"""
CLI for training TagPairTransformer on real multi-tag cluster features
(the output of cluster_features.py, e.g. cluster_features_K10_P1.csv).

Counterpart of T2TSionnaSim/MLScripts/train_realistic.py: same model
(TagPairTransformer), same loss (combined_loss), same Trainer, same targets
(4 wall distances [cm] + 4 wall angles [deg], order N, S, E, W) — only the
data loader differs.

Data format (cluster_features.py)
---------------------------------
  One row per (experiment, combination_id, permutation_id, run, frequency) with
  list-valued (JSON) cells. A *sample* is one sub-cluster measured in one run
  (or the circular mean over runs with --run_mode mean):
    theta, alpha, avgV_A, avgV_B   one value per link (K(K-1)/2), per frequency
    tag_xyz                        K x 3, permutation order
    centroid, pose                 sub-cluster centroid / quaternion [x,y,z,w]
    wall_hit [m], wall_angle [deg] order N, S, E, W

  Per pair slot the features are
    theta  (n_freq, unwrapped along frequency with period π)
    alpha  (n_freq, default on, --no_alpha to omit)          ← "attn" in the sim
    avgV_A, avgV_B (2·n_freq, --include_avgv)
    tag a / tag b XYZ relative to the centroid (6, --no_positions to omit),
      in the sub-cluster's local frame (+X = centroid → tag 0, horizontal) by
      default, or in the levelled world frame with --pos_frame world.

Train / val split
-----------------
  default (--split random)  --test_size of the samples, chosen uniformly at
                            random (seeded), go to val. The runs of one
                            sub-cluster share a ground truth and can land on
                            both sides.
  --split group             same fraction, but whole sub-clusters (all their
                            runs) go to one side (GroupShuffleSplit).
  --val_experiments 3 7     room-level holdout: those experiments are val only
                            (the honest "new room" test).

  The partition is saved next to the checkpoint as <run_stem>_split.csv, one
  line per sample:
    sample_idx  row of X_raw as returned by load_cluster_csv
    split       train | val
    experiment, combination_id, permutation_id, run   sample identity
    csv_rows    ';'-joined 0-based data-row numbers of the dataset CSV that
                make up the sample (pandas index; file line = row + 2)
  and the checkpoint carries train_idx / val_idx. ETF_eval_cluster.ipynb reads
  the split file rather than re-deriving the partition.

Examples
--------
python multitag_ml_training/train_cluster.py \\
    --dataset multitag/cluster_features_K10_P1.csv --epochs 300

python multitag_ml_training/train_cluster.py \\
    --dataset multitag/cluster_features_K10_P1.csv \\
    --val_experiments 12 --include_avgv --epochs 300

Saved: multitag_ml_training/Saved/ETF_cluster_K{K}_{n_pairs}slots{feat_tag}_d{d}_l{l}_ep{e}_s{s}_{MMDDHHMMSS}.pth
       + matching _config.json and _split.csv (read by ETF_eval_cluster.ipynb)
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))

# Model / loss / trainer live in T2TSionnaSim — import them from there so the
# sim and real pipelines share one implementation (and checkpoints stay
# interchangeable). Override the location with the T2T_ROOT env var.
T2T_ROOT = os.environ.get(
    "T2T_ROOT",
    os.path.abspath(os.path.join(_HERE, "..", "..", "T2TSionnaSim")),
)
if not os.path.isdir(os.path.join(T2T_ROOT, "MLScripts")):
    raise ImportError(f"T2TSionnaSim not found at {T2T_ROOT} — set T2T_ROOT")
if T2T_ROOT not in sys.path:
    sys.path.insert(0, T2T_ROOT)

from MLScripts.models.transformer import TagPairTransformer
from MLScripts.training.loss import combined_loss
from MLScripts.training.trainer import Trainer, ValTensors

_WALL_DIRS = ["N", "S", "E", "W"]
FREQ, RUN  = "Frequency (MHz)", "Run Exp Num"
SUB_KEYS   = ["experiment", "combination_id", "permutation_id"]   # one sub-cluster
LINK_FEATS = ["theta", "alpha", "avgV_A", "avgV_B"]
SUB_COLS   = ["tag_xyz", "centroid", "pose", "wall_hit", "wall_angle", "wall_vertical"]
_CACHE_VERSION = 2   # bump when _read_raw's output changes, to invalidate old caches


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_list_col(s: pd.Series, width: int) -> np.ndarray:
    """JSON list cells of equal length → (len(s), width) float array, without json.loads."""
    flat = np.fromstring(",".join(s.str.slice(1, -1)), sep=",")
    return flat.reshape(len(s), width)


def _parse_nested(s: pd.Series) -> np.ndarray:
    return np.array([json.loads(v) for v in s], dtype=np.float64)


def _read_raw(path: str, chunksize: int = 100_000) -> dict:
    """
    Parse the CSV once into dense arrays (cached as <csv>.parsed.npz, keyed by
    the CSV's size + mtime so a regenerated CSV is re-parsed).

    Returns dict with
      link   : (n_sub, n_run, n_freq, 4, n_pairs)  theta, alpha, avgV_A, avgV_B
      rowid  : (n_sub, n_run, n_freq) 0-based CSV data-row number of each entry
      keys   : (n_sub, 3) experiment, combination_id, permutation_id
      runs, freqs, tag_xyz (n_sub, K, 3), centroid (n_sub, 3), pose (n_sub, 4),
      wall_hit (n_sub, 4) [m], wall_angle (n_sub, 4) [deg], wall_vertical (n_sub, 4)
    """
    st = os.stat(path)
    stamp = np.array([st.st_size, int(st.st_mtime), _CACHE_VERSION])
    cache = os.path.splitext(path)[0] + ".parsed.npz"
    if os.path.isfile(cache):
        z = np.load(cache)
        if np.array_equal(z["stamp"], stamp) if z["stamp"].shape == stamp.shape else False:
            print(f"[loader] cached parse: {cache}")
            return {k: z[k] for k in z.files}

    print(f"[loader] parsing {path} (first time — cached afterwards)")
    key_parts, link_parts, sub_rows = [], [], []
    n_pairs = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if n_pairs is None:
            n_pairs = len(json.loads(chunk["theta"].iloc[0]))
        key_parts.append(chunk[SUB_KEYS + [RUN, FREQ]].to_numpy(np.int64))
        link_parts.append(np.stack(
            [_parse_list_col(chunk[f], n_pairs) for f in LINK_FEATS], axis=1
        ).astype(np.float32))                                   # (rows, 4, n_pairs)
        # Sub-cluster columns are identical across runs/freqs — keep one row each.
        sub_rows.append(chunk.drop_duplicates(SUB_KEYS)[SUB_KEYS + SUB_COLS])

    keys = np.concatenate(key_parts)
    link = np.concatenate(link_parts)
    runs, freqs = np.unique(keys[:, 3]), np.unique(keys[:, 4])

    sub_keys, sub_idx = np.unique(keys[:, :3], axis=0, return_inverse=True)
    sub_idx = sub_idx.ravel()
    n_sub = len(sub_keys)
    if len(keys) != n_sub * len(runs) * len(freqs):
        raise ValueError(
            f"{len(keys)} rows ≠ {n_sub} sub-clusters × {len(runs)} runs × "
            f"{len(freqs)} freqs — every sub-cluster needs every run and frequency"
        )
    dense = np.full((n_sub, len(runs), len(freqs), len(LINK_FEATS), n_pairs),
                    np.nan, np.float32)
    pos = (sub_idx, np.searchsorted(runs, keys[:, 3]), np.searchsorted(freqs, keys[:, 4]))
    dense[pos] = link
    rowid = np.full(dense.shape[:3], -1, np.int64)
    rowid[pos] = np.arange(len(keys))

    sub = (pd.concat(sub_rows).drop_duplicates(SUB_KEYS)
           .set_index(SUB_KEYS).loc[[tuple(k) for k in sub_keys]])
    out = dict(
        stamp=stamp, link=dense, rowid=rowid, keys=sub_keys, runs=runs, freqs=freqs,
        **{c: _parse_nested(sub[c]) for c in SUB_COLS},
    )
    np.savez(cache, **out)
    print(f"[loader] cached → {cache}")
    return out


def _local_frame(tag_xyz: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation world → local, local +X = horizontal centroid → tag 0 (as cluster_features.py)."""
    x = tag_xyz[:, 0] - centroid
    x[:, 2] = 0.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], axis=1)
    z = np.broadcast_to([0.0, 0.0, 1.0], x.shape)
    return np.stack([x, y, z], axis=1)                          # rows = local axes


def load_cluster_csv(
    path: str,
    include_alpha: bool = True,
    include_avgv: bool = False,
    include_positions: bool = True,
    pos_frame: str = "local",
    run_mode: str = "each",
) -> tuple:
    """
    Returns
    -------
    X_raw          : (n, n_pairs, feat_dim) float32
                     feat_dim = n_freq [+ n_freq alpha] [+ 2·n_freq avgV] [+ 6 positions]
    Y_dists        : (n, 4) float32  wall distances [cm]
    Y_angles       : (n, 4) float32  wall angles [deg]
    centroids      : (n, 3) float32
    poses          : (n, 4) float32  quaternion [x, y, z, w]
    n_pairs, n_freq: int
    meta_df        : DataFrame (experiment, combination_id, permutation_id, run,
                     csv_rows, wall_vertical_N/S/E/W) row-aligned with X_raw;
                     csv_rows = ';'-joined CSV data rows behind each sample
    feature_slices : dict group name → slice into feat_dim (one scaler each)
    """
    raw = _read_raw(path)
    link = raw["link"]                                  # (n_sub, n_run, n_freq, 4, n_pairs)
    n_sub, n_run, n_freq, _, n_pairs = link.shape

    theta = link[:, :, :, 0]                            # (n_sub, n_run, n_freq, n_pairs)
    other = link[:, :, :, 1:]
    if run_mode == "mean":
        # theta lives on [0, π): circular mean with period π.
        theta = (np.angle(np.exp(2j * theta).mean(axis=1, keepdims=True)) / 2) % np.pi
        other = other.mean(axis=1, keepdims=True)
        runs = np.array([-1])
        rowid = raw["rowid"].reshape(n_sub, 1, -1)          # all runs × freqs
    else:
        runs = raw["runs"]
        rowid = raw["rowid"]                                # (n_sub, n_run, n_freq)
    n_r = theta.shape[1]

    # → (n_sub·n_r, n_pairs, n_freq)
    theta = theta.transpose(0, 1, 3, 2).reshape(-1, n_pairs, n_freq).astype(np.float64)
    other = other.transpose(0, 1, 4, 3, 2).reshape(-1, n_pairs, 3, n_freq)
    theta_uw = np.unwrap(theta, axis=2, period=np.pi).astype(np.float32)

    blocks, feature_slices, cur = [theta_uw], {"theta": slice(0, n_freq)}, n_freq
    if include_alpha:
        blocks.append(other[:, :, 0])
        feature_slices["alpha"] = slice(cur, cur + n_freq); cur += n_freq
    if include_avgv:
        blocks += [other[:, :, 1], other[:, :, 2]]
        feature_slices["avgv"] = slice(cur, cur + 2 * n_freq); cur += 2 * n_freq

    rep = lambda a: np.repeat(a, n_r, axis=0)
    tag_xyz, centroid = raw["tag_xyz"], raw["centroid"]
    if include_positions:
        n_tags = tag_xyz.shape[1]
        a_idx, b_idx = np.triu_indices(n_tags, k=1)     # == itertools.combinations order
        if len(a_idx) != n_pairs:
            raise ValueError(f"{n_tags} tags imply {len(a_idx)} pairs, links have {n_pairs}")
        rel = tag_xyz - centroid[:, None, :]
        if pos_frame == "local":
            rel = np.einsum("nij,nkj->nki", _local_frame(tag_xyz, centroid), rel)
        pos = np.concatenate([rel[:, a_idx], rel[:, b_idx]], axis=2).astype(np.float32)
        blocks.append(rep(pos))
        feature_slices["pos"] = slice(cur, cur + 6); cur += 6

    X_raw    = np.concatenate(blocks, axis=2).astype(np.float32)
    Y_dists  = rep(raw["wall_hit"]).astype(np.float32) * 100   # m → cm
    Y_angles = rep(raw["wall_angle"]).astype(np.float32)        # deg
    centroids = rep(centroid).astype(np.float32)
    poses     = rep(raw["pose"]).astype(np.float32)

    meta_df = pd.DataFrame(rep(raw["keys"]), columns=SUB_KEYS)
    meta_df["run"] = np.tile(runs, n_sub)
    meta_df["csv_rows"] = [";".join(map(str, r)) for r in rowid.reshape(n_sub * n_r, -1)]
    for i, d in enumerate(_WALL_DIRS):
        meta_df[f"wall_vertical_{d}"] = rep(raw["wall_vertical"][:, i]).astype(int)

    valid = (np.isfinite(X_raw).all(axis=(1, 2))
             & np.isfinite(Y_dists).all(axis=1)
             & np.isfinite(Y_angles).all(axis=1))
    if not valid.all():
        print(f"[loader] dropped {(~valid).sum()} rows with NaN/inf (missed rays or missing links)")
        X_raw, Y_dists, Y_angles = X_raw[valid], Y_dists[valid], Y_angles[valid]
        centroids, poses = centroids[valid], poses[valid]
        meta_df = meta_df[valid].reset_index(drop=True)

    print(
        f"[loader] {n_sub} sub-clusters × {n_r} run(s) → {len(X_raw)} samples, "
        f"n_pairs={n_pairs}, n_freq={n_freq}, feat_dim={X_raw.shape[2]}, "
        f"experiments={sorted(meta_df['experiment'].unique().tolist())}"
    )
    return (X_raw, Y_dists, Y_angles, centroids, poses, n_pairs, n_freq,
            meta_df, feature_slices)


# ── Split + preprocessing ─────────────────────────────────────────────────────

def split_indices(
    meta_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    val_experiments: list[int] | None = None,
    mode: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seeded train/val sample indices (sorted).
      val_experiments given → those experiments are val, the rest train.
      mode "random"         → test_size of the samples, uniformly at random.
      mode "group"          → test_size of the sub-clusters, all runs together.
    """
    if val_experiments:
        is_val = meta_df["experiment"].isin(val_experiments).to_numpy()
        missing = set(val_experiments) - set(meta_df["experiment"].unique())
        if missing:
            raise ValueError(f"--val_experiments not in dataset: {sorted(missing)}")
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)
    if mode == "random":
        perm  = np.random.default_rng(seed).permutation(len(meta_df))
        n_val = int(round(test_size * len(meta_df)))
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])
    groups = meta_df.groupby(SUB_KEYS, sort=False).ngroup().to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, va = next(gss.split(np.zeros(len(groups)), groups=groups))
    return np.sort(tr), np.sort(va)


def save_split(path: str, meta_df: pd.DataFrame,
               train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Write the partition as <run_stem>_split.csv (see module docstring)."""
    split = np.full(len(meta_df), "", dtype=object)
    split[train_idx], split[val_idx] = "train", "val"
    out = meta_df[SUB_KEYS + ["run", "csv_rows"]].copy()
    out.insert(0, "split", split)
    out.insert(0, "sample_idx", np.arange(len(meta_df)))
    out[out["split"] != ""].to_csv(path, index=False)


def load_split(path: str, meta_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a _split.csv back into (train_idx, val_idx) for meta_df. Samples are
    matched by identity (experiment, combination_id, permutation_id, run), not
    by position, and every one of them must be found.
    """
    sp  = pd.read_csv(path)
    key = SUB_KEYS + ["run"]
    m = sp.merge(meta_df[key].reset_index(names="_idx"), on=key, how="left")
    if m["_idx"].isna().any():
        raise ValueError(f"{int(m['_idx'].isna().sum())} samples in {path} "
                         f"are not in the loaded dataset — wrong CSV or feature flags?")
    idx = m["_idx"].astype(int).to_numpy()
    if not np.array_equal(idx, sp["sample_idx"].to_numpy()):
        print("[split] sample order differs from training; matched by identity instead")
    is_val = (m["split"] == "val").to_numpy()
    return np.sort(idx[~is_val]), np.sort(idx[is_val])


def preprocess(
    X_raw: np.ndarray,
    Y_dists: np.ndarray,
    Y_angles: np.ndarray,
    feature_slices: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    centroids: np.ndarray | None = None,
    poses: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """
    Split by the given indices and fit one StandardScaler per feature group
    (theta / alpha / avgv / pos have different units) on train only.
    Same output layout as train_realistic.preprocess.
    """
    X_tr, X_va = X_raw[train_idx], X_raw[val_idx]
    feat_dim = X_raw.shape[2]
    X_tr_s = X_tr.reshape(-1, feat_dim).copy()
    X_va_s = X_va.reshape(-1, feat_dim).copy()
    scalers: dict = {}
    for name, sl in feature_slices.items():
        sc = StandardScaler()
        X_tr_s[:, sl] = sc.fit_transform(X_tr_s[:, sl])
        X_va_s[:, sl] = sc.transform(X_va_s[:, sl])
        scalers[f"x_{name}"] = sc
    X_tr_s = X_tr_s.reshape(X_tr.shape).astype(np.float32)
    X_va_s = X_va_s.reshape(X_va.shape).astype(np.float32)

    pick = lambda a, i: None if a is None else a[i]
    print(
        f"[preprocess] train={X_tr_s.shape}  val={X_va_s.shape}  "
        f"Y_d mean={Y_dists[train_idx].mean(axis=0).round(1)}  "
        f"Y_a mean={Y_angles[train_idx].mean(axis=0).round(1)}"
    )
    return dict(
        X_train=X_tr_s, X_val=X_va_s,
        Y_d_train=Y_dists[train_idx],  Y_d_val=Y_dists[val_idx],
        Y_a_train=Y_angles[train_idx], Y_a_val=Y_angles[val_idx],
        centers_train=pick(centroids, train_idx), centers_val=pick(centroids, val_idx),
        pose_train=pick(poses, train_idx),        pose_val=pick(poses, val_idx),
    ), scalers


# ── AdamW param groups ────────────────────────────────────────────────────────

def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """weight_decay on 2-D weight matrices; zero on biases, LN, cls_token, pos_embed."""
    no_decay_names = {"bias", "cls_token", "pos_embed"}
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(nd in name for nd in no_decay_names):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TagPairTransformer on real cluster_features CSV data"
    )

    # Data
    p.add_argument("--dataset", required=True,
                   help="cluster_features_K*_P*.csv from cluster_features.py")
    p.add_argument("--no_alpha", action="store_true",
                   help="Use theta features only (skip alpha)")
    p.add_argument("--include_avgv", action="store_true",
                   help="Add avgV_A / avgV_B features (2·n_freq dims per pair slot)")
    p.add_argument("--no_positions", action="store_true",
                   help="Omit centroid-relative tag XYZ features (6 dims per pair slot)")
    p.add_argument("--pos_frame", choices=["local", "world"], default="local",
                   help="Frame of the position features: local (+X = centroid → tag 0, "
                        "the frame the N/S/E/W targets live in) or levelled world")
    p.add_argument("--run_mode", choices=["each", "mean"], default="each",
                   help="each: every run is its own sample; mean: average runs per sub-cluster")
    p.add_argument("--experiments", type=int, nargs="+", default=None,
                   help="Use only these experiments (default: all)")
    p.add_argument("--val_experiments", type=int, nargs="+", default=None,
                   help="Hold these experiments out as the val set (room-level split); "
                        "--test_size is then ignored")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of samples (or sub-clusters with --split group) for val")
    p.add_argument("--split", choices=["random", "group"], default="random",
                   help="random: samples drawn at random; group: whole sub-clusters "
                        "(all runs) on one side")

    # Model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup_epochs", type=int, default=20,
                   help="Linear LR warmup (0 = ReduceLROnPlateau)")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="AdamW weight decay on weight matrices")
    p.add_argument("--l2_lambda", type=float, default=1e-4)
    p.add_argument("--huber_delta", type=float, default=0.1)
    p.add_argument("--dist_clip", type=float, default=1000.0,
                   help="Clamp GT wall distance [cm] in the loss (real rooms reach ~9 m)")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--save_dir", default=os.path.join(_HERE, "Saved"))
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index (ignored if CUDA unavailable)")

    return p.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_for_args(args) -> tuple:
    """Load + filter exactly as training does; shared with the eval notebook."""
    get = (lambda k, d=None: args.get(k, d)) if isinstance(args, dict) else \
          (lambda k, d=None: getattr(args, k, d))
    out = load_cluster_csv(
        get("dataset"),
        include_alpha=not get("no_alpha", False),
        include_avgv=get("include_avgv", False),
        include_positions=not get("no_positions", False),
        pos_frame=get("pos_frame", "local"),
        run_mode=get("run_mode", "each"),
    )
    X_raw, Y_dists, Y_angles, centroids, poses, n_pairs, n_freq, meta_df, slices = out
    if get("experiments"):
        keep = meta_df["experiment"].isin(get("experiments")).to_numpy()
        X_raw, Y_dists, Y_angles = X_raw[keep], Y_dists[keep], Y_angles[keep]
        centroids, poses = centroids[keep], poses[keep]
        meta_df = meta_df[keep].reset_index(drop=True)
        print(f"[loader] kept experiments {get('experiments')} → {len(X_raw)} samples")
    return X_raw, Y_dists, Y_angles, centroids, poses, n_pairs, n_freq, meta_df, slices


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else f"cuda:{args.gpu}" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}  seed: {args.seed}")

    # ── Load ──────────────────────────────────────────────────────────────────
    (X_raw, Y_dists, Y_angles, centroids, poses,
     n_pairs, n_freq, meta_df, feature_slices) = load_for_args(args)
    feat_dim = X_raw.shape[2]
    n_tags = int(round((1 + np.sqrt(1 + 8 * n_pairs)) / 2))

    # ── Split + preprocess ────────────────────────────────────────────────────
    train_idx, val_idx = split_indices(meta_df, args.test_size, args.seed,
                                       args.val_experiments, args.split)
    split, scalers = preprocess(X_raw, Y_dists, Y_angles, feature_slices,
                                train_idx, val_idx, centroids, poses)
    print("train experiments:", sorted(meta_df.loc[train_idx, "experiment"].unique().tolist()))
    print("val experiments  :", sorted(meta_df.loc[val_idx, "experiment"].unique().tolist()))

    # ── Tensors + DataLoader ──────────────────────────────────────────────────
    def to_t(arr) -> torch.Tensor:
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)

    X_tr, X_va   = to_t(split["X_train"]),   to_t(split["X_val"])
    Yd_tr, Yd_va = to_t(split["Y_d_train"]), to_t(split["Y_d_val"])
    Ya_tr, Ya_va = to_t(split["Y_a_train"]), to_t(split["Y_a_val"])

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(X_tr, Yd_tr, Ya_tr),
        batch_size=args.batch_size, shuffle=True, generator=g,
    )
    val_tensors = ValTensors(X=X_va, Y_d=Yd_va, Y_a=Ya_va)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = dict(
        n_slots=n_pairs,
        feat_dim=feat_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        out_dists=4,
        out_angles=4,
    )
    model = TagPairTransformer(**model_config).to(device)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(_param_groups(model, args.weight_decay), lr=args.lr)
    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1.0, (ep + 1) / args.warmup_epochs),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    def loss_fn(pred_d, pred_a, true_d, true_a, m):
        return combined_loss(pred_d, pred_a, true_d, true_a, m,
                             l2_lambda=args.l2_lambda, huber_delta=args.huber_delta,
                             dist_clip=args.dist_clip)

    # ── Output paths ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%m%d%H%M%S")
    feat_tag = (
        ("" if args.no_alpha else "_alpha")
        + ("_avgv" if args.include_avgv else "")
        + ("_nopos" if args.no_positions else f"_pos{args.pos_frame}")
        + ("_runmean" if args.run_mode == "mean" else "")
        + (f"_valexp{'-'.join(map(str, args.val_experiments))}"
           if args.val_experiments else "")
    )
    run_stem = (
        f"ETF_cluster_K{n_tags}_{n_pairs}slots{feat_tag}"
        f"_d{args.d_model}_l{args.num_layers}"
        f"_ep{args.epochs}_s{args.seed}_{ts}"
    )
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"{run_stem}.pth")
    config_path     = os.path.join(args.save_dir, f"{run_stem}_config.json")
    split_path      = os.path.join(args.save_dir, f"{run_stem}_split.csv")
    save_split(split_path, meta_df, train_idx, val_idx)
    print(f"Split      → {split_path}")

    extra_meta = dict(
        n_pairs=n_pairs,
        n_freq=n_freq,
        n_tags=n_tags,
        feat_dim=feat_dim,
        feature_slices={k: [v.start, v.stop] for k, v in feature_slices.items()},
        seed=args.seed,
        run_stem=run_stem,
        split_file=os.path.basename(split_path),
        train_idx=train_idx,
        val_idx=val_idx,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, device=device, use_bitmap=False, log_every=args.log_every,
    )
    history = trainer.fit(
        train_loader, val_tensors,
        num_epochs=args.epochs,
        save_best_path=checkpoint_path,
        model_config=model_config,
        scaler=scalers,
        extra_meta=extra_meta,
    )

    # ── Final eval (on best checkpoint) ───────────────────────────────────────
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    metrics    = trainer.evaluate(val_tensors)
    best_epoch = history.best_val_epoch()

    args_out = vars(args) | {"dataset": os.path.abspath(args.dataset)}
    with open(config_path, "w") as f:
        json.dump({
            "run_stem":      run_stem,
            "args":          args_out,
            "model_config":  model_config,
            "feature_slices": extra_meta["feature_slices"],
            "split_file":    os.path.basename(split_path),
            "n_freq":        n_freq,
            "n_train":       int(len(train_idx)),
            "n_val":         int(len(val_idx)),
            "best_epoch":    best_epoch,
            "best_val_loss": float(min(history.val_losses)),
            "final_metrics": metrics,
            "train_losses":  history.train_losses,
            "val_losses":    history.val_losses,
        }, f, indent=2)

    print(
        f"\nBest val epoch: {best_epoch}  "
        f"MAE dist: {metrics['mae_dist_m']:.3f} m  "
        f"MAE angle: {metrics['mae_angle_deg']:.2f} deg"
    )
    print(f"Checkpoint → {checkpoint_path}")
    print(f"Config     → {config_path}")


if __name__ == "__main__":
    main()

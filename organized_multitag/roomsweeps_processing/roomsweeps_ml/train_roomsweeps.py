"""
Train TagPairTransformer on room-sweep sub-clusters (the output of roomsweeps_cluster_features.py).

Room-sweep counterpart of multitag_ml_training/train_cluster.py: same model (TagPairTransformer), loss
(combined_loss), Trainer, targets (4 wall distances [cm] + 4 wall angles [deg], order N, S, E, W) and
checkpoint / _config.json / _split.csv layout, so roomsweeps_eval.ipynb mirrors ETF_eval_master.ipynb.

Features per pair slot
----------------------
  theta   (n_freq)  θ̂ unwrapped along frequency (period π)
  alpha   (n_freq)  α̂                                         (--no_alpha to omit)
  pos     (6)       tag a / tag b XYZ relative to the centroid  (OFF by default; --include_positions)
                    in the sub-cluster's local frame (+X = centroid -> tag 0, the frame the N/S/E/W targets
                    live in) or the room frame (--pos_frame room: +X towards the Edge01 wall).
  Positions encode link direction and length, so they are opt-in; run names say _pos<frame> or _nopos.

Train / val split
-----------------
  --split random (default)  --test_size of the samples at random. Optimistic: the runs and orderings of one
                            sub-cluster, and the same links, land on both sides.
  --split group             same fraction, whole sub-clusters (all runs and orderings) on one side. Still
                            shares links between sides.
  --val_experiments 8 9     those experiments are val. Every floor label always holds the same tag and the
                            sweeps overlap, so training samples containing a link (floor-label pair) measured in
                            a val experiment are dropped (--no_purge keeps them, which leaks).
  --val_columns 6 7 8       location holdout: val = sub-clusters with every tag in these grid columns (0-16),
                            train = sub-clusters with no tag in them; the rest is unused. The honest
                            "new place in the room" test.
  The partition is saved as <run_stem>_split.csv (sample_idx, split, experiment, combination_id,
  permutation_id, run); dropped samples are not listed.

Examples
--------
  python train_roomsweeps.py --dataset data/roomsweeps_K6_P1_span150cm.npz --val_columns 7 8 9
  python train_roomsweeps.py --dataset data/roomsweeps_K6_P1_span150cm.npz --val_experiments 8 --include_positions

Saved: Saved/ETF_rs_K{K}_{n_pairs}slots{feat_tag}_d{d}_l{l}_ep{e}_s{s}_{MMDDHHMMSS}.pth
       + _config.json and _split.csv (read by roomsweeps_eval.ipynb)
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))

# Model / loss / trainer live in T2TSionnaSim (shared with train_cluster.py); override with T2T_ROOT.
T2T_ROOT = os.environ.get("T2T_ROOT", os.path.expanduser("~/T2TSionnaSim"))
if not os.path.isdir(os.path.join(T2T_ROOT, "MLScripts")):
    raise ImportError(f"T2TSionnaSim not found at {T2T_ROOT} — set T2T_ROOT")
if T2T_ROOT not in sys.path:
    sys.path.insert(0, T2T_ROOT)

from MLScripts.models.transformer import TagPairTransformer  # noqa: E402
from MLScripts.training.loss import combined_loss  # noqa: E402
from MLScripts.training.trainer import Trainer, ValTensors  # noqa: E402

WALL_DIRS = ["N", "S", "E", "W"]
SUB_KEYS = ["experiment", "combination_id", "permutation_id"]   # one sub-cluster
SAMPLE_KEYS = SUB_KEYS + ["run"]


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class RoomData:
    X_raw: np.ndarray          # (n, n_pairs, feat_dim) float32
    Y_dists: np.ndarray        # (n, 4) cm
    Y_angles: np.ndarray       # (n, 4) deg
    centroids: np.ndarray      # (n, 3) m, merged-scan frame
    poses: np.ndarray          # (n, 4) quaternion [x, y, z, w]
    ray_dirs: np.ndarray       # (n, 4, 3) unit N, S, E, W directions
    labels: np.ndarray         # (n, K) floor labels, permutation order
    meta_df: pd.DataFrame      # row-aligned with X_raw
    feature_slices: dict       # group name -> slice into feat_dim (one scaler each)
    n_pairs: int
    n_freq: int
    freqs: np.ndarray
    room: dict                 # wall_names, wall_points, wall_normals, info

    def subset(self, keep: np.ndarray) -> "RoomData":
        arr = {k: getattr(self, k)[keep] for k in
               ["X_raw", "Y_dists", "Y_angles", "centroids", "poses", "ray_dirs", "labels"]}
        return RoomData(**arr, meta_df=self.meta_df[keep].reset_index(drop=True),
                        feature_slices=self.feature_slices, n_pairs=self.n_pairs, n_freq=self.n_freq,
                        freqs=self.freqs, room=self.room)


def _local_frame(xyz: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation world -> local, local +X = horizontal centroid -> tag 0 (as cluster_features.py)."""
    x = xyz[:, 0] - centroid
    x[:, 2] = 0.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], axis=1)
    z = np.broadcast_to([0.0, 0.0, 1.0], x.shape)
    return np.stack([x, y, z], axis=1)


def _room_frame(room_x: np.ndarray, n: int) -> np.ndarray:
    x = np.r_[room_x, 0.0]
    y = np.array([-x[1], x[0], 0.0])
    return np.broadcast_to(np.stack([x, y, [0.0, 0.0, 1.0]]), (n, 3, 3))


def load_roomsweeps(path: str, include_alpha: bool = True, include_positions: bool = False,
                    pos_frame: str = "local", run_mode: str = "each") -> RoomData:
    z = np.load(path)
    link = z["link"]                                   # (n_sub, n_run, n_freq, 2, n_pairs)
    n_sub, n_run, n_freq, _, n_pairs = link.shape

    theta, alpha = link[:, :, :, 0], link[:, :, :, 1]  # (n_sub, n_run, n_freq, n_pairs)
    if run_mode == "mean":
        theta = (np.angle(np.nanmean(np.exp(2j * theta), axis=1, keepdims=True)) / 2) % np.pi
        alpha = np.nanmean(alpha, axis=1, keepdims=True)
        runs = np.array([-1])
    else:
        runs = z["runs"]
    n_r = theta.shape[1]
    theta = theta.transpose(0, 1, 3, 2).reshape(-1, n_pairs, n_freq).astype(np.float64)
    alpha = alpha.transpose(0, 1, 3, 2).reshape(-1, n_pairs, n_freq)
    theta_uw = np.unwrap(theta, axis=2, period=np.pi).astype(np.float32)

    blocks, slices, cur = [theta_uw], {"theta": slice(0, n_freq)}, n_freq
    if include_alpha:
        blocks.append(alpha.astype(np.float32))
        slices["alpha"] = slice(cur, cur + n_freq); cur += n_freq

    rep = lambda a: np.repeat(a, n_r, axis=0)  # noqa: E731
    tag_xyz, centroid = z["tag_xyz"], z["centroid"]
    if include_positions:
        n_tags = tag_xyz.shape[1]
        a_idx, b_idx = np.triu_indices(n_tags, k=1)    # == itertools.combinations order
        rel = tag_xyz - centroid[:, None, :]
        if pos_frame == "local":
            R = _local_frame(tag_xyz, centroid)
        else:
            names = list(z["wall_names"])
            R = _room_frame(-z["wall_normals"][names.index("Edge01 wall")], n_sub)
        rel = np.einsum("nij,nkj->nki", R, rel)
        pos = np.concatenate([rel[:, a_idx], rel[:, b_idx]], axis=2).astype(np.float32)
        blocks.append(rep(pos))
        slices["pos"] = slice(cur, cur + 6); cur += 6

    X_raw = np.concatenate(blocks, axis=2).astype(np.float32)
    keys = rep(z["keys"])
    meta = pd.DataFrame(keys, columns=SUB_KEYS)
    meta["run"] = np.tile(runs, n_sub)
    meta["tags"] = ["-".join(map(str, t)) for t in rep(z["tag_nums"])]
    meta["labels"] = ["-".join(map(str, t)) for t in rep(z["labels"])]
    meta["span_cm"] = rep(z["span_cm"])
    cen = rep(centroid)
    meta["centroid_x"], meta["centroid_y"] = cen[:, 0], cen[:, 1]
    wall_names = [str(w) for w in z["wall_names"]]
    for i, d in enumerate(WALL_DIRS):
        meta[f"wall_vertical_{d}"] = rep(z["wall_vertical"][:, i]).astype(int)
        meta[f"wall_{d}"] = [wall_names[w] for w in rep(z["wall_id"][:, i])]

    data = RoomData(
        X_raw=X_raw,
        Y_dists=rep(z["wall_hit"]).astype(np.float32) * 100,    # m -> cm
        Y_angles=rep(z["wall_angle"]).astype(np.float32),
        centroids=cen.astype(np.float32),
        poses=rep(z["pose"]).astype(np.float32),
        ray_dirs=rep(z["ray_dirs"]).astype(np.float32),
        labels=rep(z["labels"]),
        meta_df=meta, feature_slices=slices, n_pairs=n_pairs, n_freq=n_freq, freqs=z["freqs"],
        room=dict(wall_names=wall_names, wall_points=z["wall_points"], wall_normals=z["wall_normals"],
                  info=json.loads(str(z["info"]))),
    )
    valid = np.isfinite(X_raw).all(axis=(1, 2)) & np.isfinite(data.Y_dists).all(1) & np.isfinite(data.Y_angles).all(1)
    if not valid.all():
        print(f"[loader] dropped {(~valid).sum()} samples with NaN/inf (missing runs)")
        data = data.subset(valid)
    print(f"[loader] {n_sub} sub-clusters × {n_r} run(s) → {len(data.X_raw)} samples, n_pairs={n_pairs}, "
          f"n_freq={n_freq}, feat_dim={X_raw.shape[2]} {list(slices)}, frame={data.room['info'].get('frame')}, "
          f"experiments={sorted(data.meta_df['experiment'].unique().tolist())}")
    return data


def load_for_args(args) -> RoomData:
    """Load + filter exactly as training does; shared with the eval notebook (args = Namespace or dict)."""
    get = (lambda k, d=None: args.get(k, d)) if isinstance(args, dict) else \
          (lambda k, d=None: getattr(args, k, d))
    data = load_roomsweeps(get("dataset"), include_alpha=not get("no_alpha", False),
                           include_positions=get("include_positions", False),
                           pos_frame=get("pos_frame", "local"), run_mode=get("run_mode", "each"))
    if get("experiments"):
        data = data.subset(data.meta_df["experiment"].isin(get("experiments")).to_numpy())
        print(f"[loader] kept experiments {get('experiments')} → {len(data.X_raw)} samples")
    return data


# ── Split + preprocessing ─────────────────────────────────────────────────────

def _label_pairs(labels: np.ndarray) -> list[set]:
    """Per sample: the set of floor-label pairs (links, by location) it contains."""
    K = labels.shape[1]
    a, b = np.triu_indices(K, k=1)
    lo, hi = np.minimum(labels[:, a], labels[:, b]), np.maximum(labels[:, a], labels[:, b])
    return [set(zip(l, h)) for l, h in zip(lo.tolist(), hi.tolist())]


def split_indices(data: RoomData, test_size: float = 0.2, seed: int = 42, val_experiments=None,
                  val_columns=None, mode: str = "random", purge: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Seeded (train_idx, val_idx), sorted; samples in neither are unused. See the module docstring."""
    meta = data.meta_df
    if val_experiments and val_columns:
        raise ValueError("give --val_experiments or --val_columns, not both")
    if val_columns:
        cols = (data.labels - 1) // 8
        in_val = np.isin(cols, val_columns)
        is_val, is_tr = in_val.all(axis=1), ~in_val.any(axis=1)
        print(f"[split] val columns {sorted(val_columns)}: {is_val.sum()} val, {is_tr.sum()} train, "
              f"{(~is_val & ~is_tr).sum()} straddling samples unused")
        if not is_val.any():
            raise ValueError("no sub-cluster lies entirely in --val_columns; add columns or use smaller K/span")
        return np.flatnonzero(is_tr), np.flatnonzero(is_val)
    if val_experiments:
        missing = set(val_experiments) - set(meta["experiment"].unique())
        if missing:
            raise ValueError(f"--val_experiments not in dataset: {sorted(missing)}")
        is_val = meta["experiment"].isin(val_experiments).to_numpy()
        is_tr = ~is_val
        if purge:
            pairs = _label_pairs(data.labels)
            val_pairs = set().union(*(pairs[i] for i in np.flatnonzero(is_val)))
            shared = np.array([bool(p & val_pairs) for p in pairs])
            n_drop = int((is_tr & shared).sum())
            is_tr &= ~shared
            print(f"[split] purge: dropped {n_drop} train samples sharing a link location with val "
                  f"({is_tr.sum()} train left)")
        return np.flatnonzero(is_tr), np.flatnonzero(is_val)
    if mode == "random":
        perm = np.random.default_rng(seed).permutation(len(meta))
        n_val = int(round(test_size * len(meta)))
        return np.sort(perm[n_val:]), np.sort(perm[:n_val])
    groups = meta.groupby(SUB_KEYS[:2], sort=False).ngroup().to_numpy()   # combination: all runs + orderings
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, va = next(gss.split(np.zeros(len(groups)), groups=groups))
    return np.sort(tr), np.sort(va)


def save_split(path: str, meta: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    split = np.full(len(meta), "", dtype=object)
    split[train_idx], split[val_idx] = "train", "val"
    out = meta[SAMPLE_KEYS + ["tags", "labels"]].copy()
    out.insert(0, "split", split)
    out.insert(0, "sample_idx", np.arange(len(meta)))
    out[out["split"] != ""].to_csv(path, index=False)


def load_split(path: str, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read a _split.csv back into (train_idx, val_idx) for meta, matching samples by identity."""
    sp = pd.read_csv(path)
    m = sp.merge(meta[SAMPLE_KEYS].reset_index(names="_idx"), on=SAMPLE_KEYS, how="left")
    if m["_idx"].isna().any():
        raise ValueError(f"{int(m['_idx'].isna().sum())} samples in {path} are not in the loaded dataset")
    idx = m["_idx"].astype(int).to_numpy()
    is_val = (m["split"] == "val").to_numpy()
    return np.sort(idx[~is_val]), np.sort(idx[is_val])


def preprocess(data: RoomData, train_idx, val_idx, scalers: dict | None = None) -> tuple[dict, dict]:
    """Split and standardize each feature group (fit on train unless scalers are given)."""
    X_tr, X_va = data.X_raw[train_idx], data.X_raw[val_idx]
    fd = data.X_raw.shape[2]
    X_tr_s, X_va_s = X_tr.reshape(-1, fd).copy(), X_va.reshape(-1, fd).copy()
    fixed = scalers is not None
    scalers = dict(scalers) if fixed else {}
    for name, sl in data.feature_slices.items():
        sc = scalers[f"x_{name}"] if fixed else StandardScaler().fit(X_tr_s[:, sl])
        X_tr_s[:, sl] = sc.transform(X_tr_s[:, sl])
        if len(X_va_s):
            X_va_s[:, sl] = sc.transform(X_va_s[:, sl])
        scalers[f"x_{name}"] = sc
    print(f"[preprocess] train={X_tr.shape}  val={X_va.shape}  "
          f"Y_d mean={data.Y_dists[train_idx].mean(axis=0).round(1)}")
    return dict(
        X_train=X_tr_s.reshape(X_tr.shape).astype(np.float32), X_val=X_va_s.reshape(X_va.shape).astype(np.float32),
        Y_d_train=data.Y_dists[train_idx], Y_d_val=data.Y_dists[val_idx],
        Y_a_train=data.Y_angles[train_idx], Y_a_val=data.Y_angles[val_idx],
    ), scalers


def scalers_from_checkpoint(ckpt: dict) -> dict:
    out = {}
    for name, d in ckpt["scalers"].items():
        sc = StandardScaler()
        sc.mean_ = np.asarray(d["mean"], dtype=np.float64)
        sc.scale_ = np.asarray(d["scale"], dtype=np.float64)
        sc.var_ = sc.scale_ ** 2
        sc.n_features_in_ = len(sc.mean_)
        out[name] = sc
    return out


# ── Trainer with a batched val pass (val sets here can be tens of thousands of samples) ──

class BatchedTrainer(Trainer):
    val_batch = 2048

    @torch.no_grad()
    def _predict(self, X):
        self.model.eval()
        outs = [self.model(X[i:i + self.val_batch], None) for i in range(0, len(X), self.val_batch)]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def _val_loss(self, val: ValTensors) -> float:
        pd_, pa_ = self._predict(val.X)
        loss, _, _ = self.loss_fn(pd_, pa_, val.Y_d, val.Y_a, self.model)
        return loss.item()

    def evaluate(self, val: ValTensors) -> dict:
        pd_, pa_ = self._predict(val.X)
        mae = float(torch.nanmean(torch.abs(pd_ - val.Y_d)))
        return {"mae_dist_cm": mae, "mae_dist_m": mae / 100,
                "mae_angle_deg": float(torch.nanmean(torch.abs(pa_ - val.Y_a)))}


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """weight_decay on 2-D weight matrices; none on biases, LayerNorm, cls_token, pos_embed."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or any(k in name for k in ("bias", "cls_token", "pos_embed")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TagPairTransformer on room-sweep sub-clusters",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--dataset", required=True, help=".npz from roomsweeps_cluster_features.py")
    p.add_argument("--no_alpha", action="store_true", help="θ̂ features only")
    p.add_argument("--include_positions", action="store_true",
                   help="add centroid-relative tag a / tag b XYZ (6 dims per slot); encodes link direction/length")
    p.add_argument("--pos_frame", choices=["local", "room"], default="local",
                   help="frame of the position features (with --include_positions)")
    p.add_argument("--run_mode", choices=["each", "mean"], default="each",
                   help="each: every run is a sample; mean: circular mean over runs per sub-cluster")
    p.add_argument("--experiments", type=int, nargs="+", default=None, help="use only these experiments")
    p.add_argument("--val_experiments", type=int, nargs="+", default=None,
                   help="hold these experiments out (purged of shared link locations unless --no_purge)")
    p.add_argument("--no_purge", action="store_true",
                   help="with --val_experiments, keep train samples that share a link location with val (leaks)")
    p.add_argument("--val_columns", type=int, nargs="+", default=None,
                   help="location holdout: grid columns (0-16) whose sub-clusters are val, excluded from train")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--split", choices=["random", "group"], default="random")
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
    p.add_argument("--warmup_epochs", type=int, default=20, help="linear LR warmup (0 = ReduceLROnPlateau)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--l2_lambda", type=float, default=1e-4)
    p.add_argument("--huber_delta", type=float, default=0.1)
    p.add_argument("--dist_clip", type=float, default=300.0,
                   help="clamp GT wall distance [cm] in the loss (the room's rays reach ~9 m)")
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--save_dir", default=os.path.join(_HERE, "Saved"))
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  seed: {args.seed}")

    data = load_for_args(args)
    n_pairs, n_freq, feat_dim = data.n_pairs, data.n_freq, data.X_raw.shape[2]
    n_tags = data.labels.shape[1]

    train_idx, val_idx = split_indices(data, args.test_size, args.seed, args.val_experiments,
                                       args.val_columns, args.split, purge=not args.no_purge)
    split, scalers = preprocess(data, train_idx, val_idx)
    meta = data.meta_df
    print("train experiments:", sorted(meta.loc[train_idx, "experiment"].unique().tolist()))
    print("val experiments  :", sorted(meta.loc[val_idx, "experiment"].unique().tolist()))

    to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)  # noqa: E731
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(to_t(split["X_train"]), to_t(split["Y_d_train"]),
                                            to_t(split["Y_a_train"])),
                              batch_size=args.batch_size, shuffle=True, generator=g)
    val_tensors = ValTensors(X=to_t(split["X_val"]), Y_d=to_t(split["Y_d_val"]), Y_a=to_t(split["Y_a_val"]))

    model_config = dict(n_slots=n_pairs, feat_dim=feat_dim, d_model=args.d_model, nhead=args.nhead,
                        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout, out_dists=4, out_angles=4)
    model = TagPairTransformer(**model_config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(_param_groups(model, args.weight_decay), lr=args.lr)
    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: min(1.0, (ep + 1) / args.warmup_epochs))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    def loss_fn(pred_d, pred_a, true_d, true_a, m):
        return combined_loss(pred_d, pred_a, true_d, true_a, m, l2_lambda=args.l2_lambda,
                             huber_delta=args.huber_delta, dist_clip=args.dist_clip)

    frame = data.room["info"].get("frame", "local")
    feat_tag = (("" if args.no_alpha else "_alpha")
                + (f"_pos{args.pos_frame}" if args.include_positions else "_nopos")
                + ("_runmean" if args.run_mode == "mean" else "")
                + ("_roomframe" if frame == "room" else "")
                + (f"_valexp{'-'.join(map(str, args.val_experiments))}" if args.val_experiments else "")
                + ("_nopurge" if args.val_experiments and args.no_purge else "")
                + (f"_valcol{'-'.join(map(str, args.val_columns))}" if args.val_columns else "")
                + (f"_{args.split}" if not (args.val_experiments or args.val_columns) else ""))
    ts = datetime.now().strftime("%m%d%H%M%S")
    run_stem = (f"ETF_rs_K{n_tags}_{n_pairs}slots{feat_tag}_d{args.d_model}_l{args.num_layers}"
                f"_ep{args.epochs}_s{args.seed}_{ts}")
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"{run_stem}.pth")
    config_path = os.path.join(args.save_dir, f"{run_stem}_config.json")
    split_path = os.path.join(args.save_dir, f"{run_stem}_split.csv")
    save_split(split_path, meta, train_idx, val_idx)
    print(f"Split      → {split_path}")

    extra_meta = dict(n_pairs=n_pairs, n_freq=n_freq, n_tags=n_tags, feat_dim=feat_dim,
                      feature_slices={k: [v.start, v.stop] for k, v in data.feature_slices.items()},
                      seed=args.seed, run_stem=run_stem, split_file=os.path.basename(split_path),
                      train_idx=train_idx, val_idx=val_idx)
    trainer = BatchedTrainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
                             device=device, use_bitmap=False, log_every=args.log_every)
    history = trainer.fit(train_loader, val_tensors, num_epochs=args.epochs, save_best_path=checkpoint_path,
                          model_config=model_config, scaler=scalers, extra_meta=extra_meta)

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    metrics = trainer.evaluate(val_tensors)
    best_epoch = history.best_val_epoch()
    with open(config_path, "w") as f:
        json.dump({
            "run_stem": run_stem,
            "args": vars(args) | {"dataset": os.path.abspath(args.dataset)},
            "dataset_info": data.room["info"],
            "model_config": model_config,
            "feature_slices": extra_meta["feature_slices"],
            "split_file": os.path.basename(split_path),
            "n_freq": n_freq,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "best_epoch": best_epoch,
            "best_val_loss": float(min(history.val_losses)),
            "final_metrics": metrics,
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
        }, f, indent=2)
    print(f"\nBest val epoch: {best_epoch}  MAE dist: {metrics['mae_dist_cm']:.1f} cm  "
          f"MAE angle: {metrics['mae_angle_deg']:.2f} deg")
    print(f"Checkpoint → {checkpoint_path}")
    print(f"Config     → {config_path}")


if __name__ == "__main__":
    main()

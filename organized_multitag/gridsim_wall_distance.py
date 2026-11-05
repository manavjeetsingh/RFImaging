"""Predict a simulated link's distance from the closest parallel wall from its bidirectional phase and attenuation.

The simulated counterpart of roomsweeps_wall_distance.py (whose models it imports), on the per-link table written by
grid_link_dataset.py in `grid_exp_simulated_data/data.csv`: one row per (simulation, exciter, link), every link
grid-aligned (direction 0 = along x, parallel to the y_min / y_max walls; direction 1 = along y, parallel to x_min / x_max),
with θ̂ and α̂ already computed as in roomsweeps_processing.ipynb:

    α̂12 = ½ (β̂1 / V̂2 + β̂2 / V̂1)        θ̂12 = ((Θ̂1 + Θ̂2) / 2) mod π

The target is wall_cm, the perpendicular distance of the link's midpoint from the closer of its two parallel walls.
The simulation is noise-free, so there is one "run" per link and exciter. The grid spacing differs between simulations
(20–50 cm), so `--length-cm` (default 92, as in roomsweeps_wall_distance.py: whole multiples of the real 46 cm tile)
selects links by physical length, of any number of tiles: L keeps lengths in [L − 23, L + 23) cm, so 46,92,138 together
cover 23–161 cm without gaps.

Features, over --freq-min … --freq-max (default 775–955 MHz, as in roomsweeps_wall_distance.py): θ̂ unwrapped along frequency (mod-π jumps removed, starting in [0, π) at the lowest frequency, or at 0 with
--zero-start; raw mod-π values with --no-unwrap) and α̂ per frequency (unless --no-attn). The link direction and length
are never features (only used to select links and to break down the results). Simulations are split 70 / 30 at random into train and test, so a room layout (and every
exciter's copy of its links) lands on one side. The training set is subsampled to --max-train rows to keep the SVR
tractable. Fits an SVR, an MLP and a small transformer and compares them with predicting the training mean.

The transformer reads one token per frequency (θ̂ and α̂ at that frequency, channels standardized over all frequencies so
the shape of the curve survives) plus a learned position per frequency, and a leading read-out token (learned, plus a
projection of the per-link features when a script has any, e.g. the sine fit), whose output feeds the regression head.
It trains on the same rows as the SVR and MLP. --save-model writes it (weights, standardization and the feature
settings) to a .pt file that roomsweeps_wall_distance_posttrain.py post-trains on the real links; with --kfold the saved
model is refitted on every simulation (subsampled to --max-train) after the cross-validation.

    python gridsim_wall_distance.py                       # links 69–115 cm long (≈ the real 92 cm links)
    python gridsim_wall_distance.py --length-cm 46,92,138 # links 23–161 cm long in one dataset
    python gridsim_wall_distance.py --exc 0               # one exciter per simulation
    python gridsim_wall_distance.py --no-attn             # phase only
    python gridsim_wall_distance.py --zero-start          # unwrapped phase shifted to 0 at the lowest frequency
    python gridsim_wall_distance.py --no-unwrap           # raw θ̂ in [0, π)
    python gridsim_wall_distance.py --kfold 5             # 5-fold cross-validation over simulations instead of one split
    python gridsim_wall_distance.py --no-transformer      # SVR and MLP only
    python gridsim_wall_distance.py --max-wall-cm 200     # only links within 200 cm of their closest parallel wall
    python gridsim_wall_distance.py --save-model          # also write gridsim_wall_distance_out/transformer_<tag>.pt
    python gridsim_wall_distance.py --save-model my.pt    # … or to my.pt
"""
import argparse
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from roomsweeps_wall_distance import CACHE_DIR, DIRECTION_MARKERS, FREQ_RANGE, PART_COLORS, ROOT, TILE_CM, fit_predict, metrics

DATA_PATH = ROOT / "grid_exp_simulated_data" / "data.csv"
OUT_DIR = ROOT / "gridsim_wall_distance_out"
FEATURE_PREFIXES = ("theta", "theta_uw", "alpha", "delta")
WALL_COLORS = dict(zip(["x_min", "x_max", "y_min", "y_max"], PART_COLORS))


# ---------------------------------------------------------------- data

def load_sim(path=DATA_PATH):
    """The per-link table, cached as a pickle (the CSV is ~500 MB)."""
    cache = CACHE_DIR / f"gridsim_{path.parent.name}_{path.stem}_{int(path.stat().st_mtime)}.pkl"
    if cache.exists():
        try:
            return pd.read_pickle(cache)
        except ModuleNotFoundError:  # written under another numpy major version
            print(f"  stale cache {cache.name}, rebuilding")
    print(f"  reading {path.relative_to(ROOT)} …")
    df = pd.read_csv(path)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_pickle(cache)
    return df


def freq_cols(df, prefix, freq_range=None):
    """Columns <prefix>_<f MHz> (f within freq_range) sorted by frequency, and their frequencies."""
    cols = {c: float(m[1]) for c in df.columns if (m := re.fullmatch(rf"{prefix}_(\d+(?:\.\d+)?)", c))}
    if freq_range is not None:
        cols = {c: f for c, f in cols.items() if freq_range[0] <= f <= freq_range[1]}
    cols = sorted(cols, key=cols.get)
    return cols, np.array([float(c.rsplit("_", 1)[1]) for c in cols])


def meta_cols(df):
    return [c for c in df.columns if not any(re.fullmatch(rf"{p}_\d+(?:\.\d+)?", c) for p in FEATURE_PREFIXES)]


def select_links(df, lengths_cm, exc=None, max_wall_cm=None):
    """Rows whose link length is within half a real tile (TILE_CM / 2) of one of lengths_cm, [L − ½ tile, L + ½ tile),
    labelled with that L in length_bin_cm; optionally only from the exciters in exc and at most max_wall_cm from the
    closest parallel wall."""
    length_bin = pd.Series(np.nan, index=df.index)
    for L in lengths_cm:
        length_bin[(df["length_cm"] >= L - TILE_CM / 2) & (df["length_cm"] < L + TILE_CM / 2)] = L
    keep = length_bin.notna()
    if max_wall_cm is not None:
        keep &= df["wall_cm"] <= max_wall_cm
    if exc is not None:
        keep &= df["exc_idx"].isin(exc)
    return df[keep].assign(length_bin_cm=length_bin[keep]).reset_index(drop=True)


def describe_links(links):
    print(f"\n{len(links)} link rows ({links.sim_idx.nunique()} simulations, {links.exc_idx.nunique()} exciters each), "
          f"{links.length_cm.min():.0f}–{links.length_cm.max():.0f} cm long")
    print(links.groupby(["length_bin_cm", "direction", "wall"]).agg(
        links=("sim_idx", "size"), sims=("sim_idx", "nunique"), tiles=("n_tiles", lambda n: "/".join(map(str, sorted(n.unique())))),
        min_cm=("wall_cm", "min"), max_cm=("wall_cm", "max")).round(1).to_string())


def link_features(links, extra):
    """The extra feature blocks side by side (no direction or length); drops rows with a non-finite feature."""
    X = np.column_stack(extra)
    ok = np.isfinite(X).all(axis=1)
    if not ok.all():
        print(f"  dropping {(~ok).sum()} rows with a non-finite feature")
    return X[ok], links["wall_cm"].to_numpy()[ok], links.loc[ok, meta_cols(links)].reset_index(drop=True)


def build_dataset(links, freq_range, use_attn=True, zero_start=False, unwrap=True):
    """θ̂ over freq_range, unwrapped along frequency if unwrap (re-unwrapped from the mod-π values, so it starts in [0, π)
    at the lowest selected frequency), else mod π (minus its first value if zero_start), α̂ per frequency (if use_attn)."""
    th_cols, freqs = freq_cols(links, "theta", freq_range)
    th = links[th_cols].to_numpy()
    if unwrap:
        th = np.unwrap(th, period=np.pi, axis=1)
    if zero_start:
        th = th - th[:, :1]
    extra = [th]
    if use_attn:
        extra.append(links[freq_cols(links, "alpha", freq_range)[0]].to_numpy())
    return *link_features(links, extra), freqs


# ---------------------------------------------------------------- transformer

class FreqTransformer(nn.Module):
    """Encoder over [read-out token, one token per frequency]; the read-out token's output is the prediction. The
    read-out token is learned, plus a projection of the n_global per-link features if there are any."""

    def __init__(self, n_channels, n_freqs, n_global, dim=64, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(n_channels, dim)
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_freqs, dim))
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, dim))
        self.link = nn.Linear(n_global, dim) if n_global else None
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, tokens, link):
        first = self.cls.expand(len(tokens), -1, -1)
        if self.link is not None:
            first = first + self.link(link)[:, None]
        x = torch.cat([first, self.embed(tokens) + self.pos], dim=1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


class TransformerRegressor:
    """sklearn-style fit / predict around FreqTransformer. token_cols (channels, freqs) are the columns of X that form
    the frequency tokens; every other column is a per-link feature. Inputs and target are standardized on the training
    rows; trained with AdamW, a cosine learning rate and the Huber loss for a fixed number of epochs. save / load write
    and read a fitted model, fine_tune trains a copy further (e.g. on real links, roomsweeps_wall_distance_posttrain.py)."""

    def __init__(self, token_cols, epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-2, seed=0, log_every=25, **arch):
        self.token_cols = np.asarray(token_cols)
        self.epochs, self.batch_size, self.lr, self.weight_decay, self.seed, self.log_every, self.arch = \
            epochs, batch_size, lr, weight_decay, seed, log_every, arch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _tensors(self, X):
        link_cols = np.setdiff1d(np.arange(X.shape[1]), self.token_cols.ravel())
        tokens = (X[:, self.token_cols].transpose(0, 2, 1) - self.tok_mu) / self.tok_sd     # (N, freqs, channels)
        link = (X[:, link_cols] - self.link_mu) / self.link_sd
        return [torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in (tokens, link)]

    def _set_norm(self, X, y):
        link_cols = np.setdiff1d(np.arange(X.shape[1]), self.token_cols.ravel())
        tokens = X[:, self.token_cols]
        self.tok_mu, self.tok_sd = tokens.mean(axis=(0, 2)), tokens.std(axis=(0, 2)) + 1e-8
        self.link_mu, self.link_sd = X[:, link_cols].mean(axis=0), X[:, link_cols].std(axis=0) + 1e-8
        self.y_mu, self.y_sd = y.mean(), y.std()

    def _train(self, X, y, params, epochs, batch_size, lr, weight_decay, label="transformer"):
        tokens, link = self._tensors(X)
        target = torch.as_tensor((y - self.y_mu) / self.y_sd, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        steps = -(-len(y) // batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=0.05)
        for epoch in range(epochs):
            self.model.train()
            perm, total = torch.randperm(len(y), device=self.device), 0.0
            for b in perm.split(batch_size):
                loss = nn.functional.huber_loss(self.model(tokens[b], link[b]), target[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(b)
            if epoch + 1 == epochs or (self.log_every and (epoch + 1) % self.log_every == 0):
                print(f"  {label} epoch {epoch + 1}/{epochs}: train loss {total / len(y):.4f}")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self._set_norm(X, y)
        self.model = FreqTransformer(len(self.token_cols), self.token_cols.shape[1], len(self.link_mu),
                                     **self.arch).to(self.device)
        self._train(X, y, self.model.parameters(), self.epochs, self.batch_size, self.lr, self.weight_decay)
        return self

    def fine_tune(self, X, y, epochs=200, batch_size=32, lr=1e-4, weight_decay=1e-2, renorm=False,
                  freeze_encoder=False, seed=0):
        """A copy of this fitted regressor trained further on (X, y); self is left unchanged. With renorm the input and
        target standardization is re-estimated on these rows (else the pretraining one is kept); with freeze_encoder
        only the regression head is trained."""
        ft = copy.deepcopy(self)
        torch.manual_seed(seed)
        if renorm:
            ft._set_norm(X, y)
        for p in ft.model.parameters():
            p.requires_grad_(not freeze_encoder)
        for p in ft.model.head.parameters():
            p.requires_grad_(True)
        ft._train(X, y, [p for p in ft.model.parameters() if p.requires_grad], epochs, batch_size, lr, weight_decay,
                  label="post-training")
        return ft

    NORM_KEYS = ("tok_mu", "tok_sd", "link_mu", "link_sd", "y_mu", "y_sd")

    def save(self, path, **info):
        """Weights, architecture, standardization and token layout of the fitted model, plus info (the feature settings
        it was trained with) — loadable with TransformerRegressor.load (torch.load(weights_only=True) safe)."""
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "arch": self.arch, "token_cols": self.token_cols.tolist(),
                    "norm": {k: torch.as_tensor(np.asarray(getattr(self, k), dtype=np.float64)) for k in self.NORM_KEYS},
                    "train": {"epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
                              "weight_decay": self.weight_decay, "seed": self.seed},
                    "info": info}, path)

    @classmethod
    def load(cls, path):
        """(regressor, info) from a file written by save."""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        reg = cls(ck["token_cols"], **ck["train"], **ck["arch"])
        for k in cls.NORM_KEYS:
            setattr(reg, k, ck["norm"][k].numpy())
        reg.model = FreqTransformer(len(reg.token_cols), reg.token_cols.shape[1], len(reg.link_mu), **reg.arch)
        reg.model.load_state_dict(ck["state_dict"])
        reg.model.to(reg.device)
        return reg, ck["info"]

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        tokens, link = self._tensors(X)
        out = torch.cat([self.model(t, l) for t, l in zip(tokens.split(4096), link.split(4096))])
        return out.cpu().numpy() * self.y_sd + self.y_mu


def transformer_models(args, token_cols):
    """{name: seed → regressor} for the models fitted besides those of roomsweeps_wall_distance.models."""
    if not args.transformer:
        return {}
    return {"Transformer": lambda seed: TransformerRegressor(token_cols, epochs=args.tf_epochs, seed=seed,
                                                             dim=args.tf_dim, layers=args.tf_layers)}


# ---------------------------------------------------------------- evaluation

def cap(idx, max_n, rng):
    return np.sort(rng.choice(idx, max_n, replace=False)) if max_n and len(idx) > max_n else idx


def fit_predict_all(X, y, tr, te, seed, extra_models, fitted=None):
    """fit_predict, plus every model in extra_models ({name: seed → regressor}), each also stored in fitted (a dict)."""
    preds, train_mae = fit_predict(X, y, tr, te, seed)
    for name, make in extra_models.items():
        model = make(seed).fit(X[tr], y[tr])
        if fitted is not None:
            fitted[name] = model
        preds[name] = model.predict(X[te])
        train_mae[name] = mean_absolute_error(y[tr], model.predict(X[tr]))
    return preds, train_mae


def evaluate(X, y, meta, train_frac, seed, max_train, kfold=0, extra_models=None, fitted=None):
    """Random split by simulation, or with kfold K-fold cross-validation over simulations (predictions of every fold
    pooled). The training rows are subsampled to max_train (0: all). extra_models ({name: seed → regressor}) are fitted
    alongside the SVR and MLP and stored in fitted (a dict; with kfold, those of the last fold). Prints the results;
    returns (test row indices, {model: predictions}) without the training mean."""
    extra_models = extra_models or {}
    rng = np.random.default_rng(seed)
    sims = meta["sim_idx"].to_numpy()
    n_sims = lambda idx: len(np.unique(sims[idx]))
    if not kfold:
        tr, te = next(GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed).split(X, groups=sims))
        tr_fit = cap(tr, max_train, rng)
        print(f"train: {len(tr_fit)} of {len(tr)} rows ({n_sims(tr)} simulations)    "
              f"test: {len(te)} rows ({n_sims(te)} simulations)")
        preds, train_mae = fit_predict_all(X, y, tr_fit, te, seed, extra_models, fitted)
        results = {name: {**metrics(y[te], p), "train MAE (cm)": train_mae.get(name, np.nan)} for name, p in preds.items()}
        print("\ntest set:")
        print(pd.DataFrame(results).T.round(2).to_string())
        print_by_direction(y[te], preds, meta.iloc[te])
        return te, {k: v for k, v in preds.items() if k != "mean of train"}

    pooled, folds = {}, []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=kfold).split(X, groups=sims)):
        tr_fit = cap(tr, max_train, rng)
        preds, train_mae = fit_predict_all(X, y, tr_fit, te, seed, extra_models, fitted)
        for name, p in preds.items():
            pooled.setdefault(name, np.full(len(y), np.nan))[te] = p
        folds.append({"fold": k, "test sims": n_sims(te), "test rows": len(te), "train rows": len(tr_fit),
                      **{f"{name} MAE": mean_absolute_error(y[te], p) for name, p in preds.items()},
                      **{f"{name} train MAE": v for name, v in train_mae.items()}})
    print(f"\n{kfold}-fold cross-validation over simulations:")
    print(pd.DataFrame(folds).set_index("fold").round(1).to_string())
    print("\nall held-out predictions pooled:")
    print(pd.DataFrame({name: metrics(y, p) for name, p in pooled.items()}).T.round(2).to_string())
    print_by_direction(y, pooled, meta)
    return np.arange(len(y)), {k: v for k, v in pooled.items() if k != "mean of train"}


def print_by_direction(y, preds, meta):
    d = meta["direction"].to_numpy()
    print("\ntest MAE (cm) by link direction:")
    print(pd.DataFrame({name: {f"direction {k}": mean_absolute_error(y[d == k], p[d == k]) for k in (0, 1) if (d == k).any()}
                        for name, p in preds.items()}).T.round(2).to_string())


def plot_predictions(y_test, preds, meta_test, path, title, max_points=5000, seed=0):
    """Predicted vs true distance, one panel per model, colored by the closest wall; at most max_points rows drawn."""
    show = cap(np.arange(len(y_test)), max_points, np.random.default_rng(seed))
    fig, axes = plt.subplots(1, len(preds), figsize=(5.5 * len(preds) + 2.0, 5), squeeze=False)
    hi = max(y_test.max(), *(p.max() for p in preds.values()))
    pad = 0.05 * hi                              # axes start at 0 (distances are non-negative)
    wall, direction = meta_test["wall"].to_numpy()[show], meta_test["direction"].to_numpy()[show]
    for ax, (name, p) in zip(axes.flat, preds.items()):
        ax.plot([0, hi + pad], [0, hi + pad], color="#9a9a9a", lw=1, ls="--", zorder=1)
        for w, color in WALL_COLORS.items():
            for d, marker in DIRECTION_MARKERS.items():
                sel = (wall == w) & (direction == d)
                ax.scatter(y_test[show][sel], p[show][sel], s=8, color=color, marker=marker, alpha=0.35,
                           linewidths=0, zorder=2)
        ax.set_title(f"{name}  (test MAE {mean_absolute_error(y_test, p):.1f} cm)", fontsize=11)
        ax.set_xlabel("true distance from closest parallel wall (cm)")
        ax.set_ylabel("predicted (cm)")
        ax.set_xlim(0, hi + pad)
        ax.set_ylim(0, hi + pad)
        ax.set_aspect("equal")
        ax.grid(color="#e0e0e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, markersize=8, label=w)
               for w, c in WALL_COLORS.items() if (wall == w).any()]
    handles += [plt.Line2D([], [], marker=m, ls="", color="#555555", markersize=7, label=f"direction {d} link")
                for d, m in DIRECTION_MARKERS.items()]
    axes.flat[-1].legend(handles=handles, title="closest wall", frameon=False, fontsize=8,
                         title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title + (f" ({len(show)} of {len(y_test)} shown)" if len(show) < len(y_test) else ""), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- command line

def common_args(ap):
    ap.add_argument("--data", type=Path, default=DATA_PATH, help="per-link CSV written by grid_link_dataset.py")
    ap.add_argument("--length-cm", default="92",
                    help="link length(s) in cm, comma-separated (e.g. 92,138); each a whole number of real tiles, "
                         "keeping links within half a tile of it")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--exc", default=None, help="exciter indices to keep, comma-separated (default: all)")
    ap.add_argument("--max-wall-cm", type=float, default=None,
                    help="ignore links whose midpoint is more than this many cm from the closest parallel wall")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--max-train", type=int, default=20000, help="subsample the training rows to this many (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfold", type=int, default=0,
                    help="K-fold cross-validation over simulations instead of the random split")
    ap.add_argument("--transformer", action=argparse.BooleanOptionalAction, default=True,
                    help="also fit the frequency-token transformer (--no-transformer: SVR and MLP only)")
    ap.add_argument("--tf-epochs", type=int, default=100)
    ap.add_argument("--tf-dim", type=int, default=64, help="transformer width")
    ap.add_argument("--tf-layers", type=int, default=3, help="transformer encoder layers")


def parse_lengths(args):
    """--length-cm as sorted whole multiples of TILE_CM, as in roomsweeps_wall_distance.py."""
    n_tiles = sorted({int(round(float(x) / TILE_CM)) for x in args.length_cm.split(",")})
    for x in args.length_cm.split(","):
        assert round(float(x) / TILE_CM) > 0 and abs(round(float(x) / TILE_CM) * TILE_CM - float(x)) < 1, \
            f"{x} cm is not a whole number of {TILE_CM:g} cm tiles"
    return [n * TILE_CM for n in n_tiles]


def load_links(args):
    """Parsed --length-cm (cm), and the selected link rows (printed)."""
    lengths = parse_lengths(args)
    exc = [int(x) for x in args.exc.split(",")] if args.exc else None
    print("loading simulated links …")
    links = select_links(load_sim(args.data), lengths, exc, args.max_wall_cm)
    if links.empty:
        raise SystemExit("no links match --length-cm / --exc / --max-wall-cm")
    describe_links(links)
    return lengths, links


def lengths_label(lengths):
    return ", ".join(f"{L - TILE_CM / 2:.0f}–{L + TILE_CM / 2:.0f}" for L in lengths) + " cm"


def output_tag(args):
    return ("-".join(f"{L:.0f}" for L in parse_lengths(args)) + "cm"
            + (f"_exc{args.exc.replace(',', '-')}" if args.exc else "") + (f"_kfold{args.kfold}" if args.kfold else "")
            + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else ""))


def save_outputs(out_dir, tag, y, te, preds, meta, title):
    out_dir.mkdir(exist_ok=True)
    out = meta.iloc[te].reset_index(drop=True).assign(**{f"pred_{k}": v for k, v in preds.items()})
    out.to_csv(out_dir / f"test_predictions_{tag}.csv", index=False)
    plot_predictions(y[te], preds, meta.iloc[te], out_dir / f"pred_vs_true_{tag}.png", title)
    print(f"\nwrote {(out_dir / f'test_predictions_{tag}.csv').relative_to(ROOT)}, "
          f"{(out_dir / f'pred_vs_true_{tag}.png').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--zero-start", action="store_true",
                    help="shift each link's unwrapped θ̂ so it is 0 at the lowest frequency")
    ap.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=True,
                    help="unwrap θ̂ along frequency (--no-unwrap: raw θ̂ in [0, π))")
    ap.add_argument("--save-model", nargs="?", const=True, default=None, metavar="PATH",
                    help="save the fitted transformer (default path: gridsim_wall_distance_out/transformer_<tag>.pt)")
    args = ap.parse_args()
    if args.zero_start and not args.unwrap:
        ap.error("--zero-start only works with unwrapped phase (drop --no-unwrap)")
    if args.save_model and not args.transformer:
        ap.error("--save-model needs the transformer (drop --no-transformer)")

    lengths, links = load_links(args)
    X, y, meta, freqs = build_dataset(links, (args.freq_min, args.freq_max), args.attn, args.zero_start, args.unwrap)
    phase = f"θ̂{'' if args.unwrap else ' (wrapped)'}{' (start at 0)' if args.zero_start else ''}"
    print(f"{len(X)} samples (one per link and exciter), {X.shape[1]} features: {phase}{' + α̂' if args.attn else ''} at "
          f"{len(freqs)} freqs {freqs.min():g}–{freqs.max():g} MHz")

    # frequency tokens: θ̂ in columns 0…F-1, α̂ in F…2F-1
    token_cols = np.arange(len(freqs))[None] + len(freqs) * np.arange(2 if args.attn else 1)[:, None]
    fitted = {}
    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.max_train, args.kfold,
                         transformer_models(args, token_cols), fitted)

    tag = (output_tag(args) + ("" if args.attn else "_phase_only") + ("_zero_start" if args.zero_start else "")
           + ("" if args.unwrap else "_no_unwrap"))
    save_outputs(OUT_DIR, tag, y, te, preds, meta,
                 f"simulated {lengths_label(lengths)} links, {phase}{' + α̂' if args.attn else ' only'}: "
                 "distance from closest parallel wall (test set)")

    if args.save_model:
        model = fitted["Transformer"]          # with the random split: fitted on the training simulations
        if args.kfold:
            idx = cap(np.arange(len(y)), args.max_train, np.random.default_rng(args.seed))
            print(f"\nrefitting the transformer on {len(idx)} rows of all {meta.sim_idx.nunique()} simulations to save it")
            model = transformer_models(args, token_cols)["Transformer"](args.seed).fit(X[idx], y[idx])
        path = OUT_DIR / f"transformer_{tag}.pt" if args.save_model is True else Path(args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path, freqs=freqs.tolist(), attn=args.attn, zero_start=args.zero_start, unwrap=args.unwrap,
                   freq_min=args.freq_min, freq_max=args.freq_max, length_cm=args.length_cm, exc=args.exc,
                   max_wall_cm=args.max_wall_cm, kfold=args.kfold, seed=args.seed, tag=tag, data=str(args.data))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

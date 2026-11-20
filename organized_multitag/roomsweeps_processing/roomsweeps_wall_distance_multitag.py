"""Predict the distance from the centroid of a cluster of n tags to the closest wall, from the phase and attenuation of
every link inside the cluster.

Where roomsweeps_wall_distance_alllinks.py makes one prediction per link, this script makes one per cluster: n tags of
one experiment (--n-tags, default 4) and all C(n, 2) links between them (6 links for 4 tags). The target is

    C = mean of the n tag positions (merged room scan),   target = min over the four walls of the perpendicular distance
                                                                   from C to that wall (cm)

Only compact clusters are used: the largest nominal distance between two of the tags must be at most --max-span-cm
(default 150 cm, which allows offsets up to 3 × 1 and 2 × 2 tiles). --max-clusters-per-exp randomly subsamples what is
left.

Per-link features (RF only; without --rel-pos tag positions and link lengths are never features), one median over runs
per link:

    tokens   per frequency: [θ̂ (unwrapped) minus its straight-line fit over the band, log α̂ minus its band mean]
    globals  per link:      [slope of that line, band mean of log α̂]

With --rel-pos every cluster also gets a context vector of relative tag locations: the cluster's tag with the lowest
floor label is put at (0, 0) and the other n − 1 tags, in increasing label order, contribute their (x, y) relative to it
(cm, merged room scan) — 2(n − 1) values, 12 for 7 tags. The SVR appends it to its pooled vector; every network
standardizes it on the training clusters and feeds it to its regression head next to the pooled link summary.

The links of a cluster have no natural order, so every model merges them in a way that does not depend on tag numbering:

- SVR          hand-made pooling to one fixed vector: mean and std across links of each token at each frequency, each
               global sorted across links, and the mean and std of each global. RBF kernel.
- MLP          DeepSets: one shared MLP embeds each link's flattened tokens and globals; the mean, max and std of the
               link embeddings feed the regression head.
- ResCNN       mid-level fusion along frequency: a shared residual CNN encodes each link and keeps the frequency axis, the
               mean and std across links are taken at every frequency, and further residual blocks process the merged map.
- Transformer  two levels: a shared transformer over frequency summarizes each link in a CLS token, then a transformer
               over the links relates them, with a learned attention bias for link pairs that share a tag or do not.

Evaluation is leave one experiment out. Clusters overlap heavily, so a random split would put the same measured link on
both sides. Every training cluster containing a label pair that the held-out experiment measured is dropped as well
(--no-purge: kept), so no test link position is seen in training.

    python roomsweeps_wall_distance_multitag.py                       # 4 tags, 6 links per cluster
    python roomsweeps_wall_distance_multitag.py --n-tags 3            # 3 tags, 3 links per cluster
    python roomsweeps_wall_distance_multitag.py --max-span-cm 131     # allow wider clusters
    python roomsweeps_wall_distance_multitag.py --models svr,rescnn   # only some of the models
    python roomsweeps_wall_distance_multitag.py --no-attn             # phase only
    python roomsweeps_wall_distance_multitag.py --n-tags 7 --rel-pos  # 7 tags + their locations relative to the first
"""
import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from roomsweeps_wall_distance import (BASELINE_COLOR, DATA_DIR, FREQ_RANGE, GOAL_CM, PART_COLORS, ROOT, TILE_CM,
                                      all_positions, find_walls, grid_xy, label_grid, link_runs, load_bidir,
                                      median_phase, metrics, wall_distance_cm)

OUT_DIR = Path(__file__).resolve().parent / "roomsweeps_wall_distance_multitag_out"
MODEL_NAMES = {"svr": "SVR", "mlp": "MLP (DeepSets)", "rescnn": "ResCNN (frequency fusion)",
               "transformer": "Transformer (hierarchical)"}


# ---------------------------------------------------------------- links and clusters

def load_links(positions, exps, bidir, freq_range):
    """Median θ̂ (unwrapped) and α̂ of every tag pair of every experiment: ({(exp, T1, T2): link index}, θ̂ (L, F),
    α̂ (L, F), freqs). Tag pairs without a complete run are left out."""
    index, th, al, freqs = {}, [], [], None
    for exp in exps:
        for ta, tb in itertools.combinations(sorted(positions[exp].index), 2):
            link = pd.Series({"exp": exp, "T1": ta, "T2": tb})
            theta, alpha = link_runs(bidir, link, freq_range)
            if theta.empty:
                continue
            freqs = theta.columns.to_numpy()
            index[(exp, ta, tb)] = len(th)
            th.append(median_phase(theta.to_numpy(), True))
            al.append(np.median(alpha.to_numpy(), axis=0))
    return index, np.array(th), np.array(al), freqs


def link_features(th, al, freqs, attn=True):
    """Per-link model inputs from θ̂, α̂ (L, F): tokens (L, F, channels) and globals (L, channels), see the module docstring."""
    x = (freqs - freqs.mean()) / 100
    A = np.stack([x, np.ones_like(x)], 1)
    coef = th @ np.linalg.pinv(A).T                 # (L, 2): slope, level
    tok, glob = [th - coef @ A.T], [coef[:, 0]]
    if attn:
        log_al = np.log(np.clip(al, 1e-6, None))
        mean_log = log_al.mean(1, keepdims=True)
        tok.append(log_al - mean_log)
        glob.append(mean_log[:, 0])
    return np.stack(tok, -1), np.stack(glob, -1)


def build_clusters(positions, exps, walls, link_index, n_tags, max_span_cm, max_per_exp, seed):
    """Every compact n-tag cluster with all its links measured: (meta frame, link indices (N, K), label-pair ids (N, K),
    relative tag locations (N, 2(n − 1))). The K = C(n, 2) links of a cluster are in the order of
    itertools.combinations over its sorted tags. Relative locations: the lowest-label tag at (0, 0), the others' (x, y)
    in cm relative to it, in increasing label order, flattened as x1, y1, x2, y2, …."""
    rng = np.random.default_rng(seed)
    meta, idx, pairs, rels, pair_id = [], [], [], [], {}
    for exp in exps:
        pos = positions[exp]
        rows = []
        for tags in itertools.combinations(sorted(pos.index), n_tags):
            labels = [int(pos.loc[t, "label"]) for t in tags]
            grid = np.array([label_grid(l) for l in labels], float)
            span = max(TILE_CM * np.linalg.norm(grid[i] - grid[j]) for i, j in itertools.combinations(range(n_tags), 2))
            if max_span_cm is not None and span > max_span_cm + 1e-6:
                continue
            links = [link_index.get((exp, tags[i], tags[j])) for i, j in itertools.combinations(range(n_tags), 2)]
            if any(l is None for l in links):
                continue
            xy = np.array([grid_xy(*label_grid(l)) for l in sorted(labels)])
            rel = (100 * (xy[1:] - xy[0])).ravel()
            centroid = xy.mean(0)
            dists = {name: wall_distance_cm(centroid, w) for name, w in walls.items()}
            wall = min(dists, key=dists.get)
            ids = [pair_id.setdefault(tuple(sorted((labels[i], labels[j]))), len(pair_id))
                   for i, j in itertools.combinations(range(n_tags), 2)]
            rows.append(({"exp": exp, "tags": "-".join(map(str, tags)), "labels": "-".join(map(str, labels)),
                          "span_cm": span, "centroid_x": centroid[0], "centroid_y": centroid[1],
                          "wall": wall.split(" (", 1)[0], "wall_cm": dists[wall]}, links, ids, rel))
        if max_per_exp is not None and len(rows) > max_per_exp:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), max_per_exp, replace=False))]
        for m, links, ids, rel in rows:
            meta.append(m)
            idx.append(links)
            pairs.append(ids)
            rels.append(rel)
    return pd.DataFrame(meta), np.array(idx), np.array(pairs), np.array(rels).reshape(len(meta), 2 * (n_tags - 1))


def shared_tag_matrix(n_tags):
    """(K, K) relation between the links of a cluster: 0 the same link, 1 they share a tag, 2 they share none."""
    links = list(itertools.combinations(range(n_tags), 2))
    return torch.tensor([[0 if a == b else 1 if set(a) & set(b) else 2 for b in links] for a in links])


# ---------------------------------------------------------------- SVR: hand-made pooling

def pooled_features(tok, glob, idx):
    """One fixed vector per cluster that does not depend on the link order."""
    t, g = tok[idx], glob[idx]                      # (N, K, F, C), (N, K, G)
    N = len(idx)
    return np.concatenate([t.mean(1).reshape(N, -1), t.std(1).reshape(N, -1), np.sort(g, axis=1).reshape(N, -1),
                           g.mean(1), g.std(1)], axis=1)


def fit_svr(tok, glob, idx, ctx, y, tr, te):
    X = np.concatenate([pooled_features(tok, glob, idx), ctx], axis=1)
    model = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale")),
                                       transformer=StandardScaler()).fit(X[tr], y[tr])
    return model.predict(X[te]), model.predict(X[tr])


# ---------------------------------------------------------------- neural networks: inputs (B, K, F, C), (B, K, G), (B, n_ctx)
# ctx is the per-cluster context vector (relative tag locations with --rel-pos, else n_ctx = 0); it joins the head input.

def set_std(x, dim):
    return x.std(dim, unbiased=False)


class DeepSetMLP(nn.Module):
    """Shared link MLP, then mean, max and std over the links."""

    def __init__(self, n_freqs, n_channels, n_global, n_ctx=0, width=128, dropout=0.1, **_):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_freqs * n_channels + n_global, 256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(3 * width + n_ctx, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, tok, glob, ctx):
        z = self.phi(torch.cat([tok.flatten(2), glob], -1))
        return self.rho(torch.cat([z.mean(1), z.amax(1), set_std(z, 1), ctx], -1)).squeeze(-1)


def res_block(width, dilation, dropout):
    return nn.Sequential(nn.BatchNorm1d(width), nn.GELU(), nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                         nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(width, width, 3, padding=1))


class FusionResCNN(nn.Module):
    """Shared residual CNN per link, merge across links at every frequency (mean and std), residual CNN on the merged map."""

    def __init__(self, n_freqs, n_channels, n_global, n_ctx=0, width=64, dropout=0.1, **_):
        super().__init__()
        self.inp = nn.Conv1d(n_channels, width, 3, padding=1)
        self.link_blocks = nn.ModuleList(res_block(width, d, dropout) for d in (1, 2))
        self.merge = nn.Conv1d(2 * width, width, 1)
        self.fused_blocks = nn.ModuleList(res_block(width, d, dropout) for d in (4, 1))
        self.head = nn.Sequential(nn.Linear(2 * width + 4 * n_global + n_ctx, 128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128, 1))

    def forward(self, tok, glob, ctx):
        B, K, F, C = tok.shape
        x = self.inp(tok.reshape(B * K, F, C).transpose(1, 2))
        for block in self.link_blocks:
            x = x + block(x)
        x = x.reshape(B, K, -1, F)
        x = self.merge(torch.cat([x.mean(1), set_std(x, 1)], 1))
        for block in self.fused_blocks:
            x = x + block(x)
        g = torch.cat([glob.mean(1), set_std(glob, 1), glob.amin(1), glob.amax(1)], -1)
        return self.head(torch.cat([x.mean(-1), x.amax(-1), g, ctx], -1)).squeeze(-1)


class HierarchicalTransformer(nn.Module):
    """Transformer over frequency per link (shared), then over the links with a learned attention bias per link relation.
    The links carry no position embedding, so the output does not depend on their order."""

    def __init__(self, n_freqs, n_channels, n_global, relation, n_ctx=0, dim=64, heads=4, freq_layers=2, link_layers=2,
                 dropout=0.1):
        super().__init__()
        self.tok = nn.Linear(n_channels, dim)
        self.glob = nn.Linear(n_global, dim)
        self.pos = nn.Parameter(torch.randn(1, n_freqs, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, "gelu", batch_first=True, norm_first=True)
        self.freq_encoder = nn.TransformerEncoder(layer(), freq_layers, enable_nested_tensor=False)
        self.link_layers = nn.ModuleList(layer() for _ in range(link_layers))
        self.bias = nn.Parameter(torch.zeros(link_layers, 3))
        self.register_buffer("relation", relation)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(2 * dim + n_ctx, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, tok, glob, ctx):
        B, K, F, C = tok.shape
        cls = self.cls.expand(B * K, -1, -1) + self.glob(glob.reshape(B * K, -1))[:, None]
        z = self.freq_encoder(torch.cat([cls, self.tok(tok.reshape(B * K, F, C)) + self.pos], 1))[:, 0].reshape(B, K, -1)
        for bias, layer in zip(self.bias, self.link_layers):
            z = layer(z, src_mask=bias[self.relation])
        z = self.norm(z)
        return self.head(torch.cat([z.mean(1), z.amax(1), ctx], -1)).squeeze(-1)


NETS = {"mlp": DeepSetMLP, "rescnn": FusionResCNN, "transformer": HierarchicalTransformer}


def fit_net(key, tok, glob, idx, ctx, y, tr, te, relation, args, device):
    """Train one network on the clusters tr; returns its predictions on te and tr. Inputs and target are standardized on
    the training clusters only."""
    used = np.unique(idx[tr])
    tok_mu, tok_sd = tok[used].mean((0, 1)), tok[used].std((0, 1)) + 1e-6
    glob_mu, glob_sd = glob[used].mean(0), glob[used].std(0) + 1e-6
    ctx_mu, ctx_sd = ctx[tr].mean(0), ctx[tr].std(0) + 1e-6
    y_mu, y_sd = y[tr].mean(), y[tr].std()
    T = lambda a, kind=torch.float32: torch.as_tensor(np.asarray(a), dtype=kind, device=device)
    tok_t, glob_t = T((tok - tok_mu) / tok_sd), T((glob - glob_mu) / glob_sd)
    idx_t, target, ctx_t = T(idx, torch.long), T((y - y_mu) / y_sd), T((ctx - ctx_mu) / ctx_sd)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    extra = {"relation": relation} if key == "transformer" else {}
    model = NETS[key](tok.shape[1], tok.shape[2], glob.shape[1], n_ctx=ctx.shape[1], **extra).to(device)
    epochs, lr = {"mlp": (args.mlp_epochs, args.mlp_lr), "rescnn": (args.rescnn_epochs, args.rescnn_lr),
                  "transformer": (args.tf_epochs, args.tf_lr)}[key]
    steps = -(-len(tr) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=0.05)
    tr_t = torch.as_tensor(tr)
    for _ in range(epochs):
        model.train()
        for b in tr_t[torch.randperm(len(tr), generator=gen)].split(args.batch_size):
            b = b.to(device)
            if len(b) < 2:                          # batch norm needs more than one sample
                continue
            loss = nn.functional.huber_loss(model(tok_t[idx_t[b]], glob_t[idx_t[b]], ctx_t[b]), target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    model.eval()

    @torch.no_grad()
    def predict(rows):
        out = [model(tok_t[idx_t[b]], glob_t[idx_t[b]], ctx_t[b]).cpu()
               for b in torch.as_tensor(rows).to(device).split(256)]
        return torch.cat(out).numpy() * y_sd + y_mu
    return predict(te), predict(tr)


# ---------------------------------------------------------------- leave one experiment out

def leave_one_experiment_out(tok, glob, idx, ctx, pairs, y, meta, relation, args, device):
    """({model: pooled held-out predictions} including "mean of train", per-fold table)."""
    exp_of = meta["exp"].to_numpy()
    folds = list(dict.fromkeys(exp_of))[:args.max_folds]
    pooled, table = {}, []
    for exp in folds:
        t0 = time.time()
        te, tr = np.flatnonzero(exp_of == exp), np.flatnonzero(exp_of != exp)
        n_before = len(tr)
        if args.purge:
            tr = tr[~np.isin(pairs[tr], np.unique(pairs[te])).any(1)]
        row = {"exp": exp, "test clusters": len(te), "train clusters": len(tr), "purged": n_before - len(tr)}
        preds = {"mean of train": (np.full(len(te), y[tr].mean()), np.full(len(tr), y[tr].mean()))}
        for key in args.model_keys:
            preds[MODEL_NAMES[key]] = fit_svr(tok, glob, idx, ctx, y, tr, te) if key == "svr" else \
                fit_net(key, tok, glob, idx, ctx, y, tr, te, relation, args, device)
        for name, (p_te, p_tr) in preds.items():
            pooled.setdefault(name, np.full(len(y), np.nan))[te] = p_te
            row[f"{name} MAE"] = mean_absolute_error(y[te], p_te)
            row[f"{name} train MAE"] = mean_absolute_error(y[tr], p_tr)
        table.append(row)
        scores = "  ".join(f"{name}: {row[f'{name} MAE']:.1f}" for name in preds)
        print(f"exp {exp:>2}  train {len(tr):4d} ({row['purged']} purged)  test {len(te):3d}  MAE (cm)  {scores}  "
              f"[{time.time() - t0:.0f} s]", flush=True)
    return pooled, pd.DataFrame(table).set_index("exp")


# ---------------------------------------------------------------- plots

def series_colors(names):
    colors = {"mean of train": BASELINE_COLOR}
    colors.update(zip([n for n in names if n != "mean of train"], PART_COLORS))
    return colors


def style(ax):
    ax.grid(color="#e0e0e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_predictions(y, preds, path, title):
    """Predicted against true distance, one panel per model."""
    names = [n for n in preds if n != "mean of train"]
    colors = series_colors(preds)
    ncols = min(len(names), 2)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.2 * nrows), squeeze=False)
    hi = 1.05 * max(y.max(), *(preds[n].max() for n in names))
    for ax, name in zip(axes.flat, names):
        ax.plot([0, hi], [0, hi], color="#9a9a9a", lw=1, ls="--", zorder=1)
        ax.scatter(y, preds[name], s=14, color=colors[name], alpha=0.35, linewidths=0, zorder=2)
        ax.set_title(f"{name}  (MAE {mean_absolute_error(y, preds[name]):.1f} cm)", fontsize=11)
        ax.set_xlabel("true distance from centroid to closest wall (cm)")
        ax.set_ylabel("predicted (cm)")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        style(ax)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(y, preds, path, title, bin_cm=25):
    """Cumulative absolute error, and the MAE per bin of the true distance, against predicting the training mean."""
    colors = series_colors(preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ax.axvspan(*GOAL_CM, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    for name, p in preds.items():
        a = np.sort(np.abs(p - y))
        ax.step(np.r_[0, a], np.r_[0, np.arange(1, len(a) + 1) / len(a)], where="post", color=colors[name], lw=2,
                ls="--" if name == "mean of train" else "-",
                label=f"{name}: {100 * (a <= GOAL_CM[1]).mean():.0f}% ≤ {GOAL_CM[1]} cm, median {np.median(a):.0f} cm")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"absolute error (cm), shaded: goal {GOAL_CM[0]}–{GOAL_CM[1]} cm")
    ax.set_ylabel("fraction of held-out clusters with at most this error")
    ax.set_title("Cumulative absolute error (left / higher is better)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    edges = np.arange(np.floor(y.min() / bin_cm) * bin_cm, y.max() + bin_cm, bin_cm)
    b = np.digitize(y, edges) - 1
    used = [k for k in range(len(edges) - 1) if (b == k).sum() >= 5]
    x = np.arange(len(used))
    for name, p in preds.items():
        ax.plot(x, [np.abs(p - y)[b == k].mean() for k in used], color=colors[name], lw=2, marker="o", ms=8, mec="white",
                mew=1.5, ls="--" if name == "mean of train" else "-", label=name)
    ax.axhspan(*GOAL_CM, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{edges[k]:.0f}–{edges[k + 1]:.0f}\n({(b == k).sum()})" for k in used], fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("true distance from centroid to closest wall (cm), (number of held-out clusters)")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("MAE by true distance", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        style(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tags", type=int, default=4, help="tags per cluster; all C(n, 2) links between them are used")
    ap.add_argument("--max-span-cm", type=float, default=150.0,
                    help="largest nominal distance between two tags of a cluster (negative: no limit)")
    ap.add_argument("--max-clusters-per-exp", type=int, default=None, help="randomly keep at most this many per experiment")
    ap.add_argument("--models", default="svr,mlp,rescnn,transformer", help="comma-separated: svr, mlp, rescnn, transformer")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--rel-pos", action="store_true",
                    help="add the cluster's relative tag locations as features: lowest-label tag at (0, 0), the other "
                         "n − 1 tags' (x, y) relative to it in cm")
    ap.add_argument("--purge", action=argparse.BooleanOptionalAction, default=True,
                    help="drop training clusters containing a label pair measured by the held-out experiment")
    ap.add_argument("--max-folds", type=int, default=None, help="only hold out the first this many experiments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=100)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--rescnn-epochs", type=int, default=60)
    ap.add_argument("--rescnn-lr", type=float, default=1e-3)
    ap.add_argument("--tf-epochs", type=int, default=60)
    ap.add_argument("--tf-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--device", default=None, help="torch device (default: cuda, else mps, else cpu)")
    args = ap.parse_args()
    if args.n_tags < 2:
        ap.error("--n-tags must be at least 2")
    args.model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    if unknown := set(args.model_keys) - set(MODEL_NAMES):
        ap.error(f"unknown models: {', '.join(sorted(unknown))}")
    span = None if args.max_span_cm < 0 else args.max_span_cm
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    walls = find_walls()
    print(f"experiments: {', '.join(exps)}\nloading θ̂, α̂ …", flush=True)
    bidir = {e: load_bidir(e) for e in exps}
    link_index, th, al, freqs = load_links(positions, exps, bidir, (args.freq_min, args.freq_max))
    tok, glob = link_features(th, al, freqs, args.attn)

    meta, idx, pairs, rel = build_clusters(positions, exps, walls, link_index, args.n_tags, span, args.max_clusters_per_exp, args.seed)
    if meta.empty:
        raise SystemExit("no clusters match --n-tags / --max-span-cm")
    y = meta["wall_cm"].to_numpy()
    ctx = rel if args.rel_pos else np.zeros((len(meta), 0))
    print(f"{len(meta)} clusters of {args.n_tags} tags ({idx.shape[1]} links each), span {meta.span_cm.min():.0f}–"
          f"{meta.span_cm.max():.0f} cm, {meta.labels.nunique()} distinct label sets; {len(freqs)} freqs "
          f"{freqs.min()}–{freqs.max()} MHz; device {device}"
          f"{f'; + {ctx.shape[1]} relative tag location features' if args.rel_pos else ''}")
    print(f"target: {y.min():.0f}–{y.max():.0f} cm, mean {y.mean():.0f}, std {y.std():.0f}; closest wall: "
          f"{meta.wall.value_counts().to_dict()}\n", flush=True)

    relation = shared_tag_matrix(args.n_tags).to(device)
    pooled, table = leave_one_experiment_out(tok, glob, idx, ctx, pairs, y, meta, relation, args, device)

    done = ~np.isnan(pooled["mean of train"])
    y_done, preds = y[done], {name: p[done] for name, p in pooled.items()}
    summary = pd.DataFrame({name: {**metrics(y_done, p), f"≤ {GOAL_CM[1]} cm (%)": 100 * (np.abs(p - y_done) <= GOAL_CM[1]).mean(),
                                   "median |error| (cm)": np.median(np.abs(p - y_done)),
                                   "mean train MAE (cm)": table[f"{name} train MAE"].mean()}
                            for name, p in preds.items()}).T
    print("\nall held-out predictions pooled:")
    print(summary.round(2).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"n{args.n_tags}_span{'all' if span is None else f'{span:g}cm'}" + ("" if args.attn else "_phase_only") \
        + ("" if args.purge else "_no_purge") + ("_relpos" if args.rel_pos else "")
    n_train, n_test = table["train clusters"], table["test clusters"]
    per_fold = lambda s: f"{s.min()}" if s.min() == s.max() else f"{s.min()}–{s.max()} (mean {s.mean():.0f})"
    sizes = f"train {per_fold(n_train)} clusters per fold, test {per_fold(n_test)} clusters per fold, " \
            f"{len(table)} folds, {n_test.sum()} held-out predictions"
    print(f"\ndata sizes: {sizes}")
    title = f"{args.n_tags} tags, {idx.shape[1]} links per cluster, span ≤ " \
            f"{'any' if span is None else f'{span:g} cm'}{', + relative tag locations' if args.rel_pos else ''}, " \
            f"leave one experiment out\n{sizes}"
    meta[done].reset_index(drop=True).assign(**{f"pred_{k}": v for k, v in preds.items()}).to_csv(
        OUT_DIR / f"predictions_{tag}.csv", index=False)
    summary.to_csv(OUT_DIR / f"summary_{tag}.csv")
    table.to_csv(OUT_DIR / f"folds_{tag}.csv")
    plot_predictions(y_done, preds, OUT_DIR / f"pred_vs_true_{tag}.png", title)
    plot_errors(y_done, preds, OUT_DIR / f"errors_{tag}.png", title)
    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/{{predictions,summary,folds,pred_vs_true,errors}}_{tag}.*")


if __name__ == "__main__":
    main()

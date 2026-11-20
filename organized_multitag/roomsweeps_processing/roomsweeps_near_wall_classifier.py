"""Detect whether a tag is less than 100 cm from a wall, from the phase and attenuation of all of its links under 2 m.

One sample is one tag placement (one tag in one experiment). Its input is the set of links between that tag and every other
tag of the experiment whose nominal length is below --max-link-cm (default 200 cm). Its label is

    near wall = min over the four walls of the perpendicular distance from the tag to that wall < --wall-cm (default 100 cm)

Per-link features are the ones of roomsweeps_wall_distance_multitag.py (RF only, one median over runs per link):

    tokens   per frequency: [θ̂ (unwrapped) minus its straight-line fit over the band, log α̂ minus its band mean]
    globals  per link:      [slope of that line, band mean of log α̂]

Links have no natural order and a tag can have a different number of them, so every model merges them independently of order:

- Logistic regression and RBF SVM   mean and std across links of each token, and mean, std, min and max of each global.
- DeepSets                          one shared MLP embeds each link; the mean, max and std of the embeddings feed the classifier.

The classifiers never receive tag numbers, tag locations or link lengths. Tag locations are only used to choose which links
belong to a tag (shorter than --max-link-cm) and to compute the label.

Shortcuts to be aware of. In the leapfrog sweep (experiments 1 to 15) every physical tag stays in one row of the array, and the
rows run away from one wall, so tag number and array slot nearly give the label away. Three reference baselines, which are
separate from the classifiers and are the only place this information is used, measure this:

    majority class     always predicts the more common training label
    tag number         the training near-wall rate of that physical tag
    link lengths only  RBF SVM on the sorted nominal lengths of the tag's links (no RF at all)

The "shortcut-wrong" column is the accuracy on the placements that the near-wall rate of their physical tag in the other
experiments gets wrong. This set is the same in every leave-one-out scheme. A model that only recognises the tag scores near
0 % there, and one that senses the wall scores well above 50 %.

Evaluation is leave one group out, for each scheme in --loo:

    experiment  all tags of one experiment
    tag         all placements of one physical tag
    label       all placements on one floor label
    column      all placements in one grid column
    row         all placements in one grid row
    wall        all placements whose closest wall is that wall

In every fold, training tags standing on a held-out floor label are dropped, and so is every training link whose label pair
was measured by a held-out tag. Training tags left with fewer than --min-links links are dropped too (--no-purge: keep all).

    python roomsweeps_near_wall_classifier.py                         # all schemes
    python roomsweeps_near_wall_classifier.py --loo experiment,tag    # only some schemes
    python roomsweeps_near_wall_classifier.py --wall-cm 75 --max-link-cm 150
    python roomsweeps_near_wall_classifier.py --models svm --no-attn
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from roomsweeps_wall_distance import (BASELINE_COLOR, DATA_DIR, FREQ_RANGE, PART_COLORS, ROOT, TILE_CM, all_positions,
                                      find_walls, grid_xy, label_grid, load_bidir, wall_distance_cm)
from roomsweeps_wall_distance_multitag import link_features, load_links, style

OUT_DIR = Path(__file__).resolve().parent / "roomsweeps_near_wall_classifier_out"
MODEL_NAMES = {"logreg": "Logistic regression", "svm": "RBF SVM", "deepsets": "DeepSets"}
BASELINES = ["majority class", "tag number", "link lengths only"]
LOO_SCHEMES = {"experiment": "experiment", "tag": "physical tag", "label": "floor label", "column": "grid column",
               "row": "grid row", "wall": "closest wall"}


# ---------------------------------------------------------------- samples

def build_samples(positions, exps, walls, link_index, n_links, max_link_cm, wall_cm, min_links):
    """One row per tag placement: (meta frame, link indices (N, K) padded with -1, mask (N, K), nominal length of each link (cm),
    label-pair id of each link)."""
    meta, sets, length_cm = [], [], np.full(n_links, np.nan)
    link_pair, pair_id = np.full(n_links, -1), {}
    for exp in exps:
        pos = positions[exp]
        for tag in sorted(pos.index):
            label = int(pos.loc[tag, "label"])
            grid = np.array(label_grid(label), float)
            links = []
            for other in sorted(pos.index):
                other_label = int(pos.loc[other, "label"])
                length = TILE_CM * np.linalg.norm(grid - np.array(label_grid(other_label), float))
                link = link_index.get((exp, min(tag, other), max(tag, other)))
                if other != tag and length < max_link_cm and link is not None:
                    links.append(link)
                    length_cm[link] = length
                    link_pair[link] = pair_id.setdefault(tuple(sorted((label, other_label))), len(pair_id))
            if len(links) < min_links:
                continue
            dists = {name: wall_distance_cm(grid_xy(*label_grid(label)), w) for name, w in walls.items()}
            wall = min(dists, key=dists.get)
            col, row = label_grid(label)
            meta.append({"exp": exp, "tag": tag, "label": label, "column": col, "row": row, "n_links": len(links),
                         "wall": wall.split(" (", 1)[0],
                         "wall_cm": dists[wall], "near_wall": int(dists[wall] < wall_cm)})
            sets.append(links)
    K = max(map(len, sets), default=0)
    idx = np.array([s + [-1] * (K - len(s)) for s in sets])
    return pd.DataFrame(meta), idx, idx >= 0, length_cm, link_pair


def fold_features(tok, glob, idx, mask, tr):
    """RF features for one fold, standardized on the training links only."""
    train_links = np.unique(idx[tr][mask[tr]])
    tok_mu, tok_sd = tok[train_links].mean((0, 1)), tok[train_links].std((0, 1)) + 1e-6
    glob_mu, glob_sd = glob[train_links].mean(0), glob[train_links].std(0) + 1e-6
    return (tok - tok_mu) / tok_sd, (glob - glob_mu) / glob_sd


# ---------------------------------------------------------------- classical models and baselines

def pooled_features(tok, glob, idx, mask):
    """One fixed vector per tag that does not depend on the order or number of its links."""
    t = np.where(mask[:, :, None, None], tok[idx], np.nan)      # (N, K, F, C)
    g = np.where(mask[:, :, None], glob[idx], np.nan)           # (N, K, G)
    N = len(idx)
    return np.concatenate([np.nanmean(t, 1).reshape(N, -1), np.nanstd(t, 1).reshape(N, -1),
                           np.nanmean(g, 1), np.nanstd(g, 1), np.nanmin(g, 1), np.nanmax(g, 1)], axis=1)


def fit_classical(key, X, y, tr, te):
    """Decision scores on (te, tr); a score above 0 means near the wall."""
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000) if key == "logreg" else \
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    model = make_pipeline(StandardScaler(), clf).fit(X[tr], y[tr])
    return model.decision_function(X[te]), model.decision_function(X[tr])


def tag_number_baseline(tags, y, tr, te):
    """Training near-wall rate of the physical tag, relative to the overall training rate."""
    base = y[tr].mean()
    rate = pd.Series(y[tr]).groupby(tags[tr]).mean()
    score = lambda rows: pd.Series(tags[rows]).map(rate).fillna(base).to_numpy() - base
    return score(te), score(tr)


def link_lengths_baseline(length_cm, idx, mask, y, tr, te):
    """RBF SVM on the sorted nominal lengths of the tag's links, with no RF input."""
    X = -np.sort(-np.where(mask, length_cm[idx], 0.0), axis=1)
    return fit_classical("svm", X, y, tr, te)


# ---------------------------------------------------------------- DeepSets

class DeepSetClassifier(nn.Module):
    """Shared link MLP, then the mean, max and std over the tag's links. Outputs one logit."""

    def __init__(self, n_inputs, width=64, dropout=0.2):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_inputs, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, width), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, x, mask):
        z = self.phi(x)
        m = mask[..., None].float()
        n = m.sum(1)
        mean = (z * m).sum(1) / n
        std = (((z - mean[:, None]) ** 2 * m).sum(1) / n + 1e-8).sqrt()
        return self.rho(torch.cat([mean, z.masked_fill(~mask[..., None], -torch.inf).amax(1), std], -1)).squeeze(-1)


def fit_deepsets(tok, glob, idx, mask, y, tr, te, args, device):
    """Logits on (te, tr); a logit above 0 means near the wall."""
    T = lambda a, kind=torch.float32: torch.as_tensor(np.asarray(a), dtype=kind, device=device)
    links = T(np.concatenate([tok.reshape(len(tok), -1), glob], 1))
    idx_t, mask_t, target = T(np.where(mask, idx, 0), torch.long), T(mask, torch.bool), T(y)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = DeepSetClassifier(links.shape[1]).to(device)
    steps = -(-len(tr) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    pos_weight = T((1 - y[tr].mean()) / max(y[tr].mean(), 1e-6))
    tr_t = torch.as_tensor(tr)
    for _ in range(args.epochs):
        model.train()
        for b in tr_t[torch.randperm(len(tr), generator=gen)].split(args.batch_size):
            b = b.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(links[idx_t[b]], mask_t[b]), target[b], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    model.eval()

    @torch.no_grad()
    def predict(rows):
        b = torch.as_tensor(rows).to(device)
        return model(links[idx_t[b]], mask_t[b]).cpu().numpy()
    return predict(te), predict(tr)


# ---------------------------------------------------------------- leave one group out

def shortcut_wrong_mask(meta):
    """Placements that the near-wall rate of their physical tag in the other experiments classifies wrongly."""
    y, exp_of, tags = meta["near_wall"].to_numpy(), meta["exp"].to_numpy(), meta["tag"].to_numpy()
    wrong = np.zeros(len(y), bool)
    for exp in np.unique(exp_of):
        te, tr = exp_of == exp, exp_of != exp
        rate = pd.Series(y[tr]).groupby(tags[tr]).mean()
        wrong[te] = (pd.Series(tags[te]).map(rate).fillna(y[tr].mean()).to_numpy() > y[tr].mean()) != (y[te] == 1)
    return wrong


def purged_fold(te, labels, link_pair, idx, mask, min_links):
    """Training rows and the link mask of one fold, with held-out floor labels and measured label pairs removed."""
    tr = np.flatnonzero(~np.isin(labels, labels[te]))
    held_pairs = np.unique(link_pair[idx[te][mask[te]]])
    fold_mask = mask.copy()
    fold_mask[tr] &= ~np.isin(link_pair[idx[tr]], held_pairs)
    tr = tr[fold_mask[tr].sum(1) >= min_links]
    others = np.setdiff1d(np.arange(len(idx)), tr)
    fold_mask[others] = mask[others]
    return tr, fold_mask


def leave_one_group_out(scheme, tok, glob, length_cm, link_pair, idx, mask, meta, args, device):
    """({model: pooled held-out scores}, per-fold table)."""
    y = meta["near_wall"].to_numpy()
    groups, labels, tags = meta[{"experiment": "exp"}.get(scheme, scheme)].to_numpy(), meta["label"].to_numpy(), meta["tag"].to_numpy()
    pooled, table = {}, []
    for value in list(dict.fromkeys(groups))[:args.max_folds]:
        t0 = time.time()
        te = np.flatnonzero(groups == value)
        tr, fold_mask = purged_fold(te, labels, link_pair, idx, mask, args.min_links) if args.purge else \
            (np.flatnonzero(groups != value), mask)
        if len(np.unique(y[tr])) < 2:
            print(f"{scheme} {value}: skipped, the training tags have only one class", flush=True)
            continue
        row = {"held out": value, "test tags": len(te), "test near wall": int(y[te].sum()), "train tags": len(tr),
               "train near wall": int(y[tr].sum()), "train links removed": int(mask[tr].sum() - fold_mask[tr].sum())}
        majority = 1.0 if y[tr].mean() >= 0.5 else -1.0
        scores = {"majority class": (np.full(len(te), majority), np.full(len(tr), majority)),
                  "tag number": tag_number_baseline(tags, y, tr, te),
                  "link lengths only": link_lengths_baseline(length_cm, idx, mask, y, tr, te)}
        t, g = fold_features(tok, glob, idx, fold_mask, tr)
        X = pooled_features(t, g, idx, fold_mask)
        for key in args.model_keys:
            scores[MODEL_NAMES[key]] = fit_deepsets(t, g, idx, fold_mask, y, tr, te, args, device) if key == "deepsets" \
                else fit_classical(key, X, y, tr, te)
        for name, (s_te, s_tr) in scores.items():
            pooled.setdefault(name, np.full(len(y), np.nan))[te] = s_te
            row[f"{name} accuracy"] = accuracy_score(y[te], s_te > 0)
            row[f"{name} train balanced accuracy"] = balanced_accuracy_score(y[tr], s_tr > 0)
        table.append(row)
        text = "  ".join(f"{n}: {100 * row[f'{n} accuracy']:.0f}" for n in scores)
        print(f"{scheme} {value}  train {len(tr):3d}  test {len(te):2d}  accuracy (%)  {text}  [{time.time() - t0:.0f} s]", flush=True)
    return pooled, pd.DataFrame(table).set_index("held out")


def bootstrap_interval(y, score, groups, seed, n_boot=2000):
    """95 % interval of the balanced accuracy, resampling whole held-out groups."""
    rng = np.random.default_rng(seed)
    rows = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    values = []
    for _ in range(n_boot):
        b = np.concatenate([rows[i] for i in rng.integers(len(rows), size=len(rows))])
        if len(np.unique(y[b])) == 2:
            values.append(balanced_accuracy_score(y[b], score[b] > 0))
    return np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan)


def summarize(y, scores, groups, table, shortcut_wrong, seed):
    out = {}
    for name, s in scores.items():
        p = s > 0
        low, high = bootstrap_interval(y, s, groups, seed)
        out[name] = {"balanced accuracy (%)": 100 * balanced_accuracy_score(y, p), "95% low (%)": 100 * low, "95% high (%)": 100 * high,
                     "accuracy (%)": 100 * accuracy_score(y, p), "ROC AUC": roc_auc_score(y, s),
                     "precision (%)": 100 * precision_score(y, p, zero_division=0), "recall (%)": 100 * recall_score(y, p),
                     "specificity (%)": 100 * recall_score(y, p, pos_label=0), "F1": f1_score(y, p, zero_division=0),
                     "shortcut-wrong accuracy (%)": 100 * accuracy_score(y[shortcut_wrong], p[shortcut_wrong]) if shortcut_wrong.any() else np.nan,
                     "mean train balanced accuracy (%)": 100 * table[f"{name} train balanced accuracy"].mean()}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- plots

def plot_results(y, scores, summary, n_shortcut, path, title):
    """ROC curves, and the balanced accuracy beside the accuracy on the placements the tag number baseline gets wrong."""
    models = [n for n in scores if n not in BASELINES]
    colors = dict(zip(models, PART_COLORS))
    dashes = {"tag number": "--", "link lengths only": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#c8c8c8", lw=1, zorder=1)
    for name in ["tag number", "link lengths only"] + models:
        fpr, tpr, _ = roc_curve(y, scores[name])
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, BASELINE_COLOR), ls=dashes.get(name, "-"),
                label=f"{name} (AUC {summary.loc[name, 'ROC AUC']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate (far tags called near)")
    ax.set_ylabel("true positive rate (near tags detected)")
    ax.set_title("ROC of held-out tags", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    names = BASELINES + models
    pos = np.arange(len(names))[::-1]
    series = [("balanced accuracy (%)", "balanced accuracy, all held-out tags", "o", PART_COLORS[0], 0.14),
              ("shortcut-wrong accuracy (%)", f"accuracy, the {n_shortcut} placements where the tag number shortcut is wrong", "D", PART_COLORS[1], -0.14)]
    ax.axvline(50, color="#9a9a9a", lw=1, ls="--", zorder=1)
    for column, label, marker, color, shift in series:
        values = summary.loc[names, column].to_numpy()
        ax.scatter(values, pos + shift, s=70, marker=marker, color=color, edgecolors="white", linewidths=1.5, zorder=3, label=label)
        for v, p in zip(values, pos + shift):
            ax.annotate(f"{v:.0f}", (v, p), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color="#444444")
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_xlim(-3, 108)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("percent (dashed line: chance)")
    ax.set_title("Performance against the shortcuts", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    for ax in axes:
        style(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(comparison, path, title):
    """Balanced accuracy of every model under every leave-one-out scheme."""
    schemes = list(dict.fromkeys(comparison["scheme"]))
    names = [n for n in dict.fromkeys(comparison["model"]) if n not in ("majority class", "link lengths only")]
    models = [n for n in names if n != "tag number"]
    colors = {**dict(zip(models, PART_COLORS)), "tag number": BASELINE_COLOR}
    markers = dict(zip(names, ["X", "o", "s", "^"] if "tag number" == names[0] else ["o", "s", "^", "X"]))
    pos = dict(zip(schemes, np.arange(len(schemes))[::-1]))
    fig, ax = plt.subplots(figsize=(9.5, 1.0 * len(schemes) + 2.2))
    ax.axvline(50, color="#9a9a9a", lw=1, ls="--", zorder=1)
    for k, name in enumerate(names):
        d = comparison[comparison["model"] == name]
        y = d["scheme"].map(pos).to_numpy() + 0.24 - 0.48 * k / max(len(names) - 1, 1)
        ax.scatter(d["balanced accuracy (%)"], y, s=64, marker=markers[name], color=colors[name], edgecolors="white", linewidths=1.2,
                   zorder=3, label=name)
    ax.set_yticks(list(pos.values()))
    ax.set_yticklabels([LOO_SCHEMES[s] for s in schemes])
    ax.set_xlim(30, 100)
    ax.set_xlabel("balanced accuracy of held-out tags (%), dashed line: chance")
    ax.set_ylabel("group held out")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(names))
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wall-cm", type=float, default=100.0, help="a tag closer than this to a wall is labelled near wall")
    ap.add_argument("--max-link-cm", type=float, default=200.0, help="use the tag's links shorter than this nominal length")
    ap.add_argument("--min-links", type=int, default=3, help="skip tags with fewer links than this")
    ap.add_argument("--loo", default=",".join(LOO_SCHEMES), help="comma-separated leave-one-out schemes: " + ", ".join(LOO_SCHEMES))
    ap.add_argument("--models", default="logreg,svm,deepsets", help="comma-separated: logreg, svm, deepsets")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--purge", action=argparse.BooleanOptionalAction, default=True,
                    help="drop training tags on held-out floor labels and training links with a held-out label pair")
    ap.add_argument("--max-folds", type=int, default=None, help="only hold out the first this many groups of each scheme")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150, help="DeepSets training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="DeepSets learning rate")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--device", default=None, help="torch device (default: cuda, else cpu)")
    args = ap.parse_args()
    args.model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    if unknown := set(args.model_keys) - set(MODEL_NAMES):
        ap.error(f"unknown models: {', '.join(sorted(unknown))}")
    args.loo_keys = [m.strip() for m in args.loo.split(",") if m.strip()]
    if unknown := set(args.loo_keys) - set(LOO_SCHEMES):
        ap.error(f"unknown leave-one-out schemes: {', '.join(sorted(unknown))}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    walls = find_walls()
    print(f"experiments: {', '.join(exps)}\nloading θ̂, α̂ …", flush=True)
    bidir = {e: load_bidir(e) for e in exps}
    link_index, th, al, freqs = load_links(positions, exps, bidir, (args.freq_min, args.freq_max))
    tok, glob = link_features(th, al, freqs, args.attn)

    meta, idx, mask, length_cm, link_pair = build_samples(positions, exps, walls, link_index, len(th), args.max_link_cm, args.wall_cm, args.min_links)
    if meta.empty:
        raise SystemExit("no tags match --max-link-cm / --min-links")
    y = meta["near_wall"].to_numpy()
    print(f"{len(meta)} tag placements on {meta.label.nunique()} floor labels, {meta.n_links.min()}–{meta.n_links.max()} links each "
          f"(< {args.max_link_cm:g} cm); {len(freqs)} freqs {freqs.min()}–{freqs.max()} MHz; device {device}")
    print(f"near wall (< {args.wall_cm:g} cm): {y.sum()} of {len(y)} ({100 * y.mean():.0f} %); closest wall distance "
          f"{meta.wall_cm.min():.0f}–{meta.wall_cm.max():.0f} cm\n", flush=True)

    shortcut_wrong = shortcut_wrong_mask(meta)
    OUT_DIR.mkdir(exist_ok=True)
    base = f"wall{args.wall_cm:g}cm_links{args.max_link_cm:g}cm" + ("" if args.attn else "_phase_only") \
        + ("" if args.purge else "_no_purge")
    comparison = []
    for scheme in args.loo_keys:
        print(f"\n=== leave one {LOO_SCHEMES[scheme]} out ===", flush=True)
        pooled, table = leave_one_group_out(scheme, tok, glob, length_cm, link_pair, idx, mask, meta, args, device)
        done = ~np.isnan(pooled["majority class"])
        y_done, scores, wrong = y[done], {name: s[done] for name, s in pooled.items()}, shortcut_wrong[done]
        groups = meta[{"experiment": "exp"}.get(scheme, scheme)].to_numpy()[done]
        summary = summarize(y_done, scores, groups, table, wrong, args.seed)
        print(f"\nall held-out tags pooled ({len(y_done)} tags, {len(table)} folds, {wrong.sum()} shortcut-wrong placements):")
        print(summary.round(2).to_string(), flush=True)

        tag = f"loo_{scheme}_{base}"
        title = f"Tag within {args.wall_cm:g} cm of a wall, from its links under {args.max_link_cm:g} cm, leave one {LOO_SCHEMES[scheme]} out\n" \
                f"{len(y_done)} held-out tags, {len(table)} folds, train {table['train tags'].min()}–{table['train tags'].max()} tags per fold"
        meta[done].reset_index(drop=True).assign(shortcut_wrong=wrong, **{f"score_{k}": v for k, v in scores.items()}).to_csv(
            OUT_DIR / f"predictions_{tag}.csv", index=False)
        summary.to_csv(OUT_DIR / f"summary_{tag}.csv")
        table.to_csv(OUT_DIR / f"folds_{tag}.csv")
        plot_results(y_done, scores, summary, int(wrong.sum()), OUT_DIR / f"results_{tag}.png", title)
        comparison.append(summary.reset_index(names="model").assign(scheme=scheme, folds=len(table), tags=len(y_done)))

    comparison = pd.concat(comparison, ignore_index=True)
    comparison.to_csv(OUT_DIR / f"loo_comparison_{base}.csv", index=False)
    plot_comparison(comparison, OUT_DIR / f"loo_comparison_{base}.png",
                    f"Tag within {args.wall_cm:g} cm of a wall, from its links under {args.max_link_cm:g} cm")
    print("\nbalanced accuracy (%) by leave-one-out scheme:")
    print(comparison.pivot(index="scheme", columns="model", values="balanced accuracy (%)").loc[args.loo_keys].round(1).to_string())
    print("\nshortcut-wrong accuracy (%) by leave-one-out scheme:")
    print(comparison.pivot(index="scheme", columns="model", values="shortcut-wrong accuracy (%)").loc[args.loo_keys].round(1).to_string())
    print(f"\nwrote {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

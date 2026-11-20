"""Is rfgen_wall_distance_alllinks.py learning anything from the RF? Two controls on the same links, features and
leave-one-experiment-out folds (held-out label pairs dropped from training, as --loo with the default --loo-dedup):

1. geometry-only baselines: predict each test link from the training labels alone, using no RF at all
     train mean / train median
     mean of the training links in the same angle class
     mean of the training links with the same angle class, hit wall and nominal length (falling back to angle class,
     then the train mean)
   Since link direction, length and angle are never model features, a model that only beats the train mean but not
   these is at best recovering link geometry from the RF, not the wall distance.

2. shuffled-label controls: retrain every model with the training labels permuted (the test labels stay true), repeated
   --n-shuffles times, giving the score each model gets by chance
     full       labels permuted across all training links: no link between RF and distance survives
     geometry   labels permuted only among training links with the same angle class, hit wall and nominal length: what
                link geometry says about the distance survives, what the RF says about where the link sits (e.g. the
                row nearest the wall vs the next one) does not
   The real model must beat the "full" null to have learned anything, and the "geometry" null to have learned
   something beyond link geometry.

Scores: pooled test MAE, and for the links parallel to their wall the AUC of the predictions separating far (> 75 cm)
from near (< 50 cm) links (0.5 = chance) and their Spearman ρ with the truth. The real models are also refitted with
--seeds seeds to show how much the score moves with the training seed alone.

    python rfgen_wall_distance_alllinks_controls.py --min-length-cm 46 --max-length-cm 150 --no-unwrap --max-wall-cm 100
    python rfgen_wall_distance_alllinks_controls.py --min-length-cm 46 --max-length-cm 82 --no-unwrap --max-wall-cm 100
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import mean_absolute_error

from rfgen_wall_distance import (DATA_DIR, FREQ_RANGE, BASELINE_COLOR, PART_COLORS, ROOT, all_positions,
                                 build_dataset, find_walls, load_bidir, select_links)
from rfgen_wall_distance_alllinks import OUT_DIR, all_links, make_fit_fn

NEAR_CM, FAR_CM = 50.0, 75.0      # parallel links: near = the tag row nearest the wall, far = the next row


def loo_folds(meta, loo_dedup=True):
    """[(exp, train idx, test idx)] as rfgen_wall_distance.evaluate with loo (and loo_dedup)."""
    exp_of = meta["exp"].to_numpy()
    pair_of = np.array(["-".join(map(str, sorted(p))) for p in zip(meta["label_a"], meta["label_b"])])
    folds = []
    for exp in dict.fromkeys(exp_of):
        te, tr = np.flatnonzero(exp_of == exp), np.flatnonzero(exp_of != exp)
        if loo_dedup:
            tr = tr[~np.isin(pair_of[tr], pair_of[te])]
        folds.append((exp, tr, te))
    return folds


def geometry_keys(meta):
    return pd.DataFrame({"angle": meta["angle_class"].to_numpy(), "wall": meta["wall"].to_numpy(),
                         "len": meta["length_cm"].round().to_numpy()})


def baselines(y, meta, folds):
    """{name: pooled test predictions} of the geometry-only baselines."""
    keys = geometry_keys(meta)
    out = {k: np.full(len(y), np.nan) for k in
           ["train mean", "train median", "mean | angle", "mean | angle, wall, length"]}
    for _, tr, te in folds:
        ytr = pd.Series(y[tr])
        out["train mean"][te] = ytr.mean()
        out["train median"][te] = ytr.median()
        by_angle = ytr.groupby(keys["angle"].iloc[tr].to_numpy()).mean()
        by_geom = ytr.groupby([keys[c].iloc[tr].to_numpy() for c in keys]).mean()
        for i in te:
            k = tuple(keys.iloc[i])
            out["mean | angle"][i] = by_angle.get(k[0], ytr.mean())
            out["mean | angle, wall, length"][i] = by_geom.get(k, out["mean | angle"][i])
    return out


def shuffle_labels(y, tr, groups, kind, rng):
    """y with its training entries permuted (all of them, or within each geometry group); test entries untouched."""
    yy = y.copy()
    if kind == "full":
        yy[tr] = rng.permutation(y[tr])
    else:
        for g in np.unique(groups[tr]):
            idx = tr[groups[tr] == g]
            yy[idx] = rng.permutation(y[idx])
    return yy


def run_loo(X, y, folds, fit_fn, seed, kind=None, groups=None, shuffle_seed=None):
    """Pooled test predictions {model: array} and mean training MAE {model: float} over the folds."""
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(shuffle_seed)
    pooled, train_mae = {}, {}
    for _, tr, te in folds:
        yy = y if kind is None else shuffle_labels(y, tr, groups, kind, rng)
        preds, tmae = fit_fn(X, yy, tr, te, seed)
        preds.pop("mean of train", None)
        for name, p in preds.items():
            pooled.setdefault(name, np.full(len(y), np.nan))[te] = p
            train_mae.setdefault(name, []).append(tmae.get(name, np.nan))
    return pooled, {k: float(np.nanmean(v)) for k, v in train_mae.items()}


def scores(y, p, meta):
    """MAE, and on parallel links the far-vs-near AUC and Spearman ρ."""
    par = meta["angle_class"].str.startswith("parallel").to_numpy()
    near, far = par & (y < NEAR_CM), par & (y > FAR_CM)
    auc = mannwhitneyu(p[far], p[near]).statistic / (far.sum() * near.sum()) if near.any() and far.any() else np.nan
    return {"MAE": mean_absolute_error(y, p), "AUC far/near (parallel)": auc,
            "rho (parallel)": spearmanr(p[par], y[par])[0] if par.sum() > 2 else np.nan}


def plot_controls(runs, base, path, title):
    models = list(dict.fromkeys(runs["model"]))
    colors = dict(zip(models, PART_COLORS))
    metrics = ["MAE", "AUC far/near (parallel)"]
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(4.2 * len(models), 3.6 * len(metrics)), squeeze=False)
    rows = [("shuffled: full", "full shuffle"), ("shuffled: geometry", "geometry-kept shuffle"), ("real", "real (seeds)")]
    for j, model in enumerate(models):
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            for r, (kind, label) in enumerate(rows):
                v = runs.loc[(runs.model == model) & (runs.kind == kind), metric].to_numpy()
                jitter = np.random.default_rng(r).uniform(-0.12, 0.12, len(v))
                real = kind == "real"
                ax.scatter(v, r + jitter, s=34 if real else 18, color=colors[model] if real else "#9a9a9a",
                           edgecolors="white", linewidths=0.8, zorder=3)
                if len(v) and not real:
                    lo, hi = np.percentile(v, [5, 95])
                    ax.plot([lo, hi], [r, r], color="#555555", lw=2, zorder=2, solid_capstyle="round")
            if metric == "MAE":
                for name, ls in [("train mean", "--"), ("mean | angle", ":"), ("mean | angle, wall, length", "-.")]:
                    ax.axvline(base[name], color=BASELINE_COLOR, lw=1.2, ls=ls, zorder=1,
                               label=f"{name} ({base[name]:.1f})")
                if j == 0:
                    ax.legend(title="geometry-only baseline", frameon=False, fontsize=7, title_fontsize=7,
                              loc="lower left")
            else:
                ax.axvline(0.5, color=BASELINE_COLOR, lw=1.2, ls="--", zorder=1)
            ax.set_yticks(range(len(rows)), [lab for _, lab in rows] if j == 0 else [])
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_xlabel(f"{metric}{' (cm, lower is better)' if metric == 'MAE' else ' (0.5 = chance)'}", fontsize=9)
            if i == 0:
                ax.set_title(model, fontsize=11)
            ax.grid(axis="x", color="#e0e0e0", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-length-cm", type=float, default=None)
    ap.add_argument("--max-length-cm", type=float, default=None)
    ap.add_argument("--max-angle-deg", type=float, default=None)
    ap.add_argument("--max-wall-cm", type=float, default=None)
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0])
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1])
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--zero-start", action="store_true")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--loo-dedup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nets", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tf-epochs", type=int, default=100)
    ap.add_argument("--tf-lr", type=float, default=1e-3)
    ap.add_argument("--rescnn-epochs", type=int, default=60)
    ap.add_argument("--rescnn-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--n-shuffles", type=int, default=20, help="label permutations per shuffle kind")
    ap.add_argument("--seeds", type=int, default=3, help="training seeds for the real (unshuffled) models")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    walls = find_walls()
    links = all_links(positions, exps, walls, args.min_length_cm, args.max_length_cm, args.max_angle_deg,
                      args.max_wall_cm)
    bidir = {e: load_bidir(e) for e in exps}
    links = select_links(links, bidir, (args.freq_min, args.freq_max), args.dedup)
    X, y, meta, freqs = build_dataset(links, bidir, (args.freq_min, args.freq_max), args.attn, args.zero_start,
                                      args.unwrap)
    meta = meta.reset_index(drop=True)
    folds = loo_folds(meta, args.loo_dedup)
    fit_fn = make_fit_fn(args, freqs)
    groups = geometry_keys(meta).astype(str).agg("|".join, axis=1).to_numpy()
    print(f"{len(y)} samples, {X.shape[1]} features, {len(folds)} folds, "
          f"{len(set(groups))} geometry groups (angle class × wall × length)")

    tag = (f"{args.min_length_cm or 0:g}-{args.max_length_cm or 'inf'}cm"
           + (f"_max{args.max_angle_deg:g}deg" if args.max_angle_deg is not None else "")
           + ("" if args.attn else "_phase_only") + ("_zero_start" if args.zero_start else "")
           + ("" if args.unwrap else "_no_unwrap") + ("_dedup" if args.dedup else "")
           + ("_loo-dedup" if args.loo_dedup else "_loo")
           + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else ""))

    # 1. geometry-only baselines
    base_preds = baselines(y, meta, folds)
    base_scores = pd.DataFrame({k: scores(y, p, meta) for k, p in base_preds.items()}).T

    # 2. real models (several seeds) and shuffled-label nulls, all in parallel
    jobs = [("real", s, None, None) for s in range(args.seeds)]
    jobs += [(f"shuffled: {kind}", 0, kind, 1000 * (k + 1) + i)
             for k, kind in enumerate(["full", "geometry"]) for i in range(args.n_shuffles)]
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_loo)(X, y, folds, fit_fn, seed, kind, groups, sh) for _, seed, kind, sh in jobs)
    print(f"{len(jobs)} leave-one-experiment-out runs in {time.time() - t0:.0f} s")

    rows = []
    for (label, seed, _, sh), (pooled, tmae) in zip(jobs, results):
        for model, p in pooled.items():
            rows.append({"kind": label, "seed": seed, "shuffle_seed": sh, "model": model,
                         "train MAE": tmae[model], **scores(y, p, meta)})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_DIR / f"controls_runs_{tag}.csv", index=False)

    # summary
    pd.set_option("display.width", 200)
    print("\n=== geometry-only baselines (no RF) ===")
    print(base_scores.round(2).to_string())
    print("\n=== models: real (mean over seeds) vs shuffled-label nulls (median [5–95 %]) ===")
    summary = []
    for model in dict.fromkeys(runs.model):
        r = runs[runs.model == model]
        real = r[r.kind == "real"]
        row = {"model": model}
        for metric in ["MAE", "AUC far/near (parallel)", "rho (parallel)", "train MAE"]:
            row[f"real {metric}"] = f"{real[metric].mean():.2f} ±{real[metric].std():.2f}"
            for kind in ["full", "geometry"]:
                v = r.loc[r.kind == f"shuffled: {kind}", metric]
                row[f"{kind} {metric}"] = f"{v.median():.2f} [{v.quantile(.05):.2f}–{v.quantile(.95):.2f}]"
                if metric != "train MAE":
                    better = (v <= real[metric].mean()) if metric == "MAE" else (v >= real[metric].mean())
                    row[f"{kind} {metric} p"] = f"{(better.sum() + 1) / (len(v) + 1):.3f}"
        summary.append(row)
    summary = pd.DataFrame(summary).set_index("model").T
    print(summary.to_string())
    summary.to_csv(OUT_DIR / f"controls_summary_{tag}.csv")
    base_scores.to_csv(OUT_DIR / f"controls_baselines_{tag}.csv")

    plot_controls(runs, base_scores["MAE"], OUT_DIR / f"controls_{tag}.png",
                  f"all links ({tag}): real models vs shuffled-label nulls and geometry-only baselines "
                  f"(leave one experiment out; {args.n_shuffles} shuffles per kind, {args.seeds} real seeds)")
    print(f"\nwrote {(OUT_DIR / f'controls_{tag}.png').relative_to(ROOT)} and controls_*_{tag}.csv")


if __name__ == "__main__":
    main()

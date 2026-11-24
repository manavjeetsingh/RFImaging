"""Wall-distance error per link-length bucket, for several --max-wall-cm limits, with roomsweeps_wall_distance_alllinks.py.

For every max wall distance in --max-wall-cm (default 50,100,150,200) and every nominal link-length bucket in
--buckets (default 40-80,80-120,120-160,160-200 cm, each [lo, hi)), the links of that bucket within that distance of
the wall are taken (all_links), and the models of roomsweeps_wall_distance_alllinks.py (SVR, MLP and, unless --no-nets,
the scratch transformer and ResCNN) are trained and tested on them alone, by leaving one experiment out (--loo, with
the held-out label pairs dropped from training; --no-loo-dedup keeps them). θ̂ is not unwrapped (--no-unwrap) unless
--unwrap is given. Each model gets one figure with a subplot per max wall distance: x = length bucket, y = box plot
of the model's held-out absolute errors (--signed: predicted − true), next to the mean-of-train baseline's.

    python roomsweeps_wall_distance_alllinks_buckets.py
    python roomsweeps_wall_distance_alllinks_buckets.py --no-nets                # SVR / MLP only (much faster)
    python roomsweeps_wall_distance_alllinks_buckets.py --max-wall-cm 100,200 --buckets 40-80,80-120
    python roomsweeps_wall_distance_alllinks_buckets.py --signed                 # signed instead of absolute errors

Writes roomsweeps_wall_distance_alllinks_buckets_out/errors_<tag>.csv (every held-out prediction) and
box_<model>_<tag>.png per model (svr, mlp, transformer, rescnn).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from roomsweeps_wall_distance import (DATA_DIR, FREQ_RANGE, GOAL_CM, BASELINE_COLOR, PART_COLORS, ROOT, all_positions,
                                      build_dataset, evaluate, find_walls, load_bidir)
from roomsweeps_wall_distance_alllinks import all_links, make_fit_fn
from roomsweeps_wall_distance import (add_data_version_arg, has_experiment,  # noqa: E402
                                      set_data_version, version_tag)

OUT_DIR = Path(__file__).resolve().parent / "roomsweeps_wall_distance_alllinks_buckets_out"
MIN_SAMPLES = 10          # buckets with fewer samples are skipped


def parse_buckets(s):
    return [tuple(float(v) for v in b.split("-")) for b in s.split(",")]


def model_slug(model):
    return model.lower().replace(" (scratch)", "").replace(" ", "_")


def plot_model(df, model, buckets, max_walls, signed, path, title):
    """One figure for one model: a subplot per max wall distance, x = length bucket, a box of the model's held-out
    errors per bucket next to a grey box of the mean-of-train baseline's."""
    labels = [f"{lo:g}–{hi:g}" for lo, hi in buckets]
    color = PART_COLORS[0]
    fig, axes = plt.subplots(1, len(max_walls), figsize=(3.6 * len(max_walls) + 1.5, 5.2), sharey=True,
                             squeeze=False)
    for ax, max_wall in zip(axes.flat, max_walls):
        d = df[df["max_wall_cm"] == max_wall]
        for name, c, off in [("mean of train", BASELINE_COLOR, -0.2), (model, color, 0.2)]:
            data, pos = [], []
            for k, label in enumerate(labels):
                e = d.loc[(d["bucket"] == label) & (d["model"] == name), "error"].to_numpy()
                if len(e):
                    data.append(e)
                    pos.append(k + off)
            if data:
                ax.boxplot(data, positions=pos, widths=0.34, patch_artist=True,
                           boxprops=dict(facecolor=c, alpha=0.75, edgecolor="#333333"),
                           medianprops=dict(color="#111111", lw=1.5), whiskerprops=dict(color="#555555"),
                           capprops=dict(color="#555555"),
                           flierprops=dict(marker="o", ms=2.5, mfc=c, mec="none", alpha=0.6))
        if signed:
            ax.axhline(0, color="#4a4a4a", lw=1, zorder=0)
            ax.axhspan(-GOAL_CM[1], GOAL_CM[1], color="#1baf7a", alpha=0.12, lw=0, zorder=0)
        else:
            ax.axhspan(0, GOAL_CM[1], color="#1baf7a", alpha=0.12, lw=0, zorder=0)
        counts = d[d["model"] == model].groupby("bucket").size()
        maes = d[d["model"] == model].groupby("bucket")["error"].apply(lambda e: e.abs().mean())
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{l}\nn={counts.get(l, 0)}" + (f"\nMAE {maes[l]:.0f}" if l in maes else "")
                            for l in labels], fontsize=8)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_xlabel("link length bucket (cm)")
        ax.set_title(f"≤ {max_wall:g} cm from wall", fontsize=10)
        ax.grid(axis="y", color="#e0e0e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if not signed:
        axes.flat[0].set_ylim(bottom=0)
    axes.flat[0].set_ylabel("predicted − true (cm)" if signed else "absolute error (cm)")
    axes.flat[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75, ec="#333333", label=n)
                                  for n, c in [(model, color), ("mean of train", BASELINE_COLOR)]],
                         frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buckets", default="40-80,80-120,120-160,160-200",
                    help="nominal link-length buckets in cm, lo-hi[,lo-hi…], each [lo, hi)")
    ap.add_argument("--max-wall-cm", default="50,100,150,200",
                    help="max distance to the wall along the perpendicular, comma-separated; one plot each")
    ap.add_argument("--signed", action="store_true", help="plot predicted − true instead of the absolute error")
    ap.add_argument("--max-angle-deg", type=float, default=None,
                    help="only links within this angle of the wall their perpendicular hits (0 = parallel only)")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=False,
                    help="unwrap θ̂ along frequency (default --no-unwrap: raw θ̂ in [0, π))")
    ap.add_argument("--dedup", action="store_true",
                    help="keep each label pair once, from the first experiment that measured it")
    ap.add_argument("--loo-dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="drop the training samples of every label pair the held-out experiment measured "
                         "(--no-loo-dedup: keep them)")
    ap.add_argument("--nets", action=argparse.BooleanOptionalAction, default=True,
                    help="also train a transformer and a residual CNN from scratch (--no-nets: SVR / MLP only)")
    ap.add_argument("--tf-epochs", type=int, default=100)
    ap.add_argument("--tf-lr", type=float, default=1e-3, help="peak transformer learning rate")
    ap.add_argument("--rescnn-epochs", type=int, default=60)
    ap.add_argument("--rescnn-lr", type=float, default=1e-3, help="peak ResCNN learning rate")
    ap.add_argument("--batch-size", type=int, default=32, help="transformer and ResCNN batch size")
    ap.add_argument("--weight-decay", type=float, default=1e-2, help="transformer and ResCNN weight decay")
    add_data_version_arg(ap)
    args = ap.parse_args()
    set_data_version(args.data_version)
    buckets = parse_buckets(args.buckets)
    max_walls = [float(v) for v in args.max_wall_cm.split(",")]

    positions = all_positions()
    exps = [e for e in positions if has_experiment(e)]
    print(f"experiments with a CSV: {', '.join(exps)}")
    walls = find_walls()
    print("loading θ̂, α̂ …")
    bidir = {e: load_bidir(e) for e in exps}

    tag = ("" if args.unwrap else "no_unwrap") + ("" if args.attn else "_phase_only") \
        + (f"_max{args.max_angle_deg:g}deg" if args.max_angle_deg is not None else "") \
        + ("_dedup" if args.dedup else "") + ("_loo-dedup" if args.loo_dedup else "_loo") \
        + ("" if args.nets else "_no_nets") + ("_signed" if args.signed else "")
    tag = tag.lstrip("_") or "unwrap"
    tag += version_tag()
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for max_wall in max_walls:
        links_all = all_links(positions, exps, walls, max_angle_deg=args.max_angle_deg, dedup=args.dedup,
                              max_wall_cm=max_wall)
        for lo, hi in buckets:
            label = f"{lo:g}–{hi:g}"
            links = links_all[(links_all["length_cm"] >= lo) & (links_all["length_cm"] < hi)].reset_index(drop=True) \
                if not links_all.empty else links_all
            print(f"\n================ max wall {max_wall:g} cm, length {label} cm: {len(links)} links")
            if len(links) < MIN_SAMPLES:
                print("  too few links, skipped")
                continue
            X, y, meta, freqs = build_dataset(links, bidir, (args.freq_min, args.freq_max), args.attn, False,
                                              args.unwrap)
            if len(X) < MIN_SAMPLES or meta["exp"].nunique() < 2:
                print("  too few samples / experiments for leave-one-experiment-out, skipped")
                continue
            te, preds = evaluate(X, y, meta, 0.7, args.seed, loo=True, fit_fn=make_fit_fn(args, freqs),
                                 loo_dedup=args.loo_dedup, keep_mean=True)
            for model, p in preds.items():
                ok = ~np.isnan(p)
                err = p[ok] - y[te][ok]
                rows += [{"max_wall_cm": max_wall, "bucket": label, "model": model, "exp": ex, "label_a": la,
                          "label_b": lb, "length_cm": l, "true_cm": t, "pred_cm": pr,
                          "error": e if args.signed else abs(e)}
                         for ex, la, lb, l, t, pr, e in zip(meta["exp"].iloc[te][ok], meta["label_a"].iloc[te][ok],
                                                             meta["label_b"].iloc[te][ok],
                                                             meta["length_cm"].iloc[te][ok], y[te][ok], p[ok], err)]

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no bucket had enough links")
    df.to_csv(OUT_DIR / f"errors_{tag}.csv", index=False)
    print("\nMAE (cm) per max wall distance, bucket and model:")
    print(df.assign(abs_err=lambda d: d["error"].abs())
          .pivot_table(index=["max_wall_cm", "bucket"], columns="model", values="abs_err", aggfunc="mean")
          .round(1).to_string())

    written = [OUT_DIR / f"errors_{tag}.csv"]
    feat = f"θ̂{'' if args.unwrap else ' (wrapped)'}{' + α̂' if args.attn else ' only'}"
    for model in (m for m in dict.fromkeys(df["model"]) if m != "mean of train"):
        path = OUT_DIR / f"box_{model_slug(model)}_{tag}.png"
        plot_model(df, model, buckets, max_walls, args.signed, path,
                   f"{model}: {'signed' if args.signed else 'absolute'} error by link length\n{feat}, "
                   f"leave one experiment out{' (held-out pairs dropped from training)' if args.loo_dedup else ''}")
        written.append(path)
    print("\nwrote " + ", ".join(str(p.relative_to(ROOT)) for p in written))


if __name__ == "__main__":
    main()

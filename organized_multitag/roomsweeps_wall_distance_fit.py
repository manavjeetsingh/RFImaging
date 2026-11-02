"""Predict a link's distance from the closest parallel wall from its phase-fit residuals (and attenuation).

Same links, target, split and models as roomsweeps_wall_distance.py (which this imports), but the phase features are
replaced by the linear phase fit saved by variance_processing.ipynb in `roomsweeps_data/<exp>/fits/phase_fit.csv`:

- Δ per frequency: unwrapped θ̂ minus its least-squares line over frequency (rad)
- the sine fit to Δ: amplitude A (rad) and period (MHz)

Features: Δ, A, period, α̂ per frequency at the same frequencies as Δ (median over runs; unless --no-attn), a one-hot
of the link direction and a one-hot of the link length. Run the notebook's save cell first.

    python roomsweeps_wall_distance_fit.py                     # 92 cm links
    python roomsweeps_wall_distance_fit.py --length-cm 46      # 1-tile links
    python roomsweeps_wall_distance_fit.py --length-cm 92,138  # 2- and 3-tile links in one dataset
    python roomsweeps_wall_distance_fit.py --no-attn           # phase-fit features only
    python roomsweeps_wall_distance_fit.py --dedup             # each label pair once, from its first experiment
    python roomsweeps_wall_distance_fit.py --loo               # leave one experiment out instead of the random split
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roomsweeps_wall_distance import (DATA_DIR, ROOT, TILE_CM, N_COLS, N_ROWS, all_positions, evaluate, find_walls,
                                      grid_xy, load_bidir, parallel_links, plot_predictions, wall_distance_cm)

FIT_FILE = Path("fits") / "phase_fit.csv"   # in roomsweeps_data/<exp>/
OUT_DIR = ROOT / "roomsweeps_wall_distance_fit_out"


def load_fits(exps):
    """{exp: phase_fit.csv indexed by (T1, T2)}; fails if the notebook hasn't written one."""
    fits = {}
    for exp in exps:
        path = DATA_DIR / exp / FIT_FILE
        if not path.exists():
            raise SystemExit(f"{path.relative_to(ROOT)} missing: run the save cell in variance_processing.ipynb")
        fits[exp] = pd.read_csv(path).set_index(["T1", "T2"])
    return fits


def build_dataset(links, fits, bidir, use_attn=True, n_tiles=(2,)):
    """One row per link: Δ per frequency, sine A, sine period, α̂ per frequency (if use_attn, median over runs),
    direction one-hot, length one-hot (over n_tiles)."""
    X, y, meta = [], [], []
    for _, link in links.iterrows():
        fit = fits[link.exp]
        if (link.T1, link.T2) not in fit.index:
            continue
        row = fit.loc[(link.T1, link.T2)]
        delta_cols = [c for c in fit.columns if c.startswith("delta_")]
        freqs = np.array([int(c.removeprefix("delta_")) for c in delta_cols])
        feats = [row[delta_cols].to_numpy(float), row[["sine_A", "sine_period_mhz"]].to_numpy(float)]
        n_runs = np.nan
        if use_attn:
            m = bidir[link.exp]
            m = m[(m["T1"] == link.T1) & (m["T2"] == link.T2) & m["freq"].isin(freqs)]
            alpha = m.pivot(index="run", columns="freq", values="alpha").reindex(columns=freqs).dropna()
            if alpha.empty:
                continue
            feats.append(np.median(alpha.to_numpy(), axis=0))
            n_runs = len(alpha)
        X.append(np.concatenate(feats + [np.eye(2)[link.direction],
                                         np.eye(len(n_tiles))[list(n_tiles).index(link.n_tiles)]]))
        y.append(link.wall_cm)
        meta.append({**link.to_dict(), "std_err": row["std_err"], "sine_A": row["sine_A"],
                     "sine_period_mhz": row["sine_period_mhz"], "n_runs": n_runs})
    return np.array(X), np.array(y), pd.DataFrame(meta), freqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length-cm", default="92",
                    help="link length(s) in cm, comma-separated (e.g. 92,138); each a whole number of tiles")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside the phase fit (--no-attn: phase fit only)")
    ap.add_argument("--dedup", action="store_true",
                    help="keep each label pair once, from the first experiment that measured it")
    ap.add_argument("--loo", action="store_true",
                    help="leave one experiment out: train on the other experiments, test on it, for every experiment")
    args = ap.parse_args()

    lengths = [float(x) for x in args.length_cm.split(",")]
    n_tiles = sorted({int(round(l / TILE_CM)) for l in lengths})
    for l in lengths:
        assert round(l / TILE_CM) > 0 and abs(round(l / TILE_CM) * TILE_CM - l) < 1, \
            f"{l:g} cm is not a whole number of {TILE_CM:g} cm tiles"
    lengths_cm = [f"{n * TILE_CM:.0f}" for n in n_tiles]

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    print(f"experiments with a CSV: {', '.join(exps)}  "
          f"(positions inferred for: {', '.join(e for e in exps if positions[e]['source'].iloc[0] == 'inferred') or 'none'})")
    fits = load_fits(exps)

    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = parallel_links(positions, exps, n_tiles, walls, args.dedup)
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs) of {'/'.join(map(str, n_tiles))} tiles ({'/'.join(lengths_cm)} cm; "
          f"{links.scan_length_cm.min():.0f}–{links.scan_length_cm.max():.0f} cm in the scan)")
    print(links.groupby(["n_tiles", "direction", "wall"]).agg(links=("exp", "size"), min_cm=("wall_cm", "min"),
                                                   max_cm=("wall_cm", "max")).round(1).to_string())

    bidir = {}
    if args.attn:
        print("\nloading α̂ …")
        bidir = {e: load_bidir(e) for e in exps}
    X, y, meta, freqs = build_dataset(links, fits, bidir, args.attn, n_tiles)
    print(f"{len(X)} samples (one per link and experiment), {X.shape[1]} features: Δ + sine A, period{' + α̂' if args.attn else ''} at "
          f"{len(freqs)} freqs {freqs.min()}–{freqs.max()} MHz + direction + length")

    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.loo)

    OUT_DIR.mkdir(exist_ok=True)
    tag = "-".join(lengths_cm) + "cm" + ("" if args.attn else "_fit_only") + ("_dedup" if args.dedup else "") + ("_loo" if args.loo else "")
    out = meta.iloc[te].reset_index(drop=True).assign(**{f"pred_{k}": v for k, v in preds.items()})
    out.to_csv(OUT_DIR / f"test_predictions_{tag}.csv", index=False)
    plot_predictions(y[te], preds, meta.iloc[te],
                     OUT_DIR / f"pred_vs_true_{tag}.png",
                     f"{', '.join(lengths_cm)} cm links, Δ + sine fit{' + α̂' if args.attn else ' only'}: "
                     "distance from closest parallel wall (test set)")
    print(f"\nwrote {(OUT_DIR / f'test_predictions_{tag}.csv').relative_to(ROOT)}, "
          f"{(OUT_DIR / f'pred_vs_true_{tag}.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Predict a link's distance from the closest parallel wall from its phase-fit residuals (and attenuation).

Same links, target, split, models and frequency band (--freq-min … --freq-max, default 775–955 MHz) as
roomsweeps_wall_distance.py (which this imports), but the phase features are replaced by a linear phase fit, computed
as in variance_processing.ipynb from the same θ̂ that script uses (median over runs, unwrapped along frequency):

- Δ per frequency: unwrapped θ̂ minus its least-squares line over frequency (rad)
- the sine fit to Δ: amplitude A (rad) and period (MHz)

Features: Δ, A, period and α̂ per frequency (median over runs; unless --no-attn); never the link direction or length.

    python roomsweeps_wall_distance_fit.py                     # 92 cm links
    python roomsweeps_wall_distance_fit.py --length-cm 46      # 1-tile links
    python roomsweeps_wall_distance_fit.py --length-cm 92,138  # 2- and 3-tile links in one dataset
    python roomsweeps_wall_distance_fit.py --no-attn           # phase-fit features only
    python roomsweeps_wall_distance_fit.py --freq-min 800 --freq-max 960   # the notebook's fit band
    python roomsweeps_wall_distance_fit.py --dedup             # each label pair once, from its first experiment
    python roomsweeps_wall_distance_fit.py --loo               # leave one experiment out instead of the random split
    python roomsweeps_wall_distance_fit.py --max-wall-cm 200   # only links within 200 cm of their closest parallel wall
"""
import argparse

import numpy as np
import pandas as pd

from roomsweeps_wall_distance import (DATA_DIR, FREQ_RANGE, ROOT, TILE_CM, N_COLS, N_ROWS, all_positions, evaluate,
                                      find_walls, grid_xy, link_runs, load_bidir, median_phase, parallel_links,
                                      plot_predictions, wall_distance_cm)

OUT_DIR = ROOT / "roomsweeps_wall_distance_fit_out"


def fit_sine(x, y, n_periods=400, max_cond=100):
    """Least-squares y ≈ A·sin(2π·x/P + φ) + c, as fit_sine in variance_processing.ipynb: linear in (sin, cos, 1) for
    each trial period P (2 grid steps … 2× the span), skipping ill-conditioned P. Returns (A, P)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for P in np.linspace(2 * np.min(np.diff(x)), 2 * np.ptp(x), n_periods):
        w = 2 * np.pi * x / P
        M = np.column_stack([np.sin(w), np.cos(w), np.ones_like(x)])
        if np.linalg.cond(M) > max_cond:
            continue
        coef = np.linalg.lstsq(M, y, rcond=None)[0]
        sse = np.sum((M @ coef - y) ** 2)
        if best is None or sse < best[0]:
            best = (sse, P, coef)
    _, P, (s, c, _) = best
    return np.hypot(s, c), P


def phase_fit(theta, freqs):
    """Δ (unwrapped θ̂ minus its least-squares line), the line's standard error, and the sine fit (A, P) to Δ."""
    delta = theta - np.polyval(np.polyfit(freqs, theta, 1), freqs)
    return delta, np.sqrt(np.sum(delta ** 2) / (len(freqs) - 2)), *fit_sine(freqs, delta)


def build_dataset(links, bidir, freq_range, use_attn=True):
    """One row per link (median over its runs): Δ per frequency, sine A, sine period, α̂ per frequency (if use_attn).
    No direction or length features."""
    X, y, meta = [], [], []
    for _, link in links.iterrows():
        theta, alpha = link_runs(bidir, link, freq_range)
        if theta.empty:
            continue
        freqs = theta.columns.to_numpy()
        delta, std_err, A, P = phase_fit(median_phase(theta.to_numpy()), freqs)
        X.append(np.concatenate([delta, [A, P]] + ([np.median(alpha.to_numpy(), axis=0)] if use_attn else [])))
        y.append(link.wall_cm)
        meta.append({**link.to_dict(), "std_err": std_err, "sine_A": A, "sine_period_mhz": P, "n_runs": len(theta)})
    return np.array(X), np.array(y), pd.DataFrame(meta), freqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length-cm", default="92",
                    help="link length(s) in cm, comma-separated (e.g. 92,138); each a whole number of tiles")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside the phase fit (--no-attn: phase fit only)")
    ap.add_argument("--max-wall-cm", type=float, default=None,
                    help="ignore links whose midpoint is more than this many cm from the closest parallel wall")
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

    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = parallel_links(positions, exps, n_tiles, walls, args.dedup, args.max_wall_cm)
    if links.empty:
        raise SystemExit("no links match --length-cm / --max-wall-cm")
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs) of {'/'.join(map(str, n_tiles))} tiles ({'/'.join(lengths_cm)} cm; "
          f"{links.scan_length_cm.min():.0f}–{links.scan_length_cm.max():.0f} cm in the scan)")
    print(links.groupby(["n_tiles", "direction", "wall"]).agg(links=("exp", "size"), min_cm=("wall_cm", "min"),
                                                   max_cm=("wall_cm", "max")).round(1).to_string())

    print("\nloading θ̂, α̂ …")
    bidir = {e: load_bidir(e) for e in exps}
    X, y, meta, freqs = build_dataset(links, bidir, (args.freq_min, args.freq_max), args.attn)
    print(f"{len(X)} samples (one per link and experiment, median over {meta.n_runs.min()}–{meta.n_runs.max()} runs), "
          f"{X.shape[1]} features: Δ + sine A, period{' + α̂' if args.attn else ''} at "
          f"{len(freqs)} freqs {freqs.min()}–{freqs.max()} MHz")

    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.loo)

    OUT_DIR.mkdir(exist_ok=True)
    tag = "-".join(lengths_cm) + "cm" + ("" if args.attn else "_fit_only") + ("_dedup" if args.dedup else "") + ("_loo" if args.loo else "") + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else "")
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

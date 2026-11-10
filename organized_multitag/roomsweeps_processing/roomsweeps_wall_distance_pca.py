"""PCA of the link features used by roomsweeps_wall_distance.py, to see how well they separate distance from the wall.

Builds exactly the same dataset as roomsweeps_wall_distance.py (same links, same θ̂ / α̂ features, same flags for
choosing them), but instead of training a regressor it runs a 2-component PCA over all samples and plots PC1 vs PC2:

- colour = distance of the link's midpoint from the closest parallel wall, binned into --bin-cm ranges
  (default 20 cm: 20–40, 40–60, … cm share a colour; one distinct colour per bin, blue = close, red = far)
- symbol = link length (46 cm = square, 92 cm = circle, 138 cm = triangle, …)

Phase features (as in roomsweeps_wall_distance_fit.py, from the same unwrapped median θ̂):

- default: θ̂ per frequency
- --delta-phase: Δ = θ̂ minus its least-squares line over frequency, replacing θ̂
- --delta-sine-phase: Δ minus the sine fitted to Δ (a double delta), replacing θ̂
- --sine-fit: also add the sine fit to Δ, amplitude A (rad) and period (MHz)
- --no-phase (with --sine-fit): drop the per-frequency phase block, keeping only A, period (and α̂)

α̂ per frequency is added unless --no-attn. Features are standardised before PCA (--no-scale: raw), since they have very
different scales. Also prints the explained variance and the Spearman correlation of PC1 / PC2 with the wall distance.

    python roomsweeps_wall_distance_pca.py                       # 46, 92 and 138 cm links
    python roomsweeps_wall_distance_pca.py --length-cm 92        # 2-tile links only
    python roomsweeps_wall_distance_pca.py --no-attn             # phase only
    python roomsweeps_wall_distance_pca.py --zero-start          # unwrapped phase shifted to 0 at the lowest frequency
    python roomsweeps_wall_distance_pca.py --no-unwrap           # raw θ̂ in [0, π)
    python roomsweeps_wall_distance_pca.py --dedup               # each label pair once, from its first experiment
    python roomsweeps_wall_distance_pca.py --max-wall-cm 200     # only links within 200 cm of their closest parallel wall
    python roomsweeps_wall_distance_pca.py --bin-cm 40           # coarser colour bins
    python roomsweeps_wall_distance_pca.py --no-scale            # PCA on the raw features
    python roomsweeps_wall_distance_pca.py --delta-phase         # Δ (phase minus linear fit) instead of θ̂
    python roomsweeps_wall_distance_pca.py --delta-sine-phase    # Δ minus its sine fit instead of θ̂
    python roomsweeps_wall_distance_pca.py --delta-phase --sine-fit   # Δ + sine A, period (+ α̂)
    python roomsweeps_wall_distance_pca.py --sine-fit --no-phase      # sine A, period (+ α̂) only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from roomsweeps_wall_distance import (DATA_DIR, FREQ_RANGE, N_COLS, N_ROWS, ROOT, TILE_CM, all_positions,
                                      build_dataset, find_walls, grid_xy, load_bidir, parallel_links,
                                      wall_distance_cm)
from roomsweeps_wall_distance_fit import fit_sine

OUT_DIR = Path(__file__).resolve().parent / "roomsweeps_wall_distance_pca_out"
LENGTH_MARKERS = {1: "s", 2: "o", 3: "^", 4: "D", 5: "v", 6: "P", 7: "X"}   # tiles -> symbol


def sine_curve(freqs, delta, period):
    """The least-squares A·sin(2π·f/P + φ) + c to delta at a fixed period P (the same model fit_sine fits)."""
    w = 2 * np.pi * freqs / period
    M = np.column_stack([np.sin(w), np.cos(w), np.ones_like(freqs)])
    return M @ np.linalg.lstsq(M, delta, rcond=None)[0]


def phase_features(theta, freqs, mode="raw", sine_fit=False):
    """theta: (n_links, n_freq) unwrapped θ̂. Returns (per-frequency phase block, sine block (n_links, 0 or 2)).
    mode: "raw" θ̂, "delta" θ̂ minus its linear fit, "delta_sine" that minus the sine fitted to it.
    sine_fit: the sine block is [A, period] of the sine fitted to Δ."""
    freqs = np.asarray(freqs, float)
    if mode == "raw" and not sine_fit:
        return theta, np.empty((len(theta), 0))
    phase, sine = [], []
    for th in theta:
        delta = th - np.polyval(np.polyfit(freqs, th, 1), freqs)
        if mode == "delta_sine" or sine_fit:
            A, P = fit_sine(freqs, delta)
            sine.append([A, P])
        if mode == "raw":
            phase.append(th)
        elif mode == "delta":
            phase.append(delta)
        else:
            phase.append(delta - sine_curve(freqs, delta, P))
    return np.array(phase), (np.array(sine) if sine_fit else np.empty((len(theta), 0)))


def distance_bins(y, bin_cm):
    """Bin edges covering y in steps of bin_cm, starting at a multiple of bin_cm."""
    lo = bin_cm * np.floor(y.min() / bin_cm)
    hi = bin_cm * (np.floor(y.max() / bin_cm) + 1)
    return np.arange(lo, hi + bin_cm / 2, bin_cm)


def plot_pca(Z, pca, y, meta, bin_cm, path, title):
    """PC1 vs PC2, colour = wall-distance bin (discrete, high-contrast), symbol = link length."""
    edges = distance_bins(y, bin_cm)
    n_bins = len(edges) - 1
    cmap = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.03, 0.97, n_bins)))   # one distinct colour per bin
    norm = BoundaryNorm(edges, n_bins)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    tiles = meta["n_tiles"].to_numpy()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))   # random draw order so no distance bin systematically hides another
    for n in sorted(set(tiles)):
        sel = order[tiles[order] == n]
        ax.scatter(Z[sel, 0], Z[sel, 1], c=y[sel], cmap=cmap, norm=norm, marker=LENGTH_MARKERS.get(n, "*"),
                   s=46, edgecolors="#333333", linewidths=0.4, alpha=0.95, zorder=2)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f} % of variance)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f} % of variance)")
    ax.grid(color="#e0e0e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=11)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=(edges[:-1] + edges[1:]) / 2,
                      fraction=0.04, pad=0.02)
    cb.ax.set_yticklabels([f"{a:.0f}–{b:.0f}" for a, b in zip(edges[:-1], edges[1:])], fontsize=8)
    cb.ax.tick_params(length=0)
    cb.set_label(f"distance from closest parallel wall (cm), {bin_cm:g} cm bins", fontsize=9)
    handles = [plt.Line2D([], [], marker=LENGTH_MARKERS.get(n, "*"), ls="", color="#555555", markersize=8,
                          label=f"{n * TILE_CM:.0f} cm link") for n in sorted(set(tiles))]
    ax.legend(handles=handles, title="link length", frameon=True, framealpha=0.9, edgecolor="none", fontsize=9,
              title_fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length-cm", default="46,92,138",
                    help="link length(s) in cm, comma-separated (e.g. 92,138); each a whole number of tiles")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--zero-start", action="store_true",
                    help="shift each run's unwrapped θ̂ so it is 0 at the lowest frequency")
    ap.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=True,
                    help="unwrap θ̂ along frequency (--no-unwrap: raw θ̂ in [0, π))")
    ap.add_argument("--max-wall-cm", type=float, default=None,
                    help="ignore links whose midpoint is more than this many cm from the closest parallel wall")
    ap.add_argument("--dedup", action="store_true",
                    help="keep each label pair once, from the first experiment that measured it")
    ap.add_argument("--bin-cm", type=float, default=20.0, help="width of the wall-distance colour bins (cm)")
    ap.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True,
                    help="standardise every feature before PCA (--no-scale: PCA on the raw features)")
    delta = ap.add_mutually_exclusive_group()
    delta.add_argument("--delta-phase", action="store_true",
                       help="replace θ̂ by Δ = θ̂ minus its least-squares line over frequency")
    delta.add_argument("--delta-sine-phase", action="store_true",
                       help="replace θ̂ by Δ minus the sine fitted to Δ (double delta)")
    ap.add_argument("--sine-fit", action="store_true",
                    help="add the amplitude A and period of the sine fitted to Δ as features")
    ap.add_argument("--no-phase", action="store_true",
                    help="with --sine-fit: drop the per-frequency phase block (θ̂ / Δ / double delta)")
    args = ap.parse_args()
    if args.zero_start and not args.unwrap:
        ap.error("--zero-start only works with unwrapped phase (drop --no-unwrap)")
    if (args.delta_phase or args.delta_sine_phase or args.sine_fit) and not args.unwrap:
        ap.error("the linear / sine fits need unwrapped phase (drop --no-unwrap)")
    if args.no_phase and not args.sine_fit:
        ap.error("--no-phase only works with --sine-fit")
    if args.no_phase and (args.delta_phase or args.delta_sine_phase):
        ap.error("--no-phase drops the phase block, so --delta-phase / --delta-sine-phase would have no effect")
    mode = "delta" if args.delta_phase else "delta_sine" if args.delta_sine_phase else "raw"

    lengths = [float(x) for x in args.length_cm.split(",")]
    n_tiles = sorted({int(round(l / TILE_CM)) for l in lengths})
    for l in lengths:
        assert round(l / TILE_CM) > 0 and abs(round(l / TILE_CM) * TILE_CM - l) < 1, \
            f"{l:g} cm is not a whole number of {TILE_CM:g} cm tiles"
    lengths_cm = [f"{n * TILE_CM:.0f}" for n in n_tiles]

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    print(f"experiments with a CSV: {', '.join(exps)}")

    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = parallel_links(positions, exps, n_tiles, walls, args.dedup, args.max_wall_cm)
    if links.empty:
        raise SystemExit("no links match --length-cm / --max-wall-cm")
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs) of "
          f"{'/'.join(lengths_cm)} cm")

    print("\nloading θ̂, α̂ …")
    bidir = {e: load_bidir(e) for e in exps}
    X, y, meta, freqs = build_dataset(links, bidir, (args.freq_min, args.freq_max), True, args.zero_start,
                                      args.unwrap)
    nf = len(freqs)
    phase, sine = phase_features(X[:, :nf], freqs, mode, args.sine_fit)
    blocks, names = [], []
    if not args.no_phase:
        blocks.append(phase)
        names.append({"raw": f"θ̂{'' if args.unwrap else ' (wrapped)'}{' (start at 0)' if args.zero_start else ''}",
                      "delta": "Δθ̂", "delta_sine": "Δθ̂ − sine"}[mode])
    if args.sine_fit:
        blocks.append(sine)
        names.append("sine A, period")
    if args.attn:
        blocks.append(X[:, nf:])
        names.append("α̂")
    if not blocks:
        ap.error("no features left (--no-phase with --no-attn and no --sine-fit)")
    X = np.hstack(blocks)
    feat = " + ".join(names)
    feat_detail = " + ".join(f"{name} ({b.shape[1]})" for name, b in zip(names, blocks))
    print(f"{len(X)} samples, {X.shape[1]} features: {feat_detail}; {nf} freqs {freqs.min()}–{freqs.max()} MHz")

    Xs = StandardScaler().fit_transform(X) if args.scale else X
    pca = PCA(n_components=2).fit(Xs)
    Z = pca.transform(Xs)

    rows = []
    for k in range(pca.n_components_):
        rho_all = spearmanr(Z[:, k], y).statistic
        per_len = {f"ρ {n * TILE_CM:.0f} cm": spearmanr(Z[sel, k], y[sel]).statistic if sel.sum() > 2 else np.nan
                   for n in n_tiles for sel in [meta["n_tiles"].to_numpy() == n]}
        rows.append({"PC": f"PC{k + 1}", "explained var (%)": 100 * pca.explained_variance_ratio_[k],
                                      "Spearman ρ with wall dist": rho_all, **(per_len if len(n_tiles) > 1 else {})})
    print("\nprincipal components (ρ = Spearman correlation of the component with the distance from the wall):")
    print(pd.DataFrame(rows).set_index("PC").round(3).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    tag = ("-".join(lengths_cm) + "cm" + ("" if args.attn else "_phase_only") + ("_zero_start" if args.zero_start else "")
           + ("" if args.unwrap else "_no_unwrap") + ("_dedup" if args.dedup else "")
           + {"raw": "", "delta": "_delta", "delta_sine": "_delta_sine"}[mode]
           + ("_sine" if args.sine_fit else "") + ("_no_phase" if args.no_phase else "")
           + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else "")
           + ("" if args.scale else "_noscale"))
    meta.assign(**{f"PC{k + 1}": Z[:, k] for k in range(pca.n_components_)}).to_csv(OUT_DIR / f"pca_{tag}.csv",
                                                                                   index=False)
    plot_pca(Z, pca, y, meta, args.bin_cm, OUT_DIR / f"pca_{tag}.png",
             f"PCA of {', '.join(lengths_cm)} cm links ({len(X)} samples)\n"
             f"features before PCA ({X.shape[1]}{', standardised' if args.scale else ''}): {feat_detail}, "
             f"{nf} freqs {freqs.min():g}–{freqs.max():g} MHz")
    print(f"\nwrote {(OUT_DIR / f'pca_{tag}.csv').relative_to(ROOT)}, {(OUT_DIR / f'pca_{tag}.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

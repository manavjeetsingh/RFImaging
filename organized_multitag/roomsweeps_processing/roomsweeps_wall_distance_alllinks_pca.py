"""PCA of the link features used by roomsweeps_wall_distance_alllinks.py (every link, any direction), to see how well
they separate distance from the wall.

Like roomsweeps_wall_distance_pca.py, but with the links and target of roomsweeps_wall_distance_alllinks.py: every tag
pair in every room sweep experiment that has a CSV (grid-aligned, diagonal and knight's-move links alike), and as the
target the distance from the link's midpoint, along the perpendicular to the link, to the first wall it meets
(perpendicular_hit). Builds the same θ̂ / α̂ features, runs a 2-component PCA over all samples and plots PC1 vs PC2:

- colour = distance to the closest wall along the perpendicular, binned into --bin-cm ranges
  (default 20 cm: 20–40, 40–60, … cm share a colour; one distinct colour per bin, blue = close, red = far)
- symbol = nominal link length (TILE_CM × grid distance; 46 cm = square, 65 cm = circle, 92 cm = triangle, … in
  order of length), or with --marker angle the link's angle to the hit wall (as in roomsweeps_wall_distance_alllinks.py)

Phase features (as in roomsweeps_wall_distance_fit.py, from the same unwrapped median θ̂):

- default: θ̂ per frequency
- --delta-phase: Δ = θ̂ minus its least-squares line over frequency, replacing θ̂
- --delta-sine-phase: Δ minus the sine fitted to Δ (a double delta), replacing θ̂
- --sine-fit: also add the sine fit to Δ, amplitude A (rad) and period (MHz)
- --no-phase (with --sine-fit): drop the per-frequency phase block, keeping only A, period (and α̂)

α̂ per frequency is added unless --no-attn. Features are standardised before PCA (--no-scale: raw), since they have very
different scales. Also prints the explained variance and the Spearman correlation of PC1 / PC2 with the wall distance,
overall, per link length and per angle class.

    python roomsweeps_wall_distance_alllinks_pca.py                       # every link
    python roomsweeps_wall_distance_alllinks_pca.py --length-cm 46,65,92  # 1-tile, 1-tile diagonal and 2-tile links
    python roomsweeps_wall_distance_alllinks_pca.py --max-length-cm 110   # links up to 110 cm (nominal grid length)
    python roomsweeps_wall_distance_alllinks_pca.py --max-angle-deg 5     # only links (nearly) parallel to the hit wall
    python roomsweeps_wall_distance_alllinks_pca.py --marker angle        # symbol = link ∠ hit wall instead of length
    python roomsweeps_wall_distance_alllinks_pca.py --no-attn             # phase only
    python roomsweeps_wall_distance_alllinks_pca.py --zero-start          # unwrapped phase shifted to 0 at the lowest frequency
    python roomsweeps_wall_distance_alllinks_pca.py --no-unwrap           # raw θ̂ in [0, π)
    python roomsweeps_wall_distance_alllinks_pca.py --dedup               # each label pair once, from its first experiment
    python roomsweeps_wall_distance_alllinks_pca.py --max-wall-cm 200     # only links whose ray hits a wall within 200 cm
    python roomsweeps_wall_distance_alllinks_pca.py --bin-cm 40           # coarser colour bins
    python roomsweeps_wall_distance_alllinks_pca.py --no-scale            # PCA on the raw features
    python roomsweeps_wall_distance_alllinks_pca.py --delta-phase         # Δ (phase minus linear fit) instead of θ̂
    python roomsweeps_wall_distance_alllinks_pca.py --delta-sine-phase    # Δ minus its sine fit instead of θ̂
    python roomsweeps_wall_distance_alllinks_pca.py --delta-phase --sine-fit   # Δ + sine A, period (+ α̂)
    python roomsweeps_wall_distance_alllinks_pca.py --sine-fit --no-phase      # sine A, period (+ α̂) only
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

from roomsweeps_wall_distance import (DATA_DIR, FREQ_RANGE, N_COLS, N_ROWS, ROOT, all_positions, build_dataset,
                                      find_walls, grid_xy, load_bidir, wall_distance_cm)
from roomsweeps_wall_distance_alllinks import ANGLE_CLASSES, all_links
from roomsweeps_wall_distance_pca import distance_bins, phase_features
from roomsweeps_wall_distance import (add_data_version_arg, has_experiment,  # noqa: E402
                                      set_data_version, version_tag)

OUT_DIR = Path(__file__).resolve().parent / "roomsweeps_wall_distance_alllinks_pca_out"
LENGTH_MARKERS = ["s", "o", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*", "d", "8", "H"]   # shortest link first


def marker_groups(meta, by):
    """[(label, sample mask, marker)] for the scatter symbols: one per nominal link length, or per angle class."""
    if by == "angle":
        cls = meta["angle_class"].to_numpy()
        return [(f"link ∠ wall {name}", cls == name, m) for name, _, m in ANGLE_CLASSES if (cls == name).any()]
    length = meta["length_cm"].round(1).to_numpy()
    return [(f"{l:.0f} cm link", length == l, LENGTH_MARKERS[i % len(LENGTH_MARKERS)])
            for i, l in enumerate(sorted(set(length)))]


def plot_pca(Z, pca, y, meta, bin_cm, marker_by, path, title):
    """PC1 vs PC2, colour = wall-distance bin (discrete, high-contrast), symbol = link length or angle class."""
    edges = distance_bins(y, bin_cm)
    n_bins = len(edges) - 1
    cmap = ListedColormap(plt.get_cmap("turbo")(np.linspace(0.03, 0.97, n_bins)))   # one distinct colour per bin
    norm = BoundaryNorm(edges, n_bins)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))   # random draw order so no distance bin systematically hides another
    groups = marker_groups(meta, marker_by)
    for _, mask, marker in groups:
        sel = order[mask[order]]
        ax.scatter(Z[sel, 0], Z[sel, 1], c=y[sel], cmap=cmap, norm=norm, marker=marker,
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
    cb.set_label(f"distance to closest wall along the perpendicular (cm), {bin_cm:g} cm bins", fontsize=9)
    handles = [plt.Line2D([], [], marker=marker, ls="", color="#555555", markersize=8, label=label)
               for label, _, marker in groups]
    ax.legend(handles=handles, title="link length" if marker_by == "length" else "link ∠ hit wall", frameon=True,
              framealpha=0.9, edgecolor="none", fontsize=8, title_fontsize=9, loc="best",
              ncol=2 if len(handles) > 8 else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length-cm", default=None,
                    help="only links of these nominal length(s) in cm (TILE_CM × grid distance, ±1 cm), comma-separated, "
                         "e.g. 92 or 46,65,92 (65 = 1-tile diagonal); default: every length")
    ap.add_argument("--min-length-cm", type=float, default=None, help="shortest nominal link length (TILE_CM × grid distance)")
    ap.add_argument("--max-length-cm", type=float, default=None, help="longest nominal link length")
    ap.add_argument("--max-angle-deg", type=float, default=None,
                    help="only links within this angle of the wall their perpendicular hits (0 = parallel only)")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True,
                    help="use α̂ as features alongside θ̂ (--no-attn: phase only)")
    ap.add_argument("--zero-start", action="store_true",
                    help="shift each run's unwrapped θ̂ so it is 0 at the lowest frequency")
    ap.add_argument("--unwrap", action=argparse.BooleanOptionalAction, default=True,
                    help="unwrap θ̂ along frequency (--no-unwrap: raw θ̂ in [0, π))")
    ap.add_argument("--max-wall-cm", type=float, default=None,
                    help="ignore links whose perpendicular hits the closest wall more than this many cm away")
    ap.add_argument("--dedup", action="store_true",
                    help="keep each label pair once, from the first experiment that measured it")
    ap.add_argument("--bin-cm", type=float, default=20.0, help="width of the wall-distance colour bins (cm)")
    ap.add_argument("--marker", choices=["length", "angle"], default="length",
                    help="scatter symbol: nominal link length, or the link's angle to the hit wall")
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
    add_data_version_arg(ap)
    args = ap.parse_args()
    set_data_version(args.data_version)
    if args.zero_start and not args.unwrap:
        ap.error("--zero-start only works with unwrapped phase (drop --no-unwrap)")
    if (args.delta_phase or args.delta_sine_phase or args.sine_fit) and not args.unwrap:
        ap.error("the linear / sine fits need unwrapped phase (drop --no-unwrap)")
    if args.no_phase and not args.sine_fit:
        ap.error("--no-phase only works with --sine-fit")
    if args.no_phase and (args.delta_phase or args.delta_sine_phase):
        ap.error("--no-phase drops the phase block, so --delta-phase / --delta-sine-phase would have no effect")
    mode = "delta" if args.delta_phase else "delta_sine" if args.delta_sine_phase else "raw"
    lengths = None if args.length_cm is None else [float(x) for x in args.length_cm.split(",")]

    positions = all_positions()
    exps = [e for e in positions if has_experiment(e)]
    print(f"experiments with a CSV: {', '.join(exps)}")

    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = all_links(positions, exps, walls, args.min_length_cm, args.max_length_cm, args.max_angle_deg, args.dedup,
                      args.max_wall_cm, lengths)
    if links.empty:
        raise SystemExit("no links match --length-cm, --min/--max-length-cm, --max-angle-deg, --max-wall-cm")
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs), nominal length "
          f"{links.length_cm.min():.0f}–{links.length_cm.max():.0f} cm")

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
    feat_detail = " + ".join(f"{name} ({b.shape[1]})" for name, b in zip(names, blocks))
    print(f"{len(X)} samples, {X.shape[1]} features: {feat_detail}; {nf} freqs {freqs.min()}–{freqs.max()} MHz")

    Xs = StandardScaler().fit_transform(X) if args.scale else X
    pca = PCA(n_components=2).fit(Xs)
    Z = pca.transform(Xs)

    rho = lambda k, sel: spearmanr(Z[sel, k], y[sel]).statistic if sel.sum() > 2 else np.nan
    print("\nprincipal components (ρ = Spearman correlation of the component with the distance from the wall):")
    print(pd.DataFrame([{"PC": f"PC{k + 1}", "explained var (%)": 100 * pca.explained_variance_ratio_[k],
                         "Spearman ρ with wall dist": rho(k, np.ones(len(y), bool))}
                        for k in range(pca.n_components_)]).set_index("PC").round(3).to_string())
    for by in ("length", "angle"):
        groups = marker_groups(meta, by)
        if len(groups) > 1:
            print(f"\nSpearman ρ per {'nominal link length' if by == 'length' else 'link ∠ hit wall'}:")
            print(pd.DataFrame([{"group": label, "samples": int(mask.sum()),
                                 **{f"ρ PC{k + 1}": rho(k, mask) for k in range(pca.n_components_)}}
                                for label, mask, _ in groups]).set_index("group").round(3).to_string())

    OUT_DIR.mkdir(exist_ok=True)
    tag = (("all" if args.min_length_cm is None and args.max_length_cm is None
            else f"{args.min_length_cm or 0:g}-{args.max_length_cm or 'inf'}cm")
           + ("_len" + "-".join(f"{l:g}" for l in lengths) + "cm" if lengths is not None else "")
           + (f"_max{args.max_angle_deg:g}deg" if args.max_angle_deg is not None else "")
           + ("" if args.attn else "_phase_only") + ("_zero_start" if args.zero_start else "")
           + ("" if args.unwrap else "_no_unwrap") + ("_dedup" if args.dedup else "")
           + {"raw": "", "delta": "_delta", "delta_sine": "_delta_sine"}[mode]
           + ("_sine" if args.sine_fit else "") + ("_no_phase" if args.no_phase else "")
           + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else "")
           + ("" if args.scale else "_noscale"))
    tag += version_tag()
    meta.assign(**{f"PC{k + 1}": Z[:, k] for k in range(pca.n_components_)}).to_csv(OUT_DIR / f"pca_{tag}.csv",
                                                                                   index=False)
    png = OUT_DIR / f"pca_{tag}{'' if args.marker == 'length' else '_by_angle'}.png"
    plot_pca(Z, pca, y, meta, args.bin_cm, args.marker, png,
             f"PCA of all links ({tag}, {len(X)} samples)\n"
             f"features before PCA ({X.shape[1]}{', standardised' if args.scale else ''}): {feat_detail}, "
             f"{nf} freqs {freqs.min():g}–{freqs.max():g} MHz")
    print(f"\nwrote {(OUT_DIR / f'pca_{tag}.csv').relative_to(ROOT)}, {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

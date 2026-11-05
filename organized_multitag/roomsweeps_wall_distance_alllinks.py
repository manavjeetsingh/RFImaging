"""Predict a link's distance from the closest wall along its perpendicular, for every link (any direction), from its
bidirectional phase and attenuation.

Like roomsweeps_wall_distance.py (whose functions it uses), but instead of only the links parallel to a pair of walls
it uses every tag pair in every room sweep experiment that has a CSV: grid-aligned, diagonal and knight's-move links
alike. The target is found by casting a ray from the link's midpoint perpendicular to the link, in both directions,
and taking the distance along the ray to the first of the four walls around the grid (find_walls) that it meets:

    M = (A + B) / 2,  d = ± (unit vector of B − A rotated by 90°)
    wall w (point P, unit normal n into the room) is hit at  t = −(M − P)·n / (d·n)   for d·n < 0
    target = 100 · min over walls and both directions of t  (cm)

The walls form a convex region around the grid, so the smallest positive t is where the ray leaves the room. For a
link parallel to a pair of walls this is the same as roomsweeps_wall_distance.py's target (the perpendicular distance
of the midpoint from the closer parallel wall). For a diagonal link the ray meets the wall obliquely, so the target
is longer than the midpoint's perpendicular distance from that wall (kept as `wall_perp_cm` in the output, with the
angle between the link and the hit wall as `angle_deg`, 0 = parallel).

Features, sample construction (median over runs, one sample per link and experiment, --dedup), the 70 / 30 split by
label pair, --loo / --no-loo-dedup and the SVR / MLP / mean-of-train comparison are exactly as in
roomsweeps_wall_distance.py. Next to them (--no-nets: off) two neural networks are trained from scratch on the real
training samples only, with the scratch settings of roomsweeps_wall_distance_posttrain.py:

- Transformer (scratch)   gridsim_wall_distance.TransformerRegressor, one token per frequency [θ̂, α̂], width 64,
                          3 layers, --tf-epochs (200) epochs, batch 32, peak lr --tf-lr (1e-3), weight decay 1e-2
- ResCNN (scratch)        roomsweeps_wall_distance_posttrain.ResCNNRegressor, tokens [θ̂ minus its line fit, centred
                          log α̂] plus the line slope and mean log α̂, width 64, --rescnn-epochs (60) epochs, batch 32,
                          peak lr --rescnn-lr (1e-3), weight decay 1e-2

The link direction, length and angle are never features; they are only used to select links (--min/--max-length-cm,
--max-angle-deg) and to break down the errors.

    python roomsweeps_wall_distance_alllinks.py                          # every link
    python roomsweeps_wall_distance_alllinks.py --max-length-cm 110      # links up to 110 cm (nominal grid length)
    python roomsweeps_wall_distance_alllinks.py --min-length-cm 90 --max-length-cm 110
    python roomsweeps_wall_distance_alllinks.py --max-angle-deg 5        # only links (nearly) parallel to the hit wall
    python roomsweeps_wall_distance_alllinks.py --max-wall-cm 200        # only links whose ray hits a wall within 200 cm
    python roomsweeps_wall_distance_alllinks.py --no-attn --loo          # phase only, leave one experiment out
"""
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from gridsim_wall_distance import TransformerRegressor
from roomsweeps_wall_distance import (DATA_DIR, FREQ_RANGE, GOAL_CM, BASELINE_COLOR, N_COLS, N_ROWS, PART_COLORS, ROOT,
                                      TILE_CM, all_positions, build_dataset, evaluate, find_walls, fit_predict,
                                      grid_xy, load_bidir, label_grid, link_part, wall_distance_cm)
from roomsweeps_wall_distance_posttrain import ResCNNRegressor

OUT_DIR = ROOT / "roomsweeps_wall_distance_alllinks_out"
PARALLEL_DEG = 5.0        # links within this angle of the hit wall count as parallel
ANGLE_CLASSES = [         # (name, upper bound in degrees, scatter marker), angle between the link and the hit wall
    ("parallel (< 5°)", PARALLEL_DEG, "o"),
    ("5–30°", 30.0, "^"),
    ("30–60°", 60.0, "s"),
    ("60–90°", 90.01, "D"),
]


def angle_class(angle_deg):
    return next(name for name, hi, _ in ANGLE_CLASSES if angle_deg < hi)


def short_wall(name):
    return name.split(" (", 1)[0]


# ---------------------------------------------------------------- perpendicular ray -> closest wall

def perpendicular_hit(a, b, walls):
    """Cast a ray from the midpoint of a–b (scan xy, metres) perpendicular to the link, both ways, and return the first
    wall it meets as {wall, wall_cm (along the ray), wall_perp_cm (midpoint's perpendicular distance from that wall),
    angle_deg (between the link and that wall, 0 = parallel), hit_x, hit_y}."""
    mid = (a + b) / 2
    u = (b - a) / np.linalg.norm(b - a)
    best = None
    for d in (np.array([-u[1], u[0]]), np.array([u[1], -u[0]])):
        for name, (point, normal, _) in walls.items():
            dn = d @ normal
            if dn > -1e-9:            # moving parallel to the wall or away from it (normal points into the room)
                continue
            t = -((mid - point) @ normal) / dn
            if t > 0 and (best is None or t < best[0]):
                best = (t, name, d, normal)
    t, name, d, normal = best
    hit = mid + t * d
    return {"wall": name, "wall_cm": 100 * t, "wall_perp_cm": wall_distance_cm(mid, walls[name]),
            "angle_deg": np.degrees(np.arccos(min(1.0, abs(d @ normal)))), "hit_x": hit[0], "hit_y": hit[1]}


def all_links(positions, exps, walls, min_len_cm=None, max_len_cm=None, max_angle_deg=None, dedup=False,
              max_wall_cm=None):
    """Every tag pair of every experiment with the wall its perpendicular ray hits first (perpendicular_hit). Links are
    filtered on their nominal length (TILE_CM × grid distance), their angle to the hit wall and the distance to it.
    A label pair measured in several experiments gives one row per experiment; with dedup only the first in `exps`."""
    rows, seen = [], set()
    for exp in exps:
        pos = positions[exp]
        tags = sorted(pos.index)
        for i, ta in enumerate(tags):
            for tb in tags[i + 1:]:
                la, lb = pos.loc[ta, "label"], pos.loc[tb, "label"]
                (ca, ra), (cb, rb) = label_grid(la), label_grid(lb)
                length_cm = TILE_CM * np.hypot(ca - cb, ra - rb)
                if (min_len_cm is not None and length_cm < min_len_cm - 0.5) or \
                        (max_len_cm is not None and length_cm > max_len_cm + 0.5):
                    continue
                a, b = grid_xy(ca, ra), grid_xy(cb, rb)
                hit = perpendicular_hit(a, b, walls)
                if max_angle_deg is not None and hit["angle_deg"] > max_angle_deg:
                    continue
                if max_wall_cm is not None and hit["wall_cm"] > max_wall_cm:
                    continue
                key = tuple(sorted((la, lb)))
                if dedup and key in seen:
                    continue
                seen.add(key)
                rows.append({"exp": exp, "T1": ta, "T2": tb, "label_a": la, "label_b": lb,
                             "part": link_part(ca, ra, cb, rb), "d_col": abs(ca - cb), "d_row": abs(ra - rb),
                             "length_cm": length_cm, "scan_length_cm": 100 * np.linalg.norm(a - b),
                             "mid_x": (a[0] + b[0]) / 2, "mid_y": (a[1] + b[1]) / 2, **hit,
                             "angle_class": angle_class(hit["angle_deg"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models

def make_fit_fn(args, freqs):
    """fit_fn for roomsweeps_wall_distance.evaluate: its SVR / MLP / mean of train plus (if args.nets) a transformer
    and a residual CNN trained from scratch on the training samples."""
    F = len(freqs)
    token_cols = np.arange(F)[None] + F * np.arange(2 if args.attn else 1)[:, None]     # (channels, freqs)

    def fit_fn(X, y, tr, te, seed):
        preds, train_mae = fit_predict(X, y, tr, te, seed)
        if not args.nets:
            return preds, train_mae
        nets = {
            "Transformer (scratch)": TransformerRegressor(
                token_cols, epochs=args.tf_epochs, batch_size=args.batch_size, lr=args.tf_lr,
                weight_decay=args.weight_decay, seed=seed, log_every=0, dim=64, layers=3).fit(X[tr], y[tr]),
            "ResCNN (scratch)": ResCNNRegressor(freqs, args.attn).fit(
                X[tr], y[tr], epochs=args.rescnn_epochs, batch_size=args.batch_size, lr=args.rescnn_lr,
                weight_decay=args.weight_decay, seed=seed),
        }
        for name, model in nets.items():
            preds[name] = model.predict(X[te])
            train_mae[name] = mean_absolute_error(y[tr], model.predict(X[tr]))
        return preds, train_mae
    return fit_fn


# ---------------------------------------------------------------- plots

def wall_segments(walls):
    """Each wall line clipped at its intersections with the two walls not parallel to it (the room's corners)."""
    segs = {}
    for name, (p, n, _) in walls.items():
        u = np.array([n[1], -n[0]])
        ts = []
        for other, (q, m, _) in walls.items():
            if other == name or abs(u @ m) < 0.5:
                continue
            ts.append(((q - p) @ m) / (u @ m))      # (p + t u − q)·m = 0
        if len(ts) >= 2:
            segs[name] = (p + min(ts) * u, p + max(ts) * u)
    return segs


def plot_geometry(links, walls, path):
    """The grid, the walls and, for every label pair, its link and the perpendicular ray to the wall it hits."""
    pairs = links.drop_duplicates(["label_a", "label_b"])
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, (s, e) in wall_segments(walls).items():
        ax.plot([s[0], e[0]], [s[1], e[1]], color="#2b2b2b", lw=3, solid_capstyle="round", zorder=1)
        ax.annotate(short_wall(name), (s + e) / 2, fontsize=8, ha="center", va="center", color="#2b2b2b",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), zorder=5)
    grid = np.array([grid_xy(c, r) for c in range(N_COLS) for r in range(N_ROWS)])
    ax.scatter(grid[:, 0], grid[:, 1], s=6, color="#b0b0b0", zorder=2)
    norm = plt.Normalize(pairs.wall_cm.min(), pairs.wall_cm.max())
    cmap = plt.get_cmap("viridis")
    for _, l in pairs.iterrows():
        a, b = grid_xy(*label_grid(l.label_a)), grid_xy(*label_grid(l.label_b))
        c = cmap(norm(l.wall_cm))
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=0.6, alpha=0.5, zorder=3)
        ax.plot([l.mid_x, l.hit_x], [l.mid_y, l.hit_y], color=c, lw=0.6, ls=":", alpha=0.7, zorder=3)
    ax.scatter(pairs.hit_x, pairs.hit_y, s=10, c=pairs.wall_cm, cmap=cmap, norm=norm, zorder=4)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="distance along the perpendicular (cm)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, scan)")
    ax.set_ylabel("y (m, scan)")
    ax.set_title(f"{len(pairs)} label pairs: link (solid), perpendicular from its midpoint to the closest wall (dotted)",
                 fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def style(ax):
    ax.grid(color="#e0e0e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_predictions(y_test, preds, meta_test, path, title):
    """Predicted vs true, coloured by part of the floor, marker by the link's angle to the hit wall."""
    ncols = len(preds) if len(preds) <= 3 else math.ceil(len(preds) / 2)
    nrows = math.ceil(len(preds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols + 2.5, 5 * nrows), squeeze=False)
    for ax in axes.flat[len(preds):]:
        ax.set_visible(False)
    order = [f"{a}, {b}" for a in ["Edge02", "Edge02/Center0", "Center0", "Center0/Edge01", "Edge01"]
             for b in ["Edge10", "Edge10/Center1", "Center1"]]
    present = set(meta_test["part"])
    part_colors = {part: PART_COLORS[i % len(PART_COLORS)] for i, part in enumerate(p for p in order if p in present)}
    hi = max(y_test.max(), *(p.max() for p in preds.values()))
    pad = 0.05 * hi
    parts, classes = meta_test["part"].to_numpy(), meta_test["angle_class"].to_numpy()
    for ax, (name, p) in zip(axes.flat, preds.items()):
        ax.plot([0, hi + pad], [0, hi + pad], color="#9a9a9a", lw=1, ls="--", zorder=1)
        for part, color in part_colors.items():
            for cls, _, marker in ANGLE_CLASSES:
                sel = (parts == part) & (classes == cls)
                ax.scatter(y_test[sel], p[sel], s=24, color=color, marker=marker, alpha=0.8, linewidths=0, zorder=2)
        ax.set_title(f"{name}  (test MAE {mean_absolute_error(y_test, p):.1f} cm)", fontsize=11)
        ax.set_xlabel("true distance to closest wall along the perpendicular (cm)")
        ax.set_ylabel("predicted (cm)")
        ax.set_xlim(0, hi + pad)
        ax.set_ylim(0, hi + pad)
        ax.set_aspect("equal")
        style(ax)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, markersize=8, label=part) for part, c in part_colors.items()]
    handles += [plt.Line2D([], [], marker=m, ls="", color="#555555", markersize=7, label=f"link ∠ wall {cls}")
                for cls, _, m in ANGLE_CLASSES if cls in set(classes)]
    axes.flat[ncols - 1].legend(handles=handles, title="part (dir-0, dir-1 region)", frameon=False, fontsize=8,
                         title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(y_test, preds, baseline, meta_test, path, title, bin_cm=25):
    """As roomsweeps_wall_distance.plot_errors: cumulative |error| (with the GOAL_CM band), signed error per bin_cm bin
    of the true distance, and the MAE per hit wall and angle class, each against predicting the training mean.
    Also prints the numbers."""
    colors = dict(zip(preds, PART_COLORS))
    err = {name: p - y_test for name, p in preds.items()}
    base_err = baseline - y_test
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})

    # 1. cumulative |error|
    ax = axes[0]
    ax.axvspan(*GOAL_CM, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.text(np.mean(GOAL_CM), 1.0, f"goal\n{GOAL_CM[0]}–{GOAL_CM[1]} cm", ha="center", va="top", fontsize=8,
            color="#4a4a4a")
    hi = max(np.abs(base_err).max(), *(np.abs(e).max() for e in err.values()))
    for name, e, color, ls in [("mean of train", base_err, BASELINE_COLOR, "--"),
                               *((n, e, colors[n], "-") for n, e in err.items())]:
        a = np.sort(np.abs(e))
        ax.step(np.r_[0, a], np.r_[0, np.arange(1, len(a) + 1) / len(a)], where="post", color=color, ls=ls, lw=2,
                label=f"{name}: {100 * (a <= GOAL_CM[1]).mean():.0f}% ≤ {GOAL_CM[1]} cm, median {np.median(a):.0f} cm")
    ax.set_xlim(0, hi * 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("absolute error (cm)")
    ax.set_ylabel("fraction of test links with at most this error")
    ax.set_title("Cumulative absolute error (left / higher is better)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # 2. signed error by true distance
    ax = axes[1]
    edges = np.arange(0, y_test.max() + bin_cm, bin_cm)
    b = np.digitize(y_test, edges) - 1
    used = [k for k in range(len(edges) - 1) if (b == k).sum() >= 3]
    centers = np.array([(edges[k] + edges[k + 1]) / 2 for k in used])
    ax.axhline(0, color="#4a4a4a", lw=1, zorder=1)
    ax.axhspan(-GOAL_CM[1], GOAL_CM[1], color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.plot(centers, [np.median(base_err[b == k]) for k in used], color=BASELINE_COLOR, ls="--", lw=2,
            label="mean of train (median)", zorder=2)
    width = 0.6 * bin_cm
    for i, (name, e) in enumerate(err.items()):
        x = centers + (i - (len(err) - 1) / 2) * width / max(len(err), 1)
        q = np.array([np.percentile(e[b == k], [25, 50, 75]) for k in used])
        ax.vlines(x, q[:, 0], q[:, 2], color=colors[name], lw=2, zorder=3)
        ax.plot(x, q[:, 1], "o", color=colors[name], ms=7, mec="white", mew=1.5, zorder=4,
                label=f"{name} (median, bar = middle 50 %)")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{edges[k]:.0f}–{edges[k + 1]:.0f}\n({(b == k).sum()})" for k in used], fontsize=8)
    ax.set_xlabel("true distance to closest wall along the perpendicular (cm), (number of test links)")
    ax.set_ylabel("predicted − true (cm)")
    ax.set_title("Signed error by true distance (a model that learned nothing follows the dashed line)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3. MAE per hit wall and angle class
    ax = axes[2]
    wall = meta_test["wall"].map(short_wall).to_numpy()
    classes = meta_test["angle_class"].to_numpy()
    class_order = [c for c, _, _ in ANGLE_CLASSES]
    rows = []
    for w in dict.fromkeys(wall):
        for c in (c for c in class_order if ((wall == w) & (classes == c)).any()):
            sel = (wall == w) & (classes == c)
            rows.append({"group": f"{w}, {c} ({sel.sum()})", "mean of train": np.abs(base_err[sel]).mean(),
                         **{name: np.abs(e[sel]).mean() for name, e in err.items()}})
    tab = pd.DataFrame(rows).set_index("group")
    y_pos = np.arange(len(tab))[::-1]
    h = 0.8 / max(len(err), 1)
    for i, name in enumerate(err):
        ax.barh(y_pos + (len(err) - 1) / 2 * h - i * h, tab[name], height=h * 0.85, color=colors[name], label=name,
                zorder=2)
    ax.scatter(tab["mean of train"], y_pos, marker="|", s=900, color="#4a4a4a", lw=2.5, zorder=3)
    ax.axvspan(*GOAL_CM, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tab.index, fontsize=8)
    ax.set_xlabel("MAE (cm)")
    ax.set_title("MAE by hit wall and link ∠ wall (bar left of the tick = beats the mean)", fontsize=10)
    handles = ax.get_legend_handles_labels()[0] + [
        plt.Line2D([], [], marker="|", ls="", color="#4a4a4a", markersize=12, mew=2.5, label="mean of train")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=len(handles))

    for ax in axes:
        style(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\nerror distribution (test set; links within {GOAL_CM[1]} cm, links better than the mean of train):")
    print(pd.DataFrame({name: {f"≤ {GOAL_CM[1]} cm (%)": 100 * (np.abs(e) <= GOAL_CM[1]).mean(),
                               "median |error| (cm)": np.median(np.abs(e)),
                               "90th pct |error| (cm)": np.percentile(np.abs(e), 90),
                               "beats mean of train (%)": 100 * (np.abs(e) < np.abs(base_err)).mean()
                               if name != "mean of train" else np.nan}
                        for name, e in {"mean of train": base_err, **err}.items()}).T.round(1).to_string())
    print("\nMAE (cm) by hit wall and link ∠ wall (number of test links):")
    print(tab.round(1).to_string())


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-length-cm", type=float, default=None, help="shortest nominal link length (TILE_CM × grid distance)")
    ap.add_argument("--max-length-cm", type=float, default=None, help="longest nominal link length")
    ap.add_argument("--max-angle-deg", type=float, default=None,
                    help="only links within this angle of the wall their perpendicular hits (0 = parallel only)")
    ap.add_argument("--freq-min", type=float, default=FREQ_RANGE[0], help="MHz, inclusive")
    ap.add_argument("--freq-max", type=float, default=FREQ_RANGE[1], help="MHz, inclusive")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument("--loo", action="store_true",
                    help="leave one experiment out: train on the other experiments, test on it, for every experiment")
    ap.add_argument("--loo-dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="with --loo, drop the training samples of every label pair the held-out experiment measured "
                         "(--no-loo-dedup: keep them)")
    ap.add_argument("--nets", action=argparse.BooleanOptionalAction, default=True,
                    help="also train a transformer and a residual CNN from scratch (--no-nets: SVR / MLP only)")
    ap.add_argument("--tf-epochs", type=int, default=200)
    ap.add_argument("--tf-lr", type=float, default=1e-3, help="peak transformer learning rate")
    ap.add_argument("--rescnn-epochs", type=int, default=60)
    ap.add_argument("--rescnn-lr", type=float, default=1e-3, help="peak ResCNN learning rate")
    ap.add_argument("--batch-size", type=int, default=32, help="transformer and ResCNN batch size")
    ap.add_argument("--weight-decay", type=float, default=1e-2, help="transformer and ResCNN weight decay")
    args = ap.parse_args()
    if args.zero_start and not args.unwrap:
        ap.error("--zero-start only works with unwrapped phase (drop --no-unwrap)")

    positions = all_positions()
    exps = [e for e in positions if any((DATA_DIR / e).glob("*.csv"))]
    print(f"experiments with a CSV: {', '.join(exps)}  "
          f"(positions inferred for: {', '.join(e for e in exps if positions[e]['source'].iloc[0] == 'inferred') or 'none'})")

    walls = find_walls()
    print("walls (distance of the nearest grid edge from the wall):")
    for name, w in walls.items():
        edge = grid_xy(0, 0) if "col 0" in name or "row 1" in name else grid_xy(N_COLS - 1, N_ROWS - 1)
        print(f"  {name:<30} {wall_distance_cm(edge, w):6.1f} cm")

    links = all_links(positions, exps, walls, args.min_length_cm, args.max_length_cm, args.max_angle_deg, args.dedup,
                      args.max_wall_cm)
    if links.empty:
        raise SystemExit("no links match --min/--max-length-cm, --max-angle-deg, --max-wall-cm")
    print(f"\n{len(links)} links ({links.groupby(['label_a', 'label_b']).ngroups} label pairs), nominal length "
          f"{links.length_cm.min():.0f}–{links.length_cm.max():.0f} cm "
          f"({links.scan_length_cm.min():.0f}–{links.scan_length_cm.max():.0f} cm in the scan)")
    links["wall_short"] = links["wall"].map(short_wall)
    print(links.groupby(["wall_short", "angle_class"]).agg(
        links=("exp", "size"), min_cm=("wall_cm", "min"), max_cm=("wall_cm", "max"),
        mean_extra_cm=("wall_cm", lambda s: (s - links.loc[s.index, "wall_perp_cm"]).mean())).round(1).to_string())
    print("(mean_extra_cm: how much longer the perpendicular ray is than the midpoint's perpendicular distance from the wall)")
    links = links.drop(columns="wall_short")

    OUT_DIR.mkdir(exist_ok=True)
    tag = ("all" if args.min_length_cm is None and args.max_length_cm is None
           else f"{args.min_length_cm or 0:g}-{args.max_length_cm or 'inf'}cm") \
        + (f"_max{args.max_angle_deg:g}deg" if args.max_angle_deg is not None else "") \
        + ("" if args.attn else "_phase_only") + ("_zero_start" if args.zero_start else "") \
        + ("" if args.unwrap else "_no_unwrap") + ("_dedup" if args.dedup else "") \
        + ("_loo" + ("-dedup" if args.loo_dedup else "") if args.loo else "") \
        + (f"_max{args.max_wall_cm:g}cm" if args.max_wall_cm is not None else "")
    plot_geometry(links, walls, OUT_DIR / f"geometry_{tag}.png")

    print("\nloading θ̂, α̂ …")
    bidir = {e: load_bidir(e) for e in exps}
    X, y, meta, freqs = build_dataset(links, bidir, (args.freq_min, args.freq_max), args.attn, args.zero_start, args.unwrap)
    feat = f"θ̂{'' if args.unwrap else ' (wrapped)'}{' (start at 0)' if args.zero_start else ''}{' + α̂' if args.attn else ''}"
    print(f"{len(X)} samples (one per link and experiment, median over {meta.n_runs.min()}–{meta.n_runs.max()} runs), "
          f"{X.shape[1]} features: {feat} at {len(freqs)} freqs {freqs.min()}–{freqs.max()} MHz")

    te, preds = evaluate(X, y, meta, args.train_frac, args.seed, args.loo, make_fit_fn(args, freqs),
                         loo_dedup=args.loo_dedup, keep_mean=True)
    baseline = preds.pop("mean of train")

    out = meta.iloc[te].reset_index(drop=True).assign(**{f"pred_{k}": v for k, v in preds.items()})
    out.to_csv(OUT_DIR / f"test_predictions_{tag}.csv", index=False)
    plot_predictions(y[te], preds, meta.iloc[te], OUT_DIR / f"pred_vs_true_{tag}.png",
                     f"all links ({tag}), {feat}: distance to closest wall along the perpendicular (test set)")
    plot_errors(y[te], preds, baseline, meta.iloc[te], OUT_DIR / f"errors_{tag}.png",
                f"all links ({tag}), {feat}"
                f"{' (leave one experiment out' + (', held-out pairs dropped from training)' if args.loo_dedup else ')') if args.loo else ''}"
                ": error distribution (test set)")
    print(f"\nwrote {(OUT_DIR / f'geometry_{tag}.png').relative_to(ROOT)}, "
          f"{(OUT_DIR / f'test_predictions_{tag}.csv').relative_to(ROOT)}, "
          f"{(OUT_DIR / f'pred_vs_true_{tag}.png').relative_to(ROOT)}, {(OUT_DIR / f'errors_{tag}.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

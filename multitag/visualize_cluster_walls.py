import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Usage: python visualize_cluster_walls.py CSV [--out-dir DIR] [--max-subclusters M] [--view top|3d]
#                                              [--no-rays] [--seed 0]
#   One 3-D figure per experiment from a cluster_features.py output (CSV or .pkl), in the style of
#   T2TSionnaSim MLNotebooks/vizualizeWalls.py: visualize_walls3D. Rows that differ only in run or
#   frequency share their geometry, so each (experiment, combination, permutation) is drawn once.
#   Drawn per sub-cluster: centroid, arrow to tag 0 (pose +X), dashed N/S/E/W rays and a 10x10 cm
#   vertical patch at every wall hit, oriented from wall_hit/wall_angle alone (what the model sees).
#   Hits that are not walls (wall_vertical == 0) or misses are skipped. The experiment's tags are
#   scattered and labelled. Only the CSV is used (no room scan), so the plot shows exactly the
#   training data.
#   Figures are shown interactively, or saved as experiment_{exp}.png when --out-dir is given.

LIST_COLS = ["tags", "tag_xyz", "centroid", "north_dir", "wall_hit", "wall_angle", "wall_vertical"]
DIR_NAMES = ["N", "S", "E", "W"]
DIR_COLORS = {"N": "tab:blue", "S": "tab:green", "E": "tab:red", "W": "tab:purple"}
PATCH_HALF = 0.05  # patch half-width and half-height [m]


def load(path):
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
        for c in LIST_COLS:
            df[c] = df[c].map(json.loads)
    return df.drop_duplicates(["experiment", "combination_id", "permutation_id"]).reset_index(drop=True)


def ray_dirs(north):
    """N, S, E, W unit rays from the stored North ray (W = +X = N rotated +90 deg about +z, E = -W)."""
    n = np.asarray(north, dtype=float)
    e = np.array([n[1], -n[0], 0.0])
    return np.stack([n, -n, e, -e])


def plot_experiment(exp, g, show_rays=True, view="top"):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Tags of this experiment (deduplicated across sub-clusters)
    tags = {}
    for names, xyz in zip(g.tags, g.tag_xyz):
        tags.update(zip(names, xyz))
    T = np.array(list(tags.values()))
    ax.scatter(T[:, 0], T[:, 1], T[:, 2], color="tab:orange", s=25, zorder=5, label="tags")
    for name, p in tags.items():
        ax.text(p[0], p[1], p[2], f" {name}", fontsize=7, color="tab:orange")

    n_patch = n_skip = 0
    for _, r in g.iterrows():
        c = np.asarray(r.centroid, dtype=float)
        dirs = ray_dirs(r.north_dir)
        ax.scatter(*c, color="blue", s=20, zorder=6)
        ax.quiver(*c, *(-dirs[2] * 0.2), color="blue", linewidth=1.5)  # pose +X = W = toward tag 0

        for name, d, dist, angle, vert in zip(DIR_NAMES, dirs, r.wall_hit, r.wall_angle, r.wall_vertical):
            dist = float(dist)
            if not np.isfinite(dist) or not vert:
                n_skip += 1
                continue
            hit = c + dist * d
            # Patch normal: -ray rotated CCW by the signed incidence angle (vizualizeWalls convention)
            a = np.deg2rad(float(angle))
            n0 = -d[:2]
            nx, ny = np.cos(a) * n0[0] - np.sin(a) * n0[1], np.sin(a) * n0[0] + np.cos(a) * n0[1]
            tangent, up = np.array([-ny, nx, 0.0]), np.array([0.0, 0.0, 1.0])
            corners = [hit + PATCH_HALF * (sh * tangent + sv * up) for sh, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ax.add_collection3d(Poly3DCollection([corners], alpha=0.5, facecolor=DIR_COLORS[name],
                                                 edgecolor="black", linewidth=0.6))
            n_patch += 1
            if show_rays:
                ax.plot(*np.stack([c, hit]).T, "--", color=DIR_COLORS[name], alpha=0.4, linewidth=1)

    for name, col in DIR_COLORS.items():
        ax.plot([], [], color=col, linewidth=4, label=f"{name} wall")
    ax.legend(loc="upper right", fontsize=8)

    # Equal axis scales around everything drawn
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    mid, half = lims.mean(axis=1), (lims[:, 1] - lims[:, 0]).max() / 2
    ax.set_xlim3d(mid[0] - half, mid[0] + half)
    ax.set_ylim3d(mid[1] - half, mid[1] + half)
    ax.set_zlim3d(mid[2] - half, mid[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (up)")
    if view == "top":
        ax.view_init(elev=90, azim=-90)
        ax.set_zticks([])
        ax.set_zlabel("")
    K = len(g.tags.iloc[0])
    ax.set_title(f"Experiment {exp}: {len(g)} sub-clusters, K={K}, "
                 f"{n_patch} wall hits, {n_skip} misses/non-wall skipped")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-experiment wall-hit plots from cluster_features.py output")
    ap.add_argument("csv", help="cluster_features.py output (.csv or .pkl)")
    ap.add_argument("--out-dir", default=None, help="save experiment_{exp}.png here instead of showing")
    ap.add_argument("--max-subclusters", type=int, default=None,
                    help="randomly draw at most this many sub-clusters per experiment")
    ap.add_argument("--view", choices=["top", "3d"], default="top")
    ap.add_argument("--no-rays", action="store_true", help="hide the dashed centroid -> hit rays")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(args.csv)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    for exp, g in df.groupby("experiment", sort=True):
        if args.max_subclusters and len(g) > args.max_subclusters:
            g = g.sample(args.max_subclusters, random_state=args.seed)
        fig = plot_experiment(exp, g, show_rays=not args.no_rays, view=args.view)
        if args.out_dir:
            path = os.path.join(args.out_dir, f"experiment_{exp}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"saved {path}")
    if not args.out_dir:
        plt.show()

"""
Build multi-tag sub-cluster samples from the room sweeps, for train_roomsweeps.py.

Room-sweep counterpart of multitag/cluster_features.py: same sub-cluster definition (K of an experiment's
N = 15 tags, P random orderings each), same link order, same local frame and N/S/E/W wall targets.
Differences:

  * Input is the cached bidirectional θ̂ / α̂ of every tag pair (roomsweeps_wall_distance.load_bidir), not
    link_features.csv. avgV_A / avgV_B are not available.
  * Every experiment is in the same room (OCS 2214), so tag positions come from the merged scan's floor-label
    markers (roomscans/merged_markers.csv), all in one frame. No per-experiment levelling or wall alignment.
  * The walls are the four room walls found by roomsweeps_wall_distance.find_walls() (the same walls every
    roomsweeps_* script uses), and rays are intersected with them analytically rather than cast into the
    scan, so furniture is never reported as a wall. Every hit is therefore vertical (wall_vertical = 1).
  * Output is a dense .npz (the arrays train_cluster._read_raw would build) instead of a JSON-cell CSV.

Sub-clusters
------------
  For every experiment, every K-tag combination of its tags (optionally only compact ones, --max-span-cm, and
  at most --max-combos-per-exp of them), and min(P, K!) random orderings per combination. Position i in an
  ordering is tag i; links are listed (t0,t1), (t0,t2), …, (t0,tK-1), (t1,t2), …, (tK-2,tK-1).

Frames (--frame)
----------------
  local (default, = cluster_features.py)  +X is the horizontal direction centroid -> tag 0, +Y is +X turned
                                          +90 deg about +z. Orderings whose tag 0 sits on the centroid (e.g.
                                          the middle tag of a straight line) have no +X and are not sampled.
  room                                    +X points from the Edge02 wall (beyond grid column 0) towards the
                                          Edge01 wall (beyond column 16), the same for every sub-cluster.
  Rays from the centroid: N = -Y, S = +Y, E = -X, W = +X.

Saved arrays (n_sub sub-clusters, K tags, n_pairs = K(K-1)/2, n_run = 3, n_freq)
-------------------------------------------------------------------------------
  link           (n_sub, n_run, n_freq, 2, n_pairs)  θ̂ (mod π, rad), α̂; NaN where a run is missing
  keys           (n_sub, 3)   experiment, combination_id, permutation_id
  runs, freqs    run numbers, frequencies [MHz]
  tag_nums       (n_sub, K)   tag numbers in permutation order
  labels         (n_sub, K)   floor labels (1-136) in permutation order
  tag_xyz        (n_sub, K, 3) [m], merged-scan frame, z = --tag-height
  centroid       (n_sub, 3)
  pose           (n_sub, 4)   quaternion [x, y, z, w] rotating +X onto the frame's +X (a yaw)
  ray_dirs       (n_sub, 4, 3) unit N, S, E, W ray directions
  wall_hit       (n_sub, 4)   [m] distance along each ray to the first wall
  wall_angle     (n_sub, 4)   [deg] signed azimuthal incidence angle (cluster_features.py convention)
  wall_vertical  (n_sub, 4)   all 1
  wall_id        (n_sub, 4)   index into wall_names of the wall each ray hits
  span_cm        (n_sub,)     largest nominal (tile grid) distance between two tags of the sub-cluster
  wall_names, wall_points (4, 2), wall_normals (4, 2, into the room), info (JSON string of the arguments)

Usage
-----
  python roomsweeps_cluster_features.py K P [--max-span-cm S] [--max-combos-per-exp M] [--frame local|room]
                                        [--experiments 1 2 ...] [--freq-min 775 --freq-max 995] [--seed 0]
  Default output: data/roomsweeps_K{K}_P{P}[_span{S}cm][_max{M}][_room].npz next to this file.
"""
import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # roomsweeps_processing: shared roomsweeps helpers

from roomsweeps_wall_distance import TILE_CM, all_positions, find_walls, label_grid, load_bidir, markers  # noqa: E402

DIR_NAMES = ["N", "S", "E", "W"]
TAG_HEIGHT_M = 1.0          # real tags sit ~1 m above the floor (labels are floor markers)
MIN_FRAME_ARM_M = 0.01      # tag 0 must be at least this far (horizontally) from the centroid in the local frame


def sample_permutations(tags, P, rng, valid=None):
    """min(P, #valid) distinct orderings of `tags`, uniformly at random among those for which valid(perm) holds."""
    K = len(tags)
    total = math.factorial(K)
    if total <= 50000:
        perms = [p for p in itertools.permutations(tags) if valid is None or valid(p)]
        if P >= len(perms):
            return perms
        return [perms[i] for i in sorted(rng.choice(len(perms), size=P, replace=False))]
    seen, tries = {}, 0
    while len(seen) < P and tries < 100 * P:
        tries += 1
        p = tuple(int(t) for t in rng.permutation(tags))
        if valid is None or valid(p):
            seen.setdefault(p, None)
    return list(seen)


def experiment_tensor(bidir, tags, freqs):
    """F[feat, a, b, run, freq] over the experiment's tags (index into `tags`), feat = θ̂, α̂; symmetric in a, b."""
    idx = {t: i for i, t in enumerate(tags)}
    runs = np.sort(bidir["run"].unique())
    b = bidir[bidir["freq"].isin(freqs) & bidir["T1"].isin(idx) & bidir["T2"].isin(idx)]
    a_i, b_i = b["T1"].map(idx).to_numpy(), b["T2"].map(idx).to_numpy()
    r_i, f_i = np.searchsorted(runs, b["run"]), np.searchsorted(freqs, b["freq"])
    F = np.full((2, len(tags), len(tags), len(runs), len(freqs)), np.nan, np.float32)
    for k, col in enumerate(["THETA", "alpha"]):
        v = b[col].to_numpy(np.float32)
        F[k, a_i, b_i, r_i, f_i] = v
        F[k, b_i, a_i, r_i, f_i] = v
    return F, runs


def raycast_walls(origin_xy, dirs_xy, wall_points, wall_normals):
    """
    Intersect horizontal rays with the wall lines (convex room: the first wall the ray moves towards).
    origin_xy (M, 2), dirs_xy (M, D, 2) unit; walls (W, 2) points and unit normals pointing into the room.
    Returns t_hit (M, D) [m], wall index (M, D), signed incidence angle (M, D) [deg].
    """
    dn = np.einsum("mdk,wk->mdw", dirs_xy, wall_normals)                       # d·n
    pn = np.einsum("wk,wk->w", wall_points, wall_normals)[None] - origin_xy @ wall_normals.T   # (p - c)·n  (M, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dn < -1e-9, pn[:, None, :] / dn, np.inf)
    t[t <= 0] = np.inf
    wid = np.argmin(t, axis=2)
    t_hit = np.take_along_axis(t, wid[..., None], axis=2)[..., 0]
    n = wall_normals[wid]                                                      # (M, D, 2), faces the origin
    r = -dirs_xy
    cos_az = np.clip(np.sum(r * n, axis=2), -1.0, 1.0)
    sign = np.sign(r[..., 0] * n[..., 1] - r[..., 1] * n[..., 0])
    return t_hit, wid, sign * np.degrees(np.arccos(cos_az))


def yaw_quat(x_axis):
    """Quaternions [x, y, z, w] of the yaw rotations taking +X onto the horizontal unit vectors x_axis (M, 2)."""
    half = np.arctan2(x_axis[:, 1], x_axis[:, 0]) / 2
    z = np.zeros(len(x_axis))
    return np.stack([z, z, np.sin(half), np.cos(half)], axis=1)


def build(args):
    positions = all_positions()
    exps = [str(e) for e in args.experiments] if args.experiments else sorted(positions, key=int)
    missing = set(exps) - set(positions)
    if missing:
        sys.exit(f"experiments not in roomsweeps_data: {sorted(missing, key=int)}")

    walls = find_walls()
    wall_names = [n.split(" (", 1)[0] for n in walls]
    wall_points = np.array([w[0] for w in walls.values()])
    wall_normals = np.array([w[1] for w in walls.values()])
    # Room frame +X: towards the Edge01 wall (beyond column 16), i.e. against that wall's inward normal.
    room_x = -wall_normals[wall_names.index("Edge01 wall")]

    rng = np.random.default_rng(args.seed)
    out = {k: [] for k in ["link", "keys", "tag_nums", "labels", "tag_xyz", "span_cm"]}
    runs_all = freqs = None

    for exp in exps:
        pos = positions[exp]
        tags = sorted(int(t) for t in pos.index)
        if args.K > len(tags):
            sys.exit(f"experiment {exp}: K={args.K} > N={len(tags)} tags")
        bidir = load_bidir(exp)
        f_all = np.sort(bidir["freq"].unique())
        f_sel = f_all[(f_all >= args.freq_min) & (f_all <= args.freq_max)]
        if freqs is None:
            freqs = f_sel
        elif not np.array_equal(freqs, f_sel):
            sys.exit(f"experiment {exp}: frequencies differ from the first experiment's")
        F, runs = experiment_tensor(bidir, tags, freqs)
        if runs_all is None:
            runs_all = runs
        elif not np.array_equal(runs_all, runs):
            sys.exit(f"experiment {exp}: runs {runs} differ from {runs_all}")

        labels = np.array([int(pos.loc[t, "label"]) for t in tags])
        grid = np.array([label_grid(l) for l in labels], float)                          # (N, 2) col, row
        xy = markers.loc[labels, ["x", "y"]].to_numpy(float)                              # (N, 2) m

        combos = []
        for combo in itertools.combinations(range(len(tags)), args.K):
            c = np.array(combo)
            g = grid[c]
            span = TILE_CM * max(np.linalg.norm(g[i] - g[j]) for i, j in itertools.combinations(range(args.K), 2))
            if args.max_span_cm is not None and span > args.max_span_cm + 1e-6:
                continue
            combos.append((combo, span))
        n_compact = len(combos)
        if args.max_combos_per_exp is not None and len(combos) > args.max_combos_per_exp:
            keep = sorted(rng.choice(len(combos), args.max_combos_per_exp, replace=False))
            combos = [combos[i] for i in keep]

        n_sub0 = len(out["keys"])
        n_no_frame = 0
        all_combos = {c: i for i, c in enumerate(itertools.combinations(range(len(tags)), args.K))}
        for combo, span in combos:
            ci = all_combos[combo]                       # index among all N-choose-K combinations
            cen = xy[list(combo)].mean(axis=0)
            valid = None
            if args.frame == "local":
                def valid(p, cen=cen):
                    return np.linalg.norm(xy[p[0]] - cen) >= MIN_FRAME_ARM_M
            perms = sample_permutations(list(combo), args.P, rng, valid)
            if not perms:
                n_no_frame += 1
                continue
            for pi, perm in enumerate(perms):
                perm = list(perm)
                I, J = np.array(list(itertools.combinations(perm, 2))).T
                out["link"].append(F[:, I, J].transpose(2, 3, 0, 1))                     # (run, freq, 2, pairs)
                out["keys"].append((int(exp), ci, pi))
                out["tag_nums"].append([tags[i] for i in perm])
                out["labels"].append(labels[perm])
                out["tag_xyz"].append(np.c_[xy[perm], np.full(len(perm), args.tag_height)])
                out["span_cm"].append(span)
        n_new = len(out["keys"]) - n_sub0
        msg = (f"experiment {exp:>2}: {len(tags)} tags, {n_compact} of {math.comb(len(tags), args.K)} combinations"
               f"{' within ' + format(args.max_span_cm, 'g') + ' cm' if args.max_span_cm is not None else ''}")
        if args.max_combos_per_exp is not None:
            msg += f", {len(combos)} kept"
        if n_no_frame:
            msg += f", {n_no_frame} without a valid tag 0"
        print(msg + f" -> {n_new} sub-clusters")

    if not out["keys"]:
        sys.exit("no sub-clusters; nothing saved")

    tag_xyz = np.array(out["tag_xyz"])                                                    # (n, K, 3)
    centroid = tag_xyz.mean(axis=1)
    if args.frame == "local":
        x = tag_xyz[:, 0, :2] - centroid[:, :2]
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    else:
        x = np.broadcast_to(room_x, (len(tag_xyz), 2)).copy()
    y = np.stack([-x[:, 1], x[:, 0]], axis=1)                                            # +X turned +90 deg
    dirs_xy = np.stack([-y, y, -x, x], axis=1)                                           # N, S, E, W  (n, 4, 2)
    t_hit, wid, angle = raycast_walls(centroid[:, :2], dirs_xy, wall_points, wall_normals)
    if not np.isfinite(t_hit).all():
        sys.exit("some rays hit no wall — is a centroid outside the room?")

    info = vars(args) | {"n_sub": len(out["keys"]), "wall_names": wall_names,
                         "experiments": sorted({k[0] for k in out["keys"]})}
    return dict(
        link=np.array(out["link"], np.float32),
        keys=np.array(out["keys"], np.int64),
        runs=runs_all, freqs=freqs,
        tag_nums=np.array(out["tag_nums"], np.int64),
        labels=np.array(out["labels"], np.int64),
        tag_xyz=tag_xyz,
        centroid=centroid,
        pose=yaw_quat(x),
        ray_dirs=np.concatenate([dirs_xy, np.zeros(dirs_xy.shape[:2] + (1,))], axis=2),
        wall_hit=t_hit,
        wall_angle=angle,
        wall_vertical=np.ones_like(wid),
        wall_id=wid,
        span_cm=np.array(out["span_cm"]),
        wall_names=np.array(wall_names),
        wall_points=wall_points,
        wall_normals=wall_normals,
        info=np.array(json.dumps(info)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("K", type=int, help="tags per sub-cluster")
    ap.add_argument("P", type=int, help="random orderings sampled per combination")
    ap.add_argument("--output", default=None)
    ap.add_argument("--experiments", type=int, nargs="+", default=None, help="default: every experiment folder")
    ap.add_argument("--max-span-cm", type=float, default=None,
                    help="keep only combinations whose tags are at most this far apart (nominal tile grid)")
    ap.add_argument("--max-combos-per-exp", type=int, default=None,
                    help="randomly keep at most this many combinations per experiment")
    ap.add_argument("--frame", choices=["local", "room"], default="local",
                    help="local: +X = centroid -> tag 0 (cluster_features.py); room: +X towards the Edge01 wall")
    ap.add_argument("--freq-min", type=float, default=775)
    ap.add_argument("--freq-max", type=float, default=995)
    ap.add_argument("--tag-height", type=float, default=TAG_HEIGHT_M, help="tag z [m] (the rays are horizontal)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.K < 2 or args.P < 1:
        sys.exit("need K >= 2 and P >= 1")

    suffix = (f"_span{args.max_span_cm:g}cm" if args.max_span_cm is not None else "") \
        + (f"_max{args.max_combos_per_exp}" if args.max_combos_per_exp is not None else "") \
        + ("_room" if args.frame == "room" else "")
    out_path = args.output or str(HERE / "data" / f"roomsweeps_K{args.K}_P{args.P}{suffix}.npz")
    data = build(args)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **data)
    n_sub, n_run, n_freq, _, n_pairs = data["link"].shape
    print(f"{n_sub} sub-clusters (K={args.K}, {n_pairs} links) × {n_run} runs × {n_freq} freqs "
          f"[{data['freqs'][0]}–{data['freqs'][-1]} MHz], frame={args.frame} -> {out_path}")
    print("wall distance [m] min/median/max per ray N,S,E,W:",
          [f"{a:.2f}/{b:.2f}/{c:.2f}" for a, b, c in zip(data["wall_hit"].min(0), np.median(data["wall_hit"], 0),
                                                         data["wall_hit"].max(0))])


if __name__ == "__main__":
    main()

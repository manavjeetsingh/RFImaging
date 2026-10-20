import os, sys, argparse, itertools, json, math
import numpy as np
import pandas as pd
import open3d as o3d

# Usage: python cluster_features.py K P [--input link_features.csv] [--output cluster_features_K{K}_P{P}.csv] [--seed 0]
#   Builds sub-cluster feature rows from linkFeatures.py output. Each experiment (dataset) is a cluster
#   of N tags; for every one of the N-choose-K tag combinations, P distinct orderings (permutations) of
#   the K tags are sampled at random (all K! of them if P >= K!). The ordering is the tag naming of the
#   sub-cluster: position i in the permutation is tag i, and links are listed in ascending order of
#   these names, i.e. (t1,t2), (t1,t3), ..., (t1,tK), (t2,t3), ..., (tK-1,tK).
#
#   One output row per (experiment, combination, permutation, run, frequency) with list-valued cells:
#     tags        tag names in permutation order              (K)
#     tag_nums    tag numbers in permutation order            (K)
#     tag_xyz     [x, y, z] of each tag in permutation order  (K x 3)
#     links       [tagA, tagB] of each link in link order     (K(K-1)/2 x 2)
#     theta, alpha, avgV_A, avgV_B   one value per link in link order (K(K-1)/2)
#   theta and alpha are symmetric; avgV_A/avgV_B are oriented so A is the first tag of the link in
#   the permuted order (the stored A/B values are swapped when the permutation reverses a link).
#
#   Wall features (same as T2TSionnaSim MLScripts/subdataset_creation.py, clamped to the ground plane):
#   four horizontal rays are cast from the sub-cluster centroid into the room scan (PLY_PATHS), using
#   the sim's local frame: the pose's +X points from the centroid to tag 0 (first tag of the
#   permutation) and the rays are N = -Y, S = +Y, E = -X, W = +X. So W points at tag 0, E away from
#   it, and N is the tag-0 direction rotated 90 deg clockwise about +z (seen from above).
#     centroid       [x, y, z] of the K tags
#     pose           quaternion [x, y, z, w] rotating +X onto centroid -> tag 0 (3-D, as in the sim)
#     north_dir      unit [x, y, z] of the North ray
#     wall_hit       hit distance [m] per ray, order N, S, E, W (inf on miss)
#     wall_norm      [x, y, z] surface normal at each hit, flipped to face the centroid
#     wall_angle     signed azimuthal incidence angle [deg]; 0 where the hit is not a wall
#     wall_vertical  1 if the hit surface is vertical (|n_z| < 0.5), else 0
#   Tag positions are ground truth (points.csv) for now. Scan geometry within TAG_CLEAR_RADIUS of any
#   tag of the experiment (the tags themselves, their holders) is removed before ray casting so the
#   rays pass through it instead of reporting it as a wall (--tag-clear, 0 disables).
#   Each experiment's clearing is shown in an Open3D window (removed triangles red, tags blue, wire
#   sphere = clearing radius) and must be approved in the terminal after closing it: y accepts, s skips
#   the experiment (no rows written for it), q aborts. --no-review disables this for batch runs.
#
#   Levelling: some room scans are not z-up (exp 2 and 6 are tilted ~18 deg). The tags of an experiment
#   are assumed to lie in one plane, so each experiment is rotated (about the origin) so that the
#   normal of the plane fitted through all its tags becomes +z. The rotation is applied to the tags
#   and the room scan before ray casting, and every coordinate written out (tag_xyz, centroid, pose,
#   north_dir, wall_norm) is in this levelled frame.
#     level_rot      3x3 rotation matrix R, levelled = R @ original (points.csv / .ply frame)
#   List cells are written as JSON strings in CSV; an output ending in .pkl keeps them as Python lists.

# Subdirectories (experiment numbers in link_features.csv) to use; edit to select datasets.
SUBDIRS = ["1", "2", "3", "4", "5", "6", "7", "8", "9","10",
           "11","12","13","14","15","16"]

# Subdirectory -> room scan (.ply), relative to this file's directory.
PLY_PATHS = {
    "1": "1/2214_1.ply",
    "2": "2/2214_2.ply",
    "3": "3/2214_3.ply",
    "4": "4/2214_4.ply",
    "5": "5/2214_5.ply",
    "6": "6/2214_6.ply",
    "7": "7/2214_7.ply",
    "8": "8/2214_8.ply",
    "9": "9/2214_9.ply",
    "10": "10/2214_10_filled.ply",
    "11": "11/2214_11_filled_erased.ply",
    "12": "12/2214_12_filled_erased.ply",
    "13": "13/2214_13_filled.ply",
    "14": "14/2214_14_filled_erased.ply",
    "15": "15/2214_15_erased.ply",
    "16": "16/2214_16_filled_erased.ply",
}

FREQ, RUN = "Frequency (MHz)", "Run Exp Num"
FEATS = ["theta", "alpha", "avgV_A", "avgV_B"]
WALL_COLS = ["level_rot", "centroid", "pose", "north_dir", "wall_hit", "wall_norm", "wall_angle", "wall_vertical"]
DIR_NAMES = ["N", "S", "E", "W"]
VERTICAL_NZ_MAX = 0.5  # a hit is a wall when its normal is close to horizontal
TAG_CLEAR_RADIUS = 0.2  # [m] scan triangles with a vertex this close to a tag are invisible to the rays
REVIEW_MARGIN = 1.0  # [m] the clearing review shows the scan within this distance of the tags' bounding box


def sample_permutations(tags, P, rng):
    """Return min(P, K!) distinct orderings of tags, chosen uniformly at random."""
    K = len(tags)
    total = math.factorial(K)
    if P >= total:
        return list(itertools.permutations(tags))
    if total <= 50000:
        perms = list(itertools.permutations(tags))
        return [perms[i] for i in rng.choice(total, size=P, replace=False)]
    seen = {}
    while len(seen) < P:
        p = tuple(rng.permutation(tags))
        seen.setdefault(p, None)
    return list(seen)


def experiment_tensors(g):
    """Per experiment: feature tensor F[feat, a, b, run, freq] over directed tag pairs, tag info, runs, freqs."""
    nums = sorted(set(g.tagA_num) | set(g.tagB_num))
    N = max(nums) + 1
    runs, freqs = np.sort(g[RUN].unique()), np.sort(g[FREQ].unique())
    ri, fi = np.searchsorted(runs, g[RUN]), np.searchsorted(freqs, g[FREQ])
    a, b = g.tagA_num.to_numpy(), g.tagB_num.to_numpy()

    F = np.full((len(FEATS), N, N, len(runs), len(freqs)), np.nan)
    for k, f in enumerate(FEATS):
        # Reverse direction: theta/alpha unchanged, avgV_A and avgV_B swap roles.
        rev = {"avgV_A": "avgV_B", "avgV_B": "avgV_A"}.get(f, f)
        F[k, a, b, ri, fi] = g[f].to_numpy()
        F[k, b, a, ri, fi] = g[rev].to_numpy()

    names, xyz = {}, {}
    for side in ("A", "B"):
        t = g.drop_duplicates(f"tag{side}_num")
        for _, r in t.iterrows():
            n = int(r[f"tag{side}_num"])
            names[n] = r[f"tag{side}"]
            xyz[n] = [float(r[f"tag{side}_{ax}"]) for ax in "xyz"]
    return F, nums, names, xyz, runs, freqs


def pose_xyzw(centroid, tag0):
    """Quaternion (x, y, z, w) rotating +X to the 3-D direction centroid -> tag 0 (sim's _pose_xyzw)."""
    ref = np.array([1.0, 0.0, 0.0])
    delta = np.asarray(tag0, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    norm = np.linalg.norm(delta)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    d = delta / norm
    dot = float(np.clip(ref @ d, -1.0, 1.0))
    if dot >= 1.0 - 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot <= -1.0 + 1e-9:
        perp = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        axis = np.cross(ref, perp)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(ref, d)
    axis /= np.linalg.norm(axis)
    half = np.arccos(dot) / 2.0
    return np.r_[axis * np.sin(half), np.cos(half)]


def wall_raycast(scene, tag_xyz):
    """
    Cast N/S/E/W rays from each sub-cluster's centroid. tag_xyz is (M, K, 3) in permutation order.
    Returns a list of M dicts with the WALL_COLS features.
    """
    centroid = tag_xyz.mean(axis=1)
    pose = np.array([pose_xyzw(c, t) for c, t in zip(centroid, tag_xyz[:, 0])])
    # Pose clamped to the ground plane: local +X is the horizontal direction centroid -> tag 0.
    x = tag_xyz[:, 0] - centroid
    x[:, 2] = 0.0
    x_len = np.linalg.norm(x, axis=1, keepdims=True)
    if (x_len < 1e-9).any():
        raise ValueError("tag 0 is vertically above the centroid; ray directions are undefined")
    x /= x_len
    y = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], axis=1)  # local +Y: +X rotated +90 deg
    dirs = np.stack([-y, y, -x, x], axis=1)                        # N, S, E, W  (M, 4, 3)

    M, D = dirs.shape[:2]
    origins = np.repeat(centroid, D, axis=0)
    rays = o3d.core.Tensor(np.concatenate([origins, dirs.reshape(-1, 3)], axis=1).astype(np.float32))
    res = scene.cast_rays(rays)
    t_hit = res["t_hit"].numpy().astype(np.float64).reshape(M, D)
    t_norm = res["primitive_normals"].numpy().astype(np.float64).reshape(M, D, 3)
    t_norm[np.sum(dirs * t_norm, axis=2) > 0] *= -1  # face the ray origin

    # Signed angle between -ray and the normal in the horizontal plane (sign: z of cross(-ray, normal)).
    hit = np.isfinite(t_hit)
    n_xy = t_norm[:, :, :2]
    n_xy_len = np.linalg.norm(n_xy, axis=2)
    vertical = hit & (np.abs(t_norm[:, :, 2]) < VERTICAL_NZ_MAX) & (n_xy_len > 1e-9)
    n_u = n_xy / np.where(n_xy_len > 1e-9, n_xy_len, 1.0)[..., None]
    r = -dirs[:, :, :2]
    cos_az = np.clip(np.sum(r * n_u, axis=2), -1.0, 1.0)
    sign = np.sign(r[:, :, 0] * n_u[:, :, 1] - r[:, :, 1] * n_u[:, :, 0])
    angle = np.where(vertical, sign * np.degrees(np.arccos(cos_az)), 0.0)

    return [{
        "centroid": centroid[m].tolist(),
        "pose": pose[m].tolist(),
        "north_dir": dirs[m, 0].tolist(),
        "wall_hit": t_hit[m].tolist(),
        "wall_norm": t_norm[m].tolist(),
        "wall_angle": angle[m].tolist(),
        "wall_vertical": vertical[m].astype(int).tolist(),
    } for m in range(M)]


def level_rotation(points):
    """Smallest rotation R taking the normal of the plane through `points` (Nx3) to +z."""
    pts = np.asarray(points, dtype=np.float64)
    n = np.linalg.svd(pts - pts.mean(axis=0))[2][-1]
    n *= np.sign(n[2])  # scans are roughly z-up, so the plane normal points to +z, not -z
    z = np.array([0.0, 0.0, 1.0])
    axis, c = np.cross(n, z), float(n @ z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * Kx @ Kx


def load_mesh(ply_path, R=None):
    """Room scan as a legacy TriangleMesh, rotated by R (levelled = R @ original) if given."""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"no triangles loaded from {ply_path}")
    if R is not None:
        mesh.rotate(R, center=(0.0, 0.0, 0.0))
    return mesh


def clear_around(mesh, points, radius):
    """
    Remove the triangles of `mesh` with any vertex within `radius` of any of `points` (Mx3).
    Returns (kept mesh, mesh of the removed triangles).
    """
    removed = o3d.geometry.TriangleMesh(mesh)
    if radius <= 0:
        removed.remove_triangles_by_mask(np.ones(len(mesh.triangles), dtype=bool))
        return mesh, removed
    v = np.asarray(mesh.vertices)
    near = np.zeros(len(v), dtype=bool)
    for p in np.asarray(points, dtype=np.float64):
        near |= np.sum((v - p) ** 2, axis=1) < radius ** 2
    tri_mask = near[np.asarray(mesh.triangles)].any(axis=1)
    mesh.remove_triangles_by_mask(tri_mask)
    mesh.remove_unreferenced_vertices()
    removed.remove_triangles_by_mask(~tri_mask)
    removed.remove_unreferenced_vertices()
    print(f"  removed {tri_mask.sum()} triangles within {radius} m of the tags")
    return mesh, removed


def review_clearing(mesh, removed, points, radius, title):
    """
    Show the scan around the tags with the removed triangles in red and the tags as blue spheres
    (wireframe sphere = clearing radius), then ask for approval in the terminal once the window is closed.
    Returns True to accept, False to skip the experiment; quits on 'q'.
    """
    pts = np.asarray(points, dtype=np.float64)
    box = o3d.geometry.AxisAlignedBoundingBox(pts.min(axis=0) - REVIEW_MARGIN, pts.max(axis=0) + REVIEW_MARGIN)
    kept = mesh.crop(box)
    if not kept.has_vertex_colors():
        kept.paint_uniform_color([0.7, 0.7, 0.7])
    kept.compute_vertex_normals()
    cut = o3d.geometry.TriangleMesh(removed).paint_uniform_color([1.0, 0.0, 0.0])
    cut.compute_vertex_normals()
    geoms = [kept, cut]
    for p in pts:
        tag = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(p)
        tag.paint_uniform_color([0.0, 0.3, 1.0])
        tag.compute_vertex_normals()
        ring = o3d.geometry.LineSet.create_from_triangle_mesh(
            o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8).translate(p))
        ring.paint_uniform_color([0.0, 0.3, 1.0])
        geoms += [tag, ring]
    print(f"  review: red = removed ({len(removed.triangles)} triangles), blue = tags; close the window to continue")
    o3d.visualization.draw_geometries(geoms, window_name=title, mesh_show_back_face=True)
    while True:
        ans = input("  accept clearing? [y]es / [s]kip experiment / [q]uit: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("s", "skip"):
            return False
        if ans in ("q", "quit"):
            sys.exit("aborted during clearing review")


def load_scene(ply_path, R=None, clear_points=None, clear_radius=TAG_CLEAR_RADIUS, review=False):
    """Raycasting scene of the room scan, or None when the clearing review rejects it."""
    mesh = load_mesh(ply_path, R)
    if clear_points is not None:
        mesh, removed = clear_around(mesh, clear_points, clear_radius)
        if review and clear_radius > 0 and not review_clearing(
                mesh, removed, clear_points, clear_radius, f"tag clearing: {os.path.basename(ply_path)}"):
            return None
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def build(df, K, P, rng, base_dir, tag_clear=TAG_CLEAR_RADIUS, review=False):
    rows = []
    for exp, g in df.groupby("experiment", sort=True):
        F, nums, names, xyz, runs, freqs = experiment_tensors(g)
        if K > len(nums):
            raise ValueError(f"experiment {exp}: K={K} > N={len(nums)} tags")
        subs = []  # (combination_id, permutation_id, perm) for this experiment
        for ci, combo in enumerate(itertools.combinations(nums, K)):
            for pi, perm in enumerate(sample_permutations(list(combo), P, rng)):
                subs.append((ci, pi, [int(t) for t in perm]))
        R = level_rotation([xyz[t] for t in nums])
        xyz = {t: (R @ np.asarray(p)).tolist() for t, p in xyz.items()}
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
        print(f"experiment {exp}: levelled by {tilt:.2f} deg")
        scene = load_scene(os.path.join(base_dir, PLY_PATHS[str(exp)]), R,
                           clear_points=[xyz[t] for t in nums], clear_radius=tag_clear, review=review)
        if scene is None:
            print(f"experiment {exp}: skipped (clearing rejected)")
            continue
        walls = wall_raycast(scene, np.array([[xyz[t] for t in perm] for _, _, perm in subs]))

        for (ci, pi, perm), wall in zip(subs, walls):
            wall = {"level_rot": R.tolist(), **wall}
            pairs = list(itertools.combinations(perm, 2))
            I, J = np.array(pairs).T
            V = F[:, I, J]  # (feat, link, run, freq)
            if np.isnan(V).any():
                raise ValueError(f"experiment {exp}: missing link data for tags {perm}")
            common = {
                "experiment": exp, "N": len(nums), "K": K,
                "combination_id": ci, "permutation_id": pi,
                "combination": sorted(perm),
                "tags": [names[t] for t in perm],
                "tag_nums": perm,
                "tag_xyz": [xyz[t] for t in perm],
                "links": [[names[i], names[j]] for i, j in pairs],
                **wall,
            }
            for r, run in enumerate(runs):
                for q, freq in enumerate(freqs):
                    row = dict(common)
                    row[RUN], row[FREQ] = int(run), int(freq)
                    for k, f in enumerate(FEATS):
                        row[f] = V[k, :, r, q].tolist()
                    rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("K", type=int, help="tags per sub-cluster")
    ap.add_argument("P", type=int, help="random permutations sampled per combination")
    ap.add_argument("--input", default=os.path.join(here, "link_features.csv"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag-clear", type=float, default=TAG_CLEAR_RADIUS,
                    help="radius [m] around each tag whose scan geometry the rays ignore (0 disables)")
    ap.add_argument("--no-review", action="store_true",
                    help="skip the per-experiment visual approval of the tag clearing")
    args = ap.parse_args()
    if args.K < 2 or args.P < 1:
        sys.exit("need K >= 2 and P >= 1")
    out_path = args.output or os.path.join(here, f"cluster_features_K{args.K}_P{args.P}.csv")

    df = pd.read_csv(args.input)
    missing = set(SUBDIRS) - set(df.experiment.astype(str))
    missing |= set(SUBDIRS) - PLY_PATHS.keys()
    if missing:
        sys.exit(f"subdirectories missing from {args.input} or PLY_PATHS: {sorted(missing)}")
    df = df[df.experiment.astype(str).isin(SUBDIRS)]
    out = build(df, args.K, args.P, np.random.default_rng(args.seed), here, args.tag_clear,
                review=not args.no_review)
    if out.empty:
        sys.exit("no experiments accepted; nothing saved")
    if out_path.endswith(".pkl"):
        out.to_pickle(out_path)
    else:
        list_cols = ["combination", "tags", "tag_nums", "tag_xyz", "links"] + WALL_COLS + FEATS
        out[list_cols] = out[list_cols].map(json.dumps)
        out.to_csv(out_path, index=False)
    print(f"{df.experiment.nunique()} experiments, K={args.K}, P={args.P} -> {len(out)} rows saved to {out_path}")

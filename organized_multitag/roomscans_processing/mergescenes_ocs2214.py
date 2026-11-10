"""
Merge two or more PLY scans of the same room into one scene (a union).

Pipeline
  1. Level every scan: detect the floor plane, rotate so the floor normal is +Z
     and translate so the floor sits at z = 0.
  2. Align every scan to the first one in the floor (x-y) plane: brute-force yaw
     search + FFT cross-correlation of top-down occupancy images, then ICP refine
     constrained to yaw + x/y translation.
  3. Union: start from the first scan, and from each next scan add only the
     triangles that are not already explained by the merged geometry. Objects
     that were moved between scans therefore appear at every place they were seen.

Requires: pip install open3d numpy
"""

import os

import numpy as np
import open3d as o3d

# ----------------------------------------------------------------------------
# Scenes to merge. The first one is the reference frame.
# ----------------------------------------------------------------------------
SCENE_PATHS = [
    "../roomscans/mapping_2214_1_marked.ply",
    "../roomscans/mapping_2214_2_marked.ply",
    "../roomscans/mapping_2214_3_marked.ply",
    "../roomscans/mapping_2214_4_marked.ply",
    "../roomscans/mapping_2214_5_marked.ply",
    # "../roomscans/mapping_2214_3.ply",
    # "../roomscans/mapping_2214_4.ply",
    # "../roomscans/mapping_2214_5.ply",
]
# All outputs go to OUTPUT_DIR; if it does not exist, to the folder of the first scene.
OUTPUT_DIR = "/Users/manavjeet/git/RFImaging/organized_multitag/roomscans/merged_out"
OUTPUT_NAME = "merged.ply"
SAVE_ALIGNED = True          # also write <name>_aligned.ply for every input
SAVE_ADDED = True            # also write merged_added.ply (only the added parts)
SAVE_HIGHLIGHT = True        # also write merged_highlight.ply (added parts painted red)
# Marker CSVs from markLocations.py (<scan>_markers.csv, index,x,y,z) are looked up
# next to each scene ("_marked" suffix dropped), transformed into the merged frame
# and written to merged_markers.csv (index,x,y,z,scene).
MERGE_MARKERS = True

# Leveling
UP_AXIS_HINT = None          # None = auto, or e.g. [0, 0, 1] / [0, 1, 0] in the input frame
PLANE_DIST = 0.03            # RANSAC plane inlier distance (m)

# Alignment
GRID_RES = 0.05              # occupancy image resolution (m)
YAW_STEP_DEG = 1.0           # coarse yaw search step
ICP_DIST = [0.30, 0.10, 0.04]  # coarse-to-fine ICP correspondence distances (m)
COLOR_TOL = 0.12             # mean |RGB| difference (0-1) for two points to "match" in color

# Union
MERGE_DIST = 0.06            # vertices farther than this from existing geometry count as new (m)
CLUSTER_EPS = 0.12           # new vertices closer than this belong to the same object (m)
MIN_NEW_POINTS = 30          # drop new clusters with fewer vertices than this (noise)
MIN_NEW_SIZE = 0.10          # ... or smaller than this in every direction (m)
GROW_DIST = 0.10             # grow each kept object by this much so its edges/base come along (m)

# Color merge: colored features (e.g. markers/tags) that are new by color even
# if they are geometrically too small or flat to be caught above
COLOR_MERGE = True
COLOR_SAT_MIN = 0.35         # only saturated vertices count as color features (HSV saturation)
COLOR_DIFF = 0.15            # mean |RGB| difference to every nearby existing point to count as new
COLOR_SEARCH = 0.05          # radius for "nearby existing points" (m)
COLOR_MIN_POINTS = 3         # drop color clusters with fewer vertices than this
COLOR_GROW = 0.03            # grow each color feature by this much (m)


# ----------------------------------------------------------------------------
# Leveling
# ----------------------------------------------------------------------------
def rotation_between(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        axis = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def dominant_planes(pcd, n_planes=8):
    """Iteratively RANSAC the largest planes. Returns [(normal, d, n_inliers)]."""
    planes = []
    rest = pcd
    for _ in range(n_planes):
        if len(rest.points) < 1000:
            break
        (a, b, c, d), inl = rest.segment_plane(PLANE_DIST, 3, 1000)
        n = np.array([a, b, c])
        planes.append((n / np.linalg.norm(n), d, len(inl)))
        rest = rest.select_by_index(inl, invert=True)
    return planes


def find_up_axis(pcd):
    """Pick the up direction among the room's dominant (Manhattan) plane normals.

    The up axis is the one along which the room is shortest (ceiling height is
    smaller than the floor dimensions for normal rooms). See `up_sign` for how
    floor and ceiling are told apart. UP_AXIS_HINT is used as given, sign included.
    """
    pts = np.asarray(pcd.points)
    if UP_AXIS_HINT is not None:
        up = np.asarray(UP_AXIS_HINT, float)
        return up / np.linalg.norm(up)
    else:
        planes = dominant_planes(pcd)
        # build up to three mutually orthogonal axes from the largest planes
        axes = []
        for n, _, _ in sorted(planes, key=lambda p: -p[2]):
            if all(abs(np.dot(n, a)) < 0.2 for a in axes):
                axes.append(n)
            if len(axes) == 3:
                break
        if len(axes) < 3:
            # complete the frame from the two we have (or fall back to input axes)
            if len(axes) == 2:
                axes.append(np.cross(axes[0], axes[1]))
            else:
                axes = [np.eye(3)[i] for i in range(3)]

        def extent(a):
            h = pts @ a
            return np.percentile(h, 99) - np.percentile(h, 1)

        up = min(axes, key=extent)

    return up * up_sign(pcd, up)


def up_sign(pcd, up):
    """+1 if `up` points from floor to ceiling, -1 otherwise.

    1. Color: ceilings are usually much brighter than floors.
    2. Scanner position (origin of the scan): a hand-held scanner is closer to
       the floor than to the ceiling.
    3. Mass: furniture puts more geometry in the lower half.
    """
    pts = np.asarray(pcd.points)
    h = pts @ up
    lo, hi = np.percentile(h, 1), np.percentile(h, 99)
    if pcd.has_colors():
        bright = np.asarray(pcd.colors).mean(1)
        diff = bright[h > hi - 0.1].mean() - bright[h < lo + 0.1].mean()
        if abs(diff) > 0.08:
            print(f"    up sign from color (ceiling - floor brightness {abs(diff):.2f})")
            return 1 if diff > 0 else -1
    frac = (0.0 - lo) / (hi - lo)  # height of the scan origin in the room
    if 0.05 < frac < 0.95 and abs(frac - 0.5) > 0.08:
        print(f"    up sign from scanner height ({min(frac, 1 - frac):.2f} of room height)")
        return 1 if frac < 0.5 else -1
    mid = 0.5 * (lo + hi)
    print("    up sign from mass distribution")
    return 1 if np.sum(h < mid) >= np.sum(h > mid) else -1


def level_transform(mesh):
    """4x4 transform putting the floor on z = 0 with +Z up."""
    pcd = mesh.sample_points_uniformly(200000)
    up = find_up_axis(pcd)

    # refine with a RANSAC plane fit on the lowest horizontal slab
    pts = np.asarray(pcd.points)
    h = pts @ up
    floor_h = np.percentile(h, 2)
    slab = pcd.select_by_index(np.where(np.abs(h - floor_h) < 0.15)[0])
    (a, b, c, d), _ = slab.segment_plane(PLANE_DIST, 3, 2000)
    n = np.array([a, b, c])
    n_norm = np.linalg.norm(n)
    n, d = n / n_norm, d / n_norm
    if np.dot(n, up) < 0:
        n, d = -n, -d
    if np.dot(n, up) > np.cos(np.deg2rad(10)):  # trust the fit only if it agrees
        up = n

    R = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    T = np.eye(4)
    T[:3, :3] = R
    z = (pts @ R.T)[:, 2]
    # floor height: densest height bin near the bottom
    zz = z[z < np.percentile(z, 1) + 0.3]
    hist, edges = np.histogram(zz, bins=60)
    k = np.argmax(hist)
    T[2, 3] = -0.5 * (edges[k] + edges[k + 1])
    return T


# ----------------------------------------------------------------------------
# Alignment in the floor plane
# ----------------------------------------------------------------------------
def yaw_transform(theta, tx=0.0, ty=0.0):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3] = tx, ty
    return T


def occupancy(xy, origin, shape):
    ij = np.floor((xy - origin) / GRID_RES).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    img = np.zeros(shape, np.float32)
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.log1p(img)


def wall_points(pcd):
    """Points between floor and ceiling (walls + furniture) projected to x-y."""
    p = np.asarray(pcd.points)
    zmax = np.percentile(p[:, 2], 98)
    keep = (p[:, 2] > 0.15) & (p[:, 2] < zmax - 0.15)
    return p[keep, :2]


def coarse_align_2d(src_pcd, ref_pcd, n_candidates=4):
    """Brute-force yaw + FFT translation search between top-down images.

    Returns the best `n_candidates` transforms with distinct yaws (rectangular
    rooms give several similar peaks, so ICP picks among them afterwards).
    """
    ref_xy = wall_points(ref_pcd)
    src_xy = wall_points(src_pcd)
    ref_c = ref_xy.mean(0)
    src_c = src_xy.mean(0)
    span = max(np.ptp(ref_xy, 0).max(), np.ptp(src_xy, 0).max()) * 2.0
    n = int(2 ** np.ceil(np.log2(span / GRID_RES)))
    shape = (n, n)
    origin = -0.5 * n * GRID_RES * np.ones(2)

    ref_img = occupancy(ref_xy - ref_c, origin, shape)
    F_ref = np.fft.rfft2(ref_img - ref_img.mean())

    scores = []
    for deg in np.arange(0, 360, YAW_STEP_DEG):
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        xy = (src_xy - src_c) @ np.array([[c, s], [-s, c]])
        img = occupancy(xy, origin, shape)
        corr = np.fft.irfft2(F_ref * np.conj(np.fft.rfft2(img - img.mean())), s=shape)
        k = np.unravel_index(np.argmax(corr), shape)
        shift = np.array([k[0] if k[0] < n // 2 else k[0] - n,
                          k[1] if k[1] < n // 2 else k[1] - n]) * GRID_RES
        scores.append((corr[k], th, shift))

    candidates = []
    for score, th, shift in sorted(scores, key=lambda x: -x[0]):
        if all(abs((th - t + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(10) for t, _ in candidates):
            c, s = np.cos(th), np.sin(th)
            t = ref_c + shift - np.array([[c, -s], [s, c]]) @ src_c
            candidates.append((th, yaw_transform(th, *t)))
        if len(candidates) == n_candidates:
            break
    return [T for _, T in candidates]


def project_to_yaw(T):
    """Keep only the yaw rotation and x/y translation of a rigid transform."""
    th = np.arctan2(T[1, 0], T[0, 0])
    return yaw_transform(th, T[0, 3], T[1, 3])


def refine_icp(src_pcd, ref_pcd, T):
    """Geometric ICP for the coarse scales, colored ICP for the finer ones."""
    reg = o3d.pipelines.registration
    for i, dist in enumerate(ICP_DIST):
        if i == 0 or not (src_pcd.has_colors() and ref_pcd.has_colors()):
            res = reg.registration_icp(
                src_pcd, ref_pcd, dist, T,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=60))
        else:
            res = reg.registration_colored_icp(
                src_pcd, ref_pcd, dist, T,
                reg.TransformationEstimationForColoredICP(lambda_geometric=0.9),
                reg.ICPConvergenceCriteria(max_iteration=60))
        T = project_to_yaw(res.transformation)
    return T


def match_score(src_pcd, ref_tree, ref_colors, T):
    """(geometric fitness, color-consistent fitness) of src under T."""
    pts = np.asarray(src_pcd.points) @ T[:3, :3].T + T[:3, 3]
    has_color = src_pcd.has_colors() and len(ref_colors)
    cols = np.asarray(src_pcd.colors)
    geo = col = 0
    for i, p in enumerate(pts):
        k, idx, d2 = ref_tree.search_knn_vector_3d(p, 1)
        if k and d2[0] < ICP_DIST[-1] ** 2:
            geo += 1
            if has_color and np.abs(cols[i] - ref_colors[idx[0]]).mean() < COLOR_TOL:
                col += 1
    n = len(pts)
    return geo / n, (col if has_color else geo) / n


def flip_transform(pcd):
    """Turn a leveled scan upside down (in case floor and ceiling were swapped)."""
    z = np.asarray(pcd.points)[:, 2]
    zz = z[z > np.percentile(z, 99) - 0.3]
    hist, edges = np.histogram(zz, bins=60)
    k = np.argmax(hist)
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[2, 3] = 0.5 * (edges[k] + edges[k + 1])
    return T


def align_2d(src_pcd, ref_pcd):
    """Best yaw + x/y transform (optionally after an upside-down flip) src -> ref.

    Candidates are ranked by the fraction of points that land on the reference
    AND agree with it in color, so a flipped room (walls fit, but floor and
    ceiling colors swapped) loses to the correct one.
    """
    ref_tree = o3d.geometry.KDTreeFlann(ref_pcd)
    ref_colors = np.asarray(ref_pcd.colors)
    best = None
    for flipped in (False, True):
        F = flip_transform(src_pcd) if flipped else np.eye(4)
        src = o3d.geometry.PointCloud(src_pcd).transform(F)
        for T0 in coarse_align_2d(src, ref_pcd):
            T = refine_icp(src, ref_pcd, T0)
            geo, col = match_score(src, ref_tree, ref_colors, T)
            yaw = np.rad2deg(np.arctan2(T[1, 0], T[0, 0])) % 360
            print(f"    {'flipped ' if flipped else 'upright '} yaw {yaw:6.1f} deg, "
                  f"shift ({T[0, 3]:5.2f}, {T[1, 3]:5.2f}) m: "
                  f"fitness {geo:.3f}, color-consistent {col:.3f}")
            if best is None or col > best[1]:
                best = (T @ F, col, flipped)
    if best[2]:
        print("    -> scan was upside down, flipped it")
    return best[0]


def to_pcd(mesh, n=150000):
    pcd = mesh.sample_points_uniformly(n)
    pcd = pcd.voxel_down_sample(0.02)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


# ----------------------------------------------------------------------------
# Union
# ----------------------------------------------------------------------------
def saturation(colors):
    mx = colors.max(axis=1)
    return (mx - colors.min(axis=1)) / np.maximum(mx, 1e-6)


def color_new_vertices(verts, existing_pcd):
    """Mask of saturated vertices whose color matches no existing point nearby."""
    cols = np.asarray(verts.colors)
    ex_cols = np.asarray(existing_pcd.colors)
    tree = o3d.geometry.KDTreeFlann(existing_pcd)
    pts = np.asarray(verts.points)
    new = np.zeros(len(pts), bool)
    for i in np.where(saturation(cols) > COLOR_SAT_MIN)[0]:
        k, idx, _ = tree.search_radius_vector_3d(pts[i], COLOR_SEARCH)
        if k == 0 or np.abs(ex_cols[np.asarray(idx)] - cols[i]).mean(axis=1).min() > COLOR_DIFF:
            new[i] = True
    if not new.any():
        return new

    # drop isolated speckles
    sel = np.where(new)[0]
    labels = np.asarray(verts.select_by_index(sel).cluster_dbscan(COLOR_GROW * 2, 1))
    for l in set(labels):
        members = sel[labels == l]
        if l == -1 or len(members) < COLOR_MIN_POINTS:
            new[members] = False
    return new


def new_part(mesh, existing_pcd):
    """Sub-mesh of `mesh` holding what is not already covered by `existing_pcd`.

    Geometry: vertices farther than MERGE_DIST from the existing geometry are
    clustered into objects; small clusters (registration noise) are dropped, and
    each kept object is grown by GROW_DIST so its edges and base (which sit close
    to the existing floor/walls) are included too.
    Color (COLOR_MERGE): saturated vertices whose color matches nothing nearby in
    the existing geometry are kept as well, regardless of size.

    Returns (part, color_pts, color_cols): the color-feature vertices are also
    returned so the existing surface underneath can be repainted.
    """
    verts = o3d.geometry.PointCloud(mesh.vertices)
    verts.colors = mesh.vertex_colors
    dist = np.asarray(verts.compute_point_cloud_distance(existing_pcd))
    seeds = []

    far_idx = np.where(dist > MERGE_DIST)[0]
    if len(far_idx):
        far = verts.select_by_index(far_idx)
        labels = np.asarray(far.cluster_dbscan(CLUSTER_EPS, 5))
        far_pts = np.asarray(far.points)
        for l in set(labels) - {-1}:
            q = far_pts[labels == l]
            if len(q) >= MIN_NEW_POINTS and np.ptp(q, 0).max() >= MIN_NEW_SIZE:
                seeds.append(q)
        print(f"    {len(seeds)} new objects/regions by geometry, largest:")
        for q in sorted(seeds, key=len, reverse=True)[:8]:
            c, e = q.mean(0), np.ptp(q, 0)
            print(f"      {len(q):5d} verts at ({c[0]:5.2f}, {c[1]:5.2f}, {c[2]:4.2f}), "
                  f"size {e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f} m")

    near = np.zeros(len(dist), bool)
    if seeds:
        seed_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(seeds)))
        near = np.asarray(verts.compute_point_cloud_distance(seed_pcd)) < GROW_DIST

    color_pts = np.zeros((0, 3))
    color_cols = np.zeros((0, 3))
    if COLOR_MERGE and mesh.has_vertex_colors() and existing_pcd.has_colors():
        cnew = color_new_vertices(verts, existing_pcd)
        if cnew.any():
            color_pts = np.asarray(verts.points)[cnew]
            color_cols = np.asarray(verts.colors)[cnew]
            cpcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(color_pts))
            labels = np.asarray(cpcd.cluster_dbscan(COLOR_GROW * 2, 1))
            print(f"    {len(set(labels))} new color features ({cnew.sum()} verts)")
            near |= np.asarray(verts.compute_point_cloud_distance(cpcd)) < COLOR_GROW

    tris = np.asarray(mesh.triangles)
    keep = near[tris].sum(axis=1) >= 2
    part = o3d.geometry.TriangleMesh(mesh)
    part.remove_triangles_by_mask(~keep)
    part.remove_unreferenced_vertices()
    return part, color_pts, color_cols


def repaint(mesh, pts, cols, radius):
    """Give mesh vertices within `radius` of `pts` the color of the nearest one."""
    if len(pts) == 0:
        return 0
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    tree = o3d.geometry.KDTreeFlann(src)
    mcols = np.asarray(mesh.vertex_colors)
    n = 0
    for i, v in enumerate(np.asarray(mesh.vertices)):
        k, idx, d2 = tree.search_knn_vector_3d(v, 1)
        if k and d2[0] < radius ** 2:
            mcols[i] = cols[idx[0]]
            n += 1
    mesh.vertex_colors = o3d.utility.Vector3dVector(mcols)
    return n


def existing_points(mesh):
    """Dense colored point cloud of a mesh (surface samples + its vertices)."""
    pcd = mesh.sample_points_uniformly(max(200000, 2 * len(mesh.triangles)))
    pcd.points.extend(mesh.vertices)
    if mesh.has_vertex_colors():
        pcd.colors.extend(mesh.vertex_colors)
    return pcd


def markers_path(scene_path):
    base = os.path.splitext(scene_path)[0]
    if base.endswith("_marked"):
        base = base[:-len("_marked")]
    return base + "_markers.csv"


def load_markers(path):
    """(indices, Nx3 points) from a markers CSV, or None if it does not exist."""
    if not os.path.isfile(path):
        return None
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return a[:, 0].astype(int), a[:, 1:4]


def save_markers(path, ids, pts, scenes=None):
    rows = np.column_stack([ids, pts]).astype(object)
    header, fmt = "index,x,y,z", ["%d", "%.6f", "%.6f", "%.6f"]
    if scenes is not None:
        rows = np.column_stack([rows, scenes])
        header += ",scene"; fmt.append("%s")
    np.savetxt(path, rows, delimiter=",", fmt=fmt, header=header, comments="")


def main():
    o3d.utility.random.seed(0)
    np.random.seed(0)
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [p if os.path.isabs(p) else os.path.join(here, p) for p in SCENE_PATHS]
    assert len(paths) >= 2, "need at least two scenes to merge"
    out_dir = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.join(here, OUTPUT_DIR)
    if not os.path.isdir(out_dir):
        print(f"[warn] {out_dir} does not exist, writing next to the input scenes")
        out_dir = os.path.dirname(paths[0])
    out_path = os.path.join(out_dir, OUTPUT_NAME)

    meshes = []
    transforms = []  # input frame -> merged frame, per scene
    for p in paths:
        print(f"[load] {p}")
        m = o3d.io.read_triangle_mesh(p)
        if not m.has_triangles():
            raise ValueError(f"{p} has no faces; this script expects meshes")
        print("  leveling")
        L = level_transform(m)
        m.transform(L)
        meshes.append(m)
        transforms.append(L)

    ref_pcd = to_pcd(meshes[0])
    for i in range(1, len(meshes)):
        print(f"[align] {os.path.basename(paths[i])} -> {os.path.basename(paths[0])}")
        src_pcd = to_pcd(meshes[i])
        T = align_2d(src_pcd, ref_pcd)
        meshes[i].transform(T)
        transforms[i] = T @ transforms[i]

    if SAVE_ALIGNED:
        for p, m in zip(paths, meshes):
            q = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + "_aligned.ply")
            o3d.io.write_triangle_mesh(q, m)
            print(f"[save] {q}")

    if MERGE_MARKERS:
        all_ids, all_pts, all_scenes = [], [], []
        for p, T in zip(paths, transforms):
            mk = load_markers(markers_path(p))
            if mk is None:
                print(f"[markers] none for {os.path.basename(p)}")
                continue
            ids, pts = mk
            pts = pts @ T[:3, :3].T + T[:3, 3]
            name = os.path.splitext(os.path.basename(p))[0]
            print(f"[markers] {len(ids)} from {os.path.basename(markers_path(p))}")
            if SAVE_ALIGNED:
                q = os.path.join(out_dir, name + "_markers_aligned.csv")
                save_markers(q, ids, pts)
                print(f"[save] {q}")
            all_ids.append(ids); all_pts.append(pts); all_scenes += [name] * len(ids)
        if all_ids:
            q = os.path.splitext(out_path)[0] + "_markers.csv"
            save_markers(q, np.concatenate(all_ids), np.vstack(all_pts), all_scenes)
            print(f"[save] {q}")

    merged = o3d.geometry.TriangleMesh(meshes[0])
    added = o3d.geometry.TriangleMesh()
    for i in range(1, len(meshes)):
        part, color_pts, color_cols = new_part(meshes[i], existing_points(merged))
        n_rep = repaint(merged, color_pts, color_cols, COLOR_GROW)
        print(f"[union] {os.path.basename(paths[i])}: adding {len(part.triangles)} "
              f"of {len(meshes[i].triangles)} triangles, repainted {n_rep} existing verts")
        merged += part
        added += part

    merged.remove_duplicated_vertices()
    merged.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_path, merged)
    print(f"[save] {out_path}  ({len(merged.vertices)} verts, {len(merged.triangles)} tris)")
    if SAVE_ADDED and len(added.triangles):
        q = os.path.splitext(out_path)[0] + "_added.ply"
        o3d.io.write_triangle_mesh(q, added)
        print(f"[save] {q}")
    if SAVE_HIGHLIGHT and len(added.triangles):
        base = o3d.geometry.TriangleMesh(meshes[0])
        red = o3d.geometry.TriangleMesh(added)
        red.paint_uniform_color([1.0, 0.0, 0.0])
        q = os.path.splitext(out_path)[0] + "_highlight.ply"
        o3d.io.write_triangle_mesh(q, base + red)
        print(f"[save] {q}")


if __name__ == "__main__":
    main()

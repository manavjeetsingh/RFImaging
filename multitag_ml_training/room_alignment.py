"""
Align the per-experiment coordinate systems of the multitag dataset into one
shared room frame.

Why this is needed
------------------
Every experiment (1 … 13) was recorded in the same room, but each comes with its
own room scan (multitag/<exp>/*.ply), and each scan has its own arbitrary origin
and heading. cluster_features.py then levels every experiment separately
(level_rot: the plane through its tags becomes z = 0 normal), so all the
coordinates in the cluster CSV — tag_xyz, centroid, pose — live in a per-
experiment *levelled* frame. Centroids of two experiments cannot be plotted
together until those frames are related.

How the frames are aligned
--------------------------
The levelled frames are all z-up already, so the transform between any two of
them is a yaw rotation about z + an xy translation (+ a z offset). It is found
from the room scans themselves, which all show the same walls:

  1. Each scan is loaded exactly as cluster_features.py does (PLY_PATHS,
     rotated by that experiment's level_rot) and surface-sampled. Points whose
     normal is close to horizontal (|n_z| < 0.3) are the walls / beams / large
     furniture — the part of the room seen from above.
  2. Coarse: those points are rasterised (5 cm) and, for every yaw in 1° steps,
     cross-correlated (FFT) against the reference scan's raster; the
     correlation peak gives the translation for that yaw.
  3. Fine: 2-D trimmed ICP (point-to-point, 15 cm gate) from the best few
     coarse peaks — the room is close to a rectangle, so the 180°-flipped
     solution is always a strong runner-up; the ceiling beams and the corner
     notch break the tie. Candidates are scored by the truncated mean distance
     (min(d, 10 cm)) of the wall points to the reference walls — a plain median
     is not enough, a flipped rectangle still matches most wall points.
  4. z offset: each scan's floor height (lowest dense layer of upward-facing
     points) is moved to z = 0.

The common *room frame* is the reference experiment's frame, turned so the
dominant wall direction is the x axis and shifted so the room's wall bounding
box starts at (0, 0) and the floor is z = 0. Every experiment e gets

    p_room = R_z(yaw_e) @ p_levelled + t_e

(levelled = the frame of the cluster CSV). Quality per experiment is reported as
the median distance between its wall points and the reference walls after
alignment (≈ 0.5–1.5 cm for most scans; exp 6, scanned with an ~12° tilt, ≈ 3 cm).

Usage
-----
    from room_alignment import load_alignment
    al = load_alignment(dataset_csv)               # computed once, then cached
    xyz_room   = al.to_room(exp, centroids)        # (n, 3)
    poses_room = al.poses_to_room(exp, poses)      # (n, 4) quaternions [x,y,z,w]
    hits_room  = wall_hit_points(xyz_room, poses_room, dists_m)   # (n, 4, 3)
    al.plot_check()                                # all scans overlaid in the room frame

The result is cached next to the dataset as <csv stem>.room_alignment.json
(keyed by the scan files' size + mtime; pass recompute=True to redo it).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.fft import irfft2, rfft2
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
MULTITAG_DIR = os.environ.get("MULTITAG_DIR", os.path.abspath(os.path.join(_HERE, "..", "multitag")))

WALL_NZ_MAX   = 0.3      # |n_z| below this → wall-like (vertical) surface
FLOOR_NZ_MIN  = 0.9      # n_z above this → floor-like (upward) surface
RES           = 0.05     # [m] raster cell for the coarse search
CANVAS        = 512      # raster size (CANVAS·RES = 25.6 m, rooms are ~10 m)
N_SAMPLE      = 200_000  # surface samples per scan
N_REG         = 60_000   # wall points used for registration
ICP_GATE      = 0.15     # [m] correspondences further apart are ignored
N_CANDIDATES  = 4        # coarse yaw peaks refined with ICP
_CACHE_VERSION = 2


# ── Scan loading ──────────────────────────────────────────────────────────────

def _ply_paths() -> dict[str, str]:
    """PLY_PATHS from cluster_features.py — the very scans the wall targets were cast in."""
    if MULTITAG_DIR not in sys.path:
        sys.path.insert(0, MULTITAG_DIR)
    from cluster_features import PLY_PATHS
    return {k: os.path.join(MULTITAG_DIR, v) for k, v in PLY_PATHS.items()}


def read_level_rotations(dataset_csv: str, chunksize: int = 200_000) -> dict[int, np.ndarray]:
    """level_rot (levelled = R @ scan frame) of every experiment in a cluster_features CSV."""
    out: dict[int, np.ndarray] = {}
    for ch in pd.read_csv(dataset_csv, usecols=["experiment", "level_rot"], chunksize=chunksize):
        for e, r in ch.drop_duplicates("experiment").itertuples(index=False):
            out.setdefault(int(e), np.asarray(json.loads(r), dtype=np.float64))
    return out


def load_scan_points(ply_path: str, R: np.ndarray, n: int = N_SAMPLE, seed: int = 0) -> np.ndarray:
    """(n, 6) [xyz, normal] surface samples of a scan, in the levelled frame (R applied)."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"no triangles loaded from {ply_path}")
    mesh.rotate(R, center=(0.0, 0.0, 0.0))
    o3d.utility.random.seed(seed)
    pc = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    return np.c_[np.asarray(pc.points), np.asarray(pc.normals)]


def wall_points(pts: np.ndarray) -> np.ndarray:
    return pts[np.abs(pts[:, 5]) < WALL_NZ_MAX, :3]


def floor_height(pts: np.ndarray) -> float:
    """z of the lowest dense layer of upward-facing points (2 cm bins)."""
    z = pts[pts[:, 5] > FLOOR_NZ_MIN, 2]
    if len(z) < 100:
        return float(np.percentile(pts[:, 2], 1))
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    # The floor is the lowest bin holding a substantial share of the upward points
    # (tables / beds are dense too, but higher).
    k = np.flatnonzero(hist >= 0.3 * hist.max())[0]
    k = k + int(np.argmax(hist[k:k + 5]))
    return float(0.5 * (edges[k] + edges[k + 1]))


# ── 2-D registration ──────────────────────────────────────────────────────────

def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _raster(xy: np.ndarray) -> np.ndarray:
    ij = np.floor(xy / RES).astype(int) + CANVAS // 2
    ok = ((ij >= 0) & (ij < CANVAS)).all(axis=1)
    im = np.zeros((CANVAS, CANVAS))
    np.add.at(im, (ij[ok, 0], ij[ok, 1]), 1.0)
    return gaussian_filter(np.minimum(im, 3.0), 2.0)


def coarse_search(src: np.ndarray, dst: np.ndarray, step_deg: float = 1.0) -> list[tuple]:
    """
    Exhaustive yaw search: for every yaw, the translation is the FFT
    cross-correlation peak of the two rasters. Returns [(score, theta, t)] for
    the best N_CANDIDATES yaws that are ≥ 10° apart, best first, with
    dst ≈ R(theta) @ src + t.
    """
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    FD = np.conj(rfft2(_raster(dst - dc)))
    yaws = np.radians(np.arange(0.0, 360.0, step_deg))
    peaks = []
    for th in yaws:
        cc = irfft2(rfft2(_raster((src - sc) @ _rot2(th).T)) * FD, s=(CANVAS, CANVAS))
        k = np.unravel_index(np.argmax(cc), cc.shape)
        shift = ((np.array(k) + CANVAS // 2) % CANVAS - CANVAS // 2) * RES
        peaks.append((float(cc[k]), th, dc - shift - _rot2(th) @ sc))
    order = np.argsort([-p[0] for p in peaks])
    out = []
    for i in order:
        if all(abs((peaks[i][1] - o[1] + np.pi) % (2 * np.pi) - np.pi) > np.radians(10) for o in out):
            out.append(peaks[i])
        if len(out) == N_CANDIDATES:
            break
    return out


def icp_2d(src: np.ndarray, dst_tree: cKDTree, theta: float, t: np.ndarray,
           iters: int = 60) -> tuple[float, np.ndarray, float, float]:
    """
    Trimmed point-to-point ICP. Returns theta, t, median residual [m], inlier
    (<10 cm) fraction, and the fit score: mean of min(d, 10 cm) [m], lower is better.
    """
    for _ in range(iters):
        x = src @ _rot2(theta).T + t
        d, j = dst_tree.query(x, distance_upper_bound=ICP_GATE)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            break
        a, b = src[ok], dst_tree.data[j[ok]]
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1
            Rm = Vt.T @ U.T
        theta_new, t_new = float(np.arctan2(Rm[1, 0], Rm[0, 0])), mb - Rm @ ma
        converged = abs(theta_new - theta) < 1e-6 and np.allclose(t_new, t, atol=1e-5)
        theta, t = theta_new, t_new
        if converged:
            break
    d, _ = dst_tree.query(src @ _rot2(theta).T + t)
    return theta, t, float(np.median(d)), float(np.mean(d < 0.10)), float(np.minimum(d, 0.10).mean())


def dominant_wall_yaw(xy_normals: np.ndarray) -> float:
    """Wall direction modulo 90° from horizontal normals (circular mean of 4θ)."""
    ang = np.arctan2(xy_normals[:, 1], xy_normals[:, 0])
    return float(np.angle(np.exp(4j * ang).mean()) / 4)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class RoomAlignment:
    """Per-experiment yaw [rad] + translation [m]: p_room = R_z(yaw) @ p_levelled + t."""
    yaw: dict[int, float]
    t: dict[int, np.ndarray]
    ref: int
    quality: pd.DataFrame                       # per experiment: median residual, inliers, …
    walls_room: dict[int, np.ndarray] = field(default_factory=dict, repr=False)  # (m, 3) per exp

    @property
    def experiments(self) -> list[int]:
        return sorted(self.yaw)

    def rotation(self, exp: int) -> np.ndarray:
        return Rotation.from_euler("z", self.yaw[int(exp)]).as_matrix()

    def to_room(self, exp: int, xyz: np.ndarray) -> np.ndarray:
        """(…, 3) levelled-frame points of experiment `exp` → room frame."""
        xyz = np.asarray(xyz, dtype=np.float64)
        return xyz @ self.rotation(exp).T + self.t[int(exp)]

    def vec_to_room(self, exp: int, v: np.ndarray) -> np.ndarray:
        """Directions (no translation)."""
        return np.asarray(v, dtype=np.float64) @ self.rotation(exp).T

    def poses_to_room(self, exp: int, quat_xyzw: np.ndarray) -> np.ndarray:
        """(n, 4) quaternions [x,y,z,w] → room frame (the yaw is pre-multiplied)."""
        q = Rotation.from_euler("z", self.yaw[int(exp)]) * Rotation.from_quat(np.asarray(quat_xyzw))
        return q.as_quat()

    def to_room_many(self, exps: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Row-wise: xyz[i] belongs to experiment exps[i]."""
        exps, out = np.asarray(exps), np.empty(np.shape(xyz), dtype=np.float64)
        for e in np.unique(exps):
            s = exps == e
            out[s] = self.to_room(e, np.asarray(xyz)[s])
        return out

    def poses_to_room_many(self, exps: np.ndarray, quats: np.ndarray) -> np.ndarray:
        exps, out = np.asarray(exps), np.empty(np.shape(quats), dtype=np.float64)
        for e in np.unique(exps):
            s = exps == e
            out[s] = self.poses_to_room(e, np.asarray(quats)[s])
        return out

    def room_outline(self, n: int = 40_000, seed: int = 0) -> np.ndarray:
        """(n, 3) wall points of the reference scan, room frame — a backdrop for plots."""
        w = self.walls_room[self.ref]
        idx = np.random.default_rng(seed).choice(len(w), min(n, len(w)), replace=False)
        return w[idx]

    def plot_check(self, exps=None, ncols: int = 5, colors=None):
        """One panel per experiment: its walls (colour) on the reference walls (grey)."""
        import matplotlib.pyplot as plt
        exps = self.experiments if exps is None else list(exps)
        ref = self.room_outline()
        nrows = int(np.ceil(len(exps) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.8 * nrows), squeeze=False)
        for ax, e in zip(axes.ravel(), exps):
            w = self.walls_room[e]
            w = w[np.random.default_rng(1).choice(len(w), min(20_000, len(w)), replace=False)]
            ax.scatter(ref[:, 0], ref[:, 1], s=0.2, c="0.6", lw=0)
            ax.scatter(w[:, 0], w[:, 1], s=0.2, lw=0,
                       c=[colors[e]] if colors is not None else "tab:red")
            q = self.quality.loc[e]
            ax.set_title(f"exp {e}  yaw {np.degrees(self.yaw[e]):+.1f}°\n"
                         f"median residual {q['median_resid_cm']:.1f} cm", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.ravel()[len(exps):]:
            ax.axis("off")
        fig.suptitle(f"Scan alignment check — every experiment on exp {self.ref} (grey), room frame")
        fig.tight_layout()
        return fig

    # ── (de)serialisation ────────────────────────────────────────────────────
    def to_json(self) -> dict:
        return dict(
            version=_CACHE_VERSION, ref=self.ref,
            yaw={str(k): v for k, v in self.yaw.items()},
            t={str(k): v.tolist() for k, v in self.t.items()},
            quality=self.quality.reset_index().to_dict(orient="list"),
        )

    @classmethod
    def from_json(cls, d: dict, walls_room: dict | None = None) -> "RoomAlignment":
        q = pd.DataFrame(d["quality"]).set_index("experiment")
        return cls(yaw={int(k): float(v) for k, v in d["yaw"].items()},
                   t={int(k): np.asarray(v) for k, v in d["t"].items()},
                   ref=int(d["ref"]), quality=q, walls_room=walls_room or {})


# ── Main entry point ──────────────────────────────────────────────────────────

def _sub(a: np.ndarray, e: int) -> np.ndarray:
    """Deterministic per-experiment subsample of N_REG rows (independent of run order)."""
    rng = np.random.default_rng(int(e))
    return a[rng.choice(len(a), min(N_REG, len(a)), replace=False)]


def _load_job(e: int, ply_path: str, R: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Worker: (exp, (m, 6) wall points [xyz, normal], floor z)."""
    p = load_scan_points(ply_path, R)
    return e, p[np.abs(p[:, 5]) < WALL_NZ_MAX], floor_height(p)


def _register_job(e: int, src: np.ndarray, dst: np.ndarray) -> tuple[int, list]:
    """Worker: ICP-refined coarse candidates for one experiment, best (lowest score) first."""
    tree = cKDTree(dst)
    fits = sorted((icp_2d(src, tree, th, t) for _, th, t in coarse_search(src, dst)),
                  key=lambda f: f[4])
    return e, fits


def _pool(n_workers: int):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    # spawn: open3d / BLAS are not fork-safe; the workers only need this module.
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"))


def compute_alignment(level_rot: dict[int, np.ndarray], ref: int | None = None,
                      verbose: bool = True, n_workers: int | None = None) -> RoomAlignment:
    """
    Scan loading and per-experiment registration run in parallel worker
    processes (n_workers, default: one per experiment up to the CPU count;
    n_workers=1 runs everything in-process).
    """
    from concurrent.futures import as_completed
    paths = _ply_paths()
    exps = sorted(level_rot)
    n_workers = n_workers or min(len(exps), os.cpu_count() or 1)

    def run(jobs):
        """jobs: [(fn, *args)] → results as they finish (in-process if n_workers == 1)."""
        if n_workers == 1:
            yield from (fn(*a) for fn, *a in jobs)
            return
        with _pool(n_workers) as ex:
            yield from (f.result() for f in as_completed([ex.submit(fn, *a) for fn, *a in jobs]))

    walls, floors = {}, {}
    for e, w, fz in run([(_load_job, e, paths[str(e)], level_rot[e]) for e in exps]):
        walls[e], floors[e] = w, fz
        if verbose:
            print(f"[align] exp {e:>2}: {len(w):>6} wall points, floor z = {fz:+.3f} m", flush=True)

    # Reference: the scan whose walls overlap best with all others is safest, but
    # any complete scan works; default to the one with the most wall points.
    ref = int(ref) if ref is not None else max(exps, key=lambda e: len(walls[e]))
    dst = _sub(walls[ref][:, :2], ref)

    yaw, t2, rows = {ref: 0.0}, {ref: np.zeros(2)}, {}
    rows[ref] = dict(experiment=ref, median_resid_cm=0.0, inlier_10cm=1.0, score_cm=0.0,
                     runner_up_yaw_deg=np.nan, runner_up_score_cm=np.nan, floor_z=floors[ref])
    jobs = [(_register_job, e, _sub(walls[e][:, :2], e), dst) for e in exps if e != ref]
    for e, fits in run(jobs):
        th, t, med, inl, score = fits[0]
        yaw[e], t2[e] = th, t
        # Runner-up: the best candidate that converged to a genuinely different yaw
        # (several coarse peaks usually converge to the same solution).
        other = [f for f in fits[1:] if abs((f[0] - th + np.pi) % (2 * np.pi) - np.pi) > np.radians(5)]
        rows[e] = dict(experiment=e, median_resid_cm=100 * med, inlier_10cm=inl, score_cm=100 * score,
                       runner_up_yaw_deg=np.degrees(other[0][0] - th) if other else np.nan,
                       runner_up_score_cm=100 * other[0][4] if other else np.nan,
                       floor_z=floors[e])
        if verbose:
            r = rows[e]
            print(f"[align] exp {e:>2} → {ref}: yaw {np.degrees(th):+7.2f}°  "
                  f"t ({t[0]:+.2f}, {t[1]:+.2f}) m  median {100 * med:.1f} cm  "
                  f"inliers {inl:.0%}  score {100 * score:.2f} cm  (runner-up at "
                  f"{r['runner_up_yaw_deg']:+.0f}°: score {r['runner_up_score_cm']:.2f} cm)", flush=True)
    rows = [rows[e] for e in exps]

    # Room frame: reference turned so its walls are axis-aligned, corner at the
    # origin, floor at z = 0.
    nref = walls[ref][:, 3:5]
    phi = -dominant_wall_yaw(nref)
    Rphi = _rot2(phi)
    ref_xy = walls[ref][:, :2] @ Rphi.T
    corner = np.percentile(ref_xy, 0.5, axis=0)

    yaw_room, t_room, walls_room = {}, {}, {}
    for e in exps:
        yaw_room[e] = float((yaw[e] + phi + np.pi) % (2 * np.pi) - np.pi)
        txy = Rphi @ t2[e] - corner
        t_room[e] = np.array([txy[0], txy[1], -floors[e]])
    al = RoomAlignment(yaw=yaw_room, t=t_room, ref=ref,
                       quality=pd.DataFrame(rows).set_index("experiment"))
    for e in exps:
        walls_room[e] = al.to_room(e, _sub(walls[e][:, :3], e)).astype(np.float32)
    al.walls_room = walls_room
    return al


def _stamp(level_rot: dict) -> list:
    paths = _ply_paths()
    out = []
    for e in sorted(level_rot):
        st = os.stat(paths[str(e)])
        out.append([int(e), os.path.basename(paths[str(e)]), st.st_size, int(st.st_mtime)])
    return out


def load_alignment(dataset_csv: str, ref: int | None = None, recompute: bool = False,
                   cache_path: str | None = None, verbose: bool = True,
                   n_workers: int | None = None) -> RoomAlignment:
    """
    Alignment for every experiment in `dataset_csv`, from cache when the scans
    are unchanged. The cache also stores the aligned wall points (an .npz next
    to the json) so the plots don't need to reload the scans.
    """
    base = os.path.splitext(dataset_csv)[0]
    cache_path = cache_path or base + ".room_alignment.json"
    walls_path = cache_path.replace(".json", "_walls.npz")

    if not recompute and os.path.isfile(cache_path) and os.path.isfile(walls_path):
        with open(cache_path) as f:
            d = json.load(f)
        z = np.load(walls_path)
        stamp_ok = d.get("version") == _CACHE_VERSION and (ref is None or int(d["ref"]) == int(ref))
        level_rot = {int(k): np.asarray(v) for k, v in d.get("level_rot", {}).items()}
        if stamp_ok and level_rot and d.get("stamp") == _stamp(level_rot):
            if verbose:
                print(f"[align] cached: {cache_path}")
            return RoomAlignment.from_json(d, {int(k): z[k] for k in z.files})

    if verbose:
        print(f"[align] reading level_rot from {dataset_csv}")
    level_rot = read_level_rotations(dataset_csv)
    al = compute_alignment(level_rot, ref=ref, verbose=verbose, n_workers=n_workers)
    d = al.to_json() | {"dataset": os.path.abspath(dataset_csv),
                        "level_rot": {str(k): v.tolist() for k, v in level_rot.items()},
                        "stamp": _stamp(level_rot)}
    with open(cache_path, "w") as f:
        json.dump(d, f, indent=1)
    np.savez_compressed(walls_path, **{str(k): v for k, v in al.walls_room.items()})
    if verbose:
        print(f"[align] cached → {cache_path}")
    return al


# ── Geometry helpers ──────────────────────────────────────────────────────────

def wall_ray_dirs(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    (n, 4, 3) unit N, S, E, W ray directions of each sub-cluster, exactly as
    cluster_features.wall_raycast casts them: local +X = horizontal part of the
    pose's +X (centroid → tag 0), +Y = +X turned +90° about z,
    N = -Y, S = +Y, E = -X, W = +X. Works in any z-up frame.
    """
    x = Rotation.from_quat(np.asarray(quat_xyzw)).apply([1.0, 0.0, 0.0])
    x[:, 2] = 0.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], axis=1)
    return np.stack([-y, y, -x, x], axis=1)


def wall_hit_points(centroids: np.ndarray, quat_xyzw: np.ndarray, dists_m: np.ndarray) -> np.ndarray:
    """(n, 4, 3) wall hit points: centroid + distance · ray (NaN distances stay NaN)."""
    return np.asarray(centroids)[:, None, :] + np.asarray(dists_m)[:, :, None] * wall_ray_dirs(quat_xyzw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute / refresh the room alignment cache")
    ap.add_argument("dataset", help="cluster_features_K*_P*.csv")
    ap.add_argument("--ref", type=int, default=None, help="reference experiment")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (1 = serial)")
    ap.add_argument("--plot", default=None, help="save the alignment check figure here")
    a = ap.parse_args()
    al = load_alignment(a.dataset, ref=a.ref, recompute=a.recompute, n_workers=a.workers)
    print(al.quality.round(3).to_string())
    if a.plot:
        al.plot_check().savefig(a.plot, dpi=80)
        print(f"figure → {a.plot}")

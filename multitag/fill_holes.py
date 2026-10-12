"""
Close open patches (holes) in scanned 3D room models (.ply).

A hole is a closed loop of boundary edges. Each hole is filled in three
steps (after Liepa, "Filling Holes in Meshes", 2003):

1. triangulate: project the loop onto its best-fit plane and ear-clip it;
   if the projection is not a simple polygon, use a fan around the centroid;
2. refine: insert vertices at triangle centroids and flip edges (Delaunay)
   until the patch density matches the mesh around the hole, so large holes
   get a proper mesh instead of long sliver triangles;
3. fair: solve a membrane (Laplace) equation with the hole boundary fixed,
   so non-planar holes (e.g. wrapping around a wall/floor/ceiling corner)
   get a smooth curved surface rather than a flat cap. Vertex colours are
   interpolated the same way.

Usage:
    python fill_holes.py 10/2214_10.ply                  # -> 10/2214_10_filled.ply
    python fill_holes.py 10/2214_10.ply -o out.ply --walls-only
    python fill_holes.py 10/*.ply 11/*.ply --max-perimeter 15
"""

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import trimesh


def boundary_loops(faces, n_vertices):
    """Return boundary loops as lists of vertex indices, following the
    direction of the existing (unmatched) half-edges."""
    he = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = he[:, 0].astype(np.int64) * n_vertices + he[:, 1]
    rkey = he[:, 1].astype(np.int64) * n_vertices + he[:, 0]
    bd = he[~np.isin(key, rkey)]

    out = defaultdict(list)
    for a, b in bd:
        out[a].append(b)

    used = set()
    loops = []
    for a, b in bd:
        if (a, b) in used:
            continue
        used.add((a, b))
        loop, cur = [a], b
        while cur != a:
            loop.append(cur)
            nxt = [x for x in out[cur] if (cur, x) not in used]
            if not nxt:
                break
            used.add((cur, nxt[0]))
            cur = nxt[0]
        if cur == a:
            loops.extend(l for l in _split_at_repeats(loop) if len(l) >= 3)
    return loops


def _split_at_repeats(loop):
    """Split a loop that passes through the same vertex more than once
    (pinch / bowtie vertices) into simple loops."""
    out, stack, pos = [], [], {}
    for v in loop:
        if v in pos:
            i = pos[v]
            out.append(np.array(stack[i:]))
            for u in stack[i + 1:]:
                del pos[u]
            del stack[i + 1:]
        else:
            pos[v] = len(stack)
            stack.append(v)
    out.append(np.array(stack))
    return out


def plane_fit(P):
    c = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt, s  # vt[2] is the normal


def _segments_intersect(P):
    """True if the closed 2D polygon P self-intersects (O(n^2), vectorised)."""
    A = P
    B = np.roll(P, -1, axis=0)
    n = len(P)
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if len(j) == 0:
            continue
        p, r = A[i], B[i] - A[i]
        q, s = A[j], B[j] - A[j]
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        ok = np.abs(rxs) > 1e-15
        t = np.where(ok, (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(ok, rxs, 1), -1)
        u = np.where(ok, (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / np.where(ok, rxs, 1), -1)
        if np.any(ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)):
            return True
    return False


def ear_clip(P, ids=None, edges=None):
    """Triangulate a simple 2D polygon (CCW). Returns (m, 3) local indices or
    None if it gets stuck. If `ids` (global vertex ids) and `edges` (set of
    existing undirected edges) are given, ears whose cutting diagonal is
    already an edge of the mesh are rejected."""
    idx = list(range(len(P)))
    tris = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3:
        n = len(idx)
        found = False
        Q = P[idx]
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 1e-14:
                continue  # reflex or degenerate
            # any other vertex inside triangle abc?
            d1 = (b[0] - a[0]) * (Q[:, 1] - a[1]) - (b[1] - a[1]) * (Q[:, 0] - a[0])
            d2 = (c[0] - b[0]) * (Q[:, 1] - b[1]) - (c[1] - b[1]) * (Q[:, 0] - b[0])
            d3 = (a[0] - c[0]) * (Q[:, 1] - c[1]) - (a[1] - c[1]) * (Q[:, 0] - c[0])
            inside = (d1 >= 0) & (d2 >= 0) & (d3 >= 0)
            inside[[(k - 1) % n, k, (k + 1) % n]] = False
            if inside.any():
                continue
            if edges is not None and n > 3 and _ekey(ids[i0], ids[i2]) in edges:
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            found = True
            break
        if not found:
            guard += 1
            if guard > 1:
                return None
            # drop one degenerate (collinear) vertex and retry
            for k in range(n):
                a, b, c = P[idx[k - 1]], P[idx[k]], P[idx[(k + 1) % n]]
                if abs(cross(a, b, c)) <= 1e-14:
                    del idx[k]
                    guard = 0
                    break
            else:
                return None
    tris.append(tuple(idx))
    return np.array(tris)


def _ekey(a, b):
    return (int(a), int(b)) if a < b else (int(b), int(a))


class Patch:
    """Triangle patch over a hole. Local vertices 0..nb-1 are the hole
    boundary (in CCW patch order), later vertices are inserted ones."""

    def __init__(self, X, sigma, faces, nb, ids, mesh_edges):
        self.X = [tuple(map(float, p)) for p in X]
        self.sigma = list(map(float, sigma))
        self.faces = [tuple(map(int, f)) for f in faces]
        self.nb = nb
        self.ids = ids
        self.mesh_edges = mesh_edges
        self.e2f = defaultdict(set)
        for fid, f in enumerate(self.faces):
            self._link(fid, f)

    def _link(self, fid, f):
        for k in range(3):
            self.e2f[_ekey(f[k], f[k - 2])].add(fid)

    def _unlink(self, fid, f):
        for k in range(3):
            self.e2f[_ekey(f[k], f[k - 2])].discard(fid)

    def _set(self, fid, f):
        self._unlink(fid, self.faces[fid])
        self.faces[fid] = f
        self._link(fid, f)

    def _add(self, f):
        self.faces.append(f)
        self._link(len(self.faces) - 1, f)

    def refine(self, alpha=math.sqrt(2), max_passes=100):
        X, sig = self.X, self.sigma
        self.relax()
        for _ in range(max_passes):
            inserted = 0
            for fid in range(len(self.faces)):
                i, j, k = self.faces[fid]
                c = tuple((X[i][d] + X[j][d] + X[k][d]) / 3 for d in range(3))
                sc = (sig[i] + sig[j] + sig[k]) / 3
                if all(alpha * math.dist(c, X[m]) > max(sc, sig[m]) for m in (i, j, k)):
                    ci = len(X)
                    X.append(c)
                    sig.append(sc)
                    self._set(fid, (i, j, ci))
                    self._add((j, k, ci))
                    self._add((k, i, ci))
                    inserted += 1
            if not inserted:
                break
            self.relax()

    def relax(self, max_flips=None):
        queue = [e for e, fs in self.e2f.items() if len(fs) == 2]
        max_flips = max_flips or 20 * len(self.faces) + 100
        flips = 0
        while queue and flips < max_flips:
            if self._flip(*queue.pop()):
                flips += 1
                f1, f2 = self.faces[self._last[0]], self.faces[self._last[1]]
                queue.extend({_ekey(f[k], f[k - 2]) for f in (f1, f2) for k in range(3)})

    def _flip(self, a, b):
        fs = self.e2f.get((a, b))
        if not fs or len(fs) != 2:
            return False
        f1, f2 = fs
        t1, t2 = self.faces[f1], self.faces[f2]
        # rotate t1 to (a, b, p) following its orientation
        k = next(k for k in range(3) if {t1[k], t1[(k + 1) % 3]} == {a, b})
        a, b, p = t1[k], t1[(k + 1) % 3], t1[(k + 2) % 3]
        q = next(v for v in t2 if v != a and v != b)
        if self.e2f.get(_ekey(p, q)):
            return False
        if p < self.nb and q < self.nb and _ekey(self.ids[p], self.ids[q]) in self.mesh_edges:
            return False
        X = self.X
        if _angle(X[a], X[p], X[b]) + _angle(X[a], X[q], X[b]) <= math.pi + 1e-9:
            return False  # already locally Delaunay
        n_old = _add3(_normal(X[a], X[b], X[p]), _normal(X[b], X[a], X[q]))
        if _dot(_normal(X[q], X[b], X[p]), n_old) <= 0 or _dot(_normal(X[p], X[a], X[q]), n_old) <= 0:
            return False  # quad not convex
        self._set(f1, (q, b, p))
        self._set(f2, (p, a, q))
        self._last = (f1, f2)
        return True


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _add3(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _normal(a, b, c):
    u, v = _sub(b, a), _sub(c, a)
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _angle(a, o, b):
    u, v = _sub(a, o), _sub(b, o)
    nu, nv = math.sqrt(_dot(u, u)), math.sqrt(_dot(v, v))
    if nu == 0 or nv == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, _dot(u, v) / (nu * nv))))


def harmonic(faces, nb, B, X0, tol=1e-8, max_iter=5000):
    """Membrane interpolation: solve L x = 0 (uniform weights) for the
    interior vertices nb.. with the boundary values B (nb x k) fixed.
    X0 is the initial guess for the interior. Conjugate gradient."""
    m = len(X0)
    if m == 0:
        return X0
    n = nb + m
    F = np.asarray(faces)
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    i, j = e[:, 0], e[:, 1]
    deg = np.bincount(e.ravel(), minlength=n).astype(float)

    def nsum(y):
        return np.stack([np.bincount(i, y[j, c], n) + np.bincount(j, y[i, c], n)
                         for c in range(y.shape[1])], axis=1)

    def A(x):
        y = np.zeros((n, x.shape[1]))
        y[nb:] = x
        return (deg[:, None] * y - nsum(y))[nb:]

    yb = np.zeros((n, B.shape[1]))
    yb[:nb] = B
    rhs = nsum(yb)[nb:]
    x = X0.astype(float).copy()
    r = rhs - A(x)
    p = r.copy()
    rr = (r * r).sum(0)
    stop = tol ** 2 * max((rhs * rhs).sum(), 1e-30)
    for _ in range(max_iter):
        if rr.sum() <= stop:
            break
        Ap = A(p)
        alpha = rr / np.maximum((p * Ap).sum(0), 1e-300)
        x += alpha * p
        r -= alpha * Ap
        rr_new = (r * r).sum(0)
        p = r + (rr_new / np.maximum(rr, 1e-300)) * p
        rr = rr_new
    return x


def fill_loop(V, loop, spacing, edges, next_index, attrs=None, max_faces=20000, fair=True):
    """Fill the hole bounded by `loop`.

    Returns (new_vertices, faces, new_attrs): faces use global indices, new
    vertices are numbered from `next_index`, and faces are oriented
    consistently with the surrounding mesh (they contain the reverse of the
    boundary half-edges). `attrs` (len(loop) x k, e.g. colours) are
    interpolated to the new vertices. `edges` (set of undirected mesh
    edges) is updated with the new edges."""
    P3 = V[loop]
    c, vt, _ = plane_fit(P3)
    P2 = (P3 - c) @ vt[:2].T
    area = 0.5 * np.sum(P2[:, 0] * np.roll(P2[:, 1], -1) - np.roll(P2[:, 0], -1) * P2[:, 1])
    order = np.arange(len(loop)) if area > 0 else np.arange(len(loop))[::-1]
    ids = loop[order]
    nb = len(ids)

    tris = None
    if not _segments_intersect(P2[order]):
        tris = ear_clip(P2[order], ids, edges)
    if tris is not None:
        # work in the projection plane; heights are recovered by fairing
        param = True
        X = np.column_stack([P2[order], np.zeros(nb)])
        flip = area > 0  # CCW order == loop direction -> reverse faces
    else:
        param = False
        X = np.vstack([P3[order], c])
        tris = np.column_stack([np.arange(nb), np.roll(np.arange(nb), -1), np.full(nb, nb)])
        flip = area > 0

    tri_area = 0.5 * np.linalg.norm(np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]]), axis=1).sum()
    floor = math.sqrt(4 * tri_area / (math.sqrt(3) * max_faces)) if max_faces else 0.0
    sigma = np.maximum(spacing[ids], floor)
    if not param:
        sigma = np.append(sigma, sigma.mean())

    patch = Patch(X, sigma, tris, nb, ids, edges)
    patch.refine()
    X = np.array(patch.X)
    faces = np.array(patch.faces)

    m = len(X) - nb
    if param:
        h_b = (P3[order] - c) @ vt[2]
        h = harmonic(faces, nb, h_b[:, None], np.zeros((m, 1)))[:, 0] if fair else np.zeros(m)
        new_v = c + X[nb:, :2] @ vt[:2] + h[:, None] * vt[2]
    else:
        new_v = harmonic(faces, nb, X[:nb], X[nb:]) if fair else X[nb:]

    new_attrs = None
    if attrs is not None:
        A_b = np.asarray(attrs, float)[order]
        new_attrs = harmonic(faces, nb, A_b, np.tile(A_b.mean(0), (m, 1)))

    gmap = np.concatenate([ids, next_index + np.arange(m)])
    faces = gmap[faces]
    if flip:
        faces = faces[:, ::-1]
    for f in faces:
        edges.update((_ekey(f[0], f[1]), _ekey(f[1], f[2]), _ekey(f[2], f[0])))
    return new_v, faces, new_attrs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="input .ply mesh(es)")
    ap.add_argument("-o", "--output", help="output path (single input only); default <name>_filled.ply")
    ap.add_argument("--walls-only", action="store_true", help="only fill holes whose best-fit plane is roughly vertical")
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"],
                    help="vertical axis for --walls-only; auto = axis with the smallest extent (default)")
    ap.add_argument("--wall-tol", type=float, default=0.5,
                    help="max |normal . up| for a hole to count as a wall hole (default 0.5 ~ 60deg from vertical)")
    ap.add_argument("--max-perimeter", type=float, default=np.inf, help="skip holes with a longer perimeter (m)")
    ap.add_argument("--keep-outer", type=int, default=0,
                    help="leave the N largest loops open, e.g. the border of an open scan (default 0)")
    ap.add_argument("--max-faces", type=int, default=20000,
                    help="approximate face budget per hole; very large holes get coarser triangles (default 20000)")
    ap.add_argument("--flat", action="store_true", help="skip fairing: leave patches on the hole's best-fit plane")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every loop")
    args = ap.parse_args()

    if args.output and len(args.inputs) > 1:
        ap.error("-o can only be used with a single input")

    for path in args.inputs:
        out_path = args.output or os.path.splitext(path)[0] + "_filled.ply"
        mesh = trimesh.load(path, process=False)
        V = np.asarray(mesh.vertices)
        F = np.asarray(mesh.faces)
        col = np.asarray(mesh.visual.vertex_colors) if mesh.visual.kind == "vertex" else None

        up_axis = {"x": 0, "y": 1, "z": 2}.get(args.up)
        if up_axis is None:
            up_axis = int(np.argmin(np.ptp(V, axis=0)))
        up = np.eye(3)[up_axis]

        e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
        edges = set(map(tuple, e.tolist()))
        # mean length of the edges around each vertex = local mesh density
        elen = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        spacing = np.bincount(e.ravel(), np.repeat(elen, 2), len(V)) / np.maximum(np.bincount(e.ravel(), minlength=len(V)), 1)

        loops = boundary_loops(F, len(V))
        perims = [np.linalg.norm(V[l] - V[np.roll(l, -1)], axis=1).sum() for l in loops]
        order = np.argsort(perims)[::-1]
        outer = set(order[: args.keep_outer].tolist())

        new_vs, new_fs, new_cs = [], [], []
        n_new = len(V)
        filled = 0
        skipped = defaultdict(int)
        for li in order:
            loop, per = loops[li], perims[li]
            _, vt, s = plane_fit(V[loop])
            tilt = abs(vt[2] @ up)
            reason = None
            if li in outer:
                reason = "outer border"
            elif per > args.max_perimeter:
                reason = "too large"
            elif args.walls_only and tilt > args.wall_tol:
                reason = "not a wall"
            if args.verbose:
                print(f"  loop {li:4d}: {len(loop):5d} verts, perim {per:7.3f} m, |n.up| {tilt:.2f}, "
                      f"nonplanar {s[2] / max(s[0], 1e-12):.2f} -> {reason or 'FILL'}", end="", flush=True)
            if reason:
                skipped[reason] += 1
                if args.verbose:
                    print()
                continue
            nv, nf, nc = fill_loop(V, loop, spacing, edges, n_new,
                                   attrs=None if col is None else col[loop],
                                   max_faces=args.max_faces, fair=not args.flat)
            if args.verbose:
                print(f" (+{len(nf)} faces)")
            new_vs.append(nv)
            new_fs.append(nf)
            if col is not None:
                new_cs.append(np.clip(np.round(nc), 0, 255).astype(col.dtype))
            n_new += len(nv)
            filled += 1

        V2 = np.vstack([V] + new_vs)
        F2 = np.vstack([F] + new_fs)
        kwargs = {}
        if col is not None:
            kwargs["vertex_colors"] = np.vstack([col] + new_cs)

        out = trimesh.Trimesh(V2, F2, process=False, **kwargs)
        out.export(out_path)

        e = np.sort(out.edges, axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        print(f"{path}: {len(loops)} boundary loops, filled {filled}, skipped {dict(skipped)}; "
              f"+{len(F2) - len(F)} faces, +{len(V2) - len(V)} verts; "
              f"open edges {np.sum(cnt == 1)} (was {sum(len(l) for l in loops)}) -> {out_path}")


if __name__ == "__main__":
    main()

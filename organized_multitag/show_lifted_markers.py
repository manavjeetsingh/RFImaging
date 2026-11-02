import numpy as np
import open3d as o3d

# Temporary: show merged scene + all merged markers, a copy lifted by +1 m in z, and lines between them.
OUT = "/Users/manavjeet/git/RFImaging/organized_multitag/roomscans/merged_out"
MESH = f"{OUT}/merged.ply"
CSV = f"{OUT}/merged_markers.csv"
LIFT = 1.0
R = 0.03

mesh = o3d.io.read_triangle_mesh(MESH)
mesh.compute_vertex_normals()
a = np.loadtxt(CSV, delimiter=",", skiprows=1, ndmin=2, usecols=(0, 1, 2, 3))  # skip the scene column
pts = a[:, 1:4]
up = pts + [0, 0, LIFT]
print(f"{len(pts)} markers from {CSV}")

geoms = [mesh]
for p, color in [(q, [0.9, 0.1, 0.1]) for q in pts] + [(q, [0.1, 0.4, 0.9]) for q in up]:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=R)
    s.compute_vertex_normals(); s.translate(p); s.paint_uniform_color(color)
    geoms.append(s)

lines = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(np.vstack([pts, up])),
    lines=o3d.utility.Vector2iVector([[i, i + len(pts)] for i in range(len(pts))]))
lines.paint_uniform_color([0, 0, 0])
geoms.append(lines)

o3d.visualization.draw_geometries(geoms, window_name="Markers +1 m")

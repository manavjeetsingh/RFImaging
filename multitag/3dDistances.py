import os, sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Usage: python 3dDistances.py [path/to/file.ply]
#   Shift + right-click on the model to drop a sphere.
#   Shift + right-drag an existing sphere to move it along the surface.
#   Close the window or press "Done" to print/save the sphere-to-sphere distance matrix.
PLY_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/manavjeet/git/RFImaging/multitag/4/2214_4.ply"
OUT_DIR = os.path.dirname(os.path.abspath(PLY_PATH))  # results saved next to the PLY
SINK = 0.75  # sphere center is pushed this many radii below the surface

COLORS = [[0.9, 0.1, 0.1], [0.1, 0.6, 0.9], [0.1, 0.8, 0.2], [0.95, 0.7, 0.1],
          [0.7, 0.2, 0.9], [0.1, 0.9, 0.8], [0.9, 0.4, 0.6], [0.5, 0.5, 0.5]]


class SpherePicker:
    def __init__(self, path):
        self.hits = []     # list of (surface point, inward unit vector)
        self.labels = []
        self.n_drawn = 0
        self.dragging = None  # index of sphere being moved

        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            self.geom = mesh
            self.raycaster = o3d.t.geometry.RaycastingScene()
            self.raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        else:
            pcd = o3d.io.read_point_cloud(path)
            self.geom, self.raycaster = pcd, None
            self.pcd_pts = np.asarray(pcd.points)
            mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"; mat.point_size = 3
        bbox = self.geom.get_axis_aligned_bounding_box()
        self.diag = np.linalg.norm(bbox.get_extent())

        self.window = gui.Application.instance.create_window("Place spheres", 1400, 900)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.add_geometry("model", self.geom, mat)
        self.scene.setup_camera(60, bbox, bbox.get_center())
        self.scene.set_on_mouse(self._on_mouse)

        # side panel
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Shift + right-click: place sphere\nShift + right-drag sphere: move it"))
        self.panel.add_child(gui.Label("Sphere radius"))
        self.radius = gui.Slider(gui.Slider.DOUBLE)
        self.radius.set_limits(0.0002 * self.diag, 0.02 * self.diag)
        self.radius.double_value = 0.002 * self.diag
        self.r = self.radius.double_value  # cached: the slider is destroyed when the window closes
        self.radius.set_on_value_changed(self._on_radius)
        self.panel.add_child(self.radius)
        self.list = gui.ListView()
        self.panel.add_child(self.list)
        for text, cb in [("Undo last", self._undo), ("Clear all", self._clear), ("Done", self._done)]:
            b = gui.Button(text); b.set_on_clicked(cb); self.panel.add_child(b)

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def _on_layout(self, ctx):
        r = self.window.content_rect
        w = 22 * ctx.theme.font_size
        self.scene.frame = gui.Rect(r.x, r.y, r.width - w, r.height)
        self.panel.frame = gui.Rect(r.get_right() - w, r.y, w, r.height)

    def _on_mouse(self, event):
        T = gui.MouseEvent.Type
        x = event.x - self.scene.frame.x
        y = event.y - self.scene.frame.y

        if (event.type == T.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.SHIFT)
                and event.is_button_down(gui.MouseButton.RIGHT)):
            i = self._sphere_at(x, y)
            if i is not None:
                self.dragging = i
            else:
                hit = self._surface_point(x, y)
                if hit is not None:
                    self._add(hit)
            return gui.Widget.EventCallbackResult.HANDLED

        if self.dragging is not None and event.type in (T.DRAG, T.BUTTON_UP):
            hit = self._surface_point(x, y)
            if hit is not None:
                self.hits[self.dragging] = hit
                self._draw_sphere(self.dragging)
                self._update_list()
            if event.type == T.BUTTON_UP:
                self.dragging = None
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _ray(self, x, y):
        cam = self.scene.scene.camera
        origin = np.asarray(cam.get_model_matrix())[:3, 3]
        target = np.asarray(cam.unproject(x, y, 0.5, self.scene.frame.width, self.scene.frame.height))
        d = target - origin
        return origin, d / np.linalg.norm(d)

    def _surface_point(self, x, y):
        """(surface point, inward unit vector) on the model under pixel (x, y), or None. Ignores the spheres."""
        o, d = self._ray(x, y)
        if self.raycaster is not None:
            ans = self.raycaster.cast_rays(o3d.core.Tensor([[*o, *d]], dtype=o3d.core.Dtype.Float32))
            t = ans["t_hit"].numpy()[0]
            if not np.isfinite(t):
                return None
            n = ans["primitive_normals"].numpy()[0].astype(float)
            n /= np.linalg.norm(n)
            inward = n if n @ d > 0 else -n  # face normal pointing away from the camera
            return o + t * d, inward
        # point cloud: nearest-to-camera point among those close to the ray
        v = self.pcd_pts - o
        t = v @ d
        perp = np.linalg.norm(v - t[:, None] * d, axis=1)
        near = np.where((perp < 0.002 * self.diag) & (t > 0))[0]
        if len(near) == 0:
            return None
        return self.pcd_pts[near[np.argmin(t[near])]].copy(), d  # no normals: sink along view ray

    def _sphere_at(self, x, y, px_tol=12):
        """Index of the sphere whose center projects within px_tol pixels of (x, y), else None."""
        if not self.points:
            return None
        cam = self.scene.scene.camera
        M = np.asarray(cam.get_projection_matrix()) @ np.asarray(cam.get_view_matrix())
        w, h = self.scene.frame.width, self.scene.frame.height
        best, best_d = None, px_tol
        for i, p in enumerate(self.points):
            c = M @ np.r_[p, 1.0]
            if c[3] <= 0:
                continue
            sx, sy = (c[0] / c[3] + 1) / 2 * w, (1 - c[1] / c[3]) / 2 * h
            dist = np.hypot(sx - x, sy - y)
            if dist < best_d:
                best, best_d = i, dist
        return best

    @property
    def points(self):
        """Sphere centers: surface hit pushed SINK radii into the model."""
        return [p + SINK * self.r * n for p, n in self.hits]

    def _on_radius(self, value):
        self.r = value
        self._redraw()

    def _add(self, hit):
        self.hits.append(hit)
        self._redraw()

    def _undo(self):
        if self.hits:
            self.hits.pop(); self._redraw()

    def _clear(self):
        self.hits.clear(); self._redraw()

    def _done(self):
        self.window.close()

    def _draw_sphere(self, i):
        name = f"sphere_{i}"
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)
        if i < len(self.labels):
            self.scene.remove_3d_label(self.labels[i])
        r = self.r
        p = self.points[i]
        s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        s.compute_vertex_normals(); s.translate(p)
        m = rendering.MaterialRecord(); m.shader = "defaultLit"
        m.base_color = COLORS[i % len(COLORS)] + [1.0]
        self.scene.scene.add_geometry(name, s, m)
        lab = self.scene.add_3d_label(p + [0, 0, 1.5 * r], str(i + 1))
        if i < len(self.labels):
            self.labels[i] = lab
        else:
            self.labels.append(lab)

    def _update_list(self):
        self.list.set_items([f"{i + 1}: {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}" for i, p in enumerate(self.points)])

    def _redraw(self):
        for i in range(self.n_drawn):
            self.scene.scene.remove_geometry(f"sphere_{i}")
        for lab in self.labels:
            self.scene.remove_3d_label(lab)
        self.labels = []
        for i in range(len(self.points)):
            self._draw_sphere(i)
        self.n_drawn = len(self.points)
        self._update_list()

def main():
    gui.Application.instance.initialize()
    picker = SpherePicker(PLY_PATH)
    gui.Application.instance.run()

    pts = np.array(picker.points).reshape(-1, 3)
    if len(pts) == 0:
        print("No spheres placed."); return
    D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(f"{len(pts)} sphere centers:\n{pts}\n\nDistance matrix:\n{D}")
    np.savetxt(os.path.join(OUT_DIR, "points.csv"), pts, delimiter=",", fmt="%.6f", header="x,y,z", comments="")
    np.savetxt(os.path.join(OUT_DIR, "distances.csv"), D, delimiter=",", fmt="%.4f")
    print(f"Saved points.csv and distances.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()

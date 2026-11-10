import os, sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Usage: python markLocations.py [scan name or path/to/file.ply]
#   With no argument, lists the PLY files in roomscans/ and asks which one to open.
#   Shift + right-click on the model to drop a numbered sphere (1, 2, 3, ...).
#   Edit "Next number" to jump the numbering, e.g. 4 -> 8 makes the next sphere 8, then 9, ...
#   Shift + right-drag an existing sphere to move it along the surface.
#   "Snap to common plane" projects all spheres onto their best-fit plane (also done automatically on save).
#   Close the window or press "Done" to save:
#     <name>_marked.ply   the scan with the spheres merged in
#     <name>_markers.csv  index,x,y,z of each sphere center
ROOMSCANS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "roomscans")

COLORS = [[0.9, 0.1, 0.1], [0.1, 0.6, 0.9], [0.1, 0.8, 0.2], [0.95, 0.7, 0.1],
          [0.7, 0.2, 0.9], [0.1, 0.9, 0.8], [0.9, 0.4, 0.6], [0.5, 0.5, 0.5]]


def pick_ply():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            return arg
        path = os.path.join(ROOMSCANS, arg if arg.endswith(".ply") else arg + ".ply")
        if os.path.isfile(path):
            return path
        sys.exit(f"Not found: {arg}")
    files = sorted(f for f in os.listdir(ROOMSCANS) if f.endswith(".ply") and not f.endswith("_marked.ply"))
    if not files:
        sys.exit(f"No PLY files in {ROOMSCANS}")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    return os.path.join(ROOMSCANS, files[int(input("Scan to mark: "))])


class SpherePlacer:
    def __init__(self, path):
        self.points = []   # sphere centers (clicked surface points)
        self.ids = []      # number shown/saved for each sphere
        self.labels = []
        self.n_drawn = 0
        self.dragging = None  # index of sphere being moved

        self.mesh = o3d.io.read_triangle_mesh(path)
        if len(self.mesh.triangles) == 0:
            sys.exit(f"{path} has no triangles; expected a mesh")
        self.mesh.compute_vertex_normals()
        self.raycaster = o3d.t.geometry.RaycastingScene()
        self.raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        bbox = self.mesh.get_axis_aligned_bounding_box()
        self.diag = np.linalg.norm(bbox.get_extent())

        self.window = gui.Application.instance.create_window(f"Mark locations: {os.path.basename(path)}", 1400, 900)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.add_geometry("model", self.mesh, mat)
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
        row = gui.Horiz(0.5 * em)
        row.add_child(gui.Label("Next number"))
        self.next_id = gui.NumberEdit(gui.NumberEdit.INT)
        self.next_id.set_value(1)
        self.next_id.set_limits(1, 100000)
        row.add_child(self.next_id)
        self.panel.add_child(row)
        self.list = gui.ListView()
        self.panel.add_child(self.list)
        for text, cb in [("Snap to common plane", self._flatten), ("Undo last", self._undo),
                         ("Clear all", self._clear), ("Done", self._done)]:
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
                p = self._surface_point(x, y)
                if p is not None:
                    self.points.append(p); self.ids.append(self.next_id.int_value)
                    self.next_id.set_value(self.next_id.int_value + 1)
                    self._redraw()
            return gui.Widget.EventCallbackResult.HANDLED

        if self.dragging is not None and event.type in (T.DRAG, T.BUTTON_UP):
            p = self._surface_point(x, y)
            if p is not None:
                self.points[self.dragging] = p
                self._draw_sphere(self.dragging)
                self._update_list()
            if event.type == T.BUTTON_UP:
                self.dragging = None
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _surface_point(self, x, y):
        """Point on the model under pixel (x, y), or None. Ignores the spheres."""
        cam = self.scene.scene.camera
        o = np.asarray(cam.get_model_matrix())[:3, 3]
        target = np.asarray(cam.unproject(x, y, 0.5, self.scene.frame.width, self.scene.frame.height))
        d = (target - o) / np.linalg.norm(target - o)
        ans = self.raycaster.cast_rays(o3d.core.Tensor([[*o, *d]], dtype=o3d.core.Dtype.Float32))
        t = ans["t_hit"].numpy()[0]
        return o + t * d if np.isfinite(t) else None

    def _sphere_at(self, x, y, px_tol=12):
        """Index of the sphere whose center projects within px_tol pixels of (x, y), else None."""
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

    def _on_radius(self, value):
        self.r = value
        self._redraw()

    def snap_to_plane(self):
        """Project all sphere centers onto their least-squares best-fit plane. Returns False if < 3 spheres."""
        if len(self.points) < 3:
            print("Need at least 3 spheres to fit a plane."); return False
        P = np.array(self.points)
        c = P.mean(axis=0)
        n = np.linalg.svd(P - c)[2][-1]  # normal = direction of least variance
        off = (P - c) @ n
        print(f"Plane normal {n}, max offset removed {np.abs(off).max():.4f}")
        self.points = list(P - off[:, None] * n)
        return True

    def _flatten(self):
        if self.snap_to_plane():
            self._redraw()

    def _undo(self):
        if self.points:
            self.points.pop(); self.next_id.set_value(self.ids.pop()); self._redraw()

    def _clear(self):
        self.points.clear(); self.ids.clear(); self.next_id.set_value(1); self._redraw()

    def _done(self):
        self.window.close()

    def sphere_mesh(self, i):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=self.r)
        s.compute_vertex_normals(); s.translate(self.points[i])
        s.paint_uniform_color(COLORS[i % len(COLORS)])
        return s

    def _draw_sphere(self, i):
        name = f"sphere_{i}"
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)
        if i < len(self.labels):
            self.scene.remove_3d_label(self.labels[i])
        m = rendering.MaterialRecord(); m.shader = "defaultLit"
        self.scene.scene.add_geometry(name, self.sphere_mesh(i), m)
        lab = self.scene.add_3d_label(self.points[i] + [0, 0, 1.5 * self.r], str(self.ids[i]))
        if i < len(self.labels):
            self.labels[i] = lab
        else:
            self.labels.append(lab)

    def _update_list(self):
        self.list.set_items([f"{k}: {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}" for k, p in zip(self.ids, self.points)])

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
    path = pick_ply()
    gui.Application.instance.initialize()
    placer = SpherePlacer(path)
    gui.Application.instance.run()

    if not placer.points:
        print("No spheres placed; nothing saved."); return
    placer.snap_to_plane()  # window is gone, so no redraw
    base = os.path.splitext(path)[0]
    pts = np.array(placer.points)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(f"{len(pts)} sphere centers:\n{pts}")

    rows = np.column_stack([placer.ids, pts])
    np.savetxt(base + "_markers.csv", rows, delimiter=",", fmt=["%d", "%.6f", "%.6f", "%.6f"],
               header="index,x,y,z", comments="")

    marked = placer.mesh
    if not marked.has_vertex_colors():
        marked.paint_uniform_color([0.7, 0.7, 0.7])
    for i in range(len(pts)):
        marked += placer.sphere_mesh(i)
    o3d.io.write_triangle_mesh(base + "_marked.ply", marked)
    print(f"Saved {base}_marked.ply and {base}_markers.csv")


if __name__ == "__main__":
    main()

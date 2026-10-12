import os, sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# Usage: python 3dErase.py [path/to/file.ply]
#   Shift + right-click on the model to delete a circular patch around the clicked point.
#   The circle lies in the surface's tangent plane; "Depth" limits how far off that plane it cuts,
#   so it does not punch through to surfaces behind (e.g. the other side of a corner).
#   A ring under the mouse previews what will be deleted.
#   Close the window or press "Done" to save <name>_erased.ply next to the input.
PLY_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/manavjeet/git/RFImaging/multitag/4/2214_4.ply"
OUT_PATH = os.path.splitext(os.path.abspath(PLY_PATH))[0] + "_erased.ply"
RING_SEGMENTS = 64


class CircleEraser:
    def __init__(self, path):
        self.undo_stack = []  # arrays of element indices removed by each click
        self.hover = None     # (point, normal) under the mouse, for the preview ring

        mesh = o3d.io.read_triangle_mesh(path)
        self.is_mesh = len(mesh.triangles) > 0
        if self.is_mesh:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            self.src = mesh
            self.V = np.asarray(mesh.vertices)
            self.F = np.asarray(mesh.triangles)
            self.centers = self.V[self.F].mean(axis=1)  # one element per triangle
            self.mat = rendering.MaterialRecord(); self.mat.shader = "defaultLit"
        else:
            pcd = o3d.io.read_point_cloud(path)
            self.src = pcd
            self.centers = np.asarray(pcd.points)       # one element per point
            self.mat = rendering.MaterialRecord(); self.mat.shader = "defaultUnlit"; self.mat.point_size = 3
        self.alive = np.ones(len(self.centers), bool)
        bbox = self.src.get_axis_aligned_bounding_box()
        self.diag = np.linalg.norm(bbox.get_extent())

        self.window = gui.Application.instance.create_window("Erase circles", 1400, 900)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.setup_camera(60, bbox, bbox.get_center())
        self.scene.set_on_mouse(self._on_mouse)

        # side panel
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Shift + right-click: delete circle\nRing under the mouse = preview"))
        self.panel.add_child(gui.Label("Circle radius"))
        self.radius = gui.Slider(gui.Slider.DOUBLE)
        self.radius.set_limits(0.001 * self.diag, 0.1 * self.diag)
        self.radius.double_value = 0.01 * self.diag
        self.r = self.radius.double_value  # cached: the slider is destroyed when the window closes
        self.radius.set_on_value_changed(self._on_radius)
        self.panel.add_child(self.radius)
        self.panel.add_child(gui.Label("Depth (fraction of radius)"))
        self.depth = gui.Slider(gui.Slider.DOUBLE)
        self.depth.set_limits(0.05, 2.0)
        self.depth.double_value = 0.5
        self.dfrac = self.depth.double_value
        self.depth.set_on_value_changed(self._on_depth)
        self.panel.add_child(self.depth)
        self.status = gui.Label("")
        self.panel.add_child(self.status)
        for text, cb in [("Undo last", self._undo), ("Restore all", self._restore), ("Done", self._done)]:
            b = gui.Button(text); b.set_on_clicked(cb); self.panel.add_child(b)

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        self._rebuild()

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
            hit = self._surface_point(x, y)
            if hit is not None:
                self._erase(*hit)
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == T.MOVE:
            self.hover = self._surface_point(x, y)
            self._draw_ring()
        return gui.Widget.EventCallbackResult.IGNORED

    def _ray(self, x, y):
        cam = self.scene.scene.camera
        origin = np.asarray(cam.get_model_matrix())[:3, 3]
        target = np.asarray(cam.unproject(x, y, 0.5, self.scene.frame.width, self.scene.frame.height))
        d = target - origin
        return origin, d / np.linalg.norm(d)

    def _surface_point(self, x, y):
        """(surface point, unit normal) on the remaining model under pixel (x, y), or None."""
        o, d = self._ray(x, y)
        if self.is_mesh:
            ans = self.raycaster.cast_rays(o3d.core.Tensor([[*o, *d]], dtype=o3d.core.Dtype.Float32))
            t = ans["t_hit"].numpy()[0]
            if not np.isfinite(t):
                return None
            n = ans["primitive_normals"].numpy()[0].astype(float)
            return o + t * d, n / np.linalg.norm(n)
        # point cloud: nearest-to-camera remaining point close to the ray; circle faces the camera
        pts = self.centers[self.alive]
        v = pts - o
        t = v @ d
        perp = np.linalg.norm(v - t[:, None] * d, axis=1)
        near = np.where((perp < 0.002 * self.diag) & (t > 0))[0]
        if len(near) == 0:
            return None
        return pts[near[np.argmin(t[near])]].copy(), -d

    def _selection(self, p, n):
        """Indices of remaining elements inside the circle (radius r, thickness 2*depth) at p with normal n."""
        v = self.centers - p
        along = v @ n
        across = np.linalg.norm(v - along[:, None] * n, axis=1)
        return np.where(self.alive & (across <= self.r) & (np.abs(along) <= self.dfrac * self.r))[0]

    def _erase(self, p, n):
        idx = self._selection(p, n)
        if len(idx) == 0:
            return
        self.alive[idx] = False
        self.undo_stack.append(idx)
        self._rebuild()

    def _undo(self):
        if self.undo_stack:
            self.alive[self.undo_stack.pop()] = True
            self._rebuild()

    def _restore(self):
        if self.undo_stack:
            self.undo_stack.clear()
            self.alive[:] = True
            self._rebuild()

    def _done(self):
        self.window.close()

    def _on_radius(self, value):
        self.r = value
        self._draw_ring()

    def _on_depth(self, value):
        self.dfrac = value

    def current_geometry(self):
        """The model with the erased elements removed (unreferenced vertices dropped)."""
        if self.is_mesh:
            m = o3d.geometry.TriangleMesh(self.src)
            m.remove_triangles_by_mask(~self.alive)
            m.remove_unreferenced_vertices()
            return m
        return self.src.select_by_index(np.where(self.alive)[0])

    def _rebuild(self):
        geom = self.current_geometry()
        if self.scene.scene.has_geometry("model"):
            self.scene.scene.remove_geometry("model")
        self.scene.scene.add_geometry("model", geom, self.mat)
        if self.is_mesh:  # rays must pass through the holes we cut
            self.raycaster = o3d.t.geometry.RaycastingScene()
            self.raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(geom))
        kind = "triangles" if self.is_mesh else "points"
        self.status.text = f"Removed {np.sum(~self.alive)} {kind} in {len(self.undo_stack)} circles"
        self._draw_ring()

    def _draw_ring(self):
        if self.scene.scene.has_geometry("ring"):
            self.scene.scene.remove_geometry("ring")
        if self.hover is None:
            return
        p, n = self.hover
        a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        th = np.linspace(0, 2 * np.pi, RING_SEGMENTS, endpoint=False)
        pts = p + 0.002 * self.diag * n + self.r * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
        lines = np.column_stack([np.arange(RING_SEGMENTS), np.roll(np.arange(RING_SEGMENTS), -1)])
        ring = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
        m = rendering.MaterialRecord(); m.shader = "unlitLine"; m.line_width = 3
        m.base_color = [0.9, 0.1, 0.1, 1.0]
        self.scene.scene.add_geometry("ring", ring, m)


def main():
    gui.Application.instance.initialize()
    eraser = CircleEraser(PLY_PATH)
    gui.Application.instance.run()

    removed = int(np.sum(~eraser.alive))
    if removed == 0:
        print("Nothing erased."); return
    geom = eraser.current_geometry()
    if eraser.is_mesh:
        o3d.io.write_triangle_mesh(OUT_PATH, geom)
    else:
        o3d.io.write_point_cloud(OUT_PATH, geom)
    kind = "triangles" if eraser.is_mesh else "points"
    print(f"Removed {removed} {kind} in {len(eraser.undo_stack)} circles; saved {OUT_PATH}")


if __name__ == "__main__":
    main()

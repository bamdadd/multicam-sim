"""Renderer backend: a Protocol plus an optional offscreen pyrender v1.

**Pixels are not the contract.** The manifest (analytic projection + boolean
occlusion) is the hand-off to the triangulation consumer; a renderer only exists
to *look at* a scene. So this module is deliberately decoupled:

* :class:`RendererBackend` is a ``Protocol`` — any backend that turns a
  ``Scene`` + camera + frame into an image satisfies it.
* :class:`PyrenderBackend` is a v1 offscreen implementation. ``pyrender`` /
  ``trimesh`` are an **optional extra** (``pip install multicam-sim[render]``),
  imported lazily inside :meth:`PyrenderBackend.render`. Nothing here is imported
  at package load, and CI never installs the extra — so the renderer can never
  break the manifest's green bar.

Future work is an open/closed swap of this Protocol, not a rewrite:
* a **Kubric/Blender** backend for photoreal frames (same Protocol);
* a **Rust core via pyo3** for the hot projection/occlusion path is a v2 concern
  (the geometry is analytic today and lives behind :mod:`multicam_sim.geometry`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np

from ..geometry import FloatArray

if TYPE_CHECKING:
    from ..scene import Scene


@runtime_checkable
class RendererBackend(Protocol):
    """Anything that renders one camera's view of a scene at a frame to pixels.

    The return is an ``(H, W, 3)`` uint8 image. Backends are interchangeable;
    none of them feeds the manifest.
    """

    def render(self, scene: Scene, camera_id: int, frame: int) -> FloatArray: ...


class PyrenderBackend:
    """Offscreen pyrender v1 (optional; not exercised in CI).

    Renders each entity's named points as small spheres and occluders as their
    solids, from the chosen camera. Requires the ``render`` extra; importing this
    class is cheap, but :meth:`render` lazily imports ``pyrender``/``trimesh`` and
    raises a clear error if the extra is absent.
    """

    def __init__(self, *, point_radius: float = 0.05, bg: tuple[float, float, float] = (0, 0, 0)):
        self.point_radius = point_radius
        self.bg = bg

    def render(self, scene: Scene, camera_id: int, frame: int) -> FloatArray:
        try:
            import pyrender
            import trimesh
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "PyrenderBackend needs the 'render' extra: pip install multicam-sim[render]"
            ) from exc

        cam = scene.cameras[camera_id]
        pr_scene = pyrender.Scene(bg_color=[*self.bg, 1.0], ambient_light=[0.4, 0.4, 0.4])

        # entity points -> small spheres at this frame's ground-truth coords
        for entity in scene.entities:
            match = next((f for f in entity.frames if f.frame == frame), None)
            if match is None:
                continue
            for xyz in match.points.values():
                sphere = trimesh.creation.uv_sphere(radius=self.point_radius)
                pose = np.eye(4)
                pose[:3, 3] = xyz
                pr_scene.add(pyrender.Mesh.from_trimesh(sphere), pose=pose)

        # occluders -> their solids
        for occ in scene.occluders:
            solid = _occluder_mesh(occ)
            if solid is not None:
                pr_scene.add(pyrender.Mesh.from_trimesh(solid))

        # camera: OpenCV intrinsics + OpenCV->OpenGL pose flip (RDF -> right/up/back)
        intr = cam.intrinsics
        pr_cam = pyrender.IntrinsicsCamera(fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy)
        flip = np.diag([1.0, -1.0, -1.0])
        pose = np.eye(4)
        pose[:3, :3] = cam.rotation().T @ flip
        pose[:3, 3] = cam.centre()
        pr_scene.add(pr_cam, pose=pose)
        pr_scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)

        renderer = pyrender.OffscreenRenderer(intr.width, intr.height)
        try:
            color, _ = renderer.render(pr_scene)
        finally:
            renderer.delete()
        return np.asarray(color, dtype=np.uint8)


def _occluder_mesh(occ: object) -> object | None:
    """Build a ``trimesh`` solid for a Box/Sphere occluder (best-effort, v1)."""
    import trimesh

    from ..occluders import Box, Sphere

    if isinstance(occ, Sphere):
        m = trimesh.creation.uv_sphere(radius=occ.radius)
        m.apply_translation(occ.center)
        return cast("object", m)
    if isinstance(occ, Box):
        m = trimesh.creation.box(extents=[2 * h for h in occ.half_extents])
        m.apply_translation(occ.center)
        return cast("object", m)
    return None

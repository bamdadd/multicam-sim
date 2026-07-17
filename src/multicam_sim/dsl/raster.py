"""Pure-numpy software rasterizer for :class:`RendererBackend` (zero system deps).

The offscreen backends in this package need heavy natives — ``pyrender`` wants an
OpenGL context, Kubric wants Blender. Neither runs on a bare CI box, so neither is
exercised there. This backend fills that gap: a **z-buffered triangle rasterizer
written in plain numpy** that renders on arm64 and x86_64 CI alike with nothing but
the packages already in ``dependencies`` (numpy + pydantic).

Like every renderer here, **pixels are not the contract** — this backend never
feeds the manifest. It exists so a scene can be *looked at* anywhere, and so the
test suite has a default backend that always runs.

Two properties are load-bearing and tested:

* **Projection reuse.** Every vertex is projected through the *same* OpenCV pinhole
  path the manifest uses (:meth:`multicam_sim.cameras.Camera.project`); the
  rasterizer never reimplements projection, it only z-buffers the result. See
  :func:`project_vertices`.
* **Determinism.** A z-buffer needs no sampling or jitter, so a fixed scene renders
  to fixed bytes in-process. No randomness anywhere.

Entities' named points become small spheres, Box/Sphere occluders become their
solids; everything is triangulated and fed through one perspective-correct
rasterizer (depth interpolated as ``1/w`` so occlusion ordering is exact).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..geometry import FloatArray
from ..occluders import Box, Sphere

if TYPE_CHECKING:
    from ..cameras import Camera
    from ..scene import Scene

RGB = tuple[float, float, float]


class RasterizerConfig(BaseModel):
    """Typed knobs for :class:`RasterizerBackend` (colours are linear ``0..1``)."""

    model_config = ConfigDict(frozen=True)

    point_radius: float = 0.05
    sphere_subdivisions: int = 1
    background: RGB = (0.05, 0.05, 0.08)
    point_color: RGB = (0.95, 0.40, 0.25)
    occluder_color: RGB = (0.50, 0.55, 0.60)
    ambient: float = 0.25
    near_eps: float = 1e-6


# --- primitive meshes (deterministic; no randomness) ------------------------

#: Golden-ratio icosahedron, the seed solid subdivided into a sphere.
_PHI = (1.0 + 5.0**0.5) / 2.0

_ICO_VERTS: FloatArray = np.array(
    [
        (-1.0, _PHI, 0.0),
        (1.0, _PHI, 0.0),
        (-1.0, -_PHI, 0.0),
        (1.0, -_PHI, 0.0),
        (0.0, -1.0, _PHI),
        (0.0, 1.0, _PHI),
        (0.0, -1.0, -_PHI),
        (0.0, 1.0, -_PHI),
        (_PHI, 0.0, -1.0),
        (_PHI, 0.0, 1.0),
        (-_PHI, 0.0, -1.0),
        (-_PHI, 0.0, 1.0),
    ],
    dtype=np.float64,
)

_ICO_FACES: FloatArray = np.array(
    [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ],
    dtype=np.int64,
)


def _icosphere(
    radius: float, center: FloatArray, subdivisions: int
) -> tuple[FloatArray, FloatArray]:
    """A unit icosphere subdivided ``subdivisions`` times, scaled and translated.

    Deterministic: the seed icosahedron and the midpoint split carry no ordering
    ambiguity, so the same arguments always yield byte-identical arrays.
    """
    verts = _ICO_VERTS / np.linalg.norm(_ICO_VERTS[0])
    faces = _ICO_FACES
    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)
    world = verts * radius + np.asarray(center, dtype=np.float64)
    return world, faces


def _subdivide(verts: FloatArray, faces: FloatArray) -> tuple[FloatArray, FloatArray]:
    """One loop of midpoint subdivision, each triangle -> four, verts re-normalised."""
    vert_list: list[FloatArray] = [row for row in verts]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        found = cache.get(key)
        if found is not None:
            return found
        mid = (vert_list[a] + vert_list[b]) / 2.0
        mid = mid / np.linalg.norm(mid)
        idx = len(vert_list)
        vert_list.append(mid)
        cache[key] = idx
        return idx

    new_faces: list[tuple[int, int, int]] = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
    return np.asarray(vert_list, dtype=np.float64), np.asarray(new_faces, dtype=np.int64)


#: The 8 corners of a unit cube (half-extent 1) and its 12 triangles.
_BOX_CORNERS: FloatArray = np.array(
    [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ],
    dtype=np.float64,
)

_BOX_FACES: FloatArray = np.array(
    [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (2, 3, 7),
        (2, 7, 6),
        (1, 2, 6),
        (1, 6, 5),
        (0, 4, 7),
        (0, 7, 3),
    ],
    dtype=np.int64,
)


def _box_mesh(center: FloatArray, half_extents: FloatArray) -> tuple[FloatArray, FloatArray]:
    """An axis-aligned box as 8 verts + 12 triangles (outward-wound)."""
    world = _BOX_CORNERS * np.asarray(half_extents, dtype=np.float64) + np.asarray(
        center, dtype=np.float64
    )
    return world, _BOX_FACES


# --- projection (reuses the manifest's OpenCV pinhole path) ------------------


def project_vertices(camera: Camera, verts: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Project ``(N, 3)`` world verts through :meth:`Camera.project`.

    This is the single seam where the rasterizer touches projection: it delegates
    every vertex to the camera's own OpenCV pinhole helper — the same one the
    manifest uses — and returns ``(uvs (N, 2), depths (N,))`` with ``depth == w``
    (camera-space Z; ``> 0`` is in front). No projection math is duplicated here.
    """
    points = np.asarray(verts, dtype=np.float64)
    uvs = np.empty((len(points), 2), dtype=np.float64)
    depths = np.empty(len(points), dtype=np.float64)
    for i, point in enumerate(points):
        uv, w = camera.project(point)
        uvs[i] = uv
        depths[i] = w
    return uvs, depths


# --- the rasterizer ---------------------------------------------------------


def _rasterize_mesh(
    verts: FloatArray,
    faces: FloatArray,
    color: RGB,
    camera: Camera,
    framebuffer: FloatArray,
    zbuffer: FloatArray,
    cfg: RasterizerConfig,
) -> None:
    """Z-buffer one triangle mesh into ``framebuffer``/``zbuffer`` (in place).

    Depth is perspective-correct (``1/w`` interpolated linearly in screen space);
    faces with any vertex at or behind the near plane are skipped whole (projection
    stays total, so those uv are meaningless). Flat headlight shading gives the
    solids some form without any light state to thread through.
    """
    uvs, depths = project_vertices(camera, verts)
    height, width, _ = framebuffer.shape
    cam_centre = camera.centre()
    base = np.asarray(color, dtype=np.float64)

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        d0, d1, d2 = depths[i0], depths[i1], depths[i2]
        if min(d0, d1, d2) <= cfg.near_eps:
            continue  # a vertex at/behind the near plane: skip the whole face

        (u0, v0), (u1, v1), (u2, v2) = uvs[i0], uvs[i1], uvs[i2]
        area = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
        if abs(area) < 1e-12:
            continue  # degenerate / edge-on triangle covers no pixels

        min_x = max(0, int(np.floor(min(u0, u1, u2))))
        max_x = min(width - 1, int(np.ceil(max(u0, u1, u2))))
        min_y = max(0, int(np.floor(min(v0, v1, v2))))
        max_y = min(height - 1, int(np.ceil(max(v0, v1, v2))))
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64)
        ys = np.arange(min_y, max_y + 1, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)

        w0 = (u2 - u1) * (gy - v1) - (v2 - v1) * (gx - u1)
        w1 = (u0 - u2) * (gy - v2) - (v0 - v2) * (gx - u2)
        w2 = (u1 - u0) * (gy - v0) - (v1 - v0) * (gx - u0)
        if area > 0:
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        else:
            inside = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        if not inside.any():
            continue

        l0, l1, l2 = w0 / area, w1 / area, w2 / area
        inv_depth = l0 / d0 + l1 / d1 + l2 / d2
        depth = np.where(inv_depth != 0.0, 1.0 / inv_depth, np.inf)

        y_idx, x_idx = np.nonzero(inside)
        px = x_idx + min_x
        py = y_idx + min_y
        cand = depth[y_idx, x_idx]
        closer = cand < zbuffer[py, px]
        if not closer.any():
            continue

        py, px, cand = py[closer], px[closer], cand[closer]
        zbuffer[py, px] = cand

        normal = np.cross(verts[i1] - verts[i0], verts[i2] - verts[i0])
        norm_len = float(np.linalg.norm(normal))
        centroid = (verts[i0] + verts[i1] + verts[i2]) / 3.0
        light = cam_centre - centroid
        light_len = float(np.linalg.norm(light))
        if norm_len < 1e-12 or light_len < 1e-12:
            shade = cfg.ambient
        else:
            lambert = abs(float(np.dot(normal / norm_len, light / light_len)))
            shade = cfg.ambient + (1.0 - cfg.ambient) * lambert
        framebuffer[py, px] = base * shade


class RasterizerBackend:
    """Pure-numpy z-buffered rasterizer implementing :class:`RendererBackend`.

    Zero system dependencies: it renders the same on a laptop and on a bare CI box.
    Named entity points are drawn as small spheres, Box/Sphere occluders as their
    solids. Deterministic — the same scene renders to identical bytes in-process.
    """

    def __init__(self, config: RasterizerConfig | None = None) -> None:
        self.config = config or RasterizerConfig()

    def render(self, scene: Scene, camera_id: int, frame: int) -> FloatArray:
        cfg = self.config
        camera = scene.cameras[camera_id]
        width = camera.intrinsics.width
        height = camera.intrinsics.height

        framebuffer = np.empty((height, width, 3), dtype=np.float64)
        framebuffer[:, :] = np.asarray(cfg.background, dtype=np.float64)
        zbuffer = np.full((height, width), np.inf, dtype=np.float64)

        # entity points -> small spheres at this frame's ground-truth coords
        for entity in scene.entities:
            match = next((f for f in entity.frames if f.frame == frame), None)
            if match is None:
                continue
            for xyz in match.points.values():
                verts, faces = _icosphere(
                    cfg.point_radius,
                    np.asarray(xyz, dtype=np.float64),
                    cfg.sphere_subdivisions,
                )
                _rasterize_mesh(verts, faces, cfg.point_color, camera, framebuffer, zbuffer, cfg)

        # occluders -> their solids
        for occ in scene.occluders:
            mesh = _occluder_mesh(occ, cfg)
            if mesh is None:
                continue
            verts, faces = mesh
            _rasterize_mesh(verts, faces, cfg.occluder_color, camera, framebuffer, zbuffer, cfg)

        image = np.clip(framebuffer, 0.0, 1.0) * 255.0
        return np.asarray(np.rint(image), dtype=np.uint8)


def _occluder_mesh(
    occ: Box | Sphere, cfg: RasterizerConfig
) -> tuple[FloatArray, FloatArray] | None:
    """Triangulate a Box/Sphere occluder into ``(verts, faces)`` world-space."""
    if isinstance(occ, Sphere):
        return _icosphere(
            occ.radius, np.asarray(occ.center, dtype=np.float64), cfg.sphere_subdivisions
        )
    if isinstance(occ, Box):
        return _box_mesh(
            np.asarray(occ.center, dtype=np.float64),
            np.asarray(occ.half_extents, dtype=np.float64),
        )
    return None


def default_backend() -> RasterizerBackend:
    """The suite-wide default :class:`RendererBackend`: needs no optional extra."""
    return RasterizerBackend()

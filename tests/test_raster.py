"""Tests for the pure-numpy rasterizer backend.

Two properties are load-bearing:

* projection is *reused*, not reimplemented — the rasterizer's per-vertex projection
  equals :meth:`Camera.project` to 1e-6 (helper-to-helper, not a centroid read-back
  which a pixel grid could never recover to that tolerance);
* rendering is deterministic — the same scene renders to identical bytes in-process
  (a z-buffer needs no sampling, so there is nothing to vary).
"""

from __future__ import annotations

import numpy as np

from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder
from multicam_sim.dsl.raster import (
    RasterizerBackend,
    RasterizerConfig,
    project_vertices,
)
from multicam_sim.dsl.render import RendererBackend
from multicam_sim.scene import Scene


def _scene() -> Scene:
    return (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=800.0,
                width=64,
                height_px=48,
            )
        )
        .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=1).during((3, 7)))
        .build()
    )


def test_backend_satisfies_protocol(backend: RendererBackend) -> None:
    assert isinstance(backend, RendererBackend)


def test_render_shape_and_dtype(backend: RendererBackend) -> None:
    img = backend.render(_scene(), camera_id=0, frame=5)
    assert img.shape == (48, 64, 3)
    assert img.dtype == np.uint8


def test_projection_matches_camera_helper_to_1e6() -> None:
    """The rasterizer's projection IS ``Camera.project`` — equal to 1e-6.

    Helper-to-helper on purpose: it proves ``render()`` routes every vertex through
    the manifest's OpenCV pinhole path instead of duplicating projection math. A
    rendered-blob centroid could never round-trip a point to 1e-6.
    """
    camera = _scene().cameras[0]
    world = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.3, -0.2, 0.7],
            [-0.4, 0.6, 0.4],
        ],
        dtype=np.float64,
    )
    uvs, depths = project_vertices(camera, world)
    for i, point in enumerate(world):
        uv, w = camera.project(point)
        assert np.allclose(uvs[i], uv, atol=1e-6)
        assert abs(depths[i] - w) <= 1e-6


def test_render_is_byte_stable_in_process(backend: RendererBackend) -> None:
    """Same fixed scene, rendered twice in-process -> identical bytes.

    Arch-local determinism only (no committed golden PNG byte-compared across
    architectures — float rounding at silhouette edges differs on x86_64 CI).
    """
    scene = _scene()
    first = backend.render(scene, camera_id=0, frame=5)
    second = backend.render(scene, camera_id=0, frame=5)
    assert np.array_equal(first, second)


def test_render_draws_points_over_background() -> None:
    """A point in view must paint pixels distinct from the background."""
    backend = RasterizerBackend(RasterizerConfig(point_radius=0.15))
    img = backend.render(_scene(), camera_id=0, frame=5)
    bg = np.rint(np.array([0.05, 0.05, 0.08]) * 255.0).astype(np.uint8)
    assert np.any(np.any(img != bg, axis=-1))


def test_distinct_cameras_differ() -> None:
    """Three ring cameras see the scene differently — renders must not coincide."""
    backend = RasterizerBackend()
    scene = _scene()
    cam0 = backend.render(scene, camera_id=0, frame=5)
    cam2 = backend.render(scene, camera_id=2, frame=5)
    assert not np.array_equal(cam0, cam2)

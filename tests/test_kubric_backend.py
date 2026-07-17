"""Kubric backend: Blender-free translation correctness (issue #39).

The Kubric renderer needs Blender, which cannot ``pip install`` cleanly, so these
tests **never invoke the renderer**. Instead they verify the pure translation in
:mod:`multicam_sim.dsl.kubric_spec`:

* the camera the backend builds projects a known off-axis 3D point to the **same**
  pixel as our analytic ``P = K[R|t]`` pipeline, within ``1e-6`` — and it does so
  by projecting *through Kubric's own camera parameterisation* (focal_length/mm +
  a ``(w, x, y, z)`` quaternion), not by carrying ``K`` verbatim, so the mm and
  rotation conversions are genuinely exercised;
* object positions and stable per-entity colours map correctly;
* a camera Kubric cannot represent (``fx != fy`` or off-centre principal point)
  is rejected loudly;
* importing the backend does not drag in ``kubric``.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim.cameras import Camera, Intrinsics
from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder
from multicam_sim.dsl.kubric_backend import KubricBackend
from multicam_sim.dsl.kubric_spec import (
    KubricCameraSpec,
    camera_to_kubric_spec,
    entity_color,
    scene_to_kubric_spec,
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


def _quaternion_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    """Kubric ``(w, x, y, z)`` unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project_via_kubric_spec(spec: KubricCameraSpec, point: np.ndarray) -> np.ndarray:
    """Project a world point the way Kubric/Blender will, from the spec alone.

    Rebuilds the pixel purely from the spec's mm fields + quaternion (the genuinely
    new conversion), independent of our ``K``/``R``/``t``:

    * ``fx = fy = focal_length / sensor_width * width`` (Kubric's intrinsics),
    * camera-to-world rotation from the ``(w, x, y, z)`` quaternion; its columns
      are the camera's right/up/back axes in world coordinates,
    * decompose ``point - position`` into (right, up, back) and apply the
      OpenCV<->Blender sign flips: ``u = cx - fx*a/c``, ``v = cy + fy*b/c`` where
      ``(a, b, c) = R_c2w.T @ (point - position)``. Blender looks down -Z, so a
      point in front has back-component ``c < 0``; the two negatives in
      ``u = cx - fx*a/c`` recover the OpenCV pixel (``cx + fx*x_cam/z_cam``).
    """
    fx = spec.focal_length / spec.sensor_width * spec.width
    fy = fx  # Kubric forces square pixels.
    cx = spec.width / 2.0
    cy = spec.height / 2.0

    rot_c2w = _quaternion_to_matrix(spec.quaternion)
    rel = point - np.asarray(spec.position, dtype=np.float64)
    a, b, c = rot_c2w.T @ rel  # right, up, back components
    u = cx - fx * a / c
    v = cy + fy * b / c
    return np.array([u, v], dtype=np.float64)


def test_kubric_backend_satisfies_protocol() -> None:
    assert isinstance(KubricBackend(), RendererBackend)


def test_importing_kubric_backend_does_not_import_kubric() -> None:
    import sys

    KubricBackend()
    scene_to_kubric_spec(_scene(), camera_id=0, frame=5)
    assert "kubric" not in sys.modules  # only imported inside .render()


def test_camera_spec_roundtrips_to_our_projection() -> None:
    """The Kubric camera projects a known off-axis point to the same pixel as P."""
    scene = _scene()
    # an off-axis, off-centre world point (NOT on any optical axis, so sign errors
    # do not cancel).
    point = np.array([0.7, -0.4, 0.9], dtype=np.float64)

    for cam in scene.cameras:
        uv_analytic, w = cam.project(point)
        assert w > 0.0  # in front for this scene

        spec = camera_to_kubric_spec(cam)
        uv_kubric = _project_via_kubric_spec(spec, point)

        assert np.allclose(uv_kubric, uv_analytic, atol=1e-6), (
            f"cam {cam.id}: kubric {uv_kubric} vs analytic {uv_analytic}"
        )


def test_focal_length_matches_intrinsics() -> None:
    scene = _scene()
    cam = scene.cameras[0]
    spec = camera_to_kubric_spec(cam)
    fx_reconstructed = spec.focal_length / spec.sensor_width * spec.width
    assert np.isclose(fx_reconstructed, cam.intrinsics.fx, atol=1e-9)
    assert np.isclose(fx_reconstructed, cam.intrinsics.fy, atol=1e-9)


def test_position_is_camera_centre() -> None:
    scene = _scene()
    for cam in scene.cameras:
        spec = camera_to_kubric_spec(cam)
        assert np.allclose(np.asarray(spec.position), cam.centre(), atol=1e-9)


def test_rejects_non_square_pixels() -> None:
    intr = Intrinsics(fx=800.0, fy=810.0, cx=32.0, cy=24.0, width=64, height=48)
    cam = Camera.look_at(
        id=0,
        intrinsics=intr,
        eye=np.array([4.0, 0.0, 1.5]),
        target=np.array([0.0, 0.0, 0.5]),
    )
    with pytest.raises(ValueError, match="square pixels"):
        camera_to_kubric_spec(cam)


def test_rejects_off_centre_principal_point() -> None:
    intr = Intrinsics(fx=800.0, fy=800.0, cx=30.0, cy=24.0, width=64, height=48)
    cam = Camera.look_at(
        id=0,
        intrinsics=intr,
        eye=np.array([4.0, 0.0, 1.5]),
        target=np.array([0.0, 0.0, 0.5]),
    )
    with pytest.raises(ValueError, match="centred principal point"):
        camera_to_kubric_spec(cam)


def test_objects_map_positions_and_stable_colours() -> None:
    scene = _scene()
    frame = 5
    spec = scene_to_kubric_spec(scene, camera_id=0, frame=frame)

    entity = scene.entities[0]
    match = next(f for f in entity.frames if f.frame == frame)
    assert len(spec.objects) == len(match.points)

    for obj in spec.objects:
        entity_id, point_name = obj.name.split("/", 1)
        assert entity_id == entity.id
        assert np.allclose(np.asarray(obj.position), match.points[point_name], atol=1e-12)
        # colour is stable and derived from the entity id.
        assert obj.color == entity_color(entity_id)
        assert all(0.0 <= c <= 1.0 for c in obj.color)


def test_colours_are_stable_and_distinct_per_entity() -> None:
    assert entity_color("obj") == entity_color("obj")
    assert entity_color("obj") != entity_color("person")


def test_multi_point_entity_lowers_one_sphere_per_point() -> None:
    """A multi-named-point entity (pose-shaped) yields one sphere per joint."""
    scene = _scene()
    entity = scene.entities[0]
    # synthesise a 3-point frame to prove the 1-point/17-point path is uniform.
    entity.frames[0].points = {
        "a": [0.0, 0.0, 0.5],
        "b": [0.1, 0.0, 0.5],
        "c": [0.0, 0.1, 0.5],
    }
    spec = scene_to_kubric_spec(scene, camera_id=0, frame=entity.frames[0].frame)
    names = {obj.name for obj in spec.objects}
    assert names == {"obj/a", "obj/b", "obj/c"}

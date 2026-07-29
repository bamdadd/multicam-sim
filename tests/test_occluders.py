"""Occluder geometry, composition, moving solids and scene round-trips."""

from __future__ import annotations

import numpy as np

from multicam_sim import (
    Camera,
    Cylinder,
    Intrinsics,
    PathOccluder,
    Scene,
    Sphere,
    build_manifest,
    build_smoke_scene,
)
from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder


def _cyl() -> Cylinder:
    # Unit cylinder along +Z, height 2 → caps at z=±1, radius 1.
    return Cylinder(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=1.0, height=2.0)


def test_cylinder_body_hit() -> None:
    c = _cyl()
    assert c.blocks_segment(np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))


def test_cylinder_radius_miss() -> None:
    c = _cyl()
    assert not c.blocks_segment(np.array([-3.0, 3.0, 0.0]), np.array([3.0, 3.0, 0.0]))


def test_cylinder_cap_miss() -> None:
    c = _cyl()
    # Parallel to XY, above the top cap (z=2 > 1).
    assert not c.blocks_segment(np.array([-3.0, 0.0, 2.0]), np.array([3.0, 0.0, 2.0]))


def test_cylinder_grazing_surface() -> None:
    c = _cyl()
    # Segment skims the surface at x=1, y=0 (exact radius).
    assert c.blocks_segment(np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    # Just outside.
    assert not c.blocks_segment(np.array([-3.0, 1.01, 0.0]), np.array([3.0, 1.01, 0.0]))


def test_cylinder_segment_fully_inside() -> None:
    c = _cyl()
    assert c.blocks_segment(np.array([-0.2, 0.0, 0.0]), np.array([0.2, 0.0, 0.0]))


# --------------------------------------------------------------------------- #
# Issue #32: richer occlusion regimes.
# --------------------------------------------------------------------------- #


def _ring_builder() -> SceneBuilder:
    """A standard 3-camera ring builder with one moving target entity."""
    return (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=800.0,
                width=640,
                height_px=480,
            )
        )
        .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
    )


def test_two_static_occluders_compose_as_union() -> None:
    """A point covered by the SECOND occluder is blocked; a first-only union
    would wrongly keep camera 1 visible. Camera 2 sees through both."""
    scene = (
        _ring_builder()
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=0).during((0, 10)))
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=1).during((0, 10)))
        .build()
    )
    manifest = build_manifest(scene)
    mid = manifest.entities[0].frames[5].points["center"].per_cam
    assert not mid[0].visible, "camera 0 must be blocked by the first occluder"
    assert not mid[1].visible, "camera 1 must be blocked by the second occluder"
    assert mid[2].visible, "camera 2 must see through both"


def test_path_occluder_at_frame_interpolates_keyframes() -> None:
    """A path occluder samples its trajectory and presents a static solid per frame."""
    po = PathOccluder(
        shape="sphere",
        size=0.1,
        keyframes=[
            {"frame": 0, "center": [0.0, 0.0, 0.0]},
            {"frame": 10, "center": [1.0, 0.0, 0.0]},
        ],
    )
    solid0 = po.at_frame(0)
    solid5 = po.at_frame(5)
    assert isinstance(solid0, Sphere)
    assert solid0.center == [0.0, 0.0, 0.0]
    assert isinstance(solid5, Sphere)
    assert np.allclose(np.asarray(solid5.center), [0.5, 0.0, 0.0])


def test_moving_occluder_blocks_exact_crossing_frames() -> None:
    """A sphere swept along camera 1's right axis blocks only the frame whose
    centre lies on that sightline; all other frames and cameras stay visible."""
    intr = Intrinsics.from_focal(800.0, 640, 480)
    cam1 = Camera.look_at(
        1,
        intr,
        np.array([-2.0, 2.0 * np.sqrt(3.0), 1.5]),
        np.array([0.0, 0.0, 0.5]),
    )
    point = np.array([0.0, 0.0, 0.5])
    offset = 0.15
    base = point + offset * (cam1.centre() - point)
    right = cam1.rotation()[0]
    right = right / np.linalg.norm(right)
    span = 0.5
    start = base - span * right
    end = base + span * right

    scene = (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(
            CameraRig.ring(
                n=3,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=800.0,
                width=640,
                height_px=480,
            )
        )
        .entity("obj", Path.linear((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)))
        .occlude(
            Occlusion.moving_sphere(
                size=0.08,
                path=Path.linear(tuple(start), tuple(end)),
            )
        )
        .build()
    )
    manifest = build_manifest(scene)
    blocked: list[int] = []
    for fr in manifest.entities[0].frames:
        per_cam = fr.points["center"].per_cam
        assert per_cam[0].visible, "camera 0 must stay visible"
        assert per_cam[2].visible, "camera 2 must stay visible"
        if not per_cam[1].visible:
            blocked.append(fr.frame)
    assert blocked == [5], f"expected only frame 5 blocked by the moving sphere, got {blocked}"


def test_existing_single_occluder_scene_round_trips_byte_identically() -> None:
    """The smoke scene (single static sphere) survives serialise → deserialise."""
    original = build_smoke_scene()
    json1 = original.model_dump_json(indent=2)
    restored = Scene.model_validate_json(json1)
    json2 = restored.model_dump_json(indent=2)
    assert json1 == json2
    assert original == restored


def test_multi_occluder_scene_round_trips() -> None:
    """A scene with two static occluders and a moving occluder round-trips."""
    scene = (
        _ring_builder()
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=0).during((0, 10)))
        .occlude(Occlusion.box(size=0.12).blocks(camera=1).during((0, 10)))
        .occlude(
            Occlusion.moving_sphere(
                size=0.08,
                path=Path.linear((-0.5, 0.0, 0.65), (0.5, 0.0, 0.65)),
            )
        )
        .build()
    )
    json1 = scene.model_dump_json(indent=2)
    restored = Scene.model_validate_json(json1)
    assert scene == restored

"""Occluder geometry, composition, and scene round-trips."""

from __future__ import annotations

import numpy as np

from multicam_sim import Cylinder, build_manifest
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
# Issue #32: multiple static occluders compose as a union.
# --------------------------------------------------------------------------- #


def test_two_static_occluders_compose_as_union() -> None:
    """A point covered by the SECOND occluder is blocked; a first-only union
    would wrongly keep camera 1 visible. Camera 2 sees through both."""
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
        .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=0).during((0, 10)))
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=1).during((0, 10)))
        .build()
    )
    manifest = build_manifest(scene)
    mid = manifest.entities[0].frames[5].points["center"].per_cam
    assert not mid[0].visible, "camera 0 must be blocked by the first occluder"
    assert not mid[1].visible, "camera 1 must be blocked by the second occluder"
    assert mid[2].visible, "camera 2 must see through both"

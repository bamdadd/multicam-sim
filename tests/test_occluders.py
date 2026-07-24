"""Cylinder occluder segment-blocking cases (issue #3)."""

from __future__ import annotations

import numpy as np

from multicam_sim import Cylinder


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

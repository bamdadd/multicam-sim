"""Unit checks on the mirrored camera convention and the occluder tests."""

from __future__ import annotations

import numpy as np

from multicam_sim import Box, Camera, Intrinsics, Sphere
from multicam_sim.geometry import (
    UP_WORLD,
    camera_centre,
    look_at_rotation,
)


def test_look_at_rotation_is_orthonormal_right_handed() -> None:
    eye = np.array([4.0, 0.0, 1.5])
    target = np.array([0.0, 0.0, 0.5])
    R = look_at_rotation(eye, target, UP_WORLD)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    # forward row (index 2) points from eye toward target
    forward = (target - eye) / np.linalg.norm(target - eye)
    assert np.allclose(R[2], forward, atol=1e-12)


def test_translation_recovers_centre() -> None:
    intr = Intrinsics.from_focal(800.0, 640, 480)
    eye = np.array([4.0, 0.0, 1.5])
    cam = Camera.look_at(0, intr, eye, np.array([0.0, 0.0, 0.5]))
    assert np.allclose(cam.centre(), eye, atol=1e-12)
    assert np.allclose(camera_centre(cam.rotation(), cam.translation()), eye, atol=1e-12)


def test_projection_of_centre_line_lands_on_principal_axis() -> None:
    intr = Intrinsics.from_focal(800.0, 640, 480)
    eye = np.array([4.0, 0.0, 1.5])
    target = np.array([0.0, 0.0, 0.5])
    cam = Camera.look_at(0, intr, eye, target)
    uv, w = cam.project(target)
    assert w > 0.0
    # the look-at target images at the principal point (cx, cy)
    assert np.allclose(uv, [intr.cx, intr.cy], atol=1e-9)


def test_sphere_blocks_segment_through_it() -> None:
    s = Sphere(center=[0.0, 0.0, 0.0], radius=1.0)
    assert s.blocks_segment(np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert not s.blocks_segment(np.array([-3.0, 3.0, 0.0]), np.array([3.0, 3.0, 0.0]))


def test_box_blocks_segment_through_it() -> None:
    b = Box(center=[0.0, 0.0, 0.0], half_extents=[1.0, 1.0, 1.0])
    assert b.blocks_segment(np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert not b.blocks_segment(np.array([-3.0, 5.0, 0.0]), np.array([3.0, 5.0, 0.0]))

"""Tests for the camera / intrinsics contract."""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim import Camera, Intrinsics


def test_camera_forward_is_unit_and_matches_look_at() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480)
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    cam = Camera.look_at(0, intr, eye, target, np.array([0.0, 0.0, 1.0]))
    forward = cam.forward()
    assert forward.shape == (3,)
    assert np.linalg.norm(forward) == pytest.approx(1.0, abs=1e-9)
    expected = (target - eye) / np.linalg.norm(target - eye)
    assert forward == pytest.approx(expected, abs=1e-9)


def test_from_fov_acceptance_criteria() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480)
    assert intr.fx == pytest.approx(320.0, abs=1e-9)
    assert intr.cx == pytest.approx(320.0, abs=1e-9)
    assert intr.cy == pytest.approx(240.0, abs=1e-9)


def test_from_fov_square_pixel_default() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480)
    assert intr.fx == pytest.approx(intr.fy, abs=1e-12)


def test_from_fov_with_vertical_fov() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480, fov_y_deg=60.0)
    assert intr.fx == pytest.approx(320.0, abs=1e-9)
    assert intr.fy == pytest.approx((240.0) / np.tan(np.radians(60.0) / 2.0), abs=1e-9)
    assert intr.fy != pytest.approx(intr.fx, abs=1e-6)


def test_from_fov_horizontal_edge_projects_to_border() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480)
    cam = Camera.look_at(
        0,
        intr,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    # A point on the left edge of the horizontal FOV at unit depth in front.
    # In this look-at orientation the camera +x axis points along world -y.
    point = np.array([1.0, -1.0, 0.0])
    uv, w = cam.project(point)
    assert w > 0.0
    assert uv[0] == pytest.approx(intr.width, abs=1e-9)
    assert uv[1] == pytest.approx(intr.cy, abs=1e-9)


def test_from_fov_vertical_edge_projects_to_border_with_custom_fov_y() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480, fov_y_deg=60.0)
    cam = Camera.look_at(
        0,
        intr,
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    # A point on the bottom edge of the vertical FOV at unit depth in front.
    point = np.array([1.0, 0.0, -np.tan(np.radians(60.0) / 2.0)])
    uv, w = cam.project(point)
    assert w > 0.0
    assert uv[0] == pytest.approx(intr.cx, abs=1e-9)
    assert uv[1] == pytest.approx(intr.height, abs=1e-9)


@pytest.mark.parametrize(
    ("fov_x", "fov_y", "width", "height", "match"),
    [
        (90.0, None, 0, 480, "width must be > 0"),
        (90.0, None, 640, -10, "height must be > 0"),
        (0.0, None, 640, 480, "fov_x_deg must be in"),
        (180.0, None, 640, 480, "fov_x_deg must be in"),
        (-1.0, None, 640, 480, "fov_x_deg must be in"),
        (90.0, 0.0, 640, 480, "fov_y_deg must be in"),
        (90.0, 180.0, 640, 480, "fov_y_deg must be in"),
    ],
)
def test_from_fov_validation(
    fov_x: float,
    fov_y: float | None,
    width: int,
    height: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Intrinsics.from_fov(fov_x, width, height, fov_y_deg=fov_y)


def test_fov_deg_round_trips_horizontal() -> None:
    intr = Intrinsics.from_fov(60.0, 640, 480)
    fov_x, _ = intr.fov_deg()
    assert fov_x == pytest.approx(60.0)


def test_fov_deg_square_pixel_matches_aspect() -> None:
    # Square pixels (fy == fx): the vertical FOV follows from the aspect ratio.
    intr = Intrinsics.from_fov(90.0, 640, 480)
    assert intr.fy == pytest.approx(intr.fx)
    fov_x, fov_y = intr.fov_deg()
    assert fov_x == pytest.approx(90.0)
    # width/height = 640/480, so vertical is narrower than horizontal.
    assert fov_y < fov_x


def test_fov_deg_round_trips_independent_axes() -> None:
    intr = Intrinsics.from_fov(90.0, 640, 480, fov_y_deg=60.0)
    fov_x, fov_y = intr.fov_deg()
    assert fov_x == pytest.approx(90.0)
    assert fov_y == pytest.approx(60.0)

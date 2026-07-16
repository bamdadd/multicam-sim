"""CameraRig.stations: per-camera pose, target, and intrinsics."""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim.dsl.rig import CameraRig, StationView


def test_stations_use_per_camera_pose_and_target() -> None:
    views = [
        StationView(position=(-6.0, -5.0, 1.5), look_at=(-6.0, 0.0, 1.0)),
        StationView(position=(6.0, -5.0, 1.9), look_at=(6.0, 0.0, 1.0)),
    ]
    cams = CameraRig.stations(views, width=640, height_px=480, fov_deg=55.0)
    assert len(cams) == 2
    # each camera centre is its own station position (t = -R@C round-trips)
    assert np.allclose(cams[0].centre(), [-6.0, -5.0, 1.5], atol=1e-9)
    assert np.allclose(cams[1].centre(), [6.0, -5.0, 1.9], atol=1e-9)
    # each looks at its OWN target -> that target images at the principal point
    uv0, w0 = cams[0].project(np.array([-6.0, 0.0, 1.0]))
    assert w0 > 0 and np.allclose(uv0, [cams[0].intrinsics.cx, cams[0].intrinsics.cy], atol=1e-9)


def test_stations_allow_per_camera_intrinsics() -> None:
    views = [
        StationView(position=(0.0, -5.0, 1.5), look_at=(0.0, 0.0, 1.0), fov_deg=90.0),
        StationView(position=(3.0, -5.0, 1.5), look_at=(3.0, 0.0, 1.0), focal=1200.0),
    ]
    cams = CameraRig.stations(views, width=640, height_px=480, fov_deg=55.0)
    # the wide station and the zoomed station get different focals
    assert cams[0].intrinsics.fx < cams[1].intrinsics.fx
    # zoomed station honoured its explicit focal
    assert np.isclose(cams[1].intrinsics.fx, 1200.0)


def test_station_view_rejects_two_intrinsics() -> None:
    view = StationView(position=(0.0, 0.0, 0.0), look_at=(1.0, 0.0, 0.0), focal=800.0, fov_deg=60.0)
    with pytest.raises(ValueError):
        CameraRig.stations([view], width=640, height_px=480)

"""CameraRig.grid: a planar rows x cols camera wall facing a target.

CPU-only, no renderer, no triangulation reader dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim.dsl import CameraRig


def test_grid_positions_face_target_and_preserve_extrinsic_convention() -> None:
    rows, cols = 2, 3
    corner = (-1.0, 0.0, 2.0)
    right = (1.0, 0.0, 0.0)
    down = (0.0, 0.0, -1.0)
    target = (0.5, 4.0, 0.0)
    cams = CameraRig.grid(
        rows=rows,
        cols=cols,
        corner=corner,
        right=right,
        down=down,
        look_at=target,
        focal=800.0,
        width=640,
        height_px=480,
    )
    assert len(cams) == rows * cols

    origin = np.array(corner, dtype=np.float64)
    right_vec = np.array(right, dtype=np.float64)
    down_vec = np.array(down, dtype=np.float64)
    target_arr = np.array(target, dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            cam = cams[r * cols + c]
            assert cam.id == r * cols + c
            expected_eye = origin + c * right_vec + r * down_vec
            # t = -R @ C: the stored extrinsics round-trip to the grid position.
            assert np.allclose(cam.centre(), expected_eye)
            # Every camera faces look_at: it projects near the principal point.
            uv, w = cam.project(target_arr)
            assert w > 0
            assert uv[0] == pytest.approx(cam.intrinsics.cx, abs=1e-6)
            assert uv[1] == pytest.approx(cam.intrinsics.cy, abs=1e-6)


def test_grid_supports_fov_and_overrides() -> None:
    from multicam_sim.dsl import PoseOverride

    override = PoseOverride(position=(9.0, 9.0, 9.0), look_at=(0.0, 0.0, 0.0))
    cams = CameraRig.grid(
        rows=1,
        cols=2,
        corner=(0.0, 0.0, 1.0),
        right=(1.0, 0.0, 0.0),
        down=(0.0, 1.0, 0.0),
        look_at=(0.0, 5.0, 1.0),
        fov_deg=60.0,
        width=640,
        height_px=480,
        overrides={1: override},
    )
    assert len(cams) == 2
    # Camera 0 keeps its computed grid pose; camera 1 uses the override position.
    assert np.allclose(cams[0].centre(), (0.0, 0.0, 1.0))
    assert np.allclose(cams[1].centre(), (9.0, 9.0, 9.0))


def test_grid_rejects_non_positive_dimensions() -> None:
    common = dict(
        corner=(0.0, 0.0, 0.0),
        right=(1.0, 0.0, 0.0),
        down=(0.0, 0.0, -1.0),
        look_at=(0.0, 1.0, 0.0),
        focal=800.0,
        width=640,
        height_px=480,
    )
    with pytest.raises(ValueError, match="rows >= 1"):
        CameraRig.grid(rows=0, cols=2, **common)
    with pytest.raises(ValueError, match="cols >= 1"):
        CameraRig.grid(rows=2, cols=0, **common)


def test_grid_requires_exactly_one_of_focal_or_fov() -> None:
    common = dict(
        rows=1,
        cols=1,
        corner=(0.0, 0.0, 0.0),
        right=(1.0, 0.0, 0.0),
        down=(0.0, 1.0, 0.0),
        look_at=(0.0, 1.0, 0.0),
        width=640,
        height_px=480,
    )
    with pytest.raises(ValueError, match="exactly one"):
        CameraRig.grid(**common)  # neither focal nor fov_deg
    with pytest.raises(ValueError, match="exactly one"):
        CameraRig.grid(focal=800.0, fov_deg=60.0, **common)

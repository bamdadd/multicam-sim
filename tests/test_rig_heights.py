"""Per-camera height for CameraRig.ring/line: distinct z de-degenerates geometry
and still round-trips ground truth through the REAL multicam-occlusion DLT.

CPU-only, no renderer.
"""

from __future__ import annotations

import numpy as np
import pytest

# the REAL consumer reader (dev path dependency on ../multicam-occlusion)
from multicam_occlusion.triangulation import triangulate_dlt

from multicam_sim.dsl import CameraRig


def _proj(cams: list) -> np.ndarray:
    return np.stack([c.projection_matrix() for c in cams])


def test_scalar_height_is_unchanged_behaviour() -> None:
    """Backward compat: a scalar height gives every camera the same eye z."""
    cams = CameraRig.ring(
        n=3,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    assert [c.centre()[2] for c in cams] == pytest.approx([1.5, 1.5, 1.5])


def test_per_camera_heights_give_distinct_extrinsics_and_triangulate() -> None:
    heights = [1.0, 1.8, 2.6]
    cams = CameraRig.ring(
        n=3,
        radius=4.0,
        height=heights,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    # eye z distinct -> extrinsics (t) distinct
    zs = [c.centre()[2] for c in cams]
    assert zs == pytest.approx(heights)
    ts = [tuple(c.t) for c in cams]
    assert len(set(ts)) == 3  # all three translations differ

    # project a known GT point through all three, triangulate back tightly.
    Ps = _proj(cams)
    gt = np.array([0.2, -0.1, 0.5], dtype=np.float64)
    uvs = np.array([c.project(gt)[0] for c in cams], dtype=np.float64)
    recovered = triangulate_dlt(Ps, uvs)
    assert np.allclose(recovered, gt, atol=1e-9, rtol=0.0)


def test_wrong_length_sequence_raises() -> None:
    with pytest.raises(ValueError, match="length 2, expected n=3"):
        CameraRig.ring(
            n=3,
            radius=4.0,
            height=[1.0, 2.0],
            look_at=(0.0, 0.0, 0.5),
            focal=800.0,
            width=640,
            height_px=480,
        )


def test_height_jitter_is_seeded_and_reproducible() -> None:
    kw = dict(
        n=4,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    a = CameraRig.ring(**kw, height_jitter=0.3, seed=7)
    b = CameraRig.ring(**kw, height_jitter=0.3, seed=7)
    c = CameraRig.ring(**kw, height_jitter=0.3, seed=8)
    za = [cam.centre()[2] for cam in a]
    zb = [cam.centre()[2] for cam in b]
    zc = [cam.centre()[2] for cam in c]
    assert za == pytest.approx(zb)  # same seed -> identical
    assert za != pytest.approx(zc)  # different seed -> different
    assert all(abs(z - 1.5) <= 0.3 + 1e-12 for z in za)  # within +/- jitter of base
    assert len(set(round(z, 9) for z in za)) == 4  # per-camera, not uniform


def test_line_height_override_and_backward_compat() -> None:
    base = dict(
        n=3,
        start=(-2.0, 3.0, 1.0),
        end=(2.0, 3.0, 1.0),
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    # default: z interpolated from start/end (both z=1.0 here)
    default = CameraRig.line(**base)
    assert [c.centre()[2] for c in default] == pytest.approx([1.0, 1.0, 1.0])
    # override: distinct per-camera z
    over = CameraRig.line(**base, height=[0.5, 1.5, 2.5])
    assert [c.centre()[2] for c in over] == pytest.approx([0.5, 1.5, 2.5])


def test_zero_jitter_is_identical_to_no_jitter() -> None:
    kw = dict(
        n=3,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    plain = CameraRig.ring(**kw)
    zeroed = CameraRig.ring(**kw, height_jitter=0.0, seed=99)
    for p, z in zip(plain, zeroed, strict=True):
        assert p.t == z.t

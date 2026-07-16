"""Behavioural coverage for the DSL breadth the vertical slice doesn't exercise:
CameraRig.line / .custom / .stereo / .arc and Occlusion.box / .plane.

CPU-only, no renderer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# the REAL consumer reader (dev path dependency on ../multicam-occlusion)
from multicam_occlusion.triangulation import triangulate_dlt

from multicam_sim import build_manifest
from multicam_sim.manifest import Manifest
from multicam_sim.cameras import Camera, Intrinsics
from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder


def _ring() -> list[Camera]:
    return CameraRig.ring(
        n=3,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )


def test_line_interpolates_eyes_and_matches_look_at() -> None:
    start, end, target = (-2.0, 3.0, 1.0), (2.0, 3.0, 1.0), (0.0, 0.0, 0.5)
    cams = CameraRig.line(
        n=3,
        start=start,
        end=end,
        look_at=target,
        focal=800.0,
        width=640,
        height_px=480,
    )
    intr = Intrinsics.from_focal(800.0, 640, 480)
    a, b = np.array(start), np.array(end)
    for i, cam in enumerate(cams):
        eye = a + (i / 2) * (b - a)  # 0, 0.5, 1.0
        ref = Camera.look_at(i, intr, eye, np.array(target))
        assert np.allclose(cam.rotation(), ref.rotation())
        assert np.allclose(cam.centre(), eye)  # centre round-trips through R/t


def test_custom_stores_extrinsics_verbatim() -> None:
    """custom bypasses look_at (verbatim R/t) — the centre must still round-trip."""
    ref = _ring()
    extr = [(c.R, c.t) for c in ref]
    cams = CameraRig.custom(extr, focal=800.0, width=640, height_px=480)
    for original, built in zip(ref, cams, strict=True):
        assert built.R == original.R
        assert built.t == original.t
        assert np.allclose(built.centre(), original.centre())


def _proj(cams: list[Camera]) -> np.ndarray:
    return np.stack([c.projection_matrix() for c in cams])


def test_stereo_returns_two_cameras_symmetric_about_center() -> None:
    cams = CameraRig.stereo(
        baseline=0.6,
        look_at=(0.0, 2.0, 0.5),
        height=1.5,
        focal=800.0,
        width=640,
        height_px=480,
    )
    assert len(cams) == 2
    assert cams[0].centre()[0] == pytest.approx(-0.3)
    assert cams[1].centre()[0] == pytest.approx(0.3)
    assert cams[0].centre()[1] == pytest.approx(0.0)
    assert cams[1].centre()[1] == pytest.approx(0.0)
    assert [c.centre()[2] for c in cams] == pytest.approx([1.5, 1.5])


def test_stereo_known_point_round_trips_through_manifest() -> None:
    cams = CameraRig.stereo(
        baseline=0.6,
        look_at=(0.0, 2.0, 0.5),
        height=1.5,
        focal=800.0,
        width=640,
        height_px=480,
    )
    gt = np.array([0.1, 1.0, 0.5], dtype=np.float64)
    uvs = np.array([c.project(gt)[0] for c in cams], dtype=np.float64)
    recovered = triangulate_dlt(_proj(cams), uvs)
    assert np.allclose(recovered, gt, atol=1e-9, rtol=0.0)


def _stereo_kwargs(**overrides: object) -> dict[str, object]:
    kw: dict[str, object] = dict(
        baseline=0.6,
        look_at=(0.0, 2.0, 0.5),
        height=1.5,
        focal=800.0,
        width=640,
        height_px=480,
    )
    kw.update(overrides)
    return kw


def test_stereo_validation() -> None:
    with pytest.raises(ValueError, match="baseline must be > 0"):
        CameraRig.stereo(**_stereo_kwargs(baseline=0.0))
    with pytest.raises(ValueError, match="look_at must differ from center"):
        CameraRig.stereo(**_stereo_kwargs(center=(0.0, 2.0, 0.5)))
    with pytest.raises(ValueError, match="parallel to world up"):
        CameraRig.stereo(**_stereo_kwargs(look_at=(0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="length 3, expected n=2"):
        CameraRig.stereo(**_stereo_kwargs(height=[1.0, 1.5, 2.0]))


def test_arc_returns_n_cameras_inclusive_of_endpoints() -> None:
    cams = CameraRig.arc(
        n=3,
        radius=4.0,
        start_angle=0.0,
        end_angle=math.pi / 2.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    assert len(cams) == 3
    expected_angles = [0.0, math.pi / 4.0, math.pi / 2.0]
    for cam, angle in zip(cams, expected_angles, strict=True):
        x, y, z = cam.centre()
        assert z == pytest.approx(1.5)
        assert math.atan2(y, x) == pytest.approx(angle)
        assert math.hypot(x, y) == pytest.approx(4.0)


def test_arc_known_point_round_trips_through_manifest() -> None:
    cams = CameraRig.arc(
        n=3,
        radius=4.0,
        start_angle=0.0,
        end_angle=math.pi / 2.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    gt = np.array([0.2, 0.2, 0.5], dtype=np.float64)
    uvs = np.array([c.project(gt)[0] for c in cams], dtype=np.float64)
    recovered = triangulate_dlt(_proj(cams), uvs)
    assert np.allclose(recovered, gt, atol=1e-9, rtol=0.0)


def _arc_kwargs(**overrides: object) -> dict[str, object]:
    kw: dict[str, object] = dict(
        n=3,
        radius=4.0,
        start_angle=0.0,
        end_angle=math.pi / 2.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    kw.update(overrides)
    return kw


def test_arc_validation() -> None:
    with pytest.raises(ValueError, match="arc needs n >= 1 cameras"):
        CameraRig.arc(**_arc_kwargs(n=0))
    with pytest.raises(ValueError, match="arc radius must be > 0"):
        CameraRig.arc(**_arc_kwargs(radius=0.0))


def test_arc_over_adapted_span_matches_ring() -> None:
    """Verified equivalence: arc over [0, 2*pi*(n-1)/n] equals ring()."""
    n = 4
    end = 2.0 * math.pi * (n - 1) / n
    ring_cams = CameraRig.ring(
        n=n,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    arc_cams = CameraRig.arc(
        n=n,
        radius=4.0,
        start_angle=0.0,
        end_angle=end,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    for r, a in zip(ring_cams, arc_cams, strict=True):
        assert np.allclose(r.centre(), a.centre())
        assert np.allclose(r.rotation(), a.rotation())
        assert r.t == a.t


def _scene_with(occ: Occlusion) -> Manifest:
    scene = (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(_ring())
        .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
        .occlude(occ)
        .build()
    )
    return build_manifest(scene)


def _occluded_frames(manifest: Manifest) -> list[int]:
    out = []
    for fr in manifest.entities[0].frames:
        vis = [o.visible for o in fr.points["center"].per_cam]
        if not vis[1]:
            assert vis[0] and vis[2], "occluder must stay selective to camera 1"
            out.append(fr.frame)
    return out


@pytest.mark.parametrize("shape", ["box", "plane"])
def test_box_and_plane_occluders_block_the_target_camera(shape: str) -> None:
    factory = Occlusion.box if shape == "box" else Occlusion.plane
    manifest = _scene_with(factory(size=0.2).blocks(camera=1).during((3, 7)))
    occluded = _occluded_frames(manifest)
    assert occluded, f"{shape} occluder should block camera 1 in a middle interval"
    assert 0 < len(occluded) < manifest.num_frames

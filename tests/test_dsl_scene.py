"""DSL end-to-end: a fluent scene compiles to a correct manifest and recovers
ground truth through the REAL multicam-occlusion DLT.

CPU-only, no renderer imported anywhere in this path.
"""

from __future__ import annotations

import numpy as np
import pytest

# the REAL consumer reader (dev path dependency on ../multicam-occlusion)
from multicam_occlusion.triangulation import triangulate_dlt

from multicam_sim import build_manifest
from multicam_sim.cameras import Camera
from multicam_sim.dsl import CameraRig, Occlusion, Path, SceneBuilder


def _dsl_smoke_scene() -> SceneBuilder:
    """A smoke-equivalent scene expressed entirely through the DSL."""
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
        .occlude(Occlusion.sphere(size=0.15).blocks(camera=1).during((3, 7)))
    )


def _proj_mats(manifest: dict) -> np.ndarray:
    mats = []
    for cam in manifest["cameras"]:
        assert cam["convention"] == "opencv_rdf"
        K = np.array(cam["K"], dtype=np.float64)
        R = np.array(cam["R"], dtype=np.float64)
        t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
        mats.append(K @ np.hstack([R, t]))
    return np.stack(mats)


def test_rig_matches_look_at_convention() -> None:
    """CameraRig.ring must equal hand-built Camera.look_at cameras (no drift)."""
    from multicam_sim.cameras import Intrinsics

    cams = CameraRig.ring(
        n=3,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=640,
        height_px=480,
    )
    intr = Intrinsics.from_focal(800.0, 640, 480)
    for i, cam in enumerate(cams):
        angle = 2.0 * np.pi * i / 3.0
        eye = np.array([4.0 * np.cos(angle), 4.0 * np.sin(angle), 1.5])
        ref = Camera.look_at(i, intr, eye, np.array([0.0, 0.0, 0.5]))
        assert np.allclose(cam.rotation(), ref.rotation())
        assert np.allclose(cam.translation(), ref.translation())


def test_fov_and_focal_are_consistent() -> None:
    # horizontal fov -> focal = (W/2)/tan(fov/2); a 640px sensor @ ~43.6deg -> 800px.
    fov = float(2.0 * np.degrees(np.arctan((640 / 2.0) / 800.0)))
    by_fov = CameraRig.ring(
        n=1,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        fov_deg=fov,
        width=640,
        height_px=480,
    )[0]
    assert by_fov.intrinsics.fx == pytest.approx(800.0, rel=1e-9)
    with pytest.raises(ValueError):  # exactly one of focal/fov_deg
        CameraRig.ring(
            n=1,
            radius=4.0,
            height=1.5,
            look_at=(0.0, 0.0, 0.5),
            focal=800.0,
            fov_deg=fov,
            width=640,
            height_px=480,
        )


def test_dsl_scene_shape_and_gt() -> None:
    manifest = build_manifest(_dsl_smoke_scene().build())
    assert len(manifest["cameras"]) == 3
    assert manifest["num_frames"] == 11
    frames = manifest["entities"][0]["frames"]
    assert frames[0]["points"]["center"]["xyz_gt"] == pytest.approx([0.0, -0.6, 0.5])
    assert frames[-1]["points"]["center"]["xyz_gt"] == pytest.approx([0.0, 0.6, 0.5])


def test_occlusion_is_selective_and_windowed() -> None:
    """The scheduled sphere blocks camera 1 in a genuine middle interval while
    cameras 0 and 2 keep the point — visibility is emergent geometry, asserted
    from the actual manifest (not assumed equal to the requested window)."""
    manifest = build_manifest(_dsl_smoke_scene().build())
    frames = manifest["entities"][0]["frames"]
    occluded = []
    for fr in frames:
        vis = [o["visible"] for o in fr["points"]["center"]["per_cam"]]
        if not vis[1]:
            occluded.append(fr["frame"])
            assert vis[0] and vis[2], "other cameras must keep the point (selectivity)"
            assert fr["points"]["center"]["per_cam"][1]["occ_frac"] > 0.0
    assert occluded, "expected a middle interval where camera 1 is occluded"
    assert 0 < len(occluded) < len(frames)


def test_dsl_recovers_occluded_point_through_real_dlt() -> None:
    """The whole point: a DSL-authored occluded frame triangulates to GT from the
    two remaining views through the real multicam-occlusion DLT."""
    manifest = build_manifest(_dsl_smoke_scene().build())
    Ps = _proj_mats(manifest)
    tested = False
    for fr in manifest["entities"][0]["frames"]:
        per_cam = fr["points"]["center"]["per_cam"]
        mask = np.array([o["visible"] for o in per_cam], dtype=bool)
        if mask[1]:
            continue
        tested = True
        assert mask.sum() == 2
        uvs = np.array([o["uv"] for o in per_cam], dtype=np.float64)
        gt = np.array(fr["points"]["center"]["xyz_gt"], dtype=np.float64)
        recovered = triangulate_dlt(Ps, uvs, mask=mask)
        assert np.allclose(recovered, gt, atol=1e-6, rtol=0.0)
    assert tested, "expected at least one cam-1-occluded frame"


def test_occlusion_coverage_is_monotonic_on_occ_frac() -> None:
    """The difficulty knob is monotonic: more coverage -> occ_frac never drops.
    occ_frac is a measured readback (manifest quantises to eighths), never the mask."""

    def mid_occ_frac(coverage: float) -> float:
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
            .occlude(
                Occlusion.sphere(size=0.15).targeting(coverage).blocks(camera=1).during((5, 5))
            )
            .build()
        )
        m = build_manifest(scene)
        return m["entities"][0]["frames"][5]["points"]["center"]["per_cam"][1]["occ_frac"]

    fracs = [mid_occ_frac(c) for c in (0.5, 1.0, 2.0)]
    assert fracs == sorted(fracs)  # monotonic non-decreasing
    assert fracs[-1] > 0.0

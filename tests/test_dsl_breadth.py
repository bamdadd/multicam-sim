"""Behavioural coverage for the DSL breadth the vertical slice doesn't exercise:
CameraRig.line / .custom and Occlusion.box / .plane.

CPU-only, no renderer.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim import build_manifest
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


def _scene_with(occ: Occlusion) -> dict:
    scene = (
        SceneBuilder(fps=30.0, num_frames=11)
        .cameras(_ring())
        .entity("obj", Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5)))
        .occlude(occ)
        .build()
    )
    return build_manifest(scene)


def _occluded_frames(manifest: dict) -> list[int]:
    out = []
    for fr in manifest["entities"][0]["frames"]:
        vis = [o["visible"] for o in fr["points"]["center"]["per_cam"]]
        if not vis[1]:
            assert vis[0] and vis[2], "occluder must stay selective to camera 1"
            out.append(fr["frame"])
    return out


@pytest.mark.parametrize("shape", ["box", "plane"])
def test_box_and_plane_occluders_block_the_target_camera(shape: str) -> None:
    factory = Occlusion.box if shape == "box" else Occlusion.plane
    manifest = _scene_with(factory(size=0.2).blocks(camera=1).during((3, 7)))
    occluded = _occluded_frames(manifest)
    assert occluded, f"{shape} occluder should block camera 1 in a middle interval"
    assert 0 < len(occluded) < manifest["num_frames"]

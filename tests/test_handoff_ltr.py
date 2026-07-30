"""Left-to-right cross-camera handoff scene.

One object crosses the space in a straight line (world ``-x -> +x``) under four
cameras: a LEFT camera that sees it first and loses it, two RIGHT cameras
(mid-right, right) that pick it up as it enters — with overlap bands where two
near cameras see it at once — and a WIDE camera off the far-right end, looking
down the length of the line toward the far left, that sees the object on EVERY
frame. Coverage is read straight off the manifest's per-camera ``in_view`` flags
(the sim's own projection/visibility), never recomputed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from multicam_sim import build_handoff_ltr_scene, build_manifest, write_manifest
from multicam_sim.handoff_ltr import (
    CAM_LEFT,
    CAM_MID_RIGHT,
    CAM_RIGHT,
    CAM_WIDE,
    OBJECT_ID,
)


def _load(tmp_path: Path) -> dict:
    scene = build_handoff_ltr_scene()
    path = tmp_path / "handoff_ltr.json"
    write_manifest(scene, path)
    return json.loads(path.read_text())


def _frames(manifest: dict) -> list[dict]:
    return manifest["entities"][0]["frames"]


def _in_view(frame: dict) -> list[bool]:
    return [o["in_view"] for o in frame["points"]["center"]["per_cam"]]


def test_deterministic_byte_stable_manifest() -> None:
    """Two independent builds serialise to the exact same JSON bytes."""
    first = build_manifest(build_handoff_ltr_scene()).to_json()
    second = build_manifest(build_handoff_ltr_scene()).to_json()
    assert first == second


def test_manifest_is_strict_finite_json(tmp_path: Path) -> None:
    """Every projected uv is finite (valid strict JSON, no Infinity/NaN)."""
    manifest = _load(tmp_path)
    for frame in _frames(manifest):
        for obs in frame["points"]["center"]["per_cam"]:
            assert all(math.isfinite(c) for c in obs["uv"])


def test_object_moves_left_to_right(tmp_path: Path) -> None:
    """The object's world x is monotonically non-decreasing across the frames."""
    manifest = _load(tmp_path)
    xs = [f["points"]["center"]["xyz_gt"][0] for f in _frames(manifest)]
    assert xs[0] < xs[-1], "object should travel toward +x"
    assert all(b >= a for a, b in zip(xs[:-1], xs[1:], strict=True))


def test_wide_camera_sees_object_every_frame(tmp_path: Path) -> None:
    """The far-right wide camera (looking down the line to the far left) is the
    always-on reference: the object is in_view on it on EVERY frame."""
    manifest = _load(tmp_path)
    frames = _frames(manifest)
    assert all(_in_view(f)[CAM_WIDE] for f in frames)


def test_left_camera_sees_then_loses_the_object(tmp_path: Path) -> None:
    """The LEFT camera sees the object at the start and loses it once the object
    has moved right (a true->false transition, and it does not come back)."""
    manifest = _load(tmp_path)
    seen = [_in_view(f)[CAM_LEFT] for f in _frames(manifest)]
    assert seen[0], "left camera should see the object at the start"
    assert not seen[-1], "left camera should have lost the object by the end"
    # One contiguous seen-then-gone run: after the first drop it never returns.
    last_seen = max(i for i, v in enumerate(seen) if v)
    assert all(not v for v in seen[last_seen + 1 :])
    assert all(seen[: last_seen + 1]), "left coverage is one contiguous run"


def test_right_cameras_pick_the_object_up(tmp_path: Path) -> None:
    """The two RIGHT cameras enter after the start (the object is NOT in either at
    frame 0) and hold it at the end, and the right camera is the final holder."""
    manifest = _load(tmp_path)
    frames = _frames(manifest)
    first = _in_view(frames[0])
    last = _in_view(frames[-1])
    assert not first[CAM_MID_RIGHT] and not first[CAM_RIGHT], (
        "right cameras should not yet see the object at the start"
    )
    assert last[CAM_RIGHT], "the right camera should hold the object at the end"


def test_handoff_has_overlap_and_no_near_camera_blind_gap(tmp_path: Path) -> None:
    """The handoff is smooth: on some frames two NEAR cameras (left/mid/right, not
    the wide reference) see the object at once (overlap bands), and there is never
    a frame where NO near camera sees it (no blind gap in the near coverage)."""
    manifest = _load(tmp_path)
    near = (CAM_LEFT, CAM_MID_RIGHT, CAM_RIGHT)
    near_counts = [sum(_in_view(f)[c] for c in near) for f in _frames(manifest)]
    assert max(near_counts) >= 2, "expected an overlap band of >=2 near cameras"
    assert min(near_counts) >= 1, "no near-camera blind gap along the path"


def test_stable_identity(tmp_path: Path) -> None:
    """One object with a stable id across every frame."""
    manifest = _load(tmp_path)
    assert len(manifest["entities"]) == 1
    assert manifest["entities"][0]["id"] == OBJECT_ID

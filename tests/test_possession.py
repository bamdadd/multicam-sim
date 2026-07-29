"""Carried-object attachment (#64).

Build-time baking: an object entity's ``center`` point is driven by a holder
entity over a half-open frame interval ``[start, end)``, then detaches. A
possession-GT sidecar records the holder timeline and round-trips through JSON
without touching the byte-golden analytic manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from multicam_sim import build_manifest, build_smoke_scene
from multicam_sim.dsl import CameraRig, SceneBuilder
from multicam_sim.dsl import Path as MotionPath
from multicam_sim.possession import PossessionSegment, PossessionTimeline
from multicam_sim.scene import Scene

_GOLDEN = Path(__file__).parent / "fixtures" / "manifest_golden"


def _same_shape(got: object, ref: object, path: str = "") -> None:
    """Assert identical JSON structure — key order, list length, field presence,
    and value types. Float values are compared with a tolerance (mirrors
    ``test_manifest_golden``).
    """
    assert type(got) is type(ref), f"type drift at {path or '<root>'}: {type(got)} != {type(ref)}"
    if isinstance(ref, dict):
        assert isinstance(got, dict)
        assert list(got) == list(ref), f"key set/order drift at {path or '<root>'}"
        for k in ref:
            _same_shape(got[k], ref[k], f"{path}.{k}")
    elif isinstance(ref, list):
        assert isinstance(got, list)
        assert len(got) == len(ref), f"length drift at {path or '<root>'}: {len(got)} != {len(ref)}"
        for i, (g, r) in enumerate(zip(got, ref, strict=True)):
            _same_shape(g, r, f"{path}[{i}]")
    elif isinstance(ref, float):
        assert isinstance(got, float)
        assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-12), (
            f"float drift at {path}: {got} != {ref}"
        )
    else:
        assert got == ref, f"value drift at {path}: {got!r} != {ref!r}"


def _carried_scene(*, attach: bool = True) -> Scene:
    """Holder moves one metre/frame along +x; object is far away otherwise.

    With ``attach=True`` the object is carried by the holder from frame 2 up to
    (but not including) frame 7, with a fixed (0, 1, 0) offset.
    """
    builder = (
        SceneBuilder(fps=10.0, num_frames=10)
        .cameras(
            CameraRig.ring(
                n=2,
                radius=5.0,
                height=1.0,
                look_at=(0.0, 0.0, 0.0),
                focal=800.0,
                width=640,
                height_px=480,
            )
        )
        .entity("holder", MotionPath.linear((0.0, 0.0, 0.0), (9.0, 0.0, 0.0)))
        .entity("object", MotionPath.linear((100.0, 100.0, 100.0), (100.0, 100.0, 100.0)))
    )
    if attach:
        builder.attach(
            "object",
            "holder",
            start=2,
            end=7,
            offset=(0.0, 1.0, 0.0),
        )
    return builder.build()


def test_attach_follow_detach_geometry() -> None:
    """While attached the object tracks holder + offset; after detach it resumes
    its own path."""
    scene = _carried_scene(attach=True)
    holder = next(e for e in scene.entities if e.id == "holder")
    obj = next(e for e in scene.entities if e.id == "object")

    holder_center = {f.frame: f.points["center"] for f in holder.frames}
    obj_center = {f.frame: f.points["center"] for f in obj.frames}

    for f in range(10):
        if 2 <= f < 7:
            expected = [
                holder_center[f][0] + 0.0,
                holder_center[f][1] + 1.0,
                holder_center[f][2] + 0.0,
            ]
            assert obj_center[f] == pytest.approx(expected)
        else:
            # Object must NOT track the holder outside [2, 7).
            assert obj_center[f] == pytest.approx([100.0, 100.0, 100.0])

    # Specific spot-checks for readability.
    assert obj_center[1] == pytest.approx([100.0, 100.0, 100.0])
    assert obj_center[2] == pytest.approx([2.0, 1.0, 0.0])
    assert obj_center[6] == pytest.approx([6.0, 1.0, 0.0])
    assert obj_center[7] == pytest.approx([100.0, 100.0, 100.0])


def test_possession_timeline_half_open_boundary() -> None:
    """Possession records holder id on [start, end) and None from end onward,
    with the detach frame explicit in the segment.
    """
    scene = _carried_scene(attach=True)
    assert scene.possession is not None
    timeline = scene.possession

    assert timeline.segments == [
        PossessionSegment(
            object_id="object",
            holder_id="holder",
            start_frame=2,
            end_frame=7,
        )
    ]

    # Detached before the interval.
    assert timeline.holder_at_frame("object", 0) is None
    assert timeline.holder_at_frame("object", 1) is None
    # Attached throughout the half-open interval.
    for f in range(2, 7):
        assert timeline.holder_at_frame("object", f) == "holder"
    # Detached exactly at end_frame and after.
    assert timeline.holder_at_frame("object", 7) is None
    assert timeline.holder_at_frame("object", 8) is None
    assert timeline.holder_at_frame("object", 9) is None

    # Unknown object is always detached.
    assert timeline.holder_at_frame("unknown", 4) is None


def test_scene_roundtrips_with_and_without_possession_sidecar() -> None:
    """A Scene serialises and deserialises identically, both with and without
    the optional possession sidecar."""
    attached = _carried_scene(attach=True)
    plain = _carried_scene(attach=False)

    for scene in (attached, plain):
        json_text = scene.model_dump_json(indent=2)
        restored = Scene.model_validate_json(json_text)
        assert restored.fps == scene.fps
        assert restored.num_frames == scene.num_frames
        assert [e.id for e in restored.entities] == [e.id for e in scene.entities]
        assert restored.possession == scene.possession

        # The sidecar, when present, round-trips through its own to_json.
        if scene.possession is not None:
            sidecar_json = scene.possession.to_json()
            sidecar_restored = PossessionTimeline.model_validate_json(sidecar_json)
            assert sidecar_restored == scene.possession


def test_manifest_unchanged_without_attachment() -> None:
    """Scenes that do not use the attachment helper keep the byte-golden
    analytic manifest unchanged (compared to the smoke golden fixture).
    """
    got = build_manifest(build_smoke_scene()).to_json()
    ref = (_GOLDEN / "smoke.json").read_text()
    _same_shape(json.loads(got), json.loads(ref))


def test_manifest_excludes_possession_sidecar() -> None:
    """The analytic manifest never contains possession GT, even when the scene
    carries the sidecar."""
    scene = _carried_scene(attach=True)
    manifest_json = build_manifest(scene).to_json()
    assert "possession" not in manifest_json
    assert "PossessionSegment" not in manifest_json

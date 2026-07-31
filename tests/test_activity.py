"""Activity-state ground-truth channel (#65).

A typed per-entity activity label (``standing`` / ``crouching`` / ``reaching``)
over half-open frame intervals, riding in an additive ``activity.json`` sidecar
following the ``order.py`` / ``possession.py`` precedent. The byte-golden
analytic manifest is unchanged unless the channel is opted into — and even then
the labels never enter the manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from multicam_sim import build_manifest, build_smoke_scene
from multicam_sim.activity import (
    ActivitySegment,
    ActivityState,
    ActivityTimeline,
    write_activity_json,
)
from multicam_sim.dsl import CameraRig, SceneBuilder
from multicam_sim.dsl import Path as MotionPath
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


def _labeled_scene(*, label: bool = True) -> Scene:
    """Two moving entities; with ``label=True``, ``operator`` is standing over
    [0, 4), crouching over [4, 7), reaching over [7, 10); ``walker`` is
    standing over [2, 5).
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
        .entity("operator", MotionPath.linear((0.0, 0.0, 0.0), (9.0, 0.0, 0.0)))
        .entity("walker", MotionPath.linear((0.0, 5.0, 0.0), (9.0, 5.0, 0.0)))
    )
    if label:
        builder.activity("operator", ActivityState.standing, 0, 4)
        builder.activity("operator", ActivityState.crouching, 4, 7)
        builder.activity("operator", ActivityState.reaching, 7, 10)
        builder.activity("walker", ActivityState.standing, 2, 5)
    return builder.build()


def test_activity_state_serialises_as_plain_string() -> None:
    """The label is str-backed: its JSON value is the plain state string, so
    adding a state later is a new enum member and a new string — additive,
    no schema fork."""
    assert ActivityState.standing == "standing"
    assert (
        json.loads(
            ActivitySegment(
                entity_id="e", state=ActivityState.reaching, start_frame=0, end_frame=3
            ).model_dump_json()
        )["state"]
        == "reaching"
    )


def test_segment_rejects_invalid_windows() -> None:
    """Frame indices must be >= 0 and end strictly greater than start."""
    with pytest.raises(ValueError, match=">= 0"):
        ActivitySegment(entity_id="e", state=ActivityState.standing, start_frame=-1, end_frame=3)
    with pytest.raises(ValueError, match="strictly greater"):
        ActivitySegment(entity_id="e", state=ActivityState.standing, start_frame=3, end_frame=3)
    with pytest.raises(ValueError, match="strictly greater"):
        ActivitySegment(entity_id="e", state=ActivityState.standing, start_frame=5, end_frame=3)


def test_timeline_sorts_segments_and_rejects_overlap() -> None:
    """Segments come out sorted by (entity_id, start_frame); overlapping
    intervals for the SAME entity are rejected, disjoint entities may overlap
    in time freely."""
    timeline = ActivityTimeline(
        segments=[
            ActivitySegment(
                entity_id="b", state=ActivityState.standing, start_frame=0, end_frame=5
            ),
            ActivitySegment(
                entity_id="a", state=ActivityState.crouching, start_frame=6, end_frame=9
            ),
            ActivitySegment(
                entity_id="a", state=ActivityState.standing, start_frame=0, end_frame=3
            ),
        ]
    )
    assert [(s.entity_id, s.start_frame) for s in timeline.segments] == [
        ("a", 0),
        ("a", 6),
        ("b", 0),
    ]

    with pytest.raises(ValueError, match="overlapping activity segments"):
        ActivityTimeline(
            segments=[
                ActivitySegment(
                    entity_id="a", state=ActivityState.standing, start_frame=0, end_frame=5
                ),
                ActivitySegment(
                    entity_id="a", state=ActivityState.crouching, start_frame=4, end_frame=8
                ),
            ]
        )


def test_state_at_frame_half_open_boundaries() -> None:
    """The timeline answers per-entity per-frame queries: the state holds over
    [start, end), frames outside all segments are unlabeled (None), and an
    unknown entity is always unlabeled."""
    scene = _labeled_scene(label=True)
    assert scene.activity is not None
    timeline = scene.activity

    # Contiguous intervals hand over exactly at the boundary frame.
    for f in range(0, 4):
        assert timeline.state_at_frame("operator", f) == ActivityState.standing
    for f in range(4, 7):
        assert timeline.state_at_frame("operator", f) == ActivityState.crouching
    for f in range(7, 10):
        assert timeline.state_at_frame("operator", f) == ActivityState.reaching

    # walker is labeled only on [2, 5); unlabeled before and from end_frame on.
    assert timeline.state_at_frame("walker", 1) is None
    assert timeline.state_at_frame("walker", 2) == ActivityState.standing
    assert timeline.state_at_frame("walker", 4) == ActivityState.standing
    assert timeline.state_at_frame("walker", 5) is None
    assert timeline.state_at_frame("walker", 9) is None

    # Unknown entity is always unlabeled.
    assert timeline.state_at_frame("unknown", 3) is None


def test_builder_validates_activity_window_and_entity() -> None:
    """The opt-in hook rejects out-of-range windows immediately and unknown
    entity ids at build time (mirroring ``attach``)."""
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
        .entity("operator", MotionPath.linear((0.0, 0.0, 0.0), (9.0, 0.0, 0.0)))
    )
    with pytest.raises(ValueError, match="invalid activity window"):
        builder.activity("operator", ActivityState.standing, -1, 5)
    with pytest.raises(ValueError, match="invalid activity window"):
        builder.activity("operator", ActivityState.standing, 5, 11)
    with pytest.raises(ValueError, match="invalid activity window"):
        builder.activity("operator", ActivityState.standing, 5, 5)

    builder.activity("ghost", ActivityState.standing, 0, 5)
    with pytest.raises(ValueError, match="unknown entity 'ghost'"):
        builder.build()


def test_scene_roundtrips_with_and_without_activity_sidecar() -> None:
    """A Scene serialises and deserialises identically, both with and without
    the optional activity sidecar."""
    labeled = _labeled_scene(label=True)
    plain = _labeled_scene(label=False)
    assert plain.activity is None

    for scene in (labeled, plain):
        json_text = scene.model_dump_json(indent=2)
        restored = Scene.model_validate_json(json_text)
        assert restored.fps == scene.fps
        assert restored.num_frames == scene.num_frames
        assert [e.id for e in restored.entities] == [e.id for e in scene.entities]
        assert restored.activity == scene.activity

        # The sidecar, when present, round-trips through its own to_json.
        if scene.activity is not None:
            sidecar_restored = ActivityTimeline.model_validate_json(scene.activity.to_json())
            assert sidecar_restored == scene.activity


def test_write_activity_json(tmp_path: Path) -> None:
    """The sidecar writes to ``activity.json`` and returns the dumped dict."""
    timeline = _labeled_scene(label=True).activity
    assert timeline is not None
    path = tmp_path / "activity.json"
    data = write_activity_json(timeline, path)
    assert json.loads(path.read_text()) == data
    assert data["segments"][0] == {
        "entity_id": "operator",
        "state": "standing",
        "start_frame": 0,
        "end_frame": 4,
    }


def test_manifest_unchanged_without_opt_in() -> None:
    """Scenes that do not opt into the activity channel keep the byte-golden
    analytic manifest unchanged (compared to the smoke golden fixture).
    """
    got = build_manifest(build_smoke_scene()).to_json()
    ref = (_GOLDEN / "smoke.json").read_text()
    _same_shape(json.loads(got), json.loads(ref))


def test_manifest_byte_identical_with_and_without_opt_in() -> None:
    """Opting into the activity channel changes NOTHING in the manifest: the
    serialized manifest bytes of the labeled and unlabeled scene are exactly
    equal."""
    labeled = build_manifest(_labeled_scene(label=True)).to_json().encode()
    plain = build_manifest(_labeled_scene(label=False)).to_json().encode()
    assert labeled == plain


def test_manifest_excludes_activity_sidecar() -> None:
    """The analytic manifest never contains activity GT, even when the scene
    carries the sidecar."""
    scene = _labeled_scene(label=True)
    manifest_json = build_manifest(scene).to_json()
    assert "activity" not in manifest_json
    assert "ActivitySegment" not in manifest_json

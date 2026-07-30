"""Two-agent interaction / object hand-off (#63).

Two agents on converging-then-diverging paths exchange one object at a defined
hand-off frame. The ground truth rides in the possession sidecar (#73): the
per-frame holder segments plus an :class:`InteractionEvent` (frame/time, giver,
receiver, object) — additive, never in the byte-golden analytic manifest.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from multicam_sim import build_manifest
from multicam_sim.dsl import CameraRig, SceneBuilder
from multicam_sim.dsl import Path as MotionPath
from multicam_sim.possession import InteractionEvent, PossessionSegment, PossessionTimeline
from multicam_sim.scene import Scene

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "handoff.py"

FPS = 10.0
NUM_FRAMES = 31
PICKUP_FRAME = 3
HANDOFF_FRAME = 15
OFFSET = (0.0, 0.0, 0.3)
STAGING = (-3.2, -2.4, 0.7)


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("handoff", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _load_example()


def _handoff_scene() -> Scene:
    """Giver (SW) and receiver (SE) converge to the origin at the hand-off
    frame, then diverge; the parcel is unheld until the pick-up frame."""
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(
            CameraRig.ring(
                n=2,
                radius=8.0,
                height=3.0,
                look_at=(0.0, 0.0, 1.0),
                focal=800.0,
                width=640,
                height_px=480,
            )
        )
        .entity(
            "giver", MotionPath.waypoints([(-4.0, -3.0, 1.0), (0.0, 0.0, 1.0), (-4.0, 3.0, 1.0)])
        )
        .entity(
            "receiver", MotionPath.waypoints([(4.0, -3.0, 1.0), (0.0, 0.0, 1.0), (4.0, 3.0, 1.0)])
        )
        .entity("parcel", MotionPath.linear(STAGING, STAGING))
        .handoff("parcel", "giver", "receiver", HANDOFF_FRAME, start=PICKUP_FRAME, offset=OFFSET)
        .build()
    )


def _center(scene: Scene, entity_id: str) -> dict[int, list[float]]:
    entity = next(e for e in scene.entities if e.id == entity_id)
    return {f.frame: f.points["center"] for f in entity.frames}


def test_scene_has_two_agents_and_one_object() -> None:
    scene = _handoff_scene()
    assert [e.id for e in scene.entities] == ["giver", "receiver", "parcel"]
    # The object is a plain entity with a single center point, no skeleton.
    parcel = next(e for e in scene.entities if e.id == "parcel")
    assert parcel.edges is None
    assert parcel.point_names() == {"center"}


def test_agents_converge_then_diverge() -> None:
    """Agent distance shrinks monotonically up to the hand-off frame (both at
    the exchange point there) and grows monotonically afterwards."""
    scene = _handoff_scene()
    giver = _center(scene, "giver")
    receiver = _center(scene, "receiver")

    def dist(f: int) -> float:
        return math.dist(giver[f], receiver[f])

    for f in range(HANDOFF_FRAME):
        assert dist(f + 1) < dist(f)
    assert dist(HANDOFF_FRAME) == pytest.approx(0.0)
    for f in range(HANDOFF_FRAME, NUM_FRAMES - 1):
        assert dist(f + 1) > dist(f)


def test_possession_timeline_and_single_holder_change() -> None:
    """Unheld before the pick-up, giver on [pickup, handoff), receiver on
    [handoff, end); the hand-off frame is the single frame where the holder id
    (one holder to another) changes."""
    scene = _handoff_scene()
    assert scene.possession is not None
    timeline = scene.possession

    assert timeline.segments == [
        PossessionSegment(
            object_id="parcel", holder_id="giver", start_frame=PICKUP_FRAME, end_frame=HANDOFF_FRAME
        ),
        PossessionSegment(
            object_id="parcel",
            holder_id="receiver",
            start_frame=HANDOFF_FRAME,
            end_frame=NUM_FRAMES,
        ),
    ]

    holders = [timeline.holder_at_frame("parcel", f) for f in range(NUM_FRAMES)]
    for f in range(PICKUP_FRAME):
        assert holders[f] is None  # in-flight/unheld
    for f in range(PICKUP_FRAME, HANDOFF_FRAME):
        assert holders[f] == "giver"
    for f in range(HANDOFF_FRAME, NUM_FRAMES):
        assert holders[f] == "receiver"

    holder_changes = [
        (f, holders[f - 1], holders[f])
        for f in range(1, NUM_FRAMES)
        if holders[f] != holders[f - 1] and holders[f - 1] is not None and holders[f] is not None
    ]
    assert holder_changes == [(HANDOFF_FRAME, "giver", "receiver")]


def test_interaction_event_records_frame_time_and_ids() -> None:
    scene = _handoff_scene()
    assert scene.possession is not None
    assert scene.possession.events == [
        InteractionEvent(
            frame=HANDOFF_FRAME,
            time=HANDOFF_FRAME / FPS,
            giver_id="giver",
            receiver_id="receiver",
            object_id="parcel",
        )
    ]


def test_object_tracks_current_holder_geometry() -> None:
    """While held, the object world point is holder point + offset; the holder
    switch at the hand-off frame is visible in the geometry itself."""
    scene = _handoff_scene()
    giver = _center(scene, "giver")
    receiver = _center(scene, "receiver")
    parcel = _center(scene, "parcel")

    for f in range(NUM_FRAMES):
        if f < PICKUP_FRAME:
            assert parcel[f] == pytest.approx(list(STAGING))
        else:
            holder = giver if f < HANDOFF_FRAME else receiver
            expected = [holder[f][i] + OFFSET[i] for i in range(3)]
            assert parcel[f] == pytest.approx(expected)

    # At the hand-off frame the parcel is on the receiver, not the giver.
    assert parcel[HANDOFF_FRAME] == pytest.approx(
        [receiver[HANDOFF_FRAME][i] + OFFSET[i] for i in range(3)]
    )
    assert parcel[HANDOFF_FRAME - 1] == pytest.approx(
        [giver[HANDOFF_FRAME - 1][i] + OFFSET[i] for i in range(3)]
    )


def test_interaction_event_rejects_non_finite_time() -> None:
    """NaN and +/-inf must not validate: they serialise to JSON ``null`` and
    then fail to reload, so a model-admissible value would not round-trip."""
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            InteractionEvent(frame=1, time=bad, giver_id="g", receiver_id="r", object_id="o")

    # A finite, non-negative time still validates and round-trips (the test is
    # not vacuous).
    event = InteractionEvent(frame=1, time=0.1, giver_id="g", receiver_id="r", object_id="o")
    assert InteractionEvent.model_validate_json(event.model_dump_json()) == event


def test_handoff_scene_roundtrips_through_json() -> None:
    """Scene and possession sidecar serialise and reload without loss."""
    scene = _handoff_scene()

    dumped = scene.model_dump(mode="json")
    restored = Scene.model_validate(dumped)
    assert restored == scene

    json_text = scene.model_dump_json(indent=2)
    assert Scene.model_validate_json(json_text) == scene

    assert scene.possession is not None
    timeline = scene.possession
    assert PossessionTimeline.model_validate_json(timeline.to_json()) == timeline
    assert PossessionTimeline.model_validate(timeline.model_dump(mode="json")) == timeline
    # The event survives the round-trip byte-for-byte.
    assert json.loads(timeline.to_json())["events"] == [
        {
            "frame": HANDOFF_FRAME,
            "time": HANDOFF_FRAME / FPS,
            "giver_id": "giver",
            "receiver_id": "receiver",
            "object_id": "parcel",
        }
    ]


def test_manifest_excludes_handoff_ground_truth() -> None:
    """The analytic manifest never contains possession or interaction GT."""
    manifest_json = build_manifest(_handoff_scene()).to_json()
    assert "possession" not in manifest_json
    assert "giver_id" not in manifest_json
    assert "InteractionEvent" not in manifest_json


def test_handoff_example_emits_valid_sidecars(example: ModuleType, tmp_path: Path) -> None:
    """The runnable example stages the scenario and writes manifest +
    possession sidecars whose GT matches the acceptance criteria."""
    summary = example.run(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    possession_path = tmp_path / "possession.json"
    assert manifest_path.exists() and possession_path.exists()

    manifest = json.loads(manifest_path.read_text())
    ids = {e["id"] for e in manifest["entities"]}
    assert ids == {example.GIVER_ID, example.RECEIVER_ID, example.OBJECT_ID}

    timeline = json.loads(possession_path.read_text())
    assert timeline["segments"] == [
        {
            "object_id": example.OBJECT_ID,
            "holder_id": example.GIVER_ID,
            "start_frame": example.PICKUP_FRAME,
            "end_frame": example.HANDOFF_FRAME,
        },
        {
            "object_id": example.OBJECT_ID,
            "holder_id": example.RECEIVER_ID,
            "start_frame": example.HANDOFF_FRAME,
            "end_frame": example.NUM_FRAMES,
        },
    ]
    assert timeline["events"] == [
        {
            "frame": example.HANDOFF_FRAME,
            "time": example.HANDOFF_FRAME / example.FPS,
            "giver_id": example.GIVER_ID,
            "receiver_id": example.RECEIVER_ID,
            "object_id": example.OBJECT_ID,
        }
    ]
    # The hand-off frame is the only holder-id change in the example summary.
    assert summary["changes"] == [
        (example.PICKUP_FRAME, None, example.GIVER_ID),
        (example.HANDOFF_FRAME, example.GIVER_ID, example.RECEIVER_ID),
    ]

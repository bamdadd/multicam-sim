"""Activity-state ground-truth sidecar: what each entity is doing, when.

An **activity timeline** records that an entity is in a typed activity state
(e.g. ``standing`` / ``crouching`` / ``reaching``) over a half-open frame
interval ``[start_frame, end_frame)``. It gives an activity-recognition head a
per-entity, per-frame label to score against: the state at any frame is
queryable via :meth:`ActivityTimeline.state_at_frame`.

The label is a :class:`ActivityState` ``StrEnum``: adding a state later is a
new enum member whose serialized value is just another string, so extension is
additive and needs no schema fork. This channel is distinct from the skeletal
motion DSL (a motion *producer*); it never generates or alters motion — it
only *labels* frames.

Like :mod:`multicam_sim.possession`, this module is pure typed models + logic.
The timeline rides in a JSON sidecar and is attached to
:class:`~multicam_sim.scene.Scene` via an optional field, so the byte-golden
analytic manifest is unchanged when activity GT is absent.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ActivityState(StrEnum):
    """A typed activity label for an entity over a frame interval.

    Str-backed so a new state is a new member here and a new string value on
    the wire — additive, with no schema fork for consumers.
    """

    standing = "standing"
    crouching = "crouching"
    reaching = "reaching"


class ActivitySegment(BaseModel):
    """One labeled interval: ``entity_id`` is in ``state`` over
    ``[start_frame, end_frame)``."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    state: ActivityState
    start_frame: int
    end_frame: int

    @field_validator("start_frame", "end_frame")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame indices must be >= 0")
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> ActivitySegment:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame {self.end_frame} must be strictly greater than "
                f"start_frame {self.start_frame}"
            )
        return self

    def contains(self, frame: int) -> bool:
        """True when ``frame`` is inside the half-open labeled interval."""
        return self.start_frame <= frame < self.end_frame


class ActivityTimeline(BaseModel):
    """A collection of activity segments for one or more entities.

    Segments are kept sorted by ``(entity_id, start_frame)`` and the model
    rejects overlapping intervals for the same entity, so every
    ``(entity_id, frame)`` pair has at most one state. Frames outside all of
    an entity's segments are *unlabeled* — :meth:`state_at_frame` returns
    ``None`` for them.
    """

    model_config = ConfigDict(frozen=True)

    segments: list[ActivitySegment] = []

    @field_validator("segments")
    @classmethod
    def _no_overlap(cls, value: list[ActivitySegment]) -> list[ActivitySegment]:
        by_entity: dict[str, list[ActivitySegment]] = {}
        for seg in value:
            by_entity.setdefault(seg.entity_id, []).append(seg)
        for entity_id, segs in by_entity.items():
            ordered = sorted(segs, key=lambda s: s.start_frame)
            for prev, cur in zip(ordered, ordered[1:], strict=False):
                if cur.start_frame < prev.end_frame:
                    raise ValueError(
                        f"overlapping activity segments for entity {entity_id!r}: "
                        f"[{prev.start_frame}, {prev.end_frame}) and "
                        f"[{cur.start_frame}, {cur.end_frame})"
                    )
        return sorted(value, key=lambda s: (s.entity_id, s.start_frame))

    def state_at_frame(self, entity_id: str, frame: int) -> ActivityState | None:
        """Return the activity state for ``entity_id`` at ``frame``, or ``None``
        if the entity is unlabeled at that frame.
        """
        for seg in self.segments:
            if seg.entity_id != entity_id:
                continue
            if seg.contains(frame):
                return seg.state
            if seg.start_frame > frame:
                break
        return None

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string (the ``activity.json`` sidecar payload)."""
        return self.model_dump_json(indent=indent)


def write_activity_json(timeline: ActivityTimeline, path: str | Path) -> dict[str, Any]:
    """Write an activity timeline to ``path`` as JSON.

    Returns the dumped dict so a caller can assert on it without re-reading.
    """
    data: dict[str, Any] = timeline.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))
    return data

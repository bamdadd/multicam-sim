"""Possession ground-truth sidecar: who is carrying which object, when.

A **possession timeline** records that an object entity is attached to (carried
by) a holder entity over a half-open frame interval ``[start_frame, end_frame)``.
The object's world point is driven by the holder's point during that interval
(this is baked into the object's :class:`~multicam_sim.entities.Entity` frames
at scene-build time) and is detached at ``end_frame``.

A two-agent **hand-off** is the same channel with a holder id that changes:
the giver's segment ends and the receiver's segment begins at the hand-off
frame, and an :class:`InteractionEvent` (frame/time, giver id, receiver id,
object id) records the transfer alongside the segments.

Like :mod:`multicam_sim.order`, this module is pure typed models + logic.
The timeline rides in a JSON sidecar and is attached to
:class:`~multicam_sim.scene.Scene` via an optional field, so the byte-golden
analytic manifest is unchanged when possession GT is absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PossessionSegment(BaseModel):
    """One attached interval: ``object_id`` is carried by ``holder_id`` over
    ``[start_frame, end_frame)`` and detached at ``end_frame``."""

    model_config = ConfigDict(frozen=True)

    object_id: str
    holder_id: str
    start_frame: int
    end_frame: int

    @field_validator("start_frame", "end_frame")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame indices must be >= 0")
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> PossessionSegment:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame {self.end_frame} must be strictly greater than "
                f"start_frame {self.start_frame}"
            )
        return self

    def contains(self, frame: int) -> bool:
        """True when ``frame`` is inside the half-open attached interval."""
        return self.start_frame <= frame < self.end_frame


class InteractionEvent(BaseModel):
    """One hand-off: ``object_id`` passes from ``giver_id`` to ``receiver_id``.

    ``frame`` is the hand-off frame — the single frame at which the per-frame
    possession holder changes — and ``time`` is that frame in wall-clock
    seconds (``frame / fps``), baked in by the builder so the sidecar is
    self-contained. Rides in the possession sidecar next to the segments,
    never in the byte-golden manifest.
    """

    model_config = ConfigDict(frozen=True)

    frame: int
    time: float
    giver_id: str
    receiver_id: str
    object_id: str

    @field_validator("frame")
    @classmethod
    def _frame_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("frame must be >= 0")
        return value

    @field_validator("time")
    @classmethod
    def _time_non_negative(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("time must be finite")
        if value < 0.0:
            raise ValueError("time must be >= 0")
        return value


class PossessionTimeline(BaseModel):
    """A collection of possession segments for one or more objects.

    Segments are kept sorted by ``(object_id, start_frame)`` and the model
    rejects overlapping intervals for the same object. ``events`` are the
    :class:`InteractionEvent`s (hand-offs) recorded alongside the segments,
    kept sorted by ``(frame, object_id)`` so the sidecar is deterministic;
    empty when no hand-off was staged.
    """

    model_config = ConfigDict(frozen=True)

    segments: list[PossessionSegment] = []
    events: list[InteractionEvent] = []

    @field_validator("segments")
    @classmethod
    def _no_overlap(cls, value: list[PossessionSegment]) -> list[PossessionSegment]:
        by_object: dict[str, list[PossessionSegment]] = {}
        for seg in value:
            by_object.setdefault(seg.object_id, []).append(seg)
        for object_id, segs in by_object.items():
            ordered = sorted(segs, key=lambda s: s.start_frame)
            for prev, cur in zip(ordered, ordered[1:], strict=False):
                if cur.start_frame < prev.end_frame:
                    raise ValueError(
                        f"overlapping possession segments for object {object_id!r}: "
                        f"[{prev.start_frame}, {prev.end_frame}) and "
                        f"[{cur.start_frame}, {cur.end_frame})"
                    )
        return sorted(value, key=lambda s: (s.object_id, s.start_frame))

    @field_validator("events")
    @classmethod
    def _sorted_events(cls, value: list[InteractionEvent]) -> list[InteractionEvent]:
        return sorted(value, key=lambda e: (e.frame, e.object_id))

    def holder_at_frame(self, object_id: str, frame: int) -> str | None:
        """Return the holder id for ``object_id`` at ``frame``, or ``None`` if
        the object is detached at that frame.
        """
        for seg in self.segments:
            if seg.object_id != object_id:
                continue
            if seg.contains(frame):
                return seg.holder_id
            if seg.start_frame > frame:
                break
        return None

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string (the ``possession.json`` sidecar payload)."""
        return self.model_dump_json(indent=indent)


def write_possession_json(timeline: PossessionTimeline, path: str | Path) -> dict[str, Any]:
    """Write a possession timeline to ``path`` as JSON.

    Returns the dumped dict so a caller can assert on it without re-reading.
    """
    data: dict[str, Any] = timeline.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))
    return data

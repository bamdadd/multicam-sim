"""Placement-synced operator action events (causal-fusion ground truth).

:class:`multicam_sim.order.ActionEvent` records *where* the operator's hand was
at each placement; this module authors the *causal* half: a discrete **hand
dip** — a strict local minimum of a tracked keypoint's height — timestamped
``action_lag`` (δ) frames *before* each item's placement frame, so a consumer
with only the emitted manifest can recover one dip per placement and associate
action → change in time.

Two authored negatives make the data able to *falsify* a weak association rule
rather than merely confirm a strong one:

* a **distractor action** — a dip that assembles nothing (no placement follows
  within the causal lag window);
* a **distractor change** — an item that moves with no dip inside the lag
  window before it (it is simply omitted from the ground-truth pairs).

The ground truth rides in a JSON sidecar (e.g. ``interactions.json``) listing
only the *true* ``(actor, item, action_frame, change_frame)`` pairs, so a
consumer scores precision/recall without re-deriving the truth — and any
association it makes beyond this list is a false positive by construction.

:class:`DipSchedule` is the motion producer (it rewrites one joint's height
channel of a :class:`~multicam_sim.pose.PoseTrajectory`); the rest is pure
typed models + logic in the :mod:`multicam_sim.order` /
:mod:`multicam_sim.possession` style. The sidecar never touches the
byte-golden manifest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .order import ItemPlacement
from .pose import PoseFrame, PoseTrajectory


class CausalTiming(BaseModel):
    """The timing contract between an operator action and the change it causes.

    ``action_lag`` (δ) is the authored gap in frames between a hand dip and the
    placement it causes: a dip at frame ``a`` causes the change at frame
    ``a + action_lag``. ``lag_window`` is the consumer-facing association
    window: an action at ``a`` and a change at ``c`` are causally associable
    only when ``0 < c - a <= lag_window``. Both are explicit typed parameters —
    the authored δ and the scoring window are never magic numbers buried in a
    consumer.
    """

    model_config = ConfigDict(frozen=True)

    action_lag: int
    lag_window: int

    @field_validator("action_lag")
    @classmethod
    def _lag_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("action_lag (δ) must be >= 1: the dip precedes the placement")
        return value

    @model_validator(mode="after")
    def _window_covers_lag(self) -> CausalTiming:
        if self.lag_window < self.action_lag:
            raise ValueError(
                f"lag_window {self.lag_window} must be >= action_lag {self.action_lag}, "
                "or the authored pairs fall outside their own causal window"
            )
        return self


class DipSchedule(BaseModel):
    """Hand dips: strict local minima of a tracked joint's height at given frames.

    Each dip is a reach-and-return triangle: the joint rests at ``rest_height``,
    descends to ``rest_height - depth`` at the dip frame, and climbs back over
    ``half_width`` frames either side. Dips must sit at least
    ``2 * half_width + 1`` frames apart so their profiles never overlap; every
    dip frame is then a *strict* local minimum — strictly lower than both
    neighbouring frames — recoverable from the manifest alone.
    """

    model_config = ConfigDict(frozen=True)

    frames: list[int]
    rest_height: float
    depth: float
    half_width: int = 1

    @field_validator("frames")
    @classmethod
    def _frames_valid(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("dip schedule needs at least one dip frame")
        for frame in value:
            if frame < 1:
                raise ValueError(
                    f"dip frame {frame} must be >= 1: "
                    "a strict local minimum needs a preceding frame"
                )
        if len(set(value)) != len(value):
            raise ValueError("dip frames must be unique")
        return sorted(value)

    @field_validator("depth")
    @classmethod
    def _positive_depth(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("dip depth must be > 0, or the minimum is not strict")
        return value

    @field_validator("half_width")
    @classmethod
    def _positive_half_width(cls, value: int) -> int:
        if value < 1:
            raise ValueError("half_width must be >= 1 frame")
        return value

    @model_validator(mode="after")
    def _profiles_disjoint(self) -> DipSchedule:
        for prev, cur in zip(self.frames, self.frames[1:], strict=False):
            if cur - prev < 2 * self.half_width + 1:
                raise ValueError(
                    f"dip frames {prev} and {cur} are closer than "
                    f"2*half_width+1 ({2 * self.half_width + 1}): overlapping "
                    "profiles would break the strict local minimum"
                )
        return self

    def height_at(self, frame: int) -> float:
        """Tracked-joint height at ``frame``: rest height minus the dip profile."""
        height = self.rest_height
        for dip in self.frames:
            dist = abs(frame - dip)
            height -= self.depth * max(0.0, 1.0 - dist / (self.half_width + 1))
        return height

    def author(self, trajectory: PoseTrajectory, joint: str) -> PoseTrajectory:
        """Return ``trajectory`` with ``joint``'s height channel replaced by this profile.

        The joint's x/y pass through untouched; its z becomes
        :meth:`height_at` at every frame, so outside dips the hand sits exactly
        at ``rest_height`` and the only strict local minima in the channel are
        the authored dips. Every dip needs both neighbouring frames present in
        the trajectory — a strict local minimum is defined by its neighbours.
        """
        available = {f.frame for f in trajectory.frames}
        for dip in self.frames:
            if dip - 1 not in available or dip + 1 not in available:
                raise ValueError(
                    f"dip at frame {dip} needs frames {dip - 1} and {dip + 1} in the trajectory"
                )
        frames: list[PoseFrame] = []
        for pose_frame in trajectory.frames:
            if joint not in pose_frame.joints:
                raise ValueError(f"joint {joint!r} missing at frame {pose_frame.frame}")
            joints = dict(pose_frame.joints)
            x, y, _z = joints[joint]
            joints[joint] = [x, y, self.height_at(pose_frame.frame)]
            frames.append(PoseFrame(frame=pose_frame.frame, joints=joints))
        return PoseTrajectory(id=trajectory.id, skeleton=trajectory.skeleton, frames=frames)


class ActionChange(BaseModel):
    """One ground-truth causal pair: ``actor_id``'s dip at ``action_frame``
    caused ``item_id``'s placement at ``change_frame``.

    This is the ``(actor, item, action_time, change_time)`` tuple of the
    causal-fusion contract, expressed in frames (the repo's time unit; seconds
    are ``frame / fps`` for a consumer that needs them).
    """

    model_config = ConfigDict(frozen=True)

    actor_id: str
    item_id: str
    action_frame: int
    change_frame: int

    @model_validator(mode="after")
    def _action_precedes_change(self) -> ActionChange:
        if self.change_frame <= self.action_frame:
            raise ValueError(
                f"change_frame {self.change_frame} must be strictly after "
                f"action_frame {self.action_frame}: causes precede effects"
            )
        return self


class ActionGroundTruth(BaseModel):
    """The causal-fusion sidecar payload: the true action→change pairs + contract.

    ``pairs`` holds only the *true* associations, kept sorted by
    ``(change_frame, item_id)`` so the sidecar is deterministic. Distractors
    are deliberately absent: they live in the manifest, and a consumer that
    associates one scores a false positive against this list. ``timing`` and
    ``tracked_joint`` document the channel the pairs were authored against.
    """

    model_config = ConfigDict(frozen=True)

    timing: CausalTiming
    actor_id: str
    tracked_joint: str
    pairs: list[ActionChange] = []

    @field_validator("pairs")
    @classmethod
    def _sorted_pairs(cls, value: list[ActionChange]) -> list[ActionChange]:
        return sorted(value, key=lambda p: (p.change_frame, p.item_id))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string (the ``interactions.json`` sidecar payload)."""
        return self.model_dump_json(indent=indent)


def build_action_ground_truth(
    timing: CausalTiming,
    actor_id: str,
    tracked_joint: str,
    placements: Sequence[ItemPlacement],
) -> ActionGroundTruth:
    """Derive the true pairs from placements: one dip at ``placed_at - δ`` each.

    ``placements`` must list only causally-backed placements; a distractor
    change (an item moving outside the causal lag window) is simply omitted,
    which is exactly what makes it a negative for the consumer.
    """
    pairs = [
        ActionChange(
            actor_id=actor_id,
            item_id=p.item,
            action_frame=p.placed_at_frame - timing.action_lag,
            change_frame=p.placed_at_frame,
        )
        for p in placements
    ]
    return ActionGroundTruth(
        timing=timing, actor_id=actor_id, tracked_joint=tracked_joint, pairs=pairs
    )


def write_actions_json(truth: ActionGroundTruth, path: str | Path) -> dict[str, Any]:
    """Write the action ground truth to ``path`` as JSON (e.g. ``interactions.json``).

    Returns the dumped dict so a caller can assert on it without re-reading.
    """
    data: dict[str, Any] = truth.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))
    return data

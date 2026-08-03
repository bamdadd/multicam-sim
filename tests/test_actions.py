"""Placement-synced operator action events (#34) — the library half.

A dip is a reach-and-return *strict local minimum* of a tracked joint's height:
the tests pin the profile with pasted-in literals (derived from the triangle
geometry, not from the implementation), then assert the authored trajectory
recovers exactly the authored minima — the property a manifest-only consumer
relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multicam_sim.actions import (
    ActionChange,
    CausalTiming,
    DipSchedule,
    build_action_ground_truth,
    write_actions_json,
)
from multicam_sim.order import ItemPlacement
from multicam_sim.pose import PoseFrame, PoseTrajectory, Skeleton


def _trajectory(num_frames: int, z: float = 5.0) -> PoseTrajectory:
    """A two-joint skeleton whose ``hand`` zigzags in z — authoring must replace
    the whole height channel, not just add to it."""
    skeleton = Skeleton(name="test", joints=["hand", "elbow"], edges=[("hand", "elbow")])
    frames = [
        PoseFrame(
            frame=f,
            joints={"hand": [1.0, 2.0, z + (-1.0) ** f], "elbow": [0.0, 0.0, 1.0]},
        )
        for f in range(num_frames)
    ]
    return PoseTrajectory(id="actor", skeleton=skeleton, frames=frames)


def test_causal_timing_validation() -> None:
    with pytest.raises(ValueError, match="action_lag"):
        CausalTiming(action_lag=0, lag_window=2)
    with pytest.raises(ValueError, match="lag_window"):
        CausalTiming(action_lag=3, lag_window=2)
    assert CausalTiming(action_lag=2, lag_window=2).action_lag == 2


def test_dip_profile_height_literals() -> None:
    """Triangle half_width=1: dip frame at rest-depth, neighbours at rest-depth/2."""
    dips = DipSchedule(frames=[2, 6], rest_height=1.0, depth=0.2, half_width=1)
    assert [dips.height_at(f) for f in range(8)] == pytest.approx(
        [1.0, 0.9, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9]
    )


def test_dip_profile_wider_half_width_literals() -> None:
    """Triangle half_width=2: linear return to rest over two frames either side."""
    dips = DipSchedule(frames=[3], rest_height=2.0, depth=0.6, half_width=2)
    assert [dips.height_at(f) for f in range(7)] == pytest.approx(
        [2.0, 1.8, 1.6, 1.4, 1.6, 1.8, 2.0]
    )


def test_dip_schedule_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DipSchedule(frames=[], rest_height=1.0, depth=0.2)
    with pytest.raises(ValueError, match=">= 1"):
        DipSchedule(frames=[0], rest_height=1.0, depth=0.2)  # no preceding frame
    with pytest.raises(ValueError, match="unique"):
        DipSchedule(frames=[4, 4], rest_height=1.0, depth=0.2)
    with pytest.raises(ValueError, match="depth"):
        DipSchedule(frames=[4], rest_height=1.0, depth=0.0)  # minimum not strict
    with pytest.raises(ValueError, match="half_width"):
        DipSchedule(frames=[4], rest_height=1.0, depth=0.2, half_width=0)
    with pytest.raises(ValueError, match="overlap"):
        # 2 apart with half_width=1: the profiles share a frame, breaking strictness.
        DipSchedule(frames=[4, 6], rest_height=1.0, depth=0.2)


def test_author_makes_exactly_the_authored_strict_minima() -> None:
    dips = DipSchedule(frames=[2, 6], rest_height=1.0, depth=0.2, half_width=1)
    authored = dips.author(_trajectory(9), "hand")
    z = [f.joints["hand"][2] for f in authored.frames]
    assert z == pytest.approx([1.0, 0.9, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9, 1.0])
    # The recovered strict local minima are exactly the authored dips — nothing
    # left over from the zigzag channel the schedule replaced.
    minima = [i for i in range(1, len(z) - 1) if z[i] < z[i - 1] and z[i] < z[i + 1]]
    assert minima == [2, 6]


def test_author_preserves_other_channels() -> None:
    base = _trajectory(5, z=5.0)
    authored = DipSchedule(frames=[2], rest_height=1.0, depth=0.2).author(base, "hand")
    for f_before, f_after in zip(base.frames, authored.frames, strict=True):
        assert f_after.joints["elbow"] == f_before.joints["elbow"]
        assert f_after.joints["hand"][:2] == f_before.joints["hand"][:2]  # x, y pass through
    assert authored.id == base.id and authored.skeleton == base.skeleton


def test_author_requires_neighbour_frames_and_known_joint() -> None:
    dips = DipSchedule(frames=[4], rest_height=1.0, depth=0.2)
    with pytest.raises(ValueError, match="frames 3 and 5"):
        dips.author(_trajectory(5), "hand")  # frame 5 missing: no right neighbour
    with pytest.raises(ValueError, match="missing"):
        dips.author(_trajectory(6), "ankle")


def test_action_change_requires_cause_before_effect() -> None:
    with pytest.raises(ValueError, match="strictly after"):
        ActionChange(actor_id="op", item_id="part_a", action_frame=3, change_frame=3)


def test_build_action_ground_truth_pairs_and_sorting() -> None:
    timing = CausalTiming(action_lag=1, lag_window=2)
    placements = [
        ItemPlacement(item="part_b", placed_at_frame=5, entity_id="part_b"),
        ItemPlacement(item="part_a", placed_at_frame=2, entity_id="part_a"),
    ]
    truth = build_action_ground_truth(timing, "operator", "right_wrist", placements)
    assert [(p.item_id, p.action_frame, p.change_frame) for p in truth.pairs] == [
        ("part_a", 1, 2),
        ("part_b", 4, 5),
    ]
    assert truth.actor_id == "operator" and truth.tracked_joint == "right_wrist"
    assert truth.timing == timing


def test_build_action_ground_truth_rejects_unauthorable_dips() -> None:
    """A placement at frame <= δ would put its dip where no DipSchedule can
    author one (a strict local minimum needs a preceding frame) — fail loudly
    at construction, naming the item and frames."""
    timing = CausalTiming(action_lag=5, lag_window=5)
    # the reviewer's case: placed at 3 with δ=5 → dip at frame -2
    with pytest.raises(ValueError, match=r"'part_a'.*frame 3.*frame -2"):
        build_action_ground_truth(
            timing,
            "operator",
            "right_wrist",
            [ItemPlacement(item="part_a", placed_at_frame=3, entity_id="part_a")],
        )
    # boundary: placed exactly at δ → dip at frame 0, still unauthorable
    with pytest.raises(ValueError, match=r"'part_a'.*frame 5.*frame 0"):
        build_action_ground_truth(
            timing,
            "operator",
            "right_wrist",
            [ItemPlacement(item="part_a", placed_at_frame=5, entity_id="part_a")],
        )
    # just above the boundary: placed at δ+1 → dip at frame 1, authorable
    truth = build_action_ground_truth(
        timing,
        "operator",
        "right_wrist",
        [ItemPlacement(item="part_a", placed_at_frame=6, entity_id="part_a")],
    )
    assert [(p.action_frame, p.change_frame) for p in truth.pairs] == [(1, 6)]


def test_sidecar_round_trip(tmp_path: Path) -> None:
    timing = CausalTiming(action_lag=1, lag_window=2)
    placements = [ItemPlacement(item="part_a", placed_at_frame=2, entity_id="part_a")]
    truth = build_action_ground_truth(timing, "operator", "right_wrist", placements)
    written = write_actions_json(truth, tmp_path / "interactions.json")
    assert json.loads((tmp_path / "interactions.json").read_text()) == written
    assert written == {
        "timing": {"action_lag": 1, "lag_window": 2},
        "actor_id": "operator",
        "tracked_joint": "right_wrist",
        "pairs": [
            {"actor_id": "operator", "item_id": "part_a", "action_frame": 1, "change_frame": 2}
        ],
    }

"""Group formation & dispersal: builder geometry, deterministic membership GT,
JSON round-trip, and the byte-golden manifest guarantee.

Expected values below are derived from the builder's default geometry, NOT from
the GT function: an agent's distance from the group centre falls linearly from
1.0 (frame 0) to 0.2 (frame 20), crossing radius 0.5 at frame 12.5, so every
agent is within from frame 13; with the 3-frame dwell rule, membership starts
at frame 15. During dispersal the distance grows monotonically from 0.2 back
to 1.0 over frames 40..59 (radially, along a direction rotated half a sector),
crossing 0.5 between frames 47 and 48, so membership holds through frame 47
and the group is empty again from frame 48.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from multicam_sim import Scene, build_manifest
from multicam_sim.groups import (
    GroupFrameMembership,
    GroupMembership,
    build_group_formation_scene,
    compute_group_membership,
    write_group_json,
)

_AGENT_IDS = ["agent-0", "agent-1", "agent-2", "agent-3"]
_NUM_FRAMES = 60
_FORMATION_FRAME = 15
_DISPERSAL_FRAME = 48
_CENTER = (0.0, 0.0, 0.5)


def _expected_members(frame: int) -> list[str]:
    return list(_AGENT_IDS) if _FORMATION_FRAME <= frame < _DISPERSAL_FRAME else []


def test_builder_scene_shape() -> None:
    scene = build_group_formation_scene()
    assert scene.num_frames == _NUM_FRAMES
    assert len(scene.cameras) == 3
    assert [e.id for e in scene.entities] == _AGENT_IDS
    for entity in scene.entities:
        assert len(entity.frames) == _NUM_FRAMES
        assert {f.frame for f in entity.frames} == set(range(_NUM_FRAMES))
    # During the dwell window every agent sits on the cluster ring (spread 0.2).
    for entity in scene.entities:
        xyz = entity.frames[30].points["center"]
        assert math.dist(xyz, _CENTER) == pytest.approx(0.2, abs=1e-12)


def test_formation_dispersal_and_per_frame_membership() -> None:
    scene = build_group_formation_scene()
    gt = compute_group_membership(scene)

    assert gt.group_id == "group-0"
    assert gt.formation_frame == _FORMATION_FRAME
    assert gt.dispersal_frame == _DISPERSAL_FRAME

    assert [f.frame for f in gt.frames] == list(range(_NUM_FRAMES))
    for frame in gt.frames:
        assert frame.members == _expected_members(frame.frame)
        # Symmetric ring geometry pins the centroid to the group centre.
        assert frame.centroid == pytest.approx(_CENTER, abs=1e-9)


def test_membership_rule_is_deterministic_and_scene_alone() -> None:
    scene = build_group_formation_scene()
    first = compute_group_membership(scene)
    second = compute_group_membership(scene)
    assert first == second


def test_group_id_and_rule_parameters_are_explicit() -> None:
    scene = build_group_formation_scene()
    # A custom group id rides through; a 1-frame dwell starts membership at the
    # first within frame (13) instead of 15.
    gt = compute_group_membership(scene, min_dwell_frames=1, group_id="group-7")
    assert gt.group_id == "group-7"
    assert gt.min_dwell_frames == 1
    assert gt.formation_frame == 13
    assert gt.dispersal_frame == _DISPERSAL_FRAME


def test_json_round_trip() -> None:
    scene = build_group_formation_scene()
    gt = compute_group_membership(scene)

    payload = gt.to_json()
    assert '"formation_frame": 15' in payload
    assert '"dispersal_frame": 48' in payload

    restored = GroupMembership.model_validate_json(payload)

    # Compare against independently-constructed expectations, not the original
    # object: scalar fields, then per-frame (frame, members) pairs from literals.
    assert restored.group_id == "group-0"
    assert restored.radius == 0.5
    assert restored.min_dwell_frames == 3
    assert restored.point == "center"
    assert restored.formation_frame == _FORMATION_FRAME
    assert restored.dispersal_frame == _DISPERSAL_FRAME
    assert [(f.frame, f.members) for f in restored.frames] == [
        (f, _expected_members(f)) for f in range(_NUM_FRAMES)
    ]
    for frame in restored.frames:
        assert frame.centroid == pytest.approx(_CENTER, abs=1e-9)

    # The builder's scene itself round-trips too.
    assert Scene.model_validate_json(scene.model_dump_json()) == scene


def test_sidecar_write_and_reload(tmp_path: Path) -> None:
    scene = build_group_formation_scene()
    gt = compute_group_membership(scene)

    path = tmp_path / "groups.json"
    dumped = write_group_json(gt, path)
    reloaded = GroupMembership.model_validate(json.loads(path.read_text()))

    assert dumped["group_id"] == "group-0"
    assert reloaded.formation_frame == _FORMATION_FRAME
    assert reloaded.dispersal_frame == _DISPERSAL_FRAME
    assert [(f.frame, f.members) for f in reloaded.frames] == [
        (f, _expected_members(f)) for f in range(_NUM_FRAMES)
    ]


def test_manifest_byte_identical_without_membership_gt(tmp_path: Path) -> None:
    scene = build_group_formation_scene()
    before = build_manifest(scene).to_json()

    # Requesting the GT (and writing its sidecar) must not perturb the scene
    # or the analytic manifest in any way.
    gt = compute_group_membership(scene)
    write_group_json(gt, tmp_path / "groups.json")

    after = build_manifest(scene).to_json()
    assert before == after
    assert '"members"' not in after
    assert '"centroid"' not in after
    assert '"group_id"' not in after


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="cluster_spread"):
        build_group_formation_scene(cluster_spread=0.6, radius=0.5)
    with pytest.raises(ValueError, match="arrival_frame"):
        build_group_formation_scene(arrival_frame=50, departure_frame=40)
    with pytest.raises(ValueError, match="at least 2 agents"):
        build_group_formation_scene(num_agents=1)
    with pytest.raises(ValueError, match="radius must be > 0"):
        compute_group_membership(build_group_formation_scene(), radius=0.0)
    with pytest.raises(ValueError, match="min_dwell_frames must be >= 1"):
        compute_group_membership(build_group_formation_scene(), min_dwell_frames=0)


def test_group_frame_membership_model() -> None:
    frame = GroupFrameMembership(frame=3, centroid=(0.0, 0.0, 0.5), members=["agent-0"])
    assert frame.frame == 3
    assert frame.members == ["agent-0"]

"""Skeletal-motion DSL: gaits animate all 17 COCO joints, continuously.

CPU-only, no renderer, no GL. Covers the three gaits (walk / reach / wave):
joint completeness per frame, frame-to-frame continuity against a bound
derived from each gait's own parameters, root translation over the requested
duration, determinism, and a manifest round-trip that labels every joint per
camera.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from multicam_sim import (
    COCO17_JOINTS,
    Camera,
    Intrinsics,
    PoseTrajectory,
    Scene,
    build_manifest,
)
from multicam_sim.dsl import Gait, Path
from multicam_sim.dsl.gait import ReachGait, WalkGait, WaveGait

_FPS = 30.0
_NUM_FRAMES = 61  # 2 seconds of motion


def _walk() -> WalkGait:
    root = Path.linear((0.0, -1.0, 0.9), (0.0, 1.0, 0.9)).over(2.0)
    return Gait.walk(root=root)


def _reach() -> ReachGait:
    root = Path.linear((0.0, 0.0, 0.9), (0.0, 0.5, 0.9)).over(2.0)
    return Gait.reach((0.4, 0.3, 0.4), root=root, reach_duration=1.0)


def _wave() -> WaveGait:
    root = Path.linear((0.0, 0.0, 0.9), (0.0, 0.5, 0.9)).over(2.0)
    return Gait.wave(root=root)


def _gaits() -> dict[str, WalkGait | ReachGait | WaveGait]:
    return {"walk": _walk(), "reach": _reach(), "wave": _wave()}


def _trajectories() -> dict[str, PoseTrajectory]:
    return {
        name: gait.to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
        for name, gait in _gaits().items()
    }


def _root_speed(gait: WalkGait | ReachGait | WaveGait) -> float:
    # every test root is a timed LinearPath, so the traversal speed is exact
    return gait.root.length() / gait.root.total_duration()


def _max_frame_displacement(traj: PoseTrajectory) -> float:
    worst = 0.0
    for prev, cur in zip(traj.frames[:-1], traj.frames[1:], strict=True):
        for name in COCO17_JOINTS:
            delta = np.asarray(cur.joints[name]) - np.asarray(prev.joints[name])
            worst = max(worst, float(np.linalg.norm(delta)))
    return worst


# -- joint completeness ------------------------------------------------------ #


def test_every_gait_yields_all_17_joints_every_frame() -> None:
    for name, traj in _trajectories().items():
        assert len(traj.frames) == _NUM_FRAMES, name
        entity = traj.to_entity()  # check_complete runs here
        assert entity.point_names() == set(COCO17_JOINTS)
        for frame in traj.frames:
            assert set(frame.joints) == set(COCO17_JOINTS), name
            for xyz in frame.joints.values():
                assert len(xyz) == 3


# -- continuity -------------------------------------------------------------- #


def test_joint_motion_is_continuous_frame_to_frame() -> None:
    """Per-frame displacement is bounded by what the parameters allow: the root
    path moves at ``root.length() / duration`` and the gait itself moves any
    joint at most ``gait.max_local_speed()`` (analytic bounds on the sinusoid /
    smoothstep derivatives — see each gait's docstring), so between frames
    1/fps apart no joint can travel further than their sum times 1/fps.
    """
    for name, gait in _gaits().items():
        traj = gait.to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
        bound = (_root_speed(gait) + gait.max_local_speed()) / _FPS
        worst = _max_frame_displacement(traj)
        assert worst <= bound * (1.0 + 1e-9), f"{name}: {worst} > {bound}"
        # the bound must actually bite: the gait really moves close to it
        assert worst > 0.25 * bound, f"{name}: bound {bound} is loose vs {worst}"


# -- root translation -------------------------------------------------------- #


def test_walk_root_translates_over_the_requested_duration() -> None:
    gait = _walk()
    traj = gait.to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
    # hips have no body-local horizontal motion in walk, so their world
    # displacement equals the root path's: 2.0 units in +y over 2 seconds.
    first = traj.frames[0].joints["left_hip"]
    last = traj.frames[-1].joints["left_hip"]
    assert first == pytest.approx([-0.17, -1.0, 0.9])
    assert last[1] - first[1] == pytest.approx(2.0)
    assert last[0] - first[0] == pytest.approx(0.0)


def test_untimed_root_stretches_to_scene_like_motion_dsl() -> None:
    # the timing model is the motion DSL's own: an untimed root fills the scene
    gait = Gait.walk(root=Path.linear((0.0, 0.0, 0.9), (0.0, 1.0, 0.9)))
    traj = gait.to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
    first = traj.frames[0].joints["left_hip"]
    last = traj.frames[-1].joints["left_hip"]
    assert last[1] - first[1] == pytest.approx(1.0)


# -- gait-specific behaviour ------------------------------------------------- #


def test_reach_wrist_lands_on_target_and_holds() -> None:
    traj = _reach().to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
    # root at frame 30 (t = 1.0s = reach_duration) and beyond: (0, 0.25..0.5, 0.9)
    for frame in traj.frames[30:]:
        t = frame.frame / _FPS
        root_y = 0.5 * min(t / 2.0, 1.0)
        assert frame.joints["right_wrist"] == pytest.approx([0.4, 0.3 + root_y, 0.4 + 0.9])


def test_wave_only_moves_the_waving_arm() -> None:
    traj = _wave().to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
    for frame in traj.frames:
        # the non-waving side and the legs keep their standing offsets
        assert frame.joints["left_wrist"][0] == pytest.approx(-0.165 * 1.7)
        assert frame.joints["left_ankle"][2] == pytest.approx(0.9 - 0.49 * 1.7)
    xs = [f.joints["right_wrist"][0] for f in traj.frames]
    assert max(xs) - min(xs) > 0.1  # the forearm actually oscillates


def test_validation_at_construction() -> None:
    with pytest.raises(ValidationError):
        Gait.walk(step_frequency=0.0)
    with pytest.raises(ValidationError):
        Gait.walk(stride=-1.0)
    with pytest.raises(ValidationError):
        Gait.walk(height=-1.7)
    with pytest.raises(ValidationError):
        Gait.reach((0.4, 0.3, 0.4), reach_duration=0.0)
    with pytest.raises(ValidationError):
        Gait.wave(wave_frequency=-2.0)
    with pytest.raises(ValidationError):
        Gait.wave(wave_amplitude=-0.1)


# -- determinism ------------------------------------------------------------- #


def test_compilation_is_deterministic() -> None:
    # no randomness anywhere in the layer: two compilations are byte-identical
    for name, gait in _gaits().items():
        a = gait.to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
        b = gait.to_pose_trajectory("p0", fps=_FPS, num_frames=_NUM_FRAMES)
        assert [f.joints for f in a.frames] == [f.joints for f in b.frames], name


# -- manifest round-trip ----------------------------------------------------- #


def _two_cameras() -> list[Camera]:
    intr = Intrinsics.from_focal(800.0, 640, 480)
    look = np.array([0.0, 0.0, 0.9])
    return [
        Camera.look_at(0, intr, np.array([5.0, 0.0, 1.5]), look),
        Camera.look_at(1, intr, np.array([0.0, 5.0, 1.5]), look),
    ]


def test_manifest_round_trip_labels_every_joint_per_camera() -> None:
    cameras = _two_cameras()
    for name, gait in _gaits().items():
        traj = gait.to_pose_trajectory("p0", fps=_FPS, num_frames=11)
        scene = Scene(
            fps=_FPS,
            num_frames=11,
            cameras=cameras,
            entities=[traj.to_entity()],
            occluders=[],
        )
        manifest = build_manifest(scene)
        entity = manifest.entities[0]
        assert entity.id == "p0"
        assert len(entity.edges) == 19  # COCO-17 skeleton
        for frame in entity.frames:
            assert set(frame.points) == set(COCO17_JOINTS), name
            for joint in COCO17_JOINTS:
                entry = frame.points[joint]
                assert len(entry.xyz_gt) == 3
                assert len(entry.per_cam) == len(cameras)
                for obs in entry.per_cam:
                    assert obs.visible in (True, False)
                    if obs.in_view:
                        assert obs.uv is not None

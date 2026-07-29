"""Behaviour layer: the open-closed policy above the motion DSL.

Three guarantees are proved here:

* :class:`PathBehavior` is a **strict superset** of the motion DSL — the frames
  it rolls out are byte-identical to ``path.compile_frames``, and a scene built
  through it produces a manifest byte-identical to one built from the raw path
  (the schema stays byte-identical, per ``DESIGN.md``);
* :class:`WaypointBehavior` is a correct, seeded discrete integrator that reaches
  its goals and holds at the last one;
* determinism: a fixed seed is byte-reproducible, and a >= 3-seed sweep of a
  jittered behaviour reports a stable arrival with mean +/- std (portfolio rule).

CPU-only, no renderer imported anywhere in this path.
"""

from __future__ import annotations

import numpy as np

from multicam_sim import build_manifest
from multicam_sim.cameras import Camera
from multicam_sim.dsl import (
    Behavior,
    CameraRig,
    Path,
    PathBehavior,
    SceneBuilder,
    WaypointBehavior,
)

_FPS = 30.0
_NUM_FRAMES = 11


def _ring() -> list[Camera]:
    return CameraRig.ring(
        n=3, radius=4.0, height=1.5, look_at=(0.0, 0.0, 0.5), focal=800.0, width=640, height_px=480
    )


def test_pathbehavior_is_a_protocol_member() -> None:
    assert isinstance(PathBehavior(Path.linear((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))), Behavior)
    assert isinstance(WaypointBehavior([(1.0, 0.0, 0.0)], speed=1.0), Behavior)


def test_pathbehavior_frames_match_raw_path_exactly() -> None:
    path = Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5))
    raw = path.compile_frames(_FPS, _NUM_FRAMES, name="center")
    rolled = PathBehavior(path).rollout(_FPS, _NUM_FRAMES, name="center")
    assert [f.model_dump() for f in rolled] == [f.model_dump() for f in raw]


def test_pathbehavior_manifest_is_byte_identical_to_raw_path() -> None:
    """A scene built with ``PathBehavior`` emits the same manifest JSON as the
    same scene built with the raw path — the behaviour layer changes nothing on
    the wire."""
    path = Path.linear((0.0, -0.6, 0.5), (0.0, 0.6, 0.5))

    raw_scene = SceneBuilder(_FPS, _NUM_FRAMES).cameras(_ring()).entity("obj", path).build()
    beh_scene = (
        SceneBuilder(_FPS, _NUM_FRAMES).cameras(_ring()).entity("obj", PathBehavior(path)).build()
    )
    assert build_manifest(beh_scene).to_json() == build_manifest(raw_scene).to_json()


def test_waypoint_behavior_reaches_and_holds_last_goal() -> None:
    goals = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    beh = WaypointBehavior(goals, speed=6.0, start=(0.0, 0.0, 0.0))
    frames = beh.rollout(_FPS, _NUM_FRAMES, name="center")

    assert frames[0].points["center"] == [0.0, 0.0, 0.0]
    last = np.asarray(frames[-1].points["center"])
    # speed 6 u/s over (11-1)/30 s = 2.0 u of travel; the two legs total 2.0 u,
    # so the last goal is reached exactly and held.
    assert np.allclose(last, [1.0, 1.0, 0.0], atol=1e-9)


def test_waypoint_behavior_moves_at_constant_speed() -> None:
    beh = WaypointBehavior([(2.0, 0.0, 0.0)], speed=3.0, start=(0.0, 0.0, 0.0))
    frames = beh.rollout(_FPS, _NUM_FRAMES, name="center")
    step = 3.0 / _FPS
    pts = [np.asarray(f.points["center"]) for f in frames]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        d = float(np.linalg.norm(b - a))
        # each step is either the constant step, or 0 once the goal is reached.
        assert np.isclose(d, step, atol=1e-9) or np.isclose(d, 0.0, atol=1e-9)


def test_seeded_jitter_is_deterministic() -> None:
    def roll(seed: int) -> list[list[float]]:
        beh = WaypointBehavior(
            [(2.0, 0.0, 0.0)], speed=3.0, start=(0.0, 0.0, 0.0), pacing_jitter=0.3, seed=seed
        )
        return [f.points["center"] for f in beh.rollout(_FPS, _NUM_FRAMES)]

    assert roll(7) == roll(7)  # same seed -> byte-identical
    assert roll(7) != roll(9)  # different seed -> different rollout


def test_jitter_off_by_default_is_byte_reproducible() -> None:
    """With ``pacing_jitter=0`` the rollout is a pure function of the geometry;
    two different construction seeds give identical frames, so the manifest is
    byte-reproducible regardless of seed."""
    a = WaypointBehavior([(2.0, 0.0, 0.0)], speed=3.0, seed=1).rollout(_FPS, _NUM_FRAMES)
    b = WaypointBehavior([(2.0, 0.0, 0.0)], speed=3.0, seed=2).rollout(_FPS, _NUM_FRAMES)
    assert [f.points for f in a] == [f.points for f in b]


def test_three_seed_arrival_is_stable_mean_std() -> None:
    """A jittered behaviour over 3 seeds: the object still arrives at the goal
    (jitter is only a speed wobble), so arrival distance is ~0 with tiny std."""
    goal = np.array([2.0, 0.0, 0.0])
    seeds = [0, 1, 2]
    arrivals = []
    for s in seeds:
        beh = WaypointBehavior(
            [tuple(goal)], speed=9.0, start=(0.0, 0.0, 0.0), pacing_jitter=0.2, seed=s
        )
        last = np.asarray(beh.rollout(_FPS, _NUM_FRAMES)[-1].points["center"])
        arrivals.append(float(np.linalg.norm(last - goal)))
    arr = np.asarray(arrivals)
    # speed 9 over 2/3 s = 6 u of budget >> 2 u to the goal, so every seed
    # reaches and holds the goal exactly: mean 0, std 0.
    assert arr.mean() < 1e-9
    assert arr.std() < 1e-9

"""Behaviour layer: a seeded *policy* that emits an entity's per-frame pose.

The movement DSL (:mod:`multicam_sim.dsl.motion`) is a hand-authored ``Path`` —
a trajectory fixed at authoring time. A **behaviour** is the open-closed layer
above it: a policy that *decides* where the object goes, then lowers that
decision to the same ``list[EntityFrame]`` the ``Path`` already produces. Because
the output type is unchanged, a behaviour drops into
:meth:`multicam_sim.dsl.builder.SceneBuilder.entity` with **no contract change**
(``DESIGN.md``): a behaviour only sets ``entity.frames[*].points[name]``; the
sim still runs the unchanged projection + boolean occlusion to compute
``xyz_gt`` / ``uv`` / ``visible`` / ``occ_frac``. **The behaviour never touches
ground truth** — the same discipline the DSL ``Occlusion`` layer follows.

Determinism is non-negotiable: every behaviour takes an explicit ``seed`` and
uses ``numpy.random.default_rng(seed)`` sub-streams (never the global RNG),
exactly as :mod:`multicam_sim.noise` / :mod:`multicam_sim.dropout` do. A scene
built from a seeded behaviour is byte-reproducible.

This module ships the two cheapest behaviours (``camera-driver-e2e.md`` §4.2):

* :class:`PathBehavior` wraps an existing ``Path`` verbatim — zero new physics,
  a strict superset of today's DSL. Ships first.
* :class:`WaypointBehavior` is a tiny seeded discrete integrator: move toward
  each goal at a fixed speed, advancing when within a tolerance.

Reactive behaviours (velocity control laws consulting a read-only
``SceneContext``, agent/RL pose proposers) are a later, open-closed addition on
the same :class:`Behavior` Protocol; they are intentionally NOT built here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..entities import EntityFrame
from ..geometry import FloatArray
from .motion import PathUnion, Vec3


@runtime_checkable
class Behavior(Protocol):
    """A seeded policy that emits an entity's per-frame pose.

    Deterministic: the same ``(seed, fps, num_frames)`` yields the same frames.
    A behaviour only proposes a pose; the sim computes all ground truth from it.
    """

    def rollout(
        self,
        fps: float,
        num_frames: int,
        *,
        seed: int = 0,
        name: str = "center",
    ) -> list[EntityFrame]:
        """Lower the policy to ``num_frames`` :class:`EntityFrame`s."""
        ...


class PathBehavior:
    """Wrap an existing :class:`~multicam_sim.dsl.motion.Path` verbatim.

    The trivial behaviour: the pose per frame is whatever the ``Path`` already
    compiles. Guarantees the behaviour layer is a strict superset of the motion
    DSL; ``seed`` is accepted for interface uniformity but unused (a compiled
    path is already deterministic).
    """

    def __init__(self, path: PathUnion) -> None:
        self._path = path

    @property
    def path(self) -> PathUnion:
        """The wrapped path node."""
        return self._path

    def rollout(
        self,
        fps: float,
        num_frames: int,
        *,
        seed: int = 0,
        name: str = "center",
    ) -> list[EntityFrame]:
        return self._path.compile_frames(fps, num_frames, name=name)


class WaypointBehavior:
    """Move toward each goal at a fixed ``speed``, advancing within ``tol``.

    A tiny discrete integrator: from ``start`` (or the first goal, if ``start``
    is omitted), step ``speed / fps`` world-units per frame toward the current
    goal; when within ``tol`` of it, advance to the next. Once the last goal is
    reached the object holds there for the remaining frames.

    Deterministic. ``pacing_jitter`` (default ``0.0``) adds a seeded per-step
    fractional speed wobble drawn from ``default_rng(seed)``; with the default it
    is a pure function of ``(goals, speed, fps, num_frames)`` and the manifest is
    byte-reproducible. Any reported metric depending on ``pacing_jitter`` must be
    run over >= 3 seeds (portfolio rule).
    """

    def __init__(
        self,
        goals: list[Vec3],
        *,
        speed: float,
        start: Vec3 | None = None,
        tol: float = 1e-3,
        pacing_jitter: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not goals:
            raise ValueError("waypoint behaviour needs at least one goal")
        if speed <= 0.0:
            raise ValueError("speed must be > 0")
        if tol <= 0.0:
            raise ValueError("tol must be > 0")
        if not 0.0 <= pacing_jitter < 1.0:
            raise ValueError("pacing_jitter must be in [0, 1)")
        self._goals = [np.asarray(g, dtype=np.float64) for g in goals]
        self._start = None if start is None else np.asarray(start, dtype=np.float64)
        self._speed = speed
        self._tol = tol
        self._pacing_jitter = pacing_jitter
        self._seed = seed

    def rollout(
        self,
        fps: float,
        num_frames: int,
        *,
        seed: int = 0,
        name: str = "center",
    ) -> list[EntityFrame]:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if fps <= 0.0:
            raise ValueError("fps must be > 0")
        # The explicit rollout seed overrides the construction seed when the
        # caller (e.g. a >=3-seed sweep) passes one, mirroring noise.py sub-streams.
        rng = np.random.default_rng(self._seed ^ seed)
        base_step = self._speed / fps

        pos: FloatArray = (self._start if self._start is not None else self._goals[0]).copy()
        goal_idx = 0
        frames: list[EntityFrame] = [EntityFrame(frame=0, points={name: pos.tolist()})]
        for f in range(1, num_frames):
            step = base_step
            if self._pacing_jitter > 0.0:
                step *= 1.0 + self._pacing_jitter * float(rng.uniform(-1.0, 1.0))
            remaining = step
            # Consume the per-frame budget across as many legs as it reaches.
            while remaining > 0.0 and goal_idx < len(self._goals):
                goal = self._goals[goal_idx]
                to_goal = goal - pos
                dist = float(np.linalg.norm(to_goal))
                if dist <= self._tol or dist <= remaining:
                    pos = goal.copy()
                    remaining -= dist
                    goal_idx += 1
                else:
                    pos = pos + (remaining / dist) * to_goal
                    remaining = 0.0
            frames.append(EntityFrame(frame=f, points={name: pos.tolist()}))
        return frames

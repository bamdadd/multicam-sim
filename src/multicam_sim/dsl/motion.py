"""Movement DSL: a ``Path`` is a typed, time-parameterised 3D trajectory.

A path is a pydantic discriminated union on ``kind`` (mirroring
:data:`multicam_sim.occluders.OccluderUnion`): variants store *data* (endpoints,
control points, a centre + radius) and numpy appears only inside the evaluation
methods — the same idiom as the ``Camera`` contract type.

Geometry is parameterised by ``u in [0, 1]`` over the path's own extent
(:meth:`Path.point`). Timing is a separate axis: every node carries a
``duration`` in seconds, and :meth:`Path.at_time` maps a wall-clock time to a
point. Combinators compose both axes:

* ``a.then(b)``   — traverse ``a`` then ``b`` (durations add);
* ``p.repeat(n)`` — loop ``p`` ``n`` times;
* ``p.over(s)``   — rescale the whole trajectory to last ``s`` seconds;
* ``p.at_speed(v)`` — rescale so the traversal runs at ``v`` units/second.

:meth:`Path.compile_frames` turns a path into the per-frame named points the
Scene/manifest already expect — no contract change.
"""

from __future__ import annotations

from typing import Annotated, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..entities import EntityFrame
from ..geometry import FloatArray

Vec3 = tuple[float, float, float]


def _v(xyz: Vec3) -> FloatArray:
    return np.asarray(xyz, dtype=np.float64)


class _PathNode(BaseModel):
    """Shared behaviour for every path variant. Frozen; combinators return new
    nodes rather than mutating."""

    model_config = ConfigDict(frozen=True)

    #: Wall-clock seconds the (sub)trajectory takes. ``timed`` records whether
    #: the user pinned it explicitly (``.over`` / ``.at_speed``); if not, a bare
    #: path is stretched to fill the scene at :meth:`compile_frames`.
    duration: float = 1.0
    timed: bool = False

    # -- geometry (overridden per variant) --------------------------------- #
    def point(self, u: float) -> list[float]:
        """Point at normalised parameter ``u in [0, 1]`` along the geometry."""
        raise NotImplementedError

    def length(self) -> float:
        """Approximate geometric arc length (world units)."""
        raise NotImplementedError

    # -- timing (composites override) -------------------------------------- #
    def total_duration(self) -> float:
        return self.duration

    def at_time(self, t: float) -> list[float]:
        """Point at wall-clock time ``t`` seconds (clamped into the path)."""
        dur = self.total_duration()
        u = 0.0 if dur <= 0.0 else min(max(t / dur, 0.0), 1.0)
        return self.point(u)

    def _rescale(self, factor: float) -> PathUnion:
        """Return a copy whose durations are multiplied by ``factor``."""
        return self.model_copy(update={"duration": self.duration * factor})  # type: ignore[return-value]

    # -- combinators ------------------------------------------------------- #
    def then(self, other: PathUnion) -> SequencePath:
        """Traverse ``self`` then ``other`` (a single timed trajectory)."""
        return SequencePath(children=[self, other], timed=True)  # type: ignore[list-item]

    def repeat(self, n: int) -> RepeatPath:
        """Loop this trajectory ``n`` times back to back."""
        return RepeatPath(child=self, n=n, timed=True)  # type: ignore[arg-type]

    def over(self, seconds: float) -> PathUnion:
        """Rescale the whole trajectory to last exactly ``seconds`` seconds."""
        if seconds <= 0.0:
            raise ValueError("over(seconds): seconds must be > 0")
        total = self.total_duration()
        if total <= 0.0:
            return self.model_copy(update={"timed": True})  # type: ignore[return-value]
        return self._rescale(seconds / total).model_copy(update={"timed": True})

    def at_speed(self, v: float) -> PathUnion:
        """Rescale so the traversal runs at ``v`` world-units per second."""
        if v <= 0.0:
            raise ValueError("at_speed(v): v must be > 0")
        length = self.length()
        return self.over(length / v if length > 0 else self.total_duration())

    def compile_frames(
        self,
        fps: float,
        num_frames: int,
        name: str = "center",
    ) -> list[EntityFrame]:
        """Sample the trajectory into ``num_frames`` :class:`EntityFrame`s.

        An untimed path is first stretched to fill the whole scene duration
        ``(num_frames - 1) / fps``; a timed one keeps its duration and holds at
        its final point past the end.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        scene_seconds = (num_frames - 1) / fps if num_frames > 1 else 0.0
        traj: PathUnion = (
            cast("PathUnion", self)
            if self.timed or scene_seconds == 0.0
            else self.over(scene_seconds)
        )
        frames: list[EntityFrame] = []
        for f in range(num_frames):
            t = f / fps
            frames.append(EntityFrame(frame=f, points={name: traj.at_time(t)}))
        return frames


class LinearPath(_PathNode):
    """Straight segment from ``a`` to ``b``."""

    kind: Literal["linear"] = "linear"
    a: Vec3
    b: Vec3

    def point(self, u: float) -> list[float]:
        a, b = _v(self.a), _v(self.b)
        return (a + u * (b - a)).tolist()  # type: ignore[no-any-return]

    def length(self) -> float:
        return float(np.linalg.norm(_v(self.b) - _v(self.a)))


class CirclePath(_PathNode):
    """Circle of ``radius`` about ``center`` in the plane normal to ``axis``.

    ``u`` sweeps a full turn (``2*pi*u``). The start point (``u=0``) lies along
    the first basis vector orthogonal to ``axis``.
    """

    kind: Literal["circle"] = "circle"
    center: Vec3
    radius: float
    axis: Vec3 = (0.0, 0.0, 1.0)

    @field_validator("radius")
    @classmethod
    def _positive_radius(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("circle radius must be > 0")
        return value

    def _basis(self) -> tuple[FloatArray, FloatArray]:
        axis = _v(self.axis)
        axis = axis / np.linalg.norm(axis)
        ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, axis)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        return e1, e2

    def point(self, u: float) -> list[float]:
        e1, e2 = self._basis()
        theta = 2.0 * np.pi * u
        p = _v(self.center) + self.radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        return p.tolist()  # type: ignore[no-any-return]

    def length(self) -> float:
        return float(2.0 * np.pi * self.radius)


class WaypointPath(_PathNode):
    """Piecewise-linear path through ``points`` (>= 2), uniform in ``u`` per leg."""

    kind: Literal["waypoints"] = "waypoints"
    points: list[Vec3]

    @field_validator("points")
    @classmethod
    def _at_least_two(cls, value: list[Vec3]) -> list[Vec3]:
        if len(value) < 2:
            raise ValueError("waypoints needs at least 2 points")
        return value

    def point(self, u: float) -> list[float]:
        n_legs = len(self.points) - 1
        s = min(max(u, 0.0), 1.0) * n_legs
        leg = min(int(s), n_legs - 1)
        local = s - leg
        a, b = _v(self.points[leg]), _v(self.points[leg + 1])
        return (a + local * (b - a)).tolist()  # type: ignore[no-any-return]

    def length(self) -> float:
        pts = [_v(p) for p in self.points]
        return float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:], strict=True)))


class BezierPath(_PathNode):
    """Bezier curve of arbitrary degree over ``controls`` (>= 2), via de Casteljau."""

    kind: Literal["bezier"] = "bezier"
    controls: list[Vec3]

    @field_validator("controls")
    @classmethod
    def _at_least_two(cls, value: list[Vec3]) -> list[Vec3]:
        if len(value) < 2:
            raise ValueError("bezier needs at least 2 control points")
        return value

    def point(self, u: float) -> list[float]:
        pts = [_v(c) for c in self.controls]
        while len(pts) > 1:
            pts = [pts[i] + u * (pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        return pts[0].tolist()  # type: ignore[no-any-return]

    def length(self, samples: int = 64) -> float:
        prev = _v(self.controls[0])
        pts = [_v(c) for c in self.controls]
        total = 0.0
        for i in range(1, samples + 1):
            u = i / samples
            cur = pts
            while len(cur) > 1:
                cur = [cur[j] + u * (cur[j + 1] - cur[j]) for j in range(len(cur) - 1)]
            total += float(np.linalg.norm(cur[0] - prev))
            prev = cur[0]
        return total


class SequencePath(_PathNode):
    """Concatenation of child paths; wall-clock durations add."""

    kind: Literal["sequence"] = "sequence"
    children: list[PathUnion]

    @field_validator("children")
    @classmethod
    def _non_empty(cls, value: list[PathUnion]) -> list[PathUnion]:
        if not value:
            raise ValueError("sequence needs at least one child")
        return value

    def total_duration(self) -> float:
        return float(sum(c.total_duration() for c in self.children))

    def at_time(self, t: float) -> list[float]:
        elapsed = 0.0
        for child in self.children:
            d = child.total_duration()
            if t < elapsed + d or child is self.children[-1]:
                return child.at_time(t - elapsed)
            elapsed += d
        return self.children[-1].at_time(t)

    def point(self, u: float) -> list[float]:
        return self.at_time(u * self.total_duration())

    def length(self) -> float:
        return float(sum(c.length() for c in self.children))

    def _rescale(self, factor: float) -> PathUnion:
        return self.model_copy(update={"children": [c._rescale(factor) for c in self.children]})


class RepeatPath(_PathNode):
    """Loop ``child`` ``n`` times."""

    kind: Literal["repeat"] = "repeat"
    child: PathUnion
    n: int

    @field_validator("n")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repeat n must be >= 1")
        return value

    def total_duration(self) -> float:
        return self.n * self.child.total_duration()

    def at_time(self, t: float) -> list[float]:
        base = self.child.total_duration()
        if base <= 0.0:
            return self.child.at_time(0.0)
        local = min(t, self.total_duration())
        # keep the final endpoint at the very end rather than wrapping to start
        lap_t = base if local >= self.total_duration() else local % base
        return self.child.at_time(lap_t)

    def point(self, u: float) -> list[float]:
        return self.at_time(u * self.total_duration())

    def length(self) -> float:
        return self.n * self.child.length()

    def _rescale(self, factor: float) -> PathUnion:
        return self.model_copy(update={"child": self.child._rescale(factor)})


#: Discriminated union over every path variant (parallels ``OccluderUnion``).
PathUnion = Annotated[
    LinearPath | CirclePath | WaypointPath | BezierPath | SequencePath | RepeatPath,
    Field(discriminator="kind"),
]


class Path:
    """Namespace of constructors for the movement DSL (autocomplete entry point).

    ``Path.linear(...)`` etc. return concrete nodes; combinators live on the
    returned node (``.then`` / ``.repeat`` / ``.over`` / ``.at_speed``).
    """

    @staticmethod
    def linear(a: Vec3, b: Vec3) -> LinearPath:
        return LinearPath(a=a, b=b)

    @staticmethod
    def circle(center: Vec3, radius: float, axis: Vec3 = (0.0, 0.0, 1.0)) -> CirclePath:
        return CirclePath(center=center, radius=radius, axis=axis)

    @staticmethod
    def waypoints(points: list[Vec3]) -> WaypointPath:
        return WaypointPath(points=points)

    @staticmethod
    def bezier(controls: list[Vec3]) -> BezierPath:
        return BezierPath(controls=controls)


# Recursive discriminated unions (Sequence/Repeat reference PathUnion) need a
# rebuild once every variant is defined.
SequencePath.model_rebuild()
RepeatPath.model_rebuild()

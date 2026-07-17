"""Occlusion-pattern DSL: declare *which camera* to block *when*, get real geometry.

The manifest's occluders are static, global solids and ``visible`` is hard
geometric truth (in-front AND in-image AND sightline-unblocked). So this DSL does
**not** fake the boolean — it *places real occluder geometry* on the target
camera's sightline to the entity point at the middle of the requested window, and
lets :func:`multicam_sim.manifest.build_manifest` compute ``visible`` from that
geometry.

Consequences, kept explicit (see the DSL grammar section of ``DESIGN.md``):

* the achieved occlusion window is **emergent** — whatever the placed solid
  produces — not guaranteed byte-equal to the requested ``(frame0, frame1)``;
* ``.blocks(camera=i)`` is best-effort selective: the solid sits on camera ``i``'s
  ray, but a caller should verify the other cameras stay ``visible`` (the tests
  do). ``coverage`` is a monotonic difficulty knob feeding the continuous
  ``occ_frac`` readback (which the manifest quantises to eighths); it never
  changes the hard ``visible`` mask.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..cameras import Camera
from ..entities import EntityFrame
from ..occluders import Box, HandKeyframe, HandOccluder, OccluderUnion, Sphere

Vec3 = tuple[float, float, float]


class Occlusion(BaseModel):
    """A declarative occlusion pattern that compiles to one placed occluder.

    Build with a factory (:meth:`sphere` / :meth:`box` / :meth:`plane`), then set
    the target and window fluently: ``Occlusion.sphere(0.15).blocks(camera=1)
    .during((3, 7))``. :meth:`realize` turns it into real geometry given the
    scene's cameras and the target entity's frames.
    """

    model_config = ConfigDict(frozen=True)

    shape: Literal["sphere", "box", "plane"]
    size: float
    coverage: float = 1.0
    offset: float = 0.15  # fraction from the point toward the camera centre
    camera: int | None = None
    frames: tuple[int, int] | None = None
    seconds: tuple[float, float] | None = None
    entity: str | None = None  # default: the scene's first entity
    point_name: str = "center"

    # -- factories --------------------------------------------------------- #
    @classmethod
    def sphere(cls, size: float, coverage: float = 1.0) -> Occlusion:
        """A solid sphere of radius ``size`` (scaled by ``coverage``)."""
        if size <= 0.0:
            raise ValueError("occlusion size must be > 0")
        return cls(shape="sphere", size=size, coverage=coverage)

    @classmethod
    def box(cls, size: float, coverage: float = 1.0) -> Occlusion:
        """A solid axis-aligned cube of half-extent ``size``."""
        if size <= 0.0:
            raise ValueError("occlusion size must be > 0")
        return cls(shape="box", size=size, coverage=coverage)

    @classmethod
    def plane(cls, size: float, coverage: float = 1.0) -> Occlusion:
        """A thin square slab (a finite plane approximated as a flat box)."""
        if size <= 0.0:
            raise ValueError("occlusion size must be > 0")
        return cls(shape="plane", size=size, coverage=coverage)

    # -- fluent schedule --------------------------------------------------- #
    def blocks(self, camera: int) -> Occlusion:
        """Target the sightline of camera ``camera``."""
        return self.model_copy(update={"camera": camera})

    def during(self, frames: tuple[int, int]) -> Occlusion:
        """Aim the occlusion at the frame window ``(frame0, frame1)`` inclusive."""
        if self.seconds is not None:
            raise ValueError("occlusion already has a seconds window; use .during_seconds(...)")
        f0, f1 = frames
        if f0 > f1:
            raise ValueError("during(frames): frame0 must be <= frame1")
        return self.model_copy(update={"frames": frames})

    def during_seconds(self, t0: float, t1: float) -> Occlusion:
        """Aim the occlusion at the seconds window ``(t0, t1)`` inclusive.

        The window is converted to frames by :meth:`SceneBuilder.build` using the
        scene ``fps``, rounding to the nearest frame.
        """
        if self.frames is not None:
            raise ValueError("occlusion already has a frames window; use .during(...)")
        if t0 > t1:
            raise ValueError("during_seconds(t0, t1): t0 must be <= t1")
        return self.model_copy(update={"seconds": (t0, t1)})

    def on(self, entity: str, point_name: str = "center") -> Occlusion:
        """Target a named entity/point (default: first entity, point ``center``)."""
        return self.model_copy(update={"entity": entity, "point_name": point_name})

    def targeting(self, coverage: float) -> Occlusion:
        """Set the monotonic difficulty knob (scales occluder size)."""
        if coverage <= 0.0:
            raise ValueError("coverage must be > 0")
        return self.model_copy(update={"coverage": coverage})

    # -- compile ----------------------------------------------------------- #
    def realize(self, cameras: list[Camera], entity_frames: list[EntityFrame]) -> OccluderUnion:
        """Place real occluder geometry on the target camera's sightline.

        Sits the solid a small ``offset`` fraction from the entity point toward
        the camera centre, at the middle frame of the window, so the manifest
        then computes ``visible`` geometrically.
        """
        if self.camera is None:
            raise ValueError("occlusion has no target camera; call .blocks(camera=...)")
        if self.frames is None:
            raise ValueError("occlusion has no window; call .during((f0, f1))")
        if not 0 <= self.camera < len(cameras):
            raise ValueError(f"camera {self.camera} out of range")
        f0, f1 = self.frames
        mid = (f0 + f1) // 2
        frame = next((fr for fr in entity_frames if fr.frame == mid), None)
        if frame is None or self.point_name not in frame.points:
            raise ValueError(f"no point {self.point_name!r} at frame {mid}")

        point = np.asarray(frame.points[self.point_name], dtype=np.float64)
        centre = cameras[self.camera].centre()
        occ_centre = point + self.offset * (centre - point)
        extent = self.size * self.coverage

        if self.shape == "sphere":
            return Sphere(center=occ_centre.tolist(), radius=extent)
        if self.shape == "box":
            return Box(center=occ_centre.tolist(), half_extents=[extent, extent, extent])
        # plane: a thin slab oriented flat in z (finite plane approximation)
        return Box(center=occ_centre.tolist(), half_extents=[extent, extent, extent * 0.05])


class HandSweep(BaseModel):
    """A moving hand-proxy that sweeps across a camera's view of the target.

    Domain-neutral: a hand reaching over an item on a work surface / conveyor. The
    hand is a sphere placed a small ``offset`` fraction from the target toward the
    camera centre (so it sits BETWEEN target and camera and actually occludes),
    then swept laterally along the camera's right axis. Deterministic — the
    trajectory is fixed keyframes, NO RNG.

    Two modes:

    * ``pass`` (default) — enters one side, crosses centre at the window middle,
      exits the other side. The target's ``visible_fraction`` traces a U-shaped
      occlusion dose-response (1 -> ~0 -> 1); the hero artifact.
    * ``approach`` — enters one side and STOPS centred over the target by the end
      of the window, so ``visible_fraction`` is monotone non-increasing.

    Compiles (via :meth:`realize`) to a :class:`~multicam_sim.occluders.HandOccluder`
    on the scene; the same typed keyframes can drive a photoreal backend.
    """

    model_config = ConfigDict(frozen=True)

    radius: float
    span: float = 0.35  # lateral half-width of the sweep (world units)
    offset: float = 0.15  # fraction from target toward the camera centre
    mode: Literal["pass", "approach"] = "pass"
    camera: int | None = None
    frames: tuple[int, int] | None = None
    seconds: tuple[float, float] | None = None
    entity: str | None = None
    point_name: str = "center"

    @classmethod
    def sphere(cls, radius: float, *, span: float = 0.35) -> HandSweep:
        """A hand-proxy sphere of ``radius`` sweeping ``span`` to each side."""
        if radius <= 0.0:
            raise ValueError("hand radius must be > 0")
        if span <= 0.0:
            raise ValueError("hand span must be > 0")
        return cls(radius=radius, span=span)

    def blocks(self, camera: int) -> HandSweep:
        """Target the view of camera ``camera``."""
        return self.model_copy(update={"camera": camera})

    def during(self, frames: tuple[int, int]) -> HandSweep:
        """Sweep across the inclusive frame window ``(frame0, frame1)``."""
        if self.seconds is not None:
            raise ValueError("hand sweep already has a seconds window; use .during_seconds(...)")
        f0, f1 = frames
        if f0 >= f1:
            raise ValueError("during(frames): need frame0 < frame1 for a sweep")
        return self.model_copy(update={"frames": frames})

    def during_seconds(self, t0: float, t1: float) -> HandSweep:
        """Sweep across the inclusive seconds window (converted by scene fps)."""
        if self.frames is not None:
            raise ValueError("hand sweep already has a frames window; use .during(...)")
        if t0 >= t1:
            raise ValueError("during_seconds(t0, t1): need t0 < t1 for a sweep")
        return self.model_copy(update={"seconds": (t0, t1)})

    def on(self, entity: str, point_name: str = "center") -> HandSweep:
        """Target a named entity/point (default: first entity, point ``center``)."""
        return self.model_copy(update={"entity": entity, "point_name": point_name})

    def approaching(self) -> HandSweep:
        """Switch to ``approach`` mode: enter and STOP centred (monotone curve)."""
        return self.model_copy(update={"mode": "approach"})

    def realize(self, cameras: list[Camera], entity_frames: list[EntityFrame]) -> HandOccluder:
        """Compile to a :class:`HandOccluder` with deterministic keyframes."""
        if self.camera is None:
            raise ValueError("hand sweep has no target camera; call .blocks(camera=...)")
        if self.frames is None:
            raise ValueError("hand sweep has no window; call .during((f0, f1))")
        if not 0 <= self.camera < len(cameras):
            raise ValueError(f"camera {self.camera} out of range")
        f0, f1 = self.frames
        mid = (f0 + f1) // 2
        frame = next((fr for fr in entity_frames if fr.frame == mid), None)
        if frame is None or self.point_name not in frame.points:
            raise ValueError(f"no point {self.point_name!r} at frame {mid}")

        point = np.asarray(frame.points[self.point_name], dtype=np.float64)
        cam = cameras[self.camera]
        centre = cam.centre()
        base = point + self.offset * (centre - point)
        right = cam.rotation()[0]  # world-space right axis (unit)
        right = right / np.linalg.norm(right)
        step = self.span * right

        if self.mode == "approach":
            keyframes = [
                HandKeyframe(frame=f0, center=(base - step).tolist()),
                HandKeyframe(frame=f1, center=base.tolist()),
            ]
        else:  # pass
            keyframes = [
                HandKeyframe(frame=f0, center=(base - step).tolist()),
                HandKeyframe(frame=mid, center=base.tolist()),
                HandKeyframe(frame=f1, center=(base + step).tolist()),
            ]
        return HandOccluder(radius=self.radius, keyframes=keyframes)

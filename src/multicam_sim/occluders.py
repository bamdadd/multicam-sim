"""Occluders: an abstract solid plus Box and Sphere, with segment-blocking tests.

Visibility is a hard boolean: a world point is occluded from a camera when the
segment from the point to the camera centre passes through any occluder's solid
volume. This is the ``visible`` field of the manifest — the DLT contract. The
continuous ``occ_frac`` difficulty knob is computed separately in
:mod:`multicam_sim.manifest` and kept distinct from this boolean.

Occluders are static by default, but the union is open-closed over *time*:
:meth:`Occluder.at_frame` returns the concrete solid an occluder presents at a
given frame. Static solids (Box/Sphere) return themselves; a moving occluder
(:class:`HandOccluder`) returns a solid positioned along its typed, deterministic
trajectory. :func:`multicam_sim.manifest.build_manifest` resolves every occluder
to its ``at_frame`` solid *before* any sightline test, so the boolean/``occ_frac``
path only ever sees a static Box/Sphere and stays byte-identical when no mover
exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .geometry import (
    FloatArray,
    segment_intersects_aabb,
    segment_intersects_finite_cylinder,
    segment_intersects_sphere,
)


class Occluder(BaseModel, ABC):
    """Abstract occluding solid. Subclasses tag themselves with ``kind`` so the
    scene serialises as a discriminated union."""

    @abstractmethod
    def blocks_segment(self, a: FloatArray, b: FloatArray) -> bool:
        """Does the closed segment ``a->b`` pass through this solid?"""
        raise NotImplementedError

    def at_frame(self, frame: int) -> Occluder:
        """The concrete solid this occluder presents at ``frame``.

        Static occluders are frame-invariant and return themselves; a
        time-varying occluder overrides this to return a positioned solid. The
        manifest resolves through this hook before any sightline test.
        """
        return self


class Box(Occluder):
    """Axis-aligned box centred at ``center`` with per-axis ``half_extents``."""

    kind: Literal["box"] = "box"
    center: list[float]
    half_extents: list[float]

    def blocks_segment(self, a: FloatArray, b: FloatArray) -> bool:
        c = np.asarray(self.center, dtype=np.float64)
        h = np.asarray(self.half_extents, dtype=np.float64)
        return segment_intersects_aabb(a, b, c - h, c + h)


class Sphere(Occluder):
    """Solid sphere of ``radius`` centred at ``center``."""

    kind: Literal["sphere"] = "sphere"
    center: list[float]
    radius: float

    def blocks_segment(self, a: FloatArray, b: FloatArray) -> bool:
        c = np.asarray(self.center, dtype=np.float64)
        return segment_intersects_sphere(a, b, c, self.radius)



class Cylinder(Occluder):
    """Finite solid cylinder: ``center``, unit ``axis``, ``radius``, ``height``."""

    kind: Literal["cylinder"] = "cylinder"
    center: list[float]
    axis: list[float]
    radius: float
    height: float

    @field_validator("center", "axis")
    @classmethod
    def _check_vec3(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("must be a length-3 [x, y, z] vector")
        return value

    @field_validator("radius", "height")
    @classmethod
    def _check_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("cylinder radius and height must be > 0")
        return value

    @model_validator(mode="after")
    def _normalize_axis(self) -> Cylinder:
        axis = np.asarray(self.axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError("cylinder axis must be non-zero")
        # Store a unit axis so serialised scenes stay consistent.
        object.__setattr__(self, "axis", (axis / norm).tolist())
        return self

    def blocks_segment(self, a: FloatArray, b: FloatArray) -> bool:
        c = np.asarray(self.center, dtype=np.float64)
        u = np.asarray(self.axis, dtype=np.float64)
        return segment_intersects_finite_cylinder(a, b, c, u, self.radius, self.height)


class HandKeyframe(BaseModel):
    """One waypoint of a :class:`HandOccluder`: the hand-proxy centre at ``frame``."""

    frame: int
    center: list[float]

    @field_validator("center")
    @classmethod
    def _check_center_length(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("center must be a length-3 [x, y, z] vector")
        return value


class HandOccluder(Occluder):
    """A moving hand-proxy: a sphere of ``radius`` swept along typed keyframes.

    The trajectory is a deterministic piecewise-linear interpolation of the
    keyframe centres (no RNG), clamped to the endpoints outside the keyframe
    range. It is *time-varying*: it has no single sightline test, so
    :meth:`blocks_segment` is not defined on the mover itself — callers must
    resolve it to the static solid at a frame via :meth:`at_frame` (the manifest
    does this for every frame). The spec is renderer-agnostic, so a photoreal
    backend can consume the same keyframes.
    """

    kind: Literal["hand"] = "hand"
    radius: float
    keyframes: list[HandKeyframe]

    @field_validator("radius")
    @classmethod
    def _check_radius(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("hand radius must be > 0")
        return value

    @model_validator(mode="after")
    def _check_keyframes(self) -> HandOccluder:
        if not self.keyframes:
            raise ValueError("hand needs at least one keyframe")
        frames = [k.frame for k in self.keyframes]
        if frames != sorted(frames):
            raise ValueError("hand keyframes must be sorted by frame ascending")
        if len(frames) != len(set(frames)):
            raise ValueError("hand keyframe frames must be unique")
        return self

    def center_at(self, frame: int) -> FloatArray:
        """The hand centre at ``frame`` (piecewise-linear, endpoint-clamped)."""
        keys = self.keyframes
        if frame <= keys[0].frame:
            return np.asarray(keys[0].center, dtype=np.float64)
        if frame >= keys[-1].frame:
            return np.asarray(keys[-1].center, dtype=np.float64)
        for lo, hi in zip(keys, keys[1:], strict=False):
            if lo.frame <= frame <= hi.frame:
                span = hi.frame - lo.frame
                t = (frame - lo.frame) / span
                a = np.asarray(lo.center, dtype=np.float64)
                b = np.asarray(hi.center, dtype=np.float64)
                return a + t * (b - a)
        raise AssertionError("unreachable: frame lies within the keyframe range")

    def at_frame(self, frame: int) -> Occluder:
        """The static :class:`Sphere` the hand presents at ``frame``."""
        return Sphere(center=self.center_at(frame).tolist(), radius=self.radius)

    def blocks_segment(self, a: FloatArray, b: FloatArray) -> bool:
        raise NotImplementedError(
            "HandOccluder is time-varying; resolve it with at_frame(frame) and "
            "test the returned static solid's blocks_segment"
        )


#: Discriminated union for scene (de)serialisation.
OccluderUnion = Annotated[Box | Sphere | Cylinder | HandOccluder, Field(discriminator="kind")]

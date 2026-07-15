"""Occluders: an abstract solid plus Box and Sphere, with segment-blocking tests.

Visibility is a hard boolean: a world point is occluded from a camera when the
segment from the point to the camera centre passes through any occluder's solid
volume. This is the ``visible`` field of the manifest — the DLT contract. The
continuous ``occ_frac`` difficulty knob is computed separately in
:mod:`multicam_sim.manifest` and kept distinct from this boolean.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field

from .geometry import (
    FloatArray,
    segment_intersects_aabb,
    segment_intersects_sphere,
)


class Occluder(BaseModel, ABC):
    """Abstract occluding solid. Subclasses tag themselves with ``kind`` so the
    scene serialises as a discriminated union."""

    @abstractmethod
    def blocks_segment(self, a: FloatArray, b: FloatArray) -> bool:
        """Does the closed segment ``a->b`` pass through this solid?"""
        raise NotImplementedError


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


#: Discriminated union for scene (de)serialisation.
OccluderUnion = Annotated[Box | Sphere, Field(discriminator="kind")]

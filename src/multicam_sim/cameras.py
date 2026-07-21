"""Typed camera contract: Intrinsics + Camera (pydantic v2).

Stored fields are JSON-native (floats / lists) so a Camera round-trips through
the manifest without a numpy dependency in the schema. numpy appears only inside
the derived matrices. Convention mirrored from multicam-occlusion@59f4906
(see :mod:`multicam_sim.geometry`).
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .geometry import (
    FloatArray,
    camera_centre,
    camera_translation,
    intrinsic_matrix,
    look_at_rotation,
    project_point,
    projection_matrix,
    rotation_from_axis_angle,
)
from .noise import CalibrationDrift


class Intrinsics(BaseModel):
    """Pinhole intrinsics. ``cx``/``cy`` default to the image centre when built
    via :meth:`from_focal` or :meth:`from_fov`."""

    model_config = ConfigDict(frozen=True)

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_focal(cls, focal: float, width: int, height: int) -> Intrinsics:
        """Square-pixel intrinsics with principal point at the image centre."""
        return cls(
            fx=focal,
            fy=focal,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )

    @classmethod
    def from_fov(
        cls,
        fov_x_deg: float,
        width: int,
        height: int,
        fov_y_deg: float | None = None,
    ) -> Intrinsics:
        """Intrinsics from horizontal (and optional vertical) field of view.

        When ``fov_y_deg`` is omitted, pixels are assumed square (``fy == fx``).
        The principal point is placed at the image centre.
        """
        if width <= 0:
            raise ValueError("width must be > 0")
        if height <= 0:
            raise ValueError("height must be > 0")
        if not 0.0 < fov_x_deg < 180.0:
            raise ValueError("fov_x_deg must be in (0, 180)")

        fx = (width / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
        if fov_y_deg is None:
            fy = fx
        else:
            if not 0.0 < fov_y_deg < 180.0:
                raise ValueError("fov_y_deg must be in (0, 180)")
            fy = (height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)

        return cls(
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )

    def matrix(self) -> FloatArray:
        """The ``3x3`` intrinsic matrix ``K``."""
        return intrinsic_matrix(self.fx, self.fy, self.cx, self.cy)

    def fov_deg(self) -> tuple[float, float]:
        """Horizontal and vertical field of view in degrees.

        Inverts :meth:`from_fov`: ``fov = 2 * atan((size / 2) / f)`` per axis.
        For square pixels (``fy == fx``) the two angles differ only by the
        image aspect ratio.
        """
        fov_x = 2.0 * math.degrees(math.atan((self.width / 2.0) / self.fx))
        fov_y = 2.0 * math.degrees(math.atan((self.height / 2.0) / self.fy))
        return fov_x, fov_y


class Camera(BaseModel):
    """A pinhole camera: intrinsics plus world->camera rotation ``R`` and
    translation ``t = -R @ C``. ``R`` is stored row-major (rows = camera axes in
    world coordinates); ``t`` is the world->camera translation, not the centre."""

    model_config = ConfigDict(frozen=True)

    id: int
    intrinsics: Intrinsics
    R: list[list[float]]
    t: list[float]

    @field_validator("R")
    @classmethod
    def _check_rotation_shape(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("R must be a 3x3 matrix")
        return value

    @field_validator("t")
    @classmethod
    def _check_translation_shape(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("t must be a length-3 vector")
        return value

    @classmethod
    def look_at(
        cls,
        id: int,
        intrinsics: Intrinsics,
        eye: FloatArray,
        target: FloatArray,
        up: FloatArray | None = None,
    ) -> Camera:
        """Build a Camera from an eye/target/up, deriving ``R`` and ``t``."""
        from .geometry import UP_WORLD

        rotation = look_at_rotation(eye, target, UP_WORLD if up is None else up)
        translation = camera_translation(rotation, eye)
        return cls(
            id=id,
            intrinsics=intrinsics,
            R=rotation.tolist(),
            t=translation.tolist(),
        )

    def rotation(self) -> FloatArray:
        """World->camera rotation as a ``3x3`` array."""
        return np.asarray(self.R, dtype=np.float64)

    def translation(self) -> FloatArray:
        """World->camera translation ``t`` as a length-3 array."""
        return np.asarray(self.t, dtype=np.float64)

    def centre(self) -> FloatArray:
        """Camera centre ``C = -R^T @ t`` in world coordinates."""
        return camera_centre(self.rotation(), self.translation())

    def projection_matrix(self) -> FloatArray:
        """The ``3x4`` matrix ``P = K [R | t]``."""
        return projection_matrix(self.intrinsics.matrix(), self.rotation(), self.translation())

    def project(self, point3d: FloatArray) -> tuple[FloatArray, float]:
        """Project a world point: returns ``(uv, w)`` (``w > 0`` == in front)."""
        return project_point(self.projection_matrix(), point3d)

    def in_image(self, uv: FloatArray) -> bool:
        """Is pixel ``uv`` inside the image bounds ``[0, width) x [0, height)``?"""
        u, v = float(uv[0]), float(uv[1])
        return 0.0 <= u < self.intrinsics.width and 0.0 <= v < self.intrinsics.height

    def drifted(self, rng: np.random.Generator, drift: CalibrationDrift) -> Camera:
        """A seeded copy under calibration drift — the ASSUMED calibration.

        ``self`` stays the ground truth used to project; this returns a camera
        whose ``R``/``t`` and ``fx, fy, cx, cy`` are slightly perturbed per
        ``drift`` (see :class:`multicam_sim.noise.CalibrationDrift`). The rotation
        perturbation is a small-angle axis-angle applied on the left,
        ``R' = dR @ R``. Draws are taken in a FIXED order (rotation, translation,
        focal, principal point) so the result is reproducible for a given ``rng``
        regardless of which sigmas are zero. Returns ``self`` unchanged when the
        drift is inactive (so the caller can skip recording it).
        """
        if not drift.is_active:
            return self
        sigma_rad = math.radians(drift.rotation_sigma_deg)
        axis_angle = rng.normal(0.0, sigma_rad, size=3)
        delta_t = rng.normal(0.0, drift.translation_sigma, size=3)
        delta_focal = rng.normal(0.0, drift.focal_sigma_px, size=2)
        delta_pp = rng.normal(0.0, drift.principal_point_sigma_px, size=2)
        delta_rotation = rotation_from_axis_angle(np.asarray(axis_angle, dtype=np.float64))
        rotation = delta_rotation @ self.rotation()
        translation = self.translation() + delta_t
        intr = self.intrinsics
        assumed = Intrinsics(
            fx=intr.fx + float(delta_focal[0]),
            fy=intr.fy + float(delta_focal[1]),
            cx=intr.cx + float(delta_pp[0]),
            cy=intr.cy + float(delta_pp[1]),
            width=intr.width,
            height=intr.height,
        )
        return Camera(
            id=self.id,
            intrinsics=assumed,
            R=rotation.tolist(),
            t=translation.tolist(),
        )

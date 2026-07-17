"""Typed, seeded noise / calibration-drift knobs for the observation path.

Projection is exact by default: :meth:`multicam_sim.cameras.Camera.project`
puts a world point at its true pixel and the manifest records full-precision
``uv`` and true ``K, R, t``. These knobs add controlled, reproducible error on
top **without touching ground truth**:

* :class:`PixelNoise` perturbs the OBSERVED 2D keypoint (Gaussian, sigma in
  pixels) — only the recorded ``uv`` moves; ``in_view`` / ``visible`` / ``xyz_gt``
  stay truthful.
* :class:`CalibrationDrift` produces the slightly-wrong ASSUMED calibration a
  consumer receives (small perturbations to ``R``/``t`` and ``fx, fy, cx, cy``).
  The true calibration is still used to project and is recorded unchanged; the
  drifted one is recorded *separately and additively* in the manifest.

Every knob defaults to zero/off, so an all-zero :class:`NoiseModel` leaves the
manifest byte-identical to the noiseless output (no ``uv`` perturbation, no
assumed-calibration block). Seeding is explicit and local: a single ``seed``
drives ``numpy.random.default_rng`` sub-streams; the global numpy RNG state is
never read or mutated. This mirrors the seeded-generator convention already used
by the rig's ``height_jitter`` (:func:`multicam_sim.dsl.rig._jitter_offsets`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class PixelNoise(BaseModel):
    """Gaussian noise added to the observed 2D keypoint.

    ``sigma_px`` is the standard deviation, in pixels, added independently to the
    observed ``u`` and ``v``. It perturbs only the recorded pixel; the geometric
    flags (``in_view``, ``visible``) are computed from the true projection.
    """

    model_config = ConfigDict(frozen=True)

    sigma_px: float = 0.0

    @field_validator("sigma_px")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("sigma_px must be >= 0")
        return value

    @property
    def is_active(self) -> bool:
        """Whether this knob perturbs anything (a positive sigma)."""
        return self.sigma_px > 0.0


class CalibrationDrift(BaseModel):
    """Small, seeded perturbations to the ASSUMED calibration a consumer gets.

    The true calibration (used to project, recorded in the manifest's ``cameras``
    block) is untouched; drift produces a separately-recorded assumed ``K, R, t``.
    Each field is the standard deviation of an independent zero-mean Gaussian:

    * ``rotation_sigma_deg`` — magnitude of a small-angle axis-angle rotation
      applied to ``R``, in **degrees**. Each of the three rotation-vector
      components is drawn ``N(0, rotation_sigma_deg)`` (converted to radians),
      matching the codebase's degree convention for angles (e.g. FOV).
    * ``translation_sigma`` — per-axis offset added to the world->camera
      translation ``t``, in **scene units** (same units as the scene coordinates).
    * ``focal_sigma_px`` — per-axis offset added to ``fx`` and ``fy``, in pixels.
    * ``principal_point_sigma_px`` — per-axis offset added to ``cx`` and ``cy``,
      in pixels.
    """

    model_config = ConfigDict(frozen=True)

    rotation_sigma_deg: float = 0.0
    translation_sigma: float = 0.0
    focal_sigma_px: float = 0.0
    principal_point_sigma_px: float = 0.0

    @field_validator(
        "rotation_sigma_deg",
        "translation_sigma",
        "focal_sigma_px",
        "principal_point_sigma_px",
    )
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("drift sigmas must be >= 0")
        return value

    @property
    def is_active(self) -> bool:
        """Whether this knob perturbs anything (any positive sigma)."""
        return (
            self.rotation_sigma_deg > 0.0
            or self.translation_sigma > 0.0
            or self.focal_sigma_px > 0.0
            or self.principal_point_sigma_px > 0.0
        )


class NoiseModel(BaseModel):
    """The seeded noise/drift knobs threaded through the observation path.

    Defaults are all-off: an all-zero :class:`NoiseModel` yields byte-identical
    manifest output. ``seed`` drives independent ``numpy.random.default_rng``
    sub-streams for pixel noise and for each camera's drift, so results are fully
    reproducible and the two knobs never share state.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    pixel: PixelNoise = PixelNoise()
    drift: CalibrationDrift = CalibrationDrift()

    @property
    def is_active(self) -> bool:
        """Whether any knob perturbs anything."""
        return self.pixel.is_active or self.drift.is_active

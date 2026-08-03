"""Seeded domain randomization for the Scene builder (#41).

Synthetic-to-real robustness needs varied scenes. This module adds typed,
seeded randomization knobs — background colour, key-light direction/intensity,
and a count of distractor objects — so one scenario can be sampled
deterministically N ways.

The shape mirrors :mod:`multicam_sim.noise`: a frozen pydantic *spec* carries
the knob ranges, sampling is explicit and pure
(``numpy.random.default_rng(seed)``, never the global RNG), and everything is
additive and off by default so an un-randomized scene is byte-identical to
today's output.

* **Specs** (:class:`BackgroundSpec`, :class:`LightSpec`,
  :class:`DistractorSpec`) express every knob as a closed ``(min, max)``
  interval and reject an inverted interval (``min > max``) at validation time.
* :meth:`RandomizationSpec.sample` takes a seed and returns a concrete, typed,
  fully-determined :class:`RandomizationSample` — not a mutated scene. Draws
  are taken in a FIXED order (background channels, light intensity/azimuth/
  elevation, distractor count, then three coordinates per distractor), so the
  same spec and the same seed always produce a byte-identical sample.
* The concrete results (:class:`Background`, :class:`Light`) are ordinary
  scene-level fields; the builder applies a sample and records the
  :class:`RandomizationRecord` (spec + seed) on the scene as an optional
  sidecar, so a randomized run is reproducible from its own output.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

Vec3 = tuple[float, float, float]


def _check_interval(name: str, lo: float, hi: float) -> None:
    """Reject non-finite bounds and an inverted closed interval (``min > max``)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name}: bounds must be finite; got ({lo}, {hi})")
    if lo > hi:
        raise ValueError(f"{name}: min ({lo}) must be <= max ({hi})")


class BackgroundSpec(BaseModel):
    """Uniform RGB background colour range; channels sampled independently.

    Channels are in ``[0, 1]`` (the renderer's colour convention, e.g.
    ``pyrender.Scene(bg_color=...)``). The default interval stays dark, around
    the default black background. Background was already configurable per
    renderer via ``PyrenderBackend(bg=...)``; this knob makes it a scene-level,
    randomizable value (recorded on the scene) that takes precedence over that
    constructor default when set.
    """

    model_config = ConfigDict(frozen=True)

    rgb_min: Vec3 = (0.0, 0.0, 0.0)
    rgb_max: Vec3 = (0.2, 0.2, 0.2)

    @field_validator("rgb_min", "rgb_max")
    @classmethod
    def _unit_channels(cls, value: Vec3) -> Vec3:
        for channel in value:
            if not 0.0 <= channel <= 1.0:
                raise ValueError("background channels must lie in [0, 1]")
        return value

    @model_validator(mode="after")
    def _min_le_max(self) -> BackgroundSpec:
        for lo, hi in zip(self.rgb_min, self.rgb_max, strict=True):
            _check_interval("background rgb", lo, hi)
        return self


class LightSpec(BaseModel):
    """Key-light intensity and direction ranges.

    * ``intensity`` — renderer light intensity units; today's fixed key light
      is ``3.0``, so the default interval ``(2.0, 4.0)`` varies around it.
    * ``azimuth_deg`` / ``elevation_deg`` — the direction FROM the scene TO the
      light, in **degrees** (the codebase's angle convention): azimuth measured
      in the world XY plane from +X toward +Y, elevation above the XY plane.
      The light shines along the negative of that vector (see
      :meth:`Light.direction`).
    """

    model_config = ConfigDict(frozen=True)

    intensity: tuple[float, float] = (2.0, 4.0)
    azimuth_deg: tuple[float, float] = (0.0, 360.0)
    elevation_deg: tuple[float, float] = (30.0, 90.0)

    @field_validator("intensity")
    @classmethod
    def _non_negative_intensity(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] < 0.0:
            raise ValueError("light intensity must be >= 0")
        return value

    @field_validator("elevation_deg")
    @classmethod
    def _elevation_in_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        for angle in value:
            if not -90.0 <= angle <= 90.0:
                raise ValueError("light elevation must lie in [-90, 90] degrees")
        return value

    @model_validator(mode="after")
    def _min_le_max(self) -> LightSpec:
        _check_interval("light intensity", *self.intensity)
        _check_interval("light azimuth_deg", *self.azimuth_deg)
        _check_interval("light elevation_deg", *self.elevation_deg)
        return self


class DistractorSpec(BaseModel):
    """A count of static distractor objects placed uniformly in a world box.

    ``count`` is an inclusive integer interval; each sampled distractor is a
    non-target entity at a fixed position whose ``x``/``y``/``z`` coordinates
    (scene units) are drawn independently from the given intervals. The default
    box is a 4x4x1 metre region at floor level around the origin.
    """

    model_config = ConfigDict(frozen=True)

    count: tuple[int, int] = (1, 3)
    x: tuple[float, float] = (-2.0, 2.0)
    y: tuple[float, float] = (-2.0, 2.0)
    z: tuple[float, float] = (0.0, 1.0)

    @field_validator("count")
    @classmethod
    def _non_negative_count(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 0:
            raise ValueError("distractor count must be >= 0")
        return value

    @model_validator(mode="after")
    def _min_le_max(self) -> DistractorSpec:
        if self.count[0] > self.count[1]:
            raise ValueError(
                f"distractor count: min ({self.count[0]}) must be <= max ({self.count[1]})"
            )
        _check_interval("distractor x", *self.x)
        _check_interval("distractor y", *self.y)
        _check_interval("distractor z", *self.z)
        return self


class Background(BaseModel):
    """A concrete, sampled background: one RGB colour, channels in ``[0, 1]``."""

    model_config = ConfigDict(frozen=True)

    rgb: Vec3

    @field_validator("rgb")
    @classmethod
    def _unit_channels(cls, value: Vec3) -> Vec3:
        for channel in value:
            if not 0.0 <= channel <= 1.0:
                raise ValueError("background channels must lie in [0, 1]")
        return value


class Light(BaseModel):
    """A concrete, sampled key light: intensity plus direction.

    ``azimuth_deg`` / ``elevation_deg`` give the direction FROM the scene TO
    the light (azimuth in the world XY plane from +X toward +Y, elevation above
    the XY plane); :meth:`direction` is the unit vector the light shines along.
    """

    model_config = ConfigDict(frozen=True)

    intensity: float
    azimuth_deg: float
    elevation_deg: float

    def direction(self) -> Vec3:
        """Unit vector the light travels along (the negated scene->light vector)."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        return (
            -math.cos(el) * math.cos(az),
            -math.cos(el) * math.sin(az),
            -math.sin(el),
        )


class RandomizationSample(BaseModel):
    """The concrete, fully-determined result of sampling a spec once.

    Pure data: applying it to a scene is the builder's job. ``None``/empty
    members correspond to knobs that were absent from the spec.
    """

    model_config = ConfigDict(frozen=True)

    background: Background | None = None
    light: Light | None = None
    distractor_positions: list[Vec3] = []


class RandomizationSpec(BaseModel):
    """The seeded randomization knobs for one scenario.

    Every knob is optional and off by default: an all-``None`` spec samples to
    an empty :class:`RandomizationSample`. ``seed`` is NOT stored here — it is
    passed to :meth:`sample` and recorded alongside the spec in the scene's
    :class:`RandomizationRecord` sidecar.
    """

    model_config = ConfigDict(frozen=True)

    background: BackgroundSpec | None = None
    light: LightSpec | None = None
    distractors: DistractorSpec | None = None

    def sample(self, seed: int) -> RandomizationSample:
        """Draw one concrete sample under ``seed`` (pure and deterministic).

        Uses ``numpy.random.default_rng(seed)`` with a fixed draw order, so the
        same spec and seed produce a byte-identical sample and different seeds
        produce different ones.
        """
        rng = np.random.default_rng(seed)

        background = None
        if self.background is not None:
            rgb = rng.uniform(self.background.rgb_min, self.background.rgb_max)
            background = Background(rgb=(float(rgb[0]), float(rgb[1]), float(rgb[2])))

        light = None
        if self.light is not None:
            light = Light(
                intensity=float(rng.uniform(*self.light.intensity)),
                azimuth_deg=float(rng.uniform(*self.light.azimuth_deg)),
                elevation_deg=float(rng.uniform(*self.light.elevation_deg)),
            )

        positions: list[Vec3] = []
        if self.distractors is not None:
            spec = self.distractors
            n = int(rng.integers(spec.count[0], spec.count[1] + 1))
            for _ in range(n):
                lo = [spec.x[0], spec.y[0], spec.z[0]]
                hi = [spec.x[1], spec.y[1], spec.z[1]]
                xyz = rng.uniform(lo, hi)
                positions.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

        return RandomizationSample(
            background=background, light=light, distractor_positions=positions
        )


class RandomizationRecord(BaseModel):
    """Provenance sidecar: the spec and seed that produced a scene.

    Stored on :class:`~multicam_sim.scene.Scene` as an optional field — absent
    by default, present and round-tripping when randomization was used — so a
    randomized run is reproducible from its own output:
    ``record.spec.sample(record.seed)`` regenerates the exact applied sample.
    """

    model_config = ConfigDict(frozen=True)

    spec: RandomizationSpec
    seed: int

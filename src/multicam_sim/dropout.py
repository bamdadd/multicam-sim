"""Typed, seeded per-camera **sensor dropout** (frame blackout) for the
observation path.

Stream-level and DISTINCT from scene occlusion: a dropped frame is a *sensor*
failure — the camera delivered nothing that frame — not a geometric occlusion. So
a dropped observation is *blanked* (``visible`` False, ``uv`` zeroed, the
occlusion fields absent), giving a downstream reader a genuine coverage gap. It is
never a zero occlusion score: dropout is not "fully visible with a 0.0 occluder".

Seeding mirrors :mod:`multicam_sim.noise`: one ``seed`` drives an independent
``numpy.random.default_rng`` sub-stream **per camera** (``[seed, tag, camera_id]``),
so a camera's drop schedule is byte-reproducible, independent of camera order and
of the pixel-noise / calibration-drift streams, and never reads or mutates the
global RNG. A fixed seed yields an identical schedule every run; a different seed
yields a different one.

The schedule itself is a pure function (:func:`dropped_frames`) with no Blender,
no manifest and no scene dependency, so it is unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

#: Per-camera RNG sub-stream tag for dropout, ``"DRO"`` as hex — the same ascii
#: convention as ``noise``'s ``_PIXEL_NOISE_STREAM`` / ``_DRIFT_STREAM``.
DROPOUT_STREAM: int = 0x44524F


class SensorDropout(BaseModel):
    """Seeded per-camera frame-drop schedule config.

    Each frame of each camera is independently dropped with probability
    ``drop_prob`` (a per-frame Bernoulli trial drawn from the camera's own seeded
    sub-stream). ``drop_prob = 0.0`` (the default) is off: no frame is ever
    dropped and the manifest is byte-identical to the no-dropout output.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 0
    drop_prob: float = 0.0

    @field_validator("drop_prob")
    @classmethod
    def _prob_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("drop_prob must be in [0, 1]")
        return value

    @property
    def is_active(self) -> bool:
        """Whether this drops anything (a positive per-frame probability)."""
        return self.drop_prob > 0.0


def dropped_frames(config: SensorDropout, camera_id: int, num_frames: int) -> tuple[int, ...]:
    """The sorted frame indices camera ``camera_id`` drops, for ``num_frames``.

    Deterministic in ``(config.seed, camera_id)``: the camera's own
    ``default_rng([seed, DROPOUT_STREAM, camera_id])`` draws one uniform per frame
    and the frame is dropped where the draw is below ``drop_prob``. Returns an
    empty tuple when the config is inactive, so the caller can skip all dropout
    work (and keep byte-identical output) with a cheap emptiness check.
    """
    if not config.is_active or num_frames <= 0:
        return ()
    rng = np.random.default_rng([int(config.seed), DROPOUT_STREAM, int(camera_id)])
    draws = rng.random(num_frames)
    return tuple(f for f in range(num_frames) if float(draws[f]) < config.drop_prob)

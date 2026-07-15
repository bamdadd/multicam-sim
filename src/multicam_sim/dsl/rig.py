"""Camera-rig DSL: fluent builders that emit ``list[Camera]``.

Every builder goes through :meth:`multicam_sim.cameras.Camera.look_at` (and, for
``custom``, stores caller-supplied extrinsics verbatim) so the RDF / Z-up /
``t = -R@C`` convention is never re-derived here — the contract lives in
:mod:`multicam_sim.geometry` and this layer only places eyes and targets.

Intrinsics are given as exactly one of ``focal`` (pixels) or ``fov_deg``
(horizontal field of view); the principal point sits at the image centre.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..cameras import Camera, Intrinsics

Vec3 = tuple[float, float, float]
#: A uniform height (scalar) or one height per camera (sequence of length ``n``).
Heights = float | Sequence[float]


def _base_heights(n: int, height: Heights) -> list[float]:
    """Resolve ``height`` to ``n`` per-camera base heights.

    A scalar broadcasts to every camera (uniform, the original behaviour); a
    sequence must have length ``n`` (one height per camera) or it is an error.
    """
    if isinstance(height, Sequence):
        base = [float(h) for h in height]
        if len(base) != n:
            raise ValueError(f"height sequence has length {len(base)}, expected n={n}")
        return base
    return [float(height)] * n


def _jitter_offsets(n: int, height_jitter: float, seed: int) -> list[float]:
    """Deterministic per-camera vertical offsets in ``[-jitter, +jitter]``.

    Seeded (``numpy.random.default_rng(seed)``) so a rig is fully reproducible;
    there is no unseeded RNG. Returns zeros when ``height_jitter == 0``.
    """
    if height_jitter < 0.0:
        raise ValueError("height_jitter must be >= 0")
    if height_jitter == 0.0:
        return [0.0] * n
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.uniform(-height_jitter, height_jitter, size=n)]


def _intrinsics(
    width: int,
    height: int,
    focal: float | None,
    fov_deg: float | None,
) -> Intrinsics:
    """Square-pixel intrinsics from exactly one of ``focal`` or ``fov_deg``."""
    if (focal is None) == (fov_deg is None):
        raise ValueError("give exactly one of focal= or fov_deg=")
    if fov_deg is not None:
        if not 0.0 < fov_deg < 180.0:
            raise ValueError("fov_deg must be in (0, 180)")
        focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    assert focal is not None
    if focal <= 0.0:
        raise ValueError("focal must be > 0")
    return Intrinsics.from_focal(focal, width, height)


class CameraRig:
    """Namespace of camera-array constructors (autocomplete entry point)."""

    @staticmethod
    def ring(
        n: int,
        radius: float,
        height: Heights,
        look_at: Vec3,
        *,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
        height_jitter: float = 0.0,
        seed: int = 0,
    ) -> list[Camera]:
        """``n`` cameras evenly spaced on a horizontal ring, all facing ``look_at``.

        Eye ``i`` sits at ``(radius*cos t, radius*sin t, z_i)`` with
        ``t = 2*pi*i/n`` — the ring convention of ``build_smoke_scene``. ``height``
        is either a scalar (uniform ``z``, the original behaviour) or a sequence
        of ``n`` per-camera heights; ``height_jitter`` adds a seeded, reproducible
        vertical offset on top. Heights only change eye ``z`` — ``R``/``t`` still
        follow the exact convention (``t = -R@C``, RDF, Z-up).

        Varied heights are both realism (real rigs sit at different levels) and
        better conditioning: a perfectly coplanar camera set is a degenerate,
        weaker triangulation geometry, and spreading ``z`` de-degenerates it.
        """
        if n < 1:
            raise ValueError("ring needs n >= 1 cameras")
        if radius <= 0.0:
            raise ValueError("ring radius must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        target = np.asarray(look_at, dtype=np.float64)
        heights = _base_heights(n, height)
        jitter = _jitter_offsets(n, height_jitter, seed)
        cams: list[Camera] = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            eye = np.array(
                [radius * math.cos(angle), radius * math.sin(angle), heights[i] + jitter[i]],
                dtype=np.float64,
            )
            cams.append(Camera.look_at(i, intrinsics, eye, target))
        return cams

    @staticmethod
    def line(
        n: int,
        start: Vec3,
        end: Vec3,
        look_at: Vec3,
        *,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
        height: Heights | None = None,
        height_jitter: float = 0.0,
        seed: int = 0,
    ) -> list[Camera]:
        """``n`` cameras evenly spaced from ``start`` to ``end``, all facing ``look_at``.

        By default the eye ``z`` is interpolated from ``start``/``end`` (original
        behaviour). Pass ``height`` — a scalar or a sequence of ``n`` — to
        *override* each camera's ``z``; ``height_jitter`` adds a seeded,
        reproducible vertical offset on top. Only eye ``z`` changes; ``R``/``t``
        keep the exact convention. Spreading the cameras off a single plane is
        realism and better-conditioned (less coplanar/degenerate) triangulation.
        """
        if n < 1:
            raise ValueError("line needs n >= 1 cameras")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        target = np.asarray(look_at, dtype=np.float64)
        override = None if height is None else _base_heights(n, height)
        jitter = _jitter_offsets(n, height_jitter, seed)
        cams: list[Camera] = []
        for i in range(n):
            frac = 0.0 if n == 1 else i / (n - 1)
            eye = a + frac * (b - a)
            if override is not None:
                eye[2] = override[i]
            eye[2] += jitter[i]
            cams.append(Camera.look_at(i, intrinsics, eye, target))
        return cams

    @staticmethod
    def custom(
        extrinsics: list[tuple[list[list[float]], list[float]]],
        *,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
    ) -> list[Camera]:
        """Cameras from caller-supplied ``(R, t)`` world->camera extrinsics.

        ``R``/``t`` are stored verbatim (validated by :class:`Camera`); use this
        when you already have poses in the contract convention and don't want a
        look-at re-derivation.
        """
        if not extrinsics:
            raise ValueError("custom needs at least one (R, t) pair")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        return [
            Camera(id=i, intrinsics=intrinsics, R=R, t=t) for i, (R, t) in enumerate(extrinsics)
        ]

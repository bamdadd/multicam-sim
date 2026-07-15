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

import numpy as np

from ..cameras import Camera, Intrinsics

Vec3 = tuple[float, float, float]


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
        height: float,
        look_at: Vec3,
        *,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
    ) -> list[Camera]:
        """``n`` cameras evenly spaced on a horizontal ring, all facing ``look_at``.

        Eye ``i`` sits at ``(radius*cos t, radius*sin t, height)`` with
        ``t = 2*pi*i/n`` — the same ring convention as ``build_smoke_scene``.
        """
        if n < 1:
            raise ValueError("ring needs n >= 1 cameras")
        if radius <= 0.0:
            raise ValueError("ring radius must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        target = np.asarray(look_at, dtype=np.float64)
        cams: list[Camera] = []
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            eye = np.array(
                [radius * math.cos(angle), radius * math.sin(angle), height],
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
    ) -> list[Camera]:
        """``n`` cameras evenly spaced from ``start`` to ``end``, all facing ``look_at``."""
        if n < 1:
            raise ValueError("line needs n >= 1 cameras")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        target = np.asarray(look_at, dtype=np.float64)
        cams: list[Camera] = []
        for i in range(n):
            frac = 0.0 if n == 1 else i / (n - 1)
            eye = a + frac * (b - a)
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

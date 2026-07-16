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
from pydantic import BaseModel

from ..cameras import Camera, Intrinsics
from ..geometry import UP_WORLD

Vec3 = tuple[float, float, float]
#: A uniform height (scalar) or one height per camera (sequence of length ``n``).
Heights = float | Sequence[float]


class StationView(BaseModel):
    """One camera station: an eye ``position`` looking at its OWN ``look_at``.

    Optional per-station ``focal``/``fov_deg`` and ``width``/``height_px`` override
    the rig-wide defaults, so a single :meth:`CameraRig.stations` preset serves
    both non-overlapping MTMC (separated stations, disjoint FOVs) and
    heterogeneous fusion (e.g. one wide camera framing a person, another close +
    zoomed framing items on a bench). Give at most one of ``focal``/``fov_deg``.
    """

    position: Vec3
    look_at: Vec3
    focal: float | None = None
    fov_deg: float | None = None
    width: int | None = None
    height_px: int | None = None


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
    def stereo(
        baseline: float,
        look_at: Vec3,
        *,
        center: Vec3 = (0.0, 0.0, 0.0),
        height: Heights,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
        height_jitter: float = 0.0,
        seed: int = 0,
    ) -> list[Camera]:
        """A 2-camera horizontal stereo pair, both facing ``look_at``.

        The rig centre is ``center``; the view direction is ``look_at - center``.
        The two eyes sit at ``center ± baseline/2 * right`` where ``right`` is the
        horizontal axis ``forward × up_world`` (perpendicular to the view
        direction). ``height`` sets the absolute eye ``z`` for each camera, and
        ``height_jitter`` adds a seeded, reproducible vertical offset on top.
        """
        if baseline <= 0.0:
            raise ValueError("stereo baseline must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        target = np.asarray(look_at, dtype=np.float64)
        center_arr = np.asarray(center, dtype=np.float64)
        heights = _base_heights(2, height)
        jitter = _jitter_offsets(2, height_jitter, seed)

        forward = target - center_arr
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-12:
            raise ValueError("stereo look_at must differ from center")
        forward = forward / forward_norm
        right = np.cross(forward, UP_WORLD)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-12:
            raise ValueError("stereo view direction must not be parallel to world up")
        right = right / right_norm

        cams: list[Camera] = []
        for side, sign in enumerate((-1.0, 1.0)):
            eye = center_arr + sign * (baseline / 2.0) * right
            eye[2] = heights[side] + jitter[side]
            cams.append(Camera.look_at(side, intrinsics, eye, target))
        return cams

    @staticmethod
    def arc(
        n: int,
        radius: float,
        start_angle: float,
        end_angle: float,
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
        """``n`` cameras evenly spaced along a circular arc, all facing ``look_at``.

        Endpoints are inclusive: ``n=2`` places one camera at ``start_angle`` and
        one at ``end_angle``; ``n=1`` places a single camera at ``start_angle``.
        Angles are radians, matching the convention of :meth:`ring`. ``height``
        and ``height_jitter`` behave exactly as in :meth:`ring`.
        """
        if n < 1:
            raise ValueError("arc needs n >= 1 cameras")
        if radius <= 0.0:
            raise ValueError("arc radius must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        target = np.asarray(look_at, dtype=np.float64)
        heights = _base_heights(n, height)
        jitter = _jitter_offsets(n, height_jitter, seed)
        cams: list[Camera] = []
        span = end_angle - start_angle
        for i in range(n):
            angle = start_angle if n == 1 else start_angle + span * i / (n - 1)
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

    @staticmethod
    def stations(
        views: Sequence[StationView],
        *,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
    ) -> list[Camera]:
        """Cameras at SEPARATED stations, each looking at its OWN target.

        Unlike :meth:`ring` (one target, overlapping views) this is the
        non-overlapping / heterogeneous preset: every :class:`StationView` gives
        its own eye ``position`` and ``look_at``, and may override the rig-wide
        intrinsics with its own ``focal``/``fov_deg`` and ``width``/``height_px``.
        Cameras are built through :meth:`Camera.look_at`, so the RDF / Z-up /
        ``t = -R@C`` convention is never re-derived. ``width``/``height_px`` are
        the shared defaults; ``focal``/``fov_deg`` are shared defaults used only
        for stations that don't set their own.

        Two modes off one preset:

        * MTMC: separate stations with disjoint FOVs (an object is in at most one
          view at a time; the gap between them is a genuine blind interval).
        * fusion: co-located-ish stations with different targets/zoom, so different
          entities fall in different cameras' ``in_view`` — the manifest's
          per-entity per-camera ``in_view`` captures "human in A not B, items in B
          not A" with no schema change.
        """
        if not views:
            raise ValueError("stations needs at least one StationView")
        cams: list[Camera] = []
        for i, view in enumerate(views):
            has_own = view.focal is not None or view.fov_deg is not None
            eff_focal = view.focal if has_own else focal
            eff_fov = view.fov_deg if has_own else fov_deg
            eff_w = view.width if view.width is not None else width
            eff_h = view.height_px if view.height_px is not None else height_px
            intrinsics = _intrinsics(eff_w, eff_h, eff_focal, eff_fov)
            eye = np.asarray(view.position, dtype=np.float64)
            target = np.asarray(view.look_at, dtype=np.float64)
            cams.append(Camera.look_at(i, intrinsics, eye, target))
        return cams

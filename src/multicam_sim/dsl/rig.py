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
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from ..cameras import Camera, Intrinsics
from ..geometry import UP_WORLD

Vec3 = tuple[float, float, float]
#: A uniform height (scalar) or one height per camera (sequence of length ``n``).
Heights = float | Sequence[float]


class PoseOverride(BaseModel):
    """Explicit per-camera pose override: an eye ``position`` looking at ``look_at``.

    Used by the parametric rig presets (:meth:`CameraRig.ring`, :meth:`stereo`,
    :meth:`arc`, :meth:`line`, :meth:`stations`) to replace the computed pose for
    one camera while leaving unspecified cameras on the preset path.
    """

    position: Vec3
    look_at: Vec3


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


#: Per-camera overrides: dense list (length ``n``, ``None`` = keep preset) or sparse
#: mapping keyed by camera index.
PoseOverrides = Sequence[PoseOverride | None] | Mapping[int, PoseOverride]


def _resolve_overrides(
    n: int,
    overrides: PoseOverrides | None,
) -> list[PoseOverride | None]:
    """Validate and normalize per-camera pose overrides to a length-``n`` list.

    A :class:`Sequence` must have length ``n`` (``None`` entries keep the preset
    pose for that camera). A :class:`Mapping` must use integer keys in
    ``[0, n)``; missing keys keep the preset pose. Any mismatch raises a clear
    ``ValueError`` matching the module's existing validation style.
    """
    if overrides is None:
        return [None] * n
    if isinstance(overrides, Mapping):
        out: list[PoseOverride | None] = [None] * n
        for key in overrides:
            if not isinstance(key, int) or not (0 <= key < n):
                raise ValueError(f"override key {key} out of range for n={n} cameras")
            out[key] = overrides[key]
        return out
    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        if len(overrides) != n:
            raise ValueError(f"overrides sequence has length {len(overrides)}, expected n={n}")
        return list(overrides)
    raise ValueError("overrides must be a sequence of length n or a mapping keyed by camera index")


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
        overrides: PoseOverrides | None = None,
    ) -> list[Camera]:
        """``n`` cameras evenly spaced on a horizontal ring, all facing ``look_at``.

        Eye ``i`` sits at ``(radius*cos t, radius*sin t, z_i)`` with
        ``t = 2*pi*i/n`` — the ring convention of ``build_smoke_scene``. ``height``
        is either a scalar (uniform ``z``, the original behaviour) or a sequence
        of ``n`` per-camera heights; ``height_jitter`` adds a seeded, reproducible
        vertical offset on top. Heights only change eye ``z`` — ``R``/``t`` still
        follow the exact convention (``t = -R@C``, RDF, Z-up).

        ``overrides`` replaces the computed pose for individual cameras. Pass a
        sequence of length ``n`` (``None`` keeps the preset pose for that slot) or
        a mapping from camera index to :class:`PoseOverride`; unspecified cameras
        keep the ring pose byte-for-byte.

        Varied heights are both realism (real rigs sit at different levels) and
        better conditioning: a perfectly coplanar camera set is a degenerate,
        weaker triangulation geometry, and spreading ``z`` de-degenerates it.
        """
        if n < 1:
            raise ValueError("ring needs n >= 1 cameras")
        if radius <= 0.0:
            raise ValueError("ring radius must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        base_target = np.asarray(look_at, dtype=np.float64)
        heights = _base_heights(n, height)
        jitter = _jitter_offsets(n, height_jitter, seed)
        resolved = _resolve_overrides(n, overrides)
        cams: list[Camera] = []
        for i in range(n):
            ov = resolved[i]
            if ov is not None:
                eye = np.asarray(ov.position, dtype=np.float64)
                target = np.asarray(ov.look_at, dtype=np.float64)
            else:
                angle = 2.0 * math.pi * i / n
                eye = np.array(
                    [radius * math.cos(angle), radius * math.sin(angle), heights[i] + jitter[i]],
                    dtype=np.float64,
                )
                target = base_target
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
        overrides: PoseOverrides | None = None,
    ) -> list[Camera]:
        """A 2-camera horizontal stereo pair, both facing ``look_at``.

        The rig centre is ``center``; the view direction is ``look_at - center``.
        The two eyes sit at ``center ± baseline/2 * right`` where ``right`` is the
        horizontal axis ``forward × up_world`` (perpendicular to the view
        direction). ``height`` sets the absolute eye ``z`` for each camera, and
        ``height_jitter`` adds a seeded, reproducible vertical offset on top.

        ``overrides`` replaces the computed pose for either camera. Pass a sequence
        of length 2 (``None`` keeps the preset pose) or a mapping from camera index
        to :class:`PoseOverride`.
        """
        if baseline <= 0.0:
            raise ValueError("stereo baseline must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        base_target = np.asarray(look_at, dtype=np.float64)
        center_arr = np.asarray(center, dtype=np.float64)
        heights = _base_heights(2, height)
        jitter = _jitter_offsets(2, height_jitter, seed)
        resolved = _resolve_overrides(2, overrides)

        forward = base_target - center_arr
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
            ov = resolved[side]
            if ov is not None:
                eye = np.asarray(ov.position, dtype=np.float64)
                target = np.asarray(ov.look_at, dtype=np.float64)
            else:
                eye = center_arr + sign * (baseline / 2.0) * right
                eye[2] = heights[side] + jitter[side]
                target = base_target
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
        overrides: PoseOverrides | None = None,
    ) -> list[Camera]:
        """``n`` cameras evenly spaced along a circular arc, all facing ``look_at``.

        Endpoints are inclusive: ``n=2`` places one camera at ``start_angle`` and
        one at ``end_angle``; ``n=1`` places a single camera at ``start_angle``.
        Angles are radians, matching the convention of :meth:`ring`. ``height``
        and ``height_jitter`` behave exactly as in :meth:`ring`.

        ``overrides`` replaces the computed pose for individual cameras; see
        :meth:`ring` for the accepted shapes.
        """
        if n < 1:
            raise ValueError("arc needs n >= 1 cameras")
        if radius <= 0.0:
            raise ValueError("arc radius must be > 0")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        base_target = np.asarray(look_at, dtype=np.float64)
        heights = _base_heights(n, height)
        jitter = _jitter_offsets(n, height_jitter, seed)
        resolved = _resolve_overrides(n, overrides)
        cams: list[Camera] = []
        span = end_angle - start_angle
        for i in range(n):
            ov = resolved[i]
            if ov is not None:
                eye = np.asarray(ov.position, dtype=np.float64)
                target = np.asarray(ov.look_at, dtype=np.float64)
            else:
                angle = start_angle if n == 1 else start_angle + span * i / (n - 1)
                eye = np.array(
                    [radius * math.cos(angle), radius * math.sin(angle), heights[i] + jitter[i]],
                    dtype=np.float64,
                )
                target = base_target
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
        overrides: PoseOverrides | None = None,
    ) -> list[Camera]:
        """``n`` cameras evenly spaced from ``start`` to ``end``, all facing ``look_at``.

        By default the eye ``z`` is interpolated from ``start``/``end`` (original
        behaviour). Pass ``height`` — a scalar or a sequence of ``n`` — to
        *override* each camera's ``z``; ``height_jitter`` adds a seeded,
        reproducible vertical offset on top. Only eye ``z`` changes; ``R``/``t``
        keep the exact convention. Spreading the cameras off a single plane is
        realism and better-conditioned (less coplanar/degenerate) triangulation.

        ``overrides`` replaces the computed pose for individual cameras; see
        :meth:`ring` for the accepted shapes.
        """
        if n < 1:
            raise ValueError("line needs n >= 1 cameras")
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        base_target = np.asarray(look_at, dtype=np.float64)
        z_override = None if height is None else _base_heights(n, height)
        jitter = _jitter_offsets(n, height_jitter, seed)
        resolved = _resolve_overrides(n, overrides)
        cams: list[Camera] = []
        for i in range(n):
            ov = resolved[i]
            if ov is not None:
                eye = np.asarray(ov.position, dtype=np.float64)
                target = np.asarray(ov.look_at, dtype=np.float64)
            else:
                frac = 0.0 if n == 1 else i / (n - 1)
                eye = a + frac * (b - a)
                if z_override is not None:
                    eye[2] = z_override[i]
                eye[2] += jitter[i]
                target = base_target
            cams.append(Camera.look_at(i, intrinsics, eye, target))
        return cams

    @staticmethod
    def grid(
        rows: int,
        cols: int,
        corner: Vec3,
        right: Vec3,
        down: Vec3,
        look_at: Vec3,
        *,
        width: int,
        height_px: int,
        focal: float | None = None,
        fov_deg: float | None = None,
        overrides: PoseOverrides | None = None,
    ) -> list[Camera]:
        """A planar ``rows`` x ``cols`` grid of cameras all facing ``look_at``.

        Camera ``(r, c)`` sits at ``corner + c * right + r * down`` — a camera
        wall / light-stage front. ``right`` and ``down`` are the spacing vectors
        between adjacent columns and rows. Cameras are returned row-major with
        id ``r * cols + c``. Built through :meth:`Camera.look_at`, so the
        RDF / Z-up / ``t = -R @ C`` convention is never re-derived.

        ``overrides`` replaces the computed pose for individual cameras (length
        ``rows * cols``, or a mapping keyed by camera index); see :meth:`ring`
        for the accepted shapes.
        """
        if rows < 1:
            raise ValueError("grid needs rows >= 1")
        if cols < 1:
            raise ValueError("grid needs cols >= 1")
        n = rows * cols
        intrinsics = _intrinsics(width, height_px, focal, fov_deg)
        origin = np.asarray(corner, dtype=np.float64)
        right_vec = np.asarray(right, dtype=np.float64)
        down_vec = np.asarray(down, dtype=np.float64)
        base_target = np.asarray(look_at, dtype=np.float64)
        resolved = _resolve_overrides(n, overrides)
        cams: list[Camera] = []
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                ov = resolved[i]
                if ov is not None:
                    eye = np.asarray(ov.position, dtype=np.float64)
                    target = np.asarray(ov.look_at, dtype=np.float64)
                else:
                    eye = origin + c * right_vec + r * down_vec
                    target = base_target
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
        overrides: PoseOverrides | None = None,
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

        ``overrides`` replaces the computed pose for individual stations while
        keeping the station's own intrinsics; see :meth:`ring` for the accepted
        shapes.

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
        n = len(views)
        resolved = _resolve_overrides(n, overrides)
        cams: list[Camera] = []
        for i, view in enumerate(views):
            has_own = view.focal is not None or view.fov_deg is not None
            eff_focal = view.focal if has_own else focal
            eff_fov = view.fov_deg if has_own else fov_deg
            eff_w = view.width if view.width is not None else width
            eff_h = view.height_px if view.height_px is not None else height_px
            intrinsics = _intrinsics(eff_w, eff_h, eff_focal, eff_fov)
            ov = resolved[i]
            if ov is not None:
                eye = np.asarray(ov.position, dtype=np.float64)
                target = np.asarray(ov.look_at, dtype=np.float64)
            else:
                eye = np.asarray(view.position, dtype=np.float64)
                target = np.asarray(view.look_at, dtype=np.float64)
            cams.append(Camera.look_at(i, intrinsics, eye, target))
        return cams

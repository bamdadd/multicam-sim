"""Build the manifest — the shared contract of record, as a typed pydantic model.

The manifest is the hand-off to a triangulation reader (multicam-occlusion). For
every named point of every entity at every frame it records the ground-truth
world coordinate and, per camera, the pixel observation plus occlusion signals:

* ``in_view`` (bool): projects in front of the camera AND inside the image.
* ``visible`` (bool): the hard DLT mask — ``in_view AND not occluded``.
* ``occ_frac`` (float): a continuous difficulty knob — the fraction of a small,
  configurable jittered sample around the point whose sightline is blocked.

Everything is a pydantic model built typed all the way through; there is no loose
dict assembly, and the model carries the schema validators (so it is the single
source of truth that :mod:`multicam_sim.validation` delegates to).
:meth:`Manifest.to_json` serialises via ``model_dump_json`` and is
**byte-identical** to the historical ``json.dumps(..., indent=2, allow_nan=False)``
output: field order = declaration order, optional fields (``edges``/``topology``/
per-camera ``assumed``) omitted when absent, ``occ_frac`` present when set, full
double float precision, strict/finite JSON (``uv`` is sanitised).

Seeded noise/calibration-drift knobs (:mod:`multicam_sim.noise`) can be threaded
through :func:`observe` / :func:`build_manifest` / :func:`write_manifest` to
perturb the OBSERVED ``uv`` and record a slightly-wrong ASSUMED calibration
(additively, in ``assumed``); ground truth stays exact and the all-off default
reproduces the byte layout above.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import (
    BaseModel,
    StrictBool,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .cameras import Camera
from .dropout import SensorDropout, dropped_frames
from .geometry import FloatArray
from .noise import NoiseModel, PixelNoise
from .occluders import Occluder
from .scene import Scene
from .visibility import silhouette_visible_fraction

CONVENTION = "opencv_rdf"

# Independent ``numpy.random.default_rng`` sub-stream tags derived alongside a
# NoiseModel.seed so pixel noise and per-camera drift never share RNG state (and
# a camera's drift is stable regardless of camera order). The values are
# arbitrary fixed constants ("PXL"/"DRI" as hex) that only need to differ.
_PIXEL_NOISE_STREAM = 0x50584C
_DRIFT_STREAM = 0x445249

# Fixed, deterministic jitter offsets (unit directions) for occ_frac sampling —
# no RNG, so the difficulty knob is reproducible. The first six directions are
# the axis-aligned offsets used by the original sampler; additional directions
# (face and space diagonals of a cube) extend the sample count while preserving
# the default ordering.
_JITTER_DIRS: FloatArray = np.array(
    [
        # 6 axis-aligned directions (default sample set after the centre point).
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        # 12 face diagonals.
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
        [0.0, -1.0, -1.0],
        # 8 space diagonals.
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)
# Normalise so every offset lies on a sphere of radius ``jitter``.
_JITTER_DIRS[6:18, :] /= np.sqrt(2.0)
_JITTER_DIRS[18:, :] /= np.sqrt(3.0)

# Centre point + the 6/12/8 deterministic directions above.
_MAX_SAMPLE_COUNT: int = 1 + _JITTER_DIRS.shape[0]


# --------------------------------------------------------------------------- #
# Typed manifest models. Field declaration order == emitted JSON key order.
# The validators are the schema's single source of truth (validation.py delegates
# here); their messages are consumed by multicam_sim.validation error reporting.
# --------------------------------------------------------------------------- #


class PerCamObs(BaseModel):
    """One camera's observation of a world point.

    ``visible_fraction`` / ``occluded`` are the opt-in image-space occlusion
    labels (see :mod:`multicam_sim.visibility`): the fraction of the object's
    silhouette visible on this camera and whether any nearer occluder eats into it
    (``visible_fraction < 1``). Both are ``None`` unless ``object_radius`` is
    passed to :func:`build_manifest`, so the noiseless manifest stays
    byte-identical. A DROPPED frame leaves them ``None`` too — dropout is not
    occlusion. They are declared AFTER ``occ_frac`` so, when present, they
    serialise in that order.
    """

    cam: int
    uv: list[float]
    in_view: StrictBool
    visible: StrictBool
    occ_frac: float | None = None
    visible_fraction: float | None = None
    occluded: StrictBool | None = None
    # Sensor dropout (multicam_sim.dropout): ``True`` on a blanked/dropped frame,
    # absent otherwise. Declared LAST so it serialises after the occlusion fields
    # and keeps the golden manifest byte-identical when no frame is dropped
    # (exclude_none). A dropped observation is blank: not in view, not visible,
    # no occlusion score / occlusion label — dropout is a sensor gap, never a 0.0
    # occlusion.
    dropped: StrictBool | None = None

    @field_validator("uv")
    @classmethod
    def _check_uv_length(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("uv must be a length-2 vector")
        return value

    @field_validator("occ_frac")
    @classmethod
    def _check_occ_frac_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("occ_frac must be in [0, 1]")
        return value

    @field_validator("visible_fraction")
    @classmethod
    def _check_visible_fraction_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("visible_fraction must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def _visible_implies_in_view(self) -> PerCamObs:
        if self.visible and not self.in_view:
            raise ValueError("visible implies in_view, but in_view is false")
        return self

    @model_validator(mode="after")
    def _dropped_is_blank(self) -> PerCamObs:
        if self.dropped and (
            self.in_view
            or self.visible
            or self.occ_frac is not None
            or self.visible_fraction is not None
            or self.occluded is not None
        ):
            raise ValueError(
                "a dropped frame is a sensor gap: in_view/visible must be false and "
                "occ_frac/visible_fraction/occluded absent (dropout is not a zero occlusion)"
            )
        return self


class PointObs(BaseModel):
    """A named point: ground-truth world coordinate + per-camera observations."""

    xyz_gt: list[float]
    per_cam: list[PerCamObs]

    @field_validator("xyz_gt")
    @classmethod
    def _check_xyz_gt_length(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("xyz_gt must be a length-3 vector")
        return value


class FrameObs(BaseModel):
    """One frame of an entity: frame index + its named points."""

    frame: int
    points: dict[str, PointObs]


class EntityManifest(BaseModel):
    """An entity's manifest: id, per-frame observations, optional skeleton edges.

    ``edges`` is declared AFTER ``frames`` so, when present, it serialises last —
    matching the historical byte layout; when absent it is omitted (exclude_none).
    """

    id: str
    frames: list[FrameObs]
    edges: list[list[str]] | None = None

    @model_validator(mode="after")
    def _check_edges_reference_known_points(self) -> EntityManifest:
        if self.edges is None:
            return self
        known: set[str] = set()
        for frame in self.frames:
            known.update(frame.points.keys())
        for index, edge in enumerate(self.edges):
            if len(edge) != 2:
                raise ValueError(f"edge[{index}] must be a pair of point names")
            first, second = edge
            if first not in known:
                raise ValueError(f"edge[{index}] references unknown point {first!r}")
            if second not in known:
                raise ValueError(f"edge[{index}] references unknown point {second!r}")
        return self


class AssumedCalibration(BaseModel):
    """The drifted calibration a consumer receives under calibration drift.

    Recorded (additively, on :class:`CameraManifest`) ONLY when drift is active.
    The ground-truth ``K, R, t`` live on the parent camera entry and stay exact;
    this holds the slightly-wrong ``K, R, t`` a downstream reader would use.
    """

    K: list[list[float]]
    R: list[list[float]]
    t: list[float]

    @field_validator("K")
    @classmethod
    def _check_assumed_k_shape(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("assumed.K must be a 3x3 matrix")
        return value

    @field_validator("R")
    @classmethod
    def _check_assumed_r_shape(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("assumed.R must be a 3x3 matrix")
        return value

    @field_validator("t")
    @classmethod
    def _check_assumed_t_shape(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("assumed.t must be a length-3 vector")
        return value


class CameraManifest(BaseModel):
    """A camera's serialised intrinsics + world->camera extrinsics + convention.

    ``assumed`` is declared LAST so, when present, it serialises after the
    ground-truth fields; it is omitted entirely (exclude_none) when calibration
    drift is off, keeping the manifest byte-identical to the noiseless output.
    """

    id: int
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]
    width: int
    height: int
    convention: str
    assumed: AssumedCalibration | None = None
    # The camera's seeded sensor-dropout schedule (multicam_sim.dropout): the
    # sorted frame indices this camera blanked. Absent (exclude_none) when dropout
    # is off, keeping the manifest byte-identical to the no-dropout output.
    dropped_frames: list[int] | None = None

    @field_validator("K")
    @classmethod
    def _check_k_shape(cls, value: list[list[float]], info: ValidationInfo) -> list[list[float]]:
        camera_id = info.data.get("id", "?")
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError(f"camera {camera_id}: K must be a 3x3 matrix")
        return value

    @field_validator("R")
    @classmethod
    def _check_r_shape(cls, value: list[list[float]], info: ValidationInfo) -> list[list[float]]:
        camera_id = info.data.get("id", "?")
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError(f"camera {camera_id}: R must be a 3x3 matrix")
        return value

    @field_validator("t")
    @classmethod
    def _check_t_shape(cls, value: list[float], info: ValidationInfo) -> list[float]:
        camera_id = info.data.get("id", "?")
        if len(value) != 3:
            raise ValueError(f"camera {camera_id}: t must be a length-3 vector")
        return value

    @field_validator("convention")
    @classmethod
    def _check_convention(cls, value: str, info: ValidationInfo) -> str:
        camera_id = info.data.get("id", "?")
        if value != CONVENTION:
            raise ValueError(
                f"camera {camera_id}: convention must be {CONVENTION!r}, got {value!r}"
            )
        return value


class Station(BaseModel):
    """A named station and the cameras that share (roughly) its view."""

    id: str
    camera_ids: list[int]

    @field_validator("camera_ids")
    @classmethod
    def _check_non_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("camera_ids must not be empty")
        return value


class TopoEdge(BaseModel):
    """A directed station adjacency with the transit time between them."""

    src: str
    dst: str
    transit_time_s: float

    @field_validator("transit_time_s")
    @classmethod
    def _check_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("transit_time_s must be > 0")
        return value


class Topology(BaseModel):
    """MTMC station adjacency: stations + directed transit edges."""

    stations: list[Station]
    edges: list[TopoEdge]

    @model_validator(mode="after")
    def _check_unique_station_ids(self) -> Topology:
        ids = [station.id for station in self.stations]
        if len(ids) != len(set(ids)):
            raise ValueError("station ids must be unique")
        return self

    @model_validator(mode="after")
    def _check_edges_reference_known_stations(self) -> Topology:
        known = {station.id for station in self.stations}
        for index, edge in enumerate(self.edges):
            if edge.src not in known:
                raise ValueError(
                    f"topology.edges[{index}] src {edge.src!r} references an unknown station"
                )
            if edge.dst not in known:
                raise ValueError(
                    f"topology.edges[{index}] dst {edge.dst!r} references an unknown station"
                )
        return self


class Manifest(BaseModel):
    """The full typed manifest. ``topology`` is omitted when absent."""

    cameras: list[CameraManifest]
    fps: float
    num_frames: int
    entities: list[EntityManifest]
    topology: Topology | None = None

    def to_json(self) -> str:
        """Serialise byte-identically to the historical dumps: 2-space indent,
        optional fields omitted, full float precision, strict (finite) JSON."""
        return self.model_dump_json(indent=2, exclude_none=True)


# --------------------------------------------------------------------------- #
# Compute (pure projection + boolean visibility; no renderer).
# --------------------------------------------------------------------------- #


def _any_blocks(occluders: list[Occluder], a: FloatArray, b: FloatArray) -> bool:
    return any(occ.blocks_segment(a, b) for occ in occluders)


def occlusion_fraction(
    camera: Camera,
    point3d: FloatArray,
    occluders: list[Occluder],
    sample_count: int = 7,
    jitter: float = 0.05,
) -> float:
    """Continuous occlusion: fraction of jittered sightlines that are blocked.

    ``occ_frac`` is a difficulty knob that grades how marginal an occlusion is.
    It samples the point plus a small deterministic jittered neighbourhood and
    returns the fraction of those samples whose sightline to the camera centre
    is blocked. It is distinct from ``visible`` and never feeds the
    triangulation mask.

    Args:
        sample_count: Total number of samples (centre point + jitter offsets).
            Must be positive and no greater than ``_MAX_SAMPLE_COUNT`` (27).
            The default of 7 reproduces the original behaviour: the centre
            point plus six axis-aligned offsets.
        jitter: Radius of the jitter neighbourhood (same units as the scene).
            Must be non-negative. The default of ``0.05`` reproduces the
            original behaviour.

    Returns:
        Blocked fraction in ``[0, 1]``. ``0.0`` when there are no occluders.
    """
    if not occluders:
        return 0.0
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if sample_count > _MAX_SAMPLE_COUNT:
        raise ValueError(
            f"sample_count {sample_count} exceeds the deterministic sample pool "
            f"size {_MAX_SAMPLE_COUNT}"
        )
    if jitter < 0.0:
        raise ValueError(f"jitter radius must be non-negative, got {jitter}")
    centre = camera.centre()
    num_offsets = sample_count - 1
    samples: FloatArray = point3d[None, :]
    if num_offsets > 0 and jitter > 0.0:
        offsets = jitter * _JITTER_DIRS[:num_offsets]
        samples = np.vstack([samples, point3d[None, :] + offsets])
    blocked = sum(1 for s in samples if _any_blocks(occluders, s, centre))
    return float(blocked / samples.shape[0])


def observe(
    camera: Camera,
    point3d: FloatArray,
    occluders: list[Occluder],
    occ_frac_sample_count: int = 7,
    occ_frac_jitter: float = 0.05,
    *,
    object_radius: float | None = None,
    pixel_noise: PixelNoise | None = None,
    rng: np.random.Generator | None = None,
    dropped: bool = False,
) -> PerCamObs:
    """One camera's observation of a world point.

    * ``in_view``  — projects IN FRONT (w > 0) AND inside the image bounds.
    * ``visible``  — the DLT mask: ``in_view AND not occluded`` (so ``visible``
      implies ``in_view``).
    * ``occ_frac`` — continuous occlusion difficulty, configured by
      ``occ_frac_sample_count`` and ``occ_frac_jitter`` (see
      :func:`occlusion_fraction`).
    * ``visible_fraction`` / ``occluded`` — the image-space silhouette labels
      (see :func:`multicam_sim.visibility.silhouette_visible_fraction`), emitted
      only when ``object_radius`` is given; otherwise both are ``None`` and the
      manifest stays byte-identical.

    ``occluders`` must be the per-frame static solids (already resolved via
    :meth:`Occluder.at_frame`), so the sightline / silhouette maths only ever
    sees a Box/Sphere.

    When ``pixel_noise`` is active and ``rng`` is supplied, a Gaussian offset
    (sigma ``pixel_noise.sigma_px``) is added to the OBSERVED ``uv`` only — the
    geometric flags (``in_view``/``visible``) are computed from the TRUE
    projection, so ground truth stays exact. The two draws are taken whenever
    noise is active and the projection is valid, so the RNG advances one pair per
    observed point.

    When ``dropped`` is set (a seeded sensor-dropout frame; see
    :mod:`multicam_sim.dropout`), the observation is blanked — ``uv=[0,0]``,
    ``in_view``/``visible`` false, ``occ_frac`` absent, ``dropped=True`` — so a
    downstream reader gets a coverage gap rather than a (mis)usable pixel. The
    pixel-noise draw still runs first, so dropout never perturbs the noise stream.

    Non-raising: an out-of-frame or behind-camera point is labelled, not an error.
    ``uv`` is sanitised to finite values so the manifest is always strict JSON.
    """
    uv, w = camera.project(point3d)
    in_front = w > 0.0
    in_image = bool(in_front and camera.in_image(uv))
    in_view = bool(in_front and in_image)
    unoccluded = not _any_blocks(occluders, point3d, camera.centre())
    visible = bool(in_view and unoccluded)
    u, v = float(uv[0]), float(uv[1])
    if not (in_front and np.isfinite(u) and np.isfinite(v)):
        # behind / at the image plane: pixel is meaningless, keep JSON finite.
        u, v = 0.0, 0.0
    elif pixel_noise is not None and pixel_noise.is_active and rng is not None:
        offset = rng.normal(0.0, pixel_noise.sigma_px, size=2)
        u, v = u + float(offset[0]), v + float(offset[1])
    if dropped:
        # Sensor gap: the camera delivered nothing this frame. Blank the whole
        # observation (no occ_frac / visible_fraction / occluded — dropout is not
        # a zero occlusion). The pixel-noise draw above still ran, so the noise
        # stream advances identically with or without dropout (the two are
        # independent).
        return PerCamObs(
            cam=camera.id,
            uv=[0.0, 0.0],
            in_view=False,
            visible=False,
            occ_frac=None,
            dropped=True,
        )
    visible_fraction: float | None = None
    occluded: bool | None = None
    if object_radius is not None:
        visible_fraction = silhouette_visible_fraction(camera, point3d, object_radius, occluders)
        occluded = visible_fraction < 1.0
    return PerCamObs(
        cam=camera.id,
        uv=[u, v],
        in_view=in_view,
        visible=visible,
        occ_frac=occlusion_fraction(
            camera,
            point3d,
            occluders,
            sample_count=occ_frac_sample_count,
            jitter=occ_frac_jitter,
        ),
        visible_fraction=visible_fraction,
        occluded=occluded,
    )


def _assumed_calibration(camera: Camera, noise: NoiseModel | None) -> AssumedCalibration | None:
    """The drifted (assumed) calibration for ``camera``, or ``None`` when drift
    is off. Seeded from ``noise.seed`` with a per-camera sub-stream so a camera's
    drift is reproducible and independent of camera order and the pixel-noise pass.
    """
    if noise is None or not noise.drift.is_active:
        return None
    rng = np.random.default_rng([int(noise.seed), _DRIFT_STREAM, int(camera.id)])
    drifted = camera.drifted(rng, noise.drift)
    return AssumedCalibration(
        K=drifted.intrinsics.matrix().tolist(),
        R=[list(row) for row in drifted.R],
        t=list(drifted.t),
    )


def camera_entry(
    camera: Camera,
    noise: NoiseModel | None = None,
    *,
    dropped: tuple[int, ...] = (),
) -> CameraManifest:
    """Serialise a camera: full-precision (ground-truth) K, R, t + convention.

    When ``noise`` carries an active calibration drift, the slightly-wrong
    ASSUMED calibration is recorded additively in ``assumed``; otherwise
    ``assumed`` is absent and the entry is byte-identical to the noiseless output.
    ``dropped`` is the camera's seeded dropout schedule (the frames it blanked),
    recorded in ``dropped_frames``; an empty schedule leaves the field absent.
    """
    return CameraManifest(
        id=camera.id,
        K=camera.intrinsics.matrix().tolist(),
        R=[list(row) for row in camera.R],
        t=list(camera.t),
        width=camera.intrinsics.width,
        height=camera.intrinsics.height,
        convention=CONVENTION,
        assumed=_assumed_calibration(camera, noise),
        dropped_frames=list(dropped) if dropped else None,
    )


def _topology(scene: Scene) -> Topology | None:
    if scene.topology is None:
        return None
    raw = scene.topology.to_manifest()
    return Topology(
        stations=[Station(id=s["id"], camera_ids=list(s["camera_ids"])) for s in raw["stations"]],
        edges=[
            TopoEdge(src=e["src"], dst=e["dst"], transit_time_s=e["transit_time_s"])
            for e in raw["edges"]
        ],
    )


def build_manifest(
    scene: Scene,
    occ_frac_sample_count: int = 7,
    occ_frac_jitter: float = 0.05,
    *,
    object_radius: float | None = None,
    noise: NoiseModel | None = None,
    dropout: SensorDropout | None = None,
) -> Manifest:
    """Compute the full typed manifest for ``scene`` (pure projection + boolean
    visibility; no renderer). Built typed all the way — no dict assembly.

    ``occ_frac_sample_count`` / ``occ_frac_jitter`` configure the ``occ_frac``
    sampler (see :func:`occlusion_fraction`); the defaults reproduce the original
    byte-for-byte output.

    ``object_radius`` opts into the image-space occlusion labels: when given, each
    per-camera observation carries ``visible_fraction`` (silhouette-area fraction
    visible, modelling the object as a sphere of that radius) and ``occluded``.
    ``None`` (the default) omits both fields, keeping the manifest byte-identical.
    Time-varying occluders (e.g. a :class:`~multicam_sim.occluders.HandOccluder`)
    are resolved to their per-frame static solid before any sightline/silhouette
    test, so ``visible``/``occ_frac`` are computed against the hand's true pose at
    each frame.

    ``noise`` (a :class:`multicam_sim.noise.NoiseModel`) adds seeded, reproducible
    error: Gaussian pixel noise on the observed ``uv`` and a separately-recorded
    ASSUMED calibration under drift. Ground truth (``xyz_gt`` and the true
    ``K, R, t``) stays exact. ``noise=None`` (the default) or an all-zero
    :class:`NoiseModel` reproduces the noiseless output byte-for-byte.
    """
    pixel_noise = noise.pixel if noise is not None else None
    pixel_rng: np.random.Generator | None = None
    if noise is not None and noise.pixel.is_active:
        pixel_rng = np.random.default_rng([int(noise.seed), _PIXEL_NOISE_STREAM])

    # Per-camera dropout schedule, computed once (independent seeded sub-streams).
    drop_by_cam: dict[int, frozenset[int]] = {}
    if dropout is not None and dropout.is_active:
        drop_by_cam = {
            cam.id: frozenset(dropped_frames(dropout, cam.id, scene.num_frames))
            for cam in scene.cameras
        }

    entities_out: list[EntityManifest] = []
    for entity in scene.entities:
        frames_out: list[FrameObs] = []
        for frame in entity.frames:
            # Resolve every occluder to the static solid it presents at this
            # frame (statics return themselves), so the sightline/silhouette
            # tests only ever see a Box/Sphere.
            occluders: list[Occluder] = [occ.at_frame(frame.frame) for occ in scene.occluders]
            points_out: dict[str, PointObs] = {}
            for name, xyz in frame.points.items():
                point3d = np.asarray(xyz, dtype=np.float64)
                points_out[name] = PointObs(
                    xyz_gt=[float(c) for c in xyz],
                    per_cam=[
                        observe(
                            cam,
                            point3d,
                            occluders,
                            occ_frac_sample_count=occ_frac_sample_count,
                            occ_frac_jitter=occ_frac_jitter,
                            object_radius=object_radius,
                            pixel_noise=pixel_noise,
                            rng=pixel_rng,
                            dropped=frame.frame in drop_by_cam.get(cam.id, frozenset()),
                        )
                        for cam in scene.cameras
                    ],
                )
            frames_out.append(FrameObs(frame=frame.frame, points=points_out))
        edges = None if entity.edges is None else [list(e) for e in entity.edges]
        entities_out.append(EntityManifest(id=entity.id, frames=frames_out, edges=edges))

    return Manifest(
        cameras=[
            camera_entry(cam, noise, dropped=tuple(sorted(drop_by_cam.get(cam.id, ()))))
            for cam in scene.cameras
        ],
        fps=scene.fps,
        num_frames=scene.num_frames,
        entities=entities_out,
        topology=_topology(scene),
    )


def write_manifest(
    scene: Scene,
    path: str | Path,
    *,
    noise: NoiseModel | None = None,
    dropout: SensorDropout | None = None,
) -> Manifest:
    """Build the typed manifest and write it to ``path`` as JSON.

    Serialised via :meth:`Manifest.to_json` — byte-identical to the historical
    ``json.dumps(..., indent=2, allow_nan=False)`` output when ``noise`` and
    ``dropout`` are off. ``noise`` threads the seeded noise/drift knobs and
    ``dropout`` the seeded per-camera sensor-dropout schedule into
    :func:`build_manifest`.
    """
    manifest = build_manifest(scene, noise=noise, dropout=dropout)
    Path(path).write_text(manifest.to_json())
    return manifest

"""Manifest schema validation — the consumer-side contract check.

Validates that a loaded JSON manifest conforms to the schema declared in
``DESIGN.md`` ("Manifest schema — the JSON contract"). This is the inverse of
:func:`multicam_sim.manifest.build_manifest`: every manifest produced by the
builder must pass, and every hand-written corruption the schema forbids must
fail with a message naming the offending path.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    StrictBool,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)


class _Observation(BaseModel):
    """One camera's observation of a world point."""

    cam: int
    uv: list[float]
    in_view: StrictBool
    visible: StrictBool
    occ_frac: float | None = None

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


class _Point(BaseModel):
    """Ground-truth 3D point plus per-camera observations."""

    xyz_gt: list[float]
    per_cam: list[_Observation]

    @field_validator("xyz_gt")
    @classmethod
    def _check_xyz_gt_length(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("xyz_gt must be a length-3 vector")
        return value


class _EntityFrame(BaseModel):
    """One frame of one entity: named 3D points."""

    frame: int
    points: dict[str, _Point]


class _Entity(BaseModel):
    """One entity: stable id, optional skeleton edges, and frames."""

    id: str
    edges: list[list[str]] | None = None
    frames: list[_EntityFrame]

    @model_validator(mode="after")
    def _check_edges_reference_known_points(self) -> _Entity:
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


class _Camera(BaseModel):
    """One camera entry in the manifest."""

    id: int
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]
    width: int
    height: int
    convention: str

    @field_validator("K")
    @classmethod
    def _check_K_shape(cls, value: list[list[float]], info: ValidationInfo) -> list[list[float]]:
        camera_id = info.data.get("id", "?")
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError(f"camera {camera_id}: K must be a 3x3 matrix")
        return value

    @field_validator("R")
    @classmethod
    def _check_R_shape(cls, value: list[list[float]], info: ValidationInfo) -> list[list[float]]:
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
        if value != "opencv_rdf":
            raise ValueError(f"camera {camera_id}: convention must be 'opencv_rdf', got {value!r}")
        return value


class _Station(BaseModel):
    """One station in a topology."""

    id: str
    camera_ids: list[int]

    @field_validator("camera_ids")
    @classmethod
    def _check_non_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("camera_ids must not be empty")
        return value


class _TransitEdge(BaseModel):
    """One directed transit edge between stations."""

    src: str
    dst: str
    transit_time_s: float

    @field_validator("transit_time_s")
    @classmethod
    def _check_positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("transit_time_s must be > 0")
        return value


class _Topology(BaseModel):
    """Optional MTMC station adjacency metadata."""

    stations: list[_Station]
    edges: list[_TransitEdge] = []

    @model_validator(mode="after")
    def _check_unique_station_ids(self) -> _Topology:
        ids = [station.id for station in self.stations]
        if len(ids) != len(set(ids)):
            raise ValueError("station ids must be unique")
        return self

    @model_validator(mode="after")
    def _check_edges_reference_known_stations(self) -> _Topology:
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


class _Manifest(BaseModel):
    """Top-level manifest schema."""

    cameras: list[_Camera]
    fps: float
    num_frames: int
    entities: list[_Entity]
    topology: _Topology | None = None


def _format_error(error: Any) -> str:
    """Format a pydantic ValidationError entry as a dotted path with a message."""
    location = ".".join(str(part) for part in error["loc"])
    return f"manifest validation failed at {location}: {error['msg']}"


def validate_manifest(data: dict[str, Any]) -> None:
    """Validate that ``data`` conforms to the manifest schema in DESIGN.md.

    Raises:
        ValueError: with the offending path and problem if validation fails.
    """
    try:
        manifest = _Manifest(**data)
    except ValidationError as exc:
        raise ValueError(_format_error(exc.errors()[0])) from exc

    for entity in manifest.entities:
        for frame_index, frame in enumerate(entity.frames):
            for point_name, point in frame.points.items():
                for obs_index, observation in enumerate(point.per_cam):
                    if observation.visible and not observation.in_view:
                        raise ValueError(
                            f"manifest validation failed: entities.{entity.id}.frames"
                            f"[{frame_index}].points[{point_name!r}].per_cam[{obs_index}] "
                            "visible implies in_view, but in_view is false"
                        )

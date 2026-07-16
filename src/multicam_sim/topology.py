"""Camera topology — the MTMC adjacency contract.

A :class:`CameraTopology` describes which camera *stations* are adjacent and how
long an object takes to transit between them. multicam-occlusion's multi-target
multi-camera (MTMC) path consumes this to reason about a target that leaves one
station's field of view, crosses a blind gap, and re-appears at an adjacent
station — the transit time bounds how long the re-identification gap may last.

This is a top-level, OPTIONAL manifest field: single-station scenes omit it and
their manifest is unchanged.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class Station(BaseModel):
    """A named place holding one or more cameras with a (roughly) shared view."""

    id: str
    camera_ids: list[int]

    @field_validator("camera_ids")
    @classmethod
    def _non_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("a station must hold at least one camera")
        return value


class TransitEdge(BaseModel):
    """A DIRECTED adjacency ``src -> dst`` with the transit time between stations."""

    src: str
    dst: str
    transit_time_s: float

    @field_validator("transit_time_s")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("transit_time_s must be > 0")
        return value


class CameraTopology(BaseModel):
    """Stations plus directed transit edges. Edge endpoints must be station ids."""

    stations: list[Station]
    edges: list[TransitEdge] = []

    @field_validator("stations")
    @classmethod
    def _unique_station_ids(cls, value: list[Station]) -> list[Station]:
        ids = [s.id for s in value]
        if len(ids) != len(set(ids)):
            raise ValueError("station ids must be unique")
        return value

    def model_post_init(self, __context: Any) -> None:
        known = {s.id for s in self.stations}
        for edge in self.edges:
            if edge.src not in known or edge.dst not in known:
                raise ValueError(
                    f"transit edge {edge.src!r}->{edge.dst!r} references an unknown station"
                )

    def to_manifest(self) -> dict[str, Any]:
        """Serialise the topology for the manifest."""
        return {
            "stations": [{"id": s.id, "camera_ids": list(s.camera_ids)} for s in self.stations],
            "edges": [
                {"src": e.src, "dst": e.dst, "transit_time_s": e.transit_time_s} for e in self.edges
            ],
        }

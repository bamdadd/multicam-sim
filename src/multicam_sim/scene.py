"""Scene: the top-level typed container — cameras, entities, occluders, timing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .cameras import Camera
from .entities import Entity
from .occluders import OccluderUnion
from .possession import PossessionTimeline
from .topology import CameraTopology


class Scene(BaseModel):
    """A full multi-camera scene over ``num_frames`` frames at ``fps``.

    Occluders serialise as a discriminated union on their ``kind`` tag, so a
    scene round-trips through JSON without losing the Box/Sphere distinction.
    ``topology`` is optional MTMC metadata (station adjacency + transit times),
    emitted in the manifest only when present.

    ``possession`` is an optional possession-GT sidecar (see
    :mod:`multicam_sim.possession`). It is additive: absent by default and never
    read by the manifest builder, so scenes that do not use it keep the
    byte-golden analytic manifest.
    """

    fps: float
    num_frames: int
    cameras: list[Camera]
    entities: list[Entity]
    occluders: list[OccluderUnion] = []
    topology: CameraTopology | None = None
    possession: PossessionTimeline | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.topology is None:
            return
        known = {cam.id for cam in self.cameras}
        for station in self.topology.stations:
            for camera_id in station.camera_ids:
                if camera_id not in known:
                    raise ValueError(
                        f"station {station.id!r} references unknown camera {camera_id!r}"
                    )

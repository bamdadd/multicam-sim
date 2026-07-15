"""Scene: the top-level typed container — cameras, entities, occluders, timing."""

from __future__ import annotations

from pydantic import BaseModel

from .cameras import Camera
from .entities import Entity
from .occluders import OccluderUnion


class Scene(BaseModel):
    """A full multi-camera scene over ``num_frames`` frames at ``fps``.

    Occluders serialise as a discriminated union on their ``kind`` tag, so a
    scene round-trips through JSON without losing the Box/Sphere distinction.
    """

    fps: float
    num_frames: int
    cameras: list[Camera]
    entities: list[Entity]
    occluders: list[OccluderUnion] = []

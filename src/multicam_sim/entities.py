"""Entities: a moving thing carrying NAMED 3D points per frame.

The named-points design is the forward-compatibility hinge of the contract. An
object is one entity with a single named point ``"center"``; a COCO-17 human
(a LATER layer, not built here) is one entity with 17 named points plus an
``edges`` skeleton — same schema, no fork. ``edges`` reference point names.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class EntityFrame(BaseModel):
    """One frame of an entity: a mapping of point name -> ``[x, y, z]`` (ground
    truth world coordinates)."""

    frame: int
    points: dict[str, list[float]]

    @field_validator("points")
    @classmethod
    def _check_xyz(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        for name, xyz in value.items():
            if len(xyz) != 3:
                raise ValueError(f"point {name!r} must be [x, y, z]; got {len(xyz)} coords")
        return value


class Entity(BaseModel):
    """A named collection of 3D points tracked across frames.

    ``edges`` is an optional skeleton (pairs of point names) so a later pose
    layer renders limbs; for a plain object it stays ``None``.
    """

    id: str
    edges: list[tuple[str, str]] | None = None
    frames: list[EntityFrame]

    def point_names(self) -> set[str]:
        """Every point name that appears in any frame."""
        names: set[str] = set()
        for f in self.frames:
            names.update(f.points)
        return names

"""Group formation & dispersal: a scene builder + per-frame membership GT.

Occupancy / group-behaviour analysis needs a scene where N agents converge
into a spatial cluster, dwell as a group, then disperse — plus ground truth
recording *which entities belong to the group at every frame*.

Two pieces, both self-contained:

* :func:`build_group_formation_scene` — a :class:`~multicam_sim.scene.Scene`
  whose agents lerp from a wide ring onto a tight cluster ring (both centred
  on ``group_center``), hold, then lerp radially back out along directions
  rotated half a sector from the approach. The ring
  geometry is symmetric, so the per-frame centroid of all agents stays at
  ``group_center`` throughout — the membership rule below is then exact.
* :func:`compute_group_membership` — the deterministic membership rule, pure
  on the scene alone: an entity is a member at frame ``f`` iff its tracked
  point has been within ``radius`` of the per-frame group centroid for at
  least ``min_dwell_frames`` consecutive frames ending at ``f``. Formation
  and dispersal frames fall out of that rule; nothing is hand-annotated.

The ground truth is a SINGLE group (``group_id`` defaults to ``"group-0"``).
The id field is carried so multi-group output later is an additive change —
a list of these records — not a schema fork.

**Sidecar, opt-in.** Like the order GT (:mod:`multicam_sim.order`), the
membership record rides in its own JSON sidecar via :func:`write_group_json`
and never touches the analytic manifest: it is computed FROM the scene, not
stored on it, so ``build_manifest`` output is byte-identical whether or not
the GT is requested. Everything round-trips via pydantic ``model_dump``.

Scale follows the smoke scenes (:mod:`multicam_sim.smoke`): cameras on a
radius-4 ring looking at ``[0, 0, 0.5]`` with 640x480 @ f=800, agent paths
within roughly +-1 of the origin — so ``radius=0.5`` and ``cluster_spread=0.2``
are scene-scale distances, not arbitrary constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from .cameras import Camera, Intrinsics
from .entities import Entity, EntityFrame
from .geometry import FloatArray
from .scene import Scene

# Camera rig — identical geometry to the smoke scenes so the group stays in
# frame of every camera for the whole run.
_CAM_RING_RADIUS = 4.0
_CAM_HEIGHT = 1.5
_FOCAL = 800.0
_WIDTH = 640
_HEIGHT_PX = 480

_DEFAULT_RADIUS = 0.5
_DEFAULT_MIN_DWELL_FRAMES = 3


class GroupFrameMembership(BaseModel):
    """One frame of group ground truth: the group centroid and its member ids."""

    model_config = ConfigDict(frozen=True)

    frame: int
    centroid: tuple[float, float, float]
    members: list[str]


class GroupMembership(BaseModel):
    """Per-frame membership ground truth for one group, derived from a scene.

    ``formation_frame`` is the first frame with at least one member;
    ``dispersal_frame`` is the first frame after formation with no members
    (``None`` when the group never forms or never disperses). Both fall out
    of the deterministic rule in :func:`compute_group_membership` — they are
    recorded here for convenience, not annotated by hand.
    """

    model_config = ConfigDict(frozen=True)

    group_id: str = "group-0"
    radius: float
    min_dwell_frames: int
    point: str = "center"
    formation_frame: int | None
    dispersal_frame: int | None
    frames: list[GroupFrameMembership]

    @field_validator("radius")
    @classmethod
    def _positive_radius(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("group radius must be > 0")
        return value

    @field_validator("min_dwell_frames")
    @classmethod
    def _positive_dwell(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_dwell_frames must be >= 1")
        return value

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string (the ``groups.json`` sidecar payload)."""
        return self.model_dump_json(indent=indent)


def build_group_formation_scene(
    *,
    num_agents: int = 4,
    num_frames: int = 60,
    fps: float = 30.0,
    group_center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    radius: float = _DEFAULT_RADIUS,
    cluster_spread: float = 0.2,
    start_radius: float = 1.0,
    arrival_frame: int = 20,
    departure_frame: int = 40,
) -> Scene:
    """Build a scene where ``num_agents`` agents cluster, dwell, then disperse.

    Agent ``i`` starts on a ring of ``start_radius`` around ``group_center``
    at angle ``2*pi*i/num_agents``, lerps to the same angle on a ring of
    ``cluster_spread`` by ``arrival_frame``, holds until ``departure_frame``,
    then lerps radially back out to the wide ring along a direction rotated
    half a sector (so agents leave instead of retracing their approach). All
    agents share the timing, so their per-frame centroid stays at
    ``group_center`` and every agent is inside ``radius`` of it for the whole
    dwell window
    (``cluster_spread`` must be < ``radius``). The cameras mirror the smoke
    rig (radius-4 ring aimed at ``group_center``), so all agents stay in view.

    ``radius`` is the same value the membership rule expects; pass it to
    :func:`compute_group_membership` to get matching ground truth.
    """
    if num_agents < 2:
        raise ValueError("a group needs at least 2 agents")
    if not 0 < arrival_frame < departure_frame < num_frames:
        raise ValueError("need 0 < arrival_frame < departure_frame < num_frames")
    if not 0 < cluster_spread < radius <= start_radius:
        raise ValueError("need 0 < cluster_spread < radius <= start_radius")

    center = np.asarray(group_center, dtype=np.float64)
    intrinsics = Intrinsics.from_focal(_FOCAL, _WIDTH, _HEIGHT_PX)
    cameras = []
    for i in range(3):
        angle = 2.0 * math.pi * i / 3.0
        eye = np.array(
            [_CAM_RING_RADIUS * math.cos(angle), _CAM_RING_RADIUS * math.sin(angle), _CAM_HEIGHT],
            dtype=np.float64,
        )
        cameras.append(Camera.look_at(i, intrinsics, eye, center))

    entities = []
    for i in range(num_agents):
        theta = 2.0 * math.pi * i / num_agents
        direction = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
        # Disperse radially outward along a direction rotated half a sector from
        # the approach, so agents leave the cluster instead of retracing it.
        exit_theta = theta + math.pi / num_agents
        exit_direction = np.array(
            [math.cos(exit_theta), math.sin(exit_theta), 0.0], dtype=np.float64
        )
        start = center + start_radius * direction
        cluster = center + cluster_spread * direction
        end = center + start_radius * exit_direction
        frames = [
            EntityFrame(
                frame=f,
                points={
                    "center": _agent_at(
                        f, start, cluster, end, arrival_frame, departure_frame, num_frames
                    ).tolist()
                },
            )
            for f in range(num_frames)
        ]
        entities.append(Entity(id=f"agent-{i}", frames=frames))

    return Scene(fps=fps, num_frames=num_frames, cameras=cameras, entities=entities)


def _agent_at(
    frame: int,
    start: FloatArray,
    cluster: FloatArray,
    end: FloatArray,
    arrival_frame: int,
    departure_frame: int,
    num_frames: int,
) -> FloatArray:
    """Piecewise-linear path: approach, dwell at the cluster, disperse."""
    if frame <= arrival_frame:
        return start + (cluster - start) * (frame / arrival_frame)
    if frame <= departure_frame:
        return cluster
    frac = (frame - departure_frame) / (num_frames - 1 - departure_frame)
    return cluster + (end - cluster) * frac


def compute_group_membership(
    scene: Scene,
    *,
    radius: float = _DEFAULT_RADIUS,
    min_dwell_frames: int = _DEFAULT_MIN_DWELL_FRAMES,
    point: str = "center",
    group_id: str = "group-0",
) -> GroupMembership:
    """Derive per-frame group membership from ``scene`` alone (deterministic).

    The per-frame group centroid is the mean of every entity's ``point``
    position at that frame. An entity is a member at frame ``f`` iff it was
    within ``radius`` of the centroid at each of the ``min_dwell_frames``
    consecutive frames ending at ``f`` (a causal dwell rule). An entity
    missing the point or the frame counts as not-within. Re-running on the
    same scene always yields the same record.
    """
    positions: dict[int, dict[str, tuple[float, float, float]]] = {}
    for entity in scene.entities:
        for entity_frame in entity.frames:
            xyz = entity_frame.points.get(point)
            if xyz is not None:
                positions.setdefault(entity_frame.frame, {})[entity.id] = (
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                )

    entity_ids = sorted(entity.id for entity in scene.entities)
    within: dict[str, list[bool]] = {entity_id: [] for entity_id in entity_ids}
    frames_out: list[GroupFrameMembership] = []
    for frame in range(scene.num_frames):
        here = positions.get(frame, {})
        if here:
            cx = sum(p[0] for p in here.values()) / len(here)
            cy = sum(p[1] for p in here.values()) / len(here)
            cz = sum(p[2] for p in here.values()) / len(here)
        else:
            cx = cy = cz = 0.0
        for entity_id in entity_ids:
            pos = here.get(entity_id)
            is_within = pos is not None and math.dist(pos, (cx, cy, cz)) <= radius
            within[entity_id].append(is_within)
        dwell = min_dwell_frames
        members = [
            entity_id
            for entity_id in entity_ids
            if frame + 1 >= dwell and all(within[entity_id][frame + 1 - dwell : frame + 1])
        ]
        frames_out.append(GroupFrameMembership(frame=frame, centroid=(cx, cy, cz), members=members))

    member_frames = [f.frame for f in frames_out if f.members]
    formation_frame = member_frames[0] if member_frames else None
    dispersal_frame = None
    if formation_frame is not None:
        empty_after = [f.frame for f in frames_out if f.frame > formation_frame and not f.members]
        dispersal_frame = empty_after[0] if empty_after else None

    return GroupMembership(
        group_id=group_id,
        radius=radius,
        min_dwell_frames=min_dwell_frames,
        point=point,
        formation_frame=formation_frame,
        dispersal_frame=dispersal_frame,
        frames=frames_out,
    )


def write_group_json(payload: BaseModel, path: str | Path) -> dict[str, Any]:
    """Write a group model (GroupMembership) to ``path`` as JSON.

    Mirrors :func:`multicam_sim.order.write_order_json`: returns the dumped
    dict so a caller can assert on it without re-reading.
    """
    data: dict[str, Any] = payload.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))
    return data

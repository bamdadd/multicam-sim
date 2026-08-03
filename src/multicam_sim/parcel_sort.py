"""Parcel-sort scene: a handler walks a line of sorting stations under overlapping cameras.

A domain-neutral counterpart to :mod:`multicam_sim.mtmc`. Where MTMC places
cameras at SEPARATED stations with *disjoint* fields of view (an object is in at
most one view at a time, with a genuine blind gap between them), this scene
deliberately *overlaps* adjacent cameras' coverage along a straight walkway:

  * five fixed **sorting stations** (``station_1`` .. ``station_5``) sit evenly
    spaced along the line;
  * one **handler** — a single moving point named ``center`` — walks the line
    ``station_1 -> station_5``, pausing (a **dwell**, the sort operation) at each
    station before moving on;
  * five **cameras** stand back off the line, one framing each station, aimed so
    that adjacent cameras' fields of view overlap over the walkway *between*
    stations.

The overlap is graduated, not universal: while the handler dwells at a station
it is in exactly one camera's view, and while it *transits* between two adjacent
stations it is in view on BOTH covering cameras at once. That two-camera transit
interval is the overlap this scene exists to demonstrate (issue #84) — the
manifest's per-camera ``in_view`` flags record it with no schema change, and a
consumer can triangulate the handler wherever two cameras cover it.

The handler keeps ONE stable ``entity.id`` throughout, the cross-camera
ground-truth identity a tracker must preserve down the line. The scene carries a
:class:`CameraTopology` whose stations name each sorting station's covering
camera and whose directed :class:`TransitEdge` s chain the walk ``station_i ->
station_{i+1}`` (and back) with the transit time between them.

v1 models the handler as a single point target ("center"), which keeps the
overlap proof unambiguous (one point in view on two cameras) and the manifest
compact; a skeletoned handler carrying a parcel is a deliberate future
extension, not a gap.
"""

from __future__ import annotations

from .cameras import Camera
from .dsl.motion import Path, PathUnion
from .dsl.rig import CameraRig, StationView
from .entities import Entity
from .scene import Scene
from .topology import CameraTopology, Station, TransitEdge

_WIDTH = 1280
_HEIGHT_PX = 720
_FPS = 10.0

_NUM_STATIONS = 5
#: Spacing between adjacent sorting stations along the walkway (world units, +x).
_STATION_STEP = 3.0
#: Height of the tracked handler point (torso, world +z).
_HANDLER_Z = 1.0
#: Cameras stand back off the line along -y, looking toward the walkway.
_CAM_Y = -4.0
_CAM_Z = 2.2
#: Wide-ish FOV so adjacent cameras' cones overlap over the walkway.
_CAM_FOV_DEG = 70.0

#: Seconds the handler walks between two adjacent stations.
_WALK_S = 0.5
#: Seconds the handler dwells (pauses, the sort operation) at each station.
_DWELL_S = 0.4

HANDLER_ID = "handler-1"
#: Domain-neutral sorting-station ids down the line.
STATION_IDS = [f"station_{i + 1}" for i in range(_NUM_STATIONS)]


def _station_x(index: int) -> float:
    """World ``x`` of sorting station ``index`` (0-based) along the walkway."""
    return index * _STATION_STEP


def _station_views() -> list[StationView]:
    """One camera per sorting station, standing back off the line and framing that
    station. The wide FOV makes adjacent cones overlap over the walkway between
    stations, so a transiting handler is seen by two cameras at once."""
    return [
        StationView(
            position=(_station_x(i), _CAM_Y, _CAM_Z),
            look_at=(_station_x(i), 0.0, _HANDLER_Z),
            fov_deg=_CAM_FOV_DEG,
        )
        for i in range(_NUM_STATIONS)
    ]


def _cameras() -> list[Camera]:
    return CameraRig.stations(_station_views(), width=_WIDTH, height_px=_HEIGHT_PX)


def _handler_path() -> PathUnion:
    """A timed walk ``station_1 -> station_5`` with a dwell at each station.

    Built from the motion DSL: a zero-length ``linear(p, p).over(dwell)`` segment
    holds the handler in place for the dwell, chained by ``.then`` with a
    ``linear(p_i, p_{i+1}).over(walk)`` transit segment for each leg. The dwell
    segments make the handler's position constant across consecutive frames at
    each station; the transit segments carry it (and its overlapping two-camera
    coverage) between them.
    """
    points = [(_station_x(i), 0.0, _HANDLER_Z) for i in range(_NUM_STATIONS)]
    path: PathUnion = Path.linear(points[0], points[0]).over(_DWELL_S)
    for i in range(_NUM_STATIONS - 1):
        walk = Path.linear(points[i], points[i + 1]).over(_WALK_S)
        dwell = Path.linear(points[i + 1], points[i + 1]).over(_DWELL_S)
        path = path.then(walk).then(dwell)
    return path


def _num_frames(path: PathUnion) -> int:
    """Frame count covering the whole timed walk at :data:`_FPS`, inclusive."""
    return int(round(path.total_duration() * _FPS)) + 1


def build_parcel_sort_scene() -> Scene:
    """Construct the deterministic overlapping parcel-sort scene.

    Five cameras with overlapping coverage watch one handler walk five sorting
    stations, dwelling at each. Returns a :class:`Scene` carrying the per-station
    :class:`CameraTopology` and no occluders.
    """
    cameras = _cameras()

    path = _handler_path()
    num_frames = _num_frames(path)
    frames = path.compile_frames(_FPS, num_frames, name="center")
    handler = Entity(id=HANDLER_ID, frames=frames)

    # Each sorting station is covered by its own camera (same index); the walk
    # chains station_i <-> station_{i+1} with the transit time between them.
    transit_s = _WALK_S
    stations = [Station(id=station_id, camera_ids=[i]) for i, station_id in enumerate(STATION_IDS)]
    edges: list[TransitEdge] = []
    for i in range(_NUM_STATIONS - 1):
        src, dst = STATION_IDS[i], STATION_IDS[i + 1]
        edges.append(TransitEdge(src=src, dst=dst, transit_time_s=transit_s))
        edges.append(TransitEdge(src=dst, dst=src, transit_time_s=transit_s))
    topology = CameraTopology(stations=stations, edges=edges)

    return Scene(
        fps=_FPS,
        num_frames=num_frames,
        cameras=cameras,
        entities=[handler],
        occluders=[],
        topology=topology,
    )

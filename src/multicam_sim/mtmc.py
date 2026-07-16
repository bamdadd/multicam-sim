"""MTMC (multi-target multi-camera) non-overlapping smoke scene.

Hand-specified, GL-free. Three cameras on TWO stations with disjoint fields of
view, and one object that travels through all three coverage regimes in a single
take:

  * **station A** (camera 0 alone) sees the object early — a single-camera
    interval that is (correctly) NOT triangulable;
  * a **blind gap** in the middle where NO camera sees it — labelled
    ``in_view=False`` on every camera, an interval, not an error;
  * **station B** (cameras 1 and 2, an overlapping stereo pair) sees it late —
    two views, so the real DLT recovers ground truth.

The object keeps ONE stable ``entity.id`` throughout — that id is the
cross-camera ground-truth identity a tracker must preserve across the gap
(issue #11, stable track ids). The scene carries a :class:`CameraTopology` so a
consumer knows A and B are adjacent and how long the transit takes.
"""

from __future__ import annotations

from typing import Any

from .cameras import Camera
from .dsl.motion import Path
from .dsl.rig import CameraRig, StationView
from .entities import Entity
from .scene import Scene
from .topology import CameraTopology, Station, TransitEdge

_WIDTH = 640
_HEIGHT_PX = 480
_NUM_FRAMES = 15
_FPS = 30.0

# Object sweeps in x at fixed y, z; station A frames x=-6, station B frames x=+6,
# the gap sits around x=0 (far outside every camera's cone).
_A_X = -6.0
_B_X = 6.0
_TRACK_Y = 0.0
_TRACK_Z = 1.0
_CAM_Y = -5.0  # cameras stand back along -y, looking toward the track line

TRACK_ID = "target-1"
STATION_A = "A"
STATION_B = "B"


def _station_views() -> list[StationView]:
    """Two stations, disjoint FOVs; station B is a stereo pair with slightly
    different per-camera targets/fov (exercising per-camera intrinsics)."""
    return [
        # station A: one camera on the x=-6 region
        StationView(position=(_A_X, _CAM_Y, 1.5), look_at=(_A_X, 0.0, 1.0), fov_deg=55.0),
        # station B: overlapping stereo pair on the x=+6 region, baseline in x
        StationView(position=(4.6, _CAM_Y, 1.5), look_at=(_B_X, 0.0, 1.0), fov_deg=55.0),
        StationView(position=(7.6, _CAM_Y, 1.9), look_at=(_B_X + 0.2, 0.0, 1.0), fov_deg=50.0),
    ]


def _cameras() -> list[Camera]:
    return CameraRig.stations(_station_views(), width=_WIDTH, height_px=_HEIGHT_PX)


def build_mtmc_scene() -> Scene:
    """Construct the deterministic non-overlapping MTMC smoke scene."""
    cameras = _cameras()

    # one object, stable id, sweeping A -> gap -> B
    a = (_A_X, _TRACK_Y, _TRACK_Z)
    b = (_B_X, _TRACK_Y, _TRACK_Z)
    frames = Path.waypoints([a, b]).compile_frames(_FPS, _NUM_FRAMES, name="center")
    entity = Entity(id=TRACK_ID, frames=frames)

    # station adjacency: A <-> B, transit ~ the blind-gap duration
    transit_s = _NUM_FRAMES / _FPS / 2.0
    topology = CameraTopology(
        stations=[
            Station(id=STATION_A, camera_ids=[0]),
            Station(id=STATION_B, camera_ids=[1, 2]),
        ],
        edges=[
            TransitEdge(src=STATION_A, dst=STATION_B, transit_time_s=transit_s),
            TransitEdge(src=STATION_B, dst=STATION_A, transit_time_s=transit_s),
        ],
    )

    return Scene(
        fps=_FPS,
        num_frames=_NUM_FRAMES,
        cameras=cameras,
        entities=[entity],
        occluders=[],
        topology=topology,
    )


def entity_camera_intervals(
    manifest: dict[str, Any], entity_id: str
) -> dict[int, list[tuple[int, int]]]:
    """Derive per-camera ``[enter, leave]`` frame intervals for ``entity_id``.

    Intervals are derived from the ``false -> true`` and ``true -> false``
    transitions of the per-camera ``in_view`` flag recorded in ``manifest``.
    For an entity with multiple named points, a camera is considered to see the
    entity in a frame when **any** of its points has ``in_view=True``.

    Returns a mapping ``camera_id -> list[(enter_frame, leave_frame)]`` with one
    key per camera declared in ``manifest["cameras"]``. Intervals are sorted in
    ascending frame order.

    Boundary convention:

    * Intervals are **closed**: both ``enter_frame`` and ``leave_frame`` are
      frames where the entity is in view.
    * A track visible at frame 0 enters at ``0``.
    * A track still visible at the final recorded frame leaves at that frame
      (no open-ended sentinel).
    * A camera that never sees the entity maps to an empty list.
    """
    try:
        entity = next(e for e in manifest["entities"] if e["id"] == entity_id)
    except StopIteration as exc:
        raise ValueError(f"entity {entity_id!r} not found in manifest") from exc

    camera_ids = [cam["id"] for cam in manifest["cameras"]]
    cam_index = {cam_id: idx for idx, cam_id in enumerate(camera_ids)}
    result: dict[int, list[tuple[int, int]]] = {cam_id: [] for cam_id in camera_ids}

    frames = sorted(entity["frames"], key=lambda f: f["frame"])
    if not frames:
        return result

    sequences: dict[int, list[bool]] = {cam_id: [] for cam_id in camera_ids}
    frame_numbers: list[int] = []
    for frame in frames:
        frame_numbers.append(frame["frame"])
        per_point_per_cam = [p["per_cam"] for p in frame["points"].values()]
        for cam_id, idx in cam_index.items():
            in_view = any(obs[idx]["in_view"] for obs in per_point_per_cam)
            sequences[cam_id].append(in_view)

    for cam_id, flags in sequences.items():
        intervals = result[cam_id]
        enter: int | None = None
        prev_frame: int | None = None
        for frame, flag in zip(frame_numbers, flags, strict=True):
            if flag and enter is None:
                enter = frame
            elif not flag and enter is not None:
                assert prev_frame is not None
                intervals.append((enter, prev_frame))
                enter = None
            prev_frame = frame
        if enter is not None:
            assert prev_frame is not None
            intervals.append((enter, prev_frame))

    return result

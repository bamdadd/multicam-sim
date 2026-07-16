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

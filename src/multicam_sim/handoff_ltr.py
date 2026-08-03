"""Left-to-right handoff scene: a parcel crosses the space, camera to camera.

A domain-neutral scenario built to show a moving parcel being handed off across
overlapping cameras as it travels in a straight line from left to right (world
``-x -> +x``):

  * a **left** camera frames the left of the space: the parcel starts fully in
    its view and gradually leaves it as it moves right;
  * two **right** cameras frame the middle and right of the space with
    overlapping fields of view, so as the parcel leaves the left camera it
    appears first in the middle-right camera and then, over an overlap band, in
    both right cameras at once;
  * a **wide** camera stands well back and frames the whole line, so it sees the
    parcel on EVERY frame — the always-on reference view.

The result is a graduated coverage hand-off recorded in the manifest's per-camera
``in_view`` flags with no schema change: the left camera's ``in_view`` falls from
true to false as the parcel exits, the two right cameras' rise as it enters (with
an overlap band where both are true), and the wide camera is true throughout. The
parcel keeps ONE stable ``entity.id``, the cross-camera identity a tracker must
preserve across the hand-off.
"""

from __future__ import annotations

from .cameras import Camera
from .dsl.motion import Path
from .dsl.rig import CameraRig, StationView
from .entities import Entity
from .scene import Scene
from .topology import CameraTopology, Station, TransitEdge

_WIDTH = 1280
_HEIGHT_PX = 720
_FPS = 10.0
_NUM_FRAMES = 40

#: The parcel travels this straight line, left to right, at torso height.
_START_X = -8.0
_END_X = 8.0
_PARCEL_Y = 0.0
_PARCEL_Z = 1.0

#: Near cameras stand back off the line along -y, looking toward it.
_CAM_Y = -5.0
_CAM_Z = 1.8
#: The wide reference camera stands off the FAR RIGHT end of the line and looks
#: down its length toward the far left, so the whole path stays in its frustum.
_WIDE_CAM_X = 13.0
_WIDE_CAM_Y = -6.0
_WIDE_CAM_Z = 3.0

PARCEL_ID = "parcel-1"

#: Camera ids, in declaration order.
CAM_LEFT = 0
CAM_MID_RIGHT = 1
CAM_RIGHT = 2
CAM_WIDE = 3


def _station_views() -> list[StationView]:
    """Four cameras: left, middle-right, right, and a wide always-on reference.

    The three near cameras are aimed at successive x-regions of the line with a
    moderate FOV so their coverage overlaps in bands; the wide camera sits far
    back with a large FOV so the whole line stays inside its frustum every frame.
    """
    return [
        # Left camera: frames the left of the line; the parcel starts here.
        StationView(position=(-5.0, _CAM_Y, _CAM_Z), look_at=(-5.0, 0.0, _PARCEL_Z), fov_deg=70.0),
        # Middle-right camera: frames the centre; picks the parcel up while it is
        # still in the left camera (an overlap band, no blind gap), overlapping
        # both neighbours.
        StationView(position=(0.0, _CAM_Y, _CAM_Z), look_at=(0.0, 0.0, _PARCEL_Z), fov_deg=75.0),
        # Right camera: frames the right end; overlaps the middle-right camera in
        # a band, then holds the parcel to the end.
        StationView(position=(6.0, _CAM_Y, _CAM_Z), look_at=(6.0, 0.0, _PARCEL_Z), fov_deg=60.0),
        # Wide reference camera: stands off the FAR RIGHT end and looks down the
        # length of the line toward the far left, so the whole left-to-right path
        # runs away from it and stays in view every frame.
        StationView(
            position=(_WIDE_CAM_X, _WIDE_CAM_Y, _WIDE_CAM_Z),
            look_at=(_START_X, 0.0, _PARCEL_Z),
            fov_deg=90.0,
        ),
    ]


def _cameras() -> list[Camera]:
    return CameraRig.stations(_station_views(), width=_WIDTH, height_px=_HEIGHT_PX)


def build_handoff_ltr_scene() -> Scene:
    """Construct the deterministic left-to-right cross-camera handoff scene."""
    cameras = _cameras()

    start = (_START_X, _PARCEL_Y, _PARCEL_Z)
    end = (_END_X, _PARCEL_Y, _PARCEL_Z)
    frames = Path.linear(start, end).compile_frames(_FPS, _NUM_FRAMES, name="center")
    entity = Entity(id=PARCEL_ID, frames=frames)

    # Topology: three near stations left -> middle-right -> right, plus the wide
    # reference station. Directed transit edges chain the left-to-right hand-off.
    transit_s = _NUM_FRAMES / _FPS / 3.0
    topology = CameraTopology(
        stations=[
            Station(id="left", camera_ids=[CAM_LEFT]),
            Station(id="mid_right", camera_ids=[CAM_MID_RIGHT]),
            Station(id="right", camera_ids=[CAM_RIGHT]),
            Station(id="wide", camera_ids=[CAM_WIDE]),
        ],
        edges=[
            TransitEdge(src="left", dst="mid_right", transit_time_s=transit_s),
            TransitEdge(src="mid_right", dst="right", transit_time_s=transit_s),
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

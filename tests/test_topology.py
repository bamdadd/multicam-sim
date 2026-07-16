"""CameraTopology validation: stations reference real scene cameras."""

from __future__ import annotations

import pytest

from multicam_sim.cameras import Camera, Intrinsics
from multicam_sim.entities import Entity, EntityFrame
from multicam_sim.scene import Scene
from multicam_sim.topology import CameraTopology, Station, TransitEdge


def _camera(camera_id: int) -> Camera:
    return Camera(
        id=camera_id,
        intrinsics=Intrinsics.from_focal(800.0, 640, 480),
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.0, 0.0, 0.0],
    )


def _minimal_scene(topology: CameraTopology | None) -> Scene:
    return Scene(
        fps=30.0,
        num_frames=1,
        cameras=[_camera(0), _camera(1)],
        entities=[
            Entity(
                id="obj",
                frames=[EntityFrame(frame=0, points={"center": [0.0, 0.0, 0.0]})],
            )
        ],
        topology=topology,
    )


def test_scene_rejects_unknown_station_camera_id() -> None:
    """A station that names a camera not in the scene must fail with a clear message."""
    topology = CameraTopology(
        stations=[Station(id="A", camera_ids=[0, 99])],
        edges=[],
    )
    with pytest.raises(ValueError, match="station 'A' references unknown camera 99"):
        _minimal_scene(topology)


def test_valid_scene_with_topology_constructs() -> None:
    """A station whose cameras are all present in the scene should construct."""
    topology = CameraTopology(
        stations=[Station(id="A", camera_ids=[0]), Station(id="B", camera_ids=[1])],
        edges=[TransitEdge(src="A", dst="B", transit_time_s=1.0)],
    )
    scene = _minimal_scene(topology)
    assert scene.topology is not None
    assert {s.id for s in scene.topology.stations} == {"A", "B"}


def test_scene_without_topology_constructs() -> None:
    """The topology field is optional; a scene without it should still construct."""
    scene = _minimal_scene(topology=None)
    assert scene.topology is None

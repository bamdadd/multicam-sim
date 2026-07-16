"""Fluent, typed DSL layer on top of the multicam-sim Scene contract.

Nothing here changes the manifest contract or camera convention (see
``DESIGN.md``); the DSL only *compiles down* to the existing
:class:`multicam_sim.scene.Scene` and its manifest. The renderer backend
(:mod:`multicam_sim.dsl.render`) is intentionally NOT imported here — it is an
optional extra and importing it must never be required to build a scene.
"""

from __future__ import annotations

from .builder import SceneBuilder
from .motion import (
    BezierPath,
    CirclePath,
    LinearPath,
    Path,
    PathUnion,
    RepeatPath,
    SequencePath,
    WaypointPath,
)
from .occlusion import Occlusion
from .rig import CameraRig, StationView

__all__ = [
    "BezierPath",
    "CameraRig",
    "CirclePath",
    "LinearPath",
    "Occlusion",
    "Path",
    "PathUnion",
    "RepeatPath",
    "SceneBuilder",
    "SequencePath",
    "StationView",
    "WaypointPath",
]

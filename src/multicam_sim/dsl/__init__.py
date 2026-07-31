"""Fluent, typed DSL layer on top of the multicam-sim Scene contract.

Nothing here changes the manifest contract or camera convention (see
``DESIGN.md``); the DSL only *compiles down* to the existing
:class:`multicam_sim.scene.Scene` and its manifest. The renderer backend
(:mod:`multicam_sim.dsl.render`) is intentionally NOT imported here — it is an
optional extra and importing it must never be required to build a scene.
"""

from __future__ import annotations

from ..noise import CalibrationDrift, NoiseModel, PixelNoise
from .behavior import Behavior, PathBehavior, WaypointBehavior
from .builder import SceneBuilder
from .gait import Gait, ReachGait, WalkGait, WaveGait
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
from .occlusion import HandSweep, Occlusion
from .rig import CameraRig, PoseOverride, StationView

__all__ = [
    "Behavior",
    "BezierPath",
    "CalibrationDrift",
    "CameraRig",
    "CirclePath",
    "Gait",
    "HandSweep",
    "LinearPath",
    "NoiseModel",
    "Occlusion",
    "Path",
    "PathBehavior",
    "PathUnion",
    "PixelNoise",
    "PoseOverride",
    "ReachGait",
    "RepeatPath",
    "SceneBuilder",
    "SequencePath",
    "StationView",
    "WalkGait",
    "WaveGait",
    "WaypointBehavior",
    "WaypointPath",
]

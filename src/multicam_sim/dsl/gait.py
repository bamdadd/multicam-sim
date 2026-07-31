"""Skeletal-motion DSL: parametric gaits that animate the COCO-17 skeleton.

The motion DSL (:mod:`multicam_sim.dsl.motion`) moves a single named point; the
pose layer (:mod:`multicam_sim.pose`) represents a skeleton per frame but leaves
the authoring of 17 joint positions per frame to the caller. This module closes
that gap: a **gait** generates body-local joint offsets as a function of
wall-clock time, and a **root path** (any existing ``PathUnion`` —
``LinearPath`` / ``CirclePath`` / ...) translates the whole skeleton:

    world_joint(frame) = root.at_time(t) + local_offset(gait, t)

Timing is NOT re-implemented here. Frame compilation is driven through the
existing :meth:`_PathNode.compile_frames`, so ``over(seconds)`` / ``at_speed``
on the root path stretch or retime the translation exactly as they do for a
single point, and the gait samples its own motion at the same per-frame
wall-clock times (``t = frame / fps``). Gait cadence (step/wave frequency,
reach duration) is parameterised in seconds, on the same wall-clock axis the
timing model already uses.

Everything is kinematic and fully deterministic — there is no randomness to
seed: the same ``(gait, fps, num_frames)`` yields byte-identical joint
positions. Anatomical realism is explicitly not the bar; the gaits produce all
17 COCO joints, move continuously, and round-trip through the manifest.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from ..geometry import FloatArray
from ..pose import COCO17_JOINTS, PoseFrame, PoseTrajectory, Skeleton
from .motion import LinearPath, PathUnion, Vec3

#: A gait with no explicit root path stands at the origin.
_DEFAULT_ROOT = LinearPath(a=(0.0, 0.0, 0.0), b=(0.0, 0.0, 0.0))


def _standing_offsets(height: float) -> dict[str, FloatArray]:
    """Body-local offsets of a standing COCO-17 skeleton, relative to the root.

    The root sits at the pelvis centre; Z is up, the body faces +y, and the
    left side is -x (the same convention as the pose smoke scene). Proportions
    are fractions of total body ``height`` — a first pass, meant to be tuned.
    """
    h = height
    return {
        "nose": np.array([0.0, 0.0, 0.39 * h]),
        "left_eye": np.array([-0.02 * h, 0.0, 0.41 * h]),
        "right_eye": np.array([0.02 * h, 0.0, 0.41 * h]),
        "left_ear": np.array([-0.045 * h, 0.0, 0.39 * h]),
        "right_ear": np.array([0.045 * h, 0.0, 0.39 * h]),
        "left_shoulder": np.array([-0.13 * h, 0.0, 0.29 * h]),
        "right_shoulder": np.array([0.13 * h, 0.0, 0.29 * h]),
        "left_elbow": np.array([-0.155 * h, 0.0, 0.11 * h]),
        "right_elbow": np.array([0.155 * h, 0.0, 0.11 * h]),
        "left_wrist": np.array([-0.165 * h, 0.0, -0.06 * h]),
        "right_wrist": np.array([0.165 * h, 0.0, -0.06 * h]),
        "left_hip": np.array([-0.1 * h, 0.0, 0.0]),
        "right_hip": np.array([0.1 * h, 0.0, 0.0]),
        "left_knee": np.array([-0.1 * h, 0.0, -0.26 * h]),
        "right_knee": np.array([0.1 * h, 0.0, -0.26 * h]),
        "left_ankle": np.array([-0.105 * h, 0.0, -0.49 * h]),
        "right_ankle": np.array([0.105 * h, 0.0, -0.49 * h]),
    }


def _rot_x(theta: float) -> FloatArray:
    """Rotation about the x axis (swings a hanging limb in the y-z plane)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(theta: float) -> FloatArray:
    """Rotation about the y axis (swings a raised forearm side to side)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class _GaitBase(BaseModel):
    """Shared behaviour for every gait. Frozen; compilation returns a new
    :class:`PoseTrajectory` rather than mutating."""

    model_config = ConfigDict(frozen=True)

    #: World-frame trajectory of the skeleton root (pelvis centre). Composes
    #: with the gait: every joint is ``root position + body-local offset``.
    root: PathUnion = _DEFAULT_ROOT
    #: Total body height in world units; all body-local proportions scale with it.
    height: float = 1.7

    @field_validator("height")
    @classmethod
    def _positive_height(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("height must be > 0")
        return value

    # -- gait geometry (overridden per variant) ---------------------------- #
    def _local_offsets(self, t: float, base: dict[str, FloatArray]) -> dict[str, FloatArray]:
        """Body-local joint offsets at wall-clock time ``t`` seconds."""
        raise NotImplementedError

    def max_local_speed(self) -> float:
        """Analytic upper bound on any joint's body-local speed (units/second).

        Used to derive per-frame continuity bounds from the gait's own
        parameters: a joint can move at most
        ``(root_speed + max_local_speed()) / fps`` between frames.
        """
        raise NotImplementedError

    # -- compilation ------------------------------------------------------- #
    def to_pose_trajectory(
        self,
        entity_id: str,
        *,
        fps: float,
        num_frames: int,
    ) -> PoseTrajectory:
        """Compile the gait to a :class:`PoseTrajectory` over ``num_frames``.

        The root path is sampled through the motion DSL's own
        :meth:`_PathNode.compile_frames`, so timing combinators
        (``over`` / ``at_speed`` / untimed stretch-to-scene) behave exactly as
        they do for a single moving point. Every frame carries all 17 COCO-17
        joints.
        """
        if fps <= 0.0:
            raise ValueError("fps must be > 0")
        root_frames = self.root.compile_frames(fps, num_frames, name="root")
        base = _standing_offsets(self.height)
        frames: list[PoseFrame] = []
        for ef in root_frames:
            t = ef.frame / fps
            root_pos = np.asarray(ef.points["root"], dtype=np.float64)
            offsets = self._local_offsets(t, base)
            joints = {name: (root_pos + offsets[name]).tolist() for name in COCO17_JOINTS}
            frames.append(PoseFrame(frame=ef.frame, joints=joints))
        return PoseTrajectory(id=entity_id, skeleton=Skeleton.coco17(), frames=frames)


class WalkGait(_GaitBase):
    """Translating body with periodic legs and opposite-phase arm swing.

    Legs swing sinusoidally in the facing plane at ``step_frequency`` full
    gait cycles per second (left and right legs half a cycle apart); the foot
    lifts while swinging forward. Arms hang and swing in opposite phase to the
    same-side leg. A small vertical ``bob`` at twice the step frequency moves
    the whole body. All terms are continuous in ``t``.
    """

    step_frequency: float = 1.8  # gait cycles per second
    stride: float = 0.35  # peak forward/back ankle swing (world units)
    lift: float = 0.06  # peak foot lift during the forward swing (world units)
    arm_swing: float = 0.5  # peak arm swing angle (radians)
    bob: float = 0.02  # peak vertical body bob (world units)

    @field_validator("step_frequency")
    @classmethod
    def _positive_step_frequency(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("step_frequency must be > 0")
        return value

    @field_validator("stride", "lift", "arm_swing", "bob")
    @classmethod
    def _non_negative_amplitude(cls, value: float, info: ValidationInfo) -> float:
        if value < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return value

    def _local_offsets(self, t: float, base: dict[str, FloatArray]) -> dict[str, FloatArray]:
        phi = 2.0 * math.pi * self.step_frequency * t
        offsets = dict(base)
        for side, phase in (("left", phi), ("right", phi + math.pi)):
            swing = math.sin(phase)
            hip = base[f"{side}_hip"]
            ankle = base[f"{side}_ankle"] + np.array(
                [0.0, self.stride * swing, self.lift * max(0.0, swing)]
            )
            # the knee tracks the hip-ankle midpoint, pushed forward (+y)
            offsets[f"{side}_ankle"] = ankle
            offsets[f"{side}_knee"] = 0.5 * (hip + ankle) + np.array([0.0, 0.05 * self.height, 0.0])
            # same-side arm swings in opposite phase to the leg
            rot = _rot_x(-self.arm_swing * swing)
            shoulder = base[f"{side}_shoulder"]
            elbow = shoulder + rot @ (base[f"{side}_elbow"] - shoulder)
            offsets[f"{side}_elbow"] = elbow
            offsets[f"{side}_wrist"] = elbow + rot @ (base[f"{side}_wrist"] - base[f"{side}_elbow"])
        if self.bob > 0.0:
            dz = np.array([0.0, 0.0, self.bob * (1.0 - math.cos(2.0 * phi)) / 2.0])
            offsets = {name: pos + dz for name, pos in offsets.items()}
        return offsets

    def max_local_speed(self) -> float:
        """Bound: ankle ``omega*hypot(stride, lift)`` vs wrist
        ``omega*arm_swing*(upper_arm + forearm)``, plus the ``omega*bob``
        whole-body term that rides on every joint. ``omega = 2*pi*step_frequency``.
        """
        omega = 2.0 * math.pi * self.step_frequency
        base = _standing_offsets(self.height)
        ankle_speed = omega * math.hypot(self.stride, self.lift)
        upper_arm = float(np.linalg.norm(base["left_elbow"] - base["left_shoulder"]))
        forearm = float(np.linalg.norm(base["left_wrist"] - base["left_elbow"]))
        arm_speed = omega * self.arm_swing * (upper_arm + forearm)
        return omega * self.bob + max(ankle_speed, arm_speed)


class ReachGait(_GaitBase):
    """One arm extends from its rest pose to a body-local ``target``.

    The wrist follows a smoothstep ramp (zero velocity at both ends, so the
    motion is continuous and holds at the target once reached); the elbow
    tracks toward the shoulder-target midpoint, dropped slightly below the
    line. Past ``reach_duration`` the pose holds at the target.
    """

    target: Vec3  # body-local point the wrist moves to
    reach_duration: float = 1.0  # seconds from rest to target
    arm: Literal["left", "right"] = "right"

    @field_validator("reach_duration")
    @classmethod
    def _positive_duration(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("reach_duration must be > 0")
        return value

    def _elbow_target(self, base: dict[str, FloatArray]) -> FloatArray:
        shoulder = base[f"{self.arm}_shoulder"]
        target = np.asarray(self.target, dtype=np.float64)
        mid: FloatArray = 0.5 * (shoulder + target) + np.array([0.0, 0.0, -0.05 * self.height])
        return mid

    def _local_offsets(self, t: float, base: dict[str, FloatArray]) -> dict[str, FloatArray]:
        u = min(max(t / self.reach_duration, 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smoothstep: continuous, zero end velocities
        target = np.asarray(self.target, dtype=np.float64)
        offsets = dict(base)
        wrist_rest = base[f"{self.arm}_wrist"]
        elbow_rest = base[f"{self.arm}_elbow"]
        offsets[f"{self.arm}_wrist"] = wrist_rest + s * (target - wrist_rest)
        offsets[f"{self.arm}_elbow"] = elbow_rest + s * (self._elbow_target(base) - elbow_rest)
        return offsets

    def max_local_speed(self) -> float:
        """Bound: smoothstep's peak derivative is 1.5, so a joint covering a
        rest-to-goal distance ``d`` moves at most ``1.5 * d / reach_duration``.
        """
        base = _standing_offsets(self.height)
        target = np.asarray(self.target, dtype=np.float64)
        wrist_dist = float(np.linalg.norm(target - base[f"{self.arm}_wrist"]))
        elbow_dist = float(np.linalg.norm(self._elbow_target(base) - base[f"{self.arm}_elbow"]))
        return 1.5 * max(wrist_dist, elbow_dist) / self.reach_duration


class WaveGait(_GaitBase):
    """One arm raised, forearm oscillating side to side at ``wave_frequency``.

    The upper arm is held out and slightly up; the forearm points up and
    rotates about the body-front (y) axis by
    ``wave_amplitude * sin(2*pi*wave_frequency*t)``. Everything else holds the
    standing pose.
    """

    wave_frequency: float = 2.0  # oscillations per second
    wave_amplitude: float = 0.6  # peak forearm swing angle (radians)
    arm: Literal["left", "right"] = "right"

    @field_validator("wave_frequency")
    @classmethod
    def _positive_wave_frequency(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("wave_frequency must be > 0")
        return value

    @field_validator("wave_amplitude")
    @classmethod
    def _non_negative_amplitude(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("wave_amplitude must be >= 0")
        return value

    def _local_offsets(self, t: float, base: dict[str, FloatArray]) -> dict[str, FloatArray]:
        sign = -1.0 if self.arm == "left" else 1.0
        shoulder = base[f"{self.arm}_shoulder"]
        elbow = shoulder + np.array([sign * 0.16 * self.height, 0.0, 0.04 * self.height])
        forearm = np.array([0.0, 0.0, 0.17 * self.height])
        theta = self.wave_amplitude * math.sin(2.0 * math.pi * self.wave_frequency * t)
        offsets = dict(base)
        offsets[f"{self.arm}_elbow"] = elbow
        offsets[f"{self.arm}_wrist"] = elbow + _rot_y(theta) @ forearm
        return offsets

    def max_local_speed(self) -> float:
        """Bound: only the waving wrist moves; its speed is at most
        ``wave_amplitude * 2*pi*wave_frequency * forearm_length``.
        """
        forearm_len = 0.17 * self.height
        return self.wave_amplitude * 2.0 * math.pi * self.wave_frequency * forearm_len


class Gait:
    """Namespace of constructors for the skeletal-motion DSL (autocomplete entry
    point), mirroring :class:`~multicam_sim.dsl.motion.Path`.

    ``Gait.walk(...)`` etc. return concrete gait models; the root path (with
    its ``over`` / ``at_speed`` timing combinators) lives on the ``root``
    field, and :meth:`_GaitBase.to_pose_trajectory` compiles to a
    ``PoseTrajectory``.
    """

    @staticmethod
    def walk(
        *,
        root: PathUnion = _DEFAULT_ROOT,
        height: float = 1.7,
        step_frequency: float = 1.8,
        stride: float = 0.35,
        lift: float = 0.06,
        arm_swing: float = 0.5,
        bob: float = 0.02,
    ) -> WalkGait:
        return WalkGait(
            root=root,
            height=height,
            step_frequency=step_frequency,
            stride=stride,
            lift=lift,
            arm_swing=arm_swing,
            bob=bob,
        )

    @staticmethod
    def reach(
        target: Vec3,
        *,
        root: PathUnion = _DEFAULT_ROOT,
        height: float = 1.7,
        reach_duration: float = 1.0,
        arm: Literal["left", "right"] = "right",
    ) -> ReachGait:
        return ReachGait(
            root=root,
            height=height,
            target=target,
            reach_duration=reach_duration,
            arm=arm,
        )

    @staticmethod
    def wave(
        *,
        root: PathUnion = _DEFAULT_ROOT,
        height: float = 1.7,
        wave_frequency: float = 2.0,
        wave_amplitude: float = 0.6,
        arm: Literal["left", "right"] = "right",
    ) -> WaveGait:
        return WaveGait(
            root=root,
            height=height,
            wave_frequency=wave_frequency,
            wave_amplitude=wave_amplitude,
            arm=arm,
        )

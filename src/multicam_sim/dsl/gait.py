"""Skeletal-motion DSL: parametric gaits that animate the COCO-17 skeleton.

The motion DSL (:mod:`multicam_sim.dsl.motion`) moves a single named point; the
pose layer (:mod:`multicam_sim.pose`) represents a skeleton per frame but leaves
the authoring of 17 joint positions per frame to the caller. The pose layer
alone can already be animated by hand — ``examples/assembly_station.py`` builds
a ``PoseTrajectory`` from a literal 17-joint offset table with a sinusoidal
wrist reach — but every offset is hand-authored per frame, with no timing
model, no root path, and no reusable library API. This module is that library
layer: a **gait** generates body-local joint offsets as a function of
wall-clock time, and a **root path** (any existing ``PathUnion`` —
``LinearPath`` / ``CirclePath`` / ...) translates the whole skeleton:

    world_joint(frame) = root.at_time(t) + local_offset(gait, t)

Limbs are rigid: intermediate joints (knees, the reaching elbow) are placed by
closed-form two-link inverse kinematics with segment lengths fixed from
``height``, so every COCO-17 edge keeps a constant length on every frame. These
trajectories are ground truth for downstream consumers; a bone that changes
length is structurally wrong output, not merely unrealistic.

Timing is NOT re-implemented here. Frame compilation is driven through the
existing :meth:`_PathNode.compile_frames`, so ``over(seconds)`` / ``at_speed``
on the root path stretch or retime the translation exactly as they do for a
single point, and the gait samples its own motion at the same per-frame
wall-clock times (``t = frame / fps``).

NOTE: ``over(seconds)`` / ``at_speed(v)`` retime the **root translation only**.
Gait cadence (step/wave frequency, reach duration) is parameterised in
wall-clock seconds and does NOT stretch with them.

Everything is kinematic and fully deterministic — there is no randomness to
seed: the same ``(gait, fps, num_frames)`` yields byte-identical joint
positions. Anatomical realism is explicitly not the bar; the gaits produce all
17 COCO joints, rigid limbs, continuous motion, and round-trip through the
manifest.
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
    The knees stand slightly bent (+y) so the two-link leg solve starts away
    from full extension.
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
        "left_knee": np.array([-0.1 * h, 0.05 * h, -0.26 * h]),
        "right_knee": np.array([0.1 * h, 0.05 * h, -0.26 * h]),
        "left_ankle": np.array([-0.1 * h, 0.0, -0.49 * h]),
        "right_ankle": np.array([0.1 * h, 0.0, -0.49 * h]),
    }


def _rot_x(theta: float) -> FloatArray:
    """Rotation about the x axis (swings a hanging limb in the y-z plane)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(theta: float) -> FloatArray:
    """Rotation about the y axis (swings a raised forearm side to side)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


#: Margin kept from full extension when clamping an unreachable endpoint.
#: At exactly ``l1 + l2`` the two-link solve has ``h = 0`` and its condition
#: number diverges (a tiny endpoint change moves the mid joint arbitrarily
#: fast); clamping a whisker inside the reachable sphere keeps the solve —
#: and the joint velocities it produces — bounded, while the clamped endpoint
#: still tracks the reachable sphere continuously.
_IK_MARGIN = 5e-4

#: Safety factor on the densely sampled speed used for IK mid-joint bounds:
#: finite differences on a smooth function underestimate the true peak, and
#: the grid may sit slightly off the peak phase.
_SAMPLE_SAFETY = 1.5


def _two_link(
    start: FloatArray,
    end: FloatArray,
    l1: float,
    l2: float,
    bend: Vec3,
) -> tuple[FloatArray, FloatArray]:
    """Closed-form two-link IK: place the mid joint between ``start`` and ``end``.

    ``l1`` / ``l2`` are the fixed proximal/distal segment lengths. Returns
    ``(mid, end_used)``: when the endpoint is beyond reach it is first clamped
    to just inside the reachable sphere (see ``_IK_MARGIN``), and ``end_used``
    is the clamped endpoint so the caller's end joint agrees with the solve.
    ``bend`` picks the side the mid joint bows toward (its component
    perpendicular to the limb axis); if ``bend`` is (near-)parallel to the
    limb, an arbitrary fixed perpendicular is used instead.
    """
    delta = end - start
    d = float(np.linalg.norm(delta))
    if d < 1e-12:
        # degenerate: start and end coincide; fold the limb along `bend`
        e = np.asarray(bend, dtype=np.float64)
        e = e / np.linalg.norm(e)
        return start + l1 * e, end
    cap = (l1 + l2) * (1.0 - _IK_MARGIN)
    floor = abs(l1 - l2) + _IK_MARGIN * (l1 + l2)
    d_eff = min(max(d, floor), cap)
    if d_eff != d:
        delta = delta * (d_eff / d)
        end = start + delta
        d = d_eff
    e = delta / d
    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    n = np.asarray(bend, dtype=np.float64)
    n = n - float(n @ e) * e
    if float(np.linalg.norm(n)) < 1e-9:
        ref = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = ref - float(ref @ e) * e
    n = n / np.linalg.norm(n)
    return start + a * e + h * n, end


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
        """Upper bound on any joint's body-local speed (units/second).

        Endpoints and rigidly rotated joints are bounded analytically from the
        gait's own parameters (derivatives of the sinusoid / smoothstep terms).
        IK-placed mid joints (knees, the reaching elbow) are bounded by
        sampling the gait's own offsets on a dense deterministic grid over one
        period and taking the max finite-difference speed with a safety factor
        (``_SAMPLE_SAFETY``) — the two-link solve has no simple closed-form
        velocity bound, but it is smooth, so a dense sample plus margin is
        tight and honest. Either way the bound is derived from the gait's own
        parameters, never hardcoded: a joint can move at most
        ``(root_speed + max_local_speed()) / fps`` between frames.
        """
        raise NotImplementedError

    def _sampled_max_speed(self, period: float, samples: int = 720) -> float:
        """Max per-joint body-local speed of this gait, finite-differenced on a
        deterministic grid over ``period`` seconds."""
        base = _standing_offsets(self.height)
        prev = self._local_offsets(0.0, base)
        best = 0.0
        for i in range(1, samples + 1):
            cur = self._local_offsets(period * i / samples, base)
            for name in COCO17_JOINTS:
                speed = float(np.linalg.norm(cur[name] - prev[name])) * samples / period
                best = max(best, speed)
            prev = cur
        return best

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

    Each foot rides a sphere of radius ``0.97 * (thigh + shin)`` about its hip,
    swinging in the facing plane at ``step_frequency`` gait cycles per second
    (left and right legs half a cycle apart); the spherical swing lifts the
    foot naturally at both extremes and always stays inside the leg's reach,
    so the knee's two-link solve is well-conditioned everywhere. The knee is
    placed by rigid two-link IK with fixed thigh/shin lengths and bows forward.
    Arms hang and swing in opposite phase to the same-side leg. A small
    vertical ``bob`` at twice the step frequency moves the whole body. All
    terms are continuous in ``t``.
    """

    step_frequency: float = 1.8  # gait cycles per second
    swing: float = 0.35  # peak leg swing angle about the hip (radians)
    arm_swing: float = 0.5  # peak arm swing angle (radians)
    bob: float = 0.02  # peak vertical body bob (world units)

    @field_validator("step_frequency")
    @classmethod
    def _positive_step_frequency(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("step_frequency must be > 0")
        return value

    @field_validator("swing", "arm_swing", "bob")
    @classmethod
    def _non_negative_amplitude(cls, value: float, info: ValidationInfo) -> float:
        if value < 0.0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return value

    def _local_offsets(self, t: float, base: dict[str, FloatArray]) -> dict[str, FloatArray]:
        phi = 2.0 * math.pi * self.step_frequency * t
        thigh = float(np.linalg.norm(base["left_knee"] - base["left_hip"]))
        shin = float(np.linalg.norm(base["left_ankle"] - base["left_knee"]))
        leg_radius = 0.97 * (thigh + shin)  # foot swings inside the leg's reach
        offsets = dict(base)
        for side, phase in (("left", phi), ("right", phi + math.pi)):
            theta = self.swing * math.sin(phase)
            hip = base[f"{side}_hip"]
            ankle_target = hip + leg_radius * np.array([0.0, math.sin(theta), -math.cos(theta)])
            # rigid two-link leg: fixed thigh/shin lengths, knee bows forward (+y)
            knee, ankle = _two_link(hip, ankle_target, thigh, shin, (0.0, 1.0, 0.0))
            offsets[f"{side}_ankle"] = ankle
            offsets[f"{side}_knee"] = knee
            # same-side arm swings in opposite phase to the leg
            rot = _rot_x(-self.arm_swing * math.sin(phase))
            shoulder = base[f"{side}_shoulder"]
            elbow = shoulder + rot @ (base[f"{side}_elbow"] - shoulder)
            offsets[f"{side}_elbow"] = elbow
            offsets[f"{side}_wrist"] = elbow + rot @ (base[f"{side}_wrist"] - base[f"{side}_elbow"])
        if self.bob > 0.0:
            dz = np.array([0.0, 0.0, self.bob * (1.0 - math.cos(2.0 * phi)) / 2.0])
            offsets = {name: pos + dz for name, pos in offsets.items()}
        return offsets

    def max_local_speed(self) -> float:
        """Fully analytic: the foot rides a sphere of radius ``0.97*(thigh +
        shin)`` at angular speed ``swing * omega``, so the ankle moves at most
        ``0.97*(thigh+shin)*swing*omega`` and the IK knee exactly ``thigh *
        swing * omega`` (its perpendicular and along-limb parts rotate
        together); the wrist moves at most ``arm_swing * omega * (upper_arm +
        forearm)``; ``bob * omega`` rides on every joint. ``omega =
        2*pi*step_frequency``.
        """
        omega = 2.0 * math.pi * self.step_frequency
        base = _standing_offsets(self.height)
        thigh = float(np.linalg.norm(base["left_knee"] - base["left_hip"]))
        shin = float(np.linalg.norm(base["left_ankle"] - base["left_knee"]))
        leg_speed = 0.97 * (thigh + shin) * self.swing * omega
        upper_arm = float(np.linalg.norm(base["left_elbow"] - base["left_shoulder"]))
        forearm = float(np.linalg.norm(base["left_wrist"] - base["left_elbow"]))
        arm_speed = omega * self.arm_swing * (upper_arm + forearm)
        return omega * self.bob + max(leg_speed, arm_speed)


class ReachGait(_GaitBase):
    """One arm extends from its rest pose to a body-local ``target``.

    The wrist follows a smoothstep ramp (zero velocity at both ends, so the
    motion is continuous and holds at the target once reached); the elbow is
    solved by rigid two-link IK with fixed upper-arm/forearm lengths, bowing
    outward and down from the shoulder-target line as the rest elbow does. A
    target beyond the arm's reach is clamped to the reachable sphere. Past
    ``reach_duration`` the pose holds at the target.
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

    def _effective_target(self, base: dict[str, FloatArray]) -> FloatArray:
        """The reach destination, clamped to just inside the arm's reach.

        Clamping the destination once — rather than the endpoint per frame —
        means the smoothstep approaches the reachable sphere with vanishing
        velocity and never crosses it mid-swing, so clamping cannot introduce
        a velocity spike (or any discontinuity) in the wrist or the IK elbow.
        """
        shoulder = base[f"{self.arm}_shoulder"]
        upper_arm = float(np.linalg.norm(base[f"{self.arm}_elbow"] - shoulder))
        forearm = float(np.linalg.norm(base[f"{self.arm}_wrist"] - base[f"{self.arm}_elbow"]))
        to_target = np.asarray(self.target, dtype=np.float64) - shoulder
        dist = float(np.linalg.norm(to_target))
        cap = (upper_arm + forearm) * (1.0 - _IK_MARGIN)
        if dist <= cap or dist < 1e-12:
            return np.asarray(self.target, dtype=np.float64)
        clamped: FloatArray = shoulder + to_target * (cap / dist)
        return clamped

    def _local_offsets(self, t: float, base: dict[str, FloatArray]) -> dict[str, FloatArray]:
        u = min(max(t / self.reach_duration, 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smoothstep: continuous, zero end velocities
        shoulder = base[f"{self.arm}_shoulder"]
        wrist_rest = base[f"{self.arm}_wrist"]
        upper_arm = float(np.linalg.norm(base[f"{self.arm}_elbow"] - shoulder))
        forearm = float(np.linalg.norm(wrist_rest - base[f"{self.arm}_elbow"]))
        wrist_target = wrist_rest + s * (self._effective_target(base) - wrist_rest)
        # rigid two-link arm: the elbow bows outward and down from the
        # shoulder-wrist line, as the rest elbow does relative to the rest arm
        sign = -1.0 if self.arm == "left" else 1.0
        bend = (sign * 0.4, 0.0, -1.0)
        elbow, wrist = _two_link(shoulder, wrist_target, upper_arm, forearm, bend)
        offsets = dict(base)
        offsets[f"{self.arm}_elbow"] = elbow
        offsets[f"{self.arm}_wrist"] = wrist
        return offsets

    def max_local_speed(self) -> float:
        """Analytic term: smoothstep's peak derivative is 1.5, so the wrist
        endpoint moves at most ``1.5 * |effective_target - rest| /
        reach_duration``. The IK elbow has no closed-form velocity bound, so
        the result is the max of the analytic term and the densely sampled,
        safety-factored speed over the ramp.
        """
        base = _standing_offsets(self.height)
        wrist_dist = float(np.linalg.norm(self._effective_target(base) - base[f"{self.arm}_wrist"]))
        analytic = 1.5 * wrist_dist / self.reach_duration
        sampled = _SAMPLE_SAFETY * self._sampled_max_speed(2.0 * self.reach_duration)
        return max(analytic, sampled)


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
        swing: float = 0.35,
        arm_swing: float = 0.5,
        bob: float = 0.02,
    ) -> WalkGait:
        return WalkGait(
            root=root,
            height=height,
            step_frequency=step_frequency,
            swing=swing,
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

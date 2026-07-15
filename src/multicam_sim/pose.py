"""Human-pose types layered on the named-points contract.

A human pose is not a new schema. It is one :class:`~multicam_sim.entities.Entity`
whose named points are joints and whose ``edges`` are the skeleton, so a
:class:`PoseTrajectory` lowers to a plain ``Entity`` (see :meth:`PoseTrajectory.
to_entity`) and flows through the existing manifest builder unchanged. The
manifest then carries, per joint: the ground-truth 3D position, each camera's 2D
keypoint, and the per-camera ``visible`` / ``occ_frac`` labels. A joint occluded
in one view but seen in another is labelled as such, which is exactly what a
multi-view 3D pose estimator needs.

The default skeleton is COCO-17. A dense mesh body (SMPL / SMPL-X) is an
open/closed extension point: implement :class:`MeshBackend` to emit joints, and
the rest of the pipeline does not change. No mesh backend is implemented here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationInfo, field_validator

from .entities import Entity, EntityFrame

# COCO-17 keypoint names, in the canonical dataset order (index 0..16).
COCO17_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# The canonical COCO person skeleton (19 limbs), as joint-name pairs.
COCO17_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


class Skeleton(BaseModel):
    """A named joint set plus the limb edges connecting them.

    ``edges`` are pairs of joint names and must reference joints in ``joints``;
    they become the entity ``edges`` so a consumer can draw or reason about limbs.
    """

    name: str
    joints: list[str]
    edges: list[tuple[str, str]]

    @field_validator("joints")
    @classmethod
    def _joints_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("skeleton joints must be unique")
        if not value:
            raise ValueError("skeleton must have at least one joint")
        return value

    @field_validator("edges")
    @classmethod
    def _edges_reference_joints(
        cls, value: list[tuple[str, str]], info: ValidationInfo
    ) -> list[tuple[str, str]]:
        # `info.data` holds already-validated fields; joints is validated first.
        joints = set(info.data.get("joints", []))
        for a, b in value:
            if a not in joints or b not in joints:
                raise ValueError(f"edge {(a, b)!r} references a joint not in the skeleton")
        return value

    @classmethod
    def coco17(cls) -> Skeleton:
        """The default skeleton: 17 COCO keypoints and the 19-limb COCO skeleton."""
        return cls(name="coco17", joints=list(COCO17_JOINTS), edges=list(COCO17_EDGES))


class PoseFrame(BaseModel):
    """One frame of a pose: joint name -> ``[x, y, z]`` world ground truth."""

    frame: int
    joints: dict[str, list[float]]

    @field_validator("joints")
    @classmethod
    def _check_xyz(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        for name, xyz in value.items():
            if len(xyz) != 3:
                raise ValueError(f"joint {name!r} must be [x, y, z]; got {len(xyz)} coords")
        return value


class PoseTrajectory(BaseModel):
    """A skeleton animated across frames.

    Lowers to a plain :class:`~multicam_sim.entities.Entity` with zero schema
    fork: joints become named points, the skeleton becomes ``edges``. Every frame
    must supply exactly the skeleton's joints.
    """

    id: str
    skeleton: Skeleton
    frames: list[PoseFrame]

    @field_validator("frames")
    @classmethod
    def _frames_nonempty(cls, value: list[PoseFrame]) -> list[PoseFrame]:
        if not value:
            raise ValueError("pose trajectory must have at least one frame")
        return value

    def check_complete(self) -> None:
        """Raise if any frame is missing or adds a joint relative to the skeleton."""
        expected = set(self.skeleton.joints)
        for f in self.frames:
            got = set(f.joints)
            if got != expected:
                missing = expected - got
                extra = got - expected
                raise ValueError(
                    f"frame {f.frame} joints do not match skeleton "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )

    def to_entity(self) -> Entity:
        """Lower to an :class:`Entity` (named points + skeleton edges), no fork."""
        self.check_complete()
        return Entity(
            id=self.id,
            edges=self.skeleton.edges,
            frames=[EntityFrame(frame=f.frame, points=f.joints) for f in self.frames],
        )


class MeshBackend(ABC):
    """Open/closed extension point for a dense body model (SMPL / SMPL-X).

    A backend turns its own parameters into a :class:`PoseTrajectory` on some
    skeleton, after which the pipeline is identical to the COCO-17 path. No
    concrete backend ships in this layer; a future SMPL/SMPL-X implementation
    subclasses this without touching the manifest schema.
    """

    @abstractmethod
    def to_pose_trajectory(self, entity_id: str) -> PoseTrajectory:
        """Emit a joint trajectory this simulator can project and occlude."""
        raise NotImplementedError

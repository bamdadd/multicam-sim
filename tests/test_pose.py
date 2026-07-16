"""Pose types: COCO-17 skeleton, validation, and zero-fork lowering to Entity.

Model-free and CPU-only. The manifest round-trip proves a pose entity flows
through the existing builder with per-joint 2D/3D/occlusion labels, no schema
fork.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multicam_sim import (
    COCO17_EDGES,
    COCO17_JOINTS,
    Camera,
    Entity,
    Intrinsics,
    MeshBackend,
    PoseFrame,
    PoseTrajectory,
    Scene,
    Skeleton,
    build_manifest,
)


def _coco_frame(frame: int) -> PoseFrame:
    # 17 distinct joints clustered near the origin (all in front of the test cam).
    joints = {name: [0.1 * i, 0.02 * i, 0.05 * i] for i, name in enumerate(COCO17_JOINTS)}
    return PoseFrame(frame=frame, joints=joints)


def _one_camera() -> Camera:
    intr = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    # R = identity, camera pulled back +5 in z so the near-origin joints are in front.
    return Camera(
        id=0,
        intrinsics=intr,
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.0, 0.0, 5.0],
    )


def test_coco17_skeleton_shape() -> None:
    sk = Skeleton.coco17()
    assert sk.name == "coco17"
    assert len(sk.joints) == 17
    assert len(sk.edges) == 19
    joints = set(sk.joints)
    for a, b in sk.edges:
        assert a in joints and b in joints  # every edge references a real joint
    assert tuple(sk.joints) == COCO17_JOINTS
    assert tuple(sk.edges) == COCO17_EDGES


def test_skeleton_rejects_bad_edge_and_dup_joints() -> None:
    with pytest.raises(ValidationError):
        Skeleton(name="x", joints=["a", "b"], edges=[("a", "c")])  # c not a joint
    with pytest.raises(ValidationError):
        Skeleton(name="x", joints=["a", "a"], edges=[])  # duplicate joint


def test_pose_frame_rejects_bad_xyz() -> None:
    with pytest.raises(ValidationError):
        PoseFrame(frame=0, joints={"nose": [1.0, 2.0]})  # only 2 coords


def test_to_entity_zero_fork() -> None:
    traj = PoseTrajectory(id="person0", skeleton=Skeleton.coco17(), frames=[_coco_frame(0)])
    ent = traj.to_entity()
    assert isinstance(ent, Entity)
    assert ent.id == "person0"
    assert ent.point_names() == set(COCO17_JOINTS)  # 17 named points, no fork
    assert ent.edges == list(COCO17_EDGES)


def test_incomplete_frame_raises() -> None:
    partial = PoseFrame(frame=0, joints={"nose": [0.0, 0.0, 0.0]})
    traj = PoseTrajectory(id="p", skeleton=Skeleton.coco17(), frames=[partial])
    with pytest.raises(ValueError, match="do not match skeleton"):
        traj.to_entity()


def test_pose_flows_through_manifest_per_joint() -> None:
    traj = PoseTrajectory(id="person0", skeleton=Skeleton.coco17(), frames=[_coco_frame(0)])
    scene = Scene(
        fps=30.0,
        num_frames=1,
        cameras=[_one_camera()],
        entities=[traj.to_entity()],
        occluders=[],
    )
    manifest = build_manifest(scene)
    points = manifest.entities[0].frames[0].points
    assert set(points) == set(COCO17_JOINTS)  # every joint labelled
    nose = points["nose"]
    assert len(nose.xyz_gt) == 3  # GT 3D per joint
    assert len(nose.per_cam) == 1  # per-camera 2D keypoint + occlusion label
    obs = nose.per_cam[0]
    assert hasattr(obs, "cam") and hasattr(obs, "uv") and hasattr(obs, "visible")
    assert manifest.entities[0].edges == [list(e) for e in COCO17_EDGES]


def test_mesh_backend_is_abstract() -> None:
    with pytest.raises(TypeError):
        MeshBackend()  # type: ignore[abstract]  # cannot instantiate the extension point

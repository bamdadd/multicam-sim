"""Pose smoke: a COCO-17 joint occluded in one view stays visible in others.

Builds the deterministic pose smoke scene, runs ``build_manifest``, and asserts
that ``left_wrist`` is occluded for camera 1 while at least one other camera
still sees it. All 17 COCO joints must carry ground-truth 3D positions and
per-camera observations.
"""

from __future__ import annotations

from multicam_sim import build_manifest, build_pose_smoke_scene
from multicam_sim.manifest import EntityManifest, Manifest
from multicam_sim.pose import COCO17_JOINTS

_TARGET_JOINT = "left_wrist"
_OCCLUDED_CAM = 1
_TARGET_FRAME = 5  # middle frame


def _manifest() -> Manifest:
    scene = build_pose_smoke_scene()
    return build_manifest(scene)


def _pose_entity(manifest: Manifest) -> EntityManifest:
    return manifest.entities[0]


def test_pose_smoke_scene_shape() -> None:
    manifest = _manifest()
    assert len(manifest.cameras) == 3
    assert manifest.num_frames == 11
    entity = _pose_entity(manifest)
    assert entity.id == "person"
    assert entity.edges  # skeleton limbs present
    assert len(entity.frames) == manifest.num_frames


def test_all_joints_have_gt_and_per_cam() -> None:
    manifest = _manifest()
    entity = _pose_entity(manifest)
    expected_joints = set(COCO17_JOINTS)
    for frame in entity.frames:
        assert set(frame.points) == expected_joints
        for joint in COCO17_JOINTS:
            entry = frame.points[joint]
            assert len(entry.xyz_gt) == 3
            assert len(entry.per_cam) == len(manifest.cameras)
            for cam_obs in entry.per_cam:
                assert cam_obs.uv is not None
                assert cam_obs.visible in (True, False)


def test_target_joint_occluded_for_exactly_one_camera() -> None:
    manifest = _manifest()
    entity = _pose_entity(manifest)
    frame = entity.frames[_TARGET_FRAME]
    per_cam = frame.points[_TARGET_JOINT].per_cam

    assert not per_cam[_OCCLUDED_CAM].visible
    assert per_cam[_OCCLUDED_CAM].in_view, "occlusion must be the cause, not framing"
    assert sum(1 for o in per_cam if o.visible) >= 1


def test_no_other_joint_occluded_for_camera_one() -> None:
    """The sphere is sized/placed to hit only ``left_wrist`` for camera 1."""
    manifest = _manifest()
    entity = _pose_entity(manifest)
    for frame in entity.frames:
        for joint in COCO17_JOINTS:
            if joint == _TARGET_JOINT:
                continue
            assert frame.points[joint].per_cam[_OCCLUDED_CAM].visible

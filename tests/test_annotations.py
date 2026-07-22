"""COCO / YOLO annotation exporter contract (issue #40).

The exporter turns a manifest's per-camera ``uv`` + ``in_view`` / ``visible``
signals into 2D detection + keypoint annotations. These tests pin the image /
annotation / category structure, the visibility-flag mapping, and the
round-trip: export, parse back, and check the boxes match the manifest ``uv``
within tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multicam_sim import (
    build_manifest,
    build_multi_entity_scene,
    build_pose_smoke_scene,
    export_coco,
    export_yolo,
    write_coco,
    write_yolo,
)
from multicam_sim.annotations import CocoDataset
from multicam_sim.manifest import (
    CONVENTION,
    CameraManifest,
    EntityManifest,
    FrameObs,
    Manifest,
    PerCamObs,
    PointObs,
)


def _pose_manifest() -> Manifest:
    # object_radius opts into the silhouette occlusion labels; the pose scene
    # occludes left_wrist on camera 1, exercising the visible=False flag.
    return build_manifest(build_pose_smoke_scene(), object_radius=0.1)


def _expected_bbox(manifest: Manifest, entity_id: str, frame: int, camera_id: int):
    """Bounding box of an entity's in-view pixels, straight from the manifest."""
    entity = next(e for e in manifest.entities if e.id == entity_id)
    frame_obs = next(f for f in entity.frames if f.frame == frame)
    uvs: list[tuple[float, float]] = []
    for point in frame_obs.points.values():
        obs = next(o for o in point.per_cam if o.cam == camera_id)
        if obs.in_view:
            uvs.append((obs.uv[0], obs.uv[1]))
    if not uvs:
        return None
    us = [u for u, _ in uvs]
    vs = [v for _, v in uvs]
    x, y = min(us), min(vs)
    return [x, y, max(us) - x, max(vs) - y]


def test_images_cover_every_camera_frame_pair() -> None:
    manifest = _pose_manifest()
    dataset = export_coco(manifest)
    assert len(dataset.images) == len(manifest.cameras) * manifest.num_frames
    pairs = {(img.camera_id, img.frame) for img in dataset.images}
    expected = {(cam.id, frame) for cam in manifest.cameras for frame in range(manifest.num_frames)}
    assert pairs == expected
    # image ids are unique and every annotation points at a real image.
    assert len({img.id for img in dataset.images}) == len(dataset.images)
    image_ids = {img.id for img in dataset.images}
    assert all(ann.image_id in image_ids for ann in dataset.annotations)


def test_categories_mirror_entities() -> None:
    manifest = _pose_manifest()
    dataset = export_coco(manifest)
    assert [c.name for c in dataset.categories] == [e.id for e in manifest.entities]
    person = dataset.categories[0]
    # 17 COCO joints, and the skeleton is 1-indexed pairs into them.
    assert len(person.keypoints) == 17
    assert person.skeleton  # non-empty for the coco17 pose entity
    for a, b in person.skeleton:
        assert 1 <= a <= 17 and 1 <= b <= 17


def test_coco_bbox_matches_manifest_uv() -> None:
    manifest = _pose_manifest()
    dataset = export_coco(manifest)
    cat_name = {c.id: c.name for c in dataset.categories}
    image = {img.id: img for img in dataset.images}
    assert dataset.annotations  # the standing pose is seen
    for ann in dataset.annotations:
        img = image[ann.image_id]
        expected = _expected_bbox(manifest, cat_name[ann.category_id], img.frame, img.camera_id)
        assert expected is not None
        assert ann.bbox == pytest.approx(expected)
        assert ann.area == pytest.approx(expected[2] * expected[3])
        # bbox lies within the image frame.
        assert ann.bbox[0] >= 0.0 and ann.bbox[1] >= 0.0
        assert ann.bbox[0] + ann.bbox[2] <= img.width
        assert ann.bbox[1] + ann.bbox[3] <= img.height


def test_keypoint_visibility_flags_track_in_view_and_visible() -> None:
    manifest = _pose_manifest()
    dataset = export_coco(manifest)
    cat = {c.id: c for c in dataset.categories}
    image = {img.id: img for img in dataset.images}
    saw_visible = saw_occluded = False
    for ann in dataset.annotations:
        img = image[ann.image_id]
        names = cat[ann.category_id].keypoints
        entity = next(e for e in manifest.entities if e.id == cat[ann.category_id].name)
        frame_obs = next(f for f in entity.frames if f.frame == img.frame)
        labelled = 0
        for i, name in enumerate(names):
            x, y, v = ann.keypoints[3 * i : 3 * i + 3]
            obs = next(o for o in frame_obs.points[name].per_cam if o.cam == img.camera_id)
            if not obs.in_view:
                assert (x, y, v) == (0.0, 0.0, 0.0)
            else:
                labelled += 1
                assert (x, y) == pytest.approx((obs.uv[0], obs.uv[1]))
                if obs.visible:
                    assert v == 2.0
                    saw_visible = True
                else:
                    assert v == 1.0
                    saw_occluded = True
        assert ann.num_keypoints == labelled
    # camera 1's left_wrist is occluded, so both labelled states appear.
    assert saw_visible and saw_occluded


def test_out_of_view_keypoint_is_absent_and_excluded_from_bbox() -> None:
    """A point not ``in_view`` becomes an absent keypoint (0, 0, 0) and does not
    stretch the bounding box, which is built from the in-view pixels alone."""
    cam = CameraManifest(
        id=0,
        K=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.0, 0.0, 0.0],
        width=100,
        height=100,
        convention=CONVENTION,
    )
    frame = FrameObs(
        frame=0,
        points={
            "seen": PointObs(
                xyz_gt=[0.0, 0.0, 0.0],
                per_cam=[PerCamObs(cam=0, uv=[10.0, 20.0], in_view=True, visible=True)],
            ),
            "gone": PointObs(
                xyz_gt=[1.0, 1.0, 1.0],
                # in front of the camera but off-image: a real uv, in_view False.
                per_cam=[PerCamObs(cam=0, uv=[999.0, 999.0], in_view=False, visible=False)],
            ),
        },
    )
    manifest = Manifest(
        cameras=[cam],
        fps=30.0,
        num_frames=1,
        entities=[EntityManifest(id="thing", frames=[frame])],
    )
    dataset = export_coco(manifest)
    (ann,) = dataset.annotations
    assert ann.keypoints == [10.0, 20.0, 2.0, 0.0, 0.0, 0.0]
    assert ann.num_keypoints == 1
    # only the in-view point defines the box (the off-image 999,999 is excluded).
    assert ann.bbox == [10.0, 20.0, 0.0, 0.0]


def test_yolo_roundtrip_boxes_match_manifest_uv() -> None:
    manifest = _pose_manifest()
    dataset = export_yolo(manifest)
    coco = export_coco(manifest)
    cat_name = {c.id: c.name for c in coco.categories}
    image = {img.id: img for img in coco.images}

    # index the exported YOLO lines back to (camera, frame).
    label_by_stem = {lbl.file_name: lbl for lbl in dataset.labels}
    checked = 0
    for ann in coco.annotations:
        img = image[ann.image_id]
        stem = f"cam{img.camera_id}/frame_{img.frame:06d}.txt"
        label = label_by_stem[stem]
        # find the line for this class.
        cls = dataset.names.index(cat_name[ann.category_id])
        line = next(ln for ln in label.lines if int(ln.split()[0]) == cls)
        parts = line.split()
        cx, cy, nw, nh = (float(p) for p in parts[1:5])
        # denormalise back to a pixel [x, y, w, h] box.
        w = nw * label.width
        h = nh * label.height
        x = cx * label.width - w / 2.0
        y = cy * label.height - h / 2.0
        assert [x, y, w, h] == pytest.approx(ann.bbox, abs=1e-3)
        checked += 1
    assert checked == len(coco.annotations)


def test_multi_entity_scene_yields_one_category_per_entity() -> None:
    manifest = build_manifest(build_multi_entity_scene())
    dataset = export_coco(manifest)
    assert {c.name for c in dataset.categories} == {e.id for e in manifest.entities}
    # every annotation's category id is a declared category.
    valid = {c.id for c in dataset.categories}
    assert all(ann.category_id in valid for ann in dataset.annotations)


def test_write_coco_emits_parseable_json(tmp_path: Path) -> None:
    manifest = _pose_manifest()
    path = tmp_path / "annotations.json"
    dataset = write_coco(manifest, path)
    on_disk = json.loads(path.read_text())
    assert {"images", "annotations", "categories"} <= on_disk.keys()
    assert len(on_disk["images"]) == len(dataset.images)
    # round-trips back into the typed model.
    assert CocoDataset.model_validate(on_disk) == dataset


def test_write_yolo_writes_labels_and_classes(tmp_path: Path) -> None:
    manifest = _pose_manifest()
    out = tmp_path / "labels"
    dataset = write_yolo(manifest, out)
    assert (out / "classes.txt").read_text().split() == dataset.names
    # a label file exists for every image, one .txt per (camera, frame).
    for label in dataset.labels:
        path = out / label.file_name
        assert path.is_file()
        written = [ln for ln in path.read_text().splitlines() if ln]
        assert written == label.lines

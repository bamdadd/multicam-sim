"""COCO / YOLO 2D annotation exporter for a scene manifest.

The manifest already records, for every named point of every entity at every
frame, each camera's pixel projection (``uv``) plus the ``in_view`` / ``visible``
flags. That is exactly the ground truth a 2D detection + keypoint training set
needs, so this module turns a typed :class:`~multicam_sim.manifest.Manifest` into
COCO (JSON) and YOLO (per-image TXT) annotations — the sim doubles as a synthetic
training-data generator, not just a triangulation benchmark.

One *image* is emitted per ``(camera, frame)`` pair. For every entity seen on that
image (at least one point ``in_view``) an *annotation* is emitted:

* the **bounding box** is the axis-aligned pixel bounds of that entity's
  ``in_view`` points (points outside the image are excluded, so the box always
  lies within the camera frame);
* the **keypoints** follow the entity's point set with the COCO visibility flag
  derived from ``in_view`` / ``visible``:

  - ``2`` — ``in_view`` and ``visible`` (labelled, unoccluded),
  - ``1`` — ``in_view`` but not ``visible`` (labelled, occluded),
  - ``0`` — not ``in_view`` (absent; ``x = y = 0`` per the COCO convention).

Entity ids become the category names; the per-entity skeleton (``edges``) becomes
the COCO ``skeleton`` (1-indexed pairs into that category's keypoints).

The sim does not render pixels (pixels are not the contract), so ``file_name``
values are logical references (``cam{id}/frame_{frame:06d}.jpg``) a downstream
tool can align with rendered frames if it has them.

Everything is typed all the way through with pydantic models, mirroring
:mod:`multicam_sim.manifest`. Nothing here depends on an optional extra.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .manifest import EntityManifest, FrameObs, Manifest, PerCamObs

# COCO keypoint visibility flags.
_KP_ABSENT = 0  # not in view: not labelled / not in the image (x = y = 0)
_KP_OCCLUDED = 1  # in view but not visible: labelled, occluded
_KP_VISIBLE = 2  # in view and visible: labelled, unoccluded


def _image_file_name(camera_id: int, frame: int) -> str:
    """The logical image path for a ``(camera, frame)`` pair."""
    return f"cam{camera_id}/frame_{frame:06d}.jpg"


# --------------------------------------------------------------------------- #
# Typed COCO models. Field declaration order == emitted JSON key order.
# --------------------------------------------------------------------------- #


class CocoImage(BaseModel):
    """One COCO image: a single ``(camera, frame)`` view.

    ``camera_id`` / ``frame`` are non-standard extras that keep the link back to
    the manifest explicit (a plain COCO reader ignores unknown keys)."""

    id: int
    file_name: str
    width: int
    height: int
    camera_id: int
    frame: int


class CocoAnnotation(BaseModel):
    """One COCO annotation: an entity's bbox + keypoints on a single image."""

    id: int
    image_id: int
    category_id: int
    bbox: list[float]  # [x, y, width, height], pixels
    area: float
    iscrowd: int = 0
    keypoints: list[float]  # [x1, y1, v1, x2, y2, v2, ...]
    num_keypoints: int


class CocoCategory(BaseModel):
    """One COCO category: an entity id, its keypoint names and skeleton."""

    id: int
    name: str
    keypoints: list[str]
    skeleton: list[list[int]]  # 1-indexed pairs into ``keypoints``


class CocoDataset(BaseModel):
    """A full COCO keypoint-detection dataset built from a manifest."""

    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]

    def to_json(self) -> str:
        """Serialise to pretty JSON (2-space indent, strict/finite)."""
        return self.model_dump_json(indent=2)


# --------------------------------------------------------------------------- #
# Typed YOLO models.
# --------------------------------------------------------------------------- #


class YoloLabel(BaseModel):
    """One YOLO label file: the lines for a single ``(camera, frame)`` image.

    ``lines`` holds one entry per visible entity, each
    ``class cx cy w h [px py v]...`` with the bbox/keypoint coordinates
    normalised to ``[0, 1]`` by ``width`` / ``height`` (YOLO-pose layout). An
    image with no visible entity has an empty ``lines`` list (a valid background
    label)."""

    file_name: str  # cam{id}/frame_{frame:06d}.txt
    width: int
    height: int
    lines: list[str]


class YoloDataset(BaseModel):
    """A full YOLO dataset: per-image labels + the class-index -> entity-id names.

    Categories can carry different keypoint sets, so ``names`` is the shared class
    list and each label line uses its own class's keypoint order — group lines by
    class before feeding a fixed-``kpt_shape`` trainer."""

    labels: list[YoloLabel]
    names: list[str]  # class index -> entity id


# --------------------------------------------------------------------------- #
# Build (pure, from the typed manifest).
# --------------------------------------------------------------------------- #


def _keypoint_names(entity: EntityManifest) -> list[str]:
    """Stable keypoint order for an entity: point names in first-seen order."""
    names: list[str] = []
    seen: set[str] = set()
    for frame in entity.frames:
        for name in frame.points:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _skeleton(entity: EntityManifest, keypoints: list[str]) -> list[list[int]]:
    """1-indexed skeleton pairs into ``keypoints`` from the entity's edges."""
    if entity.edges is None:
        return []
    index = {name: i + 1 for i, name in enumerate(keypoints)}
    return [[index[a], index[b]] for a, b in entity.edges if a in index and b in index]


def _obs_for_camera(per_cam: list[PerCamObs], camera_id: int) -> PerCamObs | None:
    for obs in per_cam:
        if obs.cam == camera_id:
            return obs
    return None


def _visibility_flag(obs: PerCamObs | None) -> int:
    if obs is None or not obs.in_view:
        return _KP_ABSENT
    return _KP_VISIBLE if obs.visible else _KP_OCCLUDED


def _entity_keypoints_on_camera(
    frame: FrameObs, keypoint_names: list[str], camera_id: int
) -> tuple[list[float], int, list[tuple[float, float]]]:
    """The flat COCO keypoints for one entity on one camera at one frame.

    Returns ``(keypoints, num_labelled, in_view_uvs)`` where ``keypoints`` is the
    flat ``[x, y, v, ...]`` list in ``keypoint_names`` order, ``num_labelled`` is
    the count with ``v > 0``, and ``in_view_uvs`` are the pixel coordinates of the
    ``in_view`` points (the ones that define the bounding box)."""
    keypoints: list[float] = []
    num_labelled = 0
    in_view_uvs: list[tuple[float, float]] = []
    for name in keypoint_names:
        point = frame.points.get(name)
        obs = None if point is None else _obs_for_camera(point.per_cam, camera_id)
        flag = _visibility_flag(obs)
        if flag == _KP_ABSENT or obs is None:
            keypoints.extend((0.0, 0.0, float(_KP_ABSENT)))
            continue
        u, v = float(obs.uv[0]), float(obs.uv[1])
        keypoints.extend((u, v, float(flag)))
        num_labelled += 1
        in_view_uvs.append((u, v))
    return keypoints, num_labelled, in_view_uvs


def _bbox(uvs: list[tuple[float, float]]) -> list[float]:
    """Axis-aligned ``[x, y, w, h]`` pixel bounds of ``uvs`` (non-empty)."""
    us = [u for u, _ in uvs]
    vs = [v for _, v in uvs]
    x, y = min(us), min(vs)
    return [x, y, max(us) - x, max(vs) - y]


def export_coco(manifest: Manifest) -> CocoDataset:
    """Build a typed COCO keypoint-detection dataset from ``manifest``.

    One image per ``(camera, frame)``; one annotation per entity that has at
    least one ``in_view`` point on that image. See the module docstring for the
    bbox / keypoint / visibility conventions.
    """
    categories: list[CocoCategory] = []
    category_id: dict[str, int] = {}
    entity_keypoints: dict[str, list[str]] = {}
    for i, entity in enumerate(manifest.entities):
        names = _keypoint_names(entity)
        entity_keypoints[entity.id] = names
        category_id[entity.id] = i + 1
        categories.append(
            CocoCategory(
                id=i + 1,
                name=entity.id,
                keypoints=names,
                skeleton=_skeleton(entity, names),
            )
        )

    images: list[CocoImage] = []
    annotations: list[CocoAnnotation] = []
    image_id_of: dict[tuple[int, int], int] = {}
    next_image_id = 0
    next_ann_id = 0
    for camera in manifest.cameras:
        for frame in range(manifest.num_frames):
            image_id_of[(camera.id, frame)] = next_image_id
            images.append(
                CocoImage(
                    id=next_image_id,
                    file_name=_image_file_name(camera.id, frame),
                    width=camera.width,
                    height=camera.height,
                    camera_id=camera.id,
                    frame=frame,
                )
            )
            next_image_id += 1

    for entity in manifest.entities:
        names = entity_keypoints[entity.id]
        cat_id = category_id[entity.id]
        for frame_obs in entity.frames:
            for camera in manifest.cameras:
                keypoints, num_kp, in_view_uvs = _entity_keypoints_on_camera(
                    frame_obs, names, camera.id
                )
                if not in_view_uvs:
                    continue  # entity not seen on this camera this frame
                bbox = _bbox(in_view_uvs)
                annotations.append(
                    CocoAnnotation(
                        id=next_ann_id,
                        image_id=image_id_of[(camera.id, frame_obs.frame)],
                        category_id=cat_id,
                        bbox=bbox,
                        area=bbox[2] * bbox[3],
                        keypoints=keypoints,
                        num_keypoints=num_kp,
                    )
                )
                next_ann_id += 1

    return CocoDataset(images=images, annotations=annotations, categories=categories)


def _yolo_line(
    class_index: int,
    bbox: list[float],
    keypoints: list[float],
    width: int,
    height: int,
) -> str:
    """One normalised YOLO-pose line for a bbox + flat COCO keypoints."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    parts = [str(class_index), f"{cx:.6f}", f"{cy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
    for i in range(0, len(keypoints), 3):
        px = keypoints[i] / width
        py = keypoints[i + 1] / height
        flag = int(keypoints[i + 2])
        parts.extend((f"{px:.6f}", f"{py:.6f}", str(flag)))
    return " ".join(parts)


def export_yolo(manifest: Manifest) -> YoloDataset:
    """Build a typed YOLO(-pose) dataset from ``manifest``.

    Class indices are 0-based in entity order (``names[i]`` is the entity id);
    coordinates are normalised to ``[0, 1]`` by each image's width/height. See the
    module docstring for the bbox / keypoint / visibility conventions.
    """
    coco = export_coco(manifest)
    names = [entity.id for entity in manifest.entities]
    class_index = {name: i for i, name in enumerate(names)}
    category_name = {cat.id: cat.name for cat in coco.categories}
    image_by_id = {image.id: image for image in coco.images}

    lines_by_image: dict[int, list[str]] = {image.id: [] for image in coco.images}
    for ann in coco.annotations:
        image = image_by_id[ann.image_id]
        cls = class_index[category_name[ann.category_id]]
        lines_by_image[ann.image_id].append(
            _yolo_line(cls, ann.bbox, ann.keypoints, image.width, image.height)
        )

    labels: list[YoloLabel] = []
    for image in coco.images:
        stem = _image_file_name(image.camera_id, image.frame).rsplit(".", 1)[0]
        labels.append(
            YoloLabel(
                file_name=f"{stem}.txt",
                width=image.width,
                height=image.height,
                lines=lines_by_image[image.id],
            )
        )
    return YoloDataset(labels=labels, names=names)


# --------------------------------------------------------------------------- #
# Write (persist to disk).
# --------------------------------------------------------------------------- #


def write_coco(manifest: Manifest, path: str | Path) -> CocoDataset:
    """Build the COCO dataset for ``manifest`` and write it to ``path`` as JSON."""
    dataset = export_coco(manifest)
    Path(path).write_text(dataset.to_json())
    return dataset


def write_yolo(manifest: Manifest, out_dir: str | Path) -> YoloDataset:
    """Build the YOLO dataset for ``manifest`` and write it under ``out_dir``.

    Writes one ``.txt`` label per image (``cam{id}/frame_{frame:06d}.txt``,
    parent dirs created) plus a ``classes.txt`` mapping class index -> entity id.
    """
    dataset = export_yolo(manifest)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        path = root / label.file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(label.lines)
        path.write_text(text + "\n" if text else "")
    (root / "classes.txt").write_text("\n".join(dataset.names) + "\n")
    return dataset

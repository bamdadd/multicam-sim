"""Multi-entity smoke: two entities with distinct, stable ``entity.id``s and
per-entity per-camera visibility that diverges on at least one frame.

Exercises the serialized contract end to end: build the scene, write the
manifest, reload it, and assert every entity appears exactly once per frame
under a stable id, and that at least one entity is ``visible=False`` in a
camera on some frame while another entity stays ``visible=True`` in the same
camera on the same frame. This is what a multi-object tracker consumes: the
per-frame ground-truth track id and its per-view visibility mask.
"""

from __future__ import annotations

import json
from pathlib import Path

from multicam_sim import build_multi_entity_scene, write_manifest


def _load(tmp_path: Path) -> dict:
    scene = build_multi_entity_scene()
    path = tmp_path / "multi_entity.json"
    write_manifest(scene, path)
    return json.loads(path.read_text())


def test_two_entities_with_distinct_stable_ids(tmp_path: Path) -> None:
    """The manifest carries exactly the two entities under their declared ids,
    and each id appears once per frame across every frame of the take."""
    manifest = _load(tmp_path)
    num_frames = manifest["num_frames"]

    ids = [e["id"] for e in manifest["entities"]]
    assert ids == ["obj-a", "obj-b"]
    assert len(ids) == len(set(ids))  # ids are unique (stable track keys)

    for entity in manifest["entities"]:
        frame_numbers = [f["frame"] for f in entity["frames"]]
        assert frame_numbers == list(range(num_frames))
        assert len(frame_numbers) == len(set(frame_numbers))  # each frame once


def test_per_entity_per_camera_visibility_diverges(tmp_path: Path) -> None:
    """On at least one frame, one entity is ``visible=False`` in some camera
    while the other entity stays ``visible=True`` in that same camera — the
    per-track occlusion signal a multi-object tracker relies on."""
    manifest = _load(tmp_path)
    num_cameras = len(manifest["cameras"])
    entities = {e["id"]: e for e in manifest["entities"]}
    a_frames = {f["frame"]: f for f in entities["obj-a"]["frames"]}
    b_frames = {f["frame"]: f for f in entities["obj-b"]["frames"]}

    diverging = []
    for frame_number in a_frames:
        a_vis = [o["visible"] for o in a_frames[frame_number]["points"]["center"]["per_cam"]]
        b_vis = [o["visible"] for o in b_frames[frame_number]["points"]["center"]["per_cam"]]
        for cam_index in range(num_cameras):
            if a_vis[cam_index] != b_vis[cam_index]:
                diverging.append((frame_number, cam_index, a_vis[cam_index], b_vis[cam_index]))
    assert diverging, "expected at least one frame/camera where entity visibility diverges"

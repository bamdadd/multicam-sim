"""Hero smoke: an occluded-in-one-view point is recovered from the other two
views through the REAL multicam-occlusion DLT reader.

This exercises the serialized contract end to end: build the scene, write the
manifest to JSON, reload it, rebuild ``P = K [R | t]`` from the manifest's
K/R/t, and triangulate through ``multicam_occlusion.triangulate_dlt`` using the
manifest's ``visible`` flags as the mask. The tight 1e-6 tolerance is deliberate
— the projection is analytic, so any slop would betray a wrong R/t/convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# the REAL consumer reader (dev path dependency on ../multicam-occlusion)
from multicam_occlusion.triangulation import triangulate_dlt

from multicam_sim import build_smoke_scene, write_manifest


def _proj_mats(manifest: dict) -> np.ndarray:
    """Rebuild the (N, 3, 4) projection matrices from the manifest's K, R, t."""
    mats = []
    for cam in manifest["cameras"]:
        assert cam["convention"] == "opencv_rdf"
        K = np.array(cam["K"], dtype=np.float64)
        R = np.array(cam["R"], dtype=np.float64)
        t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
        mats.append(K @ np.hstack([R, t]))
    return np.stack(mats)


def _load_manifest(tmp_path: Path) -> dict:
    scene = build_smoke_scene()
    path = tmp_path / "smoke_manifest.json"
    write_manifest(scene, path)
    return json.loads(path.read_text())  # exercise the serialized contract


def _center_frames(manifest: dict) -> list[dict]:
    return manifest["entities"][0]["frames"]


def test_smoke_scene_shape(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path)
    assert len(manifest["cameras"]) == 3
    assert manifest["num_frames"] == len(_center_frames(manifest)) == 11
    # object = one entity with a single named point "center"
    for frame in _center_frames(manifest):
        assert set(frame["points"]) == {"center"}


def test_occluder_hides_camera_one_for_the_right_reason(tmp_path: Path) -> None:
    """At the occluded interval cam 1 is in-front AND in-frame but occluded, while
    cams 0 and 2 keep the point — so ``visible=False`` is caused by the sphere,
    not by the point leaving the frustum."""
    manifest = _load_manifest(tmp_path)
    Ps = _proj_mats(manifest)

    occluded_frames = []
    for frame in _center_frames(manifest):
        per_cam = frame["points"]["center"]["per_cam"]
        vis = [o["visible"] for o in per_cam]
        if not vis[1]:
            occluded_frames.append(frame["frame"])
            # cams 0 and 2 still see it
            assert vis[0] and vis[2]
            # cam 1 is occluded, not out-of-frame: it still projects in front
            # and inside the image, and reports a positive occlusion fraction.
            gt = np.array(frame["points"]["center"]["xyz_gt"])
            homog = np.append(gt, 1.0)
            x = Ps[1] @ homog
            assert x[2] > 0.0  # in front of camera 1
            u, v = x[0] / x[2], x[1] / x[2]
            assert 0 <= u < manifest["cameras"][1]["width"]
            assert 0 <= v < manifest["cameras"][1]["height"]
            assert per_cam[1]["occ_frac"] > 0.0

    # a genuine middle interval, not every frame and not none
    assert occluded_frames, "expected some frames where camera 1 is occluded"
    assert 0 < len(occluded_frames) < manifest["num_frames"]


def test_recover_occluded_point_from_two_views(tmp_path: Path) -> None:
    """The contract: a point occluded in cam 1 is recovered to 1e-6 from the two
    remaining views, through the real DLT, using the manifest's visible mask."""
    manifest = _load_manifest(tmp_path)
    Ps = _proj_mats(manifest)

    tested_any = False
    for frame in _center_frames(manifest):
        per_cam = frame["points"]["center"]["per_cam"]
        mask = np.array([o["visible"] for o in per_cam], dtype=bool)
        if mask[1]:
            continue  # only the frames where cam 1 is masked out
        tested_any = True
        assert mask.sum() == 2  # exactly the other two views survive

        uvs = np.array([o["uv"] for o in per_cam], dtype=np.float64)
        gt = np.array(frame["points"]["center"]["xyz_gt"], dtype=np.float64)

        recovered = triangulate_dlt(Ps, uvs, mask=mask)
        assert np.allclose(recovered, gt, atol=1e-6, rtol=0.0), (
            f"frame {frame['frame']}: recovered {recovered} != gt {gt}"
        )

    assert tested_any, "expected at least one cam-1-occluded frame to triangulate"


def test_all_three_views_recover_ground_truth(tmp_path: Path) -> None:
    """Convention self-check: with no mask, all three views recover GT to ~eps.
    Separates a wrong-convention failure from a wrong-mask failure."""
    manifest = _load_manifest(tmp_path)
    Ps = _proj_mats(manifest)
    for frame in _center_frames(manifest):
        per_cam = frame["points"]["center"]["per_cam"]
        uvs = np.array([o["uv"] for o in per_cam], dtype=np.float64)
        gt = np.array(frame["points"]["center"]["xyz_gt"], dtype=np.float64)
        recovered = triangulate_dlt(Ps, uvs)
        assert np.allclose(recovered, gt, atol=1e-6, rtol=0.0)


def test_single_view_cannot_triangulate(tmp_path: Path) -> None:
    """A lone visible view constrains the point only to a ray — the reader must
    refuse it. Guards the 'need >= 2 views' half of the contract."""
    manifest = _load_manifest(tmp_path)
    Ps = _proj_mats(manifest)
    frame = _center_frames(manifest)[0]
    per_cam = frame["points"]["center"]["per_cam"]
    uvs = np.array([o["uv"] for o in per_cam], dtype=np.float64)
    lone = np.array([True, False, False])
    with pytest.raises(ValueError):
        triangulate_dlt(Ps, uvs, mask=lone)

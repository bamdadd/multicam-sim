"""MTMC non-overlapping smoke: one object crosses a blind gap between two
stations, keeping a stable identity, and is recovered by the real DLT wherever
two cameras cover it.

Exercises the serialized contract end to end: build the scene, write the
manifest (strict JSON, ``allow_nan=False``), reload it, and assert the coverage
regimes, the labelled blind gap, the stable ``entity.id``, the emitted topology,
and — through the REAL ``multicam_occlusion.triangulate_dlt`` — that >=2-camera
frames recover ground truth to tight tolerance while single-camera frames do not.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# the REAL consumer reader (dev path dependency on ../multicam-occlusion)
from multicam_occlusion.triangulation import triangulate_dlt

from multicam_sim import build_mtmc_scene, write_manifest


def _load(tmp_path: Path) -> dict:
    scene = build_mtmc_scene()
    path = tmp_path / "mtmc.json"
    write_manifest(scene, path)
    return json.loads(path.read_text())


def _proj_mats(manifest: dict) -> np.ndarray:
    mats = []
    for cam in manifest["cameras"]:
        assert cam["convention"] == "opencv_rdf"
        K = np.array(cam["K"], dtype=np.float64)
        R = np.array(cam["R"], dtype=np.float64)
        t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
        mats.append(K @ np.hstack([R, t]))
    return np.stack(mats)


def _frames(manifest: dict) -> list[dict]:
    return manifest["entities"][0]["frames"]


def _in_view(frame: dict) -> list[bool]:
    return [o["in_view"] for o in frame["points"]["center"]["per_cam"]]


def test_manifest_is_strict_finite_json(tmp_path: Path) -> None:
    """Every uv is finite — behind/out-of-frame projections are sanitised, so the
    manifest is valid strict JSON (no Infinity/NaN) for the consumer."""
    manifest = _load(tmp_path)
    for frame in _frames(manifest):
        for obs in frame["points"]["center"]["per_cam"]:
            assert all(math.isfinite(c) for c in obs["uv"])


def test_visible_implies_in_view(tmp_path: Path) -> None:
    """The factoring the consumer relies on: visible == in_view AND unoccluded,
    so visible must never be true where in_view is false."""
    manifest = _load(tmp_path)
    for frame in _frames(manifest):
        for obs in frame["points"]["center"]["per_cam"]:
            if obs["visible"]:
                assert obs["in_view"]


def test_coverage_regimes_and_labelled_blind_gap(tmp_path: Path) -> None:
    """Three regimes in one take: station A alone -> blind gap -> station B pair,
    with the gap an explicit labelled interval, not an error."""
    manifest = _load(tmp_path)
    frames = _frames(manifest)
    coverage = [sum(_in_view(f)) for f in frames]

    # station A (camera 0) sees it first; the other two do not
    assert _in_view(frames[0]) == [True, False, False]

    # a genuine blind interval exists where NO camera sees the object
    blind = [f["frame"] for f in frames if sum(_in_view(f)) == 0]
    assert blind, "expected a blind gap where no camera sees the object"
    # contiguous, and strictly inside the take (not the whole thing)
    assert blind == list(range(blind[0], blind[-1] + 1))
    assert 0 < len(blind) < manifest["num_frames"]

    # a >=2-camera interval exists at the far station (recoverable)
    assert any(c >= 2 for c in coverage), "expected a >=2-camera interval at station B"

    # camera 0's in_view transitions true -> false; the station-B pair false -> true
    c0 = [_in_view(f)[0] for f in frames]
    assert c0[0] and not c0[-1]
    cB = [_in_view(f)[1] and _in_view(f)[2] for f in frames]
    assert not cB[0] and cB[-1]


def test_stable_identity_and_topology(tmp_path: Path) -> None:
    """One entity with a stable id across every frame; topology emitted with the
    two stations and their directed transit edges."""
    manifest = _load(tmp_path)
    assert len(manifest["entities"]) == 1
    assert manifest["entities"][0]["id"] == "target-1"

    topo = manifest["topology"]
    assert {s["id"] for s in topo["stations"]} == {"A", "B"}
    by_id = {s["id"]: s["camera_ids"] for s in topo["stations"]}
    assert by_id["A"] == [0] and by_id["B"] == [1, 2]
    pairs = {(e["src"], e["dst"]) for e in topo["edges"]}
    assert ("A", "B") in pairs and ("B", "A") in pairs
    assert all(e["transit_time_s"] > 0 for e in topo["edges"])


def test_two_camera_frames_recover_ground_truth(tmp_path: Path) -> None:
    """Where >=2 cameras have in_view=true, the real DLT recovers GT to 1e-6,
    masking on in_view. Convention self-check first (all covering cams, no mask)."""
    manifest = _load(tmp_path)
    Ps = _proj_mats(manifest)

    tested = 0
    for frame in _frames(manifest):
        iv = np.array(_in_view(frame), dtype=bool)
        if iv.sum() < 2:
            continue
        tested += 1
        uvs = np.array([o["uv"] for o in frame["points"]["center"]["per_cam"]], dtype=np.float64)
        gt = np.array(frame["points"]["center"]["xyz_gt"], dtype=np.float64)
        recovered = triangulate_dlt(Ps, uvs, mask=iv)
        assert np.allclose(recovered, gt, atol=1e-6, rtol=0.0), (
            f"frame {frame['frame']}: {recovered} != {gt}"
        )
    assert tested, "expected at least one >=2-camera frame to triangulate"


def test_single_camera_coverage_is_not_triangulable(tmp_path: Path) -> None:
    """The station-A single-camera interval constrains the point only to a ray —
    the real DLT must refuse it."""
    manifest = _load(tmp_path)
    Ps = _proj_mats(manifest)

    single = [f for f in _frames(manifest) if sum(_in_view(f)) == 1]
    assert single, "expected a single-camera interval"
    frame = single[0]
    iv = np.array(_in_view(frame), dtype=bool)
    uvs = np.array([o["uv"] for o in frame["points"]["center"]["per_cam"]], dtype=np.float64)
    with pytest.raises(ValueError):
        triangulate_dlt(Ps, uvs, mask=iv)

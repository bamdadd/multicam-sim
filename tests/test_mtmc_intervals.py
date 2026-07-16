"""Synthetic tests for ``entity_camera_intervals``.

The helper derives per-camera ``[enter, leave]`` intervals from the
``in_view`` transitions recorded in a manifest. These tests use hand-built
manifests with no projection, no renderer, and no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from multicam_sim.mtmc import entity_camera_intervals


def _manifest(camera_seqs: dict[int, list[bool]]) -> dict[str, Any]:
    """Build a minimal manifest with one entity and one point per frame.

    ``camera_seqs`` maps camera ids to the per-frame ``in_view`` sequence for
    camera 0's single point ``"center"``.
    """
    camera_ids = sorted(camera_seqs)
    num_frames = len(next(iter(camera_seqs.values())))
    assert all(len(seq) == num_frames for seq in camera_seqs.values())

    cameras = [
        {"id": cam_id, "K": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
        for cam_id in camera_ids
    ]

    frames: list[dict[str, Any]] = []
    for frame_idx in range(num_frames):
        per_cam = [
            {
                "cam": cam_id,
                "uv": [0.0, 0.0],
                "in_view": camera_seqs[cam_id][frame_idx],
                "visible": camera_seqs[cam_id][frame_idx],
                "occ_frac": 0.0,
            }
            for cam_id in camera_ids
        ]
        frames.append(
            {
                "frame": frame_idx,
                "points": {"center": {"xyz_gt": [0.0, 0.0, 0.0], "per_cam": per_cam}},
            }
        )

    return {
        "cameras": cameras,
        "fps": 30.0,
        "num_frames": num_frames,
        "entities": [{"id": "obj", "frames": frames}],
    }


def test_fully_visible() -> None:
    manifest = _manifest({0: [True, True, True, True]})
    assert entity_camera_intervals(manifest, "obj") == {0: [(0, 3)]}


def test_enters_mid_sequence() -> None:
    """The acceptance-criteria example: F, F, T, T, T, F -> one interval."""
    manifest = _manifest({0: [False, False, True, True, True, False]})
    assert entity_camera_intervals(manifest, "obj") == {0: [(2, 4)]}


def test_leaves_mid_sequence() -> None:
    manifest = _manifest({0: [True, True, True, False, False, False]})
    assert entity_camera_intervals(manifest, "obj") == {0: [(0, 2)]}


def test_two_disjoint_spans() -> None:
    manifest = _manifest({0: [True, True, False, False, True, True]})
    assert entity_camera_intervals(manifest, "obj") == {0: [(0, 1), (4, 5)]}


def test_camera_never_sees_entity() -> None:
    manifest = _manifest({0: [True, True, True], 1: [False, False, False]})
    assert entity_camera_intervals(manifest, "obj") == {0: [(0, 2)], 1: []}


def test_unknown_entity_raises() -> None:
    manifest = _manifest({0: [True, True]})
    with pytest.raises(ValueError, match="entity 'missing' not found in manifest"):
        entity_camera_intervals(manifest, "missing")

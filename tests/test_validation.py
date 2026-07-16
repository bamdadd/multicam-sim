"""Manifest schema validator tests.

Every rule in DESIGN.md's "Manifest schema — the JSON contract" section gets a
failing case, and every manifest produced by ``build_manifest`` passes clean.
"""

from __future__ import annotations

from typing import Any

import pytest

from multicam_sim import build_manifest, build_mtmc_scene, build_smoke_scene, validate_manifest


def _smoke_manifest() -> dict[str, Any]:
    return build_manifest(build_smoke_scene())


def _mtmc_manifest() -> dict[str, Any]:
    return build_manifest(build_mtmc_scene())


def test_smoke_manifest_validates() -> None:
    validate_manifest(_smoke_manifest())


def test_mtmc_manifest_validates() -> None:
    validate_manifest(_mtmc_manifest())


def test_missing_camera_field() -> None:
    manifest = _smoke_manifest()
    del manifest["cameras"][0]["convention"]
    with pytest.raises(ValueError, match="cameras\\.0\\.convention"):
        validate_manifest(manifest)


def test_wrong_K_shape() -> None:
    manifest = _smoke_manifest()
    manifest["cameras"][0]["K"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="camera 0.*K must be a 3x3 matrix"):
        validate_manifest(manifest)


def test_wrong_R_shape() -> None:
    manifest = _smoke_manifest()
    manifest["cameras"][0]["R"] = [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="camera 0.*R must be a 3x3 matrix"):
        validate_manifest(manifest)


def test_wrong_t_length() -> None:
    manifest = _smoke_manifest()
    manifest["cameras"][0]["t"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="camera 0.*t must be a length-3 vector"):
        validate_manifest(manifest)


def test_wrong_convention() -> None:
    manifest = _smoke_manifest()
    manifest["cameras"][0]["convention"] = "opencv_lh"
    with pytest.raises(ValueError, match="camera 0.*convention must be 'opencv_rdf'"):
        validate_manifest(manifest)


def test_missing_xyz_gt() -> None:
    manifest = _smoke_manifest()
    del manifest["entities"][0]["frames"][0]["points"]["center"]["xyz_gt"]
    with pytest.raises(ValueError, match="entities\\.0\\.frames\\.0\\.points\\.center\\.xyz_gt"):
        validate_manifest(manifest)


def test_wrong_xyz_gt_length() -> None:
    manifest = _smoke_manifest()
    manifest["entities"][0]["frames"][0]["points"]["center"]["xyz_gt"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="xyz_gt must be a length-3 vector"):
        validate_manifest(manifest)


def test_missing_per_cam() -> None:
    manifest = _smoke_manifest()
    del manifest["entities"][0]["frames"][0]["points"]["center"]["per_cam"]
    with pytest.raises(ValueError, match="entities\\.0\\.frames\\.0\\.points\\.center\\.per_cam"):
        validate_manifest(manifest)


def test_wrong_uv_length() -> None:
    manifest = _smoke_manifest()
    manifest["entities"][0]["frames"][0]["points"]["center"]["per_cam"][0]["uv"] = [0.0]
    with pytest.raises(ValueError, match="uv must be a length-2 vector"):
        validate_manifest(manifest)


def test_bad_visible_type() -> None:
    manifest = _smoke_manifest()
    manifest["entities"][0]["frames"][0]["points"]["center"]["per_cam"][0]["visible"] = "yes"
    with pytest.raises(ValueError, match="visible"):
        validate_manifest(manifest)


def test_occ_frac_out_of_range() -> None:
    manifest = _smoke_manifest()
    manifest["entities"][0]["frames"][0]["points"]["center"]["per_cam"][0]["occ_frac"] = 1.5
    with pytest.raises(ValueError, match="occ_frac must be in \\[0, 1\\]"):
        validate_manifest(manifest)


def test_visible_implies_in_view() -> None:
    manifest = _smoke_manifest()
    observation = manifest["entities"][0]["frames"][0]["points"]["center"]["per_cam"][0]
    observation["in_view"] = False
    observation["visible"] = True
    with pytest.raises(ValueError, match="visible implies in_view"):
        validate_manifest(manifest)


def test_edge_references_unknown_point() -> None:
    manifest = _smoke_manifest()
    manifest["entities"][0]["edges"] = [["center", "nose"]]
    with pytest.raises(ValueError, match="edge\\[0\\] references unknown point 'nose'"):
        validate_manifest(manifest)


def test_edge_must_be_pair() -> None:
    manifest = _smoke_manifest()
    manifest["entities"][0]["edges"] = [["center"]]
    with pytest.raises(ValueError, match="edge\\[0\\] must be a pair of point names"):
        validate_manifest(manifest)


def test_missing_topology_station_field() -> None:
    manifest = _mtmc_manifest()
    del manifest["topology"]["stations"][0]["camera_ids"]
    with pytest.raises(ValueError, match="topology\\.stations\\.0\\.camera_ids"):
        validate_manifest(manifest)


def test_topology_empty_station_camera_ids() -> None:
    manifest = _mtmc_manifest()
    manifest["topology"]["stations"][0]["camera_ids"] = []
    with pytest.raises(ValueError, match="camera_ids must not be empty"):
        validate_manifest(manifest)


def test_topology_duplicate_station_ids() -> None:
    manifest = _mtmc_manifest()
    manifest["topology"]["stations"][1]["id"] = "A"
    with pytest.raises(ValueError, match="station ids must be unique"):
        validate_manifest(manifest)


def test_topology_edge_references_unknown_station() -> None:
    manifest = _mtmc_manifest()
    manifest["topology"]["edges"][0]["src"] = "C"
    with pytest.raises(ValueError, match="src 'C' references an unknown station"):
        validate_manifest(manifest)


def test_topology_transit_time_not_positive() -> None:
    manifest = _mtmc_manifest()
    manifest["topology"]["edges"][0]["transit_time_s"] = 0.0
    with pytest.raises(ValueError, match="transit_time_s must be > 0"):
        validate_manifest(manifest)


def test_error_message_names_offending_path() -> None:
    """The promise: corruption raises with the field path in the message."""
    manifest = _smoke_manifest()
    manifest["cameras"][2]["K"] = [[1.0]]
    with pytest.raises(ValueError) as exc_info:
        validate_manifest(manifest)
    assert "cameras.2.K" in str(exc_info.value)

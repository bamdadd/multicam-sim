"""Ground-truth coverage metrics for overlapping and complementary scenes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from multicam_sim import build_handoff_ltr_scene, build_manifest, compute_coverage_metrics
from multicam_sim.handoff_ltr import CAM_WIDE


def _assembly_example() -> ModuleType:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "assembly_station.py"
    spec = importlib.util.spec_from_file_location("assembly_station", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coverage_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "coverage_metrics.py"
    spec = importlib.util.spec_from_file_location("coverage_metrics", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_scene_reports_overlap_without_blind_gaps() -> None:
    report = compute_coverage_metrics(build_manifest(build_handoff_ltr_scene()))
    by_camera = {row.camera_id: row for row in report.per_camera}

    assert by_camera[CAM_WIDE].fraction == 1.0
    assert report.overlap_count > 0
    assert report.blind_gap_count == 0
    assert report.handoff_points


def test_assembly_scene_reports_complementary_camera_coverage() -> None:
    example = _assembly_example()
    report = compute_coverage_metrics(build_manifest(example.build_scene()))
    by_camera = {row.camera_id: row for row in report.per_camera}

    assert by_camera[0].fraction == pytest.approx(0.25)
    assert by_camera[1].fraction == pytest.approx(0.75)
    assert report.overlap_count == 0
    assert report.blind_gap_count == 0


def test_report_dict_is_json_ready_and_includes_frame_identity() -> None:
    report = compute_coverage_metrics(build_manifest(build_handoff_ltr_scene()))
    payload = report.to_dict()

    assert payload["overlap_count"] == len(payload["overlap_frames"])
    assert payload["handoff_points"][0]["entity_id"] == "parcel-1"
    assert isinstance(payload["handoff_points"][0]["entered_cameras"], tuple)


def test_panel_matrix_uses_one_camera_count_per_entity_frame() -> None:
    example = _assembly_example()
    manifest = build_manifest(example.build_scene())
    script = _coverage_script()

    frames, entity_ids, matrix = script._coverage_matrix(manifest)

    assert frames == list(range(manifest.num_frames))
    assert entity_ids == [entity.id for entity in manifest.entities]
    assert len(matrix) == len(manifest.entities)
    assert all(len(row) == manifest.num_frames for row in matrix)
    assert all(value == 1 for row in matrix for value in row)


def test_panel_renders_headless_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    manifest = build_manifest(build_handoff_ltr_scene())
    report = compute_coverage_metrics(manifest)
    script = _coverage_script()
    output = tmp_path / "nested" / "coverage.png"

    rendered = script.render_coverage_panel(manifest, report, output, title="handoff_ltr")

    assert rendered == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000

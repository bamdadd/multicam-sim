"""Parcel-sort overlapping-coverage scene.

One handler walks a line of five sorting stations, dwelling at each, watched by
five cameras with deliberately OVERLAPPING fields of view. Exercises the
serialized contract end to end: build the scene, write the manifest (strict JSON,
``allow_nan=False``), reload it, and assert determinism, the down-the-line
topology, the per-station dwell, and — the point of the scene — that the handler
is ``in_view`` on at least two cameras at once during station-to-station
transits. Overlap is read straight off the manifest's per-camera ``in_view``
flags (the sim's own projection/visibility), never recomputed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from multicam_sim import build_manifest, build_parcel_sort_scene, write_manifest
from multicam_sim.parcel_sort import HANDLER_ID, STATION_IDS


def _load(tmp_path: Path) -> dict:
    scene = build_parcel_sort_scene()
    path = tmp_path / "parcel_sort.json"
    write_manifest(scene, path)
    return json.loads(path.read_text())


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


def test_deterministic_byte_stable_manifest() -> None:
    """The builder is deterministic: two independent builds serialise to the exact
    same JSON bytes (no unseeded RNG anywhere in the scene)."""
    first = build_manifest(build_parcel_sort_scene()).to_json()
    second = build_manifest(build_parcel_sort_scene()).to_json()
    assert first == second


def test_visible_implies_in_view(tmp_path: Path) -> None:
    """visible == in_view AND unoccluded, so visible is never true where in_view
    is false."""
    manifest = _load(tmp_path)
    for frame in _frames(manifest):
        for obs in frame["points"]["center"]["per_cam"]:
            if obs["visible"]:
                assert obs["in_view"]


def test_handler_in_view_on_two_cameras_simultaneously(tmp_path: Path) -> None:
    """The overlap proof: on some frame(s) the handler is in_view on AT LEAST
    two cameras at once. Those frames are the station-to-station transits, and the
    two covering cameras are always an ADJACENT pair (graduated overlap down the
    line, not every camera seeing everything)."""
    manifest = _load(tmp_path)
    frames = _frames(manifest)
    coverage = [sum(_in_view(f)) for f in frames]

    two_cam = [i for i, c in enumerate(coverage) if c >= 2]
    assert two_cam, "expected >=2-camera overlap frames during transits"

    # Every overlap frame is covered by exactly an adjacent pair of cameras.
    for i in two_cam:
        seen = [cam for cam, iv in enumerate(_in_view(frames[i])) if iv]
        assert len(seen) == 2, f"frame {i}: expected an adjacent pair, saw {seen}"
        assert seen[1] - seen[0] == 1, f"frame {i}: cameras {seen} are not adjacent"

    # It is genuine graduated overlap, not universal: no frame is seen by all cams.
    assert max(coverage) < len(manifest["cameras"])
    # The handler is always seen by at least one camera (no blind gap here).
    assert min(coverage) >= 1


def test_handler_dwells_at_each_station(tmp_path: Path) -> None:
    """The dwell (the sort operation): the handler holds a constant position across
    consecutive frames, once per sorting-station x."""
    manifest = _load(tmp_path)
    xs = [round(f["points"]["center"]["xyz_gt"][0], 6) for f in _frames(manifest)]

    held = [a for a, b in zip(xs[:-1], xs[1:], strict=True) if a == b]
    assert held, "expected held (dwell) frames where the handler pauses"

    # A dwell happens at each of the five sorting-station x positions.
    dwell_xs = {x for x, nxt in zip(xs[:-1], xs[1:], strict=True) if x == nxt}
    assert len(dwell_xs) == len(STATION_IDS)


def test_stable_identity_and_line_topology(tmp_path: Path) -> None:
    """One handler with a stable id across every frame; topology emitted with a
    station per sorting station (its covering camera) and directed transit edges
    chaining the line both ways."""
    manifest = _load(tmp_path)
    assert len(manifest["entities"]) == 1
    assert manifest["entities"][0]["id"] == HANDLER_ID

    topo = manifest["topology"]
    assert [s["id"] for s in topo["stations"]] == STATION_IDS
    by_id = {s["id"]: s["camera_ids"] for s in topo["stations"]}
    for i, station_id in enumerate(STATION_IDS):
        assert by_id[station_id] == [i]

    pairs = {(e["src"], e["dst"]) for e in topo["edges"]}
    for a, b in zip(STATION_IDS[:-1], STATION_IDS[1:], strict=True):
        assert (a, b) in pairs and (b, a) in pairs
    assert all(e["transit_time_s"] > 0 for e in topo["edges"])

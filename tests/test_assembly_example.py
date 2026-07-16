"""The assembly-station example runs, emits valid sidecars, and its GT holds:
complementary per-entity in_view + order status. CPU-only, no GL.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "assembly_station.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assembly_station", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return _load_example()


def test_example_emits_valid_sidecars(example: ModuleType, tmp_path: Path) -> None:
    summary = example.run(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    order_path = tmp_path / "order.json"
    assert manifest_path.exists() and order_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["cameras"]) == 2
    assert manifest["num_frames"] == 11
    ids = {e["id"] for e in manifest["entities"]}
    assert ids == {"operator", "part_a", "part_b", "part_c"}

    order = json.loads(order_path.read_text())
    assert order["order_id"] == "ORD-1"
    assert summary["result"].status.value == "fulfilled"


def test_complementary_in_view(example: ModuleType, tmp_path: Path) -> None:
    """Operator only in the overview camera; items only in the worktop camera."""
    vis = example.run(tmp_path)["visibility"]

    op_ov, n = vis["operator"]["overview"]
    op_wt, _ = vis["operator"]["worktop"]
    assert op_ov == n  # operator in overview every frame
    assert op_wt == 0  # operator never in the worktop camera

    for item in ("part_a", "part_b", "part_c"):
        it_ov, _ = vis[item]["overview"]
        it_wt, total = vis[item]["worktop"]
        assert it_ov == 0  # item never in the overview camera
        assert it_wt == total  # item in the worktop camera every frame


def test_order_status_matches_placements(example: ModuleType) -> None:
    """The order GT is fulfilled: every expected part is placed exactly once."""
    from multicam_sim.order import verify_order

    order, placements = example.build_order()
    result = verify_order(order.bom, placements)
    assert result.status.value == "fulfilled"
    assert result.missing == {} and result.extra == {} and result.wrong == {}
    assert {p.item for p in placements} == {"part_a", "part_b", "part_c"}

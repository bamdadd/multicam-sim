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
    # order.json carries the synced action events (byte-additive to the sidecar)
    assert [a["action"] for a in order["actions"]] == ["place", "place", "place"]
    assert [a["frame"] for a in order["actions"]] == [2, 5, 8]


def test_action_events_synced_to_placements_and_operator_wrist(
    example: ModuleType, tmp_path: Path
) -> None:
    """One place-event per placement; frame == placed_at and hand_position == the
    operator's right_wrist at that frame."""
    summary = example.run(tmp_path)
    actions = summary["actions"]
    _, placements = example.build_order()
    assert len(actions) == len(placements)

    wrist_by_frame = {f.frame: f.joints["right_wrist"] for f in example.operator_pose().frames}
    by_item = {a.item_id: a for a in actions}
    for p in placements:
        ev = by_item[p.item]
        assert ev.action == "place"
        assert ev.frame == p.placed_at_frame
        assert ev.hand_joint == "right_wrist"
        assert ev.entity_id == "operator"
        assert list(ev.hand_position) == pytest.approx(wrist_by_frame[p.placed_at_frame])


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


# --- placement-synced preset (opt-in) --------------------------------------- #
# Everything below reads ONLY the emitted files (manifest.json /
# interactions.json on disk) — the genuine consumer path. The authored scenario
# constants are pasted as literals so the tests pin the contract, not the code.


def _strict_local_minima(series: list[float]) -> list[int]:
    """Frames whose value is strictly lower than both neighbours."""
    return [
        i
        for i in range(1, len(series) - 1)
        if series[i] < series[i - 1] and series[i] < series[i + 1]
    ]


def _joint_height_series(manifest: dict, entity_id: str, joint: str) -> list[float]:
    """A tracked joint's height per frame, read from the on-disk manifest only."""
    entity = next(e for e in manifest["entities"] if e["id"] == entity_id)
    return [fr["points"][joint]["xyz_gt"][2] for fr in entity["frames"]]


def _item_change_frames(manifest: dict, entity_id: str) -> list[int]:
    """Frames at which an item's center moved vs the previous frame (manifest only)."""
    entity = next(e for e in manifest["entities"] if e["id"] == entity_id)
    positions = [fr["points"]["center"]["xyz_gt"] for fr in entity["frames"]]
    return [f for f in range(1, len(positions)) if positions[f] != positions[f - 1]]


def test_synced_dips_recoverable_from_manifest_alone(example: ModuleType, tmp_path: Path) -> None:
    """Strict local minima of the wrist height == the synced dips + distractor.

    The placements land at frames 2/5/8 with δ=1, so dips sit at 1/4/7, plus
    the distractor dip at 10. A consumer with only manifest.json recovers them.
    """
    example.run(tmp_path, placement_synced=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    z = _joint_height_series(manifest, "operator", "right_wrist")
    dips = _strict_local_minima(z)
    assert dips == [1, 4, 7, 10]
    # each true dip is exactly δ=1 frame before its placement, strictly dipped
    for dip, placed in [(1, 2), (4, 5), (7, 8)]:
        assert placed - dip == 1
        assert z[dip] < z[dip - 1] and z[dip] < z[dip + 1]


def _naive_causal_forward_associate(
    dips: list[int], changes: list[tuple[str, int]], lag_window: int
) -> list[tuple[str, str, int, int]]:
    """The associator a consumer would write first: pair each dip with the next
    change inside the lag window. ``changes`` are ``(item_id, frame)``."""
    pairs = []
    for dip in sorted(dips):
        later = [(item, c) for item, c in changes if 0 < c - dip <= lag_window]
        if later:
            item, change = min(later, key=lambda ic: ic[1])
            pairs.append(("operator", item, dip, change))
    return pairs


def test_synced_negatives_falsify_naive_causal_association(
    example: ModuleType, tmp_path: Path
) -> None:
    """The negatives must make the naive causal-forward rule score below perfect.

    The distractor dip at 10 places nothing, but part_d's uncaused move at 11
    follows inside the lag window — the naive rule pairs them, and
    interactions.json says otherwise: exactly one false positive.
    """
    example.run(tmp_path, placement_synced=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    truth = json.loads((tmp_path / "interactions.json").read_text())
    lag_window = truth["timing"]["lag_window"]

    dips = _strict_local_minima(_joint_height_series(manifest, "operator", "right_wrist"))
    changes = [
        (item, c)
        for item in ("part_a", "part_b", "part_c", "part_d")
        for c in _item_change_frames(manifest, item)
    ]
    predicted = _naive_causal_forward_associate(dips, changes, lag_window)
    truth_pairs = {
        (p["actor_id"], p["item_id"], p["action_frame"], p["change_frame"]) for p in truth["pairs"]
    }

    tp = [p for p in predicted if p in truth_pairs]
    fp = [p for p in predicted if p not in truth_pairs]
    fn = [p for p in truth_pairs if p not in predicted]
    precision = len(tp) / len(predicted)
    recall = len(tp) / len(truth_pairs)

    # the distractor dip→part_d pairing is THE false positive, by name
    assert fp == [("operator", "part_d", 10, 11)]
    assert (len(tp), len(fp), len(fn)) == (3, 1, 0)
    assert precision == 3 / 4 < 1.0
    assert recall == 1.0

    # even a δ-informed rule trips on the confounder: a dip sits exactly δ=1
    # before part_d's move, and the ground truth still says it is no pair
    assert truth["timing"]["action_lag"] == 11 - 10
    assert all(p["item_id"] != "part_d" for p in truth["pairs"])
    # part_d itself is placed (order stays fulfilled) and fully worktop-visible
    assert _item_change_frames(manifest, "part_d") == [11]


def test_synced_interactions_sidecar_contents(example: ModuleType, tmp_path: Path) -> None:
    """The sidecar carries the timing contract and only the true pairs."""
    summary = example.run(tmp_path, placement_synced=True)
    truth = json.loads((tmp_path / "interactions.json").read_text())
    assert truth == {
        "timing": {"action_lag": 1, "lag_window": 2},
        "actor_id": "operator",
        "tracked_joint": "right_wrist",
        "pairs": [
            {"actor_id": "operator", "item_id": "part_a", "action_frame": 1, "change_frame": 2},
            {"actor_id": "operator", "item_id": "part_b", "action_frame": 4, "change_frame": 5},
            {"actor_id": "operator", "item_id": "part_c", "action_frame": 7, "change_frame": 8},
        ],
    }
    # the late distractor item is placed (order still fulfilled) but is NOT a pair
    assert summary["result"].status.value == "fulfilled"
    assert "part_d" in summary["result"].placed


def test_synced_preset_keeps_complementary_in_view(example: ModuleType, tmp_path: Path) -> None:
    """The distractor item joins the fusion story: worktop-only, every frame."""
    vis = example.run(tmp_path, placement_synced=True)["visibility"]
    d_ov, _ = vis["part_d"]["overview"]
    d_wt, total = vis["part_d"]["worktop"]
    assert d_ov == 0
    assert d_wt == total == 13


def test_default_run_has_no_interactions_sidecar(example: ModuleType, tmp_path: Path) -> None:
    """Off by default: no sidecar, no extra entities, no extra frames."""
    summary = example.run(tmp_path)
    assert not (tmp_path / "interactions.json").exists()
    assert summary["interactions"] is None
    assert summary["manifest"]["num_frames"] == 11
    assert {e["id"] for e in summary["manifest"]["entities"]} == {
        "operator",
        "part_a",
        "part_b",
        "part_c",
    }

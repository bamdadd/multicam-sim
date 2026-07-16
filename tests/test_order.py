"""Order verification: all four statuses + serialization. CPU-only, self-contained.

Neutral, synthetic item ids only (part_a/part_b/part_c/part_d); a foreign item
uses part_x. No real-domain nouns anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multicam_sim.order import (
    ActionEvent,
    BillOfMaterials,
    ItemPlacement,
    LineItem,
    Order,
    OrderResult,
    OrderStatus,
    verify_order,
    write_order_json,
)


def _bom() -> BillOfMaterials:
    # a generic pick-list: 1 part_a, 1 part_b, 2 part_c, 1 part_d
    return BillOfMaterials.from_counts({"part_a": 1, "part_b": 1, "part_c": 2, "part_d": 1})


def _place(items: list[str]) -> list[ItemPlacement]:
    return [
        ItemPlacement(item=name, placed_at_frame=i, entity_id="items")
        for i, name in enumerate(items)
    ]


def test_fulfilled_exact() -> None:
    result = verify_order(_bom(), _place(["part_a", "part_b", "part_c", "part_c", "part_d"]))
    assert result.status is OrderStatus.fulfilled
    assert result.missing == {} and result.extra == {} and result.wrong == {}
    assert result.expected == {"part_a": 1, "part_b": 1, "part_c": 2, "part_d": 1}
    assert result.placed == {"part_a": 1, "part_b": 1, "part_c": 2, "part_d": 1}


def test_missing_item() -> None:
    # one part_c short (and part_d not placed) -> missing
    result = verify_order(_bom(), _place(["part_a", "part_b", "part_c"]))
    assert result.status is OrderStatus.missing_item
    assert result.missing == {"part_c": 1, "part_d": 1}
    assert result.extra == {} and result.wrong == {}


def test_extra_item() -> None:
    # a third part_c beyond the two expected -> extra of an expected item
    result = verify_order(
        _bom(), _place(["part_a", "part_b", "part_c", "part_c", "part_c", "part_d"])
    )
    assert result.status is OrderStatus.extra_item
    assert result.extra == {"part_c": 1}
    assert result.missing == {} and result.wrong == {}


def test_wrong_item() -> None:
    # a foreign item the order never asked for -> wrong
    result = verify_order(
        _bom(), _place(["part_a", "part_b", "part_c", "part_c", "part_d", "part_x"])
    )
    assert result.status is OrderStatus.wrong_item
    assert result.wrong == {"part_x": 1}
    assert result.missing == {} and result.extra == {}


def test_status_precedence_wrong_beats_missing() -> None:
    # both a foreign item and a shortfall present -> wrong wins (most severe)
    result = verify_order(_bom(), _place(["part_a", "part_b", "part_c", "part_x"]))
    assert result.status is OrderStatus.wrong_item
    assert result.wrong == {"part_x": 1}
    assert result.missing == {"part_c": 1, "part_d": 1}


def test_deterministic_and_order_independent() -> None:
    a = verify_order(_bom(), _place(["part_c", "part_d", "part_a", "part_b", "part_c"]))
    b = verify_order(_bom(), _place(["part_a", "part_b", "part_c", "part_c", "part_d"]))
    assert a.model_dump() == b.model_dump()  # placement order irrelevant


def test_lineitem_and_bom_validation() -> None:
    with pytest.raises(ValueError):
        LineItem(name="part_a", count=0)
    with pytest.raises(ValueError):
        BillOfMaterials(items=[LineItem(name="part_c", count=1), LineItem(name="part_c", count=1)])


def test_serialises_to_order_json_sidecar(tmp_path: Path) -> None:
    order = Order(order_id="A-100", bom=_bom())
    result = verify_order(order.bom, _place(["part_a", "part_b", "part_c", "part_c", "part_d"]))

    order_path = tmp_path / "order.json"
    dumped = write_order_json(order, order_path)
    assert dumped["order_id"] == "A-100"
    # reload and reconstruct -> typed round-trip
    reloaded = Order.model_validate(json.loads(order_path.read_text()))
    assert reloaded == order

    # OrderResult is serialisable too (status as its string value)
    payload = json.loads(result.to_json())
    assert payload["status"] == "fulfilled"
    assert OrderResult.model_validate(payload) == result


def test_action_event_defaults_and_fields() -> None:
    ev = ActionEvent(frame=5, item_id="part_a", entity_id="operator", hand_position=(1.0, 2.0, 3.0))
    assert ev.action == "place"  # Literal default, room to widen
    assert ev.hand_joint == "right_wrist"
    assert ev.hand_position == (1.0, 2.0, 3.0)


def test_verify_order_carries_sorted_actions_and_order_id() -> None:
    bom = BillOfMaterials.from_counts({"part_a": 1, "part_b": 1})
    placements = _place(["part_a", "part_b"])
    actions = [
        ActionEvent(frame=5, item_id="part_b", entity_id="op", hand_position=(0.0, 0.0, 1.0)),
        ActionEvent(frame=2, item_id="part_a", entity_id="op", hand_position=(0.0, 0.0, 0.9)),
    ]
    result = verify_order(bom, placements, order_id="ORD-9", actions=actions)
    assert result.order_id == "ORD-9"
    # sorted by (frame, item_id) -> frame 2 first
    assert [a.frame for a in result.actions] == [2, 5]
    assert [a.item_id for a in result.actions] == ["part_a", "part_b"]


def test_actions_round_trip_through_sidecar_json() -> None:
    bom = BillOfMaterials.from_counts({"part_a": 1})
    ev = ActionEvent(frame=3, item_id="part_a", entity_id="op", hand_position=(1.5, 0.0, 0.95))
    result = verify_order(bom, _place(["part_a"]), order_id="ORD-1", actions=[ev])
    payload = json.loads(result.to_json())
    assert payload["actions"][0]["action"] == "place"
    assert payload["actions"][0]["hand_position"] == [1.5, 0.0, 0.95]
    assert OrderResult.model_validate(payload) == result


def test_actions_default_empty_is_backward_compatible() -> None:
    result = verify_order(BillOfMaterials.from_counts({"part_a": 1}), _place(["part_a"]))
    assert result.actions == [] and result.order_id is None

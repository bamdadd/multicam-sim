"""Order -> assembly -> verification ground truth for an assembly station.

Self-contained and generic: an **order** is a bill of materials (expected item
ids + counts) that an operator assembles into a container; **assembly** is a
sequence of :class:`ItemPlacement`s (an item appearing at a frame, tied to an
entity); **verification** compares the two and reports a deterministic
:class:`OrderResult`.

This module is pure typed models + logic — no cameras, no stations, no
``in_view``. It only knows abstract item *ids* (neutral: ``part_a``/``part_b``/
``part_c``), so it composes with any scene layer that emits placements.
Everything round-trips to an ``order.json`` sidecar via pydantic ``model_dump``.

Discrepancy vocabulary (kept precise so the four statuses are unambiguous):

* **missing** — an expected item placed *fewer* times than the order needs;
* **extra**   — an expected item placed *more* times than the order needs;
* **wrong**   — an item placed that the order never asked for (a foreign item).

The overall :class:`OrderStatus` is the most severe deviation present, in the
order ``wrong_item > missing_item > extra_item > fulfilled``.

**Action events (causal-fusion GT).** Beyond *what* was placed, a downstream
consumer (multicam-occlusion's causal fusion metric) scores the *timing*
association between an operator's action and the resulting assembly change. It
already gets per-item visibility from the manifest, but not the synced operator
action. :class:`ActionEvent` supplies exactly that: one ``place`` event per item
placement, time-synced to that item's ``placed_at`` frame and carrying the
operator's hand-joint world position at that frame. Events ride in the order GT
sidecar (:class:`OrderResult.actions`), so the byte-golden manifest is untouched.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class LineItem(BaseModel):
    """One expected item and how many of it the order needs."""

    model_config = ConfigDict(frozen=True)

    name: str
    count: int

    @field_validator("count")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LineItem count must be >= 1")
        return value


class BillOfMaterials(BaseModel):
    """The expected contents of an order: a list of :class:`LineItem`s.

    Item names must be unique (counts are aggregated per name, so a repeated name
    would be ambiguous).
    """

    model_config = ConfigDict(frozen=True)

    items: list[LineItem]

    @field_validator("items")
    @classmethod
    def _unique_names(cls, value: list[LineItem]) -> list[LineItem]:
        names = [i.name for i in value]
        if len(names) != len(set(names)):
            raise ValueError("duplicate item name in bill of materials")
        return value

    @classmethod
    def from_counts(cls, counts: dict[str, int]) -> BillOfMaterials:
        """Build from a plain ``{name: count}`` mapping."""
        return cls(items=[LineItem(name=n, count=c) for n, c in counts.items()])

    def counts(self) -> dict[str, int]:
        """Expected counts as a ``{name: count}`` mapping."""
        return {i.name: i.count for i in self.items}


class Order(BaseModel):
    """An identified order: an id plus its bill of materials."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    bom: BillOfMaterials


class ItemPlacement(BaseModel):
    """An assembly event: ``item`` placed at ``placed_at_frame`` by ``entity_id``."""

    model_config = ConfigDict(frozen=True)

    item: str
    placed_at_frame: int
    entity_id: str


class ActionEvent(BaseModel):
    """An operator action synced to an assembly change (causal-fusion GT).

    One ``place`` event per item placement, time-synced to that item's
    ``placed_at`` frame, carrying the operator's hand-joint world position at that
    frame. ``action`` is a ``Literal["place"]`` today with room to widen (e.g.
    ``pick`` / ``remove``) without a schema fork. Lives in the order GT sidecar,
    never in the byte-golden manifest.
    """

    model_config = ConfigDict(frozen=True)

    frame: int
    action: Literal["place"] = "place"
    item_id: str
    entity_id: str
    hand_joint: str = "right_wrist"
    hand_position: tuple[float, float, float]


class OrderStatus(StrEnum):
    """Outcome of verifying an assembly against an order."""

    fulfilled = "fulfilled"
    missing_item = "missing_item"
    extra_item = "extra_item"
    wrong_item = "wrong_item"


class OrderResult(BaseModel):
    """Deterministic verification result: overall status + per-item detail.

    ``missing``/``extra``/``wrong`` are ``{name: count}`` deltas (all sorted by
    name), so a consumer sees exactly which items are short, surplus, or foreign.
    ``actions`` (optional) are the placement-synced operator :class:`ActionEvent`s
    for causal fusion — additive to the order sidecar, sorted by ``(frame, item_id)``.
    """

    model_config = ConfigDict(frozen=True)

    status: OrderStatus
    expected: dict[str, int]
    placed: dict[str, int]
    missing: dict[str, int]
    extra: dict[str, int]
    wrong: dict[str, int]
    order_id: str | None = None
    actions: list[ActionEvent] = []

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string (the ``order.json`` sidecar payload)."""
        return self.model_dump_json(indent=indent)


def _sorted(counts: dict[str, int]) -> dict[str, int]:
    return {k: counts[k] for k in sorted(counts)}


def verify_order(
    bom: BillOfMaterials,
    placements: Sequence[ItemPlacement],
    *,
    order_id: str | None = None,
    actions: Sequence[ActionEvent] = (),
) -> OrderResult:
    """Compare placed items against the order; return a deterministic result.

    Independent of frame order and placement order — only aggregate per-item
    counts matter. Optional ``order_id`` and ``actions`` (placement-synced
    operator events) ride in the result; ``actions`` are sorted by
    ``(frame, item_id)`` so the sidecar is deterministic.
    """
    expected = bom.counts()
    placed = dict(Counter(p.item for p in placements))

    missing: dict[str, int] = {}
    extra: dict[str, int] = {}
    for name, want in expected.items():
        got = placed.get(name, 0)
        if got < want:
            missing[name] = want - got
        elif got > want:
            extra[name] = got - want
    wrong = {name: n for name, n in placed.items() if name not in expected}

    if wrong:
        status = OrderStatus.wrong_item
    elif missing:
        status = OrderStatus.missing_item
    elif extra:
        status = OrderStatus.extra_item
    else:
        status = OrderStatus.fulfilled

    return OrderResult(
        status=status,
        expected=_sorted(expected),
        placed=_sorted(placed),
        missing=_sorted(missing),
        extra=_sorted(extra),
        wrong=_sorted(wrong),
        order_id=order_id,
        actions=sorted(actions, key=lambda a: (a.frame, a.item_id)),
    )


def write_order_json(payload: BaseModel, path: str | Path) -> dict[str, Any]:
    """Write any order model (Order / OrderResult) to ``path`` as JSON.

    Returns the dumped dict so a caller can assert on it without re-reading.
    """
    data: dict[str, Any] = payload.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))
    return data

"""Runnable assembly-station example: heterogeneous multi-camera fusion + order GT.

Deterministic, CPU-only, no GL. Run it directly::

    python examples/assembly_station.py         # or: uv run python examples/assembly_station.py

Scenario (domain-neutral, synthetic):

* an **operator** (a COCO-17 skeleton) assembles an **order** at a station;
* three abstract **items** — ``part_a`` / ``part_b`` / ``part_c`` — are placed
  one-by-one into a **container** on a worktop over the frames;
* an **overview** camera (wide, high, north) frames the *operator*;
* a **worktop** camera (close, zoomed, east) frames the *items*.

The two cameras are aimed at spatially separated regions, so their per-entity
per-camera ``in_view`` flags are **complementary**: the operator is in view on
the overview camera and not the worktop camera, and vice-versa for the items.
This is the manifest's fusion story — different entities land in different
cameras' ``in_view`` with no schema change.

Emits two ground-truth sidecars next to this file (``--out`` to change dir):

* ``manifest.json`` — the full scene manifest (projection + in_view/visible);
* ``order.json``    — the verified order result (fulfilled / missing / …).

Opt-in **placement-synced preset** (``--placement-synced``): the continuous
wrist reach is replaced by discrete hand dips synced to the placements (a
strict local minimum of the tracked wrist's height at ``placed_at - δ`` per
placed item), plus a distractor dip that assembles nothing and a distractor
item (``part_d``) that moves outside the causal lag window. The true
``(actor, item, action_frame, change_frame)`` pairs are written to
``interactions.json`` so a causal-fusion consumer can score precision/recall.
Off by default: without the flag the scene and every emitted file are
byte-identical to before.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from multicam_sim import write_manifest
from multicam_sim.actions import (
    CausalTiming,
    DipSchedule,
    build_action_ground_truth,
    write_actions_json,
)
from multicam_sim.dsl.rig import CameraRig, StationView
from multicam_sim.entities import Entity, EntityFrame
from multicam_sim.order import (
    ActionEvent,
    BillOfMaterials,
    ItemPlacement,
    Order,
    OrderResult,
    verify_order,
    write_order_json,
)
from multicam_sim.pose import PoseFrame, PoseTrajectory, Skeleton
from multicam_sim.scene import Scene

FPS = 30.0
NUM_FRAMES = 11

# --- geometry (metres, Z-up) ------------------------------------------------ #
# Operator stands near the origin; the worktop + container sit to the east so the
# two cameras frame clearly separated regions (robust complementary in_view).
_OPERATOR_BASE = (0.0, 0.0)  # foot base (x, y)
_CONTAINER = (2.9, 0.0, 0.92)
_ITEM_STAGING = {
    "part_a": (2.75, -0.30, 0.90),
    "part_b": (2.90, -0.30, 0.90),
    "part_c": (3.05, -0.30, 0.90),
}
_PLACED_AT = {"part_a": 2, "part_b": 5, "part_c": 8}

# --- placement-synced preset (opt-in via --placement-synced) ---------------- #
# The causal half of the fusion story: the tracked hand dips (strict local
# height minimum) at ``placed_at - δ`` for each placed item; a distractor dip
# assembles nothing, and a distractor item moves outside the causal lag window.
# δ and the lag window are typed parameters (CausalTiming), not magic numbers.
_TRACKED_HAND = "right_wrist"
_SYNC_NUM_FRAMES = 13
_SYNC_TIMING = CausalTiming(action_lag=1, lag_window=2)
_DIP_DEPTH = 0.30
_DIP_HALF_WIDTH = 1
_DISTRACTOR_ITEM = "part_d"
_DISTRACTOR_STAGING = (2.60, -0.30, 0.90)
_DISTRACTOR_PLACED_AT = 10  # no dip within the lag window before it
_DISTRACTOR_DIP = 11  # assembles nothing: no placement within the window after it

# Standing COCO-17 offsets (dx, dy, dz) from the foot base; +y is the facing dir.
_JOINT_OFFSETS: dict[str, tuple[float, float, float]] = {
    "nose": (0.0, 0.10, 1.60),
    "left_eye": (0.03, 0.10, 1.64),
    "right_eye": (-0.03, 0.10, 1.64),
    "left_ear": (0.08, 0.05, 1.63),
    "right_ear": (-0.08, 0.05, 1.63),
    "left_shoulder": (0.20, 0.0, 1.45),
    "right_shoulder": (-0.20, 0.0, 1.45),
    "left_elbow": (0.26, 0.12, 1.20),
    "right_elbow": (-0.26, 0.12, 1.20),
    "left_wrist": (0.22, 0.28, 1.00),
    "right_wrist": (-0.22, 0.28, 1.00),
    "left_hip": (0.12, 0.0, 0.95),
    "right_hip": (-0.12, 0.0, 0.95),
    "left_knee": (0.12, 0.02, 0.52),
    "right_knee": (-0.12, 0.02, 0.52),
    "left_ankle": (0.10, 0.0, 0.10),
    "right_ankle": (-0.10, 0.0, 0.10),
}


def operator_pose(placement_synced: bool = False, num_frames: int = NUM_FRAMES) -> PoseTrajectory:
    """A standing COCO-17 operator whose wrists make a small assembling motion.

    Default: a single continuous sinusoidal wrist reach (order-verification
    scene). With ``placement_synced``: the wrists rest and the tracked hand
    (:data:`_TRACKED_HAND`) dips — a strict local height minimum at
    ``placed_at - δ`` for each placed item, plus one distractor dip that
    assembles nothing — so every dip is recoverable off the manifest alone.
    """
    bx, by = _OPERATOR_BASE
    frames: list[PoseFrame] = []
    for f in range(num_frames):
        phase = (
            0.0 if placement_synced else math.sin(2.0 * math.pi * f / (num_frames - 1))
        )  # -1..1, smooth
        joints: dict[str, list[float]] = {}
        for name, (dx, dy, dz) in _JOINT_OFFSETS.items():
            reach = 0.06 * phase if name.endswith("wrist") else 0.0  # wrists reach in +y
            joints[name] = [bx + dx, by + dy + reach, dz]
        frames.append(PoseFrame(frame=f, joints=joints))
    trajectory = PoseTrajectory(id="operator", skeleton=Skeleton.coco17(), frames=frames)
    if placement_synced:
        dips = DipSchedule(
            frames=[*synced_dip_frames(), _DISTRACTOR_DIP],
            rest_height=_JOINT_OFFSETS[_TRACKED_HAND][2],
            depth=_DIP_DEPTH,
            half_width=_DIP_HALF_WIDTH,
        )
        trajectory = dips.author(trajectory, _TRACKED_HAND)
    return trajectory


def synced_dip_frames() -> list[int]:
    """The dip frame (``placed_at - δ``) for each causally-backed placement."""
    return sorted(frame - _SYNC_TIMING.action_lag for frame in _PLACED_AT.values())


def item_entity(
    item_id: str,
    staging: tuple[float, float, float] | None = None,
    placed_at: int | None = None,
    num_frames: int = NUM_FRAMES,
) -> Entity:
    """An item that sits at its staging spot, then jumps into the container at its
    ``placed_at`` frame (and stays)."""
    staging = _ITEM_STAGING[item_id] if staging is None else staging
    placed_at = _PLACED_AT[item_id] if placed_at is None else placed_at
    frames = [
        EntityFrame(
            frame=f,
            points={"center": list(_CONTAINER if f >= placed_at else staging)},
        )
        for f in range(num_frames)
    ]
    return Entity(id=item_id, frames=frames)


def build_scene(placement_synced: bool = False) -> Scene:
    """Assemble the two-camera scene: overview (operator) + worktop (items)."""
    num_frames = _SYNC_NUM_FRAMES if placement_synced else NUM_FRAMES
    cameras = CameraRig.stations(
        [
            # overview: wide-ish, high, to the north (+y), framing the operator.
            StationView(position=(0.0, 4.2, 2.4), look_at=(0.0, 0.0, 1.2), fov_deg=40.0),
            # worktop: close + zoomed, to the east, framing the container/items.
            StationView(position=(2.9, -1.1, 1.7), look_at=(2.9, 0.0, 0.9), fov_deg=44.0),
        ],
        width=1280,
        height_px=720,
    )
    entities = [
        operator_pose(placement_synced, num_frames).to_entity(),
        *(item_entity(i, num_frames=num_frames) for i in _ITEM_STAGING),
    ]
    if placement_synced:
        entities.append(
            item_entity(
                _DISTRACTOR_ITEM,
                staging=_DISTRACTOR_STAGING,
                placed_at=_DISTRACTOR_PLACED_AT,
                num_frames=num_frames,
            )
        )
    return Scene(fps=FPS, num_frames=num_frames, cameras=cameras, entities=entities)


def build_order(placement_synced: bool = False) -> tuple[Order, list[ItemPlacement]]:
    """The pick-list (one of each part) and the placements as items land."""
    counts = {item: 1 for item in _ITEM_STAGING}
    placed_at = dict(_PLACED_AT)
    if placement_synced:
        counts[_DISTRACTOR_ITEM] = 1
        placed_at[_DISTRACTOR_ITEM] = _DISTRACTOR_PLACED_AT
    bom = BillOfMaterials.from_counts(counts)
    order = Order(order_id="ORD-1", bom=bom)
    placements = [
        ItemPlacement(item=item, placed_at_frame=frame, entity_id=item)
        for item, frame in placed_at.items()
    ]
    return order, placements


def build_actions(
    placements: list[ItemPlacement], trajectory: PoseTrajectory | None = None
) -> list[ActionEvent]:
    """One 'place' ActionEvent per placement, synced to its frame, carrying the
    operator's right-wrist world position at that frame (causal-fusion GT)."""
    joints_by_frame = {f.frame: f.joints for f in (trajectory or operator_pose()).frames}
    hand = "right_wrist"
    events: list[ActionEvent] = []
    for p in placements:
        wrist = joints_by_frame[p.placed_at_frame][hand]
        events.append(
            ActionEvent(
                frame=p.placed_at_frame,
                item_id=p.item,
                entity_id="operator",
                hand_joint=hand,
                hand_position=(float(wrist[0]), float(wrist[1]), float(wrist[2])),
            )
        )
    return events


def entity_in_view(manifest: dict[str, Any], entity_id: str, cam_id: int) -> tuple[int, int]:
    """(#frames with any point in_view on ``cam_id``, #frames) for an entity."""
    entity = next(e for e in manifest["entities"] if e["id"] == entity_id)
    seen = 0
    for fr in entity["frames"]:
        cams = (pc for pt in fr["points"].values() for pc in pt["per_cam"])
        if any(pc["in_view"] for pc in cams if pc["cam"] == cam_id):
            seen += 1
    return seen, len(entity["frames"])


def run(out_dir: Path, placement_synced: bool = False) -> dict[str, Any]:
    """Build, verify, write sidecars, and return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(placement_synced)
    order, placements = build_order(placement_synced)

    write_manifest(scene, out_dir / "manifest.json")
    # read back the on-disk manifest as a plain dict — the genuine consumer path.
    manifest = json.loads((out_dir / "manifest.json").read_text())

    trajectory = operator_pose(placement_synced, num_frames=scene.num_frames)
    actions = build_actions(placements, trajectory=trajectory)
    # order.json = the order GT sidecar: status + per-item deltas + the synced
    # ActionEvents (manifest stays byte-golden — actions never touch it).
    result: OrderResult = verify_order(
        order.bom, placements, order_id=order.order_id, actions=actions
    )
    write_order_json(result, out_dir / "order.json")
    write_order_json(order, out_dir / "pick_list.json")

    truth = None
    if placement_synced:
        # interactions.json = the causal GT sidecar: only the true pairs — the
        # distractor dip and the distractor (late) placement are omitted, so a
        # consumer that associates either scores a false positive.
        truth = build_action_ground_truth(
            _SYNC_TIMING,
            actor_id="operator",
            tracked_joint=_TRACKED_HAND,
            placements=[p for p in placements if p.item != _DISTRACTOR_ITEM],
        )
        write_actions_json(truth, out_dir / "interactions.json")

    OVERVIEW, WORKTOP = 0, 1
    item_ids = [*_ITEM_STAGING, *([_DISTRACTOR_ITEM] if placement_synced else [])]
    visibility = {
        "operator": {
            "overview": entity_in_view(manifest, "operator", OVERVIEW),
            "worktop": entity_in_view(manifest, "operator", WORKTOP),
        },
        **{
            item: {
                "overview": entity_in_view(manifest, item, OVERVIEW),
                "worktop": entity_in_view(manifest, item, WORKTOP),
            }
            for item in item_ids
        },
    }
    return {
        "manifest": manifest,
        "result": result,
        "actions": actions,
        "interactions": truth,
        "visibility": visibility,
        "out_dir": out_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out", help="output directory"
    )
    parser.add_argument(
        "--placement-synced",
        action="store_true",
        help="placement-synced preset: hand dips at placed_at - δ per item, a "
        "distractor dip, a late distractor item, and an interactions.json "
        "causal ground-truth sidecar",
    )
    args = parser.parse_args()
    summary = run(args.out, placement_synced=args.placement_synced)

    vis = summary["visibility"]
    print(f"[assembly_station] wrote manifest.json + order.json to {summary['out_dir']}")
    print("  camera 0 = overview (operator) | camera 1 = worktop (items)")
    for name, cams in vis.items():
        (ov, n), (wt, _) = cams["overview"], cams["worktop"]
        print(f"  {name:9s}  overview in_view {ov:2d}/{n}   worktop in_view {wt:2d}/{n}")
    print(f"  order {summary['result'].status.value}")
    for ev in summary["actions"]:
        hx, hy, hz = ev.hand_position
        print(
            f"  action {ev.action} {ev.item_id} @frame {ev.frame} "
            f"hand({ev.hand_joint})=({hx:.2f},{hy:.2f},{hz:.2f})"
        )
    truth = summary["interactions"]
    if truth is not None:
        print(f"  interactions.json: {len(truth.pairs)} causal pairs")
        for pair in truth.pairs:
            print(
                f"  {pair.actor_id} dip @frame {pair.action_frame} "
                f"-> {pair.item_id} placed @frame {pair.change_frame}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

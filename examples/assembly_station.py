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
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from multicam_sim import write_manifest
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


def operator_pose() -> PoseTrajectory:
    """A standing COCO-17 operator whose wrists make a small assembling motion."""
    bx, by = _OPERATOR_BASE
    frames: list[PoseFrame] = []
    for f in range(NUM_FRAMES):
        phase = math.sin(2.0 * math.pi * f / (NUM_FRAMES - 1))  # -1..1, smooth
        joints: dict[str, list[float]] = {}
        for name, (dx, dy, dz) in _JOINT_OFFSETS.items():
            reach = 0.06 * phase if name.endswith("wrist") else 0.0  # wrists reach in +y
            joints[name] = [bx + dx, by + dy + reach, dz]
        frames.append(PoseFrame(frame=f, joints=joints))
    return PoseTrajectory(id="operator", skeleton=Skeleton.coco17(), frames=frames)


def item_entity(item_id: str) -> Entity:
    """An item that sits at its staging spot, then jumps into the container at its
    ``placed_at`` frame (and stays)."""
    staging = _ITEM_STAGING[item_id]
    placed_at = _PLACED_AT[item_id]
    frames = [
        EntityFrame(
            frame=f,
            points={"center": list(_CONTAINER if f >= placed_at else staging)},
        )
        for f in range(NUM_FRAMES)
    ]
    return Entity(id=item_id, frames=frames)


def build_scene() -> Scene:
    """Assemble the two-camera scene: overview (operator) + worktop (items)."""
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
    entities = [operator_pose().to_entity(), *(item_entity(i) for i in _ITEM_STAGING)]
    return Scene(fps=FPS, num_frames=NUM_FRAMES, cameras=cameras, entities=entities)


def build_order() -> tuple[Order, list[ItemPlacement]]:
    """The pick-list (one of each part) and the placements as items land."""
    bom = BillOfMaterials.from_counts({item: 1 for item in _ITEM_STAGING})
    order = Order(order_id="ORD-1", bom=bom)
    placements = [
        ItemPlacement(item=item, placed_at_frame=frame, entity_id=item)
        for item, frame in _PLACED_AT.items()
    ]
    return order, placements


def build_actions(placements: list[ItemPlacement]) -> list[ActionEvent]:
    """One 'place' ActionEvent per placement, synced to its frame, carrying the
    operator's right-wrist world position at that frame (causal-fusion GT)."""
    joints_by_frame = {f.frame: f.joints for f in operator_pose().frames}
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


def run(out_dir: Path) -> dict[str, Any]:
    """Build, verify, write sidecars, and return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    order, placements = build_order()

    write_manifest(scene, out_dir / "manifest.json")
    # read back the on-disk manifest as a plain dict — the genuine consumer path.
    manifest = json.loads((out_dir / "manifest.json").read_text())

    actions = build_actions(placements)
    # order.json = the order GT sidecar: status + per-item deltas + the synced
    # ActionEvents (manifest stays byte-golden — actions never touch it).
    result: OrderResult = verify_order(
        order.bom, placements, order_id=order.order_id, actions=actions
    )
    write_order_json(result, out_dir / "order.json")
    write_order_json(order, out_dir / "pick_list.json")

    OVERVIEW, WORKTOP = 0, 1
    item_ids = list(_ITEM_STAGING)
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
        "visibility": visibility,
        "out_dir": out_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out", help="output directory"
    )
    args = parser.parse_args()
    summary = run(args.out)

    vis = summary["visibility"]
    print(f"[assembly_station] wrote manifest.json + order.json to {summary['out_dir']}")
    print("  camera 0 = overview (operator) | camera 1 = worktop (items)")
    for name, cams in vis.items():
        ov, wt = cams["overview"][0], cams["worktop"][0]
        print(f"  {name:9s}  overview in_view {ov:2d}/11   worktop in_view {wt:2d}/11")
    print(f"  order {summary['result'].status.value}")
    for ev in summary["actions"]:
        hx, hy, hz = ev.hand_position
        print(
            f"  action {ev.action} {ev.item_id} @frame {ev.frame} "
            f"hand({ev.hand_joint})=({hx:.2f},{hy:.2f},{hz:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

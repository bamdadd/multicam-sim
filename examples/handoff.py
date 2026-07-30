"""Runnable two-agent hand-off example: converging agents, one parcel, one exchange.

Deterministic, CPU-only, no GL. Run it directly::

    python examples/handoff.py         # or: uv run python examples/handoff.py

Scenario (domain-neutral, synthetic, Z-up metres):

* two **agents** — ``giver`` (south-west) and ``receiver`` (south-east) — walk
  converging paths, meet at the origin exactly at the hand-off frame, then
  diverge (giver north-west, receiver north-east);
* one **object** — ``parcel`` — rests at a staging spot, is picked up by the
  giver at ``PICKUP_FRAME``, changes hands at ``HANDOFF_FRAME`` (the single
  frame where the holder id changes), and is carried out by the receiver.

Emits two ground-truth sidecars next to this file (``--out`` to change dir):

* ``manifest.json``   — the full scene manifest (projection + in_view/visible);
* ``possession.json`` — the possession timeline (per-frame holder segments)
  plus the interaction event (frame/time, giver, receiver, object). The GT
  rides in this sidecar — the manifest stays byte-golden.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from multicam_sim import write_manifest, write_possession_json
from multicam_sim.dsl import CameraRig, SceneBuilder
from multicam_sim.dsl import Path as MotionPath
from multicam_sim.possession import InteractionEvent, PossessionTimeline
from multicam_sim.scene import Scene

FPS = 10.0
NUM_FRAMES = 31
PICKUP_FRAME = 3
HANDOFF_FRAME = 15

GIVER_ID = "giver"
RECEIVER_ID = "receiver"
OBJECT_ID = "parcel"

# --- geometry (metres, Z-up) ------------------------------------------------ #
# Both agents are at the origin at HANDOFF_FRAME (the untimed two-leg waypoint
# path is stretched over the scene, so its mid-frame lands on the middle
# waypoint), then they diverge. The parcel is held at a fixed carry offset.
_EXCHANGE = (0.0, 0.0, 1.0)
_GIVER_PATH = [(-4.0, -3.0, 1.0), _EXCHANGE, (-4.0, 3.0, 1.0)]
_RECEIVER_PATH = [(4.0, -3.0, 1.0), _EXCHANGE, (4.0, 3.0, 1.0)]
_CARRY_OFFSET = (0.0, 0.0, 0.3)
# Staging spot: where the giver's center is at PICKUP_FRAME, minus the carry
# offset, so the pick-up is spatially continuous.
_STAGING = (-3.2, -2.4, _EXCHANGE[2] - _CARRY_OFFSET[2])


def build_scene() -> Scene:
    """Assemble the ring-camera scene: two converging agents + one parcel."""
    cameras = CameraRig.ring(
        n=4,
        radius=8.0,
        height=3.0,
        look_at=_EXCHANGE,
        fov_deg=50.0,
        width=1280,
        height_px=720,
    )
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(cameras)
        .entity(GIVER_ID, MotionPath.waypoints(_GIVER_PATH))
        .entity(RECEIVER_ID, MotionPath.waypoints(_RECEIVER_PATH))
        .entity(OBJECT_ID, MotionPath.linear(_STAGING, _STAGING))
        .handoff(
            OBJECT_ID,
            GIVER_ID,
            RECEIVER_ID,
            HANDOFF_FRAME,
            start=PICKUP_FRAME,
            offset=_CARRY_OFFSET,
        )
        .build()
    )


def run(out_dir: Path) -> dict[str, Any]:
    """Build, write sidecars, and return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()

    write_manifest(scene, out_dir / "manifest.json")
    assert scene.possession is not None
    timeline: PossessionTimeline = scene.possession
    write_possession_json(timeline, out_dir / "possession.json")

    # Per-frame holder changes (the hand-off frame is the only change).
    holders = [timeline.holder_at_frame(OBJECT_ID, f) for f in range(NUM_FRAMES)]
    changes = [
        (f, holders[f - 1], holders[f])
        for f in range(1, NUM_FRAMES)
        if holders[f] != holders[f - 1]
    ]
    return {
        "scene": scene,
        "timeline": timeline,
        "holders": holders,
        "changes": changes,
        "out_dir": out_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out", help="output directory"
    )
    args = parser.parse_args()
    summary = run(args.out)

    timeline: PossessionTimeline = summary["timeline"]
    event: InteractionEvent = timeline.events[0]
    print(f"[handoff] wrote manifest.json + possession.json to {summary['out_dir']}")
    print(
        f"  hand-off {event.object_id}: {event.giver_id} -> {event.receiver_id} "
        f"@frame {event.frame} (t={event.time:.2f}s)"
    )
    for seg in timeline.segments:
        print(
            f"  segment {seg.object_id}: {seg.holder_id} over [{seg.start_frame}, {seg.end_frame})"
        )
    for frame, old, new in summary["changes"]:
        print(f"  holder change @frame {frame}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

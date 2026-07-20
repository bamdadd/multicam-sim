#!/usr/bin/env python3
"""Serve a multicam-sim scene as N live RTSP camera streams.

Each camera of a domain-neutral scene is rendered with the pure-numpy software
rasterizer (no GPU) and published as its own ``rtsp://`` stream through a local
``mediamtx`` server (each camera piped through its own ``ffmpeg`` encoder). This
turns the simulator into a rig of fake IP cameras: multicam-rt's ``ingest-rtsp``
source consumes the streams exactly like real cameras.

**Pixels only, no ground truth.** RTSP carries encoded video, not the analytic
manifest. So this is the realism / network-path test (and a rig-free way to test
the distributed path later), NOT the evaluation path: MOTA / IDF1 / pose scoring
still run offline against the manifest. See ``docs/rtsp.md``.

Usage (see ``scripts/serve-rtsp.sh`` for the dependency-checked wrapper):

    PYTHONPATH=src python3 scripts/serve_rtsp.py --cameras 3 --fps 15
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from multicam_sim.dsl import CameraRig, Path, SceneBuilder
from multicam_sim.dsl.raster import RasterizerBackend


def _find_ffmpeg(explicit: str | None) -> str:
    """Resolve the ffmpeg binary. Prefer ffmpeg@7: multicam-rt's ffmpeg-next
    consumer links FFmpeg 7, and keeping both sides on 7 avoids ABI surprises."""
    if explicit:
        return explicit
    ff7 = "/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg"
    if os.path.exists(ff7):
        return ff7
    found = shutil.which("ffmpeg")
    if not found:
        sys.exit("ffmpeg not found; install it (brew install ffmpeg@7)")
    return found


def _build_scene(cameras: int, frames: int, fps: float, width: int, height: int):
    """A domain-neutral scene: `cameras` on a ring, two items crossing the view.

    No people, no proprietary terms: just items moving through a shared volume,
    enough motion for a detector/tracker to have something to follow.
    """
    return (
        SceneBuilder(fps=fps, num_frames=frames)
        .cameras(
            CameraRig.ring(
                n=cameras,
                radius=4.0,
                height=1.5,
                look_at=(0.0, 0.0, 0.5),
                focal=500.0,
                width=width,
                height_px=height,
            )
        )
        .entity("item-0", Path.linear((-0.8, -0.6, 0.5), (0.8, 0.6, 0.5)))
        .entity("item-1", Path.linear((0.8, -0.4, 0.5), (-0.8, 0.4, 0.5)))
        .build()
    )


def _start_mediamtx(binary: str, port: int) -> subprocess.Popen | None:
    """Start a local mediamtx RTSP server that accepts publish/read on any path."""
    if not shutil.which(binary):
        sys.exit(
            f"{binary} not found; install it (brew install mediamtx), "
            f"or start one yourself and pass --no-mediamtx"
        )
    cfg = tempfile.NamedTemporaryFile(
        "w", suffix=".yml", prefix="mediamtx-", delete=False
    )
    cfg.write(f"rtsp: yes\nrtspAddress: :{port}\npaths:\n  all_others:\n")
    cfg.close()
    proc = subprocess.Popen(
        [binary, cfg.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)  # let the RTSP listener come up before publishers connect
    return proc


def _ffmpeg_publisher(ffmpeg: str, url: str, width: int, height: int, fps: float):
    """An ffmpeg process reading raw RGB frames on stdin and pushing H.264 RTSP."""
    return subprocess.Popen(
        [
            ffmpeg, "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", str(int(fps)), "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp", url,
        ],
        stdin=subprocess.PIPE,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a scene as N RTSP cameras.")
    ap.add_argument("--cameras", type=int, default=3)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg binary (default: ffmpeg@7)")
    ap.add_argument("--mediamtx", default="mediamtx", help="mediamtx binary")
    ap.add_argument(
        "--no-mediamtx",
        action="store_true",
        help="assume an RTSP server is already listening on --host:--port",
    )
    args = ap.parse_args()

    ffmpeg = _find_ffmpeg(args.ffmpeg)
    scene = _build_scene(args.cameras, args.frames, args.fps, args.width, args.height)
    backend = RasterizerBackend()
    cam_ids = [c.id for c in scene.cameras]

    # Pre-render every camera's frames once (the scene is deterministic), then
    # loop them out at the target fps.
    print(f"rendering {len(cam_ids)} cameras x {args.frames} frames ...", flush=True)
    clips = {
        cid: [backend.render(scene, cid, f).astype(np.uint8).tobytes() for f in range(args.frames)]
        for cid in cam_ids
    }

    mediamtx = None if args.no_mediamtx else _start_mediamtx(args.mediamtx, args.port)
    urls = {cid: f"rtsp://{args.host}:{args.port}/cam{cid}" for cid in cam_ids}
    publishers = {
        cid: _ffmpeg_publisher(ffmpeg, urls[cid], args.width, args.height, args.fps)
        for cid in cam_ids
    }

    print("\nRTSP camera streams (pixels only, no ground truth):")
    for cid in cam_ids:
        print(f"  cam{cid}: {urls[cid]}")
    print(f"\nfeed multicam-rt:\n  cargo run -p demo --release --features rtsp,window -- \\")
    print("    " + " ".join(urls[c] for c in cam_ids))
    print("\nCtrl-C to stop.\n", flush=True)

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    period = 1.0 / args.fps
    counters = {cid: itertools.cycle(clips[cid]) for cid in cam_ids}
    try:
        while not stop["now"]:
            t0 = time.time()
            for cid in cam_ids:
                try:
                    publishers[cid].stdin.write(next(counters[cid]))
                except BrokenPipeError:
                    stop["now"] = True
                    break
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        for p in publishers.values():
            if p.stdin:
                try:
                    p.stdin.close()
                except BrokenPipeError:
                    pass
            p.terminate()
        if mediamtx:
            mediamtx.terminate()
        print("stopped.")


if __name__ == "__main__":
    main()

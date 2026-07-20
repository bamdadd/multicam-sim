# RTSP fake cameras

`scripts/serve-rtsp.sh` turns a multicam-sim scene into a rig of **fake IP
cameras**: each camera is rendered with the pure-numpy software rasterizer (no
GPU) and published as its own live `rtsp://` stream. A consumer — multicam-rt's
`ingest-rtsp` source — pulls them exactly like real network cameras.

```
scene ── rasterizer ──▶ ffmpeg (H.264) ──▶ mediamtx ──▶ rtsp://host:8554/cam0
      └─ per camera ──▶ ffmpeg (H.264) ──▶ mediamtx ──▶ rtsp://host:8554/cam1
                                                    ──▶ rtsp://host:8554/cam2
```

## What this is for (and what it is NOT)

**Pixels only, no ground truth.** RTSP carries encoded H.264 video, not the
analytic manifest. The `uv` / `visible` / `xyz_gt` labels do not travel over the
wire. So the RTSP path is:

- a **realism / network-path test**: it exercises the real decode → detect →
  track → pose pipeline on streamed, compressed frames from N independent
  cameras, the way a deployment actually receives them;
- a **rig-free way to test the distributed path** later (streams can come from
  another machine).

It is **NOT the evaluation path.** MOTA / IDF1 / id-switch and pose MPJPE are
scored offline against the manifest ground truth (the `ManifestInference` /
`ManifestPose` path). RTSP frames have no labels to score against. Keep the two
separate: RTSP proves the *plumbing and realism*, the manifest proves the
*accuracy*.

## Dependencies (macOS / Homebrew)

```sh
brew install mediamtx        # zero-dependency RTSP server
brew install ffmpeg@7        # H.264 encode; @7 matches multicam-rt's ffmpeg-next consumer
```

`ffmpeg@7` is pinned because the Rust consumer (`ingest-rtsp`, `ffmpeg-next`)
links FFmpeg 7; FFmpeg 8 removed headers that binding does not yet support.
Keeping the encoder on 7 too avoids surprises. The script prefers
`/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg` automatically.

## Run

```sh
scripts/serve-rtsp.sh                       # 3 cameras, 640x480, 15 fps
scripts/serve-rtsp.sh --cameras 4 --fps 30  # options forwarded to serve_rtsp.py
```

It renders each camera's frames once (the scene is deterministic), starts
`mediamtx`, launches one `ffmpeg` publisher per camera, prints the `rtsp://`
URLs, and loops the clip in real time until Ctrl-C. The scene is domain-neutral
(items moving through a shared volume — no people, no proprietary terms).

Point multicam-rt at the printed URLs:

```sh
cargo run -p demo --release --features rtsp,window -- \
    rtsp://127.0.0.1:8554/cam0 rtsp://127.0.0.1:8554/cam1 rtsp://127.0.0.1:8554/cam2
```

If you already run an RTSP server, pass `--no-mediamtx` and point `--host` /
`--port` at it.

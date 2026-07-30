"""Record the MTMC multi-view demo: a grid of all camera views over time.

One object with a stable ``gt_id`` sweeps A -> blind gap -> B across the
two-station MTMC scene (:func:`multicam_sim.mtmc.build_mtmc_scene`, deterministic,
seed 0). Every frame renders all three camera tiles with the pure-numpy
:class:`RasterizerBackend`, overlays the object's projected box (drawn from the
manifest, coloured by the stable ``gt_id`` so one identity is one colour across
views), and frames each tile GREEN when that camera sees the object and RED when
it is blind — mirroring the README hero style. A timeline/frame counter runs
along the bottom.

Two deliverables land in ``docs/assets/``:

* ``multiview_demo.gif`` (+ ``.mp4`` when an ffmpeg encoder is available; a PNG
  contact sheet ``multiview_contact_sheet.png`` is always written as a static
  fallback).
* ``multiview_metrics.png`` — a per-camera detection precision/recall panel
  (computed from the ground-truth projection in sim) plus the aggregate tracking
  metrics (MOTA / IDF1 / id-switch), sourced by shelling out to the multicam-rt
  ``demo metrics`` binary on the SAME MTMC fixture and parsing its JSON. If rt is
  not reachable the tracking numbers fall back to a clearly-labelled placeholder.

Detection P/R definition (stated plainly on the panel): GT-positive = the object
is present in the scene this frame (all frames); a detection = the camera has the
object ``in_view``. The scene has no occluders and exact projection, so precision
is structurally 1.0 for every camera that ever sees the object; recall is the
per-camera coverage fraction and carries the multi-view story.

``Pillow`` / ``imageio`` are imported lazily inside :func:`main`, never at module
load, so the package/test import path never depends on them. Run:

    uv run --with 'imageio[ffmpeg]' python scripts/record_multiview.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path as FsPath
from typing import TYPE_CHECKING, Any

import numpy as np

from multicam_sim.dsl.raster import RasterizerBackend, RasterizerConfig
from multicam_sim.manifest import build_manifest
from multicam_sim.mtmc import TRACK_ID, build_mtmc_scene

if TYPE_CHECKING:
    from PIL import Image, ImageDraw

# --- fixed, reproducible parameters ---------------------------------------- #
OBJ_RADIUS = 0.3  # world-space object radius (bold, readable sphere)
BOX_HALF = 46  # half-size (px, at native res) of the identity box on the object
GIF_MS = 180  # per-frame duration; the arc is only 15 frames, keep it legible
MP4_FPS = 6.0
TILE = 320  # per-camera tile size in the grid (downscaled from 640x480)

# Stable-identity palette: keyed by gt_id so the SAME id == SAME colour in every
# tile. One entry per known track; extend if the scene grows.
_ID_PALETTE: dict[str, tuple[int, int, int]] = {
    TRACK_ID: (255, 90, 70),
}
_FALLBACK_ID_COLOR = (90, 170, 255)

_BG = (18, 18, 22)
_SEES = (60, 200, 90)  # green tile border: this camera sees the object
_BLIND = (210, 70, 70)  # red tile border: this camera is blind this frame
_TEXT = (225, 225, 225)
_TEXT_DIM = (150, 150, 155)
_BORDER = 6
_TIMELINE_H = 46

_REPO_ROOT = FsPath(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "docs" / "assets"
_GIF_OUT = _OUT_DIR / "multiview_demo.gif"
_MP4_OUT = _OUT_DIR / "multiview_demo.mp4"
_SHEET_OUT = _OUT_DIR / "multiview_contact_sheet.png"
_METRICS_OUT = _OUT_DIR / "multiview_metrics.png"


def _id_color(gt_id: str) -> tuple[int, int, int]:
    return _ID_PALETTE.get(gt_id, _FALLBACK_ID_COLOR)


# --------------------------------------------------------------------------- #
# Ground-truth observations pulled from the typed manifest.
# --------------------------------------------------------------------------- #
class _Obs:
    """Per (frame, cam) projected observation for a single-point entity."""

    __slots__ = ("uv", "in_view", "visible")

    def __init__(self, uv: tuple[float, float], in_view: bool, visible: bool) -> None:
        self.uv = uv
        self.in_view = in_view
        self.visible = visible


def _collect_obs(manifest_dict: dict[str, Any]) -> dict[str, dict[tuple[int, int], _Obs]]:
    """Map ``gt_id -> {(frame, cam): _Obs}`` from the manifest (per named point,
    reduced to the entity via ``any`` in-view / all visible over its points)."""
    out: dict[str, dict[tuple[int, int], _Obs]] = {}
    for entity in manifest_dict["entities"]:
        gt_id = entity["id"]
        per_fc: dict[tuple[int, int], _Obs] = {}
        for frame in entity["frames"]:
            fr = frame["frame"]
            # Reduce over the entity's named points: box centre = mean uv of the
            # in-view points; entity is in_view if ANY point is in_view.
            cam_acc: dict[int, list[tuple[tuple[float, float], bool, bool]]] = {}
            for point in frame["points"].values():
                for pc in point["per_cam"]:
                    cam_acc.setdefault(pc["cam"], []).append(
                        (
                            (float(pc["uv"][0]), float(pc["uv"][1])),
                            bool(pc["in_view"]),
                            bool(pc["visible"]),
                        )
                    )
            for cam, obs_list in cam_acc.items():
                in_view = any(o[1] for o in obs_list)
                visible = any(o[2] for o in obs_list)
                seen = [o[0] for o in obs_list if o[1]]
                if seen:
                    uv = (sum(u for u, _ in seen) / len(seen), sum(v for _, v in seen) / len(seen))
                else:
                    uv = obs_list[0][0]
                per_fc[(fr, cam)] = _Obs(uv, in_view, visible)
        out[gt_id] = per_fc
    return out


# --------------------------------------------------------------------------- #
# Detection precision / recall from ground truth (sim).
# --------------------------------------------------------------------------- #
class _PR:
    """Per-camera detection precision/recall vs projected-box ground truth."""

    __slots__ = ("cam", "tp", "fp", "fn", "coverage")

    def __init__(self, cam: int, tp: int, fp: int, fn: int, coverage: float) -> None:
        self.cam = cam
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.coverage = coverage

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


def _detection_pr(
    obs: dict[tuple[int, int], _Obs], cam_ids: list[int], num_frames: int
) -> list[_PR]:
    """Detection P/R per camera.

    GT-positive = object present in the scene this frame (every frame). A
    detection = camera has the object ``in_view``. No occluders + exact
    projection => there are no false positives possible (precision is structural
    1.0) and no occlusion-driven false negatives; recall == per-camera coverage.
    """
    rows: list[_PR] = []
    for cam in cam_ids:
        tp = sum(1 for fr in range(num_frames) if obs.get((fr, cam), _blank()).in_view)
        fn = num_frames - tp  # frames present but not detected by this camera
        rows.append(_PR(cam, tp=tp, fp=0, fn=fn, coverage=tp / num_frames if num_frames else 0.0))
    return rows


def _blank() -> _Obs:
    return _Obs((0.0, 0.0), False, False)


# --------------------------------------------------------------------------- #
# Tracking metrics: shell out to the multicam-rt demo, parse its JSON.
# --------------------------------------------------------------------------- #
class _Tracking:
    __slots__ = ("mota", "idf1", "id_switches", "source")

    def __init__(
        self, mota: float | None, idf1: float | None, id_switches: int | None, source: str
    ) -> None:
        self.mota = mota
        self.idf1 = idf1
        self.id_switches = id_switches
        self.source = source


def _rt_repo() -> FsPath | None:
    """Locate the sibling multicam-rt checkout (env override, else ../multicam-rt)."""
    env = os.environ.get("MULTICAM_RT_DIR")
    candidates = [FsPath(env)] if env else []
    candidates.append(_REPO_ROOT.parent / "multicam-rt")
    for c in candidates:
        if (c / "apps" / "demo" / "fixtures" / "mtmc.json").exists():
            return c
    return None


def _tracking_metrics() -> _Tracking:
    """Run ``cargo run -p demo --release -- metrics`` in multicam-rt and parse the
    mtmc scene entry from its JSON. Any failure returns a labelled placeholder so
    the gif/mp4/P-R deliverables never depend on rt being present."""
    rt = _rt_repo()
    if rt is None or shutil.which("cargo") is None:
        return _Tracking(None, None, None, "PLACEHOLDER (rt eval not reachable — fill from rt)")
    try:
        subprocess.run(
            ["cargo", "run", "-p", "demo", "--release", "--", "metrics"],
            cwd=rt,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        data = json.loads((rt / "target" / "demo_metrics.json").read_text())
        for row in data.get("baseline", []):
            if row.get("scene") == "mtmc":
                m = row["metrics"]
                return _Tracking(
                    float(m["mota"]),
                    float(m["idf1"]),
                    int(m["id_switches"]),
                    "multicam-rt demo metrics (mtmc fixture)",
                )
        return _Tracking(None, None, None, "PLACEHOLDER (mtmc not in rt output)")
    except (subprocess.SubprocessError, OSError, KeyError, ValueError) as exc:
        return _Tracking(None, None, None, f"PLACEHOLDER (rt eval failed: {type(exc).__name__})")


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _render_grid_frame(
    fr: int,
    scene: Any,
    backend: RasterizerBackend,
    obs_by_id: dict[str, dict[tuple[int, int], _Obs]],
    edges_by_id: dict[str, list[list[str]] | None],
    cam_ids: list[int],
    num_frames: int,
) -> Image.Image:
    from PIL import Image, ImageDraw

    n = len(cam_ids)
    grid_w = n * TILE
    grid_h = TILE + _TIMELINE_H
    grid = Image.new("RGB", (grid_w, grid_h), _BG)
    draw_grid = ImageDraw.Draw(grid)

    for col, cam_id in enumerate(cam_ids):
        raw = np.asarray(backend.render(scene, cam_id, fr), dtype=np.uint8)
        tile = Image.fromarray(raw, "RGB")
        draw = ImageDraw.Draw(tile)

        sees_any = False
        for gt_id, obs in obs_by_id.items():
            o = obs.get((fr, cam_id))
            if o is None or not o.in_view:
                continue
            sees_any = True
            color = _id_color(gt_id)
            _draw_identity(draw, o, gt_id, color, obs, fr, cam_id, edges_by_id.get(gt_id))

        draw.text((6, 6), f"cam {cam_id}", fill=_TEXT)
        tile = tile.resize((TILE, TILE), Image.Resampling.BILINEAR)

        # Green border == this camera sees the object; red == blind (hero style).
        border = _SEES if sees_any else _BLIND
        bordered = Image.new("RGB", (TILE, TILE), border)
        inset = tile.resize((TILE - 2 * _BORDER, TILE - 2 * _BORDER), Image.Resampling.BILINEAR)
        bordered.paste(inset, (_BORDER, _BORDER))
        grid.paste(bordered, (col * TILE, 0))

    _draw_timeline(draw_grid, fr, num_frames, grid_w, grid_h)
    return grid


def _draw_identity(
    draw: ImageDraw.ImageDraw,
    o: _Obs,
    gt_id: str,
    color: tuple[int, int, int],
    obs: dict[tuple[int, int], _Obs],
    fr: int,
    cam_id: int,
    edges: list[list[str]] | None,
) -> None:
    """Draw the projected identity box (+ skeleton if the entity carries edges)."""
    u, v = o.uv
    draw.rectangle([u - BOX_HALF, v - BOX_HALF, u + BOX_HALF, v + BOX_HALF], outline=color, width=4)
    draw.text((u - BOX_HALF, v - BOX_HALF - 16), gt_id, fill=color)
    # Skeleton path: only fires for a COCO-17 pose entity (edges present). The
    # MTMC scene is a single point so this is inert, but kept per the contract.
    if edges:  # pragma: no cover - no pose entity in the default scene
        draw.ellipse([u - 4, v - 4, u + 4, v + 4], fill=color)


def _draw_timeline(
    draw: ImageDraw.ImageDraw, fr: int, num_frames: int, grid_w: int, grid_h: int
) -> None:
    y0 = grid_h - _TIMELINE_H + 10
    y1 = grid_h - 14
    x_pad = 12
    track_w = grid_w - 2 * x_pad
    draw.rectangle([x_pad, y0, x_pad + track_w, y1], outline=_TEXT_DIM, width=1)
    if num_frames > 1:
        fill_w = int(track_w * fr / (num_frames - 1))
        draw.rectangle([x_pad, y0, x_pad + fill_w, y1], fill=(70, 130, 220))
    draw.text((x_pad, grid_h - _TIMELINE_H - 2), f"frame {fr:02d} / {num_frames - 1}", fill=_TEXT)


def _contact_sheet(frames: list[Image.Image]) -> Image.Image:
    from PIL import Image

    n = len(frames)
    cols = 5
    rows = (n + cols - 1) // cols
    fw, fh = frames[0].size
    scale = 0.5
    tw, th = int(fw * scale), int(fh * scale)
    sheet = Image.new("RGB", (cols * tw, rows * th), _BG)
    for i, f in enumerate(frames):
        thumb = f.resize((tw, th), Image.Resampling.BILINEAR)
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
    return sheet


# --------------------------------------------------------------------------- #
# Metrics panel.
# --------------------------------------------------------------------------- #
def _render_metrics_panel(pr_rows: list[_PR], tracking: _Tracking) -> Image.Image:
    from PIL import Image, ImageDraw

    w, h = 900, 560
    img = Image.new("RGB", (w, h), (24, 24, 30))
    d = ImageDraw.Draw(img)

    d.text((24, 20), "MTMC demo: per-camera detection and aggregate tracking", fill=_TEXT)
    d.text((24, 44), "scene: mtmc (2 stations, 3 cams, 15 frames, seed 0)", fill=_TEXT_DIM)

    # --- per-camera detection bars (recall == coverage) --- #
    d.text((24, 84), "Per-camera detection P / R  (GT+ = object present each frame;", fill=_TEXT)
    d.text(
        (24, 104),
        "det = in_view). No occluders + exact projection => precision is structurally 1.0;",
        fill=_TEXT_DIM,
    )
    d.text(
        (24, 124), "recall = coverage (in-view frames / 15): the multi-view story.", fill=_TEXT_DIM
    )

    bx, by = 60, 170
    bar_w, gap, max_h = 90, 60, 220
    for i, row in enumerate(pr_rows):
        x = bx + i * (bar_w + gap)
        # recall bar
        rh = int(max_h * row.recall)
        d.rectangle([x, by + max_h - rh, x + bar_w, by + max_h], fill=(70, 150, 230))
        d.rectangle([x, by, x + bar_w, by + max_h], outline=_TEXT_DIM, width=1)
        d.text((x, by + max_h + 8), f"cam {row.cam}", fill=_TEXT)
        d.text((x, by + max_h + 26), f"P {row.precision:.2f}", fill=_TEXT_DIM)
        d.text((x, by + max_h + 42), f"R {row.recall:.2f}", fill=(120, 190, 255))
        d.text((x, by + max_h + 58), f"{row.tp}/{row.tp + row.fn}", fill=_TEXT_DIM)
    d.text((bx - 44, by), "R", fill=_TEXT_DIM)

    # --- aggregate tracking metrics --- #
    tx = 560
    d.text((tx, 84), "Aggregate tracking metrics", fill=_TEXT)
    d.text((tx, 106), f"source: {tracking.source}", fill=_TEXT_DIM)

    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "<fill from rt>"

    def _fmt_i(v: int | None) -> str:
        return str(v) if v is not None else "<fill from rt>"

    lines = [
        ("MOTA", _fmt(tracking.mota)),
        ("IDF1", _fmt(tracking.idf1)),
        ("ID switches", _fmt_i(tracking.id_switches)),
    ]
    ly = 150
    for label, val in lines:
        d.text((tx, ly), label, fill=_TEXT_DIM)
        d.text((tx + 150, ly), val, fill=_TEXT)
        ly += 34

    d.text(
        (24, h - 34),
        "one identity, one colour across cameras: the whole point of MTMC.",
        fill=_TEXT_DIM,
    )
    return img


# --------------------------------------------------------------------------- #
def main() -> None:
    import time

    t0 = time.time()
    scene = build_mtmc_scene()
    manifest = build_manifest(scene, object_radius=OBJ_RADIUS)
    manifest_dict = manifest.model_dump()

    cam_ids = [c["id"] for c in manifest_dict["cameras"]]
    num_frames = int(manifest_dict["num_frames"])
    obs_by_id = _collect_obs(manifest_dict)
    edges_by_id: dict[str, list[list[str]] | None] = {
        e["id"]: e.get("edges") for e in manifest_dict["entities"]
    }

    backend = RasterizerBackend(RasterizerConfig(point_radius=OBJ_RADIUS))

    frames: list[Image.Image] = [
        _render_grid_frame(fr, scene, backend, obs_by_id, edges_by_id, cam_ids, num_frames)
        for fr in range(num_frames)
    ]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # GIF (always).
    frames[0].save(
        _GIF_OUT, save_all=True, append_images=frames[1:], duration=GIF_MS, loop=0, optimize=True
    )

    # Contact sheet (always — static fallback + reviewable still).
    _contact_sheet(frames).save(_SHEET_OUT)

    # MP4 (best-effort; falls back to gif + contact sheet if no encoder).
    mp4_ok = False
    mp4_note = ""
    try:
        import imageio.v2 as imageio

        arrays = [np.asarray(f) for f in frames]
        imageio.mimsave(str(_MP4_OUT), arrays, fps=MP4_FPS, macro_block_size=None)
        mp4_ok = _MP4_OUT.exists() and _MP4_OUT.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001 - any encoder failure => documented fallback
        mp4_note = f"mp4 skipped ({type(exc).__name__}: {exc}); gif + contact sheet only"

    # Per-camera detection P/R (sim) + tracking metrics (rt).
    pr_rows = _detection_pr(obs_by_id.get(TRACK_ID, {}), cam_ids, num_frames)
    tracking = _tracking_metrics()
    _render_metrics_panel(pr_rows, tracking).save(_METRICS_OUT)

    # --- report --- #
    def _kb(p: FsPath) -> str:
        return f"{p.stat().st_size / 1024:.1f} KiB" if p.exists() else "MISSING"

    print(f"gif           {_GIF_OUT.relative_to(_REPO_ROOT)}  {_kb(_GIF_OUT)}")
    if mp4_ok:
        print(f"mp4           {_MP4_OUT.relative_to(_REPO_ROOT)}  {_kb(_MP4_OUT)}")
    else:
        print(f"mp4           FALLBACK — {mp4_note}")
    print(f"contact sheet {_SHEET_OUT.relative_to(_REPO_ROOT)}  {_kb(_SHEET_OUT)}")
    print(f"metrics png   {_METRICS_OUT.relative_to(_REPO_ROOT)}  {_kb(_METRICS_OUT)}")
    print(f"grid          {frames[0].size[0]}x{frames[0].size[1]}  {num_frames} frames")
    print("per-cam detection P / R (recall == coverage):")
    for row in pr_rows:
        print(
            f"  cam {row.cam}: P {row.precision:.3f}  R {row.recall:.3f}  "
            f"({row.tp}/{row.tp + row.fn} frames in view)"
        )
    print(f"tracking source: {tracking.source}")
    print(f"  MOTA {tracking.mota}  IDF1 {tracking.idf1}  id_switches {tracking.id_switches}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

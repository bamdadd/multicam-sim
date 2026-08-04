"""Report public-safe coverage metrics for a multicam-sim manifest.

Usage::

    uv run python scripts/coverage_metrics.py path/to/manifest.json
    uv run --with matplotlib python scripts/coverage_metrics.py \
        path/to/manifest.json --panel docs/assets/coverage_metrics.png

JSON is always printed to stdout. ``--panel`` additionally writes a headless,
static PNG with per-camera coverage, scene-level totals, and an entity timeline
showing blind, single-camera, overlap, and handoff frames. Matplotlib is imported
lazily, so the JSON-only path keeps the package and CI dependency-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multicam_sim.coverage import CoverageReport, compute_coverage_metrics
from multicam_sim.manifest import Manifest


def _coverage_matrix(manifest: Manifest) -> tuple[list[int], list[str], list[list[int]]]:
    """Return frame ids, entity ids, and camera counts for the panel timeline.

    ``-1`` means that an entity has no sample at that frame, ``0`` is a blind
    frame, ``1`` is single-camera coverage, and values of two or more are overlap.
    Like :func:`compute_coverage_metrics`, named points are reduced to one
    entity-frame sample so skeletons do not receive extra weight.
    """
    frames = sorted({frame.frame for entity in manifest.entities for frame in entity.frames})
    entity_ids: list[str] = []
    matrix: list[list[int]] = []

    for entity in manifest.entities:
        counts: dict[int, int] = {}
        for frame in entity.frames:
            cameras = {
                observation.cam
                for point in frame.points.values()
                for observation in point.per_cam
                if observation.in_view
            }
            counts[frame.frame] = len(cameras)
        entity_ids.append(entity.id)
        matrix.append([counts.get(frame, -1) for frame in frames])

    return frames, entity_ids, matrix


def render_coverage_panel(
    manifest: Manifest,
    report: CoverageReport,
    output: Path,
    *,
    title: str,
) -> Path:
    """Render a deterministic, headless coverage-and-handoff panel to ``output``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    frames, entity_ids, matrix = _coverage_matrix(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(12, 6.75), facecolor="#111318", layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15), width_ratios=(1.7, 1.0))
    bars = figure.add_subplot(grid[0, 0])
    summary = figure.add_subplot(grid[0, 1])
    timeline = figure.add_subplot(grid[1, :])

    for axis in (bars, summary, timeline):
        axis.set_facecolor("#191c23")
        axis.tick_params(colors="#c9d1d9")
        for spine in axis.spines.values():
            spine.set_color("#343a46")

    camera_labels = [f"camera {row.camera_id}" for row in report.per_camera]
    fractions = [row.fraction for row in report.per_camera]
    positions = list(range(len(camera_labels)))
    bar_colors = ["#58a6ff" if fraction < 1.0 else "#3fb950" for fraction in fractions]
    bars.barh(positions, fractions, color=bar_colors, height=0.62)
    bars.set_yticks(positions, camera_labels)
    bars.invert_yaxis()
    bars.set_xlim(0.0, 1.08)
    bars.set_xlabel("fraction of entity-frame samples in view", color="#c9d1d9")
    bars.set_title("Per-camera coverage", color="#f0f6fc", loc="left", weight="bold")
    bars.grid(axis="x", color="#30363d", alpha=0.7, linewidth=0.8)
    bars.set_axisbelow(True)
    for position, row in zip(positions, report.per_camera, strict=True):
        bars.text(
            min(row.fraction + 0.018, 1.01),
            position,
            f"{row.fraction:.3f}  ({row.in_view_frames}/{row.total_frames})",
            va="center",
            color="#f0f6fc",
            fontsize=9,
        )

    sample_count = report.per_camera[0].total_frames if report.per_camera else 0
    summary.axis("off")
    summary.set_title("Scene summary", color="#f0f6fc", loc="left", weight="bold")
    summary_rows = (
        ("Entity-frame samples", sample_count, "#c9d1d9"),
        ("Overlap frames", report.overlap_count, "#39d0c8"),
        ("Handoff events", len(report.handoff_points), "#f2cc60"),
        ("Blind-gap frames", report.blind_gap_count, "#ff7b72"),
    )
    for row_index, (label, value, color) in enumerate(summary_rows):
        y = 0.82 - row_index * 0.2
        summary.text(0.02, y, label, color="#8b949e", fontsize=10, transform=summary.transAxes)
        summary.text(
            0.96,
            y,
            str(value),
            color=color,
            fontsize=18,
            weight="bold",
            ha="right",
            transform=summary.transAxes,
        )

    if frames and entity_ids:
        clipped = [[min(value, 2) for value in row] for row in matrix]
        colors = ListedColormap(["#30363d", "#da3633", "#388bfd", "#2ea043"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], colors.N)
        timeline.imshow(clipped, aspect="auto", interpolation="nearest", cmap=colors, norm=norm)
        timeline.set_yticks(range(len(entity_ids)), entity_ids)
        tick_step = max(1, len(frames) // 10)
        tick_positions = list(range(0, len(frames), tick_step))
        if tick_positions[-1] != len(frames) - 1:
            tick_positions.append(len(frames) - 1)
        timeline.set_xticks(tick_positions, [str(frames[index]) for index in tick_positions])
        frame_positions = {frame: index for index, frame in enumerate(frames)}
        entity_positions = {entity_id: index for index, entity_id in enumerate(entity_ids)}
        for handoff in report.handoff_points:
            x = frame_positions.get(handoff.frame)
            y = entity_positions.get(handoff.entity_id)
            if x is not None and y is not None:
                timeline.scatter(x, y, marker="v", s=58, color="#f2cc60", edgecolor="#111318")
        timeline.set_xlabel("frame", color="#c9d1d9")
        timeline.set_title(
            "Entity timeline (triangle = camera-set change)",
            color="#f0f6fc",
            loc="left",
            weight="bold",
        )
        timeline.set_xticks([index - 0.5 for index in range(1, len(frames))], minor=True)
        timeline.grid(which="minor", axis="x", color="#111318", linewidth=0.35, alpha=0.5)
    else:
        timeline.text(
            0.5,
            0.5,
            "No entity frames in manifest",
            ha="center",
            va="center",
            color="#8b949e",
            transform=timeline.transAxes,
        )

    legend = [
        Patch(facecolor="#30363d", label="no sample"),
        Patch(facecolor="#da3633", label="blind (0 cameras)"),
        Patch(facecolor="#388bfd", label="1 camera"),
        Patch(facecolor="#2ea043", label="overlap (2+ cameras)"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#f2cc60", label="handoff"),
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncols=5,
        frameon=False,
        labelcolor="#c9d1d9",
    )
    figure.suptitle(title, color="#f0f6fc", fontsize=16, weight="bold")
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def main() -> None:
    """Load a manifest, print JSON, and optionally render a static PNG panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="multicam-sim manifest JSON")
    parser.add_argument("--panel", type=Path, help="also write a static coverage panel PNG")
    parser.add_argument("--title", help="panel title (defaults to the manifest filename)")
    args = parser.parse_args()

    manifest = Manifest.model_validate_json(args.manifest.read_text())
    report = compute_coverage_metrics(manifest)
    if args.panel is not None:
        render_coverage_panel(
            manifest,
            report,
            args.panel,
            title=args.title or f"Coverage metrics — {args.manifest.stem}",
        )
        print(f"wrote coverage panel: {args.panel}", file=sys.stderr)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

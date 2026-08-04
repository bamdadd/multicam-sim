"""Coverage and handoff metrics computed only from manifest ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .manifest import Manifest


@dataclass(frozen=True)
class CameraCoverage:
    """Coverage of one camera across all entity-frame samples."""

    camera_id: int
    in_view_frames: int
    total_frames: int

    @property
    def fraction(self) -> float:
        """Return the fraction of entity-frame samples seen by this camera."""
        return self.in_view_frames / self.total_frames if self.total_frames else 0.0


@dataclass(frozen=True)
class FrameRef:
    """A frame belonging to one manifest entity."""

    entity_id: str
    frame: int


@dataclass(frozen=True)
class HandoffPoint:
    """A change in the cameras covering an entity between adjacent frames."""

    entity_id: str
    frame: int
    entered_cameras: tuple[int, ...]
    exited_cameras: tuple[int, ...]


@dataclass(frozen=True)
class CoverageReport:
    """Public-safe multi-camera coverage metrics for one manifest."""

    per_camera: tuple[CameraCoverage, ...]
    overlap_frames: tuple[FrameRef, ...]
    handoff_points: tuple[HandoffPoint, ...]
    blind_gap_frames: tuple[FrameRef, ...]

    @property
    def overlap_count(self) -> int:
        """Return the number of entity-frame samples seen by at least two cameras."""
        return len(self.overlap_frames)

    @property
    def blind_gap_count(self) -> int:
        """Return the number of entity-frame samples seen by no cameras."""
        return len(self.blind_gap_frames)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-ready representation."""
        return {
            "per_camera": [{**asdict(row), "fraction": row.fraction} for row in self.per_camera],
            "overlap_count": self.overlap_count,
            "overlap_frames": [asdict(frame) for frame in self.overlap_frames],
            "handoff_points": [asdict(point) for point in self.handoff_points],
            "blind_gap_count": self.blind_gap_count,
            "blind_gap_frames": [asdict(frame) for frame in self.blind_gap_frames],
        }


def _in_view_cameras(manifest: Manifest, entity_index: int, frame_index: int) -> set[int]:
    frame = manifest.entities[entity_index].frames[frame_index]
    return {
        observation.cam
        for point in frame.points.values()
        for observation in point.per_cam
        if observation.in_view
    }


def compute_coverage_metrics(manifest: Manifest) -> CoverageReport:
    """Compute camera coverage, overlaps, handoffs, and blind gaps.

    One sample is one entity at one frame. An entity is considered in view on a
    camera when any of its named points has ``in_view=True`` on that camera. This
    avoids overweighting skeletons merely because they have more named points
    than single-point objects.
    """
    camera_ids = sorted(camera.id for camera in manifest.cameras)
    counts = {camera_id: 0 for camera_id in camera_ids}
    total_frames = 0
    overlaps: list[FrameRef] = []
    handoffs: list[HandoffPoint] = []
    blind_gaps: list[FrameRef] = []

    for entity_index, entity in enumerate(manifest.entities):
        previous: set[int] | None = None
        previous_frame: int | None = None
        for frame_index, frame in enumerate(entity.frames):
            in_view = _in_view_cameras(manifest, entity_index, frame_index)
            frame_ref = FrameRef(entity.id, frame.frame)
            total_frames += 1
            for camera_id in in_view:
                counts[camera_id] = counts.get(camera_id, 0) + 1

            if len(in_view) >= 2:
                overlaps.append(frame_ref)
            if not in_view:
                blind_gaps.append(frame_ref)

            if (
                previous is not None
                and previous_frame is not None
                and frame.frame == previous_frame + 1
                and in_view != previous
                and in_view
                and previous
            ):
                handoffs.append(
                    HandoffPoint(
                        entity_id=entity.id,
                        frame=frame.frame,
                        entered_cameras=tuple(sorted(in_view - previous)),
                        exited_cameras=tuple(sorted(previous - in_view)),
                    )
                )
            previous = in_view
            previous_frame = frame.frame

    return CoverageReport(
        per_camera=tuple(
            CameraCoverage(camera_id, counts[camera_id], total_frames) for camera_id in camera_ids
        ),
        overlap_frames=tuple(overlaps),
        handoff_points=tuple(handoffs),
        blind_gap_frames=tuple(blind_gaps),
    )

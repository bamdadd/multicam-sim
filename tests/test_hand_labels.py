"""Moving hand occluder + analytic image-space visibility labels.

Covers the load-bearing properties agreed with orch-4:

* a nearer occluder that fully covers the object silhouette -> ``visible_fraction``
  is analytic-exact ``0.0`` and ``occluded`` is ``True``;
* no occluder -> ``visible_fraction == 1.0``;
* an APPROACH-ONLY hand (enters and stops over the target) -> ``visible_fraction``
  is monotone non-increasing (the full-pass sweep is U-shaped and reserved for the
  hero dose-response artifact, not this assertion);
* the labels are opt-in: absent ``object_radius`` -> fields are ``None`` and the
  manifest stays byte-identical;
* the manifest frame-resolves a :class:`HandOccluder` (never calls its raising
  ``blocks_segment``), so ``visible``/``occ_frac`` track the hand's per-frame pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_sim import (
    HandKeyframe,
    HandOccluder,
    build_manifest,
    silhouette_visible_fraction,
)
from multicam_sim.dsl import CameraRig, HandSweep, Path, SceneBuilder
from multicam_sim.occluders import Sphere
from multicam_sim.scene import Scene

FPS = 30.0
NUM_FRAMES = 11
TARGET_CAM = 0
OBJECT_RADIUS = 0.05


def _one_camera() -> CameraRig:
    return CameraRig.ring(
        n=3,
        radius=4.0,
        height=1.5,
        look_at=(0.0, 0.0, 0.5),
        focal=800.0,
        width=64,
        height_px=48,
    )


def _hand_scene(*, approach: bool, radius: float = 0.25, span: float = 0.5) -> Scene:
    sweep = (
        HandSweep.sphere(radius, span=span).blocks(camera=TARGET_CAM).during((0, NUM_FRAMES - 1))
    )
    if approach:
        sweep = sweep.approaching()
    return (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(_one_camera())
        .entity("obj", Path.linear((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)))
        .occlude_hand(sweep)
        .build()
    )


def _target_visible_fractions(scene: Scene) -> list[float | None]:
    """The obj-centre ``visible_fraction`` on TARGET_CAM across frames."""
    manifest = build_manifest(scene, object_radius=OBJECT_RADIUS)
    entity = next(e for e in manifest.entities if e.id == "obj")
    out: list[float | None] = []
    for frame in entity.frames:
        obs = next(pc for pc in frame.points["center"].per_cam if pc.cam == TARGET_CAM)
        out.append(obs.visible_fraction)
    return out


# --- analytic endpoints (exact) --------------------------------------------- #


def test_no_occluder_is_fully_visible() -> None:
    camera = _one_camera()[0]
    point = np.array([0.0, 0.0, 0.5])
    assert silhouette_visible_fraction(camera, point, OBJECT_RADIUS, []) == 1.0


def test_nearer_occluder_on_sightline_fully_covers() -> None:
    """A big sphere on the point->camera ray covers the silhouette exactly (0.0)."""
    camera = _one_camera()[0]
    point = np.array([0.0, 0.0, 0.5])
    # halfway from the point toward the camera centre: same pixel, nearer depth.
    occ_centre = point + 0.5 * (camera.centre() - point)
    occ = Sphere(center=occ_centre.tolist(), radius=0.3)
    assert silhouette_visible_fraction(camera, point, OBJECT_RADIUS, [occ]) == 0.0


def test_farther_occluder_does_not_cover() -> None:
    """An occluder BEHIND the object (farther from camera) never covers it."""
    camera = _one_camera()[0]
    point = np.array([0.0, 0.0, 0.5])
    behind = point + 0.5 * (point - camera.centre())  # away from the camera
    occ = Sphere(center=behind.tolist(), radius=0.3)
    assert silhouette_visible_fraction(camera, point, OBJECT_RADIUS, [occ]) == 1.0


# --- through the manifest --------------------------------------------------- #


def test_full_cover_through_manifest_sets_occluded() -> None:
    """A hand big enough to swallow the object at mid-frame -> vf 0, occluded True."""
    scene = _hand_scene(approach=False, radius=0.6, span=0.4)
    manifest = build_manifest(scene, object_radius=OBJECT_RADIUS)
    entity = next(e for e in manifest.entities if e.id == "obj")
    mid = (NUM_FRAMES - 1) // 2
    frame = next(f for f in entity.frames if f.frame == mid)
    obs = next(pc for pc in frame.points["center"].per_cam if pc.cam == TARGET_CAM)
    assert obs.visible_fraction == 0.0
    assert obs.occluded is True


def test_approach_only_is_monotone_non_increasing() -> None:
    """Hand enters and stops centred -> visible_fraction never rises."""
    fractions = _target_visible_fractions(_hand_scene(approach=True))
    values = [v for v in fractions if v is not None]
    assert len(values) == NUM_FRAMES
    for earlier, later in zip(values, values[1:], strict=False):
        assert later <= earlier + 1e-9
    # it must actually move: ends more occluded than it starts.
    assert values[-1] < values[0]


def test_full_pass_sweep_is_u_shaped() -> None:
    """The hero full-pass sweep dips to a minimum then recovers (1 -> ~0 -> 1)."""
    values = [v for v in _target_visible_fractions(_hand_scene(approach=False)) if v is not None]
    lowest = min(range(len(values)), key=lambda i: values[i])
    assert 0 < lowest < len(values) - 1  # minimum is interior, not at an end
    assert values[lowest] < values[0]
    assert values[lowest] < values[-1]


# --- opt-in / byte-identity ------------------------------------------------- #


def test_labels_absent_without_object_radius() -> None:
    """No ``object_radius`` -> visible_fraction/occluded are None and omitted."""
    scene = _hand_scene(approach=True)
    manifest = build_manifest(scene)  # feature off
    obs = manifest.entities[0].frames[0].points["center"].per_cam[0]
    assert obs.visible_fraction is None
    assert obs.occluded is None
    assert "visible_fraction" not in manifest.to_json()
    assert "occluded" not in manifest.to_json()


# --- HandOccluder unit ------------------------------------------------------ #


def test_hand_center_interpolates_and_clamps() -> None:
    hand = HandOccluder(
        radius=0.1,
        keyframes=[
            HandKeyframe(frame=0, center=[0.0, 0.0, 0.0]),
            HandKeyframe(frame=10, center=[10.0, 0.0, 0.0]),
        ],
    )
    assert np.allclose(hand.center_at(5), [5.0, 0.0, 0.0])
    assert np.allclose(hand.center_at(-3), [0.0, 0.0, 0.0])  # clamp low
    assert np.allclose(hand.center_at(99), [10.0, 0.0, 0.0])  # clamp high


def test_hand_at_frame_is_a_positioned_sphere() -> None:
    hand = HandOccluder(
        radius=0.2,
        keyframes=[
            HandKeyframe(frame=0, center=[0.0, 0.0, 0.0]),
            HandKeyframe(frame=4, center=[4.0, 0.0, 0.0]),
        ],
    )
    solid = hand.at_frame(2)
    assert isinstance(solid, Sphere)
    assert np.allclose(solid.center, [2.0, 0.0, 0.0])
    assert solid.radius == 0.2


def test_hand_blocks_segment_raises_directly() -> None:
    """The mover has no frameless sightline test — callers must use at_frame."""
    hand = HandOccluder(radius=0.1, keyframes=[HandKeyframe(frame=0, center=[0.0, 0.0, 0.0])])
    with pytest.raises(NotImplementedError):
        hand.blocks_segment(np.zeros(3), np.ones(3))


def test_manifest_frame_resolves_moving_hand() -> None:
    """visible tracks the hand's pose: blocked at mid-frame, clear at an edge.

    If the manifest did not resolve at_frame per frame it would hit the raising
    blocks_segment; that it produces per-frame booleans proves the resolution.
    """
    scene = _hand_scene(approach=False, radius=0.3, span=0.6)
    manifest = build_manifest(scene, object_radius=OBJECT_RADIUS)
    entity = next(e for e in manifest.entities if e.id == "obj")
    mid = (NUM_FRAMES - 1) // 2

    def visible_at(frame_idx: int) -> bool:
        frame = next(f for f in entity.frames if f.frame == frame_idx)
        obs = next(pc for pc in frame.points["center"].per_cam if pc.cam == TARGET_CAM)
        return bool(obs.visible)

    assert visible_at(mid) is False  # hand centred on the sightline
    assert visible_at(0) is True  # hand swept off to the side


# --- empirical rasterizer cross-check (tolerance, NOT the source) ------------ #


def test_rasterizer_pixel_count_cross_checks_analytic() -> None:
    """The rendered visible-object-pixel ratio tracks the analytic value.

    A loose tolerance on purpose: a faceted icosphere silhouette vs a perfect
    analytic disc will never match tightly. The pixel count is a CROSS-CHECK of
    the analytic ``visible_fraction`` (the manifest's source), never its source.
    """
    from multicam_sim.dsl.raster import RasterizerBackend, RasterizerConfig

    # Match the analytic proxy radius; keep the default reddish object vs grey
    # occluder so visible object pixels are the ones the hand has NOT overwritten.
    cfg = RasterizerConfig(point_radius=OBJECT_RADIUS)
    backend = RasterizerBackend(cfg)

    def object_pixels(scene: Scene, frame: int) -> int:
        """Count reddish object pixels (R >> B); grey occluder + dark bg excluded."""
        img = backend.render(scene, camera_id=TARGET_CAM, frame=frame).astype(np.int16)
        return int(((img[..., 0] - img[..., 2]) > 40).sum())

    # Approach mode gives a gradual curve with genuine partial-cover frames.
    hand_scene = _hand_scene(approach=True, radius=0.15, span=0.45)
    clear_scene = (
        SceneBuilder(fps=FPS, num_frames=NUM_FRAMES)
        .cameras(_one_camera())
        .entity("obj", Path.linear((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)))
        .build()
    )
    analytic = _target_visible_fractions(hand_scene)

    checked = 0
    for frame in range(NUM_FRAMES):
        expected = analytic[frame]
        if expected is None or not (0.15 < expected < 0.95):
            continue  # only assert in the partial regime
        base = object_pixels(clear_scene, frame)
        if base == 0:
            continue
        empirical = object_pixels(hand_scene, frame) / base
        assert abs(empirical - expected) < 0.2
        checked += 1
    assert checked >= 1  # the sweep must pass through the partial regime

"""2D keypoint/skeleton overlay exporter tests.

The exporter uses optional drawing/video dependencies imported lazily inside the
export functions, so ``import multicam_sim`` stays light. The frames path is
exercised in CI because Pillow is in the dev dependency group; the MP4 path is
guarded by ``importorskip`` because it needs the heavier ``overlay-mp4`` extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from multicam_sim import build_manifest, build_smoke_scene, export_overlay
from multicam_sim.overlay import _draw_dashed_line


def _smoke_manifest() -> dict:
    return build_manifest(build_smoke_scene()).model_dump()


def test_importing_package_does_not_import_pillow() -> None:
    # re-importing is fine; the assertion is that Pillow was not pulled in at
    # package load time.
    assert "PIL" not in sys.modules


def test_export_overlay_frames_writes_expected_files(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    manifest = _smoke_manifest()
    out_dir = tmp_path / "frames"
    export_overlay(manifest, out_dir, fmt="frames")

    for cam in manifest["cameras"]:
        cam_dir = out_dir / f"cam{cam['id']}"
        assert cam_dir.is_dir()
        for frame_idx in range(manifest["num_frames"]):
            path = cam_dir / f"frame_{frame_idx:04d}.png"
            assert path.is_file()
            assert path.stat().st_size > 0

        # Image dimensions must match the manifest camera intrinsics.
        from PIL import Image

        img = Image.open(cam_dir / "frame_0000.png")
        assert img.size == (cam["width"], cam["height"])


def test_export_overlay_frames_occluded_camera_differs(tmp_path: Path) -> None:
    """Camera 1 has occluded frames in the smoke scene; its frames must still be
    written and differ from a fully blank canvas (they contain a marker)."""
    pytest.importorskip("PIL")
    manifest = _smoke_manifest()
    out_dir = tmp_path / "frames"
    export_overlay(manifest, out_dir, fmt="frames")

    cam1_dir = out_dir / "cam1"
    # pick a frame in the middle of the occlusion window (frames 3-7).
    occluded_path = cam1_dir / "frame_0005.png"
    assert occluded_path.is_file()

    from PIL import Image

    img = Image.open(occluded_path)
    # A non-blank frame has at least some non-white pixels.
    assert any(p != (255, 255, 255) for p in img.get_flattened_data())


def test_export_overlay_mp4_writes_expected_files(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    pytest.importorskip("PIL")
    manifest = _smoke_manifest()
    out_dir = tmp_path / "videos"
    export_overlay(manifest, out_dir, fmt="mp4")

    for cam in manifest["cameras"]:
        path = out_dir / f"cam{cam['id']}.mp4"
        assert path.is_file()
        assert path.stat().st_size > 0


def test_export_overlay_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fmt must be 'frames' or 'mp4'"):
        export_overlay(_smoke_manifest(), tmp_path, fmt="gif")  # type: ignore[arg-type]


def test_draw_dashed_line_handles_zero_length() -> None:
    """The dashed-line helper must not divide by zero for coincident points."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_dashed_line(draw, (5.0, 5.0), (5.0, 5.0), fill=(0, 0, 0), width=1)
    # No exception and image stays blank.
    assert all(p == (255, 255, 255) for p in img.get_flattened_data())

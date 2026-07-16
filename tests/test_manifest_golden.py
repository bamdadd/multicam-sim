"""Byte-golden safety net for the typed Manifest.

The manifest JSON is the wire contract with multicam-occlusion (its
``triangulate_dlt`` parses this). The typed :class:`Manifest.to_json` MUST stay
byte-identical to the historical ``json.dumps(..., indent=2, allow_nan=False)``
output. These fixtures were captured from the pre-refactor ``build_manifest`` for
the smoke, MTMC, and assembly scenes; regenerate them deliberately (and review
the diff) only when the contract is intentionally changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from multicam_sim import build_manifest, build_mtmc_scene, build_smoke_scene
from multicam_sim.scene import Scene

_FIXTURES = Path(__file__).parent / "fixtures" / "manifest_golden"


def _assembly_scene() -> Scene:
    examples = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples))
    import assembly_station  # noqa: PLC0415

    return assembly_station.build_scene()


@pytest.mark.parametrize(
    ("name", "scene_factory"),
    [
        ("smoke", build_smoke_scene),
        ("mtmc", build_mtmc_scene),
        ("assembly", _assembly_scene),
    ],
)
def test_manifest_json_is_byte_identical_to_golden(name: str, scene_factory: object) -> None:
    scene = scene_factory()  # type: ignore[operator]
    got = build_manifest(scene).to_json()
    ref = (_FIXTURES / f"{name}.json").read_text()
    assert got == ref, f"{name} manifest JSON drifted from the golden fixture"


def test_optional_fields_omitted_exactly_as_before() -> None:
    """edges present only for skeletoned entities; topology only when declared;
    occ_frac always present."""
    smoke = build_manifest(build_smoke_scene()).to_json()
    mtmc = build_manifest(build_mtmc_scene()).to_json()
    # smoke: no topology, plain object has no edges; occ_frac present
    assert '"topology"' not in smoke
    assert '"edges"' not in smoke
    assert '"occ_frac"' in smoke
    # mtmc: topology present
    assert '"topology"' in mtmc

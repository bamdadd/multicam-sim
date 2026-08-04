"""Print public-safe coverage metrics for a multicam-sim manifest as JSON.

Usage::

    uv run python scripts/coverage_metrics.py path/to/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multicam_sim.coverage import compute_coverage_metrics
from multicam_sim.manifest import Manifest


def main() -> None:
    """Load a manifest and print its coverage report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="multicam-sim manifest JSON")
    args = parser.parse_args()

    manifest = Manifest.model_validate_json(args.manifest.read_text())
    report = compute_coverage_metrics(manifest)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

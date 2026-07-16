# Contributing to multicam-sim

Thanks for contributing — external PRs are welcome and reviewed quickly.

## Before you push

Run: `pre-commit install` (or `uv run ruff format .` before pushing) — CI enforces
`ruff format` and will fail otherwise.

The full CI gate that must pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

## Where to start

Browse the [good first issues](https://github.com/bamdadd/multicam-sim/labels/good%20first%20issue)
and [help wanted](https://github.com/bamdadd/multicam-sim/labels/help%20wanted) — each
issue states the outcome, acceptance criteria, and the files to touch. `DESIGN.md` is
the contract of record (camera convention + manifest schema); please keep the manifest
JSON byte-stable and the models typed (no loose dicts).

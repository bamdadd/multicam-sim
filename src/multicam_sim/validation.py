"""Manifest schema validation — delegates to the canonical pydantic Manifest.

The typed :class:`multicam_sim.manifest.Manifest` model carries the schema and
its validators; this module is a thin wrapper that validates a loaded dict
against it. The Manifest model is the single source of truth — there is no
duplicate schema here.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .manifest import Manifest


def _format_error(error: Any) -> str:
    """Format a pydantic ValidationError entry as a dotted path with a message."""
    location = ".".join(str(part) for part in error["loc"])
    return f"manifest validation failed at {location}: {error['msg']}"


def validate_manifest(data: dict[str, Any]) -> Manifest:
    """Validate that ``data`` conforms to the manifest schema and return it typed.

    The pydantic :class:`Manifest` model is the single schema source of truth;
    this delegates to it and returns the validated model (no loose dict escapes).

    Raises:
        ValueError: with the offending path and problem if validation fails.
    """
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_format_error(exc.errors()[0])) from exc

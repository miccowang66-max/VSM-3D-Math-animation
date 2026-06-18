"""
src/utils.py — Shared utility helpers.
"""

import json

# Re-export for convenience; primary definition lives in data_gen.py
from src.data_gen import kernel_lift_z  # noqa: F401


def to_json_compact(data: dict) -> str:
    """Serialize dict to compact JSON (no indentation, minimal separators)."""
    return json.dumps(data, indent=None, separators=(",", ":"))

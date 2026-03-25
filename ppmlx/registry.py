"""Built-in model registry shipped with ppmlx.

The registry provides curated aliases for popular mlx-community models,
sourced from https://huggingface.co/mlx-community. It can be disabled
via config.toml ([registry] enabled = false) or PPMLX_REGISTRY_ENABLED=0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DATA_FILE = Path(__file__).parent / "registry_data.json"
_cache: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        with open(_DATA_FILE) as f:
            _cache = json.load(f)
    return _cache


def registry_meta() -> dict[str, Any]:
    """Return registry metadata (version, updated, source)."""
    data = _load()
    return {
        "version": data.get("version", 0),
        "updated": data.get("updated", "unknown"),
        "source": data.get("source", ""),
        "count": len(data.get("models", {})),
    }


def registry_aliases() -> dict[str, str]:
    """Return {alias: repo_id} for all registry models."""
    data = _load()
    return {alias: entry["repo_id"] for alias, entry in data.get("models", {}).items()}


def registry_entries() -> dict[str, dict[str, Any]]:
    """Return the full registry entries {alias: {repo_id, params_b, size_gb, ...}}."""
    return dict(_load().get("models", {}))


def registry_lookup(alias: str) -> dict[str, Any] | None:
    """Look up a single alias. Returns the entry dict or None."""
    return _load().get("models", {}).get(alias)

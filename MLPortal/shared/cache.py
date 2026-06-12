"""Per-stage result cache — persists computed results across Streamlit restarts.

Each stage writes a JSON sidecar file alongside session_state.json:
  outputs/<project>/session/<stage>_cache.json

Usage
-----
  from shared.cache import load_stage_cache, save_stage_cache, update_stage_cache

  # On page load (once per session)
  data = load_stage_cache("clinical_insight")   # {} if not found

  # After computing a result
  update_stage_cache("clinical_insight", {"rf_result": result.model_dump()})

  # Replace the whole cache at once
  save_stage_cache("clinical_insight", {...})
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import streamlit as st


def _cache_path(stage_name: str) -> Path | None:
    """Return the cache file path for the active project, or None if no project active."""
    from shared.state import _active_base_dir
    try:
        base = _active_base_dir()
        return base / "session" / f"{stage_name}_cache.json"
    except Exception:
        return None


def load_stage_cache(stage_name: str) -> dict[str, Any]:
    """Load the cache dict for *stage_name*.  Returns {} if not found or unreadable."""
    path = _cache_path(stage_name)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_stage_cache(stage_name: str, data: dict[str, Any]) -> None:
    """Overwrite the cache file with *data* (adds a saved_at timestamp)."""
    path = _cache_path(stage_name)
    if path is None:
        return
    payload = {
        **data,
        "_saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "_stage": stage_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def update_stage_cache(stage_name: str, updates: dict[str, Any]) -> None:
    """Merge *updates* into the existing cache (partial update)."""
    existing = load_stage_cache(stage_name)
    existing.update(updates)
    save_stage_cache(stage_name, existing)


def clear_stage_cache(stage_name: str) -> None:
    """Delete the cache file for *stage_name*."""
    path = _cache_path(stage_name)
    if path and path.exists():
        path.unlink()


def restore_session_state(stage_name: str, key_mapping: dict[str, str] | None = None) -> bool:
    """Restore st.session_state keys from disk cache.

    *key_mapping* maps cache-key → session-state-key.
    If None, cache keys are used directly as session-state keys.
    Returns True if any data was restored.
    """
    cache = load_stage_cache(stage_name)
    if not cache:
        return False

    mapping = key_mapping or {k: k for k in cache if not k.startswith("_")}
    restored = False
    for cache_key, ss_key in mapping.items():
        if cache_key in cache and cache[cache_key] is not None:
            # Only restore if not already set in this session
            if ss_key not in st.session_state:
                st.session_state[ss_key] = cache[cache_key]
                restored = True
    return restored


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for non-standard types."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)

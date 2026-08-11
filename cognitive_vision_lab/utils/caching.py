"""Caching helpers: Streamlit-aware memory caching with disk fallback."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

from cognitive_vision_lab.config import CACHE_DIR


def stable_key(*parts: Any) -> str:
    """Deterministic cache key from arbitrary parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def disk_cache(namespace: str, func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Simple disk cache keyed on args repr. Falls back to computing on miss."""
    key = stable_key(namespace, func.__name__, args, sorted(kwargs.items()))
    path = CACHE_DIR / namespace / f"{key}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    result = func(*args, **kwargs)
    try:
        with open(path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass
    return result


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

"""Small utilities shared by the package: canonical JSON, hashing, stable floats."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

FLOAT_DECIMALS = 6


def _round(value: float) -> float:
    """Round to the hash precision. ``-0.0`` is normalised so equal payloads hash equal."""
    out = round(float(value), FLOAT_DECIMALS)
    return 0.0 if out == 0 else out


def canon(value: Any) -> Any:
    """Recursively normalise a payload for canonical JSON.

    - floats rounded to ``FLOAT_DECIMALS`` (see docs/03 §3)
    - numpy scalars converted to Python scalars
    - dict keys sorted; tuples become lists
    """
    if isinstance(value, dict):
        return {str(k): canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return _round(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return canon(value.tolist())
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canon(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_image(image: np.ndarray) -> str:
    """Hash of shape, dtype and bytes, so a re-render can be checked exactly."""
    arr = np.ascontiguousarray(image)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    """Canonical JSON-ready form (used for result files)."""
    return canon(value)


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

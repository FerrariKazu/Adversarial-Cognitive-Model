"""
Experiment provenance manifests — Agent A.

Every experiment writes a manifest recording exactly what produced it:
code commit, config hash, dataset version, seed, checkpoint hash,
optimizer configuration, timestamp. Refuses silent overwrite of an
existing experiment_id — a second run under the same id must fail
loudly, never silently replace the first run's record (the attribution
discipline the whole plan rests on; MASTER_PLAN Part 3's
identical-columns discipline is only auditable if provenance is
immutable).

The config hash is computed over the CANONICAL serialization of
RHANNXAConfig (noesis_vision.core.schema) — the same bytes
`to_json()` produces — so two runs with the same config always hash
equal, and any field change (including a status-flag flip) changes the
hash. Verified by tests/test_provenance_manifest.py.

Compute accounting hook (MASTER_PLAN Part 6): the manifest carries the
param-count / FLOPs / memory fields Agent A's infrastructure logs
automatically — callers pass them via `extra`; this module never
hand-computes them per experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

MANIFEST_FILENAME = "manifest.json"


def config_sha256(config: Any) -> str:
    """sha256 of the canonical serialization of `config`.

    Accepts a RHANNXAConfig (uses its to_json() — the canonical bytes) or
    any mapping (serialized with sort_keys for determinism).
    """
    if hasattr(config, "to_json"):
        payload = config.to_json()
    else:
        payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    """sha256 of a file's bytes (streamed — checkpoint files are large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    experiment_id: str,
    config: Any,
    extra: Optional[Dict[str, Any]] = None,
    root_dir: str = "runs",
    seed: Optional[int] = None,
    dataset_version: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the provenance manifest for `experiment_id` and return it.

    Fields recorded (per the Agent A contract):
      experiment_id, git_commit, config_sha256, dataset_version, seed,
      checkpoint_sha256, optimizer_config, timestamp_utc — plus any
      `extra` entries (e.g. compute accounting).

    Refuses silent overwrite: if a manifest already exists for this
    experiment_id, raises FileExistsError listing both the existing and
    the attempted config hash. A deliberate re-run must DELETE the old
    manifest explicitly (an audible action), never pass silently.
    """
    if not experiment_id or not isinstance(experiment_id, str):
        raise ValueError("experiment_id must be a non-empty string")

    exp_dir = os.path.join(root_dir, experiment_id)
    manifest_path = os.path.join(exp_dir, MANIFEST_FILENAME)

    manifest: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "git_commit": _current_code_commit(),
        "config_sha256": config_sha256(config),
        "dataset_version": dataset_version,
        "seed": seed,
        "optimizer_config": optimizer_config,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"checkpoint_path does not exist: {checkpoint_path} — a "
                f"manifest must never record a hash for a missing file")
        manifest["checkpoint_path"] = checkpoint_path
        manifest["checkpoint_sha256"] = file_sha256(checkpoint_path)
    if extra:
        overlap = set(extra) & set(manifest)
        if overlap:
            raise ValueError(
                f"extra keys {sorted(overlap)} collide with manifest core "
                f"fields — core provenance is not caller-overridable")
        manifest.update(extra)

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
        raise FileExistsError(
            f"refusing silent overwrite of experiment_id '{experiment_id}': "
            f"manifest exists at {manifest_path} (config_sha256 "
            f"{existing.get('config_sha256')}); attempted config_sha256 "
            f"{manifest['config_sha256']}. Delete the existing manifest "
            f"explicitly to re-run this id — never overwrite provenance "
            f"silently.")

    os.makedirs(exp_dir, exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, manifest_path)
    return manifest


def load_manifest(experiment_id: str, root_dir: str = "runs") -> Dict[str, Any]:
    """Load the manifest for `experiment_id`; raises FileNotFoundError with
    a clear message when it does not exist (never a silent empty dict)."""
    manifest_path = os.path.join(root_dir, experiment_id, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"no manifest for experiment_id '{experiment_id}' at "
            f"{manifest_path}")
    with open(manifest_path, "r") as f:
        return json.load(f)


def manifest_exists(experiment_id: str, root_dir: str = "runs") -> bool:
    return os.path.exists(
        os.path.join(root_dir, experiment_id, MANIFEST_FILENAME))


def _current_code_commit() -> str:
    """git HEAD SHA of the running code ('unknown' if unavailable).

    Same convention as Gen-0's checkpoint_utils.current_code_commit: every
    manifest records the exact code that produced it, so a later audit can
    ask "what did the plan say" and "what code ran" as separate questions.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:
        pass
    return "unknown"

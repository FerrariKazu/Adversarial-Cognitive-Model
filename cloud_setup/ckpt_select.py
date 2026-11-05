"""Checkpoint selection, validation, and sync scoping for matched replication runs.

The selector never trusts a filename alone. It loads checkpoint metadata, checks
the internal run identity and provenance, rejects weights-only/partial states,
and chooses the highest valid epoch across HuggingFace repositories.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_KINDS = ("rolling", "final", "best_clean", "best_robust")


class CandidateError(ValueError):
    """Raised when a checkpoint candidate is missing required internal fields."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(path: str | os.PathLike[str]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise CandidateError(f"candidate is missing: {p}")
    if p.name.endswith(".part"):
        raise CandidateError(f"partial download is not a candidate: {p}")
    try:
        import torch
    except Exception as exc:
        raise CandidateError(f"torch unavailable while reading {p}: {exc}") from exc
    try:
        state = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CandidateError(f"candidate is unreadable or partial: {p}: {exc}") from exc
    if not isinstance(state, dict):
        raise CandidateError(f"candidate is not a checkpoint dict: {p}")
    return state


def _field(state: Mapping[str, Any], name: str) -> Any:
    if name not in state:
        raise CandidateError(f"missing internal field {name!r}")
    return state[name]


def _require_commit(value: Any, name: str, expected: Optional[str] = None) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise CandidateError(f"{name} is not a lowercase git SHA")
    if expected is not None and not expected.startswith(value):
        raise CandidateError(f"{name} mismatch: expected {expected}, got {value}")
    return value


def _validate_optimizer(value: Any) -> None:
    if not isinstance(value, dict):
        raise CandidateError("optimizer state is not a dict")
    opt_state = value.get("state")
    if not isinstance(opt_state, dict) or not opt_state:
        raise CandidateError("optimizer state is empty (weights-only or reset checkpoint)")
    groups = value.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise CandidateError("optimizer param_groups are missing or empty")


def _validate_scheduler(value: Any) -> None:
    if not isinstance(value, dict) or not value:
        raise CandidateError("scheduler state is missing or empty")


def validate_candidate(
    state: Mapping[str, Any],
    *,
    expected_run: str,
    expected_phase: str,
    expected_commit: Optional[str] = None,
    expected_seed: Optional[int] = None,
    require_run_name: bool = True,
    require_provenance: bool = True,
) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        raise CandidateError("checkpoint metadata is not a mapping")

    if require_run_name:
        run_name = _field(state, "run_name")
        if run_name != expected_run:
            raise CandidateError(f"run_name mismatch: expected {expected_run}, got {run_name}")
    elif "run_name" in state and state["run_name"] != expected_run:
        raise CandidateError(f"run_name mismatch: expected {expected_run}, got {state['run_name']}")

    phase = _field(state, "phase")
    if phase != expected_phase:
        raise CandidateError(f"phase mismatch: expected {expected_phase}, got {phase}")

    epoch = _field(state, "epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CandidateError(f"epoch must be an integer >= 1, got {epoch!r}")

    if expected_seed is not None:
        seed = _field(state, "seed")
        if seed != expected_seed:
            raise CandidateError(f"seed mismatch: expected {expected_seed}, got {seed!r}")

    _validate_optimizer(_field(state, "optimizer"))
    _validate_scheduler(_field(state, "scheduler"))

    if require_provenance:
        _require_commit(_field(state, "commit_hash"), "commit_hash", expected_commit)
        source_hashes = _field(state, "source_hashes")
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise CandidateError("source_hashes is missing or empty")
        bad = [k for k, v in source_hashes.items() if not isinstance(v, str) or not SHA256_RE.fullmatch(v)]
        if bad:
            raise CandidateError(f"source_hashes contains invalid entries: {bad[:3]}")
    elif expected_commit is not None and "commit_hash" in state:
        _require_commit(state["commit_hash"], "commit_hash", expected_commit)

    return dict(state)


def validate_reuse_candidate(
    state: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_epoch: int = 15,
    expected_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(state, Mapping):
        raise CandidateError("checkpoint metadata is not a mapping")
    if state.get("phase") != expected_phase:
        raise CandidateError(f"phase mismatch: expected {expected_phase}, got {state.get('phase')!r}")
    if state.get("epoch") != expected_epoch:
        raise CandidateError(f"epoch mismatch: expected {expected_epoch}, got {state.get('epoch')!r}")
    _validate_optimizer(_field(state, "optimizer"))
    _validate_scheduler(_field(state, "scheduler"))
    if expected_seed is not None and state.get("seed") not in (None, expected_seed):
        raise CandidateError(f"seed mismatch: expected {expected_seed}, got {state.get('seed')!r}")
    return dict(state)


def select_best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise CandidateError("no valid candidates")
    epochs = []
    for c in candidates:
        state = c.get("metadata", c)
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            raise CandidateError(f"invalid candidate epoch: {epoch!r}")
        epochs.append(epoch)
    max_epoch = max(epochs)
    top = [c for c in candidates if int(c.get("metadata", c).get("epoch")) == max_epoch]
    hashes = set()
    for c in top:
        digest = c.get("sha256")
        if digest is None and c.get("path"):
            digest = sha256_file(c["path"])
        hashes.add(digest)
    if len(hashes) > 1:
        raise CandidateError(f"ambiguous candidates at epoch {max_epoch}: different file hashes")
    return dict(top[0])


def parse_checkpoint_kind(filename: str, run_name: str) -> Optional[str]:
    base = os.path.basename(filename)
    prefix = f"foundation_{run_name}_"
    if not base.startswith(prefix):
        return None
    for kind in CHECKPOINT_KINDS:
        if base == f"{prefix}{kind}.pth":
            return kind
        if re.fullmatch(rf"{re.escape(prefix + kind)}_[0-9a-f]{{6,12}}\.pth", base):
            return kind
    return None


def filename_matches_run(filename: str, run_name: str) -> bool:
    return parse_checkpoint_kind(filename, run_name) is not None


def files_for_active_runs(
    filenames: Iterable[str],
    active_runs: Iterable[str],
    *,
    kinds: Sequence[str] = CHECKPOINT_KINDS,
) -> List[str]:
    allowed = set(kinds)
    runs = tuple(active_runs)
    out: List[str] = []
    for filename in filenames:
        for run_name in runs:
            kind = parse_checkpoint_kind(filename, run_name)
            if kind in allowed:
                out.append(filename)
                break
    return sorted(set(out))


def can_overwrite_remote(local_epoch: int, remote_epoch: Optional[int]) -> bool:
    if remote_epoch is None:
        return True
    return local_epoch >= remote_epoch

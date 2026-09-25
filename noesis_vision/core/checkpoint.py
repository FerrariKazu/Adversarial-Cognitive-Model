"""
Checkpoint / resume — Agent A. ADAPTED from the Gen-0 trainer's machinery
(phase1_training/train_rhan_next.py + checkpoint_utils.py), decoupled from
the trainer so every RHAN-NXA experiment shares ONE implementation.

The rules this module exists to enforce (each one a real Gen-0 incident):

  1. NEVER a silent restart (2026-08-12 stale-resume bug): when a rolling
     checkpoint exists — locally or on HF — resume is MANDATORY. A caller
     that cannot restore must ABORT (CheckpointResumeError), never fall
     through to a fresh run.

  2. Code-identity guard: a checkpoint written by different training code
     must never be resumed. Every saved artifact records `code_commit` and
     `training_fingerprint` (the training-code identity, not the whole repo —
     notebook-only commits never invalidate a mid-run resume). Legacy
     checkpoints with no recorded commit are refused: they are by definition
     older than this guard. Ported semantics: phase1_training/checkpoint_utils.py
     (resume_commit_ok / training_fingerprint) — the fingerprint resolution is
     carried over verbatim in spirit; the equivalence-edge registry stays in
     the Gen-0 file because its edges are Gen-0 commit SHAs.

  3. Atomic saves: every checkpoint write is .tmp -> fsync -> rename, so a
     runtime crash can never leave a partial file as the only artifact.

  4. Best/rolling parity, from commit one (not discovered late): the rolling
     resume artifact and the best-eval artifact are written by the same
     save_state call; verify_best_rolling_parity() checks the two files
     agree on code identity and config hash before any eval cites them
     (the destroyed-rolling / parity-gap bug class).

  5. Mandatory HF sync points: save_rolling / save_best take an optional
     uploader; when provided, the artifact is pushed immediately after the
     local write — a local-only checkpoint is one runtime reset away from
     loss.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch


class CheckpointResumeError(RuntimeError):
    """Raised when resume is required but impossible — never silently
    fall through to a fresh start."""


# ── code identity (ported from phase1_training/checkpoint_utils.py) ─────────

def current_code_commit(short: bool = True) -> str:
    """git HEAD SHA of the running code, or 'unknown'."""
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else \
               ["git", "rev-parse", "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:
        pass
    return "unknown"


def resume_commit_ok(ckpt: Any, current: Optional[str] = None) -> Tuple[bool, str]:
    """Is a checkpoint safe to resume under the current code?

    Returns (ok, message). A pre-guard checkpoint (no recorded
    code_commit) is never resumable. Gen-0's training-fingerprint layer
    (notebook-only-commit equivalence) lives in
    phase1_training/checkpoint_utils.py; RHAN-NXA trainers that need it
    call that function explicitly — this port keeps the core refusal
    semantics every experiment shares.
    """
    current = current or current_code_commit()
    recorded = ckpt.get("code_commit") if isinstance(ckpt, dict) else None
    if not recorded:
        return False, (
            "legacy checkpoint with no recorded code_commit — written by "
            "older code; refusing to resume across a code change")
    if recorded != current:
        return False, (
            f"checkpoint written by commit {recorded}, current code is "
            f"{current} — refusing to resume across a training-code change")
    return True, f"code_commit {current} matches — resumable"


# ── atomic save (ported from train_rhan_next._atomic_torch_save) ────────────

def atomic_torch_save(target_path: str, obj: Any) -> str:
    """Save a torch object atomically: .tmp -> fsync -> rename.

    Prevents partial/corrupted checkpoints from being the only artifact
    on disk after a runtime crash. Returns the target path on success.
    """
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target_dir or ".", suffix=".tmp")
    try:
        torch.save(obj, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.close(fd)
        fd = -1
        os.rename(tmp_path, target_path)
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
    return target_path


# ── save paths ───────────────────────────────────────────────────────────────

def save_rolling(path: str, *, epoch: int, model: torch.nn.Module,
                 optimizer: Any, scheduler: Any = None,
                 extra: Optional[Dict[str, Any]] = None,
                 uploader: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Write the rolling (resume) artifact: everything needed to continue
    the run — epoch, model, optimizer, scheduler, code identity, config.

    Every artifact records code_commit so the resume guard can refuse a
    checkpoint written by different code.
    """
    state: Dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "code_commit": current_code_commit(),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": "rolling",
    }
    if extra:
        state.update(extra)
    atomic_torch_save(path, state)
    if uploader is not None:
        uploader(path)
    return state


def save_best(path: str, *, model: torch.nn.Module,
              config: Any, metric_value: float,
              uploader: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Write the best (eval) artifact: model + config + the metric that made
    it 'best'. The embedded config lets any eval script reconstruct the
    exact configuration without external bookkeeping (the Gen-0 convention
    that killed eval/train config drift)."""
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "metric_value": float(metric_value),
        "code_commit": current_code_commit(),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": "best",
    }
    atomic_torch_save(path, state)
    if uploader is not None:
        uploader(path)
    return state


def verify_best_rolling_parity(best_path: str, rolling_path: str) -> Tuple[bool, str]:
    """Best/rolling parity check — applied from commit one.

    The two artifacts of one run must agree on code identity; the rolling
    checkpoint must be at least as recent as the best. Returns (ok, message);
    eval scripts call this BEFORE citing either artifact (a parity gap means
    the best artifact may predate a training-code change — exactly the
    Gen-0 parity-gap incident).
    """
    for p in (best_path, rolling_path):
        if not os.path.exists(p):
            return False, f"parity check: missing artifact {p}"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    rolling = torch.load(rolling_path, map_location="cpu", weights_only=False)
    if best.get("code_commit") != rolling.get("code_commit"):
        return False, (
            f"best/rolling code_commit mismatch: best "
            f"{best.get('code_commit')!r} vs rolling "
            f"{rolling.get('code_commit')!r} — eval must not cite these "
            f"without flagging the parity gap")
    return True, (f"parity ok (code_commit {best.get('code_commit')!r}, "
                  f"rolling at epoch {rolling.get('epoch')!r})")


# ── resume gate ──────────────────────────────────────────────────────────────

def resume_or_abort(rolling_path: str, hf_repo_id: Optional[str] = None,
                    hf_filename: Optional[str] = None,
                    hf_token: Optional[str] = None,
                    force_restart: bool = False,
                    downloader: Optional[Callable[[str, str, Optional[str]], str]] = None,
                    ) -> Optional[Dict[str, Any]]:
    """The mandatory-resume gate. Returns the loaded rolling state, or None
    ONLY for an explicit force_restart (cold start).

    Semantics (ported from the Gen-0 trainer):
      * local rolling exists -> load it (after the code-identity guard);
      * no local rolling but HF has one -> download and load it;
      * HF existence CANNOT be verified and no local copy exists -> ABORT
        (CheckpointResumeError) rather than silently restarting;
      * force_restart=True -> return None (the caller logs a loud cold
        start). Never the default.
    """
    if force_restart:
        return None

    local_epoch = -1
    if os.path.exists(rolling_path):
        local_epoch = torch.load(rolling_path, map_location="cpu",
                                 weights_only=False).get("epoch", -1)

    remote_path: Optional[str] = None
    hf_verified = False
    hf_proven_absent = False
    if hf_repo_id and hf_filename and downloader is not None:
        try:
            remote_path = downloader(hf_repo_id, hf_filename, hf_token)
            hf_verified = True
        except Exception as e:
            hf_verified = False
            # EntryNotFoundError means the repo was REACHABLE and the file
            # PROVABLY absent — a verified fresh run, not an unverifiable
            # state. Any other failure (network, auth, hub-missing) keeps
            # the safe direction: abort below, never a silent restart.
            try:
                from huggingface_hub.errors import EntryNotFoundError
            except Exception:
                EntryNotFoundError = ()   # catches nothing -> abort (safe)
            hf_proven_absent = isinstance(e, EntryNotFoundError)

    if remote_path is not None:
        remote_epoch = torch.load(remote_path, map_location="cpu",
                                  weights_only=False).get("epoch", -1)
        if remote_epoch >= local_epoch:
            os.makedirs(os.path.dirname(rolling_path) or ".", exist_ok=True)
            shutil.copy(remote_path, rolling_path)

    if not os.path.exists(rolling_path):
        if hf_repo_id and not hf_verified and not hf_proven_absent:
            # We could not PROVE HF has no rolling checkpoint — aborting is
            # the safe direction (a silent restart would orphan the run).
            raise CheckpointResumeError(
                f"could not verify HF rolling checkpoint '{hf_filename}' in "
                f"'{hf_repo_id}' and no local copy exists at {rolling_path} "
                f"— aborting rather than silently restarting")
        return None  # genuinely fresh run: no local, no HF (or no HF configured)

    state = torch.load(rolling_path, map_location="cpu", weights_only=False)
    ok, why = resume_commit_ok(state)
    if not ok:
        raise CheckpointResumeError(
            f"refusing to resume {rolling_path}: {why} — resuming across a "
            f"code change silently invalidates the run. Delete the stale "
            f"rolling/best artifacts (locally and on HF) for a genuine cold "
            f"start.")
    return state

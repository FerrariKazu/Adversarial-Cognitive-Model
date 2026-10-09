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


def resume_commit_ok(
    ckpt: Any,
    current: Optional[str] = None,
    *,
    require_experiment_class: bool = True,
    allowed_experiment_configs: Optional[set] = None,
) -> Tuple[bool, str]:
    """Is a checkpoint safe to resume under the current code?

    Returns (ok, message). A pre-guard checkpoint (no recorded
    code_commit) is never resumable.

    This is the RHAN-NXA fingerprint-aware resume gate. It extends the
    original commit-SHA comparison with the Gen-0 training-fingerprint
    semantics (phase1_training/checkpoint_utils.py): notebook-only /
    platform-only commits do not invalidate a mid-run resume, while any
    change to the training-math ancestors still refuses it.

    For RHAN-NXA foundation runs we ALSO verify that the checkpoint's
    embedded config is compatible with the intended experiment class.
    That prevents resuming across config drift (e.g. a stale manifest vs
    a checkpoint that recorded a different pgd_steps / clean_only / recipe).

    IMPORTANT: this gate does NOT authorize arbitrary cross-commit resume.
    It authorizes resume only when (a) the training fingerprint matches
    AND (b) the checkpoint's embedded config is compatible with the
    experiment class the caller declares.
    """
    from phase1_training.checkpoint_utils import (
        resume_commit_ok as _g0_resume_commit_ok,
        training_fingerprint,
    )

    if current is None:
        current = current_code_commit()

    # 1. Legacy guard: no recorded code_commit == no resume.
    recorded_code_commit = ckpt.get("code_commit") if isinstance(ckpt, dict) else None
    if not recorded_code_commit:
        return False, (
            "legacy checkpoint with no recorded code_commit — written by "
            "older code; refusing to resume across a code change")

    # 2. Training-fingerprint identity (Gen-0 semantics).
    fp_recorded = ckpt.get("training_fingerprint") or training_fingerprint(recorded_code_commit)
    fp_current = training_fingerprint(current)
    if fp_recorded != fp_current:
        return False, (
            f"checkpoint written by commit {recorded_code_commit} (training fingerprint "
            f"{fp_recorded}), current code is {current} (training fingerprint "
            f"{fp_current}) — refusing to resume across a training-code change")

    # 3. Optional experiment-class guard for RHAN-NXA foundation runs.
    # The checkpoint's authoritative config is the embedded config in the BEST
    # checkpoint, not any stale manifest. When this gate is given a best
    # checkpoint (which carries its own embedded config) AND a declared experiment
    # class, it verifies compatibility and refuses config drift.
    #
    # IMPORTANT: for a ROLLING checkpoint this function does NOT demand an embedded
    # config. The rolling artifact intentionally carries the resume state, not the
    # authoritative config; the authoritative config lives in the BEST checkpoint.
    # The correct recovery sequence is therefore:
    #   (1) verify BEST checkpoint embedded config matches declared experiment class,
    #   (2) resume the ROLLING checkpoint under the training-fingerprint match.
    if require_experiment_class and allowed_experiment_configs is not None:
        ckpt_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if isinstance(ckpt_cfg, dict):
            # Best checkpoint path: verify embedded config compatibility.
            cfg_ok = False
            reason_parts = []
            for sig in allowed_experiment_configs:
                if not isinstance(sig, dict):
                    continue
                mismatches = []
                for key in ("clean_only", "recipe_version", "seed", "w_trades", "pgd_steps"):
                    if key in sig and ckpt_cfg.get(key) != sig[key]:
                        mismatches.append(f"{key}: checkpoint={ckpt_cfg.get(key)!r}, expected={sig[key]!r}")
                if not mismatches:
                    cfg_ok = True
                    reason_parts.append("best-checkpoint config compatible with declared experiment class")
                    break
                else:
                    reason_parts.append(
                        f"best-checkpoint config incompatible with a declared experiment signature "
                        f"({sig.get('recipe_version')!r}): " + "; ".join(mismatches)
                    )
            if not cfg_ok:
                return False, (
                    "refusing to resume: " + " || ".join(reason_parts) +
                    " — resuming across config drift silently invalidates the run."
                )
        else:
            # Rolling checkpoint path: cannot verify config here; the caller must have
            # already verified the BEST checkpoint config above. Fail closed if the
            # caller did not declare an experiment class.
            return False, (
                "rolling checkpoint has no embedded config; the caller must verify the "
                "BEST checkpoint's embedded config before resuming. Refusing to resume."
            )

    # Convenience message when we reached here via the fingerprint path only.
    if recorded_code_commit == current:
        return True, f"code_commit {current} matches — resumable"
    return True, (
        f"commit {current} changed notebooks only (training fingerprint "
        f"{fp_current} == checkpoint's {fp_recorded}) — resumable"
    )

def _checkpoint_embedded_config_hash(cfg: Any) -> Optional[str]:
    """Hash the RECONCILED authoritative experiment config.

    IMPORTANT provenance decision (recovery/rhan-next-resume-gateway):
    the experiment config hash MUST NOT include runtime/deployment fields
    (hf_token, ckpt_dir, report_dir, runs_dir, data_root). Those are
    environment/path fields, not experiment math, and they differ between
    the Kaggle runtime, the Colab runtime, and local development machines.

    The canonical experiment payload is the embedded config MINUS those five
    fields, serialized with sort_keys=True. This is the ONE canonical config
    hash used for the RHAN-NXA foundation recovery, and it is the same hash
    recorded in the reconciled manifest for foundation_backbone_only.

    NOTE: provenance.config_sha256 still exists and still hashes the FULL
    config (including those five fields) for its own existing manifest/stale
    provenance contexts. That is intentionally NOT the same number. If a
    manifest or report wants the experiment config hash, it must use this
    function, not provenance.config_sha256.
    """
    import hashlib
    import json as _json
    if not isinstance(cfg, dict):
        return None
    payload = dict(cfg)
    for _k in ("hf_token", "ckpt_dir", "report_dir", "runs_dir", "data_root"):
        payload.pop(_k, None)
    return hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()


def canonical_experiment_config_hash(cfg: Any) -> Optional[str]:
    """Public alias for the reconciled experiment config hash.

    This is the single canonical hash the RHAN-NXA foundation recovery
    asserts. Both noesis_vision/core/checkpoint.py and the recovery Kaggle
    pre-flight cell use this function so the same embedded config always
    produces the same hash regardless of which module computes it.
    """
    return _checkpoint_embedded_config_hash(cfg)


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
            # The downloader may stage the file INSIDE checkpoints/ (same
            # path) — copying onto itself is a SameFileError, not a resume
            # problem; skip the copy when source == destination.
            if os.path.abspath(remote_path) != os.path.abspath(rolling_path):
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

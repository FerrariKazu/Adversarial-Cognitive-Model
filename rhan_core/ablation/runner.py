"""
runner.py — resolves any ABLATION_MATRIX key to a ready-to-use
(config, checkpoint_path, ckpt_name) triple for training or eval.

This is what lets the eventual matched eval loop through all four entries with
one script instead of four bespoke --ckpt-specs invocations typed by hand:
phase2_attacks/eval_rhan.py --ablation-matrix [keys...] builds the specs from
this registry automatically.

No torch import at module import time — heavy imports stay lazy so this is
safe to import from notebooks and tests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from rhan_core.ablation import matrix as _matrix
from rhan_core.config.pillar_config import RHANNextConfig


def repo_root() -> str:
    """Repository root (rhan_core/ablation/runner.py -> repo root)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def resolve(key: str) -> Dict[str, Any]:
    """Resolve a matrix key to a ready-to-use entry dict.

    Returns the entry with `checkpoint` (repo-relative path as declared),
    `label`, `arch`, `config`, `status`, `ckpt_name`. `checkpoint` may be
    None (not yet trained). Raises KeyError on unknown keys.
    """
    entry = _matrix.get_entry(key)
    resolved = dict(entry)
    # ckpt_name == label for the trainable RHANNext entries; A has none.
    resolved["ckpt_name"] = (
        entry["label"] if entry["arch"] == "next" and entry["status"] != _matrix.VALIDATED
        else None)
    return resolved


def resolve_checkpoint_path(key: str) -> Optional[str]:
    """Absolute checkpoint path for a key, or None if not yet trained."""
    entry = _matrix.get_entry(key)
    rel = entry.get("checkpoint")
    if not rel:
        return None
    p = rel if os.path.isabs(rel) else os.path.join(repo_root(), rel)
    return p


def checkpoint_exists(key: str) -> bool:
    p = resolve_checkpoint_path(key)
    return bool(p and os.path.exists(p))


def train_command(key: str, extra_args: Optional[List[str]] = None,
                  ckpt_name: Optional[str] = None) -> List[str]:
    """Build the train_rhan_next.py argv for a trainable entry.

    Only entries with status PENDING may train (C this round; D is
    SCAFFOLDED_NOT_RUN and raises). The AIS-v1 halting-only variant is
    expressed with --no-ais-precision-recon (never --enable-ais alone).

    `ckpt_name` overrides the --ckpt-name (the smoke protocol runs the SAME
    matrix config under a _smoke artifact name). Returns argv WITHOUT shell
    quoting — join with shlex.join() for display.
    """
    entry = _matrix.get_entry(key)
    if entry["status"] not in _matrix.TRAINABLE_STATUSES:
        raise ValueError(
            f"{key} has status {entry['status']!r} — only "
            f"{_matrix.TRAINABLE_STATUSES} entries may train. "
            f"({key} is deliberately dormant this round.)")
    if entry["arch"] != "next":
        raise ValueError(f"{key} is not a RHANNext entry (arch "
                         f"{entry['arch']!r}) — nothing to train.")

    cfg: RHANNextConfig = entry["config"]
    argv = ["python3", "phase1_training/train_rhan_next.py"]
    if cfg.enable_ais:
        argv.append("--enable-ais")
        if not cfg.ais_precision_recon_enabled:
            argv.append("--no-ais-precision-recon")   # AIS-v1 halting-only
        if not cfg.ais_halt_enabled:
            argv.append("--no-ais-halting")
    if cfg.enable_hpc:
        argv += ["--enable-hpc", "--hpc-num-levels", str(cfg.hpc_num_levels),
                 "--w-hpc", str(cfg.hpc_error_weight)]
    argv += ["--ckpt-name", ckpt_name or entry["label"]]
    argv += list(extra_args or [])
    return argv


def eval_specs(keys: Optional[List[str]] = None,
               require_present: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build --ckpt-specs-style dicts from the registry.

    Args:
        keys: explicit key list (in order). None = every matrix entry.
        require_present: also require the checkpoint file to exist locally
            (used by the eval entrypoint for the PENDING-with-checkpoint rule).

    Returns:
        (specs, skipped): specs are {label, path, arch, freeze}; skipped is a
        list of human-readable reasons for entries that were excluded.
    """
    if keys is None:
        keys = _matrix.matrix_keys()
    specs: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for key in keys:
        entry = _matrix.get_entry(key)          # KeyError on unknown keys
        present = checkpoint_exists(key)
        eligible = _matrix.is_eval_eligible(key, present)
        if not eligible:
            if entry["status"] == _matrix.SCAFFOLDED_NOT_RUN:
                skipped.append(f"{key}: SCAFFOLDED_NOT_RUN — deliberately not "
                               "evaluated this round")
            else:
                skipped.append(f"{key}: status {entry['status']!r} and "
                               "checkpoint not present yet — train it first")
            continue
        p = resolve_checkpoint_path(key)
        if not p:
            skipped.append(f"{key}: no checkpoint path declared")
            continue
        if require_present and not os.path.exists(p):
            skipped.append(f"{key}: checkpoint missing locally "
                           f"({os.path.basename(p)})")
            continue
        specs.append({"label": entry["label"], "path": p,
                      "arch": entry["arch"], "freeze": False})
    return specs, skipped

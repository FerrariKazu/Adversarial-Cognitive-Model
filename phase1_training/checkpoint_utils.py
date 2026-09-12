import os
import subprocess
import sys
import types
from functools import lru_cache

import torch


def compat_load(path, map_location=None, **kwargs):
    _ensure_serialization_proxy()
    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def compat_save(obj, path, **kwargs):
    _ensure_serialization_proxy()
    return torch.save(obj, path, **kwargs)


def _ensure_serialization_proxy():
    if 'torch.utils.serialization' in sys.modules:
        return

    class _Attrs:
        pass

    load = _Attrs()
    load.calculate_storage_offsets = True
    load.mmap = False
    load.mmap_flags = 0
    load.endianness = 'little'

    save = _Attrs()
    save.compute_crc32 = False
    save.storage_alignment = 64
    save.use_pinned_memory_for_d2h = False

    config_mod = types.ModuleType('torch.utils.serialization.config')
    config_mod.__path__ = []
    config_mod.__package__ = 'torch.utils.serialization'
    config_mod.load = load
    config_mod.save = save
    sys.modules['torch.utils.serialization.config'] = config_mod

    parent_mod = types.ModuleType('torch.utils.serialization')
    parent_mod.__path__ = []
    parent_mod.__package__ = 'torch.utils'
    parent_mod.config = config_mod
    sys.modules['torch.utils.serialization'] = parent_mod


def current_code_commit(short=True):
    """git HEAD SHA of the code that is running, or 'unknown' if unavailable.

    Stored in every checkpoint ('code_commit') and compared at resume time so
    a checkpoint written by DIFFERENT code is never silently resumed — the
    2026-08-12 stale-resume bug resumed old dead-head weights at epoch 12 and
    produced a meaningless Stage 2 health-gate verdict.

    NOTE: HEAD-only identity does NOT capture an uncommitted dirty working
    tree. That is intentional and acceptable here because every training
    session runs `git reset --hard origin/...` before training, so HEAD == the
    exact pushed code that produced the checkpoint.
    """
    try:
        args = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
        out = subprocess.run(args, capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:
        pass
    return 'unknown'


# ── Training fingerprint (2026-09-12) ────────────────────────────────────────
# The raw HEAD-commit guard refused a legitimate resume whenever ANY commit
# landed between training sessions, even when the commit touched only the
# platform notebooks (the SBR-2 incident: checkpoint written by 54c3984, next
# session checked out 47d30b2 which changed cloud_setup/*.py only — the guard
# demanded deleting ~6 hours of real SBR-2 progress).
#
# The fingerprint is the identity of the TRAINING CODE, not the whole repo:
# a commit that changes only cloud_setup/ or docs/ resolves to its nearest
# training-relevant ancestor, so checkpoints written before it stay resumable.
#
# PROTOCOL when landing a commit: if it touches ONLY cloud_setup/*.py,
# docs/, scripts/, or report/ — add an edge here mapping the new commit to
# its nearest training-relevant ancestor. NEVER add an edge for a commit
# that touches phase1_training/ or rhan_core/ — those change training math
# and MUST invalidate older rolling checkpoints. The ONLY entries here are
# dated equivalence grants for math-adjacent edits PROVEN training-neutral,
# so a live run is never orphaned by its own repair commit.
FINGERPRINT_EQUIVALENCE = {
    # 3d0b0e4 (2026-09-12 guard repair) ≡ 8abb343: the repair touched
    # train_rhan_next.py (resume barrier, fingerprint stamping, DDP launch
    # plumbing) but no training math — no loss/optimizer/curriculum/model
    # change (verified by diff). Without this grant the guard's own fix
    # would orphan every run written by 54c3984/47d30b2 (e.g. the live
    # SBR-2 rolling checkpoint) — the exact failure mode it repairs.
    '3d0b0e4': '8abb343',
}

# Directories whose contents define the training fingerprint. checkpoint_utils
# is excluded: it is resume plumbing, and including it would make every
# guard change invalidate live runs (self-orphaning).
_TRAINING_PATHS = (
    'phase1_training/',
    'rhan_core/',
    ':(exclude)phase1_training/checkpoint_utils.py',
)


@lru_cache(maxsize=None)
def _cached_fp(commit, _stamp):
    return _training_fingerprint_uncached(commit)


def _git_fp_once(commit):
    """Nearest ancestor of `commit` that touched the training-math paths."""
    try:
        out = subprocess.run(
            ['git', 'log', '-1', '--format=%h', commit, '--',
             *_TRAINING_PATHS],
            capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:
        pass
    return commit


def _training_fingerprint_uncached(commit):
    """Resolve `commit` to its training fingerprint (equivalence fixpoint).

    The equivalence grant is applied BEFORE each git resolution: a granted
    commit is typically a math-path-toucher itself (its own git resolution
    would return it unchanged), so the grant is what redirects the walk to
    the equivalent ancestor.
    """
    for _ in range(8):
        commit = FINGERPRINT_EQUIVALENCE.get(commit, commit)
        resolved = _git_fp_once(commit)
        if resolved == commit:
            break
        commit = resolved
    return commit


def training_fingerprint(commit=None):
    """Canonical identity of the training code at a given commit.

    Resolved by git history: the nearest ancestor that touched
    phase1_training/ or rhan_core/ (resume plumbing excluded), then fixed
    through FINGERPRINT_EQUIVALENCE. Notebook-only commits therefore never
    change the fingerprint, while any change to the training math moves it
    and invalidates older rolling checkpoints. Unknown commits resolve to
    themselves — they will simply never match an older checkpoint's
    fingerprint, which is the safe direction to fail in.
    """
    commit = commit or current_code_commit()
    _stamp = os.path.getmtime(__file__) if os.path.exists(__file__) else 0.0
    return _cached_fp(commit, _stamp)


def resume_commit_ok(ckpt, current=None):
    """Is a checkpoint safe to resume under the current code?

    Returns (ok: bool, message: str). Compares the TRAINING FINGERPRINT of
    the checkpoint against the current code, so notebook-only commits do not
    spuriously invalidate a mid-run resume, while any change to
    phase1_training/ or rhan_core/ still refuses it. A pre-guard legacy
    checkpoint (no recorded commit) is never resumable.
    """
    current = current or current_code_commit()
    recorded = ckpt.get('code_commit') if isinstance(ckpt, dict) else None
    if not recorded:
        return False, (
            "legacy checkpoint with no recorded code_commit — written by older "
            "code; refusing to resume across a code change")
    fp_recorded = ckpt.get('training_fingerprint') or training_fingerprint(recorded)
    fp_current = training_fingerprint(current)
    if fp_recorded != fp_current:
        return False, (
            f"checkpoint written by commit {recorded} (training fingerprint "
            f"{fp_recorded}), current code is {current} (training fingerprint "
            f"{fp_current}) — refusing to resume across a training-code change")
    if recorded == current:
        return True, f"code_commit {current} matches — resumable"
    return True, (
        f"commit {current} changed notebooks only (training fingerprint "
        f"{fp_current} == checkpoint's {fp_recorded}) — resumable")

import subprocess
import sys
import types

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


def resume_commit_ok(ckpt, current=None):
    """Is a checkpoint safe to resume under the current code?

    Returns (ok: bool, message: str). A checkpoint written by a different git
    commit — or a pre-guard legacy checkpoint with no recorded commit — is NOT
    resumable: resuming across a code change silently invalidates the run.
    """
    current = current or current_code_commit()
    recorded = ckpt.get('code_commit') if isinstance(ckpt, dict) else None
    if not recorded:
        return False, (
            "legacy checkpoint with no recorded code_commit — written by older "
            "code; refusing to resume across a code change")
    if recorded != current:
        return False, (
            f"checkpoint written by commit {recorded}, current code is {current} "
            f"— refusing to resume across a code change")
    return True, f"code_commit {current} matches — resumable"

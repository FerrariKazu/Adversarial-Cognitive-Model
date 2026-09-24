"""
Seed management — Agent A.

Sets torch / numpy / python seeds in one call. Documents (and returns)
which CUDA ops remain non-deterministic so no caller claims full
determinism the stack cannot back — an honest determinism boundary is
part of provenance.

What IS deterministic after set_all_seeds (CUDA >= 10.2 with these
flags): torch.matmul / conv forward+backward kernels routed through
cudnn/cublas deterministic algorithms. What is NOT guaranteed:
torch.nn.functional.interpolate backward, scatter-add /
index_put-accumulate backward, atomicAdd-based kernels generally, and
any torch.compile-generated kernel. Callers must not claim
run-to-run bit equality across machines; per-seed RESULT equivalence
(accuracy within seed noise) is the reproducibility standard, per the
project's 8+-seed discipline.
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch

#: The honest determinism ledger, returned by set_all_seeds so every
#: caller can log it alongside the seed instead of asserting more than
#: is true. Keyed by op class; values state the guarantee.
NONDETERMINISTIC_OPS: Dict[str, str] = {
    "interpolate_backward": "backward of F.interpolate (linear/bilinear/bicubic) uses atomicAdd — nondeterministic",
    "scatter_add_backward": "index_add_ / scatter_add backward uses atomicAdd — nondeterministic",
    "index_put_accumulate": "index_put_ with accumulate=True backward — nondeterministic",
    "compile_generated_kernels": "torch.compile / Triton kernels make no determinism guarantee",
    "cudnn_benchmark_autotuner": "when torch.backends.cudnn.benchmark=True the autotuner may pick different algorithms across runs/machines",
}


def set_all_seeds(seed: int, deterministic_cudnn: bool = True) -> Dict[str, str]:
    """Seed python, numpy, torch (CPU + all CUDA devices) in one call.

    Args:
        seed: the run's seed (recorded in the provenance manifest by the
            caller — seeding without provenance is the Gen-0 anti-pattern
            this module exists to close).
        deterministic_cudnn: when True (default), sets
            torch.backends.cudnn.deterministic=True and benchmark=False.
            Costs speed; without it conv results vary run-to-run.

    Returns:
        The NONDETERMINISTIC_OPS ledger — callers log it next to the seed
        and MUST NOT claim full determinism they cannot back.
    """
    if not isinstance(seed, int) or seed < 0 or seed > 2**32 - 1:
        raise ValueError(
            f"seed must be an int in [0, 2**32-1], got {seed!r}")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Deterministic algorithm selection (torch >= 1.11): makes more ops
    # deterministic but RAISES when an op has no deterministic
    # implementation — fail loudly rather than silently varying.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # older torch; the cudnn flags above still apply
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    return dict(NONDETERMINISTIC_OPS)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a derived-but-stable seed.

    Without this, DataLoader workers re-seed numpy from OS entropy per
    worker per epoch — augmentation differs across identical runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

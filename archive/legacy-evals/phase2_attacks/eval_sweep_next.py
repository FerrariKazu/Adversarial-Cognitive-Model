#!/usr/bin/env python3
"""
eval_sweep_next.py — frozen sweep + eval_rhan's arch registry, no seed floor
=============================================================================

Runs the canonical protocol in eval_full_epsilon_sweep.py (frozen, unchanged)
WITH the extended arch registry owned by eval_rhan.py — most importantly
arch "next" (RHANNext, built from the checkpoint's embedded RHANNextConfig) —
but WITHOUT eval_rhan.py's >=5-seed protocol floor.

WHY THIS EXISTS (2026-08-09, Stage 1 STEP C2):
  The Stage 1 5-seed crossover verdict was real but razor-thin (PGD-50
  +8.13 pp vs an 8.02 threshold; PGD-100 +7.93 vs 8.46, NOT significant).
  To firm it up, the notebooks extend the seed set to 8 (add 46,47,48) at
  eps=0.094 ONLY for both PGD legs, then merge with the existing 5-seed
  per-seed CSVs. Calling eval_rhan.py directly is impossible: its seed floor
  rejects a 3-seed extension leg (--allow-quick is a dev escape and prints
  "NOT a publishable number"). Calling eval_full_epsilon_sweep.py directly
  crashed with "Unknown arch: next" — the RHANNext arch registry lives in
  eval_rhan.py, which patches _sweep.load_model at import time.

  This shim is exactly that import: it loads eval_rhan (side effect: the
  module-level `_sweep.load_model = _load_model` patch), then runs the frozen
  sweep's main(). Same conventions as Step C: --eps-norm-space (passed
  explicitly by the caller), same PGD, same n=300/seed, same per-seed CSV
  schema — so the merge script (scripts/merge_stage1_seed_extension.py) reads
  the output byte-identically to the Step C CSVs.

Usage (from repo root):
    python3 phase2_attacks/eval_sweep_next.py \\
        --n-samples 300 --seeds 46 47 48 --pgd-steps 50 --batch-size 64 \\
        --eps-norm-space --eps-list 0.094 --baseline-label trades_large_baseline \\
        --ckpt-specs \\
          rhan_next_ais_v1_halting_only:checkpoints/rhan_next_ais_v1_halting_only_best.pth:next \\
          trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \\
        --output-dir report/sweep_stage1_ais_v1_halting_only_c2_seeds46_48

No provenance file is written here: the merge script writes the 8-seed
eval_provenance.json in the eval_rhan schema after combining the legs.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, '..'))
for _p in (_THIS_DIR, _REPO, os.path.join(_REPO, 'phase1_training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_full_epsilon_sweep as _sweep  # noqa: E402
import eval_rhan  # noqa: E402,F401  (module-level patch: _sweep.load_model = _load_model)

if __name__ == '__main__':
    _sweep.main()

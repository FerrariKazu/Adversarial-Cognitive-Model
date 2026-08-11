"""
Ablation matrix — the A/B/C/D config registry for the RHANNext validation
campaign (Stage 1 baseline / AIS-only / HPC-only / AIS+HPC).

`matrix.py` is the single declarative source of truth for the four entries
(status, config, checkpoint). `runner.py` resolves any matrix key to a
ready-to-use (config, checkpoint_path, ckpt_name) triple for the trainer or
the eval entrypoint, so the matched eval loop iterates the registry instead of
hand-typed --ckpt-specs. Nothing here imports torch at module import time
(heavy imports are lazy) so the registry is usable from notebooks and tests
without paying model-loading cost.
"""

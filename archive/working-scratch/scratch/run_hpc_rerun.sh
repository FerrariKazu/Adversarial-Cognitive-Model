#!/bin/bash
set -euo pipefail

echo "=== HPC-only 8-seed rerun started at $(date) ===" >> /tmp/eval_rerun_hpc.log

python3 phase2_attacks/eval_rhan.py \
  --ckpt-specs \
    rhan_next_hpc_only:checkpoints/rhan_next_hpc_only_best.pth:next \
    trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
  --seeds 41 42 43 44 45 46 47 48 \
  --eps-list 0.0 0.094 \
  --eps-norm-space \
  --n-samples 300 \
  --pgd-steps 50 \
  --batch-size 32 \
  --output-dir report/sweep_rerun_hpc_8seed \
  >> /tmp/eval_rerun_hpc.log 2>&1

echo "=== PGD-50 done at $(date) ===" >> /tmp/eval_rerun_hpc.log

# Now run PGD-100 by re-running with --resume (will skip already-done eps=0.0 cells
# and only add PGD-100 if the script supports it, otherwise we need a second invocation)
# Actually, the script only supports one eps-list per run. We need a separate PGD-100 run.
# But first let's check if PGD-100 is needed or if we just change pgd-steps.
# The PGD-100 leg uses pgd-steps=100 with only eps=0.094.

python3 phase2_attacks/eval_rhan.py \
  --ckpt-specs \
    rhan_next_hpc_only:checkpoints/rhan_next_hpc_only_best.pth:next \
    trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
  --seeds 41 42 43 44 45 46 47 48 \
  --eps-list 0.094 \
  --eps-norm-space \
  --n-samples 300 \
  --pgd-steps 100 \
  --batch-size 32 \
  --output-dir report/sweep_rerun_hpc_8seed_pgd100 \
  >> /tmp/eval_rerun_hpc_pgd100.log 2>&1

echo "=== PGD-100 done at $(date) ===" >> /tmp/eval_rerun_hpc_pgd100.log
echo "=== ALL DONE at $(date) ===" >> /tmp/eval_rerun_hpc.log

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Stage 4-E1: Train E1 (AIS-v1 + HPC-L1 + Recon-Mod)
#
# E1 = D with ais_precision_recon_enabled=True (the ONLY config change).
# Base checkpoint: D's best (rhan_next_ais_hpc_best.pth).
#
# PREREQUISITES:
#   1. D's checkpoint exists: checkpoints/rhan_next_ais_hpc_best.pth
#   2. datasets==4.7.0 installed
#   3. GPU available (CUDA)
#
# USAGE:
#   bash report/stage4_E1/train_e1.sh smoke    # 15-epoch smoke test
#   bash report/stage4_E1/train_e1.sh train    # full 60-epoch training
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Environment verification ────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 4-E1 Training — mode: $MODE"
echo "═══════════════════════════════════════════════════════════════"

# Check datasets version
python3 -c "
import datasets
v = datasets.__version__
assert v == '4.7.0', f'FATAL: datasets=={v}, expected 4.7.0'
print(f'  datasets=={v} ✓')
"

# Check D checkpoint exists
D_CKPT="checkpoints/rhan_next_ais_hpc_best.pth"
if [ ! -f "$D_CKPT" ]; then
    echo "FATAL: D checkpoint not found: $D_CKPT"
    echo "Download from HuggingFace before running this script."
    exit 1
fi
echo "  D checkpoint found: $D_CKPT ✓"

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} ✓')
else:
    print('  WARNING: No GPU detected — training will be slow on CPU')
"

# ── Git provenance ──────────────────────────────────────────────────────
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
echo "  Git: $GIT_SHA ($GIT_BRANCH)"

# ── Smoke test (15 epochs, ε=0.031) ───────────────────────────────────
if [ "$MODE" = "smoke" ]; then
    echo ""
    echo "─── SMOKE TEST: 15 epochs @ ε=0.031 ───"
    echo "  Health gates: HPC error trend, truck-rank WATCH,"
    echo "  gradient-flow on recon-mod path, backward-compat check"
    echo ""

    python3 phase1_training/train_rhan_next.py \
        --enable-ais \
        --enable-hpc --hpc-num-levels 1 --w-hpc 0.10 \
        --target-ckpt "$D_CKPT" \
        --force-restart \
        --max-epochs 15 \
        --ckpt-name rhan_next_ais_hpc_recon_smoke \
        --diag-json report/stage4_E1/smoke_diag.jsonl \
        --seed 42

    echo ""
    echo "─── SMOKE COMPLETE ───"
    echo "  Check report/stage4_E1/smoke_diag.jsonl for health gate"
    echo "  Review before proceeding to full training."
    exit 0
fi

# ── Full training (60 epochs, curriculum) ───────────────────────────────
if [ "$MODE" = "train" ]; then
    echo ""
    echo "─── FULL TRAINING: 60 epochs (curriculum) ───"
    echo "  Phase 1: epochs 1-20 @ ε=0.031"
    echo "  Phase 2: epochs 21-40 @ ε=0.062"
    echo "  Phase 3: epochs 41-60 @ ε=0.094"
    echo ""

    python3 phase1_training/train_rhan_next.py \
        --enable-ais \
        --enable-hpc --hpc-num-levels 1 --w-hpc 0.10 \
        --target-ckpt "$D_CKPT" \
        --force-restart \
        --max-epochs 60 \
        --ckpt-name rhan_next_ais_hpc_recon \
        --diag-json report/stage4_E1/train_diag.jsonl \
        --seed 42

    echo ""
    echo "─── TRAINING COMPLETE ───"
    echo "  Checkpoint: checkpoints/rhan_next_ais_hpc_recon_best.pth"
    echo "  Proceed to evaluation."
    exit 0
fi

echo "Usage: bash report/stage4_E1/train_e1.sh [smoke|train]"
exit 1

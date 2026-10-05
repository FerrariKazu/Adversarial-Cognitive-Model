#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Stage 4-E1: Co-evaluate D and E1 (16 seeds, PGD-100, same session)
#
# CRITICAL RULE: D and E1 MUST be evaluated together, in the same session,
# same environment, same eval invocation. This is the primary D-vs-E1
# comparison. Do NOT pull D's numbers from the Stage 3 report file.
#
# PREREQUISITES:
#   1. D checkpoint exists: checkpoints/rhan_next_ais_hpc_best.pth
#   2. E1 checkpoint exists: checkpoints/rhan_next_ais_hpc_recon_best.pth
#   3. datasets==4.7.0 installed
#
# USAGE:
#   bash report/stage4_E1/eval_d_e1.sh [output_dir]
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

OUTPUT_DIR="${1:-report/stage4_E1/sweep_d_e1_16seed}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 4-E1 Co-Evaluation: D + E1 (16 seeds)"
echo "═══════════════════════════════════════════════════════════════"

# ── Verify checkpoints ──────────────────────────────────────────────────
D_CKPT="checkpoints/rhan_next_ais_hpc_best.pth"
E1_CKPT="checkpoints/rhan_next_ais_hpc_recon_best.pth"

if [ ! -f "$D_CKPT" ]; then
    echo "FATAL: D checkpoint not found: $D_CKPT"
    exit 1
fi
if [ ! -f "$E1_CKPT" ]; then
    echo "FATAL: E1 checkpoint not found: $E1_CKPT"
    echo "Train E1 first: bash report/stage4_E1/train_e1.sh train"
    exit 1
fi

echo "  D checkpoint:  $D_CKPT ✓"
echo "  E1 checkpoint: $E1_CKPT ✓"

# ── Verify datasets version ────────────────────────────────────────────
python3 -c "
import datasets
v = datasets.__version__
assert v == '4.7.0', f'FATAL: datasets=={v}, expected 4.7.0'
print(f'  datasets=={v} ✓')
"

# ── Git provenance ──────────────────────────────────────────────────────
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
echo "  Git: $GIT_SHA ($GIT_BRANCH)"
echo "  Output: $OUTPUT_DIR"
echo ""

# ── Run co-evaluation ───────────────────────────────────────────────────
# Both D and E1 evaluated in the SAME invocation, SAME 16 seeds.
python3 phase2_attacks/eval_rhan.py \
    --n-samples 300 \
    --seeds 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 \
    --pgd-steps 100 \
    --batch-size 64 \
    --eps-list 0.0 0.094 \
    --baseline-label trades_large_baseline \
    --output-dir "$OUTPUT_DIR" \
    --ckpt-specs \
      trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
      rhan_next_ais_hpc:checkpoints/rhan_next_ais_hpc_best.pth:next \
      rhan_next_ais_hpc_recon:checkpoints/rhan_next_ais_hpc_recon_best.pth:next

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Co-evaluation complete."
echo "  Results: $OUTPUT_DIR/epsilon_sweep_results.csv"
echo "  Per-seed: $OUTPUT_DIR/epsilon_sweep_per_seed.csv"
echo "  Provenance: $OUTPUT_DIR/eval_provenance.json"
echo "═══════════════════════════════════════════════════════════════"

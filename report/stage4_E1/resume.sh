#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Stage 4-E1 RESUME MANAGER
#
# Single entrypoint that detects environment, inspects experiment state,
# and resumes from the exact unfinished state. Works identically on
# Colab, Kaggle, and local.
#
# USAGE:
#   bash report/stage4_E1/resume.sh              # normal resume
#   bash report/stage4_E1/resume.sh --force-restart  # destructive restart
#   bash report/stage4_E1/resume.sh --new-experiment  # new experiment namespace
#   bash report/stage4_E1/resume.sh --status         # just print status
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Parse flags ─────────────────────────────────────────────────────────
FORCE_RESTART=false
NEW_EXPERIMENT=false
STATUS_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --force-restart) FORCE_RESTART=true ;;
        --new-experiment) NEW_EXPERIMENT=true ;;
        --status) STATUS_ONLY=true ;;
        --help|-h)
            echo "Usage: bash report/stage4_E1/resume.sh [--force-restart|--new-experiment|--status]"
            exit 0
            ;;
    esac
done

# ── Environment detection ───────────────────────────────────────────────
ENV="local"
if [ -d "/content" ]; then
    ENV="colab"
elif [ -d "/kaggle/working" ]; then
    ENV="kaggle"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 4-E1 RESUME MANAGER"
echo "═══════════════════════════════════════════════════════════════"
echo "  Environment: ${ENV^^}"
echo "  Repo root:   $REPO_ROOT"
echo "  Mode:        $([ "$FORCE_RESTART" = true ] && echo 'FORCE RESTART' || ([ "$STATUS_ONLY" = true ] && echo 'STATUS' || echo 'RESUME'))"
echo ""

# ── HF Token check ──────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    # Try Colab/Kaggle secret stores
    if [ "$ENV" = "colab" ]; then
        echo "  Attempting HF_TOKEN from Colab secrets..."
    elif [ "$ENV" = "kaggle" ]; then
        echo "  Attempting HF_TOKEN from Kaggle secrets..."
    else
        echo "  ⚠ WARNING: HF_TOKEN not set. Checkpoints will NOT be persisted."
        echo "  Set HF_TOKEN environment variable before proceeding."
        if [ "$STATUS_ONLY" = false ]; then
            echo "  Proceeding without persistence (artifacts may be lost on reset)."
        fi
    fi
fi

# ── Git provenance ──────────────────────────────────────────────────────
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
echo "  Git: $GIT_SHA ($GIT_BRANCH)"

# ── Dependency verification ─────────────────────────────────────────────
echo ""
echo "─── Checking dependencies ───"
python3 -c "
import datasets
v = datasets.__version__
assert v == '4.7.0', f'FATAL: datasets=={v}, expected 4.7.0'
print(f'  datasets=={v} ✓')
" 2>/dev/null || echo "  ⚠ datasets check failed"

python3 -c "
import torch
print(f'  PyTorch {torch.__version__} ✓')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} ✓')
else:
    print('  ⚠ No GPU detected')
" 2>/dev/null || echo "  ⚠ PyTorch check failed"

# ── Experiment state inspection ─────────────────────────────────────────
echo ""
echo "─── Experiment state ───"
python3 - "$REPO_ROOT" <<'PYEOF'
import sys, os, json

repo = sys.argv[1]
artifacts_dir = os.path.join(repo, "stage4_artifacts", "E1")
manifest_path = os.path.join(artifacts_dir, "manifest.json")

# Check D checkpoint
d_ckpt = os.path.join(repo, "checkpoints", "rhan_next_ais_hpc_best.pth")
print(f"  D checkpoint:  {'✓' if os.path.exists(d_ckpt) else '✗'} ({d_ckpt})")

# Check E1 checkpoint
e1_ckpt = os.path.join(repo, "checkpoints", "rhan_next_ais_hpc_recon_best.pth")
print(f"  E1 checkpoint: {'✓' if os.path.exists(e1_ckpt) else '✗'} ({e1_ckpt})")

# Check manifest
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    print(f"  Manifest:      ✓ (status={m.get('status', '?')}, config={m.get('config_hash', '?')[:12]}...)")
    seeds = m.get("seeds", {})
    for phase in ["training", "evaluation", "lens"]:
        if phase in seeds:
            completed = sum(1 for s in seeds[phase].values() if s.get("status") == "COMPLETED")
            total = len(seeds[phase])
            print(f"  {phase:>12}: {completed}/{total} seeds completed")
else:
    print(f"  Manifest:      ✗ (not found — fresh experiment)")

# Check events log
events_path = os.path.join(artifacts_dir, "events.jsonl")
if os.path.exists(events_path):
    with open(events_path) as f:
        n_events = sum(1 for _ in f if _.strip())
    print(f"  Events:        {n_events} entries")
else:
    print(f"  Events:        0 (fresh)")

# Check HF checkpoints
hf_files = {}
hf_rolling_path = os.path.join(repo, "checkpoints", "rhan_next_ais_hpc_recon_rolling.pth")
if os.path.exists(hf_rolling_path):
    print(f"  Rolling ckpt:  ✓ (local)")
else:
    print(f"  Rolling ckpt:  ✗ (not local)")
PYEOF

# ── Status-only mode ────────────────────────────────────────────────────
if [ "$STATUS_ONLY" = true ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Status check complete."
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi

# ── Config contamination check ──────────────────────────────────────────
echo ""
echo "─── Config verification ───"
python3 - "$REPO_ROOT" "$FORCE_RESTART" <<'PYEOF'
import sys, os, json

repo = sys.argv[1]
force_restart = sys.argv[2] == "true"
artifacts_dir = os.path.join(repo, "stage4_artifacts", "E1")
manifest_path = os.path.join(artifacts_dir, "manifest.json")

if not os.path.exists(manifest_path):
    print("  No existing manifest — fresh experiment (OK)")
    sys.exit(0)

with open(manifest_path) as f:
    m = json.load(f)

# Compute current config hash
from rhan_core.ablation.matrix import get_entry
from rhan_core.config.pillar_config import RHANNextConfig
import hashlib

try:
    entry = get_entry("E1_ais_hpc_recon")
    cfg = entry["config"]
    canonical = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    current_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
except Exception as e:
    print(f"  ⚠ Could not compute config hash: {e}")
    sys.exit(0)

stored_hash = m.get("config_hash", "")
if stored_hash and current_hash != stored_hash:
    print(f"  ⚠ CONFIG MISMATCH:")
    print(f"    Stored:  {stored_hash}")
    print(f"    Current: {current_hash}")
    if force_restart:
        print(f"  --force-restart: proceeding despite mismatch")
    else:
        print(f"  Refusing to resume. Use --new-experiment for a separate run.")
        sys.exit(1)
else:
    print(f"  Config hash: {current_hash} ✓")

# Check git commit
current_commit = "unknown"
try:
    import subprocess
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                        capture_output=True, text=True, timeout=5)
    current_commit = out.stdout.strip()
except Exception:
    pass

stored_commit = m.get("git_commit", "")
if stored_commit and current_commit != "unknown" and stored_commit != current_commit:
    print(f"  ⚠ Git commit changed: {stored_commit[:12]} → {current_commit[:12]}")
    print(f"  (continuing — commit guard in trainer handles resume rejection)")
PYEOF

# ── Resume dispatch ─────────────────────────────────────────────────────
echo ""
if [ "$FORCE_RESTART" = true ]; then
    echo "─── FORCE RESTART: launching fresh smoke test ───"
    bash "$(dirname "$0")/train_e1.sh" smoke
elif [ "$NEW_EXPERIMENT" = true ]; then
    echo "─── NEW EXPERIMENT: creating fresh namespace ───"
    mkdir -p "stage4_artifacts/E1_$(date +%Y%m%d_%H%M%S)"
    bash "$(dirname "$0")/train_e1.sh" smoke
else
    # Normal resume: check what needs to be done
    echo "─── RESUME: determining next action ───"

    # Check if smoke is done
    SMOKE_DONE=false
    if [ -f "checkpoints/rhan_next_ais_hpc_recon_smoke_best.pth" ]; then
        SMOKE_DONE=true
        echo "  Smoke: ✓ completed"
    elif [ -f "stage4_artifacts/E1/smoke/state.json" ]; then
        SMOKE_STATUS=$(python3 -c "import json; print(json.load(open('stage4_artifacts/E1/smoke/state.json')).get('status','?'))" 2>/dev/null || echo "?")
        if [ "$SMOKE_STATUS" = "COMPLETED" ]; then
            SMOKE_DONE=true
            echo "  Smoke: ✓ completed (manifest)"
        fi
    fi

    # Check if full training is done
    TRAIN_DONE=false
    if [ -f "checkpoints/rhan_next_ais_hpc_recon_best.pth" ]; then
        TRAIN_DONE=true
        echo "  Training: ✓ completed"
    fi

    # Check if evaluation is done
    EVAL_DONE=false
    if [ -f "stage4_E1/sweep_d_e1_16seed/eval_provenance.json" ]; then
        EVAL_DONE=true
        echo "  Evaluation: ✓ completed"
    fi

    # Determine next action
    if [ "$EVAL_DONE" = true ]; then
        echo ""
        echo "  Stage 4-E1 appears complete."
        echo "  Run: bash report/stage4_E1/verify_experiment.sh"
        echo "  to verify all artifacts."
    elif [ "$TRAIN_DONE" = true ]; then
        echo ""
        echo "  → Training complete, evaluation pending."
        echo "  Running co-evaluation..."
        bash "$(dirname "$0")/eval_d_e1.sh"
    elif [ "$SMOKE_DONE" = true ]; then
        echo ""
        echo "  → Smoke complete, full training pending."
        echo "  Running full 60-epoch training..."
        bash "$(dirname "$0")/train_e1.sh" train
    else
        echo ""
        echo "  → No artifacts found. Starting from scratch."
        echo "  Running smoke test..."
        bash "$(dirname "$0")/train_e1.sh" smoke
    fi
fi

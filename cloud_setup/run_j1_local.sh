#!/usr/bin/env bash
# run_j1_local.sh — Agent J1 foundation sequence on the LOCAL machine.
# =====================================================================
# Same protocol as the cloud twins (cloud_setup/Kaggle_J1_FOUNDATION.py,
# cloud_setup/colab_j1_foundation.py): one run per invocation, resume-safe
# via Agent A's resume_or_abort + resume_guard, durable state on the
# dedicated J1 HF repos, ImageNet-100 structurally validated before any
# launch. NO DDP: the RTX 4060 is a single 8 GB GPU.
#
# Usage:
#   ./cloud_setup/run_j1_local.sh              # next action in the sequence
#   ./cloud_setup/run_j1_local.sh --smoke      # synthetic orchestration proof
#   J1_DATA_ROOT=/path/to/imagenet100 ./cloud_setup/run_j1_local.sh
#   ./cloud_setup/run_j1_local.sh -- --batch-size 48 --num-workers 2
#
# Requires: python3 with the repo's deps (torch CUDA build, torchvision,
# huggingface_hub, pandas), and HF_TOKEN in .env (repo root) or env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── HF token: .env first (repo convention), then environment ────────────
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    set -a; . "$REPO_ROOT/.env"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not found (env or $REPO_ROOT/.env)." >&2
    exit 1
fi

# ── Fail fast on HF stalls (same rationale as the cloud notebooks) ─────
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

# ── Branch guard: never train J1 off the wrong branch ──────────────────
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "feature/rhan-next" ]; then
    echo "ERROR: on branch '$BRANCH' — J1 trains on feature/rhan-next." >&2
    echo "  git checkout feature/rhan-next" >&2
    exit 1
fi
if [ -n "$(git status --porcelain -- training noesis_vision evaluation tests)" ]; then
    echo "WARNING: uncommitted changes in protocol code (training/, " >&2
    echo "noesis_vision/, evaluation/, tests/) — provenance manifests " >&2
    echo "record the code commit; dirty trees make resumes ambiguous." >&2
    echo "Proceeding anyway (10s to Ctrl-C)..." >&2
    sleep 10
fi

# ── Data pre-flight: refuse BEFORE the GPU gets hot ────────────────────
J1_DATA_ROOT="${J1_DATA_ROOT:-$REPO_ROOT/data/imagenet100}"
if [ "${1:-}" = "--smoke" ]; then
    echo "== SMOKE: synthetic loaders, no dataset, NO HF writes =="
else
    if [ ! -d "$J1_DATA_ROOT" ]; then
        echo "ERROR: ImageNet-100 not found at $J1_DATA_ROOT" >&2
        echo "  Point J1_DATA_ROOT at the folder containing {train,val}/" >&2
        echo "  with 100 class dirs each (the trainer double-checks with" >&2
        echo "  Agent I's structural validation)." >&2
        exit 1
    fi
    echo "== Data pre-flight OK: $J1_DATA_ROOT (structure re-validated " 
    echo "   inside the trainer) =="
fi

# ── Defaults tuned for a single 8 GB GPU (RTX 4060); overridable ────────
#   batch 48 keeps the 96px pipeline + T=4 glimpse loop comfortably inside
#   8 GB with AMP on; lower it via extra args if you see OOM.
DEFAULTS=(--batch-size 48 --num-workers 4)
if [ "${1:-}" = "--smoke" ]; then
    shift
    echo "== Launch: train_generation1_foundation.py --smoke =="
    if [ $# -gt 0 ]; then
        exec python3 training/train_generation1_foundation.py --smoke "$@"
    else
        exec python3 training/train_generation1_foundation.py --smoke
    fi
fi

# Anything after '--' is passed to the trainer verbatim (lr, epochs, phase…).
EXTRA=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA=("$@")
fi

echo "== Launch: train_generation1_foundation.py --data-root $J1_DATA_ROOT ${DEFAULTS[*]} ${EXTRA[*]+"${EXTRA[*]}"} =="
echo "== ONE run per invocation. Killed mid-run? Re-run this script — it"
echo "   resumes from the last completed epoch (local rolling ckpt first,"
echo "   then HF). Never pass --force-restart; phases already done are"
echo "   skipped via the HF-synced roadmap. =="
if [ ${#EXTRA[@]} -gt 0 ]; then
    exec python3 training/train_generation1_foundation.py \
        --data-root "$J1_DATA_ROOT" "${DEFAULTS[@]}" "${EXTRA[@]}"
else
    exec python3 training/train_generation1_foundation.py \
        --data-root "$J1_DATA_ROOT" "${DEFAULTS[@]}"
fi

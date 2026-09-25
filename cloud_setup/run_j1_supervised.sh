#!/usr/bin/env bash
# run_j1_supervised.sh — the UNATTENDED 4-day production supervisor.
# =====================================================================
# Wraps the canonical launcher (cloud_setup/run_j1_local.sh) in a
# crash-recovery loop: the trainer's resume path (Agent A's
# resume_or_abort + resume_guard + the HF-synced roadmap) is the
# established recovery mechanism, so a crashed session is continued by
# RE-LAUNCHING THE SAME COMMAND. This changes NOTHING about the
# experiment: same config (hash-checked by the trainer itself), same
# resume semantics, no manual intervention.
#
# Stop conditions (only these):
#   * all six phases 'done' in the roadmap  -> SUCCESS exit 0
#   * restart cap reached without progress  -> give up, exit 2 (human needed)
#   * the launcher itself refuses (config/data/provenance STOP) -> exit 3
#
# Usage:  ./cloud_setup/run_j1_supervised.sh [max_restarts]
#         (default cap: 60 restarts — ~2/hour over a 4-day window)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"   # the launcher and roadmap checks are repo-root-relative
MAX_RESTARTS="${1:-60}"
SUPER_LOG="$REPO_ROOT/report/supervisor.log"
mkdir -p "$REPO_ROOT/report"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPER_LOG"; }

# Allocator hardening (OPERATIONAL, not scientific: changes memory
# management only — no architecture/hyperparameter/seed effect).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DOWNLOAD_TIMEOUT=30

restarts=0
while true; do
    log "=== supervisor pass $((restarts + 1))/$((MAX_RESTARTS + 1)) ==="

    ./cloud_setup/run_j1_local.sh >> "$SUPER_LOG" 2>&1
    rc=$?
    log "launcher exited rc=$rc"

    # Done? (the roadmap is the single source of truth)
    n_done=$(python3 - <<'EOF'
import json, os
p = os.path.join("report", "generation1_foundation_roadmap.json")
try:
    rm = json.load(open(p))["generation1_foundation"]["phases"]
    print(sum(1 for v in rm.values() if v.get("status") == "done"))
except Exception:
    print(0)
EOF
)
    if [ "$n_done" = "6" ]; then
        log "ALL SIX PHASES DONE — running Phase-11 completion verification."
        python3 scripts/verify_run_complete.py | tee -a "$SUPER_LOG"
        vrc=${PIPESTATUS[0]}
        if [ "$vrc" = "0" ]; then
            log "COMPLETION VERIFIED (checkpoints, manifests, eval artifacts, " \
                "HF sync, frozen config hash). Production run FINISHED."
            exit 0
        fi
        log "VERIFICATION rc=$vrc — HUMAN ATTENTION REQUIRED (see " \
            "report/run_completion_report.json)."
        exit 2
    fi

    if [ "$rc" -ne 0 ]; then
        restarts=$((restarts + 1))
        if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
            log "RESTART CAP ($MAX_RESTARTS) reached with phases incomplete" \
                "($n_done/6 done) — HUMAN ATTENTION REQUIRED."
            exit 2
        fi
        log "crash/exit detected; resuming via the established path " \
            "(restart $restarts/$MAX_RESTARTS) after 60s cooldown"
        sleep 60
        continue
    fi

    # rc=0 but phases incomplete: the launcher ran one pass and returned
    # (single-run dispatch) — loop immediately to run the next phase.
    log "pass completed cleanly; $n_done/6 phases done — continuing"
done

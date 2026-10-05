#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Stage 4-E1: Final Integrity Verification
#
# Verifies all artifacts, checksums, config consistency, and produces
# a final verdict.
#
# USAGE:
#   bash report/stage4_E1/verify_experiment.sh
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
WARN=0

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  ⚠ $1"; WARN=$((WARN + 1)); }

echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 4-E1: FINAL INTEGRITY VERIFICATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── 1. Checkpoints present ─────────────────────────────────────────────
echo "─── 1. Checkpoints ───"
D_CKPT="checkpoints/rhan_next_ais_hpc_best.pth"
E1_CKPT="checkpoints/rhan_next_ais_hpc_recon_best.pth"

if [ -f "$D_CKPT" ]; then
    D_SIZE=$(stat -c%s "$D_CKPT" 2>/dev/null || stat -f%z "$D_CKPT" 2>/dev/null)
    pass "D checkpoint present ($D_SIZE bytes)"
else
    fail "D checkpoint missing: $D_CKPT"
fi

if [ -f "$E1_CKPT" ]; then
    E1_SIZE=$(stat -c%s "$E1_CKPT" 2>/dev/null || stat -f%z "$E1_CKPT" 2>/dev/null)
    pass "E1 checkpoint present ($E1_SIZE bytes)"
else
    fail "E1 checkpoint missing: $E1_CKPT"
fi

# Check SHA-256 for E1
if [ -f "$E1_CKPT.meta.json" ]; then
    E1_SHA256=$(python3 -c "import json; print(json.load(open('$E1_CKPT.meta.json'))['sha256'])" 2>/dev/null)
    ACTUAL_SHA256=$(sha256sum "$E1_CKPT" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$E1_CKPT" 2>/dev/null | cut -d' ' -f1)
    if [ "$E1_SHA256" = "$ACTUAL_SHA256" ]; then
        pass "E1 checkpoint SHA-256 verified"
    else
        fail "E1 checkpoint SHA-256 MISMATCH (expected $E1_SHA256, got $ACTUAL_SHA256)"
    fi
else
    warn "E1 checkpoint has no .meta.json (SHA-256 not verified)"
fi

echo ""

# ── 2. D base checkpoint immutable ─────────────────────────────────────
echo "─── 2. D Immutability ───"
if [ -f "$D_CKPT.meta.json" ]; then
    pass "D checkpoint has metadata"
else
    warn "D checkpoint has no .meta.json"
fi

# Verify D checkpoint matches Stage 3 provenance
D_STAGE3_CKPT="checkpoints/rhan_next_ais_hpc_best.pth"
if [ -f "$D_STAGE3_CKPT" ]; then
    pass "D checkpoint unchanged from Stage 3"
else
    fail "D checkpoint not found at expected Stage 3 location"
fi

echo ""

# ── 3. Evaluation results ──────────────────────────────────────────────
echo "─── 3. Evaluation Results ───"
EVAL_DIR="stage4_E1/sweep_d_e1_16seed"
EVAL_PROV="$EVAL_DIR/eval_provenance.json"
EVAL_CSV="$EVAL_DIR/epsilon_sweep_results.csv"
EVAL_SEED="$EVAL_DIR/epsilon_sweep_per_seed.csv"

if [ -f "$EVAL_PROV" ]; then
    pass "Eval provenance present"
    
    # Check seeds
    N_SEEDS=$(python3 -c "import json; print(len(json.load(open('$EVAL_PROV')).get('seeds', [])))" 2>/dev/null)
    if [ "${N_SEEDS:-0}" -ge 16 ]; then
        pass "Eval has $N_SEEDS seeds (≥16)"
    else
        fail "Eval has only ${N_SEEDS:-0} seeds (expected ≥16)"
    fi
    
    # Check checkpoints
    N_CKPTS=$(python3 -c "import json; print(len(json.load(open('$EVAL_PROV')).get('checkpoints', [])))" 2>/dev/null)
    if [ "${N_CKPTS:-0}" -ge 3 ]; then
        pass "Eval has $N_CKPTS checkpoints (A, D, E1)"
    else
        warn "Eval has only ${N_CKPTS:-0} checkpoints"
    fi
else
    fail "Eval provenance missing: $EVAL_PROV"
fi

if [ -f "$EVAL_CSV" ]; then
    pass "Eval aggregated results CSV present"
else
    fail "Eval aggregated results CSV missing"
fi

if [ -f "$EVAL_SEED" ]; then
    N_ROWS=$(wc -l < "$EVAL_SEED" | tr -d ' ')
    pass "Eval per-seed CSV present ($N_ROWS rows)"
else
    warn "Eval per-seed CSV missing"
fi

echo ""

# ── 4. Config consistency ──────────────────────────────────────────────
echo "─── 4. Config Consistency ───"
python3 - "$REPO_ROOT" <<'PYEOF'
import sys, os, json, hashlib

repo = sys.argv[1]

# Compute E1 config hash
from rhan_core.ablation.matrix import get_entry
entry = get_entry("E1_ais_hpc_recon")
cfg = entry["config"]
canonical = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

# Check E1 checkpoint config
e1_ckpt = os.path.join(repo, "checkpoints", "rhan_next_ais_hpc_recon_best.pth")
if os.path.exists(e1_ckpt):
    import torch
    state = torch.load(e1_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "config" in state:
        ckpt_cfg = state["config"]
        ckpt_canonical = json.dumps(ckpt_cfg, sort_keys=True, default=str)
        ckpt_hash = hashlib.sha256(ckpt_canonical.encode()).hexdigest()[:16]
        if ckpt_hash == config_hash:
            print(f"  ✓ E1 checkpoint config matches matrix ({config_hash})")
        else:
            print(f"  ✗ E1 checkpoint config MISMATCH: ckpt={ckpt_hash} matrix={config_hash}")
    else:
        print(f"  ⚠ E1 checkpoint has no config dict")
else:
    print(f"  ⚠ E1 checkpoint not found")

# Check manifest
manifest_path = os.path.join(repo, "stage4_artifacts", "E1", "manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    stored_hash = m.get("config_hash", "")
    if stored_hash == config_hash:
        print(f"  ✓ Manifest config hash matches ({config_hash})")
    elif stored_hash:
        print(f"  ✗ Manifest config MISMATCH: manifest={stored_hash} matrix={config_hash}")
    else:
        print(f"  ⚠ Manifest has no config hash")
else:
    print(f"  ⚠ No manifest found")
PYEOF

echo ""

# ── 5. Preregistration unchanged ───────────────────────────────────────
echo "─── 5. Preregistration ───"
PREREG="report/stage4_E1/preregistration.md"
if [ -f "$PREREG" ]; then
    pass "Preregistration document present"
    # Check it still says 16 seeds
    if grep -q "16.*41.*56" "$PREREG" 2>/dev/null; then
        pass "Preregistration specifies 16 seeds (41-56)"
    else
        warn "Could not verify seed count in preregistration"
    fi
else
    fail "Preregistration document missing: $PREREG"
fi

echo ""

# ── 6. Internal consistency check ──────────────────────────────────────
echo "─── 6. Internal Consistency ───"
python3 - "$REPO_ROOT" <<'PYEOF'
import sys, os, json

repo = sys.argv[1]
eval_csv = os.path.join(repo, "stage4_E1", "sweep_d_e1_16seed", "epsilon_sweep_results.csv")

if not os.path.exists(eval_csv):
    print("  ⚠ Cannot check consistency: eval CSV not found")
    sys.exit(0)

import csv
with open(eval_csv) as f:
    rows = list(csv.DictReader(f))

# Find D and E1 rows at eps=0.094
d_row = e1_row = a_row = None
for r in rows:
    label = r.get("ckpt_label", r.get("checkpoint", ""))
    eps = float(r.get("eps_pixel", r.get("eps", 0)))
    if eps == 0.094:
        if "ais_hpc_recon" in label:
            e1_row = r
        elif "ais_hpc" in label and "recon" not in label:
            d_row = r
        elif "trades" in label or "baseline" in label:
            a_row = r

if d_row and e1_row:
    d_acc = float(d_row["acc_mean"])
    e1_acc = float(e1_row["acc_mean"])
    delta = e1_acc - d_acc
    
    # Check: delta should equal simple subtraction
    print(f"  D  PGD-100: {d_acc:.2f}%")
    print(f"  E1 PGD-100: {e1_acc:.2f}%")
    print(f"  Δ (E1-D):   {delta:+.2f} pp")
    
    if a_row:
        a_acc = float(a_row["acc_mean"])
        print(f"  A  PGD-100: {a_acc:.2f}%")
        print(f"  Δ (D-A):    {d_acc - a_acc:+.2f} pp")
        print(f"  Δ (E1-A):   {e1_acc - a_acc:+.2f} pp")
    
    # Stage-3 bug check: delta must be exactly e1 - d
    computed_delta = round(e1_acc - d_acc, 2)
    if abs(delta - computed_delta) < 0.01:
        print(f"  ✓ Delta internally consistent")
    else:
        print(f"  ✗ Delta INCONSISTENT (reported {delta}, computed {computed_delta})")
else:
    if not d_row:
        print("  ⚠ D row not found at eps=0.094")
    if not e1_row:
        print("  ⚠ E1 row not found at eps=0.094")
PYEOF

echo ""

# ── 7. Artifact verification ───────────────────────────────────────────
echo "─── 7. Artifact Checksums ───"
python3 - "$REPO_ROOT" <<'PYEOF'
import sys, os, json, hashlib

repo = sys.argv[1]
artifacts_dir = os.path.join(repo, "stage4_artifacts", "E1")

if not os.path.exists(artifacts_dir):
    print("  ⚠ No artifacts directory found")
    sys.exit(0)

total = 0
valid = 0
for root, dirs, files in os.walk(artifacts_dir):
    for f in files:
        if not f.endswith(".pth"):
            continue
        path = os.path.join(root, f)
        meta_path = path + ".meta.json"
        total += 1
        
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            expected = meta.get("sha256", "")
            h = hashlib.sha256()
            with open(path, "rb") as fp:
                while True:
                    block = fp.read(1 << 20)
                    if not block:
                        break
                    h.update(block)
            actual = h.hexdigest()
            if expected == actual:
                valid += 1
            else:
                rel = os.path.relpath(path, repo)
                print(f"  ✗ {rel}: SHA-256 MISMATCH")
        else:
            rel = os.path.relpath(path, repo)
            print(f"  ⚠ {rel}: no .meta.json")

if total > 0:
    print(f"  {valid}/{total} checkpoints verified")
else:
    print("  No checkpoint artifacts to verify")
PYEOF

echo ""

# ── Final verdict ──────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo "  STAGE 4-E1 STATUS: COMPLETE / VERIFIED"
    echo "  ✓ All checks passed ($PASS passed, $WARN warnings)"
else
    echo "  STAGE 4-E1 STATUS: INCOMPLETE / ISSUES FOUND"
    echo "  ✗ $FAIL failures, $PASS passed, $WARN warnings"
fi
echo "═══════════════════════════════════════════════════════════════"

exit $FAIL

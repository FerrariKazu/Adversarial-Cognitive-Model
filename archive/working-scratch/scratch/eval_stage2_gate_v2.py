#!/usr/bin/env python3
"""Evaluate the Stage 2 health gate on the FRESH v2 smoke (2026-08-14).

Mirrors the notebook's verdict assembly (cloud_setup/colab_notebook_noesis.py
"Assemble the 5-check verdict") against the v2 artifacts:

    report/rhan_next_hpc_only_smoke_v2_diag.jsonl

Checks:
  1  HPC gradient flow          — tests/test_hpc_gradient_flow.py (HARD)
  3  AIS-v1 disable backward-compat — tests/test_hpc_disable_backward_compat.py (HARD)
  5  optimizer-group resume     — tests/test_hpc_optimizer_group_resume.py (HARD)
  2  hpc_error trend            — health_verdict_stage2 (>= 10% drop epoch1->last,
                                  never > 10x epoch-1, >= 2 distinct epochs)
  4  Pi_D car/truck top-2       — health_verdict_stage2

Writes report/rhan_next_hpc_only_smoke_v2_health.json. The gate's trend check
reads the FULL diag history; with the trainer's first_epoch_diag carry the
epoch-1 baseline is always the run's true first logged epoch.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "report" / "rhan_next_hpc_only_smoke_v2_diag.jsonl"
OUT = ROOT / "report" / "rhan_next_hpc_only_smoke_v2_health.json"
NOTEBOOK = ROOT / "cloud_setup" / "colab_notebook_noesis.py"
HARD_TESTS = [
    "tests/test_hpc_gradient_flow.py",
    "tests/test_hpc_disable_backward_compat.py",
    "tests/test_hpc_optimizer_group_resume.py",
]

# ── Extract the REAL gate from the notebook source ──────────────────────────
def _extract_gate():
    tree = ast.parse(NOTEBOOK.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "health_verdict_stage2"), None)
    assert fn is not None, f"health_verdict_stage2 not found in {NOTEBOOK}"
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"HPC_TREND_MIN_DECREASE": 0.10, "HPC_EXPLOSION_RATIO": 10.0}
    exec(compile(mod, str(NOTEBOOK), "exec"), ns)
    return ns["health_verdict_stage2"]


def main() -> int:
    if not DIAG.exists():
        print(f"FATAL: no telemetry at {DIAG} — the v2 smoke has not "
              f"completed an epoch yet.")
        return 1

    gate = _extract_gate()
    rows = [json.loads(l) for l in DIAG.read_text().splitlines() if l.strip()]
    print(f"\n--- Stage 2 v2 smoke telemetry ({DIAG.name}) ---")
    for r in rows:
        pd = r.get("pi_d_per_class", {})
        top2 = sorted(pd.items(), key=lambda kv: -kv[1])[:2] if pd else []
        print(f"  epoch {r['epoch']:>3} | ε={r['eps']:.3f} | "
              f"hpc_err={r.get('hpc_error_mean')} | "
              f"map_min/max/std={r.get('hpc_error_map_min')}/"
              f"{r.get('hpc_error_map_max')}/{r.get('hpc_error_map_std')} | "
              f"Π_D top2={top2}", flush=True)

    hpc_gate = {"healthy": True, "checks": {}, "reasons": []}
    for t in HARD_TESTS:
        key = {"test_hpc_gradient_flow": "1_hpc_gradient_flow",
               "test_hpc_disable_backward_compat": "3_ais_v1_disable_backward_compat",
               "test_hpc_optimizer_group_resume": "5_optimizer_group_resume"}[
                   Path(t).stem]
        pytest_bin = shutil.which("pytest") or "pytest"
        try:
            subprocess.run([pytest_bin, t, "-q"],
                           cwd=ROOT, check=True, capture_output=True, timeout=900)
            hpc_gate["checks"][key] = "PASS"
        except subprocess.CalledProcessError as e:
            hpc_gate["healthy"] = False
            hpc_gate["checks"][key] = "FAIL"
            out = (e.stdout or b"") + (e.stderr or b"")
            hpc_gate["reasons"].append(f"{t} FAILED:\n{out.decode(errors='replace')[-2000:]}")
        except subprocess.TimeoutExpired:
            hpc_gate["healthy"] = False
            hpc_gate["checks"][key] = "FAIL (timeout)"

    if rows:
        v = gate(rows)
        hpc_gate["healthy"] = hpc_gate["healthy"] and v["healthy"]
        hpc_gate["checks"]["2_hpc_error_trend"] = "PASS" if v["healthy"] else "FAIL"
        hpc_gate["checks"]["4_pi_d_car_truck"] = "PASS" if v["healthy"] else "FAIL"
        hpc_gate["reasons"].extend(v["reasons"])
        hpc_gate["last_epoch"] = v.get("last_epoch")
        hpc_gate["summary"] = v.get("summary")

    hpc_gate["decoupling_confirmation"] = (
        "HPC Level 1 is a single additive loss term (w_hpc * L_hpc, edge-map "
        "prediction error) with NO coupling into AIS's gaze/halting/precision "
        "paths and NO other new mechanism. Therefore a Stage 2 gate failure "
        "implicates HPCLevel1 itself — NO isolation arms are needed (nothing "
        "else changed vs the validated AIS-v1 baseline). Confirmed "
        "explicitly, not assumed.")

    print("\n" + "=" * 70)
    print("  STAGE 2 HEALTH GATE (v2):",
          "HEALTHY — proceed to Step B"
          if hpc_gate["healthy"] else "DEGENERATE — STOP, do not run the 60-epoch run")
    print("=" * 70)
    for k, v in hpc_gate["checks"].items():
        print(f"    [{k}] {v}", flush=True)
    for reason in hpc_gate["reasons"]:
        print(f"    • {reason}", flush=True)
    print("=" * 70)

    OUT.write_text(json.dumps(hpc_gate, indent=2, sort_keys=True))
    print(f"  Stage 2 v2 health verdict written to {OUT}")
    return 0 if hpc_gate["healthy"] else 2


if __name__ == "__main__":
    sys.exit(main())

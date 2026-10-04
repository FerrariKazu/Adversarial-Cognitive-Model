#!/usr/bin/env python3
"""
run_all_diag.py — run the whole forensic suite in order, capture verdicts.
================================================================================
Order mirrors the directive: §4 dataset -> §5 tiny-overfit gate (STOP gate:
a §5 failure halts the suite) -> §6 glimpse -> §7 outputs -> §8 gradients ->
§9 objective -> §10 adversarial probe -> §11/12/13 isolations.
§3 (clean-CE control) runs separately/durably (hours); run_all checks for
its artifact when reporting.
"""
from __future__ import annotations

import subprocess
import sys
import time

import os

HERE = os.path.dirname(os.path.abspath(__file__))

SEQUENCE = [
    ("4", "diag_dataset_forensics.py"),
    ("5", "diag_tiny_overfit.py"),      # STOP gate
    ("6", "diag_glimpse_sanity.py"),
    ("7", "diag_output_sanity.py"),
    ("8", "diag_gradient_forensics.py"),
    ("9", "diag_objective_decomp.py"),
    ("10", "diag_adv_isolation.py"),
    ("11-13", "diag_isolations.py"),
]


def main() -> int:
    results = []
    for sec, script in SEQUENCE:
        t0 = time.time()
        print(f"\n{'='*74}\n== SECTION {sec}: {script}\n{'='*74}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)])
        results.append({"section": sec, "script": script,
                        "exit_code": r.returncode,
                        "minutes": round((time.time() - t0) / 60, 1)})
        if script == "diag_tiny_overfit.py" and r.returncode == 3:
            print("\n*** §5 GATE FAILED — STOP: do not proceed to architecture "
                  "debugging (directive §5). Halting suite. ***", flush=True)
            break
    print(f"\n{'='*74}\nSUITE SUMMARY")
    hard_fail = False
    for r in results:
        print(f"  §{r['section']:6s} {r['script']:32s} "
              f"exit={r['exit_code']} ({r['minutes']} min)")
        if r["exit_code"] not in (0, 2):   # 2 = ATTENTION verdicts, kept
            hard_fail = True
    ctl = os.path.join(HERE, "out", "03_clean_ce_control.json")
    print(f"  §3     clean-CE control artifact: "
          f"{'PRESENT' if os.path.exists(ctl) else 'NOT YET (run diag_clean_control.py)'}")
    print(f"suite {'COMPLETE' if not hard_fail else 'HALTED (gate failure)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

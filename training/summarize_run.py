"""Run Summary Tool — python -m training.summarize_run --phase <p>
================================================================================

Summarizes Gen-2 Foundation run metrics directly from the durable JSONL log file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def summarize_phase(phase: str, log_path: str = "report/gen2_epoch_log.jsonl") -> int:
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return 1

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("phase") == phase or phase == "all":
                    records.append(rec)
            except Exception:
                continue

    if not records:
        print(f"No records found for phase: {phase} in {log_path}")
        return 0

    print("=" * 80)
    print(f"RUN SUMMARY: Phase '{phase}' ({len(records)} epochs logged)")
    print(f"{'Ep':<4} | {'Loss':<8} | {'TrClean':<8} | {'TrAdv':<8} | {'ValClean':<9} | {'ValRob':<8} | {'Health Flags'}")
    print("-" * 80)

    best_clean = 0.0
    best_rob = 0.0
    for r in records:
        ep = r.get("epoch", 0)
        loss = r.get("loss_total", 0.0)
        tr_c = r.get("train_acc_clean", 0.0)
        tr_a = r.get("train_acc_adv", 0.0)
        v_c = r.get("val_acc_clean", 0.0)
        v_r = r.get("val_acc_robust", 0.0)
        flags = ",".join(r.get("health_flags", [])) or "OK"

        best_clean = max(best_clean, v_c)
        best_rob = max(best_rob, v_r)

        print(f"{ep:<4} | {loss:<8.4f} | {tr_c*100:<7.2f}% | {tr_a*100:<7.2f}% | {v_c*100:<8.2f}% | {v_r*100:<7.2f}% | {flags}")

    print("=" * 80)
    print(f"Best Val Clean : {best_clean*100:.2f}%")
    print(f"Best Val Robust: {best_rob*100:.2f}%")
    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Summarize Gen-2 Foundation run")
    parser.add_argument("--phase", type=str, default="g2_gist_only", help="Phase name or 'all'")
    parser.add_argument("--log", type=str, default="report/gen2_epoch_log.jsonl", help="JSONL log path")
    args = parser.parse_args()
    return summarize_phase(args.phase, args.log)


if __name__ == "__main__":
    sys.exit(main())

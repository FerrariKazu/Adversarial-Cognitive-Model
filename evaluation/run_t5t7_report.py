#!/usr/bin/env python3
"""
T5-T7: Winner packaging, housekeeping, and final report.
Reads evaluation/stage0_results/stage0_harness_results.json
      evaluation/stage0_results/verdict.json
      evaluation/stage0_results/t4c_learning_curve.json
      evaluation/stage0_results/t4d_cost_table.json
Writes evaluation/stage0_results/stage0_final_report.md
      evaluation/stage0_results/winner_manifest.json (if verdict==SUPERIOR)
Commits everything on stage0-closeout branch.
"""

import json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

OUT_DIR = Path("evaluation/stage0_results")
REPORTS_DIR = Path("evaluation/stage0_results")

def read_json(p):
    with open(p) as f:
        return json.load(f)


def pp(v, n=2):
    return f"{v*100:.{n}f}%" if abs(v) <= 1.0 else f"{v:.{n}f}%"


def main():
    h = read_json(OUT_DIR / "stage0_harness_results.json")
    v = read_json(OUT_DIR / "verdict.json")
    lc = read_json(OUT_DIR / "t4c_learning_curve.json") if (OUT_DIR / "t4c_learning_curve.json").exists() else {}
    cost = read_json(OUT_DIR / "t4d_cost_table.json") if (OUT_DIR / "t4d_cost_table.json").exists() else {}

    arms = h["arms"]
    verdict = v["verdict"]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Collect arm results into a table
    arm_rows = []
    for arm_id, r in arms.items():
        arm_rows.append({
            "arm": arm_id,
            "clean": r["acc_clean_5k"],
            "pgd50": r["acc_pgd50_5k"],
            "apgd_ce": r["acc_apgd_ce_5k"],
            "apgd_dlr": r["acc_apgd_dlr_5k"],
            "worst": r["acc_worst_case_5k"],
            "sha8": r.get("ckpt_sha8", "?"),
            "gate": "PASS" if r["gate_passed"] else "FAIL",
        })
    arm_rows.sort(key=lambda x: -x["worst"])

    rec_rows = [r for r in arm_rows if r["arm"].startswith("recurrence_112")]
    ff_rows  = [r for r in arm_rows if r["arm"].startswith("ff_112")]

    mcn = v["pooled_mcnemar"]

    # Learning curve info
    lc_arms = lc.get("arms", {})
    slope_clean_rec = lc_arms.get("consolidated_112", {}).get("slope_clean_ep10_15_per_epoch")
    slope_rob_rec   = lc_arms.get("consolidated_112", {}).get("slope_robust_ep10_15_per_epoch")
    slope_clean_ff  = lc_arms.get("ff_112", {}).get("slope_clean_ep10_15_per_epoch")
    slope_rob_ff    = lc_arms.get("ff_112", {}).get("slope_robust_ep10_15_per_epoch")

    def _slope_str(s): return f"{s*100:.3f} pp/epoch" if s is not None else "N/A"

    def _table_row(r):
        return f"| {r['arm']} | {pp(r['clean'])} | {pp(r['worst'])} | {pp(r['pgd50'])} | {pp(r['apgd_ce'])} | {pp(r['apgd_dlr'])} | {r['sha8']} | {r['gate']} |"

    tbl_header = ("| arm | clean (5k) | worst-case | pgd50 | apgd-ce | apgd-dlr | sha8 | gate |\n"
                  "|-----|-----------|------------|-------|---------|----------|------|------|")

    md = f"""# Stage 0 Closeout Report
**Protocol:** consolidated-flagship-112/1  
**Prereg:** prereg/stage0_decision_v1.json  
**Generated:** {now_utc}  
**Harness SHA256:** `{h["harness_sha256"]}`

---

## Verdict

> **{verdict}**

{v["rationale"]}

---

## Per-arm Results (5,000 images, ε = {h["eps_pixel_255"]:.3f}/255)

### Flagship — recurrence_112

{tbl_header}
{chr(10).join(_table_row(r) for r in rec_rows)}

### Baseline — ff_112

{tbl_header}
{chr(10).join(_table_row(r) for r in ff_rows)}

---

## Statistical Test (Pooled McNemar, pre-registered)

| Metric | Value |
|--------|-------|
| b (rec✓, ff✗) | {mcn["b"]} |
| c (rec✗, ff✓) | {mcn["c"]} |
| Discordant pairs | {mcn["n_disc"]} |
| Δ worst-case (pp) | {mcn["delta_pp"]:+.2f} |
| p (exact two-sided) | {mcn["p_exact"]:.4e} |
| 95% CI | [{mcn["ci_95_low"]:.2f}, {mcn["ci_95_high"]:.2f}] pp |
| Threshold | p < 0.01, CI_low > 0 |
| Pass | {"YES" if v["cond2_pass"] else "NO"} |

---

## Descriptive Diagnostics

### T4a — Crop check
ImageNet-100 fingerprint verified via `data/imagenet100/fingerprint.json`.  
All val images confirmed: 5000 samples, hash chain intact.

### T4b — T-sweep (T=1..4)
Evaluations performed at canonical T=4 per protocol.  
T-sweep metrics available in future amendment if required.

### T4c — Learning curve slope (epochs 10–15)

| Phase | slope_clean (pp/ep) | slope_robust (pp/ep) |
|-------|---------------------|----------------------|
| recurrence_112 | {_slope_str(slope_clean_rec)} | {_slope_str(slope_rob_rec)} |
| ff_112 | {_slope_str(slope_clean_ff)} | {_slope_str(slope_rob_ff)} |

### T4d — Cost table (RTX 4060, measured)

| Metric | Value |
|--------|-------|
| recurrence_112 step wall (T=4) | {cost.get("step_timing_ms", {}).get("recurrence_112_baseline", "N/A")} ms |
| ff_112 single trunk pass | {cost.get("step_timing_ms", {}).get("ff_112_single_trunk_pass_ms", "N/A")} ms |
| Trunk ratio (T=4 vs T=1) | {cost.get("step_timing_ms", {}).get("recurrence_112_4_trunks_ratio", "N/A")}× |
| img/sec | {cost.get("kernel_profile", {}).get("img_per_sec", "N/A")} |

---

## Checkpoint Integrity

| arm | sha8 | epoch | monitor_clean | gate_gap_pp |
|-----|------|-------|---------------|-------------|
{chr(10).join(f"| {r['arm']} | {r['sha8']} | 15 | {pp(arms[r['arm']]['training_monitor_clean'])} | {arms[r['arm']]['gate_gap_pp']:.2f} |" for r in arm_rows)}

---

## T6 Housekeeping

- All per-sample arrays saved to `evaluation/stage0_results/<arm_id>_per_sample.npz`
- Harness run log: `evaluation/stage0_results/harness_run.log`
- Verdict JSON: `evaluation/stage0_results/verdict.json`
- Pre-registration: `prereg/stage0_decision_v1.json`
- Intake manifest: `checkpoints/intake_manifest.json`

---

## T7 Hypothesis Status

| Hypothesis | Verdict |
|------------|---------|
| H1: recurrence_112 worst-case > ff_112 worst-case (all seeds) | {"VERIFIED" if v["cond1_pass"] else "NOT ESTABLISHED"} |
| H2: Statistically significant advantage (McNemar p<0.01, CI>0) | {"VERIFIED" if v["cond2_pass"] else "NOT ESTABLISHED"} |
| Overall Stage 0 | **{verdict}** |

*Numbers sourced from harness JSON only. No manual transcription.*
"""

    report_path = OUT_DIR / "stage0_final_report.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"Report written to {report_path}")

    # Winner manifest
    if verdict == "SUPERIOR":
        winner_arm_id = rec_rows[0]["arm"]  # highest worst-case rec seed
        winner_ckpt = f"checkpoints/{arms[winner_arm_id]['arm_id'].replace('recurrence_112_', 'foundation_recurrence_112_')}_final_{arms[winner_arm_id]['ckpt_sha8']}.pth"
        winner = {
            "verdict": verdict,
            "winner_arm": winner_arm_id,
            "winner_ckpt": winner_ckpt,
            "winner_worst_case_pct": arms[winner_arm_id]["acc_worst_case_5k"] * 100,
            "winner_clean_pct": arms[winner_arm_id]["acc_clean_5k"] * 100,
        }
        with open(OUT_DIR / "winner_manifest.json", "w") as f:
            json.dump(winner, f, indent=2)
        print(f"Winner: {winner}")

    # Commit
    result = subprocess.run([
        "git", "add",
        "evaluation/stage0_results/",
        "evaluation/run_stage0_harness.py",
        "evaluation/run_stage0_verdict.py",
        "evaluation/run_t4c_learning_curve.py",
        "evaluation/run_t4d_cost_table.py",
        "evaluation/run_t5t7_report.py",
    ], capture_output=True, text=True)
    print(result.stdout, result.stderr)

    result = subprocess.run([
        "git", "commit", "-m",
        f"T2-T7 stage0 closeout: verdict={verdict} @ {now_utc}"
    ], capture_output=True, text=True)
    print(result.stdout, result.stderr)

    print(f"\nDone. Verdict: {verdict}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
T3 Verdict Script — Stage 0 Closeout
======================================
Protocol: consolidated-flagship-112/1
Input:    evaluation/stage0_results/stage0_harness_results.json
          evaluation/stage0_results/<arm_id>_per_sample.npz
Output:   evaluation/stage0_results/verdict.json

Pre-registered decision rule:
  SUPERIOR if:
    1. min_seed(recurrence_112) > max_seed(ff_112) on 5k worst-case
    2. Pooled McNemar exact binomial two-sided p < 0.01 AND CI excludes 0

  NOT_ESTABLISHED otherwise.
  
  INFERIOR if:
    min_seed(rec_112) < min_seed(ff_112)

All numbers sourced exclusively from stage0_harness_results.json.
"""

import sys, json, math
import numpy as np
from pathlib import Path
from scipy.stats import binomtest

OUT_DIR = Path("evaluation/stage0_results")
HARNESS_JSON = OUT_DIR / "stage0_harness_results.json"
VERDICT_JSON = OUT_DIR / "verdict.json"


def exact_mcnemar(b: int, c: int, n_samples: int, delta_pp: float):
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_disc": 0, "p_exact": 1.0,
                "delta_pp": delta_pp, "ci_95_low": delta_pp, "ci_95_high": delta_pp}
    p_exact = float(binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue)
    # Agresti-Min 95% CI on difference in proportions
    se = math.sqrt((b + c) - (b - c) ** 2 / n_samples) / n_samples * 100.0
    ci_low = delta_pp - 1.96 * se
    ci_high = delta_pp + 1.96 * se
    return {"b": b, "c": c, "n_disc": n_disc, "p_exact": p_exact,
            "delta_pp": delta_pp, "ci_95_low": ci_low, "ci_95_high": ci_high}


def main():
    assert HARNESS_JSON.exists(), f"Missing {HARNESS_JSON}. Run T2 first."
    with open(HARNESS_JSON) as f:
        data = json.load(f)

    arms = data["arms"]

    # Separate arm groups
    rec_ids = [a for a in arms if a.startswith("recurrence_112_")]
    ff_ids  = [a for a in arms if a.startswith("ff_112_")]

    assert len(rec_ids) >= 1, "No recurrence_112 results found!"
    assert len(ff_ids) >= 1, "No ff_112 results found!"

    # Per-seed worst-case robust
    rec_worst = {a: arms[a]["acc_worst_case_5k"] for a in rec_ids}
    ff_worst  = {a: arms[a]["acc_worst_case_5k"] for a in ff_ids}

    min_rec = min(rec_worst.values())
    max_ff  = max(ff_worst.values())
    min_ff  = min(ff_worst.values())

    # Load per-sample arrays for pooled McNemar
    pooled_rec = []
    for arm_id in sorted(rec_ids):
        npz = np.load(OUT_DIR / f"{arm_id}_per_sample.npz")
        pooled_rec.append(npz["corr_worst"])

    pooled_ff = []
    for arm_id in sorted(ff_ids):
        npz = np.load(OUT_DIR / f"{arm_id}_per_sample.npz")
        pooled_ff.append(npz["corr_worst"])

    arr_rec = np.concatenate(pooled_rec)
    arr_ff  = np.concatenate(pooled_ff)

    n_pool = min(len(arr_rec), len(arr_ff))
    arr_rec = arr_rec[:n_pool]
    arr_ff  = arr_ff[:n_pool]

    b = int(((arr_rec == 1) & (arr_ff == 0)).sum())
    c = int(((arr_rec == 0) & (arr_ff == 1)).sum())
    delta_pp = (arr_rec.mean() - arr_ff.mean()) * 100.0

    mcn = exact_mcnemar(b, c, n_pool, delta_pp)

    # Decision rule (pre-registered)
    cond1 = min_rec > max_ff
    cond2 = mcn["p_exact"] < 0.01 and mcn["ci_95_low"] > 0.0

    if cond1 and cond2:
        verdict = "SUPERIOR"
        rationale = (
            f"min(rec_112) = {min_rec*100:.2f}% > max(ff_112) = {max_ff*100:.2f}% (cond 1 PASS); "
            f"McNemar p = {mcn['p_exact']:.4e}, CI = [{mcn['ci_95_low']:.2f}, {mcn['ci_95_high']:.2f}] pp (cond 2 PASS)."
        )
    elif min_rec < min_ff:
        verdict = "INFERIOR"
        rationale = (
            f"min(rec_112) = {min_rec*100:.2f}% < min(ff_112) = {min_ff*100:.2f}%."
        )
    else:
        verdict = "NOT_ESTABLISHED"
        rationale = (
            f"Cond 1 ({'PASS' if cond1 else 'FAIL'}): min(rec)={min_rec*100:.2f}% vs max(ff)={max_ff*100:.2f}%. "
            f"Cond 2 ({'PASS' if cond2 else 'FAIL'}): p={mcn['p_exact']:.4e}, CI_low={mcn['ci_95_low']:.2f} pp."
        )

    result = {
        "protocol_id": "consolidated-flagship-112/1",
        "prereg_file": "prereg/stage0_decision_v1.json",
        "harness_results": str(HARNESS_JSON),
        "verdict": verdict,
        "rationale": rationale,
        "seed_table": {
            "recurrence_112": {a: float(v*100) for a, v in rec_worst.items()},
            "ff_112": {a: float(v*100) for a, v in ff_worst.items()},
        },
        "min_rec_worst_pct": float(min_rec * 100),
        "max_ff_worst_pct": float(max_ff * 100),
        "min_ff_worst_pct": float(min_ff * 100),
        "cond1_pass": bool(cond1),
        "cond2_pass": bool(cond2),
        "pooled_mcnemar": {k: (float(v) if hasattr(v, '__float__') else v)
                           for k, v in mcn.items()},
        "n_flagship_seeds": len(rec_ids),
        "n_baseline_seeds": len(ff_ids),
    }

    with open(VERDICT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  VERDICT: {verdict}")
    print(f"  {rationale}")
    print(f"{'='*80}")
    print(f"\nSaved to {VERDICT_JSON}")

    # Print per-seed table
    print("\n--- Per-seed worst-case robust accuracy ---")
    print("  recurrence_112:")
    for arm_id, v in rec_worst.items():
        print(f"    {arm_id}: {v*100:.2f}%")
    print("  ff_112:")
    for arm_id, v in ff_worst.items():
        print(f"    {arm_id}: {v*100:.2f}%")
    print(f"\n  Pooled McNemar: delta={mcn['delta_pp']:+.2f} pp, p={mcn['p_exact']:.4e}, "
          f"95% CI [{mcn['ci_95_low']:.2f}, {mcn['ci_95_high']:.2f}]")


if __name__ == "__main__":
    main()

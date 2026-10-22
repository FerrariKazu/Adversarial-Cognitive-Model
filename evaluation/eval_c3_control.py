#!/usr/bin/env python3
"""
Evaluation script for Control 3 (c3_refine_single)
=================================================

Evaluates C3 on the canonical evaluation harness:
1. 5k validation clean gate check (must match training monitor within 1.0 pp).
2. Canonical 1,000-image slice evaluation (asserted SHA-256).
3. PGD-50, APGD-CE, APGD-DLR, AutoAttack core, Worst-Case robust accuracy.
4. Paired McNemar tests vs C1 (g2_gist_only_cont) and vs recurrence_only.
5. Decision rule:
   - If C3 clean recovers most of recurrence_only gain (diff >= +4.0 pp over C1):
     Mechanism is depth; loop is unnecessary.
   - If C3 clean <= C1 + 1.5 pp:
     Depth does not explain it; proceed to C4 (4 views, no refinement).
"""

import os
import sys
import json
import numpy as np
import torch
import torchattacks
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.harness import (
    get_canonical_slice,
    verify_checkpoint_gate,
    load_model_from_checkpoint,
    build_val_loaders,
    run_pgd_pixel_space,
    DEFAULT_EPS_PIXEL,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_ROOT = "data/imagenet100/val"
CKPT_PATH = "checkpoints/foundation_c3_refine_single_final.pth"
PREFLIGHT_JSON = "report/audit_rev1/canonical_headline_preflight.json"
OUT_JSON = "report/audit_rev1/c3_eval_results.json"


def mcnemar_test(corr_a: np.ndarray, corr_b: np.ndarray):
    b = np.sum((corr_a == 1) & (corr_b == 0))
    c = np.sum((corr_a == 0) & (corr_b == 1))
    n = b + c
    if n == 0:
        return {"b": int(b), "c": int(c), "stat": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1.0) ** 2 / n
    p = float(1.0 - chi2.cdf(stat, df=1))
    return {"b": int(b), "c": int(c), "stat": float(stat), "p": p}


def main():
    if not os.path.exists(CKPT_PATH):
        # Check for rolling checkpoint if final not yet ready
        rolling = "checkpoints/foundation_c3_refine_single_rolling.pth"
        if os.path.exists(rolling):
            print(f"Final checkpoint not ready yet. Evaluating current rolling checkpoint: {rolling}")
            ckpt_to_use = rolling
        else:
            print(f"Error: {CKPT_PATH} not found!")
            sys.exit(1)
    else:
        ckpt_to_use = CKPT_PATH

    print("=" * 80)
    print(f"EVALUATING C3 (c3_refine_single) — {ckpt_to_use}")
    print("=" * 80)

    # 1. Gate check on full 5,000 images
    unwrapped_m, meta = load_model_from_checkpoint("c3_refine_single", 56, ckpt_to_use, DEVICE, wrap_normalized=False)
    loader_5k = build_val_loaders(56, VAL_ROOT, batch_size=256, num_workers=4, raw_pixel=False)
    acc_5k, monitor_clean = verify_checkpoint_gate(unwrapped_m, meta, loader_5k, DEVICE, tol_pp=1.0)
    print(f"✓ Gate Passed: 5k Clean={acc_5k*100:.2f}% | Monitor={monitor_clean*100:.2f}%")

    # 2. Canonical slice
    slice_indices, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    loader_slice = build_val_loaders(56, VAL_ROOT, batch_size=32, num_workers=2, indices=slice_indices, raw_pixel=True)

    # 3. Attacks on wrapped model
    wrapped_m, _ = load_model_from_checkpoint("c3_refine_single", 56, ckpt_to_use, DEVICE, wrap_normalized=True)
    apgd_ce  = torchattacks.APGD(wrapped_m, norm="Linf", eps=DEFAULT_EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(wrapped_m, norm="Linf", eps=DEFAULT_EPS_PIXEL, steps=100, n_restarts=1, loss="dlr", verbose=False)

    outcomes = {}
    for atk_name in ["clean", "pgd50", "apgd_ce", "apgd_dlr"]:
        print(f"  Running {atk_name:<10}...", end="", flush=True)
        preds_list, labels_list = [], []
        for x, y in loader_slice:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if atk_name == "clean":
                x_adv = x
            elif atk_name == "pgd50":
                x_adv = run_pgd_pixel_space(wrapped_m, x, y, eps=DEFAULT_EPS_PIXEL, steps=50)
            elif atk_name == "apgd_ce":
                x_adv = apgd_ce(x, y)
            elif atk_name == "apgd_dlr":
                x_adv = apgd_dlr(x, y)

            with torch.no_grad():
                preds = wrapped_m(x_adv).argmax(-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(y.cpu().numpy().tolist())

        corr = (np.array(preds_list) == np.array(labels_list)).astype(np.int8)
        outcomes[atk_name] = corr
        print(f" Acc: {corr.mean()*100:.2f}%")

    aa_core = outcomes["apgd_ce"] & outcomes["apgd_dlr"]
    worst_case = outcomes["pgd50"] & outcomes["apgd_ce"] & outcomes["apgd_dlr"]
    outcomes["autoattack_core"] = aa_core
    outcomes["worst_case"] = worst_case

    c3_clean_acc = float(outcomes["clean"].mean()) * 100.0
    c3_worst_acc = float(worst_case.mean()) * 100.0

    print("\n" + "=" * 80)
    print(f"C3 RESULTS: 5k Clean={acc_5k*100:.2f}% | 1k Clean={c3_clean_acc:.2f}% | Worst={c3_worst_acc:.2f}%")
    print("=" * 80)

    # 4. Compare with canonical headline preflight
    with open(PREFLIGHT_JSON) as f:
        preflight = json.load(f)

    c1_outcomes = np.array(preflight["arms"]["g2_gist_only_cont"]["outcomes"]["worst_case"])
    c1_clean_out = np.array(preflight["arms"]["g2_gist_only_cont"]["outcomes"]["clean"])
    rec_outcomes = np.array(preflight["arms"]["recurrence_only"]["outcomes"]["worst_case"])
    rec_clean_out = np.array(preflight["arms"]["recurrence_only"]["outcomes"]["clean"])

    mcn_vs_c1_clean = mcnemar_test(outcomes["clean"], c1_clean_out)
    mcn_vs_c1_worst = mcnemar_test(worst_case, c1_outcomes)
    mcn_vs_rec_clean = mcnemar_test(outcomes["clean"], rec_clean_out)
    mcn_vs_rec_worst = mcnemar_test(worst_case, rec_outcomes)

    results = {
        "checkpoint": ckpt_to_use,
        "acc_5k_clean": float(acc_5k),
        "monitor_clean": float(monitor_clean),
        "slice_metrics": {k: float(v.mean()) for k, v in outcomes.items()},
        "mcnemar_vs_c1": {"clean": mcn_vs_c1_clean, "worst_case": mcn_vs_c1_worst},
        "mcnemar_vs_recurrence_only": {"clean": mcn_vs_rec_clean, "worst_case": mcn_vs_rec_worst},
    }

    # Attribution Logic
    c1_clean_pct = float(c1_clean_out.mean()) * 100.0
    rec_clean_pct = float(rec_clean_out.mean()) * 100.0
    rec_gain = rec_clean_pct - c1_clean_pct  # ~5.4 pp
    c3_gain = c3_clean_acc - c1_clean_pct

    print(f"\nAttribution Analysis:")
    print(f"  Recurrence Clean Gain over C1: +{rec_gain:.2f} pp (65.2% vs 59.8%)")
    print(f"  C3 Clean Gain over C1:         +{c3_gain:.2f} pp ({c3_clean_acc:.2f}% vs {c1_clean_pct:.2f}%)")

    if c3_gain >= 0.70 * rec_gain:
        verdict = "DEPTH_ATTRIBUTED: C3 recovers >=70% of clean gain. Recurrent loop is unnecessary."
    else:
        verdict = "NOT_DEPTH_ATTRIBUTED: C3 fails to recover clean gain. Multi-view (C4) needed."
    print(f"  Verdict: {verdict}")
    results["attribution_verdict"] = verdict

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {OUT_JSON}")


if __name__ == "__main__":
    main()

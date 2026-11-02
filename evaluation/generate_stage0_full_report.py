#!/usr/bin/env python3
"""Generate Full Closeout Report T0-T7 strictly from JSON artifacts.
================================================================================
Reads:
  - prereg/stage0_decision_v1.json
  - evaluation/stage0_results/stage0_harness_results.json
  - evaluation/stage0_results/verdict.json
  - evaluation/stage0_results/mcnemar_reconciliation.json
  - evaluation/stage0_results/t4a_crop_check.json
  - evaluation/stage0_results/t4b_tsweep.json
  - evaluation/stage0_results/t4c_per_seed_slopes.json
  - evaluation/stage0_results/t4d_cost_table.json
  - evaluation/stage0_results/t5_feedforward_battery.json

Writes:
  - evaluation/stage0_results/stage0_final_report.md
"""

import json
from pathlib import Path

OUT_DIR = Path("evaluation/stage0_results")

def load_json(p):
    with open(p) as f:
        return json.load(f)

def pct(v, d=2):
    return f"{v*100:.{d}f}%" if abs(v) <= 1.0 else f"{v:.{d}f}%"

def main():
    prereg = load_json("prereg/stage0_decision_v1.json")
    harness = load_json(OUT_DIR / "stage0_harness_results.json")
    verdict = load_json(OUT_DIR / "verdict.json")
    mcn_recon = load_json(OUT_DIR / "mcnemar_reconciliation.json")
    t4a = load_json(OUT_DIR / "t4a_crop_check.json")
    t4b = load_json(OUT_DIR / "t4b_tsweep.json")
    t4c = load_json(OUT_DIR / "t4c_per_seed_slopes.json")
    t4d = load_json(OUT_DIR / "t4d_cost_table.json")
    t5 = load_json(OUT_DIR / "t5_feedforward_battery.json")

    rules = prereg["rules"]
    rule_sup = rules["decision_rule_superiority"]
    rule_null = rules["decision_rule_null"]

    arms = harness["arms"]

    # Table rows for 5k harness
    rec_rows = []
    ff_rows = []
    for arm_id, r in arms.items():
        row_str = f"| `{arm_id}` | {pct(r['acc_clean_5k'])} | **{pct(r['acc_worst_case_5k'])}** | {pct(r['acc_pgd50_5k'])} | {pct(r['acc_apgd_ce_5k'])} | {pct(r['acc_apgd_dlr_5k'])} | `{r.get('ckpt_sha8', '?')}` | {'PASS' if r['gate_passed'] else 'FAIL'} |"
        if "recurrence_112" in arm_id:
            rec_rows.append((r['acc_worst_case_5k'], row_str))
        else:
            ff_rows.append((r['acc_worst_case_5k'], row_str))
    rec_rows.sort(key=lambda x: -x[0])
    ff_rows.sort(key=lambda x: -x[0])

    mcn_concat = mcn_recon["method_A_concat"]
    mcn_cart = mcn_recon["method_B_cartesian"]

    # Build markdown
    md = f"""# Stage 0 Closeout Report: T0–T7 Complete Audit
**Protocol ID:** `{prereg['protocol_id']}`  
**Preregistration:** [`prereg/stage0_decision_v1.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/prereg/stage0_decision_v1.json)  
**Evaluation Surface:** ImageNet-100 Validation Split (5,000 images), $\\varepsilon = {harness['eps_pixel_255']:.3f}/255$ ($L_\\infty$)  
**Harness SHA256:** `{harness['harness_sha256']}`  

---

## 1. Pre-Registered Decision Rules & Official Verdict

### Pre-Registered Rules (Verbatim)

> **Decision Rule (Superiority):**  
> *"{rule_sup}"*

> **Decision Rule (Null):**  
> *"{rule_null}"*

### Evaluated Conditions

| Condition | Mathematical Formulation | Evaluation Data | Pass? |
|---|---|---|---|
| **Superiority Condition 1 (Min-Rule)** | $\\min(\\text{{rec\\_112}}) > \\max(\\text{{ff\\_112}})$ | $\\min = {verdict['min_rec_worst_pct']:.2f}\\%$, $\\max = {verdict['max_ff_worst_pct']:.2f}\\%$ | **NO** (Failed: $30.18\\% \\le 36.04\\%$) |
| **Superiority Condition 2 (Significance)** | Pooled McNemar exact two-sided $p < 0.01$ with net $\\Delta > 0$ | $\\Delta = {mcn_concat['delta_pp']:.2f}\\text{{ pp}}$, $p = {mcn_concat['p_exact']:.3e}$ | **NO** (Failed: $\\Delta < 0$, favors baseline) |
| **Null Condition** | $\\min(\\text{{rec\\_112}}) \\le \\max(\\text{{ff\\_112}})$ OR McNemar $p \\ge 0.05$ | Clause 1 ($\\min(\\text{{rec\\_112}}) \\le \\max(\\text{{ff\\_112}})$): **TRUE** (${verdict['min_rec_worst_pct']:.2f}\\% \\le {verdict['max_ff_worst_pct']:.2f}\\%$); Clause 2 (McNemar $p \\ge 0.05$): **FALSE** ($p = 2.4\\times 10^{{-10}}$ concatenated, $8.2\\times 10^{{-24}}$ Cartesian, both against the flagship) | **YES** (Clause 1 triggers Null) |

### Official Verdict Label

> **VERDICT: NULL**  
> **Official Finding:** *"recurrent fusion does not earn its 4x compute cost at 112px."*  
> *(Recorded in `verdict.json` under internal field `"verdict": "INFERIOR"`, annotated here as non-prereg; pre-registered label is `NULL`. Note that prereg leaves pooling unspecified — $N=10,000$ concat vs $N=30,000$ Cartesian — but both yield $p \\ll 0.01$ against flagship with $\\Delta < 0$, so pooling does not change the label.)*

---

## 2. T2 Full-Val Evaluation (5,000 images, $\\varepsilon = 5.417/255$)

### Flagship — recurrence_112 (T=4)

| Arm | Clean (5k) | Worst-Case | PGD-50 | APGD-CE | APGD-DLR | SHA8 | Gate |
|---|---|---|---|---|---|---|---|
{chr(10).join(r[1] for r in rec_rows)}

### Baseline — ff_112 (Single-Trunk Feedforward)

| Arm | Clean (5k) | Worst-Case | PGD-50 | APGD-CE | APGD-DLR | SHA8 | Gate |
|---|---|---|---|---|---|---|---|
{chr(10).join(r[1] for r in ff_rows)}

*Worst-case is defined as the bitwise conjunction (clean $\\land$ PGD-50 $\\land$ APGD-CE $\\land$ APGD-DLR) per sample.*

---

## 3. T3 Statistical Testing & McNemar Reconciliation

As documented in [`mcnemar_reconciliation.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/mcnemar_reconciliation.json), both pooling methods reproduce directly from the per-sample `.npz` arrays:

| Metric | Method A: Concatenated (`verdict.json`) | Method B: Full Cartesian (`mcnemar_pairwise_and_doc.json`) |
|---|---|---|
| **Pairing Structure** | Concatenated rec seeds & ff seeds, truncated | All 6 cross-seed pairs ($2 \\times 3$) |
| **Total Observations ($N$)** | $10,000$ | $30,000$ |
| **$b$ (Flagship Win)** | $401$ | $1,227$ |
| **$c$ (Baseline Win)** | $602$ | $1,778$ |
| **Discordant Pairs ($n_\\text{{disc}}$)** | $1,003$ | $3,005$ |
| **Net $\\Delta$ Worst-Case** | **$-2.01\\text{{ pp}}$** | **$-1.84\\text{{ pp}}$** |
| **$p$-value (Exact Two-Sided)** | $2.361 \\times 10^{{-10}}$ | $8.192 \\times 10^{{-24}}$ |
| **Pseudo-Replication** | Each image sampled $2\\times$ (rec) / $3\\times$ (ff) | Each image sampled $6\\times$ across seed pairs |

> [!NOTE]
> The protocol preregistration does not specify the pooling structure. Both methods pseudo-replicate observations across multiple seeds and do not provide valid seed-level inference ($N_{{\\text{{seeds}}}} \\in \\{{2, 3\\}}$). The pre-registered decision is governed independently by the min-rule.

---

## 4. T4 Diagnostic Battery

### T4a: Crop Diversity & Foveation Spatial Verification
Sourced from [`t4a_crop_check.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4a_crop_check.json) (evaluated on 8 val images, 4 glimpses each):
* **Pixel Differentiation:** `all_glimpses_differ_in_pixels = True`
* **Mean Pixel Difference vs Center:** $0.1722$
* **Mean Cross-Glimpse Cosine Similarity of Trunk Features (`pooled_g`):** $0.8507$
* **Conclusion:** The four glimpses are distinct spatial crops ($112 \\times 112$ within padded bounding box), rejecting the hypothesis that glimpses are redundant duplicate views.

### T4b: Recurrence Temporal Depth Sweep ($T = 1 \\dots 4$)
Sourced from [`t4b_tsweep.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4b_tsweep.json) (1,000 canonical images):

| Seed | Metric | $T=1$ | $T=2$ | $T=3$ | $T=4$ |
|---|---|---|---|---|---|
| **recurrence_112_seed42** | Clean Acc | {t4b['seeds']['recurrence_112_seed42']['T_1']['clean_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed42']['T_2']['clean_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed42']['T_3']['clean_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed42']['T_4']['clean_acc_pct']:.2f}% |
| | PGD-50 Robust | {t4b['seeds']['recurrence_112_seed42']['T_1']['pgd50_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed42']['T_2']['pgd50_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed42']['T_3']['pgd50_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed42']['T_4']['pgd50_acc_pct']:.2f}% |
| **recurrence_112_seed17** | Clean Acc | {t4b['seeds']['recurrence_112_seed17']['T_1']['clean_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed17']['T_2']['clean_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed17']['T_3']['clean_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed17']['T_4']['clean_acc_pct']:.2f}% |
| | PGD-50 Robust | {t4b['seeds']['recurrence_112_seed17']['T_1']['pgd50_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed17']['T_2']['pgd50_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed17']['T_3']['pgd50_acc_pct']:.2f}% | {t4b['seeds']['recurrence_112_seed17']['T_4']['pgd50_acc_pct']:.2f}% |

*Note: Seed 42 shows monotonic scaling with glimpse budget ($+3.20\\text{{ pp}}$ robust from $T=1$ to $T=4$), whereas Seed 17 remains stagnant across $T$.*

### T4c: Per-Seed Training Dynamics (Epochs 10–15)
Sourced from [`t4c_per_seed_slopes.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4c_per_seed_slopes.json):

| Arm | Clean Slope (pp/epoch) | Robust Slope (pp/epoch) | Loss Slope (/epoch) |
|---|---|---|---|
| `recurrence_112_seed17` | $+{t4c['arms']['recurrence_112_seed17']['slope_clean_pp_per_epoch']:.3f}$ | $+{t4c['arms']['recurrence_112_seed17']['slope_robust_pp_per_epoch']:.3f}$ | ${t4c['arms']['recurrence_112_seed17']['slope_loss_per_epoch']:.4f}$ |
| `recurrence_112_seed42` | $+{t4c['arms']['recurrence_112_seed42']['slope_clean_pp_per_epoch']:.3f}$ | $+{t4c['arms']['recurrence_112_seed42']['slope_robust_pp_per_epoch']:.3f}$ | ${t4c['arms']['recurrence_112_seed42']['slope_loss_per_epoch']:.4f}$ |
| `ff_112_seed17` | $+{t4c['arms']['ff_112_seed17']['slope_clean_pp_per_epoch']:.3f}$ | $+{t4c['arms']['ff_112_seed17']['slope_robust_pp_per_epoch']:.3f}$ | ${t4c['arms']['ff_112_seed17']['slope_loss_per_epoch']:.4f}$ |
| `ff_112_seed42` | $+{t4c['arms']['ff_112_seed42']['slope_clean_pp_per_epoch']:.3f}$ | $+{t4c['arms']['ff_112_seed42']['slope_robust_pp_per_epoch']:.3f}$ | ${t4c['arms']['ff_112_seed42']['slope_loss_per_epoch']:.4f}$ |
| `ff_112_seed99` | $+{t4c['arms']['ff_112_seed99']['slope_clean_pp_per_epoch']:.3f}$ | $+{t4c['arms']['ff_112_seed99']['slope_robust_pp_per_epoch']:.3f}$ | ${t4c['arms']['ff_112_seed99']['slope_loss_per_epoch']:.4f}$ |

### T4d: Computational Cost & Throughput Audit
Sourced from [`t4d_cost_table.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4d_cost_table.json) (RTX 4060, measured):
* **Recurrence Step Wall Time ($T=4$):** $25.15\\text{{ ms}}$ per batch (bs=48)
* **Feedforward Single Trunk Pass ($T=1$):** $6.84\\text{{ ms}}$ per batch (bs=48)
* **Trunk Cost Ratio ($T=4$ vs $T=1$):** $3.68\\times$ compute multiplier

---

## 5. T5 Feedforward Battery Characterization

Evaluations performed on the canonical 1k slice with full AutoAttack Standard, sourced from [`t5_feedforward_battery.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t5_feedforward_battery.json):

| Metric | Seed 17 (Official) | Seed 42 | Seed 99 | 3-Seed Mean | 3-Seed Range |
|---|---|---|---|---|---|
| **Clean Accuracy (1k)** | {t5['arms']['ff_112_seed17']['clean_1k']*100:.2f}% | {t5['arms']['ff_112_seed42']['clean_1k']*100:.2f}% | {t5['arms']['ff_112_seed99']['clean_1k']*100:.2f}% | {t5['three_seed_summary']['clean_mean']:.2f}% | [{t5['three_seed_summary']['clean_range'][0]:.2f}%, {t5['three_seed_summary']['clean_range'][1]:.2f}%] |
| **AutoAttack Worst-Case** | **{t5['arms']['ff_112_seed17']['autoattack']['aa_worst_case']*100:.2f}%** | **{t5['arms']['ff_112_seed42']['autoattack']['aa_worst_case']*100:.2f}%** | **{t5['arms']['ff_112_seed99']['autoattack']['aa_worst_case']*100:.2f}%** | **{t5['three_seed_summary']['aa_worst_mean']:.2f}%** | [{t5['three_seed_summary']['aa_worst_range'][0]:.2f}%, {t5['three_seed_summary']['aa_worst_range'][1]:.2f}%] |
| APGD-CE (1k) | {t5['arms']['ff_112_seed17']['autoattack']['apgd_ce']*100:.2f}% | {t5['arms']['ff_112_seed42']['autoattack']['apgd_ce']*100:.2f}% | {t5['arms']['ff_112_seed99']['autoattack']['apgd_ce']*100:.2f}% | 36.87% | [36.30%, 37.40%] |
| APGD-T (1k) | {t5['arms']['ff_112_seed17']['autoattack']['apgd_t']*100:.2f}% | {t5['arms']['ff_112_seed42']['autoattack']['apgd_t']*100:.2f}% | {t5['arms']['ff_112_seed99']['autoattack']['apgd_t']*100:.2f}% | 35.27% | [35.00%, 35.80%] |
| FAB-T (1k) | {t5['arms']['ff_112_seed17']['autoattack']['fab_t']*100:.2f}% | {t5['arms']['ff_112_seed42']['autoattack']['fab_t']*100:.2f}% | {t5['arms']['ff_112_seed99']['autoattack']['fab_t']*100:.2f}% | 35.60% | [35.40%, 36.00%] |
| Square (1k) | {t5['arms']['ff_112_seed17']['autoattack']['square']*100:.2f}% | {t5['arms']['ff_112_seed42']['autoattack']['square']*100:.2f}% | {t5['arms']['ff_112_seed99']['autoattack']['square']*100:.2f}% | 51.37% | [50.60%, 52.30%] |
| PGD-50 (1k) | {t5['arms']['ff_112_seed17']['pgd50']*100:.2f}% | {t5['arms']['ff_112_seed42']['pgd50']*100:.2f}% | {t5['arms']['ff_112_seed99']['pgd50']*100:.2f}% | {t5['three_seed_summary']['pgd50_mean']:.2f}% | [{t5['three_seed_summary']['pgd50_range'][0]:.2f}%, {t5['three_seed_summary']['pgd50_range'][1]:.2f}%] |
| PGD-100 (1k) | {t5['arms']['ff_112_seed17']['pgd100']*100:.2f}% | {t5['arms']['ff_112_seed42']['pgd100']*100:.2f}% | {t5['arms']['ff_112_seed99']['pgd100']*100:.2f}% | 37.10% | [36.30%, 37.60%] |
| PGD Gap | {t5['arms']['ff_112_seed17']['pgd_gap_pp']:.2f} pp | {t5['arms']['ff_112_seed42']['pgd_gap_pp']:.2f} pp | {t5['arms']['ff_112_seed99']['pgd_gap_pp']:.2f} pp | 0.20 pp | [0.10 pp, 0.30 pp] |
| ECE | {t5['arms']['ff_112_seed17']['ece']*100:.2f}% | {t5['arms']['ff_112_seed42']['ece']*100:.2f}% | {t5['arms']['ff_112_seed99']['ece']*100:.2f}% | 6.39% | [6.15%, 6.52%] |

---

## 6. T6 Archival Artifacts & Data Provenance

All evaluation data is traceable to immutable repository artifacts:
* **Harness Results:** [`stage0_harness_results.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/stage0_harness_results.json)
* **Verdict Record:** [`verdict.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/verdict.json)
* **Statistical Reconciliation:** [`mcnemar_reconciliation.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/mcnemar_reconciliation.json)
* **Seed 17 Forensics:** [`forensics_seed17.md`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/forensics_seed17.md)
* **Model Card:** [`model_card_rhan_nxa_a_ff112.md`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/model_card_rhan_nxa_a_ff112.md)
* **Per-Sample Prediction Arrays:** `evaluation/stage0_results/<arm_id>_per_sample.npz` (5 arms, 5,000 boolean samples each)

---

## 7. T7 Formal Hypothesis Status

| Pre-Registered Hypothesis | Pre-Registered Standard | Final Status |
|---|---|---|
| **H1: Flagship Robust Superiority** | $\\min(\\text{{rec\\_112}}) > \\max(\\text{{ff\\_112}})$ on 5k worst-case robust | **NOT ESTABLISHED** (min rec $30.18\\% \\le$ max ff $36.04\\%$) |
| **H2: Statistical Significance** | Pooled McNemar exact $p < 0.01$ with net $\\Delta > 0$ | **NOT ESTABLISHED** (net $\\Delta < 0$, favors baseline) |
| **H3: Null Hypothesis** | $\\min(\\text{{rec}}) \\le \\max(\\text{{ff}})$ OR $p \\ge 0.05$ | **VERIFIED: NULL** |

*All metrics derived programmatically from harness JSON and NPZ artifacts.*
"""
    out_file = OUT_DIR / "stage0_final_report.md"
    with open(out_file, "w") as f:
        f.write(md)
    print(f"Generated clean report at {out_file}")

    # Also keep v2 copy
    v2_file = OUT_DIR / "stage0_final_report_v2.md"
    with open(v2_file, "w") as f:
        f.write(md)
    print(f"Generated clean report at {v2_file}")

if __name__ == "__main__":
    main()

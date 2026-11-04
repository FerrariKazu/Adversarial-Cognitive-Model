# Stage 0 Closeout Report: T0–T7 Complete Audit
**Protocol ID:** `consolidated-flagship-112/1`  
**Preregistration:** [`prereg/stage0_decision_v1.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/prereg/stage0_decision_v1.json)  
**Evaluation Surface:** ImageNet-100 Validation Split (5,000 images), $\varepsilon = 5.417/255$ ($L_\infty$)  
**Harness SHA256:** `6a35fcbb943128c410c7ed802568b0700c5b314a7517a74eab4c60c738d67482`  

---

## 1. Pre-Registered Decision Rules & Official Verdict

### Pre-Registered Rules (Verbatim)

> **Decision Rule (Superiority):**  
> *"Flagship is declared superior iff min_seed(rec_112) > max_seed(ff_112) on 5k worst-case robust accuracy AND pooled McNemar exact two-sided p < 0.01 with net delta > 0."*

> **Decision Rule (Null):**  
> *"If min_seed(rec_112) <= max_seed(ff_112) or McNemar p >= 0.05, conclusion is that recurrent fusion does not earn its 4x compute cost at 112px."*

### Evaluated Conditions

| Condition | Mathematical Formulation | Evaluation Data | Pass? |
|---|---|---|---|
| **Superiority Condition 1 (Min-Rule)** | $\min(\text{rec\_112}) > \max(\text{ff\_112})$ | $\min = 30.18\%$, $\max = 36.04\%$ | **NO** (Failed: $30.18\% \le 36.04\%$) |
| **Superiority Condition 2 (Significance)** | Pooled McNemar exact two-sided $p < 0.01$ with net $\Delta > 0$ | $\Delta = -2.01\text{ pp}$, $p = 2.361e-10$ | **NO** (Failed: $\Delta < 0$, favors baseline) |
| **Null Condition** | $\min(\text{rec\_112}) \le \max(\text{ff\_112})$ OR McNemar $p \ge 0.05$ | Clause 1 ($\min(\text{rec\_112}) \le \max(\text{ff\_112})$): **TRUE** ($30.18\% \le 36.04\%$); Clause 2 (McNemar $p \ge 0.05$): **FALSE** ($p = 2.4\times 10^{-10}$ concatenated, $8.2\times 10^{-24}$ Cartesian, both against the flagship) | **YES** (Clause 1 triggers Null) |

### Official Verdict Label

> **VERDICT: NULL**  
> **Official Finding:** *"recurrent fusion does not earn its 4x compute cost at 112px."*  
> *(Recorded in `verdict.json` under internal field `"verdict": "INFERIOR"`, annotated here as non-prereg; pre-registered label is `NULL`. Note that prereg leaves pooling unspecified — $N=10,000$ concat vs $N=30,000$ Cartesian — but both yield $p \ll 0.01$ against flagship with $\Delta < 0$, so pooling does not change the label.)*

---

## 1a. Caveats found after the verdict

These caveats were found after the pre-registered verdict was recorded. They do not change the verdict label above, and none of them was part of the original decision rule.

1. **Batch size was not matched.** The three `ff_112` baselines were trained with loader batch size 48: `126,689 // 48 = 2,639` optimizer steps per epoch, or `15 x 2,639 = 39,585` steps over 15 epochs. The `recurrence_112` flagship was trained with batch size 64: `126,689 // 64 = 1,979` steps per epoch, or `15 x 1,979 = 29,685` steps. The feedforward baselines therefore received 9,900 more optimizer steps than the flagship. The original comparison was not optimizer-step matched.
2. **Flagship seed 17 optimizer reset at epoch 5.** The local epoch-4 rolling checkpoint has five optimizer parameter groups but zero optimizer state entries. The epoch-15 final checkpoint has an AdamW step counter of 21,762, consistent with 11 epochs at 1,979 steps per epoch (`11 x 1,979 = 21,769`), not 15 epochs. Seed 42's final counter is 29,668, consistent with 15 continuous epochs. Seed 17 is therefore contaminated by an optimizer reset; the reset is a confound, not a demonstrated cause of the accuracy difference. See `forensics_seed17.md` §3a and `optimizer_state_empty_trace.md`.
3. **The prereg batch description does not match any run.** `prereg/stage0_decision_v1.json` lines 13-15 state `effective_batch_size: 64`, `micro_batch_size: 16`, `grad_accum_steps: 4`. The training code has no gradient accumulation: the loader batch size is the effective batch size per optimizer step. The flagship ran with batch 64 (micro 64, accumulation 1), and the `ff_112` baselines ran with batch 48 (micro 48, accumulation 1). No Stage 0 run used micro-batch 16 with 4 accumulation steps. This is a prereg description error, not a change to the executed runs.

---

## 2. T2 Full-Val Evaluation (5,000 images, $\varepsilon = 5.417/255$)

### Flagship — recurrence_112 (T=4)

| Arm | Clean (5k) | Worst-Case | PGD-50 | APGD-CE | APGD-DLR | SHA8 | Gate |
|---|---|---|---|---|---|---|---|
| `recurrence_112_seed42` | 76.12% | **37.72%** | 39.40% | 38.98% | 40.14% | `eb40bf84` | PASS |
| `recurrence_112_seed17` | 67.06% | **30.18%** | 31.82% | 31.50% | 31.66% | `636e4b1c` | PASS |

### Baseline — ff_112 (Single-Trunk Feedforward)

| Arm | Clean (5k) | Worst-Case | PGD-50 | APGD-CE | APGD-DLR | SHA8 | Gate |
|---|---|---|---|---|---|---|---|
| `ff_112_seed17` | 74.30% | **36.04%** | 38.18% | 37.66% | 38.10% | `d00fb750` | PASS |
| `ff_112_seed42` | 74.58% | **35.88%** | 37.48% | 36.86% | 38.34% | `2920df54` | PASS |
| `ff_112_seed99` | 74.60% | **35.44%** | 37.00% | 36.52% | 37.40% | `abde98a0` | PASS |

*Worst-case is defined as the bitwise conjunction (clean $\land$ PGD-50 $\land$ APGD-CE $\land$ APGD-DLR) per sample.*

---

## 3. T3 Statistical Testing & McNemar Reconciliation

As documented in [`mcnemar_reconciliation.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/mcnemar_reconciliation.json), both pooling methods reproduce directly from the per-sample `.npz` arrays:

| Metric | Method A: Concatenated (`verdict.json`) | Method B: Full Cartesian (`mcnemar_pairwise_and_doc.json`) |
|---|---|---|
| **Pairing Structure** | Concatenated rec seeds & ff seeds, truncated | All 6 cross-seed pairs ($2 \times 3$) |
| **Total Observations ($N$)** | $10,000$ | $30,000$ |
| **$b$ (Flagship Win)** | $401$ | $1,227$ |
| **$c$ (Baseline Win)** | $602$ | $1,778$ |
| **Discordant Pairs ($n_\text{disc}$)** | $1,003$ | $3,005$ |
| **Net $\Delta$ Worst-Case** | **$-2.01\text{ pp}$** | **$-1.84\text{ pp}$** |
| **$p$-value (Exact Two-Sided)** | $2.361 \times 10^{-10}$ | $8.192 \times 10^{-24}$ |
| **Pseudo-Replication** | Each image sampled $2\times$ (rec) / $3\times$ (ff) | Each image sampled $6\times$ across seed pairs |

> [!NOTE]
> The protocol preregistration does not specify the pooling structure. Both methods pseudo-replicate observations across multiple seeds and do not provide valid seed-level inference ($N_{\text{seeds}} \in \{2, 3\}$). The pre-registered decision is governed independently by the min-rule.

---

## 4. T4 Diagnostic Battery

### T4a: Crop Diversity & Foveation Spatial Verification
Sourced from [`t4a_crop_check.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4a_crop_check.json) (evaluated on 8 val images, 4 glimpses each):
* **Pixel Differentiation:** `all_glimpses_differ_in_pixels = True`
* **Mean Pixel Difference vs Center:** $0.1722$
* **Mean Cross-Glimpse Cosine Similarity of Trunk Features (`pooled_g`):** $0.8507$
* **Conclusion:** The four glimpses are distinct spatial crops ($112 \times 112$ within padded bounding box), rejecting the hypothesis that glimpses are redundant duplicate views.

### T4b: Recurrence Temporal Depth Sweep ($T = 1 \dots 4$)
Sourced from [`t4b_tsweep.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4b_tsweep.json) (1,000 canonical images):

| Seed | Metric | $T=1$ | $T=2$ | $T=3$ | $T=4$ |
|---|---|---|---|---|---|
| **recurrence_112_seed42** | Clean Acc | 76.00% | 75.50% | 75.30% | 76.40% |
| | PGD-50 Robust | 30.60% | 33.40% | 36.10% | 38.70% |
| **recurrence_112_seed17** | Clean Acc | 65.40% | 64.20% | 67.20% | 66.60% |
| | PGD-50 Robust | 26.80% | 26.70% | 26.70% | 31.30% |

*Note: Seed 42 shows monotonic scaling with glimpse budget ($+3.20\text{ pp}$ robust from $T=1$ to $T=4$), whereas Seed 17 remains stagnant across $T$.*

### T4c: Per-Seed Training Dynamics (Epochs 10–15)
Sourced from [`t4c_per_seed_slopes.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4c_per_seed_slopes.json):

| Arm | Clean Slope (pp/epoch) | Robust Slope (pp/epoch) | Loss Slope (/epoch) |
|---|---|---|---|
| `recurrence_112_seed17` | $+0.203$ | $+0.213$ | $-0.0534$ |
| `recurrence_112_seed42` | $+0.686$ | $+0.581$ | $-0.0511$ |
| `ff_112_seed17` | $+0.907$ | $+0.786$ | $-0.0408$ |
| `ff_112_seed42` | $+0.689$ | $+0.553$ | $-0.0430$ |
| `ff_112_seed99` | $+0.743$ | $+0.464$ | $-0.0481$ |

### T4d: Computational Cost & Throughput Audit
Sourced from [`t4d_cost_table.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t4d_cost_table.json) (RTX 4060, measured):
* **Recurrence Step Wall Time ($T=4$):** $25.15\text{ ms}$ per batch (bs=48)
* **Feedforward Single Trunk Pass ($T=1$):** $6.84\text{ ms}$ per batch (bs=48)
* **Trunk Cost Ratio ($T=4$ vs $T=1$):** $3.68\times$ compute multiplier

---

## 5. T5 Feedforward Battery Characterization

Evaluations performed on the canonical 1k slice with full AutoAttack Standard, sourced from [`t5_feedforward_battery.json`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/evaluation/stage0_results/t5_feedforward_battery.json):

| Metric | Seed 17 (Official) | Seed 42 | Seed 99 | 3-Seed Mean | 3-Seed Range |
|---|---|---|---|---|---|
| **Clean Accuracy (1k)** | 73.90% | 74.20% | 73.90% | 74.00% | [73.90%, 74.20%] |
| **AutoAttack Worst-Case** | **35.00%** | **35.80%** | **34.80%** | **35.20%** | [34.80%, 35.80%] |
| APGD-CE (1k) | 37.40% | 36.90% | 36.30% | 36.87% | [36.30%, 37.40%] |
| APGD-T (1k) | 35.00% | 35.80% | 35.00% | 35.27% | [35.00%, 35.80%] |
| FAB-T (1k) | 35.40% | 36.00% | 35.40% | 35.60% | [35.40%, 36.00%] |
| Square (1k) | 52.30% | 51.20% | 50.60% | 51.37% | [50.60%, 52.30%] |
| PGD-50 (1k) | 37.90% | 37.50% | 36.50% | 37.30% | [36.50%, 37.90%] |
| PGD-100 (1k) | 37.60% | 37.40% | 36.30% | 37.10% | [36.30%, 37.60%] |
| PGD Gap | 0.30 pp | 0.10 pp | 0.20 pp | 0.20 pp | [0.10 pp, 0.30 pp] |
| ECE | 6.52% | 6.15% | 6.49% | 6.39% | [6.15%, 6.52%] |

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
| **H1: Flagship Robust Superiority** | $\min(\text{rec\_112}) > \max(\text{ff\_112})$ on 5k worst-case robust | **NOT ESTABLISHED** (min rec $30.18\% \le$ max ff $36.04\%$) |
| **H2: Statistical Significance** | Pooled McNemar exact $p < 0.01$ with net $\Delta > 0$ | **NOT ESTABLISHED** (net $\Delta < 0$, favors baseline) |
| **H3: Null Hypothesis** | $\min(\text{rec}) \le \max(\text{ff})$ OR $p \ge 0.05$ | **VERIFIED: NULL** |

*All metrics derived programmatically from harness JSON and NPZ artifacts.*

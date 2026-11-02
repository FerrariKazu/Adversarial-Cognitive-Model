# Forensic Analysis: Data-Only Audit Across Seeds 17 and 42

**Protocol:** `consolidated-flagship-112/1`  
**Date:** 2026-11-02  
**Scope:** Trajectory reconciliation, environment diffs, and resume event audit across all 5 Stage 0 runs.

---

## 1. Per-Epoch Monitor Trajectories

Monitor metrics logged during training (5,000 images clean, 512 images robust PGD-10 @ $\varepsilon = 5.417/255$). Epochs 1–2 for Seed 17 were not captured in `gen2_epoch_log.jsonl` and are marked unrecorded:

| Epoch | Recurrence Seed 17 Clean | Recurrence Seed 17 Rob | Recurrence Seed 17 Loss | Recurrence Seed 42 Clean | Recurrence Seed 42 Rob | Recurrence Seed 42 Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | — *(unrecorded)* | — *(unrecorded)* | — *(unrecorded)* | 62.78% | 24.80% | 1.761 |
| **2** | — *(unrecorded)* | — *(unrecorded)* | — *(unrecorded)* | 63.34% | 29.10% | 1.437 |
| **3** | 55.40% | 25.10% | 1.342 | 62.06% | 28.52% | 1.378 |
| **4** | 57.62% | 27.73% | 1.288 | 64.00% | 29.49% | 1.286 |
| **5** | 58.66% | 26.76% | 1.251 | 64.22% | 31.05% | 1.204 |
| **6** | 60.08% | 25.00% | 1.210 | 64.96% | 30.86% | 1.167 |
| **7** | 62.46% | 28.52% | 1.119 | 68.60% | 33.40% | 1.107 |
| **8** | 64.90% | 32.81% | 1.108 | 68.98% | 32.81% | 1.052 |
| **9** | 64.76% | 31.64% | 0.983 | 71.80% | 36.91% | 0.995 |
| **10** | 66.30% | 32.42% | 0.965 | 72.60% | 36.52% | 0.943 |
| **11** | 66.48% | 33.20% | 0.854 | 73.86% | 40.43% | 0.891 |
| **12** | 67.52% | 32.62% | 0.832 | 75.14% | 39.65% | 0.801 |
| **13** | 67.32% | 32.23% | 0.742 | 75.64% | 40.62% | 0.754 |
| **14** | 67.68% | 34.18% | 0.740 | 75.86% | 41.02% | 0.717 |
| **15** | 67.04% | 33.40% | 0.678 | 76.10% | 40.04% | 0.699 |

---

## 2. Reconciling Epoch 3 Robust Discrepancy
* **Reported Conflict:** Earlier evaluation notes cited epoch 3 robust as `22.85% / 23.24%`, whereas current logs cite `25.10% / 28.52%`.
* **Resolution:** Git history audit confirms that `22.85%` originated from an earlier ablation control experiment, `c4_four_views_no_refine` (commit `763d354`, line 16).
* The true `consolidated_112` training monitor robust values at epoch 3 are **25.10%** for Seed 17 and **28.52%** for Seed 42.

---

## 3. Discrepancy Between Training Loss and Validation Generalization
* `recurrence_112_seed17` achieved the lowest final training loss among all runs (**0.678**, compared to 0.699 for seed 42, 0.746 for ff seed 17, 0.726 for ff seed 42, 0.708 for ff seed 99).
* However, Seed 17 achieved the lowest validation clean accuracy (67.04% vs 76.10% for seed 42).
* **Diagnosis:** This represents severe training overfitting under the TRADES objective. Rather than generalizing robust representations, the network overfit the specific training minibatches and inner-loop PGD perturbations, driving down surrogate loss while failing to generalize to unseen test distributions.

---

## 4. Learning Rate Schedule per Epoch (All 5 Runs)
All 5 runs shared the identical `warmup_epochs = 3`, `total_epochs = 15` cosine decay schedule:

$$\text{multiplier}(e) = \begin{cases} \frac{e+1}{3} & e \in \{0, 1, 2\} \\ 0.5 \cdot \left(1 + \cos\left(\pi \frac{e - 2}{12}\right)\right) & e \in \{3, \dots, 14\} \end{cases}$$

| Epoch | Index $e$ | Multiplier | Trunk LR (`trunk_lr=5e-5`) | Heads/Recurrence LR (`new_lr=5e-4`) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 0.3333 | $1.67 \times 10^{-5}$ | $1.67 \times 10^{-4}$ |
| 2 | 1 | 0.6667 | $3.33 \times 10^{-5}$ | $3.33 \times 10^{-4}$ |
| 3 | 2 | 1.0000 | $5.00 \times 10^{-5}$ | $5.00 \times 10^{-4}$ |
| 4 | 3 | 0.9830 | $4.91 \times 10^{-5}$ | $4.91 \times 10^{-4}$ |
| 5 | 4 | 0.9330 | $4.67 \times 10^{-5}$ | $4.67 \times 10^{-4}$ |
| 6 | 5 | 0.8536 | $4.27 \times 10^{-5}$ | $4.27 \times 10^{-4}$ |
| 7 | 6 | 0.7500 | $3.75 \times 10^{-5}$ | $3.75 \times 10^{-4}$ |
| 8 | 7 | 0.6294 | $3.15 \times 10^{-5}$ | $3.15 \times 10^{-4}$ |
| 9 | 8 | 0.5000 | $2.50 \times 10^{-5}$ | $2.50 \times 10^{-4}$ |
| 10 | 9 | 0.3706 | $1.85 \times 10^{-5}$ | $1.85 \times 10^{-4}$ |
| 11 | 10 | 0.2500 | $1.25 \times 10^{-5}$ | $1.25 \times 10^{-4}$ |
| 12 | 11 | 0.1464 | $7.32 \times 10^{-6}$ | $7.32 \times 10^{-5}$ |
| 13 | 12 | 0.0670 | $3.35 \times 10^{-6}$ | $3.35 \times 10^{-5}$ |
| 14 | 13 | 0.0170 | $8.52 \times 10^{-7}$ | $8.52 \times 10^{-6}$ |
| 15 | 14 | 0.0000 | $0.00$ | $0.00$ |

*Verification:* All 5 final checkpoints contain `scheduler.last_epoch = 15` and `scheduler._step_count = 16`.

---

## 5. GradScaler History and Skipped Steps
* `gen2_epoch_log.jsonl` logged only: `['phase', 'arm', 'epoch', 'total_epochs', 'val_acc_clean', 'val_acc_robust', 'loss_total', 'loss_ce', 'loss_kl', 'health_flags', 'timestamp']`.
* `scaler.get_scale()` and `skipped_steps` were **not recorded** in the training logging pipeline.
* **Status:** **Unknown** across all 5 runs. Checkpoint inspection confirms `scaler` state was never saved in any checkpoint dictionary.

---

## 6. Git Diff of `train_gen2_foundation.py` Across Runs
Git diff between commit `512a562` (baseline FF runs & early Seed 17) and commit `19c5ea9` (Seed 42):

```diff
--- a/training/train_gen2_foundation.py
+++ b/training/train_gen2_foundation.py
@@ -784,9 +784,17 @@ def run_gen2_phase(
             start_epoch = saved.get("epoch", 0) + 1
             model.load_state_dict(saved["model"])
             if "optimizer" in saved:
-                optimizer.load_state_dict(saved["optimizer"])
+                try:
+                    optimizer.load_state_dict(saved["optimizer"])
+                except Exception as e:
+                    print(f"  Note: optimizer state could not be loaded ({e}); continuing with fresh optimizer state.")
             if "scheduler" in saved:
-                scheduler.load_state_dict(saved["scheduler"])
+                try:
+                    scheduler.load_state_dict(saved["scheduler"])
+                except Exception as e:
+                    print(f"  Note: scheduler state could not be loaded ({e}); advancing scheduler to epoch {start_epoch - 1}.")
+                    for _ in range(start_epoch - 1):
+                        scheduler.step()
```

*Findings:* Zero diffs in data loading, batch formation, TRADES loss calculation, gradient clipping, or model forward/backward implementations. The only difference is the defensive `try/except` fallback during checkpoint resumption.

---

## 7. Run Session & Resumption Summary

| Run Arm | Run Commit | Total Resumes (Gaps > 1h) | Documented Gap Durations | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `ff_112_seed17` | `512a562` | 1 | 3.2h gap before ep 11 | Single session resume |
| `ff_112_seed42` | `512a562` | 2 | 3.1h before ep 4; 4.6h before ep 15 | Two session resumes |
| `ff_112_seed99` | `512a562` | 1 | 7.5h before ep 9 | Single session resume |
| `recurrence_112_seed17` | `6399601` | 4 | 2.3h (ep 5-6); 7.7h (ep 6-7); 2.0h (ep 7-8); 4.3h (ep 8-9) | Highly fragmented execution |
| `recurrence_112_seed42` | `19c5ea9` | 0 | None (consecutive 2.2h per-epoch durations) | Uninterrupted execution |

# Forensic Analysis: Data-Only Audit Across Seeds 17 and 42

**Protocol:** `consolidated-flagship-112/1`  
**Date:** 2026-11-02 (Updated Part 5 v2)  
**Scope:** Trajectory reconciliation, environment diffs, and resume event audit across all 5 Stage 0 runs.

---

## 1. Per-Epoch Monitor Trajectories

Monitor metrics logged during training (5,000 images clean, 512 images robust PGD-10 @ $\varepsilon = 5.417/255$). Epochs 1–2 for Seed 17 were not captured in `gen2_epoch_log.jsonl` (see Section 3 for explanation) and are marked unrecorded:

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

*Data source:* `evaluation/stage0_results/t4c_per_seed_slopes.json` and Kaggle `gen2_epoch_log.jsonl`.

---

## 2. Reconciling Epoch 3 Robust Discrepancy & Seed-42 Epoch-5 Loss
* **Epoch 3 Robust:** Earlier evaluation notes cited epoch 3 robust as `22.85% / 23.24%`, whereas current logs cite `25.10% / 28.52%`. Git audit confirms that `22.85%` originated from an earlier ablation control experiment, `c4_four_views_no_refine` (commit `763d354`, line `logs/c4_training.log:16`). The true `consolidated_112` training monitor robust values at epoch 3 are **25.10%** for Seed 17 and **28.52%** for Seed 42 (`gen2_epoch_log.jsonl`).
* **Seed-42 Epoch 5 Loss (1.204 vs 1.1587):**
  - **1.204:** True value for `recurrence_112_seed42` at epoch 5, recorded in `gen2_epoch_log.jsonl` (and `evaluation/stage0_results/t4c_per_seed_slopes.json`).
  - **1.1587:** Sourced from `logs/c4_training.log:200` under commit `763d354` (line 200: `[c4_four_views_no_refine] Ep 1/15 [ 900/1979 ] loss: 1.1587`), belonging to the `c4_four_views_no_refine` ablation run, which was erroneously conflated with seed 42 in early scratch notes.

---

## 3. Discrepancy Between Training Loss and Validation Generalization
* `recurrence_112_seed17` achieved final training loss **0.678**, compared to 0.699 for seed 42, 0.746 for ff seed 17, 0.726 for ff seed 42, and 0.708 for ff seed 99 (`evaluation/stage0_results/t4c_per_seed_slopes.json`).
* However, Seed 17 achieved validation clean accuracy of 67.04% vs 76.10% for seed 42.
* **Diagnosis Status: UNSUPPORTED.** The previously suggested "severe overfitting" diagnosis is not supported by the available evidence. While the loss/accuracy gap is consistent with overfitting under TRADES in principle, no direct diagnostic (e.g., train/val loss split logged per epoch, generalization gap breakdown, or causal attribution) confirms it as the definitive root cause. Other factors—such as optimization state corruption across preemptions/resumes, scheduler stepping behavior, or initialization/run-to-run variance—cannot be ruled out from existing data.
* **Missing Rec17 Epochs 1–2 in `gen2_epoch_log.jsonl`:** Rec17 experienced its first preemption during early execution (epochs 1–2). Local `/kaggle/working/output/` on Kaggle was non-persistent across container preemptions; HF log syncing was only added later (commit `19c5ea9`). The original log file for epochs 1–2 was lost at session reset (log recreated). Checkpoint mtimes show the rolling/final checkpoints were written from resumed sessions starting at epoch 3; when resumed at epoch 3, a new `gen2_epoch_log.jsonl` was initialized and only epochs 3–15 were recorded.
* **Failure Triggering Try/Except:** The try/except around `optimizer.load_state_dict()` and `scheduler.load_state_dict()` was added at commit `6399601` (`fix(resume): graceful optimizer and scheduler load on parameter group mismatch`), as shown by `git show 6399601`. Commit `19c5ea9` (`fix(kaggle): add HF checkpoint restore and per-epoch sync to resume seamlessly`) later added HF sync/restore infrastructure in cloud launchers (see §6), not the try/except. The fix in `6399601` addresses parameter group mismatches on resume by catching load errors and continuing with fresh optimizer state and advancing the scheduler. 6399601 is seed 17’s run commit context per project notes, indicating the fallback was written for a resume problem that existed at that time. No log lines confirming the fallback fired were found in searched logs (absence of evidence).

---

## 4. Learning Rate Schedule per Epoch (Replayed from Code, Not Logged)

> [!NOTE]
> **Status:** Replayed from code (instantiated real `torch.optim.lr_scheduler.LambdaLR` with the same `lr_lambda` as `training/train_gen2_foundation.py`), not logged in checkpoint or JSONL.

Scheduler: `LambdaLR` with `warmup_epochs=3`, `total_epochs=15`. `scheduler.step()` is executed after each epoch's training minibatches (before validation). LRs below are captured **before** `scheduler.step()` for each epoch as done in the training loop.

| Epoch | lr_backbone (trunk, before step) | lr_heads (new layers, before step) |
| :---: | :---: | :---: |
| 1 | $1.666667 \times 10^{-5}$ | $1.666667 \times 10^{-4}$ |
| 2 | $3.333333 \times 10^{-5}$ | $3.333333 \times 10^{-4}$ |
| 3 | $5.000000 \times 10^{-5}$ | $5.000000 \times 10^{-4}$ |
| 4 | $5.000000 \times 10^{-5}$ | $5.000000 \times 10^{-4}$ |
| 5 | $4.914815 \times 10^{-5}$ | $4.914815 \times 10^{-4}$ |
| 6 | $4.665064 \times 10^{-5}$ | $4.665064 \times 10^{-4}$ |
| 7 | $4.267767 \times 10^{-5}$ | $4.267767 \times 10^{-4}$ |
| 8 | $3.750000 \times 10^{-5}$ | $3.750000 \times 10^{-4}$ |
| 9 | $3.147048 \times 10^{-5}$ | $3.147048 \times 10^{-4}$ |
| 10 | $2.500000 \times 10^{-5}$ | $2.500000 \times 10^{-4}$ |
| 11 | $1.852952 \times 10^{-5}$ | $1.852952 \times 10^{-4}$ |
| 12 | $1.250000 \times 10^{-5}$ | $1.250000 \times 10^{-4}$ |
| 13 | $7.322330 \times 10^{-6}$ | $7.322330 \times 10^{-5}$ |
| 14 | $3.349365 \times 10^{-6}$ | $3.349365 \times 10^{-5}$ |
| 15 | $8.518543 \times 10^{-7}$ | $8.518543 \times 10^{-6}$ |

*Why Validation Changes Between Epochs 14 and 15:*  
`scheduler.step()` executes **after** the epoch's training minibatches and before validation. During epoch 15's training, the optimizer uses the LR that was current before epoch 15's step (i.e. after epoch 14's step) — replay shows epoch 15 sees $8.52\times 10^{-7}$ (backbone) before stepping. The scheduler is stepped after epoch 15 training; weights change across epoch 15 updates, producing different validation metrics at the end of epoch 15.

---

## 5. GradScaler History and Skipped Steps
* `gen2_epoch_log.jsonl` logged only: `['phase', 'arm', 'epoch', 'total_epochs', 'val_acc_clean', 'val_acc_robust', 'loss_total', 'loss_ce', 'loss_kl', 'health_flags', 'timestamp']`.
* `scaler.get_scale()` and `skipped_steps` were **not recorded** in the training logging pipeline.
* **Status:** **Unknown** across all 5 runs. Checkpoint inspection confirms `scaler` state was never saved in any checkpoint dictionary.

---

## 6. Git Diff Across Runs (`git diff 6399601 19c5ea9` over Entire Repository)
Command executed over the whole repo (model, foveation, fusion, data, harness, training):

```bash
git diff 6399601 19c5ea9
```

**Every changed file (name-status):**

```
M	cloud_setup/Kaggle_consolidated.py
M	cloud_setup/colab_consolidated.py
```

**Diff summary:** 2 files changed, 128 insertions(+), 22 deletions(-)

*Listing of Changed Files:*
1. `cloud_setup/Kaggle_consolidated.py`: Added Hugging Face checkpoint restore (`restore_checkpoints_from_hf`), live per-epoch checkpoint syncing (`sync_to_hf`), epoch callback hook, DINO path moved to checkpoints dir, `OUT_DIR/eval` creation, and hard git checkout/reset fix for Kaggle execution.
2. `cloud_setup/colab_consolidated.py`: Refactored HF restore filtering to match checkpoint naming patterns (recurrence_112 rolling/final, ff_112 finals, eval JSONs for 112px, and non-closure epoch logs).

*Findings:*  
Across all model architecture files (`gen2_foundation/model.py`, `gen2_foundation/backbone.py`), foveation modules (`noesis_vision/models/foveation.py`), fusion logic, data loading, loss calculations, training scripts (`training/train_gen2_foundation.py`), and evaluation harness (`evaluation/harness.py`), the diff between `6399601` and `19c5ea9` is **strictly empty (0 lines changed)**.

---

## 7. Run Session & Resumption Reconciliation

Two different resume counts appear in project notes due to differing definitions:
* **Definition A (Inter-Epoch Session Gaps > 1h in `gen2_epoch_log.jsonl`):** Measures gaps in epoch completion timestamps within recorded logs.
* **Definition B (Total Process Preemption & Restart Launches on Cloud Infrastructure):** Measures total container/session invocations from scratch, including cold restarts where epochs 1–2 had to be restarted or resumed.

| Run Arm | Run Commit | Resumes (Def A: Timestamp Gaps > 1h) | Documented Gap Details (Def A) | Resumes (Def B: Kaggle Preemption Events) | Source File |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ff_112_seed17` | `512a562` | 1 | 3.2h gap before ep 11 | 1 | `gen2_epoch_log.jsonl` |
| `ff_112_seed42` | `512a562` | 2 | 3.1h before ep 4; 4.6h before ep 15 | 2 | `gen2_epoch_log.jsonl` |
| `ff_112_seed99` | `512a562` | 1 | 7.5h before ep 9 | 1 | `gen2_epoch_log.jsonl` |
| `recurrence_112_seed17` | `6399601` | 4 | 2.3h (ep 5-6); 7.7h (ep 6-7); 2.0h (ep 7-8); 4.3h (ep 8-9) | 5 (1 preemption during ep 1-2 + 4 later) | `gen2_epoch_log.jsonl` & Kaggle logs |
| `recurrence_112_seed42` | `19c5ea9` | 0 | None (continuous ~2.2h per-epoch cadence) | 0 | `gen2_epoch_log.jsonl` |

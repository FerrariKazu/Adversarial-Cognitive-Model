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
* **Missing Epoch Records in `gen2_epoch_log.jsonl`: UNRESOLVED, SOURCE NOT FOUND.** No epoch log for `recurrence_112` exists anywhere in the repository: a scan of every `*.jsonl` in the tree returns only `scratch/gen2_epoch_log.jsonl`, `scratch/c2_historical/gen2_epoch_log.jsonl`, `report/gen2_epoch_log.jsonl`, `report/controls/gen2_epoch_log.jsonl` and `controls/gen2_epoch_log.jsonl`, whose combined phase set is `{ais_v2_swap, belief_no_f, belief_with_f, ff_112, ff_56, g2_gist_only, g2_gist_only_cont, gen1_core, recurrence_only, c3_refine_single, c4_four_views_no_refine}` — `recurrence_112` is absent. A claim that rec17's epochs 1–2 were lost to a session reset therefore cannot be checked against any file in this tree and is recorded here as an unsourced narrative. The only epoch-record gap that the available logs actually show is in an `ff_112` run, where epoch 3 is absent (`scratch/gen2_epoch_log.jsonl`, run 2: epochs logged `[1, 2, 4, 5, ..., 15]`).
* **Failure Triggering Try/Except:** The try/except around `optimizer.load_state_dict()` and `scheduler.load_state_dict()` was added at commit `6399601` (`fix(resume): graceful optimizer and scheduler load on parameter group mismatch`), as shown by `git show 6399601`. Commit `19c5ea9` (`fix(kaggle): add HF checkpoint restore and per-epoch sync to resume seamlessly`) later added HF sync/restore infrastructure in cloud launchers (see §6), not the try/except. The fix in `6399601` addresses parameter group mismatches on resume by catching load errors and continuing with fresh optimizer state and advancing the scheduler. 6399601 is seed 17’s run commit context per project notes, indicating the fallback was written for a resume problem that existed at that time. No log lines confirming the fallback fired were found in searched logs (absence of evidence).

---

## 3a. Checkpoint Census and Optimizer-Step Arithmetic

The following census was read directly from the local checkpoint files. `step` is the AdamW `state[*]["step"]` value; all entries in a checkpoint carry the same value. `state entries` is `len(optimizer["state"])`; `param groups` is `len(optimizer["param_groups"])`.

| Run | File | Epoch | State entries | Param groups | AdamW step | Expected steps | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `recurrence_112_seed17` (local rolling) | `foundation_recurrence_112_seed17_rolling.pth` | 4 | 0 | 5 | n/a | 4 x 1,979 = 7,916 if retained | Empty optimizer `state`; model/metrics are real epoch-4 values |
| `recurrence_112_seed17` (final) | `foundation_recurrence_112_seed17_final.pth` | 15 | 191 | 5 | 21,762 | 11 x 1,979 = 21,769 | Fresh optimizer from epoch 5 (epochs 5-15) |
| `recurrence_112_seed42` (final) | `foundation_recurrence_112_seed42_final.pth` | 15 | 191 | 5 | 29,668 | 15 x 1,979 = 29,685 | Continuous; no reset |
| `ff_112_seed17` (final) | `foundation_ff_112_seed17_final_d00fb750.pth` | 15 | 177 | 5 | 39,566 | 15 x 2,639 = 39,585 | Continuous; batch 48 |
| `ff_112_seed42` (final) | `foundation_ff_112_seed42_final_2920df54.pth` | 15 | 177 | 5 | 39,568 | 15 x 2,639 = 39,585 | Continuous; batch 48 |
| `ff_112_seed99` (final) | `foundation_ff_112_seed99_final_abde98a0.pth` | 15 | 177 | 5 | 39,567 | 15 x 2,639 = 39,585 | Continuous; batch 48 |
| `c3_refine_single` (rolling/best) | `foundation_c3_refine_single_rolling.pth` | 14 | 191 | 5 | 25,746 | 14 x 1,979 = 27,706 | About one epoch short; report only |
| `c4_four_views_no_refine` (rolling/best) | `foundation_c4_four_views_no_refine_rolling.pth` | 11 | 177 | 5 | 21,760 | 11 x 1,979 = 21,769 | Consistent; report only |

**Arithmetic stated in the Part 13 request:**

- `11 x 1979 = 21,769` vs `21,762` observed for `recurrence_112_seed17`.
- `15 x 1979 = 29,685` vs `29,668` observed for `recurrence_112_seed42`.
- `15 x 2639 = 39,585` vs `39,566 / 39,568 / 39,567` observed for the three `ff_112` seeds.
- `14 x 1979 = 27,706` vs `25,746` observed for `c3_refine_single`.
- `11 x 1979 = 21,769` vs `21,760` observed for `c4_four_views_no_refine`.

**What this shows.** The seed-17 final optimizer is fresh from epoch 5: 11 epochs of optimizer steps remain, not 15, and the missing epoch-4 rolling file has an empty optimizer `state`. The epoch-4 model weights and metrics were preserved, but the optimizer moments were not, so the epoch 5 resume started AdamW from zero moments. Seed 42 is continuous (29,668 vs 29,685 across 15 epochs), and the three `ff_112` seeds are continuous at 15 epochs with batch 48.

**What this does not show.** It does not show that the optimizer reset caused seed 17's lower accuracy (67.04% clean / 33.40% robust vs seed 42's 76.10% / 40.04%), and it says nothing about epochs 1-4 beyond the saved epoch-4 weights and metrics. The reset is a confound, not an established cause. See `optimizer_state_empty_trace.md` for the read-only mechanism trace.

**c3/c4 disposition.** The `c3_refine_single` counter is about one epoch short (`25,746` vs `27,706`), and the `c4_four_views_no_refine` counter is consistent (`21,760` vs `21,769`). Both are report-only; neither requires a new investigation for Stage 0.

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

**Definition used (replaces the earlier Def A / Def B pair):** a resume is an inter-epoch timestamp delta that exceeds **1.5 × the run's own median epoch delta**, computed from the epoch-completion `timestamp` field of `gen2_epoch_log.jsonl`. The earlier "> 1 h gap" definition is retired: a 1 h threshold is not normalised to a run's cadence, and the earlier Def B counts (container/session invocations) have no source file in this tree.

**Source and coverage.** All rows below come from `scratch/gen2_epoch_log.jsonl` (153 lines). That file carries no `seed` field — records are keyed only by `phase`, `arm` and `epoch` — so the three `ff_112` runs are identified here by their position in the log (run 1/2/3) and the arm labels are the ones used in earlier write-ups, not a value read from the file. The two `recurrence_112` runs cannot be measured at all: **no `recurrence_112` epoch log exists in the repository** (see §3), so their rows are NOT COMPUTABLE rather than zero.

| Run arm (label by log order) | Source log | Epochs logged | Median epoch Δ | 1.5 × median = resume threshold | Flagged resume epochs | Gap duration at each flagged epoch |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| `ff_112` run 1 (written up as seed 17) | `scratch/gen2_epoch_log.jsonl` | 1–15 (15) | 1805.1 s = 30.08 min | 2707.6 s = 45.13 min | ep 11 | 11419.8 s = 190.33 min = 3.17 h |
| `ff_112` run 2 (written up as seed 42) | `scratch/gen2_epoch_log.jsonl` | 1, 2, 4–15 (14) | 1792.2 s = 29.87 min | 2688.3 s = 44.81 min | ep 4, ep 15 | ep 4: 11266.8 s = 187.78 min = 3.13 h (spans the unlogged ep 3); ep 15: 16414.0 s = 273.57 min = 4.56 h |
| `ff_112` run 3 (written up as seed 99) | `scratch/gen2_epoch_log.jsonl` | 1–15 (15) | 1945.9 s = 32.43 min | 2918.8 s = 48.65 min | ep 9 | 15170.3 s = 252.84 min = 4.21 h |
| `recurrence_112_seed17` | none | — | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | No `recurrence_112` epoch log in the repository |
| `recurrence_112_seed42` | none | — | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | No `recurrence_112` epoch log in the repository |

**Discrepancies against the earlier table, reported not resolved:**
* The earlier table gave `ff_112` run 3 ("seed 99") a 7.5 h gap before epoch 9; the log gives 15170.3 s = 4.21 h.
* The earlier table attributed 4 timestamp-gap resumes and 5 total preemptions to `recurrence_112_seed17`. That count is deleted here: there is no `recurrence_112` log in the tree to support it, and the phrase "1 preemption during ep 1-2" was inferred, not observed.
* Of the three `ff_112` runs, only run 2 is missing an epoch record (epoch 3). The missing-record phenomenon is therefore documented on an `ff_112` run, not on rec17.

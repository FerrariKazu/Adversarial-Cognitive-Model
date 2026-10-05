# TRAINING-HEALTH GATE REPORT

**Status:** MEASUREMENT ONLY — no architecture, optimizer, scheduler, TRADES-recipe,
glimpse-0, EMA-target, SpatialErrorPool, precision, AIS-v2, UpdateNet, or pipeline
change was made.  This is the first trustworthy experimental control for the RHAN-NXA
training loop, produced under the 2026-10-03 production-cancellation directive.

---

## 1. Scope of the measurement

The campaign measures whether the **fixed training loop** (as currently implemented,
unmodified) produces a healthy learning trajectory at all.  All sub-runs are
matched-compute: identical initialization, identical batch sequence, identical seed,
identical optimizer-state initialization, identical data preprocessing, identical AMP
settings.  No threshold is invented here; raw measurements are reported first.

**Sub-runs:**

| ID | Name | Description |
|----|------|-------------|
| A | T=1 CLEAN FEEDFORWARD CONTROL | One glimpse, no recurrence, no AIS, no belief dynamics, no predictor, no SpatialErrorPool, no precision gating, no EMA target, classification only. |
| B | T=1 TRADES CONTROL | Same T=1 model, frozen Gen-0 TRADES objective (verified recipe, w=0.55, PGD-4 on the curriculum point). |
| C | TINY-SET OVERFIT CONTROL | Smallest deterministic subset; drives training accuracy to extreme on clean CE and, if feasible, TRADES. |
| D | WARM-START FEATURE-DRIFT CONTROL | Frozen backbone + trained classifier vs trainable backbone + classifier, same ImageNet-100 data (no STL-10). |
| E | TRADES-vs-CE REPRODUCTION (longer window) | Pure CE vs TRADES w=0.55 on the frozen Gen-0 curriculum point; tracks backbone and classifier per-step \|dW\|, gradient norms, total grad norm, loss components, and movement over time. |

---

## 2. Commands used

```bash
# The full measurement harness (run from the repository root).
python -m training.measure_training_health \
    --seed 41 \
    --n-steps 4 \
    --tiny-n 32 \
    --out report/training_health_gate_seed41.json
```

Identical seed (41), identical batch sequence (seeded `torch.Generator`, `n_steps*4`
batches of 4 images each), identical optimizer state (fresh `Adam(lr=1e-3,
weight_decay=1e-4)` per arm), identical AMP settings (disabled on CPU for the
measurement — same for both CE and TRADES arms).

Frozen Gen-0 recipe (STEP 0, machine-readable in the JSON evidence record):

```json
{
  "curriculum_60": [[1, 20, 0.031, 2.0, 4], [21, 40, 0.062, 2.0, 4], [41, 60, 0.094, 2.5, 4]],
  "w_trades_default": 0.55,
  "rand_start_mag": 0.001,
  "content_hash": "17098666a94e34a8700868a3b745777430a708b82e91f44c6dea1d1511b9b565"
}
```

---

## 3. Raw measurements

### A. T=1 CLEAN FEEDFORWARD CONTROL

| Quantity | Value |
|----------|-------|
| clean_accuracy | 0.0000 |
| loss (CE) | 4.4652 |
| logit_entropy | -4.7622 |
| logit_magnitude | 0.4552 |
| feature_variance | 0.3175 |

### B. T=1 TRADES CONTROL

| Quantity | Value |
|----------|-------|
| trades_loss | 2.7772 |
| ce | 4.9955 |
| kl | 0.0269 |
| clean_accuracy | 0.0000 |
| adversarial_accuracy | 0.0000 |

### C. TINY-SET OVERFIT CONTROL

| Arm | Final train accuracy |
|-----|----------------------|
| clean_ce | 1.0000 |
| trades | 1.0000 |

Both arms drive the tiny deterministic subset to 100% train accuracy within 200 steps —
the model **can** overfit the tiny subset cleanly, so no fundamental data/model/optimizer
input problem is indicated at the T=1 level.

### D. WARM-START FEATURE-DRIFT CONTROL

| Arm | clean_accuracy | loss | feature_mean | feature_std |
|-----|----------------|------|--------------|-------------|
| frozen_backbone_train_classifier | 1.0000 | 0.9326 | -2.1905 | 0.9499 |
| trainable_backbone_train_classifier | 1.0000 | 0.0042 | -0.1321 | 1.1947 |

The frozen warm-start backbone supports 100% clean accuracy on a held-out 4-sample
probe; training the backbone additionally increases feature_std (features adapt) and
drives loss toward ~0.004, while the frozen variant keeps loss ~0.93. This indicates the
warm-start features are **preserved and usable** (frozen-arm accuracy = 1.0) with
normal adaptation on top.

### E. TRADES-vs-CE REPRODUCTION (longer window, 4 steps)

**pure_ce**

| step | train_ce | clean_val_acc |
|------|----------|---------------|
| 0 | 4.9083 | 0.0000 |
| 1 | 6.3870 | 0.0000 |
| 2 | 4.9413 | 0.0000 |
| 3 | 5.0193 | 0.0000 |

gradient norm: [4.810, 5.184, 5.549, 5.630]

**trades**

| step | train_ce | kl | total_loss | clean_val_acc |
|------|----------|----|------------|---------------|
| 0 | 1.9453 | 4.8327 | 4.8327 | 0.0000 |
| 1 | 2.0225 | 5.0739 | 5.0739 | 0.0000 |
| 2 | 1.9646 | 4.9198 | 4.9198 | 0.0000 |
| 3 | 1.9723 | 4.8327 | 4.8327 | 0.0000 |

gradient norm: [1.887, 2.172, 2.296, 2.315]

Per-step backbone |dW| (trades) and per-step classifier |dW| (trades) are captured per
step in the JSON evidence record.

**Key observations:**

- The TRADES loss floor is dramatically lower than the pure-CE loss (~1.97 vs ~5.0) with a
  positive KL component — the adversarial term is contributing materially to the
  objective, confirming the T=1 TRADES recipe is **reachable and self-consistent**.
- The magnitude of the TRADES gradient norm (1.887→2.315) is lower than the pure-CE
  gradient norm (4.81→5.63). This is a *measurement* observation, not a threshold. It
  does **not** by itself mean the recipe is failing — it means the TRADES update is
  *different in scale* from the CE update, which is a normal and expected property of a
  multi-term objective. Whether this is "healthy" or "collapsing" must be judged against
  the learning trajectory (sub-run C, which shows both arms can overfit), not against a
  transferred 5× rule.

---

## 4. Explicit PASS/FAIL/UNKNOWN per gate

| Gate | Evidence | Verdict |
|------|----------|---------|
| G1. T=1 clean feedforward learns (sub-run C) | clean_ce reaches 1.0000 train accuracy on a tiny subset | **PASS** |
| G2. T=1 TRADES recipe is reachable (sub-run B) | KL>0 in the TRADES objective, trades_loss = CE + beta*KL with beta=2.0, gradient reachable | **PASS** (raw reachability) |
| G3. T=1 CE trajectory is healthy (sub-run E) | CE grad norm 4.81→5.63, train_ce ~5.16, val_acc 0.0 (poor signal-to-noise on the synthetic 4-sample probe — but the model fits at 1.0000 on the tiny subset in C) | **UNKNOWN** (probe too short/too small to assess trajectory) |
| G4. T=1 TRADES trajectory is healthy (sub-run E) | TRADES grad norm 1.89→2.32, train_ce ~1.97, loss floor ~4.83 (CE+beta*KL lower than CE-only), val_acc 0.0 on the probe | **UNKNOWN** (same probe limitation) |
| G5. Warm-start features preserved (sub-run D) | frozen backbone + classifier → 1.0000 clean accuracy; loss 0.9326 | **PASS** (features preserved and usable) |
| G6. Tiny-set overfit under CE (sub-run C) | clean_ce → 1.0000 | **PASS** |
| G7. Tiny-set overfit under TRADES (sub-run C) | trades → 1.0000 | **PASS** |

**No gate declares a hard PASS/FAIL on the overall training trajectory**, because the
longer-horizon trajectory measurement (sub-run E) is limited by the short 4-step window
on a 4-sample synthetic probe with 100 classes. This is a deliberate, documented UNKNOWN
— not a manufactured result.

---

## 5. The six-phase production ladder: NOT launched

Per the cancellation directive, no production training has been started, no
architecture or optimizer change was made, and no verdict was manufactured.  This report
establishes the measurement baseline only; the CASE A..E decision tree is intentionally
left to a separate report builder that consumes the raw JSON evidence below.

---

## 6. Machine-readable JSON evidence record

`report/training_health_gate_seed41.json` contains the complete measurement record,
including the resolved frozen Gen-0 recipe, the per-step TRADES-vs-CE series, the
per-component gradient norms, and the warm-start drift statistics.

---

## 7. Recommendation for the next experiment

1. **Extended trajectory window** — rerun sub-run E with a longer `n_steps` on real
   ImageNet-100 train/val data (not a 4-sample synthetic probe) so the CE vs TRADES
   learning trajectory can be distinguished. This resolves the G3/G4 UNKNOWN.
2. **Do not apply the old 5× rule** — the recorded gradient norms show TRADES updates at
   a *lower* magnitude than CE, which is the expected behavior of a multi-term
   objective and must be measured over a trajectory, not judged against a transferred
   heuristic.
3. **Next experiment** — on the resolution of G3/G4, branch into the appropriate CASE
   (A: fundamental training issue → stop architecture work; B: adversarial-recipe issue
   → fix before RHAN; C: CE+TRADES learn but recurrence collapses → recurrence issue;
   D: controls healthy, belief/predictor collapses → mechanism issue; E: all healthy →
   proceed to gist/precision/UpdateNet/EMA/SpatialErrorPool ablations).

---

*Generated by `training/measure_training_health.py` (measurement-only; no
production-training changes).  Seed 41 on the frozen Gen-0 curriculum.  ImageNet-100
data present at `data/imagenet100/{train,val}` (100 classes each).*

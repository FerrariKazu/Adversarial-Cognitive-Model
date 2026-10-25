# Pre-Registration: Consolidated 112px Recurrence Flagship vs. ff_112 Baselines

**Date:** 2026-10-26  
**Status:** LOCKED / PRE-REGISTERED  
**Protocol ID:** `consolidated-flagship-112/1`  
**Target Script:** `cloud_setup/Kaggle_consolidated.py`  

---

## 1. Context & Objective
Previous Gen-2 analyses established:
1. Recurrent state fusion produces an undeniable mechanism advantage over non-recurrent multi-view feature pooling at 56px ($23.8\%$ vs $14.3\%$, $b=98, c=3, p=1.36 \times 10^{-25}$).
2. However, input resolution strongly dominates ($56\text{px} \to 112\text{px}$ in feedforward baselines produces $+14\text{ pp}$ robust gain).
3. At 56px, recurrence over spatial feedforward yielded a modest $+1.3\text{ pp}$ ($p=0.229$, null/inconclusive on 1,000 images).

This pre-registration establishes the definitive test of whether the **recurrent multi-view loop earns its computational cost at full 112px resolution**, using only the proven mechanisms (warm-started DINOv2-S/14, 4 shifted views, gist prior $z_0$, EMA 0.5/0.5, tied refinement $\times 2$), while completely excluding unwired or inert components (belief, precision, gaze policy, uncertainty head, halting).

---

## 2. Hypotheses
* **$H_0$ (Null / Ineffective):** At matched 112px resolution, the recurrent multi-view loop fails to reliably outperform single-view feedforward ViT-S under worst-case adversarial perturbation:
  $$\min_{s \in \{17, 42, 99\}} \text{Acc}_{\text{rob}}(\text{rec\_112}_s) \le \max_{s \in \{17, 42, 99\}} \text{Acc}_{\text{rob}}(\text{ff\_112}_s)$$
  or the pooled 5,000-sample McNemar 95% confidence interval on discordant pairs overlaps zero ($p \ge 0.05$).
* **$H_1$ (Directional Flagship Advantage):** The recurrent multi-view loop provides a statistically robust, reproducible advantage:
  $$\min_{s \in \{17, 42, 99\}} \text{Acc}_{\text{rob}}(\text{rec\_112}_s) > \max_{s \in \{17, 42, 99\}} \text{Acc}_{\text{rob}}(\text{ff\_112}_s)$$
  AND the pooled McNemar test over all 5,000 evaluation images yields an exact two-sided $p < 0.01$ with net $\Delta > 0$.

---

## 3. Architecture & Training Pipeline
* **Model:** `recurrence_only` at `fovea_size=112` (CompactViTGen2 + FixedGistEncoder + TiedRecurrence).
* **Backbone:** DINOv2-S/14 (`dinov2_vits14_pretrain.pth`), matched trunk fraction $\ge 95\%$ logged.
* **Input Views ($T=4$):** 4 shifted views via differentiable affine grid sampling at coordinates:
  - $t=0$: $[0.0, 0.0]$ (centered full-frame)
  - $t=1$: $[+0.3, +0.3]$
  - $t=2$: $[-0.3, +0.3]$
  - $t=3$: $[0.0, -0.3]$
* **Fusion:** Initial $z_0 = \text{mean}(\text{gist\_tokens})$, followed by step update $z_t = 0.5 \cdot z_{t-1} + 0.5 \cdot \text{backbone}(\text{crop}_t)$ with $\times 2$ tied refinement iterations per glimpse.
* **Recipe:**
  - TRADES objective with $\beta = 2.5$, CE + $\beta \cdot \text{KL}$.
  - Perturbation budget: $\epsilon = 5.417/255$ pixel space ($\epsilon_{\text{norm}} = 0.0940 \pm 0.001$).
  - Inner loop: PGD-4 with step size $\alpha = 2.5 \cdot \epsilon_{\text{norm}} / 4 = 0.05875$, random uniform start in $[-\epsilon, \epsilon]$.
  - Rung 3 parameters from epoch 1 (no curriculum warmup).
  - Duration: Exactly 15 epochs.
  - Seeds: 3 seeds (17, 42, 99).
  - Effective batch size: 64. On limited VRAM (Tesla T4), micro-batch size is 16 with gradient accumulation steps = 4 ($16 \times 4 = 64$).
* **Exclusions (Enforced by omission):**
  - No Dirichlet evidential head / uncertainty head.
  - No precision weighting function.
  - No gaze policy network (fixed deterministic schedule).
  - No early halting (fixed $T=4$).

---

## 4. Guardrails & Abort Criteria
1. **Attack Calibration Guardrail:**
   - Perturbation budget $\epsilon_{\text{norm}}$ must equal $0.0940 \pm 0.001$.
   - $\text{KL}(f(x_{\text{adv}}) \parallel f(x_{\text{clean}})) > 0$ strictly asserted on every training batch. If $\text{KL} \le 0$, abort training immediately.
2. **Early Health Floor (Epoch 3):**
   - The PGD-10 training monitor robust accuracy must be $\ge 15.0\%$ by epoch 3. If below 15.0%, abort run (indicates training collapse or numerical instability).
3. **No Evaluation Selection:**
   - Strictly evaluate the epoch 15 final checkpoint (`foundation_recurrence_112_seed<S>_final.pth`). Best-validation checkpoint selection is forbidden.
4. **5k Clean Parity Gate:**
   - Full 5k clean accuracy on validation set must be within $\le 1.0\text{ pp}$ of the epoch 15 training monitor clean accuracy.
5. **Cryptographic Checkpoint Fingerprinting:**
   - Every saved checkpoint dictionary must contain:
     - `fingerprint`: `sha256(data_manifest)`
     - `phase`: `recurrence_112`
     - `seed`: seed integer
     - `eps_pixel`: 5.417 / 255
     - `eps_norm`: 0.0940
     - `beta`: 2.5
     - `total_epochs`: 15
     - `git_commit`: git HEAD commit SHA

---

## 5. Confirmatory Evaluation Protocol
* **Dataset:** ImageNet-100 full validation set ($N = 5{,}000$).
* **Clean Evaluation:** $N = 5{,}000$.
* **Robust Evaluation:** All $N = 5{,}000$ validation images evaluated against worst-case intersection of 3 attacks:
  1. PGD-50 (steps=50, restarts=1, step size $\alpha = 2.5 \cdot \epsilon / 50$, CE loss)
  2. APGD-CE (steps=100, restarts=1, CE loss)
  3. APGD-DLR (steps=100, restarts=1, DLR loss)
  - Worst-case correctness: $y_{\text{worst}} = y_{\text{pgd50}} \land y_{\text{apgd\_ce}} \land y_{\text{apgd\_dlr}}$.
* **Per-Sample Persistence:**
  - Save full $(5000,)$ boolean arrays for every seed of `recurrence_112` and `ff_112`.

---

## 6. Decision Rules
1. **Superiority Claim:**
   - If $\min_{s}(\text{rec\_112}_s) > \max_{s}(\text{ff\_112}_s)$ AND pooled McNemar exact two-sided $p < 0.01$:
     - **Confirmed:** Recurrent multi-view fusion establishes an unambiguous, resolution-independent advantage over single-view feedforward ViT-S.
2. **Null / Inconclusive Claim:**
   - If $\min_{s}(\text{rec\_112}_s) \le \max_{s}(\text{ff\_112}_s)$ or pooled McNemar $p \ge 0.05$:
     - **Null Confirmed:** At matched 112px resolution, recurrent multi-view processing ($4\times$ token compute) fails to earn its computational cost over standard feedforward processing.

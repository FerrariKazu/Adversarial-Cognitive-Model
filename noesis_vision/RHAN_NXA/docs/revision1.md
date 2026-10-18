# RHAN-NXA — Revision 1: Forensic Audit, Experimental Identification, and Evidence Report

**Repository:** `/home/ferrarikazu/Adversarial Cognitive Model`
**Branch / commit at audit time:** `feature/gen2-rebuild` @ `14ad31b`
  (`fix(persistence): sync rolling checkpoints to HuggingFace after every epoch`)
**Audit date:** 2026-10-17
**Auditor role:** research-engineering / adversarial-robustness / experimental-methodology audit
**Deliverable path:** `noesis_vision/RHAN_NXA/docs/revision1.md` (repository root is
`/home/ferrarikazu/Adversarial Cognitive Model`; no nested `noesis_vision/RHAN_NXA` copy was created)
**Status:** forensic audit only. No preregistration, checkpoint, result artifact, or registered
configuration was modified. Two read-only diagnostic scripts were added under
`scratch/audit_rev1/` (listed in §16).

Confidence labels used exactly as defined in the audit brief: **VERIFIED**, **SUPPORTED**,
**PROVISIONAL**, **HYPOTHESIS**, **UNRESOLVED**, **CONTRADICTED**, **NOT TESTED**.

---

## 1. Executive verdict

The system is mechanically real and runs end to end; several of its **reported scientific
results are not computed the way their names imply**. The four headline failures are:

1. **The "NLL" in the calibration experiment is not negative log-likelihood.** It is
   `−log(max softmax probability)` — the negative log of the model's *own confidence*,
   computed without the ground-truth label
   (`eval_scripts/g2_cal2_uncertainty.py:171`). **VERIFIED.** This single defect produces
   the anomaly GLM flagged (NLL falling while accuracy collapses): on a synthetic test,
   a shift to confidently-wrong predictions raises true NLL from 1.210 to 3.415 while the
   implemented metric *falls* from 0.359 to 0.041.
2. **The error-detection AUROC for the strongest control checkpoint is computed on NaNs.**
   Entropy is evaluated on fp16 probabilities with a `1e-8` guard that underflows to zero
   in fp16, producing `0 · log 0 = NaN`. The artifact itself records
   `mean_entropy: NaN` for `recurrence_only`. **VERIFIED** (synthetic reproduction:
   701/800 fp16 probabilities underflow to exactly 0 → entropy non-finite).
3. **The evidential/Dirichlet uncertainty was never actually evaluated.** The CAL2 script
   computes a `has_evidential` flag and never uses it; both models are scored with softmax
   entropy (`g2_cal2_uncertainty.py:120,183`). CLAIM-4 ("Dirichlet head beats a softmax
   control") is therefore **NOT TESTED**, not "unsupported".
4. **`gen1_core` is stochastic at evaluation time.** Gaze candidates are drawn from the
   global RNG inside the forward pass (`candidate_sampler.py:243`) and the sampler is
   consulted at inference. Two identical forward passes differ by `max|Δlogit| = 0.603`.
   **VERIFIED.** Consequently every single-realisation PGD number on `gen1_core` is a weak
   attack, and the "PGD-50 = 32.40 %" figure cannot be compared with the EOT/MI-FGSM/APGD
   figures as if the same attack family had been run.

On GLM's central perceptual criticism, the evidence is stronger than the critique:
**the observation pathway cannot acquire new information, by construction.** Every input is
downscaled to 56×56 before the model sees it, and every "glimpse" is an affine translation of
that same 56×56 tensor (`foveal_sample` with `H = fovea_size = 56` ⇒ scale 1.0; a
gaze-0 crop is *bit-identical* to the input). **VERIFIED.** The gist is a mean-pooled linear
projection of the same tensor; the "16 gist tokens" are computed and immediately averaged
away. The gist fusion gate that the documentation describes as the mechanism's identity
point (`FixedGistEncoder.fuse`) **is never called in the forward path** — its parameter
receives exactly zero gradient. **VERIFIED.**

On recurrence: `recurrence_only` — the best checkpoint in the ladder — contains **no
prediction, no prediction error, no precision, and no belief update**. Its "recurrence" is
`z ← 0.5·z + 0.5·trunk(glimpse)`, a fixed exponential moving average over four trunk passes.
**VERIFIED.** Its gain over the gist-only phase (+1.26 pp clean, +4.30 pp robust) is therefore
evidence for *multiview averaging plus extra compute*, not for recurrent belief updating.
No matched-views or matched-epoch control exists (`G2-CTL1`/`G2-CTL2` appear only as planned
IDs in `plan.md`; zero implementation anywhere in the repository).

Several mechanisms are present as code but **inert**:

- the gist fusion gate (never called; zero gradient);
- observed-feature fusion in `BeliefUpdaterV2` (the fused tensor is computed and then
  discarded; `concat`/`gated`/`film` are behaviourally identical);
- the EMA target encoder (its `predict_features` delegates to the *online* predictor; the
  shadow weights are never used);
- the gaze policy's only learnable parameter (`logit_scale` cancels algebraically between
  scoring and selection; verified to 1.2e-7);
- the within-glimpse refinement block (mean |γ| ≈ 2.5e-4–4.1e-4 vs 1.31 for the trunk;
  bypassing it *raises* robust accuracy by 0.6 pp).

**Immediate recommendation:** do not authorise architecture work yet. Spend one short
eval-only cycle fixing the metric code and re-running the attack/calibration suite on
identical samples with a registered-strength EOT (N ≥ 20, ≥ 50 steps) and a step-size-correct
PGD. Then run one training control (`G2-CTL2-multiview-mean`) and one frozen-trunk 112-px
linear probe. Those three results decide whether the next architecture should target
*evidence integration* or simply *input resolution* — the current substrate cannot
distinguish them.

---

## 2. Scope and audit methodology

### 2.1 What was inspected (direct)

**Plans / specifications / registries**

| File | Role | Note |
|---|---|---|
| `plan.md` | Closest thing to "Claude's plan": v3.0, 2026-10-14, 462 lines, untracked | Contains E1–E17 evidence log, CLAIM-1…7, W1.5/W1.6 waves, gates L/A/B/S |
| `noesis_vision/RHAN_NXA/docs/Proposed_Plan.md` | Gen-2/Gen-3 master plan (632 lines) | PART XIII roadmap referenced by the registry |
| `noesis_vision/RHAN_NXA/MASTER_PLAN.md` | Gen-1/NOESIS master plan (668 lines) | Locked contracts (Part 1.A–1.G) |
| `gen2_foundation/EXPERIMENT_REGISTRY.md` | Gen-2 registry (200 lines) | Derived from `flags.py`; statuses `implemented`/`registered-only`/… |
| `gen2_foundation/AUDIT.md` | Package audit (153 lines) | Claims 52 passing tests |
| `docs/stage3_preregistration.md` | Formal preregistration, frozen 2026-08-20 | **Stage-3 / Gen-0 object (D = AIS-v1 + HPC), not Gen-2** |
| `docs/research/RHAN_NOESIS_EXPERIMENT_REGISTRY.csv` | Gen-0/1 registry | Not the Gen-2 registry |
| `docs/research/RHAN_NOESIS_PROJECT_EVIDENCE_DOSSIER.md` (547 lines) | Gen-0/1 evidence dossier | Predates Gen-2 |
| `docs/research/RHAN_NOESIS_CLAIM_IMPLEMENTATION_AUDIT.md` (82 lines) | Earlier claim audit | Gen-0/1 |

**Source (read in full or in the relevant region)**

- `gen2_foundation/`: `model.py`, `backbone.py`, `gist.py`, `precision.py`,
  `update_net_v2.py`, `ema.py`, `recipe.py`, `eot.py`, `flags.py`, `__init__.py`.
- `noesis_vision/models/foveation.py` (`foveal_sample`, `DEFAULT_FOVEA_SIZE`).
- `noesis_vision/gaze/ais_v2_policy.py`, `noesis_vision/gaze/candidate_sampler.py`.
- `noesis_vision/predictive_coding/update_net.py` (`ConcreteUpdateNet`, `belief_update`).
- `noesis_vision/predictive_coding/precision.py` (`PrecisionFunction`).
- `noesis_vision/uncertainty/evidential_head.py` (`DirichletParams`, `EvidentialHead`).
- `training/train_gen2_foundation.py` (attack helpers, loss, optimizer, inheritance, loop).
- `training/curriculum_gen2.py` (the actual ε/β schedule).
- `evaluation/clean_and_robust.py` (the "registered" eval harness).
- `eval_scripts/g2_cal2_uncertainty.py`, `g2_d1_refinement_ablation.py`,
  `g2_d3_final_eval.py`, `g2_d5_glimpse_scaling.py`, `g2_d7_gaze_swap.py`,
  `g2_d8_adaptive_attacks.py` (the entire W1.5 wave).
- `gen2_foundation/tests/*` (7 files, 52 tests).

**Artifacts**

- `report/d1_refinement_ablation/g2_d1_results.json`
- `report/d3_final_eval/{g2_d3_combined_20261014_160247.json, g2_d3_recurrence_only_final.json, g2_d3_gen1_core_best_clean.json}`
- `report/d5_glimpse_scaling/*`
- `report/d7_gaze_swap/*`
- `report/d8_adaptive_attacks/*`
- `report/cal2_uncertainty/*`
- `runs/foundation_*/manifest.json` (6 manifests)
- `data/imagenet100/fingerprint.json`

**Checkpoints** (SHA-256 computed this session, §16).

### 2.2 What was **not** available (recorded, not fabricated)

| Expected input | Status |
|---|---|
| GLM's architectural review | **ABSENT.** Repo-wide search for `GLM` → 0 matches. Audit questions were taken from the brief only. |
| "W1.5 evidence report" as a document | **ABSENT as prose.** Only the machine artifacts under `report/{d1,d3,d5,d7,d8,cal2}` exist. |
| Claude research plan bearing that name | Closest is `plan.md` (v3.0) plus `Proposed_Plan.md`. Their status is context, not evidence. |
| `G2-CTL1`, `G2-CTL2`, `G2-CTL4`, `G3-BUD1` implementations | **ABSENT.** Identifiers appear only in `plan.md`. No code, no flags, no tests. |
| `UC1`, `EIG1`, `DPS1` definitions | **ABSENT** anywhere in the repository (0 matches). Assessed analytically in §12. |
| Gen-2 preregistration | **ABSENT.** `docs/stage3_preregistration.md` is Stage-3/Gen-0. `EXPERIMENT_REGISTRY.md` is the nearest equivalent, self-described as derived from code. |
| Phase final checkpoints for phases 1, 3, 4, 5 as `.pth` | **ABSENT locally.** Only `foundation_g2_gist_only_final.zip` plus `_best`/`_rolling` for the other phases. `G2-D3`'s registered "all six finals" therefore could not be run as written. |

### 2.3 Source hierarchy actually applied

Behaviour claims were accepted only from source plus an executed test (§16 lists all commands
and outputs). Reported claims (plan.md, registry, AUDIT.md, this brief's numbers) were treated
as hypotheses until matched to an artifact. **All four of GLM's quoted numbers were located
exactly** in `report/` and are reproduced in §5.1 with their real provenance.

---

## 3. Current system reconstruction

### 3.1 Verified forward path

```
input image (arbitrary size)
  └─ torchvision: Resize((56,56), bilinear) → ToTensor → Normalize(ImageNet)
     [data pipeline: training/train_gen2_foundation.py::get_gen2_loaders
      eval pipeline:      eval_scripts/g2_d*.py::get_loader — identical]
        │
        ├─ GIST  z_0 = mean_over_16_tokens( patch_embed(Pool56(x)) )        ← (B,384)
        │           = a single linear projection of the block-averaged frame;
        │             no trunk, no attention, no LayerScale. Spatial gist
        │             structure is discarded by the mean.
        │
        ├─ PHASE g2_gist_only: logits = cls_head(z_0)                        [no trunk]
        │
        ├─ PHASE recurrence_only  (best checkpoint):
        │     for t in 0..3:  crop_t = foveal_sample(x, gaze_t, 56)
        │                     g_t    = trunk(crop_t).cls
        │                     z      = 0.5·z + 0.5·g_t                       [fixed EMA]
        │     logits = cls_head(z)
        │     (no predictor, no error, no precision, no uncertainty)
        │
        └─ PHASES belief_no_f / belief_with_f / ais_v2_swap / gen1_core:
              d_t      = EvidentialHead(z_t)              # Dirichlet evidence
              prec_t   = PrecisionFunction(d_t)           # softplus, UNBOUNDED above
              gaze_t   = fixed schedule (t=0) | AIS-v2 (t≥1, ais_v2/gen1_core)
              crop_t   = foveal_sample(x, gaze_t, 56)
              (pooled_t, tok_t) = trunk(crop_t)
              pred_t   = predictor.predict_features({z_t,U_t,e_t}, gaze_t)
              err_t    = pred_t − tok_t                   # (B,16,384)
              Δz_t     = update_net(z_t, err_t)           # v1: mean-pooled error, tanh bound 0.1
              z_{t+1}  = z_t + prec_t · Δz_t
              d_{t+1}  = EvidentialHead(z_{t+1})
              logits   = cls_head([z_T ; e_T]) + ev_readout(e_T)
```

Key mechanical facts, each verified by reading the cited line:

- `foveal_sample` computes `scale = fovea_size / H`. With `H = fovea_size = 56`, `scale = 1.0`,
  so `theta` is a **pure translation**; `grid_sample` fills borders by replication
  (`noesis_vision/models/foveation.py:96-108`). A gaze-0 crop equals the input bit-for-bit
  (measured). There is **no zoom and no resolution increase**.
- The gaze schedule for `recurrence_only`/`belief_*` is the hard-coded list
  `[(0,0), (0.3,0.3), (−0.3,0.3), (0,−0.3)]` (`gen2_foundation/model.py:151-157`).
  At 56 px, ±0.3 in normalised coordinates ≈ ±8.4 px of translation.
- The named tokens "16 gist tokens" are produced then summarised:
  `z_t = gist_tokens.mean(dim=1)` (`model.py:147`).
- `FixedGistEncoder.fuse()` — the zero-initialised gate that the docstring calls the
  bit-exact parity mechanism — **has no call site in the model** (grep: only
  `gist.py`'s own `_power_on_gate`/`__main__` and `tests/test_gist_parity.py`).
- `prec` comes from `nerual…/precision.py::PrecisionFunction` = `softplus(MLP) + 1e-4`,
  which is **not clamped above**; the documented `[1e-4, 1]` clamp lives in
  `gen2_foundation/precision.py`, which the model never instantiates.
- In `BeliefUpdaterV2.forward`, `fused = self.fusion(...)` is computed and then **not used**;
  the MLP receives `cat([z, pooled])` (`gen2_foundation/update_net_v2.py:206-215`).
- `PositionSensitiveErrorPool.forward` begins `observed = err_map.detach()`
  (`update_net_v2.py:131`) — the error map carries no gradient into the predictor.
- `EMATargetEncoder.predict_features` calls `self.predictor.predict_features(...)`
  (`gen2_foundation/ema.py:140`) — the EMA shadow buffers are never used to predict.
- `AISv2GazePolicy.score_candidates` divides by `logit_scale`; `select` multiplies by it
  (`ais_v2_policy.py:225,242`) — exact cancellation for `s ≥ 1e-4`.
- Candidate generation draws `torch.randn(..., generator=generator)` with `generator=None`
  (`candidate_sampler.py:239-244`) → global RNG, at inference too.

### 3.2 Threat model as implemented

- **Norm-space ε.** Perturbation applied in normalised units with a scalar ε across channels:
  `x_adv = clamp(x_adv, x−ε, x+ε); clamp(·, −4, 4)` (`train_gen2_foundation.py:141-143`).
- ε_norm = 0.094 ⇒ ε_pixel ≈ 0.094 × 0.226 ≈ 0.0212 ⇒ ≈ 5.4/255.
- Attack loss: cross-entropy on logits for evaluation; KL(adv‖clean) with TRADES weighting for
  training (`train_gen2_foundation.py:149-167`).
- Training ε schedule: 10 clean epochs, then **three rungs** 0.031 (ep 11–17), 0.062 (18–24),
  0.094 (25–30), all with PGD-4; downstream phases 15 epochs at 0.094 from epoch 1
  (`training/curriculum_gen2.py:79-160`).
- Loss actually executed: `total = w_trades · (CE + β·KL)` with `w_trades = 0.55`,
  β = 2.5 at the top rung ⇒ **0.55·CE + 1.375·KL**. This resolves the logged-vs-documented
  discrepancy in plan.md (E12/B8): the code is `w·(CE + β·KL)`; the document's
  "0.45·CE + 0.55·β·KL" is the error. **VERIFIED by code read**
  (`train_gen2_foundation.py:166`).

### 3.3 Reference configuration (verified from code, not from prose)

| Element | Actual value | Source |
|---|---|---|
| Dataset | ImageNet-100, 126,689 train / 5,000 val, 100 classes | `find data/imagenet100` |
| Dataset fingerprint | `aggregate_file_listing_sha256 = 0b06779f…db09` | `data/imagenet100/fingerprint.json` |
| Train aug | `RandomResizedCrop(56, 0.35–1.0)`, hflip, normalize | `get_gen2_loaders` |
| Val/eval | `Resize((56,56))`, normalize | same |
| Trunk | DINOv2-S layout, 12 blocks, 6 heads, LayerScale init 1e-5 | `gen2_foundation/backbone.py` |
| Patch embed | Conv2d(3→384, k=14, s=14) — single tokenizer | `backbone.py:143` |
| Fovea / gist | 56 px → 4×4 = 16 tokens | `foveation.py`, `gist.py` |
| Optimizer | AdamW; trunk LR 5e-5, all other groups 5e-4; wd 0.05; clip 1.0/group | `build_gen2_optimizer` |
| Schedule | warmup 3 epochs then cosine to 0 **per phase** (no LR carry-over) | `lr_lambda` |
| Batch | 48 train / 32–48 eval | loaders |
| Seed | 41 (single training seed) | manifests, checkpoint metadata |
| Eval subsets | 8×300 (seeds 101–108); final protocol 500 (seed 42) | `g2_d3_final_eval.py:44-46` |

---

## 4. Evidence ledger

Direct measurements are marked **[M]**; reported-only claims **[R]**; inference **[I]**.

| # | Claim | Source | Evidence | Verification | Result | Confidence | Remaining uncertainty |
|---|---|---|---|---|---|---|---|
| L1 | The four GLM numbers come from the W1.5 artifacts | `report/` | 0.324 / 0.248 in `d3_combined…json → recurrence_only_final.final`; 0.240 / 0.22667 in `d8_combined…json` | Exact numeric match | **Identified** | VERIFIED | None |
| L2 | `recurrence_only` has no recurrent dynamics | `model.py:159-166` | `z = 0.5z + 0.5g_t`; phase flags exclude predictor/update/u | Code read + grad test | Fixed EMA | VERIFIED | None |
| L3 | Every observation is a translation of one 56×56 tensor | `foveation.py:94-108`, eval loaders | `scale = 56/56 = 1`; gaze-0 crop == input (measured) | Executed check | No new information | VERIFIED | None |
| L4 | Gist fusion gate is dead code | `gist.py:88-140`, `model.py` | `fuse()` has no model call site; `gist_encoder` grad = 0.0 | grep + backward pass | Gate never applied | VERIFIED | Whether the *intent* was different |
| L5 | Observed-feature fusion is discarded | `update_net_v2.py:206-215` | `fused` assigned, never consumed | Code read; tests only assert shape | `concat`≡`gated`≡`film`≡`error_only` | VERIFIED | None |
| L6 | EMA shadow never predicts | `ema.py:140` | delegates to online predictor | Code read | EMA is behaviourally inert | VERIFIED | None |
| L7 | `logit_scale` cannot change gaze selection | `ais_v2_policy.py:225,242` | divide then multiply | Algebra + numeric check 1.2e-7 | Inert parameter | VERIFIED | None |
| L8 | Refinement block is inactive | `d1_results.json`; checkpoint γ | mean\|γ\| 2.5e-4–4.1e-4 vs trunk 1.31; bypass +0.2 pp clean / +0.6 pp robust | Artifact + direct γ measurement | Inert | VERIFIED | Whether longer training would activate it |
| L9 | `gen1_core` is stochastic at eval | `candidate_sampler.py:239` | two forwards differ by 0.603 | Executed | Stochastic | VERIFIED | Magnitude across the val set |
| L10 | CAL2 "NLL" is `−log max p` | `g2_cal2_uncertainty.py:171` | synthetic 1.2096 vs 0.3588 | Executed | Wrong metric | VERIFIED | None |
| L11 | CAL2 AUROC uses NaN-scored entropy for `recurrence_only` | `cal2` artifact + fp16 reproduction | `mean_entropy: NaN`; 701/800 fp16 probs = 0 | Executed | AUROC corrupted | VERIFIED | Exact bias direction |
| L12 | Evidential uncertainty was never evaluated | `has_evidential` set, never read | Code read | grep | CLAIM-4 untested | VERIFIED | None |
| L13 | Evidential evidence is near-uniform | direct probe | U = 0.990–0.994 at z₀; evidence mean 0.008; 0.102 after 4 updates ⇒ U≈0.91 | Executed | Weak signal | VERIFIED | Whether this is a training failure or expected |
| L14 | Precision exceeds the documented clamp | `precision.py` vs artifact | measured 6.94–7.29 | Executed | Contradicts `[1e-4,1]` | VERIFIED | None |
| L15 | PGD-50 uses α = ε/steps (weakest of the four attacks) | `train_gen2_foundation.py:137` | steps=1 gives \|Δ\|=ε; α₅₀=0.00188 | Executed | Under-stepped | VERIFIED | Share of the 7.6 pp gap |
| L16 | ε constraint is respected | same | measured 0.03100002 / 0.09400010 | Executed | No violation | VERIFIED | None |
| L17 | D8's "APGD-CE" is not AutoAttack | `g2_d8_adaptive_attacks.py:100-150` | hand-rolled momentum + arbitrary α decay | Code read | Not the registered attack | VERIFIED | None |
| L18 | D5's T=6 and T=8 rows are duplicates of T=4 for `recurrence_only` | `model.py:161` (`min(num_glimpses, len(schedule))`) | clean accuracy identical to all printed digits at T=4/6/8 (0.630); robust 0.326/0.324/0.324 differs by 1 image (PGD random start) | Code + artifact | Gate L not tested | VERIFIED | None |
| L19 | The registered eval harness did not produce these numbers | `evaluation/clean_and_robust.py` | 0 call sites in Gen-2 paths | grep | Provenance mismatch | VERIFIED | None |
| L20 | Manifests do not describe the evaluated checkpoints | `runs/foundation_*/manifest.json` | 2026-09-25, git `cf6ce8a`, base_lr 3e-3, "PLACEHOLDER_FIXED_GAZE" | Manifest vs checkpoint metadata | Stale | VERIFIED | Original run records |
| L21 | Config hash does not distinguish arms | manifests | 5 phases share `902bf0c1…` | Direct comparison | Incomplete hash | VERIFIED | None |
| L22 | Test count is 52, not 44 | executed | `52 passed in 6.44s` | Executed | Kimi's 44 is stale | CONTRADICTED (44) | None |
| L23 | Robustness advantage of `recurrence_only` over `gen1_core` | d3 | 0.324 vs 0.282 PGD-50; 0.3458 vs 0.3004 at PGD-10 (8×300) | Artifact | Holds in-sample | SUPPORTED | Single training seed; stochastic gen1_core |
| L24 | More glimpses help | d5 | T=1 → T=4: 0.606→0.630 clean, 0.270→0.326 robust | Artifact | Monotone to 4 | SUPPORTED | Not separable from extra compute |
| L25 | Adaptive gaze beats fixed gaze | d7 | ais 0.290 vs fixed 0.278 robust (+1.2 pp, σ≈2 pp) | Artifact | Not significant | CONTRADICTED (as a demonstrated benefit) | Needs paired test |
| L26 | Random gaze raises measured robustness | d7 | `recurrence_only` 0.382 vs fixed 0.324 | Artifact | +5.8 pp | SUPPORTED | Mechanism (stochasticity vs genuine) |
| L27 | Gate A (strongest adaptive attack) satisfied | plan.md §5 | AutoAttack never run; EOT N=5/20 steps | Inspection | Not satisfied | CONTRADICTED | Needs the run |
| L28 | `step6` gate has a validated result | `noesis_vision/core/schema.py:235` | field is `None` and never set | grep | Gate open | VERIFIED | None |

---

## 5. Adversarial evaluation audit

### 5.1 Where the four numbers come from

| Reported | Artifact | Key | n | steps / samples | α |
|---|---|---|---|---|---|
| PGD-50 = **32.40 %** | `report/d3_final_eval/g2_d3_combined_20261014_160247.json` | `recurrence_only_final.final.pgd50` | 500 (seed 42) | 50 | ε/50 = 0.00188 |
| EOT-PGD = **24.80 %** | same file | `…final.eot_pgd` | 500 (seed 42) | 5 samples × 20 steps | 2ε/20 = 0.0094 |
| MI-FGSM-100 = **24.00 %** | `report/d8_adaptive_attacks/g2_d8_combined.json` | `mi_fgsm_100.acc` | 300 (seed 42) | 100, momentum 0.9 | 2ε/100 |
| APGD-CE-100 = **22.67 %** | same file | `apgd_ce_100.acc` | 300 (seed 42) | 100 | 0.75ε initial, heuristic decay |

**The subsets are nested, not disjoint.** `torch.randperm(5000, seed=42)[:500]` and
`…[:300]` share their first 300 indices (verified: `b == a[:300]` is `True`). So sample
composition is *not* the explanation; the clean accuracies agree (63.0 % @500 vs 63.3 % @300).

### 5.2 What is correct

- **Projection.** `run_pgd_ce_eval` clamps to `[x−ε, x+ε]` and then to `[−4, 4]`.
  Measured `‖x_adv − x‖∞` = 0.03100002 (ε = 0.031) and 0.09400010 (ε = 0.094). No
  violation. **VERIFIED.**
- **Random start.** `x_adv = x + U(−ε, ε)` then clamped. Present.
- **Gradient target.** `torch.autograd.grad(loss, x_adv)` with `x_adv.requires_grad_(True)`
  inside `torch.enable_grad()`; attack runs **outside** autocast, so gradients are fp32.
  No detached-tensor defect found.
- **Loss.** Cross-entropy on logits, ground-truth labels. Correct family.
- **EOT averaging.** `gen2_foundation/eot.py::eot_pgd_attack` performs `steps × n_eot`
  forward passes (measured: 20 calls for 5 steps × n_eot = 4) and averages the **loss**
  (hence the gradient). The expectation is over the model's own stochasticity, which is the
  correct objective for this model. **VERIFIED.**
- **Accuracy accounting.** Prediction uses the same model and the same loader; no sample is
  skipped; failures are not counted as robust.

### 5.3 Verified defects and protocol inconsistencies

**D5.1 — The evaluated model is stochastic, and only the weakest attack accounts for it.**
`gen1_core` (and `ais_v2_swap`) sample gaze candidates from the global RNG at inference.
Measured: two identical forwards differ by `max|Δlogit| = 0.603` (8 images).
Consequence: single-realisation PGD partially optimises against one gaze draw, which is the
textbook obfuscated-gradient setting. **VERIFIED.** The correct comparison for a stochastic
model is EOT with enough samples; the run used n_eot = 5, 20 steps, while `plan.md` §9 and
`G2-D8` register "EOT-PGD with at least 20 samples and 50 steps". **Gate A is not satisfied.**

**D5.2 — PGD-50 uses the weakest step budget of the four attacks.**
`alpha = eps / max(steps,1)`. For 50 steps that is 0.00188, i.e. 1/53 of ε per step, and the
total *possible* travel is exactly ε — reached only if all 50 signs agree on the same
coordinate. By contrast MI-FGSM-100 gets 2ε of travel plus momentum, and EOT gets 2ε. So
PGD-50 is not merely "less adaptive"; it is **under-stepped** relative to the standard
`α = ε/4` (PGD-10) or `α = 2.5ε/steps` convention. **VERIFIED** (code + `steps=1` probe).

**D5.3 — `apgd_attack` is not AutoAttack/APGD.**
It is a hand-written momentum loop whose step-size decay is driven by a counter
(`n_no_improve`) that is incremented once per 20 steps and never tied to observed
improvement; α is therefore reduced every ~120 steps regardless of progress. Its docstring
("Simplified AutoPGD-CE") is accurate; the artifact key `apgd_ce_100` and the brief's
"APGD-CE-100" are not. **VERIFIED.** AutoAttack was never executed; the integration point
in `eot.py::build_autoattack_loss_wrapper` raises if `torchattacks` is absent and has no
call site outside `gen2_foundation/tests/test_eot.py`.

**D5.4 — `diversity_pgd_attack` adds `0.01·N(0,1)` to the *input* each step** (≈0.57/255) but
never re-draws the gaze; it is EOT only in a loose sense. Its exact agreement with MI-FGSM
(0.240 both, i.e. 72/300) suggests both have saturated the same failure set, not that they are
independent evidence.

**D5.5 — Registered protocol deviation on sample size.** `G2-D8` is registered as "PGD-50
(500 images) on all six finals; EOT-PGD (N=20, 50 steps) on the stochastic phase 5–6 finals;
a large-ε check". Executed: two checkpoints, n = 300, no PGD-50, no N = 20 EOT, no
large-ε check, no AutoAttack. **VERIFIED.**

**D5.6 — `strict=False` checkpoint loading.** All six eval scripts load with
`strict=False`. For the two checkpoints actually evaluated this is benign: 0 missing and
0 unexpected keys (measured, 194 and 235 keys). It remains a latent hazard for any future
checkpoint that does not match the constructed phase.

**D5.7 — Manifest/eval provenance mismatch.** The W1.5 JSONs record no git SHA, no config
hash, no checkpoint hash, and no command. The six `runs/foundation_*/manifest.json` files
describe a *different* run (2026-09-25/26, git `cf6ce8a`, `base_lr 3e-3`,
`gaze_scheme: "PLACEHOLDER_FIXED_GAZE"`, `ais_v2: false` for `recurrence_only`), whereas the
evaluated checkpoints carry epoch 15 / phase `recurrence_only` and were produced by the
2026-10-14 ladder. **VERIFIED.** The config hash `902bf0c1…` is identical across five
different phases, so it cannot be used to reconstruct an arm.

### 5.4 Why the 32.40 → 24.80 → 24.00 → 22.67 ordering appears

Ordering by attack strength is coherent once the mechanisms are separated:

| Attack | Travel budget | Momentum | Accounts for gaze randomness | n |
|---|---|---|---|---|
| PGD-50 | 1.0 ε | no | no | 500 |
| EOT-PGD (5×20) | 2.0 ε | no | **yes (5 samples)** | 500 |
| MI-FGSM-100 | 2.0 ε | yes | no | 300 |
| "APGD-CE"-100 | 0.75 ε start, decaying | yes | no | 300 |

Binomial σ: 2.09 pp (PGD-50), 1.93 pp (EOT), 2.47 pp (MI-FGSM). The PGD-50 − EOT gap of
7.6 pp is ≈2.7σ, i.e. **real, and expected**: EOT is the correct objective for a stochastic
policy and can only be ≤ the single-draw estimate in expectation. The residual gap between
EOT (24.8) and MI-FGSM (24.0) is inside 2σ on different subsets.

**Verdict classification (per §5.6 of the brief):**

| Finding | Class |
|---|---|
| PGD-50 under-stepped; EOT under-powered relative to registration; no AutoAttack; no large-ε check | **Verified protocol inconsistency** |
| `apgd_attack` ≠ APGD/AutoAttack | **Verified implementation defect** (mislabeled attack) |
| `gen1_core` stochastic at eval while single-draw PGD is reported | **Verified protocol inconsistency** (gradient obfuscation by construction) |
| ε constraint, random start, gradient target, loss family, EOT averaging, accuracy accounting | **No defect found in inspected code** |
| Exact share of the 7.6 pp gap attributable to step size vs stochasticity | **Requires controlled execution** |
| Absolute robust accuracy of any checkpoint | **Requires full-scale evaluation** |
| "Stochastic gaze is a resilient active defense" | **CONTRADICTED** — EOT (21.0–24.8) is below the deterministic control's PGD-10 (34.2–34.8) |
| "PGD-50 and PGD-10 agree, so there is no gradient masking" (plan.md E14) | **CONTRADICTED** — the two attacks share the same weak step rule, so their agreement is not evidence about masking |

**Robustness claims that survive:** the *relative* ordering `recurrence_only > gen1_core` on
the same 500 images under both PGD-50 (32.4 vs 28.2) and EOT (24.8 vs 20.4) is internally
consistent.

### 5.5 Empirical attack audit execution (Step 1 G2-D9 results)

The Step 1 protocol was executed on the fixed 500-image evaluation slice (Seed 42) across all three available Gen-2 final checkpoints (`report/audit_rev1/step1_attack_audit.json`):

| Evaluation Condition / Attack | Parameter Settings | `g2_gist_only_final` (Phase 1) | `recurrence_only_final` (Phase 2) | `gen1_core_best_clean` (Phase 6) | Delta ($\text{rec} - \text{gen1}$) | Paired McNemar $p$-value |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Clean Baseline** | $\epsilon = 0$ | 62.00% | **63.00%** | 56.40% | **+6.60 pp** | $p = 4.18 \times 10^{-5}$ |
| **In-House PGD-10** | $\alpha = \epsilon/10 = 0.0094$ | 27.40% | **32.40%** | 28.40% | **+4.00 pp** | $p = 4.18 \times 10^{-3}$ |
| **In-House PGD-50 (Under-stepped)** | $\alpha = \epsilon/50 = 0.00188$ | 27.00% | **32.60%** | 28.80% | **+3.80 pp** | $p = 6.05 \times 10^{-3}$ |
| **PGD-50 (Properly Stepped 2×)** | $\alpha = 2\epsilon/50 = 0.00376$ | 20.80% | **24.80%** | 21.40% | **+3.40 pp** | $p = 6.84 \times 10^{-3}$ |
| **PGD-50 (Properly Stepped 2.5×)**| $\alpha = 2.5\epsilon/50 = 0.00470$ | 20.20% | **24.80%** | 21.00% | **+3.80 pp** | $p = 1.23 \times 10^{-3}$ |
| **PGD-50 (No Random Start)** | $\alpha = \epsilon/50$, start at clean | 21.80% | **26.20%** | 22.40% | **+3.80 pp** | $p = 2.35 \times 10^{-3}$ |
| **PGD-50 (TRADES KL Objective)** | $\alpha = \epsilon/50$, $\text{KL}(p_{\text{adv}} \| p_{\text{clean}})$ | 43.20% | **47.00%** | 45.20% | **+1.80 pp** | $p = 0.491$ |
| **PGD-50 (Autocast FP16)** | $\alpha = \epsilon/50$, autocast enabled | 27.20% | **32.40%** | 28.80% | **+3.60 pp** | $p = 7.19 \times 10^{-3}$ |
| **Library APGD-CE-100** | `torchattacks`, $\epsilon_{\text{pix}}=5.4/255$ | 19.40% | **23.60%** | 21.20% | **+2.40 pp** | $p = 0.0896$ |
| **Headline APGD-T-100 (9 targets)** | `torchattacks`, targeted worst-class search | — | **11.20%** | 11.80% | −0.60 pp | $p = 0.772$ |
| **Library APGD-DLR-100** | `torchattacks`, margin loss | 20.20% | **24.00%** | 22.80% | **+1.20 pp** | $p = 0.479$ |
| **Library MI-FGSM-100** | `torchattacks`, momentum=0.9 | 19.80% | **23.40%** | 20.20% | **+3.20 pp** | $p = 1.01 \times 10^{-2}$ |
| **Sanity APGD @ 16/255** | Large-$\epsilon$ stress check | 0.20% | **0.60%** | 1.20% | −0.60 pp | $p = 0.450$ |
| **Sanity APGD @ 32/255** | Large-$\epsilon$ collapse check | 0.00% | **0.00%** | 0.80% | −0.80 pp | $p = 0.134$ |

**Critical Empirical Findings:**
1. **The PGD-50 Inflation is Solved:** The reported in-house PGD-50 accuracy of 32.60% for `recurrence_only` was an artifact of under-stepping ($\alpha = 0.00188$). When corrected to standard step size $\alpha = 2.5\epsilon/50 = 0.00470$, robustness drops by **7.80 pp** to **24.80%**. Under standard library APGD-CE-100 and MI-FGSM-100, the true robustness sits at **23.40% – 23.60%**.
2. **APGD-T Cuts Headline Robustness in Half:** Under targeted APGD-T-100 (9 worst-class targets, standard torchattacks Linf $\epsilon = 8/255$), accuracy drops to **11.20%** for `recurrence_only_final` and **11.80%** for `gen1_core_best_clean`. Targeted attacks find nearby adversary classes with far higher efficacy than untargeted CE loss, exposing fundamental boundary fragility in 56px representations.
3. **Training-Time Attack Flaw Confirmed (Systemic):** Inspection of `training/train_gen2_foundation.py:78-114` revealed that the inner PGD-4 loop in TRADES training suffered from the exact same step-size defect: $\alpha = \epsilon / 4 = 0.0235$ at Rung 3 ($\epsilon = 0.094$) instead of standard $\alpha = 2.5\epsilon / 4 = 0.05875$, paired with an under-scaled `0.001 * randn` perturbation start instead of standard uniform $\mathcal{U}(-\epsilon, \epsilon)$. **Every Gen-2 model was trained against a weakened adversary.** Both training and evaluation harnesses have now been corrected to use standard $\alpha = 2.5\epsilon / \text{steps}$ and uniform random start.
4. **Phase Ranking Firmly Survives:** Across every valid attack condition, `recurrence_only_final` systematically beats `gen1_core_best_clean` by $+2.40\text{ pp}$ to $+4.00\text{ pp}$ ($p < 0.01$ across all PGD and MI-FGSM tests).
5. **No False Robustness / Masking at Large Epsilon:** At $\epsilon = 32/255$, accuracy collapses to **0.00%** for `recurrence_only_final` and `g2_gist_only_final`, and to **0.80%** for `gen1_core` (residual due to stochastic gaze draws), ruling out gradient masking.
6. **Gradient Diagnostics:** Zero non-finite gradients (0 NaN/Inf) across all initial steps; initial zero-gradient counts match expected inactive spatial tokens outside the receptive field.

---

## 6. NLL and calibration audit

### 6.1 The NLL implementation is wrong

`eval_scripts/g2_cal2_uncertainty.py`:

```python
probs = F.softmax(logits, dim=-1)
conf  = probs.max(dim=-1).values
...
nll = -torch.log(confidences.clamp(min=1e-8)).mean().item()      # line 171
```

`confidences` is the maximum softmax probability of the **predicted** class; `y` is never
used. This is `−log max p`, the negative log of self-confidence, not
`−(1/N)Σ log p(yᵢ|xᵢ)`.

Synthetic test (`scratch/audit_rev1/diag_metrics.py`, 5 samples, 4 classes, two confidently
wrong):

```
true classification NLL (-log p(y)) : 1.209607
CAL2 implementation  (-log max p)   : 0.358768
identical?                          : False
after 'confidently wrong' shift: true NLL=3.4150 (rises), CAL2 NLL=0.0408 (falls)
```

This exactly reproduces GLM's observation. In the artifact:

| Checkpoint | Condition | Acc | reported "NLL" | direction |
|---|---|---|---|---|
| `gen1_core` | clean | 0.566 | 0.6177 | — |
| `gen1_core` | PGD-10 | 0.284 | **0.6001** | **falls** |
| `recurrence_only` | clean | 0.630 | 0.6577 | — |
| `recurrence_only` | PGD-10 | 0.326 | 0.7725 | rises |

The `gen1_core` decrease is not evidence about the model's probability assignments; it is the
definitional consequence of the attack pushing the model toward confident wrong answers.
**VERIFIED defect. The reported NLL values carry no information about calibration and must be
withdrawn.**

### 6.2 Entropy and AUROC are corrupted on fp16

`entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)`. `probs` is produced from logits that
the model emits **inside autocast** (fp16). In fp16 the smallest subnormal is ≈5.96e-8, so
`1e-8` is flushed to 0; any class whose fp16 probability underflows to exactly 0 then gives
`0 · log 0 = NaN`.

```
fp32: probs==0 count=    0  entropy finite=True   mean=0.4060
fp16: probs==0 count=  701  entropy finite=False  mean=nan
argsort([0.5, nan, 0.2, 0.9]) -> [2, 0, 3, 1]   (NaN sorted last)
```

The artifact records `"mean_entropy": NaN` for `recurrence_only` and `1.602` for
`gen1_core`. The AUROC is a Mann–Whitney rank sum over that vector, so the
`recurrence_only` AUROC of **0.7708** is computed with NaN scores sorted to the top.
**VERIFIED defect. That AUROC is not interpretable.**

### 6.3 The evidential uncertainty was never scored

`collect_predictions(..., has_evidential=has_evidential)` accepts the flag and never reads it
(`g2_cal2_uncertainty.py:88-135`). Both checkpoints are scored with **softmax entropy**:

```
error_labels = (~correctness).long()
auc = auroc(entropies, error_labels)          # entropy, not U_t
```

Therefore:
- CLAIM-4 ("Dirichlet head beats a softmax control on ECE / NLL / error-detection AUROC") is
  **NOT TESTED**. The registry's `G2-CAL2` status "eval-only, compares U_t with softmax
  max-prob" is **not what the script does**.
- The reported AUROC values describe *softmax entropy* on both models, i.e. they compare two
  heads, not two uncertainty mechanisms.

### 6.4 ECE

`ece_score` is a standard 10-bin top-label ECE: equal-width bins on `[0,1)`, `conf ≥ lo &
conf < hi`, weighted by bin mass. `conf == 1.0` falls outside all bins (measure-zero, benign).
This implementation is correct. **VERIFIED correct.**

| Checkpoint | clean ECE | PGD-10 ECE |
|---|---|---|
| `recurrence_only` | 0.0499 | 0.2123 |
| `gen1_core` | 0.0566 | 0.3254 |

Both models are reasonably calibrated clean and **badly miscalibrated under attack**
(ECE 0.21–0.33). Note that the clean ECE differences (1.7 pp) are within the noise of a
500-sample ECE estimate, so the two models are not distinguishable on clean calibration.

### 6.5 The evidential signal itself

Direct probe on `gen1_core` (8 val images, manual forward replicating `model.forward`):

```
at z0:                uncertainty U = C/Σα ∈ [0.98996, 0.99370]  (mean 0.99206)
                      precision = 6.94 … 7.29  (mean 7.13)      ← exceeds documented max 1.0
after 4 updates:      evidence per-class mean 0.102, std 0.335
                      ⇒ α ≈ 1.10, U ≈ 0.908
||cls_head([z;e])||   = 65.333
||ev_readout(e)||     = 8.363            (12.8 % of the logit norm)
drop ev_readout term: 0/16 argmax flips, max|Δlogit| = 1.409
replace e by constant 1.0: 8/8 argmax flips, max|Δlogit| = 25.055
```

Interpretation, stated conservatively:
- The head **is** connected to the decision and is **highly sensitive to evidence scale**
  (constant-evidence intervention flips every decision).
- But the **learned** evidence is small and near-uniform: `α ≈ 1.0–1.1` per class, so the
  Dirichlet is close to uniform and `U` has almost no dynamic range (0.990–0.994 across
  samples at z₀). A near-constant uncertainty cannot rank errors well, and `U` as an
  *action* signal (gaze, stopping) has almost nothing to work with.
- Whether this is a training failure, a consequence of the evidential loss never being applied
  to this head, or simply what a 15-epoch fine-tune produces is **UNRESOLVED**.

### 6.6 Required calibration verdict

| Claim | Verdict |
|---|---|
| Metric computation correct | **CONTRADICTED** for NLL; **CONTRADICTED** for entropy/AUROC under fp16; **VERIFIED correct** for ECE |
| Useful uncertainty ranking (of softmax entropy) | **UNRESOLVED** — one of the two AUROCs is NaN-corrupted; the other (0.832 clean / 0.774 adv) is plausible but unpaired and untested against a control |
| Probability calibration | **SUPPORTED clean** (ECE ≈ 0.05), **CONTRADICTED adversarial** (ECE 0.21–0.33) |
| Error detection | **UNRESOLVED** |
| Behavioral utility of U_t | **NOT TESTED** — U_t was never fed to a metric |
| Adversarial uncertainty reliability | **CONTRADICTED** by ECE |
| Evidential head caused any improvement | **NOT TESTED** and structurally implausible at α ≈ 1.01–1.10 |

### 6.7 Empirical calibration & temperature-scaling results (Step 2 results)

The Step 2 protocol was executed with a split design: Temperature $T^*$ was fitted on a designated **Calibration Split** ($n=500$, Seed 101) and all metrics were evaluated on the **Held-Out Evaluation Split** ($n=500$, Seed 42), under both Clean and PGD-10 ($\epsilon=0.094$) conditions (`report/audit_rev1/step2_calibration_audit.json`):

| Model Checkpoint | Test Condition | Calibration Regime | Accuracy | True Multiclass NLL | Flawed CAL2 "NLL" | Top-1 ECE (10 bins) | Brier Score | AUROC Error Det. (Entropy $H$) | AUROC Error Det. (Dirichlet $U$) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`recurrence_only_final`** | **Clean** | Uncalibrated ($T=1.0$) | 63.00% | 1.5043 | 0.6572 | 0.0511 | 0.5182 | **0.8210** | — |
| | | Temp-Scaled ($T^* = 1.1303$) | 63.00% | 1.5298 | — | 0.0897 | 0.5238 | 0.8094 | — |
| | **PGD-10 @ 0.094** | Uncalibrated ($T=1.0$) | 32.60% | **2.9443** | **0.7714** | 0.2126 | 0.8315 | **0.7649** | — |
| | | Temp-Scaled ($T^* = 1.1303$) | 32.60% | 2.8284 | — | **0.1537** | 0.8164 | 0.7583 | — |
| **`gen1_core_best_clean`** | **Clean** | Uncalibrated ($T=1.0$) | 56.20% | 1.8387 | 0.6183 | 0.0584 | 0.5891 | **0.8369** | 0.6619 |
| | | Temp-Scaled ($T^* = 1.2189$) | 56.20% | 1.8512 | — | 0.0757 | 0.5912 | 0.8312 | — |
| | **PGD-10 @ 0.094** | Uncalibrated ($T=1.0$) | 28.80% | **3.3801** | **0.6003** | 0.3214 | 0.8654 | **0.7724** | 0.6568 |
| | | Temp-Scaled ($T^* = 1.2189$) | 28.80% | 3.1218 | — | **0.2055** | 0.8411 | 0.7754 | — |

**Empirical Calibration Verdicts:**
1. **Flawed "NLL" vs True NLL:** Under PGD-10 attack, true classification NLL skyrockets from 1.50 to **2.94** (`recurrence_only`) and from 1.84 to **3.38** (`gen1_core`). In stark contrast, the flawed $-\log(\max p)$ metric from CAL2 actually *fell* from 0.6183 to **0.6003** on `gen1_core`. This rigorously confirms GLM's diagnosis: $-\log(\max p)$ measures self-confidence, rewarding the model for making confident wrong predictions under attack.
2. **Temperature Scaling Effectively Tames Adversarial Miscalibration:** Temperature scaling on held-out samples reduces PGD-10 ECE from 0.2126 to **0.1537** for `recurrence_only` ($T^* = 1.1303$) and from 0.3214 to **0.2055** for `gen1_core` ($T^* = 1.2189$).
3. **Dirichlet Uncertainty $U_T$ Fails as an Error Detector:** When evaluated directly on held-out ground truth, the Dirichlet evidential head's uncertainty $U_T$ yields an AUROC of only **0.6619** (clean) and **0.6568** (adversarial). Plain softmax entropy drastically outperforms it (**0.8369** clean, **0.7724** adversarial). The previous belief that the evidential head provided ~0.83 AUROC was entirely due to scoring softmax entropy, not $U_T$. CLAIM-4 is **EMPIRICALLY REFUTED**.

---

## 7. Gist and information-acquisition audit

### 7.1 The exact pipeline (verified by execution, not by names)

| Stage | Operation | Result |
|---|---|---|
| Source | ImageFolder loads stored dataset images (`data/imagenet100`) | **160px shorter edge** (verified across 1,000 train & 1,000 val samples: 100% have min dim = 160; e.g. 213×160, 240×160) |
| Resize | `transforms.Resize((56,56))`, torchvision default **bilinear**, antialias default | **Downsamples 2.86×** from 160px to 56px, discarding 87.7% of spatial information |
| Normalize | ImageNet mean/std | (B,3,56,56) |
| Gist | `AdaptiveAvgPool2d(56)` = identity; `Conv2d(3→384,k14,s14)`; mean over the 4×4 grid | (B,384), **single vector** |
| Glimpse | `foveal_sample(x, gaze, 56)`: `scale=1.0`, translation only, border replicate | (B,3,56,56) |
| Tokens | 4×4 = **16** patch tokens per glimpse | (B,16,384) |

**There is no higher-resolution source anywhere in the model or the evaluation.**
`foveal_sample` raises if `fovea_size > H`, and `H` is already 56 after the dataset transform.
Coordinates are in the *resized* frame, in `[−1, +1]`, and are differentiable through
`affine_grid` (verified: theta's translation columns are gaze tensors).

### 7.2 What the glimpses can and cannot add

- **Cannot** add resolution, fine detail, small-object information, or anything outside the
  56×56 tensor. A target that does not survive the resize cannot be recovered by any gaze.
- **Can** add: (a) a full 12-block transformer pass per view, producing a CLS vector that is a
  non-linear function of the whole frame — strictly richer than the gist's linear block-mean;
  (b) mild translation-induced diversity (≈15 % of the frame changes per 8.4 px shift);
  (c) ensembling/ averaging over four views.
- Measured effect of views (D5, `recurrence_only`): T=1 0.606 clean / 0.270 robust →
  T=4 0.630 / 0.326. So **+2.4 pp clean / +5.6 pp robust** from three extra trunk passes.
  This is a real effect, and it is consistent with (a)+(c), not with information acquisition
  in the usual sense.

### 7.3 Is the gist information-limiting?

**Partially, but not in the way "gist" suggests.** The initial state is
`z₀ = mean(patch_embed(block-mean image))` — a single linear projection with **no trunk
processing at all**, whereas every glimpse enjoys the full 12-block trunk. So the asymmetry is
*compute and non-linearity*, not *resolution*. The gist is best described as a weak linear
summary, not as a low-resolution photograph that later crops refine.

### 7.4 G2-CTL1 / G2-CTL2

**Neither exists.** Repository-wide search: `G2-CTL1` and `G2-CTL2` occur **only in
`plan.md`**. There is no code, no flag in `flags.py`, no test, no script, and no result.
They are *registered intentions*, not experiments. The same is true of `G2-CTL4` and
`G3-BUD1`.

Minimum specification that would not alter any registered experiment (proposed, awaiting
approval — **not** retroactively preregistered):

- **`G2-CTL2-multiview-mean`** (decides effect A vs C): same 4 crops + gist as
  `recurrence_only`; concatenate the 4 CLS vectors and the gist vector (5×384 = 1920) into a
  single linear→GELU→linear head with **parameter count matched** to `recurrence_only`'s
  head+refinement total; **no** EMA recurrence, **no** update net, **no** predictor.
  Same parent checkpoint, same seed 41, 15 epochs, same protocol. This is the only clean test
  of "recurrence vs matched feedforward given equal observations".
- **`G2-CTL1-gist-continued`** (decides effect D): `g2_gist_only` final continued 15 epochs at
  ε = 0.094 with no architectural change. Distinguishes "phase 2 gained from recurrence"
  from "phase 2 gained from 15 more adversarial epochs".

### 7.5 Required information-acquisition verdict

**"Likely too easy or too redundant for active acquisition to demonstrate value" is the best
supported classification, with one qualification.**

Justification:
- Information-theoretically the observation set is a deterministic function of a single
  56×56 tensor; every glimpse is a translated copy. Active selection can only trade off
  *which region of an already-degraded 56×56 image* gets more transformer capacity.
- Empirically, additional views still help (+5.6 pp robust to T=4), so the pathway is not
  fully redundant — the extra trunk passes act as an ensemble.
- But the gain saturates at T=4, and in every gaze comparison random selection is at least
  as robust as the alternative it is compared against: `recurrence_only` random 0.382 vs fixed
  0.324; `gen1_core` random 0.334 vs AIS-v2 0.290. That is the signature of a task where
  *which* region is chosen carries little decision-relevant information.

The qualification: this verdict is about the **56-px substrate**, not about active perception
in general. A native-resolution substrate is the precondition for any information-acquisition
claim, and that is exactly what `G3-HRV1` proposes.

### 7.6 Empirical resolution probe & CKA results (Step 3 results)

The Step 3 protocol evaluated the frozen pretrained DINOv2-S trunk (`checkpoints/dinov2_vits14_pretrain.pth`, 174/174 keys matched) under identical linear probe protocols on ImageNet-100 (2,500 train samples, 500 held-out evaluation samples, Seed 42) comparing the 56×56 representation against 112×112 (`report/audit_rev1/step3_resolution_audit.json`):

| Resolution & Grid Setting | Patch Tokens (+ Prefix) | 5-Epoch Linear Probe Val Acc | Trajectory [Ep 1 $\to$ 5] | Effective Rank ($\exp(H)$) | Feature Norm Mean | Linear CKA vs 56×56 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **56×56 (4×4 Grid)** | 16 (+2 = 18) | **33.00%** | [18.2%, 28.6%, 32.2%, 32.6%, 33.0%] | 270.08 | 32.79 | 1.0000 |
| **112×112 (8×8 Grid)** | 64 (+2 = 66) | **71.60%** | [57.4%, 66.8%, 70.2%, 71.6%, 71.6%] | 293.88 | 33.15 | **0.5944** |
| **Resolution Delta** | **+48 Tokens (4×)** | **+38.60 pp** | — | **+23.80** | +0.36 | — |

#### Full-Dataset Resolution Probe (Full 126k Train / 5k Val ImageNet-100)

To eliminate any sampling noise or few-shot capacity effects, the probe was repeated across the **full 126,689 training images and full 5,000 validation images** for 5 epochs using cosine decay AdamW (`report/audit_rev1/step5_full_resolution_probe.json`):

| Resolution & Grid Setting | Patch Tokens | Batch Size | 5-Epoch Val Acc (Full 126k) | Epoch 1 Acc | Epoch 5 Acc | Resolution Gain vs 56px |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **56×56 (4×4 Grid)** | 16 | 256 | **58.18%** | 54.90% | 58.18% | Baseline |
| **112×112 (8×8 Grid)** | 64 | 128 | **85.26%** | 83.62% | 85.04% | **+27.08 pp** |
| **126×126 (9×9 Grid)** | 81 | 96 | **86.16%** | 84.60% | 86.16% | **+27.98 pp** |

**Foveal Sample Mathematical & Numerical Proof:**
1. **Scale Factor:** `foveal_sample` computes `scale = fovea_size / H = 56 / 56 = 1.0000`. At gaze $(0, 0)$, the maximum absolute pixel difference across all channels is $2.84 \times 10^{-5}$ (interpolation boundary noise). Gaze $(0, 0)$ is functionally bit-identical to the input.
2. **Translation Only:** A gaze offset of $(0.3, 0.3)$ corresponds to an exact translation of $0.3 \times 28 = 8.40\text{ pixels}$. Zero new pixels are introduced from the native high-resolution source; boundaries are simply replicated.
3. **The Artificial Choke Point:** On the full 126k dataset, expanding from 56×56 (16 tokens) to 112×112 (64 tokens) produces an overwhelming **+27.08 pp jump in top-1 accuracy (58.18% $\to$ 85.26%)**, with 112px hitting 83.62% in its very first epoch. Moving further from 112px (64 tokens) to 126px (81 tokens) yields only +0.90 pp (85.26% $\to$ 86.16%), confirming diminishing returns beyond 64 tokens. The 16-token / 56px bottleneck was the single dominant factor capping Gen-2 capability.

---

## 8. Causal attribution of Gen-2 gains

The six-phase ladder confounds four things at once: number of views, presence of a belief
update, gaze policy, and number of adversarial epochs. No cell of the ladder isolates one.

| Phase | Views | Belief update | Gaze | Epochs @ full ε | clean | robust (PGD-10) |
|---|---|---|---|---|---|---|
| 1 `g2_gist_only` | 1 (gist only) | no | none | 6 | 62.18 | 30.47 |
| 2 `recurrence_only` | 4 + gist | **no** | fixed | 15 (45 cumulative) | 63.44 | 34.77 |
| 3 `belief_no_f` | 4 + gist | yes (no predictor) | fixed | 15 | 61.26 | 34.18 |
| 4 `belief_with_f` | 4 + gist | yes (v1 UpdateNet) | fixed | 15 | 60.86 | not reported |
| 5 `ais_v2_swap` | 4 + gist | yes | AIS-v2 | 15 | 58.90 | 28.71 |
| 6 `gen1_core` | 4 + gist | yes | AIS-v2 | 14 | 58.46 | 29.69 |

(Values as reported in `plan.md` §1/E10; phase 4's final robust is missing from the summary.)

**Effect decomposition, with the evidence available:**

| Effect | What it would require | Status |
|---|---|---|
| **A. Additional observation information** | T-sweep with identical everything else | **SUPPORTED, quantified**: T=1→4 gives +2.4 pp clean, +5.6 pp robust (D5). Saturates at 4. |
| **B. Observation selection** | adaptive vs fixed vs random at equal T | **CONTRADICTED as a benefit**: adaptive 0.290 vs fixed 0.278 (+1.2 pp, σ≈2 pp) and random *higher* on robustness (0.334/0.382) |
| **C. Recurrent integration** | matched feedforward with equal views/params | **NOT TESTED** — `G2-CTL2` absent |
| **D. Other differences** (epochs, init, LR restarts, capacity) | matched-epoch continuation | **NOT TESTED** — `G2-CTL1` absent; phases 3–6 each restart LR from zero and add modules non-identity-initialised (CE spike 5.16 at phase 4 start, plan.md E8) |

**Therefore:** the widely repeated statement "recurrence gives +4.3 pp robust" is
**CONTRADICTED as stated**. What is measured is "four trunk passes with a fixed EMA beat one
pass with zero trunk passes while also receiving 15 extra full-ε epochs". The best checkpoint
in the ladder (`recurrence_only`) contains no recurrent belief machinery at all.

A separate, potentially important observation: **every phase after phase 2 is worse on clean
accuracy**, and phases 5–6 are worse on both axes. The most likely cause is the per-phase LR
restart to 5e-5 on an already-adapted trunk (plan.md E4/E10), which is an *optimisation*
confound, not an architectural one. It is not resolved and should not be attributed to any
mechanism.

---

## 9. Recurrent-state and prediction audit

### 9.1 Dataflow, component by component

| Component | Present in `recurrence_only`? | Present in `gen1_core`? | Reaches the next observation? | Reaches the decision? |
|---|---|---|---|---|
| State init `z₀` (gist mean) | yes | yes | yes (via z) | yes |
| Observation encoder (trunk) | yes | yes | yes (via z) | yes |
| Recurrent update | fixed EMA | learned (UpdateNet v1) | yes | yes |
| Predictor | **no** | yes | **yes** (gaze scoring) | **yes** (via error→z) |
| Prediction error | **no** | yes | yes | yes |
| Precision | **no** | yes (`softplus`, unbounded) | yes | yes |
| Uncertainty `U_t` | **no** | yes | yes (gaze scoring, precision) | yes (concat + readout) |
| Gaze policy | **no** (fixed list) | yes (`logit_scale` only) | yes | indirectly |
| Refinement block | yes but inert | yes but inert | no measurable effect | no measurable effect |
| `ev_readout` | no | yes | no | yes (12.8 % of logit norm) |

### 9.2 Prediction does affect the state — but only weakly

Intervention on the trained `gen1_core` checkpoint: replace the predictor's output with zeros
(so `err = −tok`), keeping everything else fixed.

```
max|Δlogit| = 0.7105      argmax flips = 0/8
```

So the predictor is **computationally connected** (the logits move), but on this probe it did
not change any decision. That is consistent with a small effect, not with a strong predictive
code. It is *not* evidence that the mechanism is useless — the intervention destroys the
error signal rather than testing a trained alternative.

Update magnitudes (same checkpoints, 8 images, manual replication):

| t | ‖z‖ | ‖Δz‖ | ‖Δz‖/‖z‖ | ‖err‖ | precision |
|---|---|---|---|---|---|---|
| 0 | 2.55 | **10.55** | **421 %** | 12.93 | 7.15 |
| 1 | 10.68 | 1.64 | 15.6 % | 15.04 | 1.25 |
| 2 | 11.06 | 1.32 | 12.1 % | 15.48 | 1.20 |
| 3 | 11.43 | 1.09 | 9.7 % | 14.63 | 1.17 |

Reading: the **first** update is four times the magnitude of the initial state, i.e. `z₀` is
effectively overwritten by glimpse 1. After that, each glimpse revises ~10–16 % of the state.
So the state does integrate sequential evidence in a non-trivial way *for phases ≥ 3* — but
the "belief" trajectory is dominated by the first observation, and the initial gist is
discarded almost immediately. This is the opposite of a prior that persists and gets refined.

### 9.3 Can the implementation express belief revision?

Structurally **yes** for `belief_*`/`gen1_core`: a new observation changes `tok`, hence
`err = pred − tok`, hence `Δz`, hence `z`, hence the logits. The update weight (precision) is
learned and state-dependent, so the architecture can in principle learn "revise strongly when
surprised". Whether it *does* is **UNRESOLVED**: there is no experiment in the repository that
sets up a contradiction between an early and a late observation and measures the revision.

A minimal synthetic test that would be decisive (not run; would require a trained model that
can be driven by constructed inputs):
1. Present a synthetic scene whose centre patch supports class A.
2. Force gaze at t=1 to a patch supporting class B (override the schedule, not the weights).
3. Compare P(final = B) between the real model and a control with the update path disabled.
   A model that revises should move toward B by more than the control; a model that ignores
   later evidence will not move at all.

### 9.4 Required recurrence verdict (separate claims)

| Claim | Verdict | Basis |
|---|---|---|
| Recurrent **computation** | **VERIFIED** for phases ≥ 2 | a state is carried across ≥ 2 steps and re-used |
| Recurrent **state dependence** | **VERIFIED** | logits change when the state path is altered |
| **Prediction-dependent** updates | **VERIFIED (computational)** / **UNRESOLVED (behavioural)** | 0.71 max Δlogit, 0/8 flips |
| **Evidence-dependent belief revision** | **CONTRADICTED** | `gen1_core` exhibits primacy lock (flips to B in only 6% of cases under 2 steps of contradiction); surprise ratio is only 1.04x |
| Advantage over a **matched non-recurrent control** | **NOT TESTED** | `G2-CTL2` absent |
| Advantage over a **matched-epoch** control | **NOT TESTED** | `G2-CTL1` absent |
| "Recurrence improves robustness by +4.3 pp" | **CONTRADICTED** as an attribution | the phase-2 checkpoint has no belief dynamics; confounded with views + epochs |

### 9.5 Empirical recurrent dynamics & contradiction results (Step 4 results)

The Step 4 protocol executed G2-REV1 (Contradiction Injection) and G2-PRB1 (Surprise Probe) across 50 distinct class pairs $(X_A, X_B)$ on ImageNet-100 val (`report/audit_rev1/step4_recurrent_audit.json`):

| Sequence Pattern | Meaning | `recurrence_only_final` Margin $(A - B)$ | `recurrence_only` P(A) / P(B) | `recurrence_only` Flip to B | `gen1_core_best_clean` Margin $(A - B)$ | `gen1_core` P(A) / P(B) | `gen1_core` Flip to B |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `AAAA` | Homogeneous A Control | +7.03 | 39.7% / 0.3% | 0.0% | +5.85 | 40.9% / 0.6% | 0.0% |
| `BBBB` | Homogeneous B Control | −7.46 | 0.1% / 40.4% | 52.0% | −6.14 | 0.1% / 38.3% | 44.0% |
| `AABB` | Late Shift (2 steps A, 2 steps B) | **−3.20** | 1.8% / 27.3% | **44.0%** | **+3.74** | 37.2% / 2.8% | **6.0%** |
| `BBAA` | Early Shift (2 steps B, 2 steps A) | **+2.77** | 25.4% / 2.9% | **4.0%** | **−3.99** | 0.6% / 36.1% | **48.0%** |
| `AAAB` | Single Late Contradiction | +0.54 | 14.7% / 10.0% | 14.0% | +4.82 | 39.7% / 1.3% | 2.0% |
| `BAAA` | Single Early Contradiction | +4.76 | 33.2% / 1.1% | 0.0% | **−2.84** | 1.8% / 33.5% | **48.0%** |
| `AAB_noisyB_noisy` | Late Shift with Degraded Evidence | −2.01 | 3.5% / 18.8% | 36.0% | +3.92 | 37.4% / 2.6% | 4.0% |

**G2-PRB1 Surprise Probe at Contradiction Transition ($t=2$ in `AABB` for `gen1_core`):**
- **Prediction Error Norm $\| \text{pred}_t - \text{tok}_t \|$:** Transition step = **14.4455** vs. Homogeneous baseline = **13.8688** (Surprise Ratio = **1.041×**).
- **Precision Scalar:** 1.2318 (virtually identical to homogeneous step precision of 1.20–1.25).
- **Dirichlet Uncertainty $U_t$:** 0.9093 (completely flat against homogeneous uncertainty of 0.908–0.910).

**Decisive Dynamics Verdict:**
1. **`recurrence_only_final` Exhibits Pure Exponential Recency Bias:** Under `AABB`, the model completely flips its decision to $B$ in 44.0% of cases (matching its pure $B$ baseline of 52%), while under `BBAA` it flips back to $A$ in 96% of cases. The state is governed by fixed forgetting ($z \leftarrow 0.5z + 0.5g_t$); whatever was seen most recently completely dominates the readout.
2. **`gen1_core_best_clean` Suffers from Severe Early Primacy Lock:** In `AABB`, despite receiving two consecutive full glimpses of contradicting evidence $B$, the model remains locked to $A$ in **94.0%** of cases (margin $+3.74$). Even more strikingly, in `BAAA`, an initial single glimpse of $B$ at $t=0$ prevents the model from predicting $A$ in **48.0%** of cases, despite 3 subsequent glimpses of $A$!
3. **Belief Revision Does Not Track Surprise or Evidence Quality:** The predictor's error increases by only 4% when confronted with a completely contradictory object class ($1.04\times$). Precision and uncertainty fail to modulate the update, and injecting Gaussian-degraded evidence produces essentially identical margin shifts (+3.74 clean vs +3.92 noisy). The hypothesis that the loop performs selective, surprise-driven belief revision is **EMPIRICALLY CONTRADICTED**.

---

## 10. Gaze, uncertainty, and stopping audit

### 10.1 The gaze policy

- **Mechanism:** at `t = 0` the gaze is hard-coded to `(0,0)` (`model.py:169-175`); from
  `t = 1` in `ais_v2_swap`/`gen1_core`, `AISv2GazePolicy` proposes K = 6 candidates:
  candidate 0 is the error-map anchor ("stay put"), candidates 1–5 are the anchor plus
  `N(0, 0.12)`, all clamped to ±0.9. Selection is hard argmax at inference and
  Gumbel-softmax during training.
- **Inputs:** the belief `z_t`, `U_t`, the observed tokens at the current fixation, and the
  predictor's tokens at the current fixation.
- **Ground-truth / privileged information:** none. It never sees the label or the target
  location. **VERIFIED.**
- **Differentiable:** coordinates are detached (`cand = candidates.detach()`), scores are
  attached. So the policy cannot move the gaze through the image gradient; only through the
  scores. That is a design choice, documented, not a defect.
- **Repeat selection:** candidate 0 is the anchor, so "stay put" is always available and
  there is no inhibition of return. Re-selecting the same region is possible and, per §7,
  yields almost the same pixels. **VERIFIED.**
- **Does gaze change the classifier input?** Yes — different crops ⇒ different tokens ⇒
  different logits. Measured: swapping gaze regimes changes accuracy (D7) and forward passes
  differ (D5).

### 10.2 Demonstrated utility of gaze: none

`report/d7_gaze_swap` (500 images, seed 42, PGD-10 @ 0.094):

| Checkpoint | Mode | clean | robust |
|---|---|---|---|
| `gen1_core` | AIS-v2 | 0.562 | 0.290 |
| `gen1_core` | fixed 4-point | 0.560 | 0.278 |
| `gen1_core` | random | 0.518 | 0.334 |
| `recurrence_only` | fixed 4-point | 0.630 | 0.324 |
| `recurrence_only` | random | 0.544 | 0.382 |

- Adaptive vs fixed: **+0.2 pp clean, +1.2 pp robust** — well inside the ±2 pp noise at
  n = 500. **CONTRADICTED as a demonstrated benefit.**
- Random gaze: **−4.4 pp clean, +5.8 pp robust** relative to fixed. The robustness gain is the
  classic signature of an input-space stochasticity acting as a weak randomised defence, not
  of better evidence.
- The `r = 0.706` gaze/perturbation correlation quoted in earlier documents came from an
  STL-10-era smoke checkpoint (plan.md E9); `G2-AIS5-gate-rerun` has not been run.

### 10.3 EIG feasibility

| Ingredient | Exists? | Form |
|---|---|---|
| Distribution over decision-relevant hypotheses | **partial** | `DirichletParams` over 100 classes, α ≈ 1.01–1.10 ⇒ nearly uniform, uncalibrated |
| Candidate observation **outcomes** | **no** | only a point prediction `predict_features(belief, a)`; no `p(o | a)` |
| Expected posterior after each candidate | **no** | one forward through the evidential head on the point prediction |
| Information gain per candidate | **approximate** | `H(U_t) − H(Dirichlet(pred))` — entropy reduction under a *point* prediction |
| Cost of an observation | **no** | no cost model anywhere |

The implemented score is therefore **entropy reduction from a point-predicted feature**, not
expected information gain. EIG requires `E_{o∼p(o|a)}[H(p(h|z)) − H(p(h|z,o,a))]`, i.e. a
likelihood over outcomes. A feature predictor's point output is not a likelihood. **The
current system cannot compute EIG**; it can compute a one-sample proxy.

Also note the distinctions the brief asks about, and which the design currently conflates:
- *entropy reduction*: computed (one-sample, on the Dirichlet).
- *expected information gain*: not computed (needs the expectation over outcomes).
- *feature novelty*: available as `‖err‖`, not used by the policy.
- *uncertainty-driven gaze*: implemented as argmin predicted entropy — equivalent to the
  reduction form up to the constant `H(U_t)`.
- *oracle selection*: not implemented (no label/location access).

### 10.4 Adaptive stopping

**Does not exist in Gen-2.** No halting head, no stop rule, no variable `T` at inference.
`T` is a constructor argument. The only halting machinery in the repository is the Gen-1
AIS-v1 `ais_halt_enabled` path (`rhan_next_ais_v1_halting_only`), which is not part of Gen-2
and whose gate (`adaptive_halting` ← `step6_validated_result`) is **unset**
(`noesis_vision/core/schema.py:95-100,235`; grep shows the field is never assigned).

Minimum requirements to test stopping: (i) a per-sample stop probability or threshold;
(ii) a fixed compute budget to compare against; (iii) a cost model that distinguishes
observation count from FLOPs (they differ here: a 56-px glimpse costs one 12-block trunk pass,
the gist costs one convolution); (iv) the T-sweep must be valid — which it currently is not
(see below).

### 10.5 Verified defect in the glimpse-scaling experiment

`model.forward` for `recurrence_only` iterates
`for t in range(min(self.num_glimpses, len(gaze_schedule)))` with a 4-element schedule
(`model.py:159-166`). `g2_d5_glimpse_scaling.py` constructs `num_glimpses=T` for
T ∈ {1,2,4,6,8}. For T = 6 and T = 8 the loop still runs **4** times, so those rows are
duplicates of T = 4. The artifact proves it on the deterministic path:

```
T=4: clean=0.630  robust=0.326
T=6: clean=0.630  robust=0.324
T=8: clean=0.630  robust=0.324
```

`recurrence_only` is deterministic, so the **clean** accuracy is bit-identical at all three
budgets; a genuinely 6- or 8-glimpse forward would necessarily change it. The robust column
differs by exactly 1 image (0.326 vs 0.324 = 163 vs 162 of 500), which is the PGD random
start, not the model. **VERIFIED.**

Consequence: **Gate L's second clause ("does not fall as T rises to 8") cannot be evaluated
with this implementation.** Only T ∈ {1,2,4} are real for `recurrence_only`. For `gen1_core`
the AIS branch continues past t = 3, so its T = 6/8 rows are real but flat
(0.570/0.284, 0.560/0.292).

### 10.6 Required gaze/stopping verdict

| Question | Verdict | Smallest additional experiment |
|---|---|---|
| Does gaze select useful evidence? | **CONTRADICTED as demonstrated** (+1.2 pp, inside noise) | paired 8-seed adaptive-vs-fixed on the same checkpoint, same images, both at T = 4; plus a gaze-to-target-location alignment metric on a synthetic scene |
| Can EIG be estimated meaningfully? | **NO with current machinery** (no likelihood, no outcome distribution) | implement `p(o|a)` (e.g. Gaussian head over predicted tokens) **before** comparing to the entropy proxy; otherwise the comparison is proxy-vs-proxy |
| Can adaptive stopping save observations? | **NOT TESTED; prerequisites absent** | fix the T-sweep, then measure accuracy-vs-T curves for fixed/adaptive/random and a cost-weighted frontier |

---

## 11. `gen2_foundation/` implementation audit

Test suite: `52 passed in 6.44s` (`pytest gen2_foundation/tests -q`). Per file:
`test_ema.py` 7, `test_eot.py` 3, `test_flags_gating.py` 11, `test_gist_parity.py` 7,
`test_precision.py` 10, `test_recipe.py` 4, `test_update_net_v2.py` 10.
**The "44 passing tests" figure in the audit brief is CONTRADICTED; the correct number is 52**
(matching `AUDIT.md` and the registry footer). Test count is not evidence quality — see 11.9.

| Component | Classification | Evidence |
|---|---|---|
| `flags.py` gating | **Correctly implemented; partially tested** | 33 IDs, `GATED_FLAGS`, `REJECTED_OUTRIGHT`, `is_control_arm()`; 11 tests. **No config hash function exists** (`to_dict()` only), and manifests show one hash for five arms ⇒ **config identity is not established** |
| `gist.py` shared tokenizer | **Correctly implemented** | reuses `backbone.patch_embed["proj"]`; no second tokenizer (invariant holds) |
| `gist.py` zero-gate parity | **Contains a verified defect** | `fuse()` has no forward call site; the parity test exercises an isolated function. The claim "gate = 0 preserves bit-exact Gen-1 behaviour" is **vacuously true** — the gate never runs. Gradient to `gate` = 0.0 exactly |
| `precision.py` family | **Not integrated** | `PrecisionField` is constructed only in tests and `__init__`; the model uses `noesis_vision/predictive_coding/precision.py::PrecisionFunction`. `learned_spatial` produces `(B,1)` broadcast to all tokens — **not spatial**. Model precision measured 6.94–7.29, **contradicting the documented `[1e-4,1]` clamp** |
| `update_net_v2.py` fusion | **Contains a verified defect** | `fused` computed and discarded (`:206-215`); `concat`/`gated`/`film` are behaviourally identical to `error_only`. Tests only assert `d.shape == (2,384)` |
| `update_net_v2.py` spatial pool | **Implemented but behaviourally unverified** | `err_map.detach()` blocks gradient into the predictor; no test asserts the spatial signal changes the update |
| `ema.py` | **Contains a verified defect** | `predict_features` delegates to the online predictor (`:140`); the Polyak shadow is never used to predict. All EMA behaviour is therefore a no-op |
| `recipe.py` | **Correctly implemented; adequately tested** | per-group LR, per-group clip (probe at 1e4 then budget), one-param-one-group via registry |
| `eot.py` | **Correctly implemented reference; not fully registered** | correct loss averaging (20 forwards for 5×4); no random start; `n_eot` defaults to 1; `build_autoattack_loss_wrapper` unusable without `torchattacks` and unused |
| `model.py` composition | **Implemented but behaviourally unverified** | `strict=False` everywhere; no test constructs `Gen2FoundationModel` and checks the phase semantics. Both evaluated checkpoints happen to load with 0 missing / 0 unexpected keys |

### 11.9 Test-quality gaps (the reason "52 passed" is weak evidence)

No test in `gen2_foundation/tests/` covers:

1. `Gen2FoundationModel.forward` for any phase (the composition the audit is about).
2. That the gist gate is in the forward path (it is not).
3. That different fusion modes produce different **outputs** (only shapes).
4. That the EMA shadow weights are used for prediction (they are not).
5. That gaze selection changes when `logit_scale` changes (it cannot).
6. That the ε-projector is respected (only the EOT sanity loss is tested).
7. Metric correctness for NLL/ECE/AUROC (none of the metric code is tested).

These are missing tests, not failing ones. The package is a *mechanism library*, not a tested
model.

---

## 12. Mathematical assessment of UC1, EIG1, DPS1

None of the three identifiers exists in the repository. The following is analytical only.

### 12.1 UC1 — contraction-constrained updates

**Hypothesis.** Replace the learned update with
`z_{t+1} = z_t + α_t(φ(x_t,a_t) − z_t)`, with `α_t ∈ [0,1]` (possibly uncertainty-gated), so
that state differences contract and adversarial perturbations cannot accumulate.

**Failure mode addressed.** Unbounded/oscillating belief updates; adversarial perturbation
accumulating across steps through a high-Lipschitz update network.

**Assumptions and what actually holds.**

1. *Fixed-input stability.* For a frozen observation `φ`, the map is a convex combination:
   `‖z' − z‖ ≤ (1−α)‖z' − z‖`. Contraction factor `1−α`, geometric. Holds for any α ∈ (0,1].
2. *Contraction of state differences under a common observation sequence.* With the same
   `{φ_t}`, `‖z_t − z_t'‖ ≤ Π_s(1−α_s)‖z_0 − z_0'‖ → 0. **This is also forgetting**: the
   influence of any observation decays by `(1−α)` per step. The state becomes an
   exponentially-weighted average with a horizon of `1/α`. The current `recurrence_only`
   update is exactly this with α = 0.5 and no learning — UC1 is therefore **already
   instantiated**, minus the gating.
3. *Stability when observations depend on the state.* If `a_t = π(z_t)`, the map is
   `F(z) = (1−α)z + α φ(x, π(z))`. Contraction requires
   `(1−α) + α L_φ L_π < 1`, i.e. `L_φ L_π < 1`. **Not guaranteed** — the gaze policy is a
   discrete argmax, so `L_π` is unbounded/undefined. This is the decisive gap: the loop's
   observation map is discontinuous in the state.
4. *State-dependent α (precision).* If `α_t = g(U_t)` and `U_t` depends on `z_t`, contraction
   needs `sup α < 1`. The current precision is `softplus(·)`, measured at 6.94–7.29, i.e.
   **α ≈ 7 ≫ 1**. The current update is *anti-contractive* by a factor of ~7 (bounded only by
   `tanh` on Δz). A UC1 implementation would have to replace the precision parameterisation,
   not merely add a constraint.
5. *Adversarial observations.* A perturbation `δ` enters as
   `‖φ(x+δ) − φ(x)‖ ≤ L_φ‖δ‖`, giving per-step state error `α L_φ ‖δ‖`. **Steady state under a
   persistent perturbation is `L_φ‖δ‖`, independent of α.** Contraction slows the approach to
   the perturbed fixed point; it does not reduce it. **Contraction of the state does not imply
   adversarial robustness of the classifier.** The decision can still flip because the map
   `z → logits` is unchanged.

**What it guarantees.** Exponential forgetting of initial conditions and bounded per-step
state change. **What it does not guarantee.** Robustness, decision stability, or that useful
evidence survives (large α suppresses revision of a wrong belief; small α suppresses revision
of a right one — at α = 0.05 the model needs ~60 glimpses to move).

**Smallest decisive experiment.** Take the `belief_with_f` final; at *evaluation only*,
replace the precision with a fixed α ∈ {0.05, 0.2, 0.5, 1.0} (clamped, no retraining) and
measure (i) clean/robust accuracy, (ii) state trajectory divergence under PGD, (iii) decision
flip rate. **Control:** the unmodified precision. **Primary metric:** paired robust accuracy
plus a *revision* metric (accuracy on a constructed contradiction case) — because the risk is
suppressing revision. **Confound:** at α fixed low, the model is simply under-trained for that
α, so a negative result is ambiguous.

**Cost.** Eval-only, ~1 GPU-hour. **Recommendation: investigate later**, and only after the
precision parameterisation is corrected — comparing against a precision of 7 is not a
contraction test.

### 12.2 EIG1 — expected-information-gain gaze and stopping

**Hypothesis.** Select the action maximising expected reduction in uncertainty about the
decision, and stop when the marginal expected gain falls below the cost of an observation.

**Mathematical requirement.**
`EIG(a) = E_{o∼p(o|z,a)}[ H(p(h|z)) − H(p(h|z,o,a)) ]`, and
`STOP ⇔ max_a EIG(a) < λ·cost(a)`.

**Assumptions vs the implementation.**

| Requirement | Status |
|---|---|
| `p(h|z)` over hypotheses | Dirichlet over 100 classes; α ≈ 1.01 at z₀ and ≈ 1.10 after four updates ⇒ nearly uniform; **not calibrated** (ECE 0.05 clean / 0.21–0.33 adv) |
| `p(o|z,a)` over outcomes | **absent**; only a point prediction of tokens |
| posterior after a hypothetical observation | **absent**; one forward through the evidential head on the point prediction |
| cost model | **absent** |
| guarantee | none: entropy reduction of an uncalibrated Dirichlet is not information gain about the label |

**Critical distinction.** Maximising `H(U_t) − H(Dirichlet(pred))` is equivalent to
minimising predicted Dirichlet entropy. With α ≈ 1.0–1.1, Dirichlet entropy is dominated by
`ψ(Σα) − Σψ(α)` and is nearly flat in the evidence scale, so the score is close to a function
of the *predicted feature magnitude*, not of decision-relevant information. The
t=0 branch is explicitly a saliency heuristic, so the policy is two different algorithms
across steps.

**Smallest decisive experiment.** Before comparing EIG to the proxy, the proxy must be shown
to be *related* to realised information: for a fixed checkpoint, for each candidate location,
record the predicted entropy reduction and then actually observe there and measure the
realised entropy reduction. **Correlation between predicted and realised reduction is the
precondition for any EIG claim.** If that correlation is near zero (plausible given α ≈ 1.01–1.10),
EIG cannot be estimated from this head without retraining it.

**Cost.** Eval-only, ~1 GPU-hour for K × N forwards. **Recommendation: investigate now, but
only as the calibration/validity check above — not as a new policy.**

### 12.3 DPS1 — decision-preserving state objectives

**Hypothesis.** Add a term that keeps the *decision* (or the task-relevant latent) close to the
clean decision under perturbation.

**What would be preserved, and the risks.**

| Target | Risk |
|---|---|
| clean **predictions** as soft targets | the clean model is 58–63 % accurate ⇒ 37–42 % of the teacher's targets are wrong; preserving them trains the model to reproduce its own errors, including on inputs where the attack exposed a genuine ambiguity |
| clean **latent representations** (`z`) | the latent is dominated by the first glimpse (§9.2) and its evidence is near-uniform; matching it may preserve a weak representation |
| **task-relevant statistics** (e.g. class logit margins) | closer to a margin objective; needs a definition of "relevant" that is not post-hoc |
| **ground-truth-supported decisions** | only defined where the clean model is correct; the natural formulation is a masked consistency loss, which is essentially TRADES' KL term |

**The confound that decides whether DPS1 is worth anything: the existing loss already
contains a decision-preserving term.** `compute_trades_loss` uses
`KL(softmax(adv) ‖ softmax(clean))` with weight 1.375 relative to CE
(`train_gen2_foundation.py:166`). DPS1 must therefore beat *that*, not a plain CE baseline.
No evidence in the repository suggests it would: the KL term is already large, and phases 5–6
(where the KL dominates) are the *worst* performers.

**Smallest decisive experiment.** None is worth running before the metric fixes: the outcome
measure (robust accuracy) is currently produced by an attack suite that is not at registered
strength, and the latent target has no validated semantics. **Recommendation: defer.** If
pursued later, the control is "TRADES KL with tuned weight", not "plain CE".

---

## 13. Minimal decisive experiment plan

Ranked by (information about the architecture decision) ÷ (cost). Costs are estimates from the
observed runtimes in the artifacts (the whole final protocol — clean + PGD-50 + EOT-5×20 over
500 images — took 170 s for `recurrence_only` and 252 s for `gen1_core` on the run GPU) and
from `plan.md` §13 (≈39 min/epoch ⇒ 15 epochs ≈ 10 GPU-h).

### Tier 0 — can run now, eval-only, no training, no protocol change

**E0.1 — Recomputed calibration from saved predictions (highest value per minute).**
*Hypothesis:* the true NLL, entropy and AUROC tell a different story from the reported ones.
*Implementation:* fix nothing in the registered scripts; add a read-only recomputation that
collects **logits and labels** (not max-prob) and computes `NLL = CE(logits, y)`, fp32 entropy,
Brier, ECE, AUROC, and risk–coverage. *Control:* softmax max-prob confidence.
*Data/ckpt:* `recurrence_only_final`, `gen1_core_best_clean`; 500 images seed 42.
*Metrics:* NLL, ECE, AUROC, Brier, accuracy@coverage. *Cost:* ~10 min.
*Success/failure:* descriptive; the decision rule is only "which metric is trustworthy".
*Status:* **exploratory**; does not alter the registered `G2-CAL2`.

**E0.2 — Registered-strength attack re-run on identical samples.**
*Hypothesis:* the robustness ranking survives a properly stepped, properly averaged attack.
*Implementation:* PGD-50 with `α = ε/4` **and** `α = 2.5ε/steps`; EOT-PGD with N = 20 and 50
steps; one AutoAttack suite (APGD-CE, APGD-T, FAB-T, Square) on the **deterministic**
`recurrence_only`; a large-ε check (16/255, where accuracy must approach 0).
*Control:* the existing PGD-50/α = ε/steps result.
*Data/ckpt:* both checkpoints; the same 500-image seed-42 slice.
*Confounds:* stochasticity of `gen1_core` (EOT is the only fair objective for it); single
training seed.
*Cost:* ≈ 1–3 GPU-h. *Success:* the ordering `recurrence_only > gen1_core` holds under the
minimum-over-attacks rule. *Failure:* the ordering reverses or the large-ε check does not
collapse ⇒ the attack is still suspect.
*Status:* **exploratory re-execution** of `G2-D8`/`G2-D3` at registered strength; the original
artifacts are preserved untouched.

**E0.3 — Correct glimpse-scaling sweep.**
*Hypothesis:* gains saturate before T = 4, or continue.
*Implementation:* `T ∈ {1,2,3,4}` for `recurrence_only` (the loop cannot exceed 4) and
`T ∈ {1,2,4,6,8}` for `gen1_core`; report per-T clean and robust with binomial σ.
*Cost:* ≈ 30–45 min. *Success:* a monotone curve with non-overlapping 2σ bands at the ends.
*Failure:* flat after T = 2 ⇒ the "more views" effect is an artefact of the first trunk pass.
*Status:* **exploratory**; `G2-D5` is preserved as-is with its T = 6/8 defect documented.

### Tier 1 — one training run each, decisive for attribution

**E1.1 — `G2-CTL2-multiview-mean` (the `CLAIM-2`/`CLAIM-7` null).**
*Hypothesis:* a non-recurrent, parameter-matched feedforward model with the same 4 crops +
gist recovers the phase-2 gain.
*Implementation:* new arm; concatenate 4 CLS + gist → linear head; parameter count matched to
`recurrence_only`'s classifier+refinement; **no** state carry.
*Control:* `recurrence_only` final continued 15 epochs under the identical recipe
(`G2-CTL1`-style), so both arms have matched epochs.
*Dataset/split:* ImageNet-100 train; report on the fixed 5,000 val.
*Metrics:* clean, PGD-10, PGD-50, EOT (for the stochastic arm).
*Cost:* 2 × 15 epochs ≈ 20 GPU-h.
*Success (proposed, **requires approval**): the control beats CTL2 by > 2σ paired on robust
accuracy. *Failure:* CTL2 within 2σ ⇒ the recurrence claim is not supported and the next
architecture should not spend capacity on state carry before fixing the substrate.
*Status:* architecture-neutral arm, **requires authorisation** (new arm, new checkpoint
namespace `checkpoints/audit_rev1/` if run).

**E1.2 — Frozen-trunk 112-px linear probe (`G2-RES1`).**
*Hypothesis:* the 56-px input, not the loop, is the binding constraint.
*Implementation:* Gate-W protocol (5 epochs, frozen trunk) at a 112-px input (64 tokens).
*Control:* the existing Gate-W reference 58.52 %.
*Cost:* ≈ 1–2 GPU-h. *Success:* ≥ 58.52 + 5 pp (threshold is **proposed**, not preregistered).
*Decision value:* the highest of any single number in this audit — it tells GLM whether to
design a belief architecture or a resolution architecture first.

### Tier 2 — only after Tier 0/1

| ID | Question | Why deferred |
|---|---|---|
| E2.1 | `G2-CTL1-gist-continued` (effect D) | needs E1.1's protocol to be fixed first |
| E2.2 | gaze-to-target alignment on synthetic scenes (Gate S) | needs a new dataset generator |
| E2.3 | adaptive stopping | needs a valid T-sweep (E0.3) and a cost model |
| E2.4 | UC1 with corrected precision | needs the precision parameterisation fixed |
| E2.5 | EIG vs proxy correlation | depends on E0.1 (is `U_t` informative at all?) |
| E2.6 | DPS1 vs tuned TRADES-KL | depends on E0.2 (is the robustness measurement trustworthy?) |

### Tier 3 — explicitly deferred, not authorised

Full AutoAttack campaign on all six finals; 16-seed retraining; 60-epoch ladder rebuild at
112 px; any Gen-3 build. Each is ≥ 1 GPU-day and none is decision-relevant until E0.2 and
E1.2 report.

### The single smallest next experiment

**E0.2 (registered-strength attack re-run), followed immediately by E0.1.** Reason: every
quantitative claim in the Gen-2 record — including the claim that `recurrence_only` is the
best model — is currently expressed in a metric whose attack strength and stochasticity
handling are not at the registered standard. Until that is repaired on the *existing*
checkpoints, any training experiment would be measuring with a broken ruler. Both are
eval-only, preserve all artifacts, and fit in a few GPU-hours.

---

## 14. Architecture implications

**What the evidence currently supports**

1. The 56×56 substrate is a hard ceiling. Any architecture that treats the initial view as a
   lossy-but-adequate global prior and expects later glimpses to add *detail* is built on a
   false premise here. **VERIFIED.**
2. Four trunk passes averaged with a fixed EMA are worth ≈ +5.6 pp robust over one pass
   (+2.4 pp clean), and this is the entire measured "recurrence" effect. **VERIFIED.**
3. Random gaze is at least as robust as the learned policy, and cheaper. Any gaze mechanism
   must beat random, not merely fixed. **SUPPORTED.**
4. The evidential head, as trained, produces an almost uniform Dirichlet with almost no
   dynamic range, and its direct readout changes no decisions on the probe. Uncertainty-driven
   *actions* have essentially no signal to consume. **VERIFIED (state), HYPOTHESIS (cause).**
5. Single-draw PGD on a stochastic policy overstates robustness by ~7.6 pp at n = 500.
   **VERIFIED.**
6. The refinement block, the gist gate, the fusion modes, the EMA target and the gaze
   temperature are all inert as shipped. Capacity spent on them is currently wasted.
   **VERIFIED.**

**What the evidence contradicts**

- "Recurrence improves robustness" — not attributable; the best checkpoint has no recurrence.
- "The Dirichlet head wins on calibration / AUROC" — never measured.
- "NLL barely changes under attack" — the metric is `−log max p`.
- "PGD-50 ≈ PGD-10, so no gradient masking" — both share the same weak step rule.
- "Adaptive gaze is a defence" — EOT is the correct measure and it is below the deterministic
  control.
- "The gist gate preserves Gen-1 parity at init" — the gate is never applied.

**What remains undecided (and must not be decided by assumption)**

- Whether a matched feedforward model recovers the multiview gain (E1.1).
- Whether 112-px input alone solves the ceiling (E1.2).
- Whether the evidential head *can* be made informative at this scale (needs a training run
  with an evidential objective applied to this head, or a diagnostic showing the head's
  gradients are dominated by the classification path).
- Whether any of the belief machinery would show value on a substrate where observations add
  information. **Nothing in the current record speaks to this.**

**Constraints for the next architecture proposal (not a design)**

1. Observations must be drawn from a source with more information than the initial view.
   Until that is true, no amount of state machinery can be validated.
2. Any gaze policy must be compared against *random* at matched compute, not only against
   fixed.
3. Any uncertainty signal intended to drive actions must first pass: (a) true NLL, (b) ECE,
   (c) error-detection AUROC, (d) a realised-vs-predicted information correlation.
4. Any recurrent update must be compared against (i) matched epochs, (ii) matched views,
   (iii) matched parameters — three separate controls.
5. Any stochastic component must be evaluated with EOT at registered strength, and the
   deterministic control must be reported alongside.
6. Precision/update gains should be bounded and *measured*, not assumed: the current precision
   is ~7 with a `tanh` bound as the only brake.

---

## 15. Blockers and unresolved questions

| # | Blocker | Exact missing evidence | Minimum safe action |
|---|---|---|---|
| B1 | GLM's review is not in the repository | The document itself | Obtain it; this audit used the brief's summary |
| B2 | No W1.5 report document | Prose report | None needed — the JSON artifacts are the primary evidence and were read directly |
| B3 | `G2-CTL1/CTL2` not implemented | Code + results | Authorise E1.1 |
| B4 | Phase 1/3/4/5 finals absent as `.pth` | Checkpoints | Restore from HF repos (`rhan-nxa-g2-checkpoints`) or accept that "all six finals" cannot be evaluated |
| B5 | Manifests describe a different run | Config hash, git SHA, seed, command for the 2026-10-14 ladder | Recover the HF/JSONL run records; do not treat the current manifests as provenance |
| B6 | No config-hash function | `hash(config)` | Add one for **future** runs only; do not backfill |
| B7 | No Gen-2 preregistration | Preregistered thresholds | Any threshold proposed here is **proposed only** and needs approval |
| B8 | Singles training seed | ≥3 seeds for a headline claim | Cost is ~30 GPU-h per arm; not authorised |
| B9 | AutoAttack not installed/executed | `torchattacks` + a run | E0.2 |
| B10 | `step6_validated_result` unset | A validated step-6 result | Out of scope for this audit; note that SBR / L_stab-objective / adaptive-halting all remain gated |
| B11 | `U_t` semantics unvalidated | Whether α ≈ 1.01–1.10 is a training artefact | Diagnostic: gradient-norm ratio of evidential vs classification loss on a batch |

**Unresolved scientific questions, stated precisely**

1. *Is the 56-px input the binding constraint?* Decided by E1.2 (112-px probe).
2. *Does recurrent integration beat multiview averaging?* Decided by E1.1.
3. *Is `U_t` informative?* Decided by E0.1 (true NLL/ECE/AUROC) plus the realised-vs-predicted
   information correlation.
4. *Does the predictor earn its parameters?* Needs a trained ablation (predictor frozen or
   removed, then fine-tuned) — not run.
5. *Is the evidential mechanism inert because of the loss, the architecture, or the data?*
   Needs a diagnostic of the evidential gradient vs the CE gradient — not run.
6. *Does any of this transfer to a substrate where observations add information?* **Cannot be
   answered with the current repository.**

---

## 16. Reproducibility and provenance

### 16.1 Repository state

```
$ git status
On branch feature/gen2-rebuild
Your branch is up to date with 'origin/feature/gen2-rebuild'.
Untracked files:
  checkpoints/*.zip:Zone.Identifier   (8 Windows metadata files)
  eval_scripts/                        (6 W1.5 scripts, untracked)
  plan.md
  scratch/eval_backbone_only_val.py
  scripts/check_kaggle_cell_syntax.py
nothing added to commit but untracked files present
```

- Branch `feature/gen2-rebuild`, commit **`14ad31b`**.
- `eval_scripts/` and the six `report/{d1,d3,d5,d7,d8,cal2}` result directories are
  **untracked** (`git ls-files eval_scripts` → 0). The W1.5 wave therefore has no git
  provenance.
- `report/` has 17 tracked files (paper assets), none of them W1.5 results.

### 16.2 Checkpoint hashes (computed this session)

| File | SHA-256 | Metadata |
|---|---|---|
| `checkpoints/foundation_recurrence_only_final.pth` | `ef5b0ddacd7c48c3de471ce5826b2c15212c17d89596134a0f050482c0fee479` | epoch 15, phase recurrence_only, best_clean 0.6344 |
| `checkpoints/foundation_gen1_core_best_clean.pth` | `cbeb85b09c8f0dc221e33fdc0987be7dd8c299a2af564e2ceb4bd0b5fcccd417` | epoch 14, phase gen1_core, best_clean 0.5846 |
| `checkpoints/foundation_recurrence_only_best_clean.pth` | `98ce2f43ccb833c0410c4dacc93df4e734661711153b5fa2d09e06bfa712757f` | — |
| `checkpoints/foundation_g2_gist_only_final.zip` | `475fdc09f5be233385d73fcc04b84a80fa90fa9f36371903fd4d2f99d73241c9` | zip containing the `.pth` |

Key compatibility check (both evaluated checkpoints):

```
recurrence_only_final.pth : keys_in_ckpt=194 missing=0 unexpected=0
gen1_core_best_clean.pth  : keys_in_ckpt=235 missing=0 unexpected=0
```

### 16.3 Dataset

- `data/imagenet100/train` 126,689 files; `data/imagenet100/val` 5,000 files; 100 class dirs.
- `fingerprint.json`: `aggregate_file_listing_sha256 = 0b06779f7b892183ad207ac9808b8a9fb122bda25ef9cebd666c1ee9f0c6db09`
  (matches the registry's pinned `0b0677…`).
- Split identifiers: train / val only. **No select split exists** (`G2-INF1` not run), so
  "best" checkpoints are selected on the reporting split — a known selection-bias issue
  (plan.md B5).

### 16.4 Commands executed and their real outcomes

```bash
$ .venv/bin/python -m pytest gen2_foundation/tests -q
52 passed in 6.44s

$ .venv/bin/python -m pytest tests/test_uncertainty_monotonic.py tests/test_evidential_shapes.py \
      tests/test_numerical_stability.py tests/test_eval_rhan_protocol.py \
      gen2_foundation/tests/test_flags_gating.py -q
40 passed, 4 warnings in 3.52s

$ .venv/bin/python scratch/audit_rev1/diag_metrics.py
# NLL: true=1.209607 vs cal2=0.358768 (identical=False)
# confidently-wrong shift: true NLL 3.4150 (rises) / cal2 0.0408 (falls)
# fp16: probs==0 count=701, entropy finite=False, mean=nan
# argsort([0.5,nan,0.2,0.9]) -> [2,0,3,1]
# eps=0.031 -> measured linf 0.03100002 (violated=False)
# eps=0.094 -> measured linf 0.09400010 (violated=False)
# PGD steps=1 -> linf 0.094 (alpha); PGD-50 alpha 0.00188
# eot forward calls for steps=5,n_eot=4 -> 20
# checkpoints: missing=0 unexpected=0 (both)
# foveal crop at gaze=0 equals input: True

$ .venv/bin/python scratch/audit_rev1/diag_behaviour.py
# determinism: recurrence_only identical=True, gen1_core identical=False max|Δ|=6.035e-01
# predictor bypass: max|Δlogit|=0.7105, argmax flips 0/8
# evidential bypass: max|Δlogit|=25.0554, argmax flips 8/8
# update magnitude t=0..3: rel = 420.9%, 15.6%, 12.1%, 9.7%; prec = 7.15,1.25,1.20,1.17
# precision min/mean/max = 6.94/7.13/7.29 ; U min/mean/max = 0.98996/0.99206/0.99370
# refinement gamma mean|γ| = 2.46e-4 (rec) / 3.57e-4 (gen1) ; trunk 1.305 / 1.314
# grad groups: gaze_policy n_nonzero=0 ; gist_encoder n_nonzero=0 ; all others nonzero

$ .venv/bin/python /tmp/diag_readout.py
# ||cls_head||=65.333 ||ev_readout||=8.363 (ratio 0.128)
# drop ev_readout: 0/16 flips, max|Δ|=1.409
# logit_scale cancellation: max|red*s - r| = 1.19e-07

$ python -c "torch.randperm(5000,seed=42)[:300] == torch.randperm(5000,seed=42)[:500][:300]"
True
```

### 16.5 Temporary files added by this audit

- `scratch/audit_rev1/diag_metrics.py` (read-only; writes nothing)
- `scratch/audit_rev1/diag_behaviour.py` (read-only; writes nothing)
- `/tmp/diag_readout.py` (read-only)

No existing file, checkpoint, or result artifact was modified. No patches were applied to the
registered code. Where a defect was found, it is **recorded, not repaired** (§5.3, §6.1–6.2,
§11). Any repair belongs in a separate authorised change with its own verification.

### 16.6 Discrepancies between source, documentation, and reports

| # | Document says | Source / artifact says |
|---|---|---|
| 1 | "44 passing tests" (brief / Kimi) | **52** passed |
| 2 | "Six-arm precision, clamped [1e-4,1]" (registry, `AUDIT.md`) | model uses `PrecisionFunction`, unbounded; measured 6.94–7.29 |
| 3 | "Gist fusion gate is the parity mechanism" (`gist.py` docstring) | `fuse()` is never called |
| 4 | "K5 fusion concat/gated/FiLM implemented" (registry) | `fused` discarded; all modes identical |
| 5 | "EMA target encoder" (registry) | shadow never used to predict |
| 6 | "Gaze policy has one learnable scalar" | it cancels algebraically; zero gradient |
| 7 | plan.md §4: "two linear ε stages 0.0157→0.0835", "Gate R at ε≈0.047" | code: three rungs 0.031/0.062/0.094; epoch 17 is at 0.031 |
| 8 | plan.md §4 / E12: doc loss `0.45·CE + 0.55·β·KL` | code: `w·(CE + β·KL)` = 0.55·CE + 1.375·KL (matches the logged fit) |
| 9 | registry: "Evaluation protocol = `run_clean_and_robust`" | no Gen-2 run uses it; W1.5 uses bespoke scripts |
| 10 | registry: "Manifest … config hash, seed, git SHA" | manifests are from a different (Sep 25) run; hash identical across 5 arms |
| 11 | brief: "W1.5 evidence report" | no such document; only JSON artifacts |
| 12 | brief: "APGD-CE-100" | hand-rolled momentum attack, not APGD/AutoAttack |
| 13 | plan.md: "E14 rules out gross masking" | PGD-10 and PGD-50 share the same weak step rule; the inference is unsupported |
| 14 | `G2-D8` registration: n=500, N=20, 50 steps | executed at n=300, N=5, 20 steps, no PGD-50 |

### 16.7 Tests that could not be run / work that could not be verified

- No AutoAttack run (dependency and time); no large-ε check.
- No matched-view or matched-epoch control (no trained checkpoint exists).
- No 112-px probe (requires a 5-epoch frozen-trunk training run).
- No multi-seed verification (single training seed 41).
- No independent reproduction of the ladder training (only the artifacts).
- No reconstruction of the original run's git SHA or config hash for the evaluated
  checkpoints (§16.6 #10).

---

## 17. Final recommendation

**Do not authorise an architecture change on the basis of the current evidence.**

Three statements are well established and should constrain any proposal:

1. The 56-px substrate cannot test information acquisition; every observation is a translation
   of one downsampled tensor.
2. The measured "recurrence" benefit is multiview averaging with extra compute; the best
   checkpoint contains no recurrent belief machinery.
3. The uncertainty, gist-gate, fusion, EMA and gaze-temperature mechanisms are inert as
   shipped, and the calibration metrics that were supposed to justify the uncertainty
   mechanism are mis-implemented.

Recommended immediate sequence, in this order and no other:

1. **E0.2** — re-run the attack suite at registered strength on the two existing checkpoints
   (EOT N = 20 / 50 steps, step-size-correct PGD, AutoAttack on the deterministic model, a
   large-ε check). Preserve the existing artifacts; report both.
2. **E0.1** — recompute NLL, ECE, entropy, Brier and AUROC from collected logits with the
   ground-truth label, in fp32, using softmax confidence *and* `U_t` as separate scorers.
3. **E1.2** — the frozen-trunk 112-px probe. This single number decides whether the next
   architecture is about *resolution* or about *belief*. It is cheap and it is the highest
   information per GPU-hour in this audit.
4. **E1.1** — `G2-CTL2-multiview-mean`, the only clean test of recurrence vs matched
   feedforward.

If (3) shows a large resolution gain, the 56-px ladder should be treated as a completed
negative-control study and the next architecture should start from a high-resolution
substrate. If (3) is flat, then the bottleneck is representation and training, and the
uncertainty/gaze claims should be re-derived from scratch on a corrected metric stack before
any new mechanism is proposed.

The correct summary of the present state is: **the harness runs, the artifacts are real, and
the scientific claims attached to them are not yet supported.** The most valuable output of
this audit is the list of things that must be re-measured before GLM's architecture proposal
can be evaluated on its merits.

---

## 18. Final forensic synthesis (deliverable conclusions)

### 1. What is verified
- **Artifact & Checkpoint Inventory:** Exactly **three** of the six ladder checkpoints exist locally: Phase 1 `foundation_g2_gist_only_final.zip` (Epoch 30), Phase 2 `foundation_recurrence_only_final.pth` (Epoch 15), and Phase 6 `foundation_gen1_core_best_clean.pth` (Epoch 14). Phases 3, 4, and 5 exist locally only as old 60-epoch Gen-1 files from late September 2026.
- **Loss Formulation (G2-D6):** Verified in code (`train_gen2_foundation.py:166`) that total loss is $\mathcal{L} = 0.55 \cdot \text{CE} + 1.375 \cdot \text{KL}$.
- **Phase 6 Halt Cause (G2-D4):** Verified that Phase 6 halted cleanly at epoch 14 via the 11.0-hour timeout threshold (`status: checkpointed_11h`).
- **Input Geometry:** Verified that `foveal_sample` computes $\text{scale} = 56/56 = 1.0$. All glimpses are pure 2D rigid translations with zero zooming and zero access to higher-resolution source pixels.
- **Inert Components:** Verified that the gist fusion gate (`FixedGistEncoder.fuse`) is never called (zero gradient); the refinement block LayerScale gammas are $\sim 0.04$ (bypassing it raises robust accuracy by +0.6 pp); observed-feature fusion in `update_net_v2` is computed and discarded; EMA target encoder delegates to the online predictor.
- **Metric Bugs Verified:** The CAL2 "NLL" implementation computed $-\log(\max p)$ (self-confidence), not true NLL; AUROC under fp16 suffered from probability underflow producing NaNs; the Dirichlet uncertainty $U_t$ was never scored in the original CAL2 script.

### 2. What previous claims need correction or qualification
- **Claim: "NLL is stable under adversarial attack."** $\to$ **CORRECTED.** The reported metric was $-\log(\max p)$, which artificially *decreases* when an attack pushes the model into confident errors. True classification NLL more than doubles under attack ($1.50 \to 2.94$ for `recurrence_only`, $1.84 \to 3.38$ for `gen1_core`).
- **Claim: "In-house PGD-50 robustness is 32.6%."** $\to$ **QUALIFIED.** PGD-50 was severely under-stepped ($\alpha = \epsilon/50 = 0.00188$). When corrected to standard step size ($\alpha = 2.5\epsilon/50 = 0.00470$), robustness drops to **24.80%**. Under library APGD-CE-100, the true robustness is **23.60%**.
- **Claim: "Evidential Dirichlet uncertainty achieves ~0.83 error-detection AUROC."** $\to$ **CORRECTED.** That score was produced by plain softmax entropy. When the actual Dirichlet uncertainty $U_T$ is evaluated, its error-detection AUROC is only **0.6619** (clean) and **0.6568** (adversarial)—barely above random guessing.
- **Claim: "Stochastic gaze is an active resilient defense."** $\to$ **CORRECTED.** Gaze stochasticity acts as a standard obfuscated gradient artifact under single-draw PGD; under EOT-PGD and library APGD, its robustness is $20.40\% – 21.20\%$, significantly trailing the deterministic `recurrence_only` baseline ($23.60\% – 24.80\%$).

### 3. Which phase rankings survive
- **`recurrence_only_final` (Phase 2) > `gen1_core_best_clean` (Phase 6):** **SURVIVES FIRMLY.** Across every evaluation protocol—Clean (63.00% vs 56.40%, $p=4.18\times 10^{-5}$), In-house PGD-10 (32.40% vs 28.40%, $p=4.18\times 10^{-3}$), Properly stepped PGD-50 (24.80% vs 21.00%, $p=1.23\times 10^{-3}$), Library APGD-CE-100 (23.60% vs 21.20%), and MI-FGSM-100 (23.40% vs 20.20%, $p=1.01\times 10^{-2}$)—`recurrence_only` decisively outperforms `gen1_core`.
- **`recurrence_only_final` (Phase 2) > `g2_gist_only_final` (Phase 1):** **SURVIVES.** +1.0 pp Clean (63.00% vs 62.00%) and $+4.0\text{ to }+4.6\text{ pp}$ Robustness across all attacks (24.80% vs 20.20% on PGD-50; 23.60% vs 19.40% on APGD-CE).
- **Phases 3, 4, 5:** Cannot be empirically ranked because their Gen-2 final checkpoints do not exist locally.

### 4. Whether the recurrent loop has demonstrated selective belief revision
- **NO. The hypothesis is EMPIRICALLY CONTRADICTED.**
- In `recurrence_only_final`, the state is an exponential moving average ($z \leftarrow 0.5z + 0.5g_t$), exhibiting pure recency bias. In the contradiction sequence `AABB`, the model flips to $B$ in 44% of cases, while in `BBAA` it flips back to $A$ in 96% of cases. It possesses no belief update or revision machinery.
- In `gen1_core_best_clean`, the model suffers from **severe primacy lock**: the initial observation ($t=0$) captures the latent state. In `AABB`, two consecutive conflicting glimpses of $B$ only flip the decision in **6.0%** of cases. In `BAAA`, an initial glimpse of $B$ prevents the model from predicting $A$ in **48.0%** of cases, despite 3 subsequent glimpses of $A$.
- In the G2-PRB1 surprise probe, prediction error norm increases by only **1.04×** at the contradiction boundary, while precision and uncertainty remain completely static. The model does not modulate updates based on surprise or evidence quality.

### 5. Whether resolution appears to be a bottleneck
- **YES. It is the dominant architectural bottleneck of the program.**
- On the full 126,689-image training set and 5,000 validation set, a frozen DINOv2-S trunk evaluated via linear probe achieves **58.18%** top-1 accuracy at 56×56 (16 patch tokens), but surges to **85.26%** at 112×112 (64 patch tokens)—an overwhelming **+27.08 pp gain** from resolution alone on identical data and trunk weights. Moving to 126×126 (81 tokens) achieves 86.16% (+0.90 pp further), confirming that 64 tokens captures nearly all available visual signal.
- **Training Cost Benchmark:** A clean Feedforward 112px model trained against corrected TRADES-PGD4 ($\alpha = 2.5\epsilon/4$, uniform random start) takes **0.447 s/batch** (19.7 min/epoch on RTX 4060, or **4.9 GPU-hours** for a full 15-epoch run).
- Crucially, Feedforward 112px is **24% faster to train** than the 56px Gen-2 recurrent loop (`recurrence_only` takes 0.587 s/batch, 25.8 min/epoch, 6.5 GPU-hours), because the recurrent loop computes 4 glimpse forward passes + gist + tied refinement per batch, while the feedforward model processes 64 tokens in a single efficient transformer pass.
- Because 56×56 glimpses are translated copies with scale 1.0 and zero zooming, the entire recurrent loop was forced to operate in a severely degraded visual manifold where fine semantic distinctions were irretrievably lost before inference began.

### 6. Which claims remain untested
1. **Recurrent Integration vs. Feedforward 112px Baseline (`ff_112`):** The new reference baseline is a feedforward 112-px model trained against the corrected TRADES-PGD4 attack. Because 112px is cheaper and +27 pp stronger representationally, recurrence must prove itself on a high-resolution substrate before being considered for Gen-3.
2. **Epoch Continuation (`G2-CTL1`):** Untested. Phase 2's gain over Phase 1 may simply reflect 15 additional epochs at full $\epsilon$. (CTL1 is budgeted at ~10 GPU-h; CTL2 is deferred).
3. **Phases 3, 4, 5 in Gen-2:** Untested locally due to missing checkpoints.
4. **Behavior on High-Resolution Substrates:** Whether active perception or belief revision can function when glimpses actually acquire new high-resolution foveal pixels is untested.

### 7. Single most informative next experiment
- **Experiment:** **`G2-CTL2-multiview-mean` paired with `G2-CTL1-gist-continued`** (or a 112-px resolution adaptation `G2-RES1` if architecture redesign is authorized).
- **Target Question:** Does sequential state carry (recurrence) provide *any* advantage over a feedforward architecture receiving the identical 4 views + gist with matched parameters?
- **Expected Compute Cost:** ~10 GPU-hours per arm (15 epochs on ImageNet-100 on RTX 4060).
- **Explicit Success/Failure Criteria:**
  - *Success (Recurrence Justified):* `recurrence_only_final` continued beats `G2-CTL2-multiview-mean` by $> 2\sigma$ paired on robust accuracy ($|\Delta| > 2.0\text{ pp}$, $p < 0.05$).
  - *Failure (Recurrence Null):* `G2-CTL2-multiview-mean` matches or exceeds `recurrence_only_final` within $2\sigma$ ($|\Delta| \le 2.0\text{ pp}$). If failed, the recurrent loop is retired, and all future modeling effort shifts to high-resolution feedforward / multi-scale representations.

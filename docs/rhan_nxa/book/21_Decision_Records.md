# Chapter 21 — Formal Decision Records: DR-001 through DR-010

> *Level 2–3 reading. Institutional memory: why each non-obvious architectural choice was made, what alternatives were evaluated and rejected, and what pre-registered empirical conditions could overturn it.*

---

## 1. Overview and Record Schema

The architecture of RHAN-NXA is governed by formal Decision Records. Each record preserves the historical rationale, mathematical reasoning, empirical evidence, rejected alternatives, remaining uncertainties, and the falsifying experiment for key architectural commitments.

Every record conforms to a strict 8-point schema:

```text
Decision:                         The core architectural or procedural choice.
Current choice:                   The active implementation setting in Generation-1.
Status:                           LOCKED | REQUIRED | EXPERIMENTAL CANDIDATE | 
                                  DEFERRED | PENDING DECISION | REJECTED.
Why:                              The foundational theoretical and causal rationale.
Evidence:                         Empirical numbers, seed counts, statistical tests.
What was rejected:                Specific alternatives dismissed and why.
What remains uncertain:           Unresolved trade-offs and theoretical boundaries.
Experiment that could change it:  Pre-registered empirical trigger for revision.
```

---

## 2. DR-001 — Why $S_t = \text{None}$ Initially

```text
Decision:                         Exclude explicit structural state from the Gen-1 core.
Current choice:                   S_t = None through the first integrated system and its
                                  first ablation matrix.
Status:                           REQUIRED (as the default configuration).
Why:                              The burden of proof is on added structural machinery. 
                                  The clean recurrent predictive coding core must validate
                                  its baseline capability before carrying complex, high-risk
                                  object-centric modules.
Evidence:                         Doubled empirical evidence:
                                  1. sbr0 ablation: zeroing out the top slot improved clean
                                     accuracy (retained ratio = 1.0157 vs a 0.7 floor designed
                                     to catch slot dependency).
                                  2. Clean collapse: a -9 to -10 pp clean accuracy degradation
                                     reproducibly appeared whenever the Gen-0 slot implementation
                                     was active—both intentionally (E2, sbr2–4) and 
                                     accidentally (D2, D3).
What was rejected:                Carrying forward the 16-slot SBR "just in case" or assuming
                                  that passing a low-threshold gate implied genuine object grouping.
What remains uncertain:           Whether any slot-based mechanism can pass a probe that detects
                                  true slot distinguishability.
Experiment that could change it:  Post-ImageNet-100 validation: a compact 2–4 slot variant evaluated
                                  with per-slot decodability distinguishable from each other and
                                  from chance under tightened Gate 1 conditions.
```

---

## 3. DR-002 — Why Pixel Reconstruction Was Rejected

```text
Decision:                         Do not use pixel-space reconstruction as E_t's prediction target.
Current choice:                   Rejected for Generation-1.
Status:                           REJECTED (for Gen-1).
Why:                              Two independent arguments:
                                  1. Mechanistic: pixel reconstruction penalizes high-frequency 
                                     textures, diluting precision Π_t and increasing belief drift
                                     (observed across Gen-0 Lens analysis).
                                  2. Engineering: pixel generation requires a heavy spatial decoder
                                     absent from the compact ViT substrate, inflating parameter 
                                     budgets and complicating gradient isolation.
Evidence:                         Gen-0 Lens analysis demonstrated increased drift. Experiment E1 
                                  measured a -0.90 pp clean drop vs Configuration D (statistically 
                                  inconclusive, but mechanistically disfavored and computationally 
                                  burdensome).
What was rejected:                Claiming "pixel reconstruction was proven catastrophically harmful" 
                                  (the effect was statistically marginal, but engineering cost was 
                                  unjustified).
What remains uncertain:           Whether a decoder-free dense spatial target could provide value in a 
                                  future multi-task setting.
Experiment that could change it:  A visual substrate with native dense spatial representations evaluated
                                  with significant, Lens-verified stability gains under matched controls.
```

---

## 4. DR-003 — Why Latent Glimpse Prediction Was Selected

```text
Decision:                         E_t is computed as the mismatch between predicted and observed 
                                  patch-level latent features at the next glimpse location.
Current choice:                   Locked as the Gen-1 design default.
Status:                           EXPERIMENTAL CANDIDATE (Design LOCKED, empirical value under 
                                  evaluation due to Gen-0 D3 confounding).
Why:                              1. Representationally symmetric by construction: predicted features 
                                     and observed features share the identical ViT patch-embedder space.
                                  2. Zero auxiliary decoder: the patch embedding output serves as both
                                     target format and ground truth.
                                  3. Unification with active sensing: a single predictor serves both 
                                     belief updating (E_t) and AIS-v2 gaze candidate scoring.
What was rejected:                Global state prediction (predicting z_{t+1} directly from z_t, which
                                  is trivial and prone to identity shortcuts); naive additive fusion 
                                  (z_t + λ Π E_t), which failed due to dimensional and scale mismatch.
What remains uncertain:           Whether latent prediction beats the null baseline (F = identity, no E_t) 
                                  under strictly unconfounded training.
Experiment that could change it:  The pre-registered unconfounded rerun of Configuration D3 (SBR disabled)
                                  benchmarked against Configuration D (null prediction).
```

---

## 5. DR-004 — Why $T = 4$ Glimpses

```text
Decision:                         Execute exactly T = 4 discrete glimpses per image forward pass.
Current choice:                   T = 4 for the initial build and validation matrix.
Status:                           REQUIRED for the first build (revisable via ablation).
Why:                              Reuses the validated setting from the project's empirical lineage
                                  rather than arbitrarily introducing another uncalibrated hyperparameter.
Evidence:                         Empirically verified in Gen-0 as balancing classification accuracy, 
                                  gaze trajectory convergence, and compute budgets.
What was rejected:                Deriving T dynamically at build time or using variable unrolled depths
                                  without empirical baselines.
What remains uncertain:           Whether T = 4 remains optimal when scaling from 96x96 inputs (STL-10) 
                                  to 224x224 inputs (ImageNet-100).
Experiment that could change it:  A rigorous T-sweep ablation (T ∈ {2, 4, 6, 8}) under parameter-matched
                                  and FLOP-matched controls.
```

---

## 6. DR-005 — Why Tied Within-Glimpse Recurrence

```text
Decision:                         Run a single shared Transformer encoder block iteratively 2–3 times 
                                  per glimpse before spatial pooling (Universal Transformer style).
Current choice:                   Weight-tied iterative token refinement.
Status:                           LOCKED as design (Option C — hybrid — selected narrowly).
Why:                              Option A alone (unrolled across glimpses with no internal refinement) 
                                  abandons the iterative token hypothesis. Option B alone (deep untied 
                                  stack per glimpse) balloons parameters and obscures whether gains stem 
                                  from depth or recurrence. Tied weights scale compute without adding 
                                  parameters, preserving the strict 20–25M budget.
Evidence:                         Theoretical capacity analysis and the SBR capacity-vs-mechanism risks.
What was rejected:                Untied stacked layers per glimpse; single-pass superficial token pooling.
What remains uncertain:           The exact performance delta between 2 versus 3 within-glimpse iterations.
Experiment that could change it:  Within-glimpse iteration count ablation (1 vs 2 vs 3 iterations) 
                                  evaluated under FLOP-matched baselines.
```

---

## 7. DR-006 — Why Adaptive Halting Is Deferred

```text
Decision:                         Enforce fixed glimpse depth T = 4; no learned or thresholded halting 
                                  in the initial Generation-1 build.
Current choice:                   Deferred.
Status:                           DEFERRED.
Why:                              Adaptive halting introduces an auxiliary halting policy and loss term 
                                  that has a documented history in this project of conflicting with primary 
                                  classification and stability objectives. The core loop must validate at 
                                  constant depth first.
Evidence:                         Gen-0 empirical record: in Model v10, a halting objective triggered 
                                  severe loss conflicts and numerical instabilities; AIS-v1 halting showed 
                                  marginal gains while complicating optimization.
What was rejected:                Simultaneous co-training of adaptive halting with the unvalidated core loop.
What remains uncertain:           Whether halting yields meaningful computational savings on natural images 
                                  without sacrificing worst-case adversarial robustness.
Experiment that could change it:  Reintroducing a halting head on a frozen, validated Step 6 core checkpoint 
                                  with strict halting-on vs halting-off controls.
```

---

## 8. DR-007 — Why AIS-v2 Uses the Shared Predictor

```text
Decision:                         Deploy a single GlimpseFeaturePredictor consumed jointly by the realized 
                                  glimpse prediction error (E_t) and the candidate gaze scoring policy.
Current choice:                   Shared predictor; gaze scoring via predicted uncertainty reduction from 
                                  Dirichlet evidential entropy.
Status:                           REQUIRED as a foundational design rule.
Why:                              Maintaining two separate predictors (one for state updates and one for gaze) 
                                  creates representation drift between where the model looks and how it updates. 
                                  A shared predictor enforces representational parsimony.
Evidence:                         Strong mechanism validation: Gen-0 Pearson correlation r = 0.706 between 
                                  predicted uncertainty reduction and actual error reduction.
What was rejected:                A dedicated, independent gaze-scoring network; predicting pooled global 
                                  vectors instead of localized patch tokens.
What remains uncertain:           Whether long-run training maintains alignment between realization updates 
                                  and planning scoring without loss interference.
Experiment that could change it:  Empirical divergence between update gradients and scoring gradients 
                                  under the shared predictor parameter group.
```

---

## 9. DR-008 — Why Uncertainty Derives from Dirichlet Evidence

```text
Decision:                         U_t is parameterized as class-conditioned Dirichlet evidence via the 
                                  EvidentialHead.
Current choice:                   Retained for Generation-1.
Status:                           REQUIRED for Gen-1 (with explicit acknowledgment of the scoping compromise: 
                                  readout-level, not belief-content-level).
Why:                              The EvidentialHead provides an established, closed-form epistemic uncertainty 
                                  metric ($K / \sum \alpha_k$) with analytical properties. Representation-level 
                                  uncertainty lacks a validated ground-truth target.
Evidence:                         Component validation across the Gen-0 lineage and literature (Sensoy et al.).
What was rejected:                Deferring all uncertainty representations (which would disable precision 
                                  weighting and AIS-v2 scoring); inventing untested self-supervised uncertainty 
                                  schemes during core integration.
What remains uncertain:           The theoretical gap between uncertainty about task labels and uncertainty 
                                  about visual latent features.
Experiment that could change it:  Validation of an unconfounded, self-supervised epistemic uncertainty head 
                                  trained directly on latent representation manifolds.
```

---

## 10. DR-009 — Why Representation-Level Uncertainty Is Deferred

```text
Decision:                         Do not attempt to construct direct uncertainty metrics over z_t or S_t 
                                  tensor contents in Generation-1.
Current choice:                   Pending decision; deferred past initial core validation.
Status:                           PENDING DECISION.
Why:                              Inventing complex representation-level density estimators or variance 
                                  heads mid-build repeats the Generation-0 failure pattern of stacking 
                                  unvalidated mechanisms onto an unverified base.
Evidence:                         The Master Implementation Plan's formal Tension Specification (Part 1.A).
What was rejected:                Equating class evidential uncertainty with visual feature entropy, or 
                                  claiming that U_t represents complete perceptual doubt.
What remains uncertain:           What supervision signal or generative target could calibrate latent-level 
                                  uncertainty without destabilizing ViT embeddings.
Experiment that could change it:  Same as DR-008.
```

---

## 11. DR-010 — Why the 16-Slot SBR Was Not Carried Forward

```text
Decision:                         Do not port the Generation-0 16-slot Spatial Bottleneck Recurrence (SBR) 
                                  into the RHAN-NXA architecture.
Current choice:                   Rejected as an implementation.
Status:                           REJECTED (Implementation only; the abstract concept of structural state 
                                  remains an EXPERIMENTAL CANDIDATE, DEFERRED).
Why:                              **Rejected Implementation $\neq$ Rejected Idea.** The 16-slot implementation 
                                  lacked evidence of functional specialization and imposed a severe, reproducible 
                                  performance degradation.
Evidence:                         1. sbr0 gate probes: individual slot classification accuracies ranged from 
                                     0.44 to 0.51 (statistically indistinguishable from random chance).
                                  2. Ablation paradox: zeroing out the dominant slot yielded a retained accuracy 
                                     of 1.0157, proving the network was not functionally dependent on the slots.
                                  3. Clean collapse: a persistent 9–10 percentage point penalty across E2, sbr2–4, 
                                     D2, and D3.
What was rejected:                Treating "gate passed" as evidence of genuine capability when the verification 
                                  gate was structurally incapable of detecting vacuous solutions.
What remains uncertain:           Whether any slot-based architectural mechanism can achieve functional 
                                  differentiation in compact visual backbones.
Experiment that could change it:  The DR-001 re-entry protocol: a clean 2–4 slot variant evaluated under 
                                  strictly hardened decodability gates.
```

---

## 12. Summary Matrix of Decision Records

| Record | Architectural Choice | Disposition | Status | Primary Rationale |
|---|---|---|---|---|
| **DR-001** | Initial Structural State $S_t = \text{None}$ | Locked Core | **REQUIRED** | Burden of proof on structure; avoid 16-slot collapse |
| **DR-002** | Rejection of Pixel Reconstruction | Dismissed Target | **REJECTED** | Mechanistically dilutes precision; requires heavy decoder |
| **DR-003** | Latent Glimpse Prediction Target | Core Target | **EXPERIMENTAL CANDIDATE** | Representationally symmetric; unifies prediction & gaze |
| **DR-004** | Fixed Glimpse Depth $T = 4$ | Recurrence Depth | **REQUIRED** | Validated project lineage; compute/accuracy balance |
| **DR-005** | Tied Within-Glimpse Recurrence | Block Iteration | **LOCKED** | Adds compute, not parameters; preserves 20–25M budget |
| **DR-006** | Adaptive Halting Deferral | Termination Policy | **DEFERRED** | Prevents objective conflict; isolates core loop validation |
| **DR-007** | Shared Predictor for Error & AIS-v2 | Gaze / Error Sync | **REQUIRED** | Eliminates dual-predictor drift; r = 0.706 mechanism link |
| **DR-008** | Dirichlet Evidential Uncertainty $U_t$ | Uncertainty Type | **REQUIRED** | Validated closed-form prior; avoids mid-build invention |
| **DR-009** | Deferral of Representation Uncertainty | Latent Entropy | **PENDING DECISION** | Lack of validated self-supervised training targets |
| **DR-010** | Rejection of 16-Slot SBR Code | Slot Attention | **REJECTED** | Chance-level slot decoding; reproducible -10 pp collapse |

---

## 13. Related Components and System Context

- [Chapter 06 — Structure State $S_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/06_Structure_State.md): Context for DR-001 and DR-010.
- [Chapter 08 — Prediction Error $E_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md): Context for DR-002 and DR-003.
- [Chapter 09 — Recurrence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/09_Recurrence.md): Context for DR-004, DR-005, and DR-006.
- [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md): Context for DR-007.
- [Chapter 19 — Generation-0 Evidence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/19_Gen0_Evidence.md): Deep-dive into the empirical numbers underlying these records.

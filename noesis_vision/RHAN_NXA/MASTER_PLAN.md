# RHAN-NXA — Generation-1 Master Implementation & Experiment Plan
## (Authorized documentation scope: Part 0 + Part 1.A–1.F)

> **Status of this file:** this is the source-of-truth capture of the
> RHAN-NXA plan as authorized for Gen-1 documentation. Sections 8+ of the
> master plan are **NOT** documented yet and must not be inferred.
> Every architecture chapter under `noesis_vision/RHAN_NXA/docs/` carries a
> `Source decision` pointer back to a section of this file.

---

# PART 0 — Gen-0 Confound Scope

*Verified against the provenance paths, not assumed.*

**Affected: ais_v2 (D2), hpc_belief (D3)** — both evaluated via
`sweep_rhan_nx_*` / the `_nx_trainer` runner, which hardcodes
`--enable-sbr` for every stage. These were supposed to be isolated
swap tests (AIS-v2 alone; belief-HPC alone). They are not. Both carry
legacy SBR (16 slots, 512-dim) unintentionally.

**NOT affected: D, E1, E2, E3** — evaluated via the older
`sweep_stage3_d_*` / `sweep_stage4_*` provenance path, which predates
this runner. E2's SBR is INTENTIONAL there (that is what E2 tests).
sbr2/sbr3/sbr4 also run through `_nx_trainer`, but SBR being active
there is correct by design — no confound.

**THE TELL:** D3's clean accuracy (45.06±2.64) and clean-delta-vs-D
(−9.90, REAL) are statistically indistinguishable from E2/SBR's
(45.06±3.30, −9.90, REAL) — the SAME two numbers, to two decimal
places, from two experiments that were supposed to be testing
completely different mechanisms. That is not coincidence. That is the
SBR-legacy signature dominating both results.

**CONSEQUENCE:** AIS-v2's and belief-HPC's TRUE isolated effect on
16-seed clean/robust accuracy is **UNKNOWN**. Not "weak" — unknown.
Everything else in the Gen-0 table stands as reported.

**ONE SALVAGEABLE RESULT:** AIS-v2's smoke-gate candidate-preference
correlation (**r = 0.706**, 512 samples, schema `ais_v2_smoke_gate_v1`)
is a narrower, more targeted diagnostic than the confounded 16-seed
sweep and is very likely NOT affected the same way — it measures
whether the candidate-scoring head's predictions correlate with policy
choice, a property largely orthogonal to whether SBR happens to be
active. This is graded **REQUIRED** evidence below. It is the single
most valuable new fact Gen 0 produced.

---

# PART 1 — Architecture (Sections 1.A–1.F only)

## 1.A — BeliefState `B_t = (z_t, S_t, U_t, E_t, A_t)`

- `z_t: (B, D_z)` — global content, `D_z` = backbone embed dim
  (locked once the substrate (Agent C) sets it). Dense, continuous,
  differentiable. Pooled CLS-token-equivalent output of the
  within-glimpse recurrent transformer (see 1.C).
- `S_t: Optional[(B, K, D_s)]`. **DEFAULT: None** for the core build
  (see 1.D). When present, an instance of the existing BeliefState
  ABC's structured variant — not a new type.
- `U_t` — Dirichlet evidence `e_t: (B, C)` [C = num classes],
  non-negative (softplus), `alpha_t = e_t + 1`. Uncertainty scalar =
  `C / sum(alpha_t)`.
  - TENSION (stated explicitly): `U_t` as specified is uncertainty over
    the **classification readout**, not over `B_t`'s content in
    general. This sits oddly against "classification is a readout, not
    the purpose" — a genuinely belief-centric uncertainty would measure
    confidence in `z_t`/`S_t` themselves. **Resolution: KEEP
    class-conditioned Dirichlet for Gen-1** (the EvidentialHead already
    exists, is validated, and a representation-level uncertainty has no
    existing target to train against without inventing new machinery).
    Document as a scoping compromise. Representation-level uncertainty
    = **PENDING DECISION**, deferred past Gen-1's first integrated build.
- `E_t` — resolved in 1.B; the answer to the prediction-error-space audit.
- `A_t` — `(gaze_history: list of (B,2) coordinates up to length T,
  current_glimpse_idx: int)`. Not a learned tensor itself — a
  structured record the gaze policy reads and writes.
- **None-propagation:** every consumer of BeliefState (`drift_to`,
  `as_tensor`, the classifier head, `F` itself) must have an explicit
  branch for `S_t is None`, tested directly.
- **Gradients through `B_t`:** `z_t` and `U_t` carry gradients always.
  `S_t` carries gradients only when not None. `E_t` is computed FROM a
  detached "observed" target and a non-detached "predicted" value —
  `E_t` itself must never be detached before use in the update (the
  single most repeated failure mode in Gen 0's history was exactly
  this class of bug). `A_t` carries no gradient (it is a coordinate
  record; differentiability lives in the policy that produced it).
- **Serialization:** `B_t` is NOT checkpointed as a persistent object
  across images — it exists only within one image's forward pass
  (T glimpses), then is discarded. What gets checkpointed is the MODEL
  (the parameters that produce `B_t`), never a specific `B_t` instance.
  There is no "B_t persistence across images" in Gen-1 at all.

## 1.B — Prediction Error `E_t` (the resolution of the Section-3 audit)

**REJECTED (for Gen-1): pixel-space reconstruction as `E_t`'s target.**
Two independent reasons, not one:
1. mechanistic — Gen 0's Lens analysis found reconstruction dilutes
   precision and increases belief drift, a real if not statistically
   dramatic cost;
2. engineering — pixel-space prediction requires a decoder that does
   not exist in the compact-ViT substrate and would need its own
   parameter budget and gradient-isolation treatment for no
   established benefit.

Note precisely, for the record: the E1 STATISTICAL result was
"−0.90pp vs D, NOT significant" — the mechanistic cost is real and
repeated, but calling it "definitively proven harmful" overstates what
the significance test showed. **Grade: mechanistically disfavored,
statistically inconclusive, engineering cost not justified.**

**REJECTED (for Gen-1, same reasoning): a separate edge-map/HPC-style
hand-designed feature target.** It requires its own extractor
(non-learnable Sobel-equivalent), doesn't share `z_t`'s space, and
belief-HPC's actual comparative evidence for it is confounded (Part 0).
No standing justification to build it fresh.

**LOCKED DEFAULT: `E_t` lives in `z_t`'s OWN latent feature space —**
specifically, patch/token-level features at the NEXT GLIMPSE's
fixation, predicted from the current belief. This is deliberately NOT
"predict the next global `z_t` from the current global `z_t`" (a
single pooled vector self-predicting across recurrent steps within one
static image has very little genuine signal). It IS "given `B_t` and a
candidate/chosen gaze location `a`, predict what the ENCODER will
produce when it actually looks there."

This single design choice does three things at once:
1. Makes the update equation mathematically valid BY CONSTRUCTION —
   predicted and observed local features are computed by the same
   encoder, guaranteed comparable space, no ad-hoc projection needed.
2. Requires no new decoder infrastructure — reuses the backbone's own
   patch-embedding output as both the prediction target's format and
   its ground truth.
3. UNIFIES with AIS-v2: the exact same predictor, run on K candidate
   locations before committing to one, IS AIS-v2's candidate-scoring
   mechanism (predicted error / expected uncertainty reduction per
   candidate). One predictor module, two consumers (the
   realized-glimpse error signal for F, and the candidate-scoring
   signal for gaze selection) — not two separate mechanisms that could
   drift out of sync.

**REVISED UPDATE EQUATION** (supersedes the earlier `z_t + λ·Π·E_t`
draft, which assumed E and z were trivially addable):

    z_{t+1} = z_t + Π_t · UpdateNet(z_t, E_t)

`UpdateNet`: a small learned MLP or single GRU cell, NOT a raw linear
addition — local glimpse-feature error and the global pooled vector
are not assumed compatible without a learned mapping between them.
This is a small, new, independently gradient-isolated component (own
optimizer group, own pre-flight |dW| check before any smoke test — the
standing Gen-0 rule, applied from day one).

**First-glimpse boundary condition (t = 0) — LOCKED:** the first
glimpse has no predecessor belief, so there is no predict step to
compare against. `E_0 := 0` (zero tensor, correct shape, no gradient
contribution) — the update at t = 0 reduces to
`z_1 = z_0 + Π_0 · UpdateNet(z_0, 0)`, i.e. pure observation with no
correction term. Prediction — and therefore `E_t` as a genuine signal
— begins at t = 1, once a real belief exists to generate a prediction
from. AIS-v2's candidate scoring at t = 0 falls back to the same
heuristic-saliency sampling used for candidate GENERATION (see 1.E
above) rather than uncertainty-reduction scoring, since there is no
`U_0`-conditioned prediction to score against yet. This is a LOCKED
decision, not PENDING: it is a boundary-condition fill-in required for
the equations above to be well-defined at all, not an open empirical
question.

**Interaction with `S_t`:** none in the core build (`S_t` is None). If
`S_t` is reintroduced later, its own update follows the same
predict/observe/error/precision/update shape independently — not
derived from `z_t`'s `E_t`.

**Status: EXPERIMENTAL CANDIDATE, not REQUIRED** — the best-reasoned
default given the constraints, not an empirically proven-superior
choice (D3's evidence for it is confounded). The experiment that
resolves it: a clean, SBR-disabled rerun, once the Gen-1 substrate
exists, comparing this latent-glimpse target against a null (no `E_t`
at all — F reduces to identity).

## 1.C — Recurrence

**LOCKED: Hybrid (Option C)** — but narrowly scoped: not chosen because
"more expressive." Chosen because Option A alone abandons the
belief-state thesis entirely (it becomes a plain recurrent ViT, useful
only as a CONTROL) and Option B alone likely under-represents each
glimpse (no iterative token refinement before pooling into `z_t`).

- **Within-glimpse:** a SHARED-WEIGHT (tied parameters) transformer
  block, run 2–3 times per glimpse before pooling — Universal
  Transformer-style. Tied weights mean iteration count changes
  compute, NOT parameter count, which keeps the compactness target
  simple to account for regardless of how many refinement steps are used.
- **Across-glimpse:** T fixed glimpses per image (**start at T=4**,
  reusing the STL-10 project's own validated convention rather than
  re-deriving a number from nothing), each producing one `B_t` via the
  predict→observe→error→precision→update→gaze cycle.
- **Adaptive halting: DEFERRED for the first build.** Fixed T=4. Gen 0's
  own halting history (v10's loss-conflict-with-Banach-proof incident,
  AIS-v1's modest halting effect) argues for validating the core loop
  at fixed depth before reintroducing a mechanism with a documented
  history of fighting other objectives.
- **Controls required for every recurrence ablation:** a
  parameter-matched control (same total params, recurrence
  disabled/T=1) AND a compute-matched control (same total FLOPs,
  achieved via width instead of depth) — both must exist before any
  "recurrence helps" claim is made, per the capacity-vs-mechanism
  warning Gen 0 already proved is a live risk (the SBR
  clean-gain-from-capacity pattern).

## 1.D — Structure `S_t`

**LOCKED: `S_t = None` for the core RHAN-NXA build** through the first
full integrated system and its first ablation matrix.

**Grounds (doubled evidence, not single-source):**
1. sbr0's own "everything-slot" ablation — zeroing the highest-norm
   slot IMPROVED accuracy (retained = 1.0157 against a floor meant to
   catch the opposite failure) — direct, gate-verified evidence the
   current slots carry no unique information;
2. the SBR clean-collapse signature (−9 to −10pp) now confirmed
   appearing whenever this specific slot implementation is active,
   deliberately (E2, sbr2–4) or accidentally (D2, D3 — Part 0).

This is REJECTED as currently implemented, not "object structure in
general." The concept of structural representation is graded
**EXPERIMENTAL CANDIDATE, DEFERRED** until after the None-`S_t` core
system validates on ImageNet-100. If revisited: start at 2–4 slots
(not 16), with per-slot decodability required to show slots are
STATISTICALLY DISTINGUISHABLE from each other and from chance BEFORE
any scale-up — sbr0's gate technically passed while measuring
near-uniform, near-chance per-slot content; that gate's floor needs
tightening before it's trusted again, not just re-run at larger K.

**Graceful None operation:** already specified in 1.A. This is not a
hypothetical — it is the actual default configuration for Gen-1's
first several phases.

## 1.E — AIS-v2

**LOCKED: one mechanism (no v3).**

- **Candidate generation:** K = 4–8 locations sampled around current
  highest-uncertainty regions of the belief (reuse whatever sampling
  heuristic the validated STL-10 AIS-v2 implementation used — do not
  redesign from scratch without cause).
- **Candidate scoring:** the SAME predictor from 1.B, evaluated at
  each candidate location, scored by predicted uncertainty reduction
  (via `U_t`'s Dirichlet entropy, not a separate scoring head — one
  uncertainty representation). At t = 0 there is no `U_0`-conditioned
  prediction to score against: candidate scoring falls back to the
  same heuristic-saliency sampling used for generation — LOCKED (see
  1.B's first-glimpse boundary condition).
- **Selection:** soft (differentiable, e.g. Gumbel-softmax or a
  straight-through estimator over the K candidates) during training
  for gradient flow; hard argmax at inference. Gaze history enters
  `A_t` as specified in 1.A.
- **Compute cost:** K forward passes of the (cheap) glimpse-feature
  predictor per step, NOT K forward passes of the full backbone —
  this must be enforced as an interface constraint, not left
  ambiguous, or candidate scoring becomes prohibitively expensive at
  ImageNet scale.
- **Center-bias prevention:** track and report the empirical
  distribution of chosen gaze locations across a validation batch; a
  policy that always centers is a **FAILURE CONDITION** for the gate,
  not a passing result with an asterisk.
- **Status:** mechanism DESIGN = **REQUIRED** (r = 0.706 is real,
  targeted evidence). Mechanism's 16-seed ACCURACY/ROBUSTNESS
  contribution = **PENDING DECISION** (Part 0's confound) — re-test
  cleanly once the Gen-1 substrate exists; do not assume the
  confounded +11.42 number as settled fact.

---

## 1.F — Uncertainty (Section 7) — locked, per Part 1.A's tension noted

One representation only: EvidentialHead's Dirichlet formulation,
**ported (not reimplemented)** from the STL-10 project. Feeds `Π_t`
(precision) in Part 1.B's update, feeds AIS-v2's candidate scoring,
feeds L_stab's drift metric. Calibration (ECE) evaluated in Agent I,
not trained against directly in Gen-1's first pass.

## 1.G — L_stab (Section 8)

**LOCKED: staged** —
- (A) **diagnostic only** through the entire core build and first
  ablation matrix;
- (B) promoted to a **training objective ONLY after** the core system
  (without L_stab) has a validated ImageNet-100 result to compare
  against.

**Mandatory responsiveness guard** (restated as a gate here): a model
minimizing adversarial belief drift by becoming insensitive to ALL new
evidence is a **FAILURE, not a success**. Gate 9 (Section 19) requires
L_stab's adversarial-drift reduction to be reported ALONGSIDE an
OOD/novel-evidence responsiveness score computed the same way, on a
disjoint probe set. A result showing low drift on BOTH is graded
**FAILED, full stop** — this is a hard, pre-registered stopping
condition, not a caveat added after the fact.

**Status: EXPERIMENTAL CANDIDATE, explicitly staged, not REQUIRED.**

## 1.H — Memory (Section 9)

- **Parametric memory** `θ` = model weights (trivial, always present).
- **Recurrent working state** `h_t` = the within-image glimpse
  trajectory (already covered by `B_t` across T steps — not a separate
  thing).
- **Episodic/temporal persistence ACROSS images** = DEFERRED, matches
  the existing "video is deferred pending a temporal experiment"
  framing — no work until a specific experiment shows temporal
  persistence is a limiting capability.
- **External retrieval/RAG = REJECTED outright.** No evidence anywhere
  in this project's history motivates it; it would be introduced
  purely because the word "memory" appears in the architecture name —
  exactly the trap to avoid.

## 1.I — V1 Frontend (Section 10)

**EXPERIMENTAL CANDIDATE.** Fixed (non-learnable), small parameter
cost, some added compute for the Gabor convolution. Built **LAST**,
after the core recurrent-belief loop validates without it — not
first — because it is a low-level, largely orthogonal intervention
whose effects (frequency/robustness properties) would otherwise
contaminate attribution for whether the belief-dynamics idea itself
works. **Standalone ablation required** before it is folded into any
"integrated" configuration.

---

# PART 2 — Revised Training Phase DAG

Two graphs, kept explicitly separate.

## Build dependency (what can be implemented/tested in parallel)

```text
Infra (Agent A)
  ├─→ Compact ViT + recurrence (Agent C)  [buildable independent of
  │     belief machinery — it's the substrate]
  ├─→ BeliefState scaffold + None-safety (Agent B)  [buildable
  │     independent of C — interface + composition logic, testable
  │     with dummy tensors]
  └─→ EvidentialHead port (Agent D)  [buildable independent of both —
        a self-contained head]

Once C + B + D exist:
  Unified predictor (glimpse-feature prediction + AIS-v2 candidate
  scoring, Part 1.B/E — Agents E+F coordinate tightly, see Part 4).

L_stab scaffolding (diagnostic-only): any time after B exists — it
only needs BeliefState.drift_to(), not a trained model.
```

## Scientific experiment dependency (EVALUATED in this order, or
## attribution is lost)

1. **Backbone-only baseline** (no recurrence, no belief) — parameter-
   and compute-matched controls established HERE, reused for every
   later comparison.
2. **+ Recurrence only** (T=4 glimpses, fixed, NO belief update —
   glimpses are independent, no F). Isolates: does recurrent
   multi-glimpse processing help AT ALL, before any belief machinery.
3. **+ BeliefState + U_t** (Dirichlet), NO F (identity update).
   Isolates: does an explicit typed belief + calibrated uncertainty
   help, independent of predictive dynamics.
4. **+ F / belief dynamics** (Part 1.B's update, NO AIS-v2 — gaze
   fixed/heuristic). Isolates: does predictive updating help
   independent of active gaze selection.
5. **+ AIS-v2** (full loop: predict → observe → error → precision →
   update → gaze-select). First end-to-end test of the active
   perception thesis.
6. **Integrated system**, still `S_t=None`, L_stab=diagnostic-only.
   **THIS is the Generation-1 core result** — the number this whole
   plan exists to produce.
7. **ONLY AFTER 6:** `S_t` reintroduction (minimal, 2–4 slots) as its
   OWN isolated arm vs step 6's result.
8. **ONLY AFTER 6 (not gated on 7):** L_stab promoted from diagnostic
   to training objective, its own isolated arm vs step 6, WITH the
   responsiveness guard as a hard gate.
9. **Full ablation matrix** (Part 3) once steps 1–6 exist cleanly.
10. **ImageNet-1K final run** — the single best-validated
    configuration from steps 1–9, run once, not per-ablation.
11. **Robustness/human-alignment evaluation suite** (Agent I) —
    Tier 1 (PGD/AutoAttack subset, ImageNet-C, shape/texture bias)
    required; Tier 2 (Brain-Score, new psychophysics) only if time
    remains within the 3-month window.

**Steps 7 and 8 are DELIBERATELY parallel-buildable (build dependency)
but NOT parallel-EVALUABLE against step 6 simultaneously in the same
run** — each needs its own clean comparison to step 6's frozen result,
never to each other directly, or you reproduce exactly the D2/D3
confound this plan opened with.

---

# PART 3 — Ablation Matrix

| Arm | vs. step 6 config, changes | Isolates | Seeds | Must stay identical |
|---|---|---|---|---|
| Backbone-only | no recurrence, no belief | Base capacity | 8+ | data, curriculum, warm-start |
| +Recurrence | T=4, no belief | Multi-glimpse value alone | 8+ | everything except recurrence flag |
| +Belief, no F | identity update | Explicit state + uncertainty alone | 8+ | everything except F flag |
| +F, no AIS-v2 | fixed/heuristic gaze | Predictive dynamics alone | 8+ | everything except F flag |
| Full (step 6) | AIS-v2 active | The complete Gen-1 thesis | 16 | — (this is the reference) |
| +S_t (2–4 slots) | vs step 6 | Minimal structure's marginal value | 8+ | everything except S_t flag |
| +L_stab | vs step 6 | Stability objective's marginal value + responsiveness guard | 8+ | everything except L_stab flag |
| +V1 frontend | vs step 6 | Frontend's marginal value | 8+ | everything except frontend flag |
| Param-matched control | backbone-only, width-matched to full | Rules out "more params" as full's explanation | 8+ | total param count |
| Compute-matched control | backbone-only, FLOPs-matched to full | Rules out "more compute" as full's explanation | 8+ | total FLOPs |

**Seed rationale:** 8 minimum given Gen 0's own finding that
TRADES-class baselines carry higher per-seed variance than mechanism
variants; 16 reserved for the single reference config (step 6) and
the ImageNet-1K final run; extend per-arm only if a result lands
borderline, per the established no-third-extension discipline.

---

# PART 4 — Multi-Agent Dependency Graph & Ownership

Two adjustments the audit requires:

**ADJUSTMENT 1:** Agent E (Predictive Dynamics) and Agent F (AIS-v2)
must share ONE interface contract from Agent 0, because Part 1.B
unified their predictor. They are NOT merged into one agent (F owns
candidate generation/selection/gaze-state bookkeeping; E owns the
predictor module itself and the belief-update network), but Agent 0's
spec must define the predictor's exact call signature ONCE, and both
E and F consume it identically. **Neither may implement its own copy.**

**ADJUSTMENT 2:** Agent G (Structural Representation) is EXPLICITLY
gated — it does not start until Agent J confirms step 6 (Part 2) has
a validated result. Agent 0's spec must record this gate as a literal
precondition Agent G checks before writing any code.

```text
           Agent 0 (Spec/Interfaces — runs first, always)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Agent A         Agent B         Agent D
        (Infra)      (BeliefState)   (Uncertainty)
              │             │             │
              └──────┬──────┴──────┬──────┘
                     ▼             ▼
                Agent C      [Agents E+F, sharing
              (Recurrent      Agent 0's predictor
               ViT core)      interface contract]
                     │             │
                     └──────┬──────┘
                            ▼
                    Agent J (Integration) — produces
                    the step-6 reference checkpoint
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Agent G          Agent H       Agent I
     (Structure —      (Objectives —  (Evaluation)
      GATED on J's      L_stab, staged
      step-6 result)    per Part 1.G)
```

Every agent's prompt uses the 17-point contract template, enumerated:

1. Mission
2. Scientific purpose
3. Inputs
4. Outputs
5. Files it may modify
6. Files it must not modify
7. Existing infrastructure to reuse
8. API/interface contract
9. Tensor shapes
10. Gradient requirements
11. Tests
12. Smoke experiment
13. Acceptance criteria
14. Failure conditions (stop and report — do not improvise)
15. Scientific interpretation
16. Commit/PR boundary
17. Handoff artifact

Plus: the explicit non-improvisation clause appears in every single
prompt, not just Agent 0's.

---

# PART 5 — Infrastructure Port Table (Section 15)

| Item | Disposition | Reason |
|---|---|---|
| Multi-group optimizer + registry | **PORT VERBATIM** | Validated across 3+ Gen-0 components; zero reason to touch |
| Gradient-isolation pre-flight (`|dW|` measurement) | **PORT VERBATIM** | (same validation) |
| Checkpoint/resume + best/rolling parity check | **PORT VERBATIM** | Learned the hard way once already |
| Comparator registry | **ADAPT** | Same concept, new dataset/checkpoint namespace |
| Structural-consistency assertion (Summary Table vs CSV) | **PORT VERBATIM** | Non-negotiable given its history |
| Attack evaluation (norm-space PGD/AutoAttack conventions) | **ADAPT** | Same conventions, new input resolution/normalization stats |
| EvidentialHead | **NEW** | First implementation in this project of a published formulation (Sensoy et al.), not ported from anywhere |
| AIS-v1's relocated-Eq.-II gaze code | **REJECT** | Superseded by AIS-v2's design; do not carry forward as a fallback path |
| Legacy Slot Attention (16-slot) module | **REJECT** | Doubled evidence against it (Part 1.D) |
| Stage-state machine | **ADAPT** | Same pattern, new phase list (Part 2) |

---

# PART 6 — Compute Accounting, Gates, Scope (Sections 18–21, condensed)

**Compute accounting:** every experiment in Part 2/3 reports param
count (total + trainable), FLOPs/image, latency, peak memory, glimpse
count, effective compute/image — logged automatically by Agent A's
infrastructure, not hand-computed per experiment.

**Gates:**

| Gate | Status |
|---|---|
| G6 — uncertainty correlates with error/ambiguity | **PENDING DECISION** — no threshold justified yet; report the correlation, don't gate on an arbitrary cutoff until one experiment exists to calibrate against |
| G7 — AIS-v2 avoids trivial center-bias | **REQUIRED** check, threshold **PENDING DECISION** (same reasoning) |
| G8 — S_t contains measurable structure | **N/A** while `S_t=None`; **PENDING DECISION** once Part 3's structure arm runs |
| G9 — L_stab reduces adversarial drift without collapsing OOD responsiveness | **REQUIRED as designed** (Part 1.G); the responsiveness-floor threshold is PENDING DECISION, calibrated from the diagnostic-only phase's own baseline distribution before L_stab ever becomes a loss |
| G10 — integration doesn't silently alter unrelated mechanisms | **REQUIRED** — enforced by the ablation matrix's "everything else stays identical" columns, not a separate check |

**Scope (Section 21):** 3D — DEFERRED, no 2D evidence yet suggests
depth is limiting. Video/temporal — DEFERRED, preserved conceptually
via the existing `frame_t → B_t → predict → frame_{t+1}` formulation,
revisited only if a small temporal probe shows occlusion/persistence
is a real gap. Relational graphs — DEFERRED, contingent on first
showing appearance+structure (once S_t is revisited at all) is
insufficient — do not build before that negative result exists.

**Literature classification (honest, unverified-citation-list
caveat):** ViT/DINOv2 warm-start = KNOWN PRIOR. Universal-Transformer-
style tied recurrence = KNOWN PRIOR. Predictive coding as inspiration,
not implementation = ENGINEERING ADAPTATION. JEPA-style latent-space
prediction = KNOWN PRIOR, adapted here to glimpse-level rather than
masked-region prediction = ENGINEERING ADAPTATION. Evidential deep
learning = KNOWN PRIOR. The unification of one predictor serving both
belief-update error AND active-gaze candidate scoring, inside an
explicitly typed, None-safe belief object = SCIENTIFIC COMBINATION at
best — novelty not verified and not claimed stronger without checking.

---

# AGENT 0 — Canonical Interface Specification Contract

**MISSION.** Own the canonical RHAN-NXA interface specification. No
model code, training loops, or loss functions. ONE artifact: a
versioned interface/schema document plus corresponding (empty-bodied,
type-annotated, docstring-only) Python interface stubs that every
other agent imports and implements against. Every ambiguity marked
PENDING DECISION or EXPERIMENTAL CANDIDATE becomes an explicit config
flag with a documented default and a documented "this is not yet
decided" comment — never silently resolved.

**SCIENTIFIC PURPOSE.** Generation 0 lost real time to interface
drift: multiple eval scripts disagreeing on epsilon convention, a
runner hardcoding a flag that silently confounded two "isolated"
experiments (Part 0). Agent 0 exists specifically to make that class
of failure structurally impossible for Generation 1 — one schema, one
source of truth, every downstream agent imports it rather than
reimplementing its own interpretation.

**INPUTS.** This master plan (Parts 0–6). Nothing else — no external
sources, no invented requirements.

**OUTPUTS.**
1. `noesis_vision/core/schema.py` — RHANNXAConfig (dataclass,
   versioned via `schema_version` field), covering every locked
   decision AND every PENDING DECISION / EXPERIMENTAL CANDIDATE flag
   from Part 1, each with its status as a code comment.
2. `noesis_vision/beliefs/interfaces.py` — BeliefState ABC (Part 1.A
   shapes exactly), stub-only.
3. `noesis_vision/predictive_coding/interfaces.py` — the SHARED
   predictor interface consumed by both Agent E and Agent F (Parts
   1.B/E, Adjustment 1) — the single most load-bearing artifact, since
   it prevents E and F from diverging.
4. `noesis_vision/core/dependency_graph.md` — the Part 2/4 graphs, in
   this repo, as the literal source both later agents and future
   sessions check before starting work.
5. `noesis_vision/core/agent_contract_template.md` — the 17-point
   template, placeholders filled, for Agents A–J to inherit.

**FILES MAY MODIFY.** Only the five files above, newly created.
Nothing in an existing directory (there is no existing `noesis_vision/`
at this stage — if one exists from an earlier, less-audited pass,
STOP and report the discrepancy rather than overwriting it).

**FILES MUST NOT MODIFY.** Anything under the STL-10 project's
`rhan_core/` — historical reference only, per the no-parallel-port
rule.

**EXISTING INFRASTRUCTURE TO REUSE.** None at this stage — Agent 0
creates the interfaces other agents will reuse infrastructure against.
Part 5's port table is informational for the spec (e.g., schema.py's
optimizer-group config shape should match what the ported multi-group
optimizer expects), not something Agent 0 implements.

**API/INTERFACE CONTRACT.** Exactly the shapes and signatures in Part
1.A (BeliefState), Part 1.B (predictor: takes current belief +
candidate location(s), returns predicted glimpse features + a
per-candidate scalar score), Part 1.C (recurrence config:
shared-weight flag, iteration count, glimpse count T). Do not add
fields not present in this plan without flagging them as a question
back rather than deciding silently.

**TENSOR SHAPES.** Part 1.A verbatim. Every stub signature must have
shapes in its docstring, not just types.

**GRADIENT REQUIREMENTS.** N/A directly (stub-only) — but every
interface must be DESIGNED so gradient flow is checkable by
construction (e.g., no interface that returns a detached tensor by
default from a method whose whole purpose is to carry gradient).

**TESTS.**
- `tests/test_schema_version.py`: config round-trips through
  serialization; every PENDING DECISION flag defaults to its
  documented safe value (False/None); attempting to set a
  PENDING-DECISION-gated flag True without its documented prerequisite
  (e.g., `enable_sbr=True` without step-6 validation recorded) raises,
  not silently succeeds.
- `tests/test_interface_imports.py`: every stub file imports cleanly;
  every ABC method raises NotImplementedError naming which future
  agent is responsible for it.

**SMOKE EXPERIMENT.** Import every file, instantiate RHANNXAConfig
with defaults, call every stub method once (all raise
NotImplementedError cleanly; none raise unrelated errors).

**ACCEPTANCE CRITERIA.** All five files exist. Both test files pass.
Every locked decision from Part 1 appears as a config field. Every
PENDING DECISION appears as a config field with its gating comment.
The dependency-graph markdown matches Part 2/4 exactly (no reordering,
no simplification).

**FAILURE CONDITIONS — STOP AND REPORT, DO NOT IMPROVISE.** If any
part of Part 1's audit is genuinely ambiguous about a shape or
signature not resolved above, STOP and list the specific ambiguity
rather than guessing. If an existing `noesis_vision/` directory
exists with different interface shapes, STOP and report the conflict —
do not silently reconcile it.

**SCIENTIFIC INTERPRETATION.** This artifact establishes NOTHING
empirically. Pure interface design. No experiment, no claim, no
number.

**COMMIT/PR BOUNDARY.** One commit: "Agent 0: RHAN-NXA canonical
interfaces and dependency graph." No implementation logic anywhere in
this commit.

**HANDOFF ARTIFACT.** The five files above, reviewed before Agent A
starts. Agent A (and every subsequent agent) imports from
`noesis_vision/core/schema.py` and the relevant `interfaces.py` —
never redefines its own version.

---

## What is NOT in this capture (do not document it)

- Agent prompts A–J (gated on review of Agent 0's handoff artifact;
  only Agent 0's prompt is authorized and captured above).
- Any architecture fact not stated above.
- Exact numerical gate thresholds (the plan itself marks them PENDING
  DECISION).
- Detailed ImageNet-1K training hyperparameters (only the step-10
  run-once rule is specified).

*Documentation pass boundary: Part 0, Part 1.A–1.I, Parts 2–6, and the
Agent 0 contract. Nothing beyond.*

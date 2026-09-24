# 18 — Glossary

*One-line definition first; technical definition where useful.
Terms defined elsewhere get a chapter pointer.*

---

**RHAN** — The recurrent, belief-driven visual-perception research
program this project belongs to. *(Predecessor generations: Gen 0.)*

**RHAN-NXA** — The Generation-1 architecture documented here: an
investigation-style perception system with an explicit belief state,
latent prediction error, and uncertainty-driven gaze.

**Gen-0** — The previous generation of experiments (STL-10 ladder:
gen0, sbr0–sbr4, ais_v2, hpc_belief); source of the verified numbers,
the SBR rejection, and the D2/D3 confound. → `16_Gen0_Evidence_And_Confounds.md`

**Gen-1** — The generation documented here: the RHAN-NXA core build
and its first ablations.

**Perception** — In this project: the *process* of building an
internal state sufficient to answer questions from — not the
classification output itself.

**Belief** — The model's current internal state about the image; a
computational variable, not a claim about mentality.

**Belief state (`B_t`)** — The typed five-part snapshot
`(z_t, S_t, U_t, E_t, A_t)` carried across one image's T glimpses. → `04_Belief_State.md`

**Glimpse** — One deliberate look: encoder observation at a chosen
gaze location, plus the belief update it triggers.

**Recurrence** — Re-applying computation across steps rather than
stacking new parameters. Two kinds here (below). → `09_Recurrence.md`

**Within-glimpse recurrence** — Re-running a single weight-tied
transformer block 2–3 times inside one glimpse to refine token
features before pooling.

**Across-glimpse recurrence** — The T = 4-glimpse loop itself: each
glimpse produces one `B_t` via the predict→observe→error→precision
→update→gaze cycle.

**Transformer** — The neural architecture built on attention between
tokens; the substrate's building block. (No chapter; background.)

**ViT (Vision Transformer)** — A transformer applied to image patches;
RHAN-NXA uses a compact one as its visual substrate.

**Latent representation** — A learned vector encoding of data inside
the network, as opposed to raw pixels or labels.

**Feature / token** — One per-patch (or per-candidate-location) vector
produced by the encoder; the space `E_t` lives in.

**`z_t`** — The belief state's global content vector `(B, D_z)`;
pooled from the within-glimpse recurrent transformer. → `05_z_State.md`

**`S_t`** — The belief state's optional structural component;
**None** in the Gen-1 core. → `06_Structure_State.md`

**`U_t`** — The belief state's uncertainty component; Dirichlet
evidence over the classification readout in Gen-1. → `07_Uncertainty.md`

**`E_t`** — The belief state's prediction-error component; mismatch
between predicted and observed next-glimpse latent features. → `08_Prediction_Error.md`

**`A_t`** — The belief state's gaze record: gaze history (list of
`(B,2)` coordinates up to length T) plus current glimpse index;
non-learned. → `04_Belief_State.md`

**Precision** — How much the current evidence should be trusted; the
`Π_t` factor scaling the belief update. (Relationship to `U_t`'s
Dirichlet machinery is a Gen-1 design matter; exact formulation
outside this documentation pass.)

**Uncertainty** — The model's estimate of its own confidence; here,
readout-level (see `U_t`). → `07_Uncertainty.md`

**Dirichlet evidence** — Evidential-deep-learning machinery: emit
non-negative evidence `e_t` per class; `alpha_t = e_t + 1` gives a
Dirichlet concentration; total evidence inversely tracks uncertainty
(`C / sum(alpha_t)`). → `07_Uncertainty.md`

**EvidentialHead** — The existing, validated module that produces the
Dirichlet evidence above; Gen-1's uncertainty source. → `07_Uncertainty.md`

**Prediction error** — The informative mismatch between what was
predicted and what was observed; `E_t` is its Gen-1 latent-form. → `08_Prediction_Error.md`

**UpdateNet** — The small learned MLP/GRU mapping `(z_t, E_t)` into
`z_t`'s update; explicitly *not* a raw addition; own optimizer group
and pre-flight `|dW|` check. → `08_Prediction_Error.md`, `13_Gradient_Flow.md`

**AIS (Active Information Selection)** — The active-perception
mechanism family in RHAN; chooses where to look.

**AIS-v2** — The Gen-1 gaze mechanism: K = 4–8 candidates around
high-uncertainty regions, scored by predicted uncertainty reduction
via the shared predictor and Dirichlet entropy; soft selection in
training, hard argmax at inference. → `10_AIS_v2.md`

**Gaze** — The chosen fixation location for a glimpse; recorded in
`A_t`.

**Candidate location** — One of the K positions AIS-v2 scores before
committing to the next gaze.

**Active perception** — Perception in which the observer *chooses*
what to observe next, rather than passively receiving fixed views.

**SBR** — "Slot-based structural representation," the Gen-0 16-slot
implementation; **REJECTED as implemented** (retained-ablation 1.0157;
clean-collapse signature), which is a statement about the
implementation, not about structure in general. → `06_Structure_State.md`

**Structural representation** — The general idea of explicit
object/relationship state in the belief; EXPERIMENTAL CANDIDATE,
DEFERRED, with pre-registered re-entry rules. → `06_Structure_State.md`

**Parameter-matched** — A control with the same total parameter count
as the tested model, with the mechanism disabled; separates mechanism
from capacity.

**Compute-matched** — A control with the same total FLOPs (e.g. via
width instead of depth); separates mechanism from raw compute.

**Ablation** — Removing/isolating one component to measure its causal
contribution under the controls above.

**Control** — The comparison run that makes a claim falsifiable;
Gen-1 requires parameter- and compute-matched controls for mechanism
claims. → `09_Recurrence.md`

**Gradient isolation** — Giving a component its own optimizer group
and verifying its gradient flow (`|dW|` pre-flight) so it cannot
silently starve. → `13_Gradient_Flow.md`

**Adversarial robustness** — Accuracy under deliberately perturbed
inputs; measured here as PGD-100/PGD-50 accuracy at eps = 0.094
(norm space), 16 seeds. → `16_Gen0_Evidence_And_Confounds.md`

---

### Generation-1 planning vocabulary

**`L_stab`** — The belief-stability objective: measures adversarial
belief drift via `BeliefState.drift_to()`; LOCKED to a staged protocol
(diagnostic-only → training objective, gated on a validated core).
→ `21_L_stab_Stability.md`

**Responsiveness guard (Gate 9's rule)** — The hard, pre-registered
rule that a model achieving low adversarial drift by becoming
insensitive to ALL new evidence is graded FAILED, not robust; drift
reduction must be reported alongside OOD/novel-evidence responsiveness
on a disjoint probe set. → `21_L_stab_Stability.md`

**Parametric memory (θ)** — The model's weights; the only cross-image
memory in Gen-1. → `22_Memory.md`

**Recurrent working state (h_t)** — The within-image belief trajectory
`B_1…B_T`; not a separate module — it *is* the belief state evolving.
→ `22_Memory.md`

**RAG (external retrieval)** — Fetching evidence from an external
store; **REJECTED outright** in Gen-1 (no project evidence motivates
it). → `22_Memory.md`

**V1 frontend** — A fixed, non-learnable Gabor-convolution input
stage; EXPERIMENTAL CANDIDATE, built LAST with a standalone-ablation
requirement. → `23_V1_Frontend.md`

**Gabor filter/convolution** — A fixed orientation/frequency-tuned
filter bank used as the frontend's first stage (structural analogy to
cortical V1, not a neural-fidelity claim). → `23_V1_Frontend.md`

**Training phase DAG** — The two-graph plan: build dependency
(parallel-friendly construction) vs scientific experiment dependency
(the 11-step evaluation ladder, steps 1–11). → `24_Training_Phase_DAG.md`

**Step-6 reference** — The integrated Gen-1 core result
(S_t=None, L_stab diagnostic-only, AIS-v2 active) that every ablation
arm compares against; frozen. → `24_Training_Phase_DAG.md`, `25_Ablation_Matrix.md`

**Ablation matrix** — The ten-arm comparison plan (backbone-only,
+recurrence, +belief-no-F, +F-no-AIS, Full, +S_t, +L_stab, +V1,
param-matched, compute-matched); each arm flips one flag vs step 6. → `25_Ablation_Matrix.md`

**Param-matched control** — Backbone-only run with total params matched
to the full model; rules out "more parameters" explanations. → `25_Ablation_Matrix.md`

**Compute-matched control** — Backbone-only run with FLOPs matched to
the full model; rules out "more compute" explanations. → `25_Ablation_Matrix.md`

**No-third-extension discipline** — Extend a per-arm seed count once
if borderline; never repeatedly. → `25_Ablation_Matrix.md`

**Agent 0** — The spec/interfaces agent that runs first, always;
produces the versioned schema, interface stubs, dependency graph, and
contract template; writes no implementation logic. → `30_Agent0_Interface_Contract.md`

**Agents A–J** — The build agents: Infra (A), BeliefState (B),
Recurrent ViT core (C), Uncertainty (D), Predictive dynamics (E),
AIS-v2 (F), Structure (G — gated on J), Objectives (H), Evaluation
(I), Integration (J). → `26_Agent_Organization.md`

**Shared predictor contract (Adjustment 1)** — Agent 0 defines the
predictor's call signature ONCE; Agents E and F consume it identically;
neither may implement its own copy. → `26_Agent_Organization.md`

**Agent G's gate (Adjustment 2)** — Structure work does not start
until Agent J confirms step 6's validated result; recorded as a
literal precondition, not a norm. → `26_Agent_Organization.md`

**17-point contract template** — The per-agent prompt template
(mission, scientific purpose, inputs/outputs, allowlist/denylist,
interface contract, shapes, gradients, tests, smoke, acceptance,
failure conditions, interpretation, commit boundary, handoff), used
verbatim with the non-improvisation clause in every prompt. → `26_Agent_Organization.md`

**PORT VERBATIM / ADAPT / REJECT** — The three infrastructure
dispositions: copy validated code as-is / rebind semantics to new
surfaces / refuse to carry forward (with reasons attached). → `27_Infrastructure_Port_Table.md`

**No-parallel-port rule** — Gen-1 builds against Gen-1 interfaces;
`rhan_core/` stays historical reference only; ports happen only
through the port table. → `27_Infrastructure_Port_Table.md`

**G6–G10** — The Gen-1 gates: G6 uncertainty-error correlation
(PENDING), G7 center-bias (REQUIRED, threshold PENDING), G8 structure
(N/A while S_t=None), G9 drift-without-collapse (REQUIRED as designed,
floor PENDING), G10 no-silent-alteration (REQUIRED, enforced by the
matrix's identical columns). → `28_Gates_and_Compute_Accounting.md`

**RHANNXAConfig** — The versioned config dataclass (schema_version
field) in `noesis_vision/core/schema.py`; every locked decision and
every PENDING-DECISION flag appears as a field with a status comment. → `30_Agent0_Interface_Contract.md`

**KNOWN PRIOR / ENGINEERING ADAPTATION / SCIENTIFIC COMBINATION** —
The literature labels: established idea used as-is / known prior
modified for this setting / combination possibly new (unverified;
claimed at no greater strength). → `29_Literature_Classification.md`

---

> **Source decision:** glossary assembled from Parts 1.A–1.F and the
> documentation task's mandated term list; no term is defined here in
> a way that contradicts its chapter.

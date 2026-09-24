# 17 — Common Misunderstandings

*Mandatory reading before repeating or implementing anything here.
Each entry: the wrong claim, the correction, and where to verify.*

---

## "RHAN just looks at the image four times."

**Not exactly.** Repeated forward passes over an image (e.g. multi-crop
ensembles) are *independent* looks aggregated at the end. RHAN-NXA's
glimpses are **belief-driven and sequential**: each look is chosen by
the current belief's uncertainty, compared against an explicit
prediction, and folded into a persistent-per-image belief that changes
what happens next. The count (T = 4) is the least interesting part of
that. → `03_Perception_As_Investigation.md`, `09_Recurrence.md`.

## "`U_t` is uncertainty about everything the model believes."

**No.** Current `U_t` is **class-conditioned** uncertainty from the
evidential readout — uncertainty over the classification readout, not
over `B_t`'s content in general. This is an intentional, documented
Gen-1 scoping compromise: a genuinely belief-centric uncertainty has
no existing training target and would require new machinery.
Representation-level uncertainty is **PENDING DECISION**, deferred
past the first integrated build. → `07_Uncertainty.md`.

## "`S_t = None` means RHAN cannot represent objects."

**No.** It means **no explicit structural state module is active in
the core Gen-1 configuration**. The pooled `z_t` will still encode
object-relevant features implicitly (it is trained on object
classification). What is absent is the explicit, structured,
inspectable state. → `06_Structure_State.md`.

## "SBR was rejected, so object-centric perception was disproven."

**No.** REJECTED IMPLEMENTATION ≠ REJECTED IDEA. What Gen 0 rejected
is the **16-slot SBR as tested** — on doubled evidence: the
everything-slot ablation improved accuracy (retained 1.0157), and the
−9 to −10pp clean-collapse signature appeared every time that
implementation was active. The *concept* of structural representation
remains an EXPERIMENTAL CANDIDATE, DEFERRED, with pre-registered
re-entry rules (2–4 slots; per-slot decodability distinguishable from
chance; tightened gates). → `06_Structure_State.md`.

## "Prediction error means reconstructing pixels."

**No.** The current Gen-1 candidate is **latent next-glimpse feature
prediction**: predict the encoder's own patch/token features at the
next fixation, in the encoder's own space. Pixel reconstruction is
REJECTED for Gen-1 — and precisely, not rhetorically: mechanistically
disfavored (precision dilution, belief drift in Lens analysis),
statistically **inconclusive** (E1's −0.90pp vs D was NOT
significant), and its engineering cost (a decoder that doesn't exist
in the substrate) unjustified. Do not claim "pixel reconstruction was
proven harmful" — that overstates the significance test. →
`08_Prediction_Error.md`.

## "AIS-v2 is proven to improve robustness."

**Not yet.** Its candidate-preference diagnostic (r = 0.706, 512
samples) is real, targeted, REQUIRED evidence that the scoring head's
predictions correlate with policy choice. But its isolated 16-seed
robustness/accuracy contribution is currently **UNKNOWN** — the Gen-0
experiment was confounded by unintentionally-active legacy SBR. The
confounded +11.42pp-vs-TRADES number is not settled fact and must be
re-tested cleanly. → `10_AIS_v2.md`, `16_Gen0_Evidence_And_Confounds.md`.

## "More recurrence automatically means better perception."

**No.** That is exactly why parameter-matched and compute-matched
controls are mandatory before any recurrence claim: without them,
"recurrence helped" may just mean "more parameters/FLOPs helped." Gen
0 proved this failure mode is live — the SBR rungs' clean gains
tracked added capacity. → `09_Recurrence.md`, `15_Status_And_Decision_System.md`.

## Bonus confusions (from the implementation side)

### "The belief state is saved between images."
**No.** `B_t` exists only within one image's forward pass and is
discarded after T glimpses. What is checkpointed is the *model*. There
is no cross-image belief persistence in Gen-1 — any code that caches
belief state across images implements something not specified. →
`04_Belief_State.md`, `12_Architecture_Data_Flow.md`.

### "`A_t` should be differentiable so gaze learns."
**No.** `A_t` is a record. Gaze learning happens through the **policy's
selection mechanism** (soft selection during training), not through
the coordinates. At inference, selection is hard argmax. →
`13_Gradient_Flow.md`.

### "`E_t` can be detached to stabilize training."
**Never.** Detaching `E_t` before the update severs the predictor's
realized-glimpse learning path — the exact bug class that repeatedly
starved components in Gen 0. The observed side is detached; the
predicted side and `E_t` itself are not. → `08_Prediction_Error.md`,
`13_Gradient_Flow.md`.

### "The gate passed, so the mechanism works."
**Careful.** sbr0's gate passed while measuring near-uniform,
near-chance per-slot content — a floor calibrated to detect collapse
cannot detect vacuity. "Gate passed" means "the pre-registered
criteria passed," nothing more. Gates need the power to fail for the
right reasons. → `06_Structure_State.md`, `16_Gen0_Evidence_And_Confounds.md`.

### "Low belief drift under attack means the model is robust."
**Not by itself.** A model can achieve low adversarial drift by
becoming insensitive to ALL new evidence — which is a FAILURE, not a
success. That is why Gate 9 reports L_stab's drift reduction
ALONGSIDE an OOD/novel-evidence responsiveness score on a disjoint
probe set, and low drift on BOTH is graded FAILED, full stop — a hard,
pre-registered rule. → `21_L_stab_Stability.md`.

### "Memory means adding retrieval (RAG)."
**No.** Gen-1's memory is the model's weights plus the within-image
belief trajectory. Episodic persistence across images is DEFERRED
pending a specific motivating experiment, and external retrieval is
REJECTED outright — no project evidence motivates it. → `22_Memory.md`.

### "A fixed Gabor frontend is harmless — just add it."
**Not first.** It is a low-level, largely orthogonal intervention
whose frequency/robustness effects would contaminate attribution for
whether the belief-dynamics idea itself works. It is built LAST, with
a standalone ablation required before any integrated use. →
`23_V1_Frontend.md`.

### "Steps 7 and 8 can share one evaluation run since both build on step 6."
**No.** They are parallel-buildable but NOT parallel-evaluable against
step 6 simultaneously — each needs its own clean comparison to step
6's frozen result, never to each other directly, or you reproduce
exactly the D2/D3 confound. → `24_Training_Phase_DAG.md`.

### "Two mechanisms can be compared arm-to-arm in the ablation matrix."
**No.** Every arm's primary comparison is against the step-6 frozen
reference — never another arm. Arm-vs-arm comparison reintroduces
multi-mechanism attribution collapse. → `25_Ablation_Matrix.md`.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan (misunderstanding examples mandated by the documentation task,
> grounded in Parts 1.A–1.F and Part 0); `report/Gen0.md` for the
> underlying numbers.

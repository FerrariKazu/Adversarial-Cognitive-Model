# 03 — The Central Idea: Perception as an Investigation

*Level 0 reading. This chapter defines the loop everything else hangs
off. ~15 minutes.*

---

## The canonical cycle

RHAN-NXA's entire architecture is an implementation of one loop:

> ## **Predict → Observe → Error → Precision → Update → Attention → Repeat**

Before any mathematics, here is every word in plain English.

### Predict
**"What do I expect to find if I look there?"**

Before gathering new evidence, the model forms an explicit expectation
about what a candidate observation would reveal, given everything it
currently believes.

### Observe
**"What did the visual system actually receive?"**

The model takes the glimpse — the actual encoder output at the chosen
location.

### Error
**"How different was reality from my prediction?"**

The mismatch between prediction and observation is computed and kept
as a first-class signal (`E_t`). Large error means the current belief
was missing something. Small error means the belief already accounted
for what was seen.

### Precision
**"How much should I trust this evidence?"**

Not all evidence is equally reliable. The precision/uncertainty step
weights how strongly the new evidence should influence the belief
(`U_t` and the precision factor `Π_t`).

### Update
**"How should my current belief change?"**

The belief is revised using the error signal, scaled by precision —
not overwritten, *revised*.

### Attention
**"What should I inspect next?"**

The updated belief includes an updated uncertainty estimate, and
uncertainty tells the model where looking next is most likely to
resolve remaining ambiguity (AIS-v2).

### Repeat
**"Is the current belief good enough, or should perception continue?"**

Gen-1 runs this cycle a **fixed T = 4** times per image. Adaptive
"am I done yet?" halting is deliberately deferred — see
`09_Recurrence.md`.

## Why this loop is the center

Everything else in the architecture is one step of this loop wearing a
specific implementation:

| Loop step | Implementation | Chapter |
|---|---|---|
| Predict | latent glimpse predictor | `08_Prediction_Error.md` |
| Observe | compact ViT at the chosen gaze | `09_Recurrence.md` |
| Error | `E_t` in latent feature space | `08_Prediction_Error.md` |
| Precision | Dirichlet evidence / `Π_t` | `07_Uncertainty.md` |
| Update | `UpdateNet` belief revision | `08_Prediction_Error.md` |
| Attention | AIS-v2 candidate scoring | `10_AIS_v2.md` |
| Repeat | fixed T = 4 (halting deferred) | `09_Recurrence.md` |

If a proposed feature cannot be placed in this table, it is not part
of the Gen-1 core — that is what `20_Scope_Boundaries.md` enforces.

## What this is, and is not, scientifically

This loop is **predictive-coding-inspired** — it has the
predict-then-correct shape associated with predictive coding in the
literature. RHAN-NXA does **not** claim to be a full or literal
implementation of predictive coding as a neurobiological theory, and
does not claim the loop is how humans perceive. It is a computational
organization whose *consequences* (robustness, calibration,
sample-efficiency) are open experimental questions.

> **FACT:** the loop's steps and their ordering are fixed by the
> specification.
> **HYPOTHESIS:** that running perception this way produces measurably
> better robustness or calibration than a single-pass classifier.
> Every status label in this documentation exists to keep that
> distinction visible.

## A first taste of the running example

The running example used throughout this documentation: **an image
where an object is partly hidden.** A single-pass classifier must guess
from one ambiguous view. The investigation loop instead:

1. looks once, forms a preliminary belief with high uncertainty;
2. predicts what a second look elsewhere would reveal;
3. compares prediction to observation, and lets the mismatch
   reorganize the belief;
4. chooses the next look based on where ambiguity remains.

The full step-by-step version with concrete states is
`11_Complete_Perceptual_Loop.md`.

---

## Reading path

Next: `04_Belief_State.md` — the state this loop maintains.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.B (predict→observe→error→precision→update→gaze cycle),
> Part 1.C (fixed T), Part 1.E (AIS-v2 step placement).

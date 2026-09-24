# 21 — `L_stab`: Belief-Stability Objective (Staged)

*Level 2–3 reading. A mechanism whose most important property is when
it is allowed to act.*

---

## In one sentence

`L_stab` is a stability objective over belief trajectories — and in
Gen-1 it is **LOCKED to a staged protocol**: diagnostic-only first,
promotable to a training objective only after the clean core has a
validated result to compare against.

## The intuition

Adversarial perturbations don't just change what a model outputs —
they can derail *how* it thinks. In RHAN-NXA, they can flip the belief
trajectory: small input corruption produces a wildly different path of
belief states than the clean image did. `L_stab` targets that: how
similar is the belief trajectory under a perturbed image to the
trajectory under the clean one?

That is a useful diagnostic immediately, and a tempting training
objective immediately. The staging rule exists because of the
temptation.

## The mechanism, conceptually

- Take a clean image and an adversarially perturbed copy.
- Run both through the perception loop; obtain two belief
  trajectories.
- `L_stab` measures drift between them — the plan specifies the drift
  metric comes through `BeliefState.drift_to()`, fed by the one
  Dirichlet uncertainty representation (Part 1.F).
- As a **diagnostic**: report it, gate nothing on it, train nothing
  against it.
- As a **training objective** (only when promoted): penalize
  adversarial belief drift during training.

## The staging protocol (LOCKED)

```text
(A) DIAGNOSTIC ONLY
    through the entire core build and first ablation matrix
    — computed, reported, never optimized against.

(B) TRAINING OBJECTIVE
    ONLY after the core system (WITHOUT L_stab) has a validated
    ImageNet-100 result to compare against — its own isolated arm
    (Part 2 step 8, Part 3's +L_stab arm), evaluated against step 6's
    frozen result.
```

Why the staging is non-negotiable: promoting a stability loss before
the clean core is validated makes attribution impossible. If the core
+ L_stab behaves differently, you cannot tell whether the
belief-dynamics thesis or the stability objective produced it — the
exact single-mechanism-attribution failure the plan is built to avoid.

## The responsiveness guard (the hard, pre-registered rule)

This is the most important paragraph in this chapter, verbatim in
substance:

> **A model minimizing adversarial belief drift by becoming
> insensitive to ALL new evidence is a FAILURE, not a success.**

**Gate 9** therefore requires L_stab's adversarial-drift reduction to
be reported **ALONGSIDE** an **OOD / novel-evidence responsiveness
score**, computed the same way, on a **disjoint probe set**. A result
showing low drift on BOTH — stable under perturbation *and* unresponsive
to genuinely new evidence — is graded **FAILED, full stop**.

This is a **hard, pre-registered stopping condition**, not a caveat
added after the fact. A stability objective that flattens perception
into indifference has optimized the metric by destroying the
capability the architecture exists to build. (The pointer in Part 0's
terminology: gates must be able to fail for the right reasons.)

## Ownership

L_stab is Agent H's component (Objectives), staged per Part 1.G; the
drift metric it reads is defined on the BeliefState interface (Agent
B/Agent 0), and it only needs `drift_to()` — not a trained model — so
its scaffolding can be built any time after B exists (Part 2 build
graph).

## Scientific status

| Claim | Status |
|---|---|
| Staged protocol (diagnostic → objective, gated on validated core) | **LOCKED** |
| L_stab as a mechanism | **EXPERIMENTAL CANDIDATE, explicitly staged, not REQUIRED** |
| The responsiveness guard / Gate 9 two-sided rule | **REQUIRED as designed** (threshold PENDING DECISION — calibrated from the diagnostic-only phase's own baseline distribution before L_stab ever becomes a loss) |

## What this does NOT mean

- "Diagnostic only" does not mean ignored — it is computed and
  reported from the first build; it is just never optimized against
  during that phase.
- The guard does not pre-judge L_stab as harmful; it makes the harmful
  failure mode *detectable and terminal* rather than rationalizable.
- L_stab's eventual promotion is not guaranteed. If step 6's core
  result is never validated, L_stab stays a diagnostic.

## Open questions

- The OOD-responsiveness floor threshold — **PENDING DECISION** by
  design, to be calibrated from the diagnostic-phase baseline
  distribution (Gate 9, Part 6).
- Which belief components the drift metric weighs (the plan specifies
  the metric lives on `drift_to()`; component weights are an Agent-H
  implementation matter, not documented here).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.G — L_stab (staging, responsiveness guard, Gate 9 rule,
> status — captured verbatim in `../MASTER_PLAN.md`); Part 2 (step 8
> gating); Part 6 (Gate 9 threshold status).

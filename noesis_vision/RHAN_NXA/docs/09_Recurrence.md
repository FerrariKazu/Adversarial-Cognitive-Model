# 09 — Recurrence

*Level 1 reading. Covers both kinds of recurrence and the controls
any recurrence claim requires.*

---

## In one sentence

A normal feed-forward classifier gets one main computational pass;
RHAN-NXA deliberately allows perception to unfold over multiple steps
— *inside* each glimpse (weight-tied refinement) and *across* glimpses
(fixed T = 4).

## Why recurrence at all

Perception as an investigation (`03_Perception_As_Investigation.md`)
needs time: predictions must precede observations, and beliefs must
accumulate across looks. A single forward pass has no room for either.
Recurrence is how the architecture buys that time — in two distinct
places.

## Two kinds of recurrence

### 1. Within-glimpse recurrence (weight-tied refinement)

Inside one glimpse, the transformer block is run **2–3 times** before
pooling, with **shared (tied) weights** — Universal-Transformer-style.

Beginner layer: instead of a stack of different layers, the *same*
layer is applied repeatedly, letting the token representations
iteratively refine before they are summarized into `z_t`.

**Why tied weights matter (the distinction that trips people up):**

> Weight tying means running the same block multiple times increases
> **computation** without proportionally increasing **parameter
> count**.

Keep these four concepts separate — they are *not* the same thing:

| Concept | Meaning | Changes when we iterate the tied block? |
|---|---|---|
| **depth** | how many sequential applications | yes |
| **recurrence** | re-applying the same weights | the mechanism itself |
| **parameter count** | how many learnable weights exist | **no** (weights are shared) |
| **compute (FLOPs)** | how much arithmetic runs | yes, roughly linearly |

This is critical for team members: an experiment comparing
"2 refinements vs 3 refinements" changes compute but *not* capacity —
which is exactly what makes such comparisons clean to interpret, and
what keeps the compactness accounting simple regardless of how many
refinement steps are used.

### 2. Across-glimpse recurrence (the investigation loop)

The system observes **T fixed glimpses per image**:

```text
t = 1 … T        with the initial Gen-1 setting:  T = 4
```

Each glimpse produces one `B_t` via the
predict→observe→error→precision→update→gaze cycle. The belief state
carries the accumulation; T is the loop's trip count.

**Why T = 4?** Because it reuses the STL-10 project's own validated
convention rather than re-deriving a number from nothing. This is a
**design decision grounded in project precedent**, not a claim that 4
is optimal — and it is the kind of number a future ablation may
revise, with proper controls.

## Adaptive halting: DEFERRED

Gen-1 runs the loop at **fixed T = 4**. Adaptive halting — letting the
model decide "I've seen enough" — is **deferred for the first build**.

Why:

> Gen 0's own halting history (v10's loss-conflict-with-Banach-proof
> incident, AIS-v1's modest halting effect) argues for validating the
> core loop at fixed depth before reintroducing a mechanism with a
> documented history of fighting other objectives.

In plainer terms: halting adds a second objective ("stop when done")
that must coexist with the learning objectives, and Gen 0 twice saw
that coexistence go badly. Fixed T removes that entire failure class
until the core loop is proven on its own.

## The controls rule (non-negotiable)

> **Controls required for every recurrence ablation:** a
> **parameter-matched control** (same total params, recurrence
> disabled / T=1) **and** a **compute-matched control** (same total
> FLOPs, achieved via width instead of depth) — both must exist before
> any "recurrence helps" claim is made.

Why this is mandatory: otherwise a team member might falsely conclude
"recurrence helped" when the actual cause was "the model simply had
more parameters or FLOPs." Gen 0 already proved this risk is *live* —
the SBR rungs' clean-accuracy gains tracked added capacity, not
mechanism (see `16_Gen0_Evidence_And_Confounds.md`).

Concretely, before any of these claims is written down:
- "T=4 beats T=1" → needs param-matched AND compute-matched controls
- "3 refinements beat 2" → compute changes, params don't; still needs
  the compute-matched width control for the across-the-board claim
- "recurrence is why robustness improved" → both controls, always

## Scientific status

| Claim | Status |
|---|---|
| Hybrid recurrence (tied within-glimpse + fixed-T across-glimpse) | **LOCKED** as design — but narrowly scoped: chosen because Option A alone abandons the belief-state thesis (it becomes a plain recurrent ViT, useful only as a CONTROL) and Option B alone likely under-represents each glimpse (no iterative token refinement before pooling into `z_t`). Not chosen because "more expressive." |
| T = 4 | **REQUIRED** for the first build (project-precedent default; revisable via controlled ablation) |
| Within-glimpse refinement count 2–3 | **DESIGN DECISION** — exact count is a substrate/implementation parameter |
| Adaptive halting | **DEFERRED** |

## What this does NOT mean

- Recurrence is not claimed to be biologically rhythm-driven or
  oscillatory; "steps" here are computational iterations.
- Fixed T is not a claim that adaptive halting is a bad idea — it is a
  claim that halting must be validated *after* the core loop, not
  simultaneously with it.
- Weight tying is not a cost-saving trick here; it is what keeps
  capacity-vs-compute distinguishable in every ablation.

## Open questions

- The exact within-glimpse iteration count (2 vs 3) — to be fixed by
  the substrate and validated under the controls rule.
- Whether T=4 remains right as resolution/dataset scale changes.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.C — Recurrence (LOCKED status, Option A/B/C scoping,
> weight-tying rationale, T=4 rationale, halting deferral, and the
> controls requirement are captured verbatim in `MASTER_PLAN.md`).

# 01 — What Is RHAN-NXA?

*Level 0 reading. No ML background assumed. ~10 minutes.*

---

## In one sentence

RHAN-NXA is an image-understanding system that does not classify an
image in a single pass; it **investigates** the image over several
deliberate looks, maintaining an explicit, updatable **belief state**
about what it is seeing, and only reads a classification out of that
belief at the end.

## The problem it is trying to solve

### The problem with "look once, answer immediately"

A standard image classifier — even a very good one — commits to an
answer from one view of the image. If that view is ambiguous, partial,
or adversarially corrupted, the model has no mechanism to notice its
own confusion and *do something about it*. It cannot look again. It
cannot ask itself "what would I expect to see next, and does reality
match?"

RHAN-NXA's premise: perception should behave more like an
**investigation** — a sequence of evidence-gathering steps, each one
informed by what is already believed and by what is still uncertain.

### What "perception" means in this project

Perception is **not** the classification output. Perception is the
*process* of building an internal state that is good enough to answer
questions from. Classification is a **readout** — a small head attached
to the end of the perceptual state. This distinction is foundational:
the architecture's internal state is defined by what it believes about
the image, not by the label it eventually emits.

### Why robustness is part of the same problem

RHAN's research program studies images that have been deliberately
perturbed (adversarial examples — see the Glossary). A model that
perceives by accumulating and cross-checking evidence has a
structurally different relationship with corrupted input than a model
that maps one corrupted view straight to a label. Whether that
structural difference actually buys robustness is precisely what the
experiments are designed to test — it is a **hypothesis**, not an
established result.

## What is a belief state?

A **belief state** is a structured snapshot of everything the model
currently "thinks" about the image:

> **"RHAN-NXA needs somewhere to represent what it currently believes
> about the image."**

At every investigation step *t*, that snapshot is:

```text
B_t = (z_t, S_t, U_t, E_t, A_t)
```

In plain English: what the model thinks it sees (`z_t`), optional
structural detail (`S_t`), how uncertain it is (`U_t`), how surprising
the latest evidence was (`E_t`), and where it has looked (`A_t`).

You do not need to understand the mathematics yet. The full walk-through
is `04_Belief_State.md`.

## Why does the model look more than once?

Because one look is not always enough, and — critically — because the
*choice of where to look next* can itself be informative. RHAN-NXA
takes **T = 4 glimpses** per image (the initial Gen-1 setting). Each
glimpse is a deliberate act: the model predicts what it expects to find
at a candidate location, compares that prediction to what it actually
finds, updates its belief, and chooses the next location.

## Why does it predict before observing?

Because the **difference between prediction and observation** is
information. If you expect to see a red cup and instead see a blue
bottle, that mismatch tells you something your previous belief was
wrong about. This is the predictive step at the center of the loop —
see `08_Prediction_Error.md`.

## Why does it care about uncertainty?

Because "what the model believes" and "how sure it is" are different
facts, and the second one drives the first one's updates. Uncertainty
is what makes looking again *targeted* instead of random. See
`07_Uncertainty.md`.

## Why does it decide where to look next?

Because in a fixed-compute budget, *where* you look matters as much as
*how many* times you look. The AIS-v2 mechanism scores candidate
locations by predicted uncertainty reduction and commits to the most
informative one. See `10_AIS_v2.md`.

## What RHAN-NXA is not

Before going further, calibrate expectations — RHAN-NXA is not "a
model of human consciousness," not "a proven-robust classifier," and
not a bundle of independently impressive components. It is a
**hypothesis about computational organization**. The full list of
disclaimers is `17_Common_Misunderstandings.md` and
`15_Status_And_Decision_System.md`.

---

## Reading path

Next: `02_Why_RHAN_NXA_Exists.md` (why a new generation was needed) →
`03_Perception_As_Investigation.md` (the central loop).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A–1.F (introductory framing); the "perception as
> investigation" formulation is from the plan's canonical cycle.

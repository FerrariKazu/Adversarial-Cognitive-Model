# 29 — Literature Classification

*Level 3 reading. What RHAN-NXA borrows, what it adapts, and the one
place a novelty claim is even entertained — with the caveat the plan
itself attaches.*

---

## Why this chapter exists

Research documents drift toward novelty inflation: ideas that are
known priors get described as contributions. The plan counters this
with an explicit classification of every recognizable external idea —
and an honest caveat about what has *not* been verified.

## The classification labels

| Label | Meaning |
|---|---|
| **KNOWN PRIOR** | An established idea from the literature, used here as-is |
| **ENGINEERING ADAPTATION** | A known prior, modified for this architecture's setting |
| **SCIENTIFIC COMBINATION** | The combination itself may be new — *not verified*, claimed only at that strength |

## The classifications (as stated in the plan)

| Element | Classification |
|---|---|
| ViT / DINOv2 warm-start | **KNOWN PRIOR** |
| Universal-Transformer-style tied recurrence | **KNOWN PRIOR** |
| Predictive coding (as inspiration, not implementation) | **ENGINEERING ADAPTATION** |
| JEPA-style latent-space prediction | **KNOWN PRIOR** — adapted here to **glimpse-level** rather than masked-region prediction, which makes it an **ENGINEERING ADAPTATION** |
| Evidential deep learning (Dirichlet evidence) | **KNOWN PRIOR** |
| One predictor serving both belief-update error AND active-gaze candidate scoring, inside an explicitly typed, **None-safe** belief object | **SCIENTIFIC COMBINATION at best** |

## The novelty claim, stated at its honest strength

> The unification of one predictor serving both belief-update error
> and active-gaze candidate scoring, inside an explicitly typed,
> None-safe belief object = **SCIENTIFIC COMBINATION at best** —
> the plan has **not verified** this exact pairing exists in the
> literature and **will not claim novelty stronger than that without
> checking**.

Read that carefully: even the architecture's central unification is
claimed only as a *possible* combination, pending a literature check
that has not been done. This documentation preserves that boundary —
no sentence anywhere in `RHAN_NXA/` claims "first," "novel," or "new"
beyond what this chapter licenses.

## The caveat that travels with this chapter

> *(Honest, unverified-citation-list caveat, from the plan.)*

The classifications above are the plan's own honest labeling of ideas
it recognizes — **no citation metadata has been verified**. Specific
papers, authors, years, and DOIs are deliberately absent from this
documentation because verifying them is a separate, unperformed task.
Anyone writing a paper or report from this material must do that
verification first; `docs/research/` and the project's reference
corpus in the main repository are the starting points, not this
chapter.

## Conceptual relationships (explained, not claimed)

For readers who want the intuition of each borrowed idea:

- **ViT/DINOv2 warm-start** — begin the compact ViT from pretrained
  weights rather than scratch. A training-efficiency prior, not a
  mechanism.
- **Universal Transformer** — apply one transformer block repeatedly
  with tied weights (depth without new parameters). RHAN-NXA uses this
  for within-glimpse refinement (`09_Recurrence.md`).
- **Predictive coding** — the brain-inspired schema of predicting
  inputs and correcting by errors. RHAN-NXA adopts the *shape* of this
  loop (predict→observe→error→precision→update) as engineering, and
  explicitly does **not** claim a full predictive-coding
  implementation or biological equivalence (`03_Perception_As_Investigation.md`).
- **JEPA-style latent prediction** — predict in representation space
  rather than pixel space. RHAN-NXA adapts the *space* (latent) but
  changes the *target*: next-glimpse features at a chosen fixation,
  not masked regions (`08_Prediction_Error.md`).
- **Evidential deep learning** — emit evidence and derive
  uncertainty (Dirichlet). Used as the one uncertainty representation
  (`07_Uncertainty.md`).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 6 — literature classification paragraph (captured
> verbatim, caveat included, in `../MASTER_PLAN.md`).

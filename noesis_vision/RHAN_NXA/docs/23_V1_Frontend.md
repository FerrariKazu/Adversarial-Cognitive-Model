# 23 — V1 Frontend (Section 10)

*Level 2 reading. A small, fixed component whose main interest is its
place in the build order.*

---

## In one sentence

A fixed (non-learnable), biologically-inspired V1-style frontend — a
Gabor-convolution layer at the input — graded EXPERIMENTAL CANDIDATE
and deliberately built **LAST**.

## The intuition

Biological visual systems begin with orientation- and frequency-tuned
filters (V1 simple cells). A Gabor frontend gives the network a fixed
bank of such filters as its first stage: cheap, well-understood, and
plausibly helpful for robustness properties (frequency structure
survives some perturbations that fool learned first layers). "V1"
here refers to that first cortical processing stage — the label is a
structural analogy, not a claim of neural fidelity.

## Properties (as specified)

| Property | Value |
|---|---|
| Learnable? | **No — fixed (non-learnable)** |
| Parameter cost | Small |
| Compute cost | Some added compute for the Gabor convolution |
| Build order | **LAST** — after the core recurrent-belief loop validates without it |
| Isolation requirement | **Standalone ablation required** before folding into any "integrated" configuration |

## Why built last (the attribution argument)

The frontend is a **low-level, largely orthogonal intervention**. If
it were present from the start, its frequency/robustness effects would
be entangled with everything downstream — and every result of the core
build would carry the question "was that the belief dynamics, or the
Gabor bank?"

Built last, the order of evidence is clean:

1. the core recurrent-belief loop validates **without** it → any
   result is attributable to the core thesis;
2. the frontend runs as its **own standalone ablation** (Part 3's
   +V1-frontend arm, 8+ seeds, everything except the frontend flag
   identical) → its marginal value is measured directly against the
   step-6 reference;
3. only then, if it earns it, is it folded into an integrated
   configuration.

This is the same single-mechanism-attribution discipline as every
other staging decision in the plan (L_stab's staging, S_t's gating) —
applied to the component most likely to be added prematurely because
it is small and easy.

## Where it sits in the architecture

```text
image → [V1 frontend (fixed Gabor bank)] → compact ViT → …core loop…
```

It is the only documented component upstream of the substrate. It has
no gradients (fixed), so it never appears in `13_Gradient_Flow.md`'s
tables beyond this note: a non-learnable frontend must be *excluded*
from trainable-parameter counts but *included* in FLOPs accounting
(Part 6's compute accounting covers both automatically via Agent A's
logging).

## Scientific status

**EXPERIMENTAL CANDIDATE** — plausible, specified, deliberately
sequenced last; its marginal value is unknown until its own ablation
runs.

## What this does NOT mean

- "Fixed" does not mean frozen-forever — a future variant could make
  the frontend learnable, but that would be a new decision requiring
  new evidence (and it would change the parameter-matched accounting).
- "Experimental candidate" does not mean scheduled — it means the
  ablation exists in the matrix (Part 3) and nothing more is promised.
- No claim is made that Gabor filters match human V1; the analogy is
  structural.

## Open questions

- Gabor bank specifics (frequencies, orientations, spatial support) —
  implementation parameters, not specified in the authorized plan
  material.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.I — V1 Frontend (properties, built-last rationale,
> standalone-ablation requirement, status — captured verbatim in
> `../MASTER_PLAN.md`); Part 3 (+V1 frontend arm); Part 6 (compute
> accounting coverage).

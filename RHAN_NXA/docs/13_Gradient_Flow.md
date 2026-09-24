# 13 — Gradient Flow

*Level 2 reading, written beginner-first. The rules here are
specification, not style.*

---

## First, why this chapter exists

> A tensor can participate in the forward computation but still
> accidentally stop learning if its gradient is detached.

**Detachment** (beginner layer): in automatic differentiation, every
value in a computation remembers how it was made, so that "blame"
(gradients) can flow backwards to the parameters that produced it.
*Detaching* a value cuts that memory — the value still flows forward,
but no blame flows backward through it. That is sometimes exactly what
you want (ground truth should not be "learned from" via the loss
sneaking through it) — and catastrophic when done by accident, because
the affected component silently stops learning while everything looks
fine on the surface.

Gen 0's single most repeated failure mode was exactly this class of
bug. Hence: gradient reachability is written down here as part of the
architecture.

## The per-component gradient table

| Component | Gradients? | The rule | Why the rule exists |
|---|---|---|---|
| `z_t` | **always** | gradient-bearing | the primary learning pathway of the perception loop; every objective reaches the backbone through it |
| `U_t` | **always** | gradient-bearing | the evidential head must learn to concentrate evidence when evidence warrants it |
| `S_t` | **only when not None** | gradient-bearing iff present | in the core build `S_t` is None; if structure returns, it must learn like any other component |
| `E_t` | **through the predicted side only** | observed target **detached**; predicted value **not detached**; `E_t` itself **never detached before use in the update** | the observation is ground truth (nothing should learn *through* it), the prediction must learn (blame must reach the predictor), and the update must remain differentiable or `UpdateNet` and everything upstream of `E_t` silently starves |
| `A_t` | **never** | coordinate/state record | differentiability lives in the **policy that produced** the gaze choice, not in the record of where it looked; the record is bookkeeping |

## The `E_t` rule, expanded (the one to get right)

The prediction-error pipeline has exactly three gradient facts:

1. **Observed features: detached.** They are the target. If gradients
   flowed through them, the model could reduce error by *changing what
   it treats as reality* rather than by predicting better.
2. **Predicted features: not detached.** The predictor improves only
   if blame reaches it through the mismatch.
3. **`E_t` entering the update: differentiable.** The update
   `z_{t+1} = z_t + Π_t · UpdateNet(z_t, E_t)` must carry gradients
   through `E_t` back into the predictor (and, via the belief's own
   path, into everything upstream). Detaching `E_t` "to be safe" is
   the bug — it severs the predictor's second learning pathway (the
   realized-glimpse one) and the update's sensitivity to prediction
   quality, silently.

## The gaze path (where differentiability actually lives)

`A_t` — the list of coordinates and the step index — carries **no
gradient**. This is correct and deliberate:

- at **inference**, selection is hard argmax — no gradients exist by
  construction;
- at **training**, selection is soft (e.g. Gumbel-softmax /
  straight-through over the K candidates) precisely so gradients flow
  through the *selection mechanism*, not through the coordinate
  record. If you find yourself needing `A_t` to be differentiable, the
  actual bug is upstream in how the policy's gradients are routed.

## `UpdateNet`'s isolation contract

`UpdateNet` is a small, new, independently gradient-isolated
component:

- **own optimizer group** (it must not share a learning-rate group
  with the backbone — its inputs and scale differ);
- **own pre-flight `|dW|` check before any smoke test** — verify, on a
  tiny run, that its parameter gradients are non-zero and sane before
  believing any training curve.

This is the standing Gen-0 rule applied from day one, in response to
a documented history of starved components discovered only after
wasted runs.

## The None-state gradient rule

When `S_t` is None (the core build), there is simply no structural
gradient path — and the explicit None branch must not *simulate* one
(e.g. by routing structural placeholders through `z_t`'s update). When
`S_t` returns in a future build, its update is gradient-bearing in its
own right, following the same predict/observe/error/precision/update
shape independently — **not derived from `z_t`'s `E_t`**.

## Why this is "part of the scientific specification"

> Gradient reachability is part of the scientific specification, not
> merely an implementation detail.

An ablation that concludes "the prediction-error update didn't help"
is *meaningless* if `E_t` was accidentally detached — the mechanism
was never actually learning. Every negative result depends on the
gradient contract having been honored; that is why it is written here
with the same authority as the architecture itself.

## Quick self-check for implementers

Before a smoke run, verify all of:

- [ ] predictor gradients non-zero via the realized-glimpse (`E_t`) path
- [ ] predictor gradients non-zero via the AIS candidate-scoring path
- [ ] `UpdateNet` in its own optimizer group; `|dW|` pre-flight passed
- [ ] observed-side features detached (and *only* the observed side)
- [ ] `E_t` not detached anywhere between computation and update
- [ ] `A_t` requires no grad anywhere
- [ ] explicit `S_t is None` branch exercised directly by a test

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A — BeliefState (the "Gradients through B_t" paragraph,
> preserved verbatim in `MASTER_PLAN.md`), Part 1.B (UpdateNet
> isolation, detach rules), Part 1.E (soft/hard selection).

# 05 — `z_t`: The Global Content State

*Level 1 reading. Builds on `04_Belief_State.md`.*

---

## In one sentence

`z_t` is the dense, continuous, differentiable vector that holds what
the model currently thinks the image contains, globally.

## The intuition

If the belief state `B_t` is the investigation's notebook, `z_t` is
the main page of running notes — the summary of perceptual content
that everything else reads and updates. It is *not* the raw pixels,
and it is not any single glimpse's output; it is the accumulated
global content after `t` glimpses and `t` belief updates.

## A concrete example

At glimpse 1 of the partly-hidden-animal image, `z_t` might encode
"animal-like texture, quadruped posture, moderate confidence." After a
well-chosen second glimpse resolves the head region, `z_t` shifts
toward "deer-like: muzzle + antler geometry consistent." Nothing about
`z_t` is a label — a classifier reads it later — but it is the thing
that *would* support the label.

## What it is made of (beginner layer)

You do not need to understand transformers yet. At this level:

> `z_t` is the model's current internal representation of what it
> thinks it is seeing — one vector per image, revised after every
> glimpse.

## Formal definition (technical layer)

```text
z_t : (B, D_z)
```

- `B` = batch size
- `D_z` = **backbone embedding dimension**, locked once the compact-ViT
  substrate sets it (the exact numeric value is an implementation
  parameter, not specified in this documentation scope)
- dense, continuous, differentiable
- obtained as the **pooled CLS-token-equivalent output** of the
  within-glimpse recurrent transformer (see `09_Recurrence.md`)

### Do not confuse these four things

| Thing | What it is | Is it `z_t`? |
|---|---|---|
| raw pixels | the input image itself | no |
| patch/token features | per-patch encoder outputs (local, spatial) | no — these are `E_t`'s space |
| pooled representation | the aggregation step over tokens | the *source* of `z_t` |
| `z_t` | the belief-state's global content vector after the update | **yes** |

The distinction matters because `E_t` lives in the *token-feature*
space (predicted/observed local features), while `z_t` lives in the
*pooled global* space, and the architecture deliberately does **not**
assume the two are directly addable — that is what `UpdateNet` is for
(`08_Prediction_Error.md`).

## Gradient behavior

`z_t` carries gradients **always**. It is the primary pathway through
which the perception loop learns: classification loss, uncertainty
loss, and the prediction-error update all reach the backbone through
`z_t`.

## Lifecycle

- Initialized from glimpse 1's encoder output.
- Updated once per glimpse by
  `z_{t+1} = z_t + Π_t · UpdateNet(z_t, E_t)`.
- Read at the end of the investigation by the classification readout.
- Discarded when the image's forward pass ends (never checkpointed as
  a per-image artifact — see `04_Belief_State.md`).

## Scientific status

**REQUIRED** — as the belief state's global content carrier. This is
an architectural fact, not an empirical claim: no experiment is needed
to know the core build has a `z_t`; experiments are needed to know
whether maintaining it this *way* is what buys robustness (that is the
hypothesis layer).

## What this does NOT mean

- `z_t` is not guaranteed interpretable — "what the model thinks" is
  functional language, not a claim about human-readable semantics.
- `z_t` is not the whole belief: uncertainty, error, and gaze are
  separate components precisely so the model can reason about *its own
  state*, not just its content.

## Open questions

- The exact value of `D_z` (set by the substrate; outside this
  documentation's scope).
- Whether `z_t` itself should ever carry uncertainty beyond the
  class-readout `U_t` — **PENDING DECISION** (`07_Uncertainty.md`).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A — BeliefState (`z_t` bullet: shape, pooling origin,
> gradient status), and Part 1.B (update equation placement).

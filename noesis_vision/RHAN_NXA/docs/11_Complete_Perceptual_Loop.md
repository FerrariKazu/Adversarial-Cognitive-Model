# 11 — The Complete Perceptual Loop

*Level 1 reading. Everything assembled — one worked example, start to
finish.*

---

## The loop, as a diagram

```text
                 ┌──────────────────┐
                 │ Current Belief Bₜ│
                 └────────┬─────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Predict       │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Observe       │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Prediction    │
                  │ Error Eₜ      │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Precision / Uₜ│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Update zₜ     │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Choose where  │
                  │ to look next  │
                  └───────┬───────┘
                          │
                          ▼
                       Bₜ₊₁
                          │
                          └──────────► repeat
```

One full traversal of this diagram = one glimpse. The Gen-1 core runs
it **T = 4** times per image.

## Worked example: the partly-hidden animal

Setting: an STL-10-scale image, a quadruped roughly half-hidden behind
foliage. 10 classes, T = 4 glimpses.

### Glimpse 1 — the first look

- **Observe:** the encoder (with its 2–3 tied refinement steps) sees
  the image at an initial fixation — say, image center by convention
  for the first glimpse.
- **Belief formed:** `z_1` pools into something like "fur texture,
  quadruped stance." The evidential head emits evidence spread across
  {deer, horse, dog} → **`U_1` high**. `A_1` records the center.
- **E on the first glimpse: locked to zero.** There is no predecessor
  belief — nothing to be surprised against. The t = 0 convention is
  **LOCKED** in `../MASTER_PLAN.md` §1.B: `E_0 := 0` (zero tensor,
  correct shape, no gradient contribution), the update reduces to
  `z_1 = z_0 + Π_0 · UpdateNet(z_0, 0)` — pure observation, no
  correction term — and prediction (with genuine `E_t`) begins at
  t = 1. Not a configuration matter (`08_Prediction_Error.md`).

### Between glimpses — predict and choose

- **Predict:** the shared predictor, given `B_1` and K = 4–8 candidate
  locations around the highest-uncertainty regions, estimates what
  features each look would produce and how much `U` would drop
  (Dirichlet entropy). Candidates near the obscured head region score
  well; a re-look at the already-seen flank scores poorly.
- **First-glimpse exception (t = 0):** with no `U_0`-conditioned
  prediction to score against, candidate scoring at t = 0 falls back
  to the same heuristic-saliency sampling used for candidate
  generation — **LOCKED** (`../MASTER_PLAN.md` §1.B, `10_AIS_v2.md`).
- **Choose:** soft selection during training / hard argmax at
  inference. The head-area candidate wins. `A_2` will record it.

### Glimpse 2 — evidence against the expectation

- **Predict (pre-observe):** "if this is a deer, looking at the head
  region yields muzzle/antler-like local features."
- **Observe:** the encoder looks. It finds a rounded, short-muzzled
  profile — more dog-like.
- **Error:** predicted deer-ish features vs observed dog-ish features
  → non-trivial `E_2`, in the shared token-feature space.
- **Precision:** `U_1` was high, and the evidence is a *direct* look
  at a high-signal region → `Π` substantial.
- **Update:** `z_2 = z_1 + Π_2 · UpdateNet(z_2's inputs)` shifts
  toward "dog-like head on quadruped body." Evidence re-concentrates →
  **`U_2` lower than `U_1`** — the look was informative, and the
  uncertainty drop confirms the choice.

### Glimpse 3 — residual ambiguity

- Uncertainty remains about the obscured flank (body proportions
  distinguish dog breeds/classes). Candidates around the flank score
  highest; the model looks; the predicted-vs-observed match is close
  (`E_3` small) — the belief already anticipated this evidence, which
  *confirms* rather than corrects. `z_3` stabilizes; `U_3` drops
  further.

### Glimpse 4 — final evidence, then readout

- One more informative look per the same rule. After the update,
  `z_4` holds a consolidated percept; `U_4` is its residual
  uncertainty — possibly still nonzero (the foliage really did hide
  the decisive region), and that's fine: the belief *and its
  uncertainty* are both output.
- **Readout:** the classification head reads `z_4` (and the evidential
  state) → label + calibrated-ish confidence. The investigation ends;
  `B_4` is discarded with the forward pass.

### What a single-pass classifier did instead

One look, one guess — with the ambiguous flank and the hidden head
both unexamined, and no mechanism to *know* which one mattered.

## Where each component acts (trace table)

| Loop stage | Component acting | Reference |
|---|---|---|
| Observe | compact ViT + tied within-glimpse refinement | `09_Recurrence.md` |
| Predict | shared glimpse-feature predictor | `08_Prediction_Error.md` |
| Choose (between glimpses) | AIS-v2 (K candidates, Dirichlet-entropy scoring) | `10_AIS_v2.md` |
| Error | `E_t` in latent token space | `08_Prediction_Error.md` |
| Precision | `Π_t` from `U_t`'s Dirichlet evidence | `07_Uncertainty.md` |
| Update | `UpdateNet` (learned, not raw addition) | `08_Prediction_Error.md` |
| Record | `A_t` gaze history | `04_Belief_State.md` |
| Readout | classification head (readout, not the purpose) | `07_Uncertainty.md` |

## What the loop does not include

Conspicuously absent, by documented decision: structural state
(`S_t = None`), adaptive halting (fixed T), pixel reconstruction
(rejected), any memory across images (none exists — `B_t` is
per-image). See `20_Scope_Boundaries.md` for the full list.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A–1.F (the loop's steps and their locked parameters);
> the worked example is illustrative teaching material, consistent
> with — not adding to — the specification.

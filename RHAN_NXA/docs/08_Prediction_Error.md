# 08 — `E_t`: Prediction Error

*Level 1 reading. This is the chapter that answers "what does the
model predict, and how does the error update the belief?"*

---

## In one sentence

`E_t` is the mismatch between what the model predicted it would observe
and what it actually observed — computed in the encoder's own latent
feature space at the next glimpse's fixation.

## Why prediction error exists

> If you expect to see a red cup and instead observe a blue bottle,
> the difference between expectation and observation is informative.

The mismatch localizes *what the current belief was missing*. It is
the learning signal inside perception itself — not just between the
model and a label, but between the model and the world, one glimpse at
a time.

## What RHAN-NXA explicitly rejects (and exactly how far)

### Rejected for Gen-1: pixel-space reconstruction

The earlier idea — predict the raw pixels (or edges) of what you'll
see next, error = pixel MSE — is **rejected for Gen-1**, for two
independent reasons:

1. **Mechanistic:** Gen 0's Lens analysis found reconstruction dilutes
   precision and increases belief drift — a real, if not statistically
   dramatic, cost.
2. **Engineering:** pixel-space prediction requires a decoder that does
   not exist in the compact-ViT substrate and would need its own
   parameter budget and gradient-isolation treatment, for no
   established benefit.

**The precise record** (do not strengthen this in either direction):

> The E1 **statistical** result was "−0.90pp vs D, **NOT significant**"
> — the mechanistic cost is real and repeated, but calling it
> "definitively proven harmful" **overstates** what the significance
> test showed.
>
> **Grade: mechanistically disfavored, statistically inconclusive,
> engineering cost not justified.** Not attempting it is the right call
> for reasons beyond the disputed statistic.

### Rejected for Gen-1: edge-map / HPC-style hand-designed targets

A separate fixed-feature target (edge maps, hand-designed feature
extractors) is rejected on the same reasoning: it requires its own
non-learnable extractor, does not share `z_t`'s space, and belief-HPC's
comparative evidence for it is **confounded** (the Part-0 SBR
contamination — `16_Gen0_Evidence_And_Confounds.md`). No standing
justification to build it fresh.

## The Gen-1 candidate: latent next-glimpse prediction

**LOCKED DEFAULT (as a design):** `E_t` lives in `z_t`'s own latent
feature space — specifically, **patch/token-level features at the NEXT
GLIMPSE's fixation, predicted from the current belief.**

Note what this is deliberately *not*: it is not "predict the next
global `z_t` from the current global `z_t`." A single pooled vector
self-predicting across recurrent steps within one static image has
very little genuine signal to predict. It **is**: given `B_t` and a
candidate/chosen gaze location `a`, predict what the encoder will
produce when it actually looks there.

```text
current belief B_t  +  candidate gaze a  →  predicted local feature
actual observation at a                →  actual local feature
E_t = mismatch(predicted, actual)
```

### Why this design choice does three jobs at once

1. **Mathematically valid by construction.** Predicted and observed
   local features are computed by the *same encoder* — guaranteed
   comparable space, no ad-hoc projection needed to make dimensions
   match.
2. **No new decoder infrastructure.** The backbone's own
   patch-embedding output serves as both the prediction target's
   format and its ground truth.
3. **Unifies with AIS-v2.** The exact same predictor, run on K
   candidate locations before committing to one, *is* AIS-v2's
   candidate-scoring mechanism (predicted error / expected uncertainty
   reduction per candidate). **One predictor module, two consumers** —
   the realized-glimpse error signal for the update, and the
   candidate-scoring signal for gaze selection — not two separate
   mechanisms that could drift out of sync.

## The update equation

```text
z_{t+1} = z_t + Π_t · UpdateNet(z_t, E_t)
```

In plain English: the next belief's content is the current content,
plus a learned transformation of (current content, prediction error),
scaled by precision — how much this evidence should be trusted.

Every symbol:

| Symbol | Meaning | Shape / type |
|---|---|---|
| `z_t` | current global content | `(B, D_z)` |
| `Π_t` | precision factor from `U_t` | scalar or broadcast per-sample |
| `E_t` | latent prediction error at the glimpse's fixation | token-feature space, same space as the predictor's outputs |
| `UpdateNet` | small learned MLP **or** single GRU cell | maps `(z_t, E_t)` into `z`'s update space |

**`UpdateNet` is NOT a raw addition of `E_t`.** Local glimpse-feature
error and the global pooled vector are *not assumed compatible*
without a learned mapping between them — the earlier
`z_t + λ·Π·E_t` draft, which assumed trivial addability, is
superseded.

**Engineering contract:** `UpdateNet` is a small, new, independently
gradient-isolated component — its own optimizer group, its own
pre-flight `|dW|` (gradient-magnitude) check before any smoke test.
This is the standing Gen-0 rule applied from day one rather than
discovered after a starved smoke run.

**First glimpse, t = 0 (LOCKED):** the first glimpse has no
predecessor belief — there is no predict step to compare against, so
`E_0 := 0` (zero tensor, correct shape, no gradient contribution). The
update at t = 0 reduces to `z_1 = z_0 + Π_0 · UpdateNet(z_0, 0)` —
pure observation, no correction term. `E_t` becomes a genuine signal
at t = 1, once a real belief exists to predict from. This is a LOCKED
boundary condition required for the update equation to be well-defined
at all — not an open question, and not agent discretion: Agents E and
F must not choose their own first-glimpse convention.

**Interaction with `S_t`:** none in the core build (`S_t` is None).
If structure returns later, its update follows the same
predict/observe/error/precision/update shape *independently* — not
derived from `z_t`'s `E_t`.

## Gradient rules (the part Gen 0 got wrong repeatedly)

- The **observed** target is **detached** (it is ground truth).
- The **predicted** value is **not detached** (the predictor must
  learn).
- `E_t` itself must **never be detached before use in the update** —
  this exact bug class was the single most repeated failure mode in
  Gen 0's history.

Full context: `13_Gradient_Flow.md`.

## Scientific status

| Claim | Status |
|---|---|
| Latent next-glimpse prediction as `E_t`'s design | **EXPERIMENTAL CANDIDATE** — the best-reasoned default given the constraints, *not* an empirically proven-superior choice (D3's evidence for it is confounded) |
| Pixel reconstruction | **REJECTED for Gen-1** (mechanistically disfavored, statistically inconclusive, engineering cost unjustified) |
| Edge/HPC-style targets | **REJECTED for Gen-1** (no standing justification; comparative evidence confounded) |
| One predictor, two consumers (update + AIS scoring) | **REQUIRED** design rule (single shared predictor; E and AIS must not be independently reimplemented) |
| First-glimpse convention: `E_0 := 0`, no predict step at t = 0 | **LOCKED** boundary condition (well-definedness fill-in for the update equation; not agent discretion) |

### The experiment that resolves the candidate

> A clean, SBR-disabled rerun, once the Gen-1 substrate exists,
> comparing this latent-glimpse target against a **null** (no `E_t` at
> all — the update function `F` reduces to identity), under
> parameter- and compute-matched controls.

Until that runs, any statement stronger than "current design proposes"
is a documentation error.

## What this does NOT mean

- "Prediction error" here does **not** mean reconstructing pixels.
- `E_t` is not a loss function — it is a *state signal* consumed by
  the update; how (and whether) prediction quality is additionally
  supervised by training loss is a Section-8+ matter, outside this
  documentation pass.
- The unification with AIS-v2 does not mean gaze selection is "free" —
  it costs K cheap predictor passes per step, not K backbone passes
  (`10_AIS_v2.md`).
- `E_0 = 0` does not mean prediction is "disabled" — the first glimpse
  simply *is* the observation step; the predict half of the loop
  engages from t = 1 (see the LOCKED t = 0 boundary condition above).

## Open questions

- Exact predictor architecture (small head on the backbone's patch
  embedding — specifics are substrate/implementation-level, outside
  this scope).
- Whether predicted-uncertainty-reduction scoring and realized-error
  updating remain aligned over long training (the reason one shared
  predictor is mandated is precisely to keep them from drifting).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.B — Prediction Error (all rejections, the LOCKED
> DEFAULT, the three-way justification, and the REVISED UPDATE
> EQUATION are captured verbatim in `MASTER_PLAN.md`).

# 10 — AIS-v2: Choosing Where to Look

*Level 1 reading. The active-perception mechanism, its rules, and the
one distinction its status hinges on.*

---

## In one sentence

AIS-v2 is RHAN-NXA's gaze mechanism: it chooses where to look next
based on **expected information** — predicted uncertainty reduction —
using the *same* predictor that powers the belief update.

## The intuition

> If you are trying to identify something hidden in a room, staring at
> the center of the room is not necessarily the best strategy. You
> want to look where the next observation is most likely to resolve
> your uncertainty.

In RHAN-NXA, "where to look" is not a fixed scan pattern or an
afterthought — it is a decision the belief state itself informs: high
uncertainty regions nominate candidates, the shared predictor estimates
what each candidate would reveal, and the most informative one wins.

## A concrete example

Partly-hidden-animal image, after glimpse 1: uncertainty concentrates
around two regions — the obscured head area and an ambiguous flank.
AIS-v2 generates K candidates around those high-uncertainty regions,
asks the predictor "if I looked *here*, what features would I see, and
how much would my uncertainty drop?", and commits to the head-area
candidate that promises the largest drop.

## The pipeline (seven steps)

1. **Identify uncertain regions** — where the belief's uncertainty is
   highest.
2. **Generate K candidate locations** around them.
3. **Predict what each candidate might reveal** — the shared
   glimpse-feature predictor (the *same* module as `E_t`'s predictor).
4. **Estimate uncertainty reduction** per candidate — via `U_t`'s
   **Dirichlet entropy**, not a separate scoring head (one uncertainty
   representation).
5. **Select the next observation.**
6. **Observe it** — the encoder actually looks.
7. **Update the belief** — the normal predict→observe→error→precision
   →update cycle continues.

## Design parameters (as locked)

| Parameter | Value | Notes |
|---|---|---|
| `K` (candidates) | **4–8** | sampled around current highest-uncertainty regions; reuse whatever sampling heuristic the validated STL-10 AIS-v2 implementation used — do not redesign from scratch without cause |
| Scoring | predicted uncertainty reduction | Dirichlet entropy of `U_t`; **no separate scoring head** |
| Scoring at t = 0 | **heuristic-saliency fallback** (same as generation) | **LOCKED** — no `U_0`-conditioned prediction exists to score against yet (`../MASTER_PLAN.md` §1.B; `08_Prediction_Error.md`) |
| Selection, training | **soft** (differentiable — e.g. Gumbel-softmax or a straight-through estimator over the K candidates) | so gradients flow through the choice |
| Selection, inference | **hard argmax** | deterministic at test time |
| Gaze history | enters `A_t` | the record the policy reads and writes (`04_Belief_State.md`) |

## Two hard engineering rules

### 1. The compute constraint

> **K candidate evaluations must use the cheap glimpse-feature
> predictor, NOT K full backbone passes.**

This must be enforced as an **interface constraint**, not left
ambiguous — otherwise candidate scoring becomes prohibitively
expensive at ImageNet scale. One candidate = one cheap predictor pass
over that location's would-be features, never a re-encode of the
image.

### 2. Center-bias is a failure condition

> Track and report the empirical distribution of chosen gaze locations
> across a validation batch; **a policy that always centers is a
> FAILURE CONDITION for the gate — not a passing result with an
> asterisk.**

If the model always looks at the center, it is not demonstrating
meaningful active perception, no matter what its other metrics say.

## The unification with `E_t` (why there is one predictor, not two)

The predictor that scores candidates *is* the predictor whose realized
error becomes `E_t`:

- **before** the glimpse: run on K candidates → scores gaze selection;
- **after** the glimpse: run on the chosen location → prediction vs
  observation → `E_t` → belief update.

One predictor module, two consumers. This is a **REQUIRED** design
rule: the two functions must not be independently reimplemented, because
two separately-trained predictors could drift out of sync — the exact
duplicated-effort split the plan warns against. Details of the
predictor's target space: `08_Prediction_Error.md`.

## Scientific status — the distinction this whole chapter hinges on

| Claim | Status |
|---|---|
| AIS-v2 **mechanism design** (shared predictor, K candidates, uncertainty-reduction scoring, soft/hard selection) | **REQUIRED** |
| AIS-v2's **isolated accuracy/robustness benefit** at 16-seed scale | **PENDING DECISION** — the Gen-0 experiment was confounded |

Why the split: the mechanism-level diagnostic is real and targeted —
candidate-preference correlation **r = 0.706** (512 samples, schema
`ais_v2_smoke_gate_v1`) is graded REQUIRED evidence, and is largely
orthogonal to whether SBR was active. But the confounded 16-seed sweep
(+11.42pp adv vs TRADES, −6.71pp clean vs D) **cannot** be attributed
cleanly to the gaze mechanism, because the runs unintentionally carried
legacy SBR. Therefore:

> **Re-test cleanly once the Gen-1 substrate exists; do not assume the
> confounded +11.42 number as settled fact.**

Full confound accounting: `16_Gen0_Evidence_And_Confounds.md`.

## What this does NOT mean

- AIS-v2 is not claimed to work like human eye movements or saccades;
  "gaze" is a computational choice over candidate locations.
- r = 0.706 does not mean "the model knows where to look" in any
  stronger sense than *its scoring head's predictions correlate with
  its policy choices at 512-sample scale* — the diagnostic's own
  narrow, honest reading.
- Soft selection during training does not mean the model "looks
  everywhere a little bit" at inference; inference is hard argmax.
- The t = 0 heuristic-saliency fallback does not mean scoring is
  heuristic everywhere — from t = 1 on, uncertainty-reduction scoring
  applies as specified.

## Open questions

- The exact candidate-sampling heuristic (inherited from the validated
  STL-10 implementation; do not redesign without cause).
- How center-bias reporting integrates into the Gen-1 gate suite
  (the failure condition is locked; the reporting cadence is a
  Section-8+ matter).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.E — AIS-v2 (all parameters, both engineering rules, and
> the REQUIRED/PENDING status split are captured verbatim in
> `MASTER_PLAN.md`); Part 1.B (predictor unification).

# 25 — Ablation Matrix (Part 3)

*Level 2–3 reading. The complete attribution experiment: ten arms,
their isolations, and the discipline that keeps them comparable.*

---

## In one sentence

The ablation matrix is the full set of comparison runs against the
step-6 reference — each arm flips exactly one flag, each answers one
attribution question, and everything else stays identical.

## The matrix

| Arm | vs. step 6 config, changes | Isolates | Seeds | Must stay identical |
|---|---|---|---|---|
| **Backbone-only** | no recurrence, no belief | Base capacity | 8+ | data, curriculum, warm-start |
| **+Recurrence** | T=4, no belief | Multi-glimpse value alone | 8+ | everything except recurrence flag |
| **+Belief, no F** | identity update | Explicit state + uncertainty alone | 8+ | everything except F flag |
| **+F, no AIS-v2** | fixed/heuristic gaze | Predictive dynamics alone | 8+ | everything except F flag |
| **Full (step 6)** | AIS-v2 active | The complete Gen-1 thesis | **16** | — (this is the reference) |
| **+S_t (2–4 slots)** | vs step 6 | Minimal structure's marginal value | 8+ | everything except S_t flag |
| **+L_stab** | vs step 6 | Stability objective's marginal value **+ responsiveness guard** | 8+ | everything except L_stab flag |
| **+V1 frontend** | vs step 6 | Frontend's marginal value | 8+ | everything except frontend flag |
| **Param-matched control** | backbone-only, width-matched to full | Rules out "more params" as full's explanation | 8+ | total param count |
| **Compute-matched control** | backbone-only, FLOPs-matched to full | Rules out "more compute" as full's explanation | 8+ | total FLOPs |

## How to read it

- **The first four arms** are the ladder's lower rungs (Part 2 steps
  1–4) restated as matrix rows — they exist in both documents because
  they are both build phases *and* ablation arms.
- **Full (step 6) is the reference**, not an arm among equals: every
  other row is a delta against it, with 16 seeds (the protocol's
  full strength) reserved for it.
- **The two control rows are arms too.** They are not formalities:
  Gen 0's Pareto lesson means "more params" and "more compute" are
  *live alternative explanations* for any clean-accuracy gain. Until
  both controls exist, the Full arm's advantage over backbone-only is
  unattributed.
- **+L_stab's isolate column carries the guard**: that arm reports
  drift reduction *and* OOD responsiveness (Gate 9, `21_L_stab_Stability.md`),
  with the two-sided failure rule pre-registered.

## Seed policy (and its rationale)

| Allocation | Seeds | Why |
|---|---|---|
| Every arm by default | **8 minimum** | Gen 0's own finding: TRADES-class baselines carry **higher per-seed variance than mechanism variants** — arms differ in variance needs |
| Reference config (step 6) + ImageNet-1K final run | **16** | The full protocol's strength is reserved for the numbers everything else is compared against |
| Borderline results | extend per-arm | Per the established **no-third-extension discipline** — extend once if borderline, not repeatedly |

## The "must stay identical" discipline (this is Gate 10)

Every arm row's identical-columns is not bookkeeping — it is the
enforcement mechanism for **Gate 10** ("integration doesn't silently
alter unrelated mechanisms"). Any difference between arms other than
the flagged one — data, curriculum, warm-start, seeds policy, eval
protocol — re-introduces a two-variable comparison, which is the
attribution failure this entire plan exists to prevent.

Practical rule for implementers: an arm is configured by starting from
the step-6 reference config and changing **one flag** — never by
assembling a "similar" config from memory. (This is the same
single-source-of-truth principle Agent 0's schema exists to enforce.)

## What the matrix does NOT include

- Any arm vs *another arm* as its primary comparison (only vs step 6;
  see `24_Training_Phase_DAG.md`'s steps-7/8 rule).
- Tier-2 evaluation suites (Brain-Score, new psychophysics) — gated on
  time, Part 2 step 11.
- Any threshold on what a "marginal value" must be to count — that is
  Gate territory (Part 6) and several thresholds are honestly PENDING
  DECISION.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 3 — Ablation Matrix (the table and seed rationale
> captured verbatim in `../MASTER_PLAN.md`); Part 6 (Gate 10's
> enforcement-by-identical-columns reading).

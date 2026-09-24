# 24 — Training Phase DAG (Part 2)

*Level 2–3 reading. Two graphs, kept explicitly separate — and why
conflating them loses attribution.*

---

## Why two graphs

**Build dependency** answers: *what can be implemented and tested in
parallel, right now?*

**Scientific experiment dependency** answers: *in what order must
results be EVALUATED so that each mechanism's contribution is
attributable?*

These are different orders. Parallelizable *construction* does not
license parallelized *evaluation* — the D2/D3 confound (Part 0)
happened precisely because evaluation attribution wasn't protected.

## Graph 1 — Build dependency (parallel-friendly)

```text
Infra (Agent A)
  ├─→ Compact ViT + recurrence (Agent C)      [substrate — buildable
  │                                            independent of belief machinery]
  ├─→ BeliefState scaffold + None-safety
  │   (Agent B)                               [interface + composition
  │                                            logic; testable with
  │                                            dummy tensors — independent of C]
  └─→ EvidentialHead port (Agent D)           [self-contained head —
                                               independent of both]

Once C + B + D exist:
  Unified predictor (glimpse-feature prediction + AIS-v2 candidate
  scoring — Agents E+F coordinate tightly; see Part 4 / `26_`).

L_stab scaffolding (diagnostic-only): any time after B exists —
  it needs only BeliefState.drift_to(), not a trained model.
```

## Graph 2 — Scientific experiment dependency (the evaluation ladder)

Each step adds exactly one thing relative to its predecessor. That is
what makes each rung's comparison attributable.

| Step | Configuration | What it isolates |
|---|---|---|
| 1 | **Backbone-only baseline** (no recurrence, no belief) | Base capacity. **Parameter- and compute-matched controls are established HERE** and reused for every later comparison. |
| 2 | **+ Recurrence only** — T=4 glimpses, fixed, **NO belief update** (glimpses independent, no F) | Does recurrent multi-glimpse processing help AT ALL, before any belief machinery exists? |
| 3 | **+ BeliefState + U_t** (Dirichlet), **NO F** (identity update) | Does an explicit typed belief + calibrated uncertainty help, independent of any predictive dynamics? |
| 4 | **+ F / belief dynamics** (Part 1.B's update, **NO AIS-v2** — gaze fixed/heuristic) | Does predictive updating help independent of active gaze selection? |
| 5 | **+ AIS-v2** — full loop: predict → observe → error → precision → update → gaze-select | The first end-to-end test of the active-perception thesis. |
| 6 | **Integrated system** — still `S_t=None`, L_stab diagnostic-only | **The Generation-1 core result** — the number this whole plan exists to produce. |
| 7 | **+ S_t** (minimal, 2–4 slots) — own isolated arm vs step 6 — **ONLY AFTER 6** | Minimal structure's marginal value. |
| 8 | **+ L_stab** promoted to training objective, own arm vs step 6, responsiveness guard as a hard gate — **ONLY AFTER 6 (not gated on 7)** | The stability objective's marginal value. |
| 9 | **Full ablation matrix** (Part 3) — once steps 1–6 exist cleanly | The complete attribution picture. |
| 10 | **ImageNet-1K final run** — the single best-validated configuration from steps 1–9, **run once, not per-ablation** | The generation's headline result. |
| 11 | **Robustness/human-alignment evaluation suite** (Agent I) — Tier 1 (PGD/AutoAttack subset, ImageNet-C, shape/texture bias) required; Tier 2 (Brain-Score, new psychophysics) only if time remains within the **3-month window** | Honest external evaluation. |

## The steps-7/8 rule (the confound lesson made structural)

> Steps 7 and 8 are **DELIBERATELY parallel-buildable** (build
> dependency) but **NOT parallel-EVALUABLE against step 6
> simultaneously in the same run** — each needs its own clean
> comparison to step 6's frozen result, **never to each other
> directly**, or you reproduce exactly the D2/D3 confound this plan
> opened with.

Concretely: G's structure arm and H's L_stab arm may be *built* at the
same time, but each *evaluation* must run against step 6's frozen
reference checkpoint, in its own run. Comparing arm-vs-arm (or pooling
their runs) would recreate two-mechanism attribution collapse — the
precise failure Part 0 documented.

## Reading the ladder against the failures it prevents

| Ladder property | Failure it prevents |
|---|---|
| Controls established at step 1 | "More params / more compute" explanations discovered too late |
| No-belief recurrence at step 2 | Recurrence's value smuggled into belief-machinery claims |
| Identity-F at step 3 | Belief-container value confused with predictive-dynamics value |
| Heuristic gaze at step 4 | Prediction-error value confused with active-gaze value |
| Step 6 as frozen reference | Arm comparisons against a moving target |
| Own-arm-only comparisons (7/8) | Two-mechanism confounds (the D2/D3 lesson) |
| One ImageNet-1K run (10) | Per-ablation compute inflation and selection noise |

## Scientific status

The DAG itself is **LOCKED** as plan (Part 2). Individual step
*outcomes* remain unknown — nothing in this chapter asserts any
mechanism works; it asserts the order in which claims will be allowed
to exist.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 2 — Revised Training Phase DAG (both graphs, all 11
> steps, and the steps-7/8 rule captured verbatim in
> `../MASTER_PLAN.md`).

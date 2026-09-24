# 28 — Gates, Compute Accounting, and Scope (Part 6)

*Level 2–3 reading. How Gen-1 experiments are measured, bounded, and
honestly gated — including which thresholds deliberately don't exist
yet.*

---

## Compute accounting (automatic, not hand-computed)

Every experiment in the DAG (Part 2) and ablation matrix (Part 3)
reports:

| Field | Notes |
|---|---|
| Param count | **total + trainable** (the distinction matters: the V1 frontend and any fixed component count in FLOPs but not trainable params) |
| FLOPs/image | the compute-matched control's currency |
| Latency | wall-clock per image |
| Peak memory | training and eval |
| Glimpse count | T, and any effective variation |
| Effective compute/image | the combined cost of the perception loop (incl. K cheap predictor passes per step) |

These are **logged automatically by Agent A's infrastructure** — never
hand-computed per experiment. Hand-computed accounting is exactly the
kind of drift Agent 0's single-schema rule exists to eliminate.

## The gates (with honest threshold statuses)

| Gate | What it checks | Status |
|---|---|---|
| **G6** | Uncertainty correlates with error/ambiguity | **PENDING DECISION** — no threshold justified yet; **report the correlation, don't gate on an arbitrary cutoff** until one experiment exists to calibrate against |
| **G7** | AIS-v2 avoids trivial center-bias | **REQUIRED** check; threshold **PENDING DECISION** (same reasoning). The failure condition itself is already locked: a policy that always centers is a FAILURE, not a passing result with an asterisk (`10_AIS_v2.md`) |
| **G8** | `S_t` contains measurable structure | **N/A** while `S_t=None`; **PENDING DECISION** once Part 3's structure arm runs |
| **G9** | L_stab reduces adversarial drift **without collapsing OOD responsiveness** | **REQUIRED as designed** (Part 1.G); the responsiveness-floor threshold is **PENDING DECISION**, calibrated from the diagnostic-only phase's own baseline distribution **before L_stab ever becomes a loss** |
| **G10** | Integration doesn't silently alter unrelated mechanisms | **REQUIRED** — enforced by the ablation matrix's "everything else stays identical" columns, **not a separate check** (`25_Ablation_Matrix.md`) |

### Why "PENDING DECISION" thresholds are a feature

A gate with an invented threshold is a gate that can be failed or
passed arbitrarily — Gen 0's sbr0 gate passed while measuring
near-chance slot content because its floor couldn't detect vacuity
(`16_Gen0_Evidence_And_Confounds.md`). The plan's rule: **calibrate
thresholds from measured baseline distributions** (one experiment's
worth), then gate. Until then, report the quantity; don't pretend the
cutoff exists.

The two REQUIRED gates (G9, G10) are required *in design* — their
structure is fixed — while their numeric thresholds are the parts
still pending calibration.

## Scope boundaries (Section 21, condensed)

| Direction | Disposition | Reason |
|---|---|---|
| **3D / depth** | **DEFERRED** | No 2D evidence yet suggests depth is limiting |
| **Video / temporal** | **DEFERRED** | Preserved conceptually via the existing `frame_t → B_t → predict → frame_{t+1}` formulation; revisited only if a small temporal probe shows occlusion/persistence is a real gap |
| **Relational graphs** | **DEFERRED, with a precondition** | Contingent on **first showing** appearance+structure (once `S_t` is revisited at all) is insufficient — **do not build before that negative result exists** |

Note the asymmetry in the last row: relational graphs are not merely
waiting for time — they are waiting for a *specific negative result*
to exist first. That is the strongest deferral in the plan.

## The honesty norms this chapter encodes

1. Every gate states its threshold status — REQUIRED / PENDING
   DECISION / N-A — and never dresses a guess as a cutoff.
2. Compute accounting is infrastructure, not per-experiment prose.
3. Deferrals name their re-entry condition; rejections name their
   evidence.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 6 — Compute Accounting, Gates, Scope (the gate table,
> scope dispositions, and literature-classification preamble captured
> verbatim in `../MASTER_PLAN.md`).

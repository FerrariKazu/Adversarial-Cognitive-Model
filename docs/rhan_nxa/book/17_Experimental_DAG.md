# Chapter 17 — Experimental DAG: Staged Isolation Protocol

## 1. In one sentence
The RHAN-NXA experiment plan is structured as a two-graph DAG: a **build dependency graph** (what must exist before what can be built) and a **scientific experiment dependency graph** (Steps 1–11, the validation ladder), where every arm's primary comparison is against a **frozen Step-6 reference**, never against another arm.

## 2. The isolation principle
The defining failure of Generation-0 was that D2 (AIS-v2 test) and D3 (belief-HPC test) both ran with unintentionally-active SBR, making their results **uninterpretable as isolated tests**. The Gen-1 protocol was designed specifically to prevent this:

> **Every ablation arm flips exactly one flag versus the Step-6 frozen reference. Every arm's primary comparison is to Step-6, never to another arm.**

Comparing two arms against each other reintroduces multi-mechanism attribution collapse—exactly the D2/D3 confound reproduced.

## 3. The Step-6 frozen reference
Step 6 is the **integrated Gen-1 core**:
- $S_t = \text{None}$ (no SBR)
- $L_\text{stab}$ diagnostic-only (not an objective)
- AIS-v2 active
- Full belief loop: $z_t$, $U_t$, $E_t$, $\Pi_t$, $A_t$

Step-6 must produce a validated ImageNet-100 result before any ablation arm begins evaluation. This result is frozen; ablation arms compare against it.

## 4. The eleven-step validation ladder

| Step | What it tests | Primary comparison |
|:---:|:---|:---|
| 1 | Backbone only (no belief loop) | TRADES baseline |
| 2 | Backbone + within-glimpse recurrence | Step 1 |
| 3 | + Belief state (no $E_t$) | Step 2 |
| 4 | + Prediction error $E_t$ (no AIS-v2) | Step 3 |
| 5 | + AIS-v2 (no $L_\text{stab}$) | Step 4 |
| **6** | **Full core** ($S_t=\text{None}$, $L_\text{stab}$ diagnostic) | Step 5 |
| 7 | + $S_t$ (2–4 slots, gated) | Step 6 |
| 8 | + $L_\text{stab}$ as objective (gated) | Step 6 |
| 9 | + V1 Gabor frontend (standalone ablation first) | Step 6 |
| 10 | Param-matched backbone-only control | Step 6 |
| 11 | Compute-matched backbone-only control | Step 6 |

Steps 7 and 8 can be **built in parallel** (they share Step-6's substrate), but must be **evaluated independently against Step-6**, not against each other.

## 5. Gate structure
Each step has pre-registered pass/fail criteria:

| Gate | Criterion | Status |
|:---:|:---|:---:|
| G6 | Uncertainty–error correlation (Pearson $r$ threshold, PENDING) | PENDING |
| G7 | Center-bias: gaze distribution not center-concentrated (threshold PENDING) | REQUIRED |
| G8 | Structure gate (per-slot decodability, N/A while $S_t=\text{None}$) | N/A |
| G9 | Drift without collapse: $L_\text{stab}$ reduction alongside OOD responsiveness | REQUIRED as designed |
| G10 | No silent alteration: all arms have identical ablation columns | REQUIRED |

**Gate 9 critical rule**: low adversarial drift achieved by becoming insensitive to all new evidence is a **FAILURE**, not a success. Drift reduction must be reported alongside an OOD/novel-evidence responsiveness score on a disjoint probe set. Low drift on both simultaneously = FAILED.

## 6. Matched controls (mandatory for mechanism claims)
Before claiming any mechanism improves robustness, two controls must pass:

- **Parameter-matched**: backbone-only run with total parameters equal to the full model. Excludes "more parameters helped."
- **Compute-matched**: backbone-only run with FLOPs equal to the full model (via width expansion). Excludes "more compute helped."

The Gen-0 SBR rungs proved this failure mode is live: their clean accuracy gains tracked added capacity, not structural mechanism.

## 7. The confound documentation requirement
Every experiment that touches a mechanism that was active in Gen-0's D2/D3 (AIS-v2, belief-HPC) must document the confound clearly and run a clean re-test. The `_nx_trainer` SBR bug must not be silently forgotten.

## 8. Scientific status
- **Step-6 as frozen reference**: **REQUIRED** (isolation protocol).
- **Matched controls**: **REQUIRED** (standing rule from Gen-0 Pareto lesson).
- **Per-arm flag isolation**: **REQUIRED** (no arm may swap two flags simultaneously).

## 9. Source references
- `noesis_vision/RHAN_NXA/docs/24_Training_Phase_DAG.md`
- `noesis_vision/RHAN_NXA/docs/25_Ablation_Matrix.md`
- `noesis_vision/RHAN_NXA/docs/28_Gates_and_Compute_Accounting.md`
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `GATED_FLAGS`, `step6_validated_result`
- Illustrated in Figure 9 (`experiment_isolation.svg`).
- Connected to Chapters 16 (training system) and 19 (Gen-0 evidence).

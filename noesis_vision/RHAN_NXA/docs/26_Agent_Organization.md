# 26 — Multi-Agent Organization & Ownership (Part 4)

*Level 2–3 reading. Who builds what, in what order, under which
contracts.*

---

## In one sentence

Eleven agents (0–J) with non-overlapping ownership build RHAN-NXA;
two explicit adjustments — a shared predictor contract for E+F and a
hard gate on Agent G — keep the decomposition from recreating the
confounds the plan exists to prevent.

## The dependency graph

```text
           Agent 0 (Spec/Interfaces — runs first, always)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Agent A         Agent B         Agent D
        (Infra)      (BeliefState)   (Uncertainty)
              │             │             │
              └──────┬──────┴──────┬──────┘
                     ▼             ▼
                Agent C      [Agents E+F, sharing
              (Recurrent      Agent 0's predictor
               ViT core)      interface contract]
                     │             │
                     └──────┬──────┘
                            ▼
                    Agent J (Integration) — produces
                    the step-6 reference checkpoint
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Agent G          Agent H       Agent I
     (Structure —      (Objectives —  (Evaluation)
      GATED on J's      L_stab, staged
      step-6 result)    per Part 1.G)
```

## Ownership table

| Agent | Owns | Key constraints |
|---|---|---|
| **Agent 0** | Canonical spec, interfaces, dependency graph, contract template (see `30_Agent0_Interface_Contract.md`) | Spec-only; no implementation logic |
| **Agent A** | Infrastructure (optimizer groups, checkpoint/resume, logging incl. Part-6 compute accounting) | Everything downstream reuses, never re-implements |
| **Agent B** | BeliefState scaffold + None-safety | Testable with dummy tensors; independent of C |
| **Agent C** | Compact ViT + recurrence (substrate) | Independent of belief machinery |
| **Agent D** | EvidentialHead **port** (not reimplementation) | Self-contained head |
| **Agent E** | Predictor module + belief-update network (`UpdateNet`) | **Shares Agent 0's predictor interface — may not implement its own copy** |
| **Agent F** | Candidate generation/selection, gaze-state bookkeeping (AIS-v2) | **Same shared predictor contract — same prohibition** |
| **Agent G** | Structural representation (S_t) | **EXPLICITLY GATED: does not start until Agent J confirms step 6's validated result**; the gate is a literal precondition checked before writing any code |
| **Agent H** | Objectives (L_stab), staged per Part 1.G | Diagnostic-only until promotion conditions met; owns the responsiveness-guard evaluation |
| **Agent I** | Evaluation suite (Tier 1 required; Tier 2 time-gated), incl. ECE calibration | 3-month window discipline (Part 2 step 11) |
| **Agent J** | Integration — produces **the step-6 reference checkpoint** | Its validated result is the gate for G and the reference for the matrix |

## The two adjustments (why this isn't a naive decomposition)

### Adjustment 1 — Agents E and F share ONE interface contract

Part 1.B unified the predictor (one module, two consumers). The
agentic decomposition must preserve that unification:

- E and F are **NOT merged** — the work is separable (F: candidate
  generation/selection/gaze bookkeeping; E: predictor module +
  belief-update network).
- But **Agent 0's spec defines the predictor's exact call signature
  ONCE**, and both E and F consume it identically.
- **Neither may implement its own copy.**

This is the documentation-level rule from `08_Prediction_Error.md`
and `10_AIS_v2.md` turned into an organizational invariant: the thing
that prevents predictor drift is a single interface artifact, not good
intentions.

### Adjustment 2 — Agent G is explicitly gated

Agent G (Structure) **does not start until Agent J confirms step 6 has
a validated result**. Agent 0's spec must record this gate as a
**literal precondition Agent G checks before writing any code** — not
an understanding, not a norm: a checkable precondition. (Mirrors the
S_t re-entry rules in `06_Structure_State.md`.)

## The 17-point contract template

Every agent's prompt uses this template **verbatim**. The plan names
it a *17-point* template and enumerates these points explicitly:

1. Mission
2. Scientific purpose
3. Inputs
4. Outputs
5. Files it may modify
6. Files it must not modify
7. Existing infrastructure to reuse
8. API/interface contract
9. Tensor shapes
10. Gradient requirements
11. Tests
12. Smoke experiment
13. Acceptance criteria
14. Failure conditions (stop and report — do not improvise)
15. Scientific interpretation
16. Commit/PR boundary
17. Handoff artifact

Plus: the **explicit non-improvisation clause appears in every single
prompt** — not just Agent 0's. "Stop and report" is the default
behavior on ambiguity; guessing a reasonable-sounding default is the
documented anti-pattern (it is how Gen-0's interface drift started).

## Sequencing discipline

Agent 0 runs first, **always** — every other agent's interface contract
depends on what it produces. Agents A–J follow in subsequent turns,
**each gated on review of the prior one's handoff artifact**.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 4 — Multi-Agent Dependency Graph & Ownership (graph,
> both adjustments, template, and non-improvisation clause captured
> verbatim in `../MASTER_PLAN.md`); the Agent 0 contract (see
> `30_Agent0_Interface_Contract.md`).

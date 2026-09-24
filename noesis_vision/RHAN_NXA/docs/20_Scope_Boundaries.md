# 20 — Scope Boundaries

*Level 2–3 reading. What is explicitly OUTSIDE the Gen-1 core, so
nobody implements things merely because they sound useful.*

---

## Why this chapter exists

The most expensive code is code that shouldn't exist. Every item below
has a documented reason for being out of scope **now**; several have
pre-registered conditions for returning. If you are about to build
something absent from the core chapters (`04`–`10`), check here first.

## Explicitly outside the current Gen-1 scope

| Item | Status | Reason / re-entry condition | Reference |
|---|---|---|---|
| Representation-level uncertainty (uncertainty over `z_t`/`S_t` content) | **UNRESOLVED — PENDING DECISION** | no existing training target; would require new machinery mid-build | `07_Uncertainty.md`, DR-8/DR-9 |
| Episodic memory (across images) | **DEFERRED** | no work until a specific experiment shows temporal persistence is a limiting capability | `22_Memory.md` |
| External retrieval / RAG | **REJECTED outright** | no project evidence motivates it; would exist only because "memory" is in the name | `22_Memory.md` |
| Temporal / video persistence (belief across images or time) | **DEFERRED** | preserved conceptually via `frame_t → B_t → predict → frame_{t+1}`; revisited only if a temporal probe shows occlusion/persistence is a real gap | `04_Belief_State.md`, `22_Memory.md` |
| Structural representation (slots, objects) | **EXPERIMENTAL CANDIDATE, DEFERRED** | waits on the None-core validating on ImageNet-100; re-entry rules pre-registered (2–4 slots; distinguishability gates) | `06_Structure_State.md`, DR-1 |
| Adaptive halting | **DEFERRED** | documented history of fighting other objectives; core loop validates first at fixed T | `09_Recurrence.md`, DR-6 |
| Pixel reconstruction as `E_t` target | **REJECTED for Gen-1** | mechanistically disfavored, statistically inconclusive, engineering cost unjustified | `08_Prediction_Error.md`, DR-2 |
| Edge-map / HPC-style hand-designed targets | **REJECTED for Gen-1** | own extractor, foreign space, confounded comparative evidence | `08_Prediction_Error.md` |
| The Gen-0 16-slot SBR implementation | **REJECTED** | retained-ablation 1.0157; clean-collapse signature; gates unable to detect vacuity | `06_Structure_State.md`, DR-10 |
| AIS-v1 relocated-Eq.-II gaze code | **REJECTED (do not port)** | superseded by AIS-v2's design; not even as a fallback path | `27_Infrastructure_Port_Table.md` |
| 3D / depth | **DEFERRED** | no 2D evidence yet suggests depth is limiting | `28_Gates_and_Compute_Accounting.md` |
| Relational graphs | **DEFERRED with precondition** | contingent on first showing appearance+structure is insufficient — the negative result must exist first | `28_Gates_and_Compute_Accounting.md` |
| `L_stab` as a training objective | **EXPERIMENTAL CANDIDATE, staged** | diagnostic-only until the core (without it) has a validated ImageNet-100 result; responsiveness guard is a hard gate | `21_L_stab_Stability.md` |
| V1 frontend in integrated configs | **EXPERIMENTAL CANDIDATE, sequenced last** | standalone ablation required before folding into any integrated configuration | `23_V1_Frontend.md` |
| Agents A–J prompts | **NOT YET WRITTEN** | gated on review of Agent 0's handoff artifact; only Agent 0's contract is authorized | `30_Agent0_Interface_Contract.md` |
| Detailed ImageNet-1K training hyperparameters | **NOT SPECIFIED** | only the run-once rule (Part 2 step 10) is specified | `24_Training_Phase_DAG.md` |

## The do-not-implement list (documentation-task boundary)

This documentation pass does **not** include, and no reader should
infer from it:

- model classes or training loops (RHAN-NXA is specified, not
  implemented);
- porting Gen-0 code into Gen-1 except through the Part-5 port table
  (`27_Infrastructure_Port_Table.md`); `rhan_core/` stays historical
  reference only;
- any mechanism whose chapter status is DEFERRED, PENDING DECISION, or
  REJECTED;
- numeric gate thresholds the plan marks PENDING DECISION (G6's
  correlation cutoff, G7's threshold, G9's responsiveness floor) —
  they are calibrated from measured baseline distributions, not
  invented;
- numeric values the plan does not fix (`D_z`, token grids, crop
  sizes, iteration counts) — marked UNKNOWN in `14_Tensor_Shape_Reference.md`;
- new mechanisms, new losses, or "improvements" not present in
  `../MASTER_PLAN.md`.

## What IS in scope (the whole core, for contrast)

The belief state and its None-propagating contract; `z_t`; the
class-readout Dirichlet `U_t`; latent next-glimpse `E_t` with
`UpdateNet`; tied within-glimpse recurrence; fixed T = 4
across-glimpse recurrence; the shared predictor; AIS-v2 as specified
(K = 4–8, soft/hard selection, compute constraint, center-bias failure
condition); `A_t` as a non-learned record; the gradient contract;
the controls rule for all mechanism claims.

That is the entire Gen-1 core. If a proposed component is not on this
list and not in `15_Status_And_Decision_System.md`'s table, it is out
of scope until the plan says otherwise.

---

> **Source decision:** noesis_vision/RHAN_NXA/MASTER_PLAN.md (scope of authorized
> material: Part 0, Parts 1.A–1.I, Parts 2–6, and the Agent 0
> contract; the out-of-scope table restates decisions documented in
> those parts; nothing here adds new architecture).

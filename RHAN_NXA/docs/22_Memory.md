# 22 — Memory (Section 9): What Counts, What Waits, What's Rejected

*Level 2 reading. Mostly a chapter about dispositions — including one
outright rejection.*

---

## In one sentence

RHAN-NXA has exactly two forms of memory in Gen-1 — the model's own
weights and the within-image belief trajectory — with episodic
persistence **deferred** and external retrieval **rejected outright**.

## The four dispositions

### 1. Parametric memory `θ` — the model weights

**Always present, trivially.** Everything the model learns across
images and epochs lives in its parameters: the backbone, the tied
refinement block, the shared predictor, `UpdateNet`, the evidential
head. This is the only cross-experience memory Gen-1 has, and it needs
no mechanism beyond training.

### 2. Recurrent working state `h_t` — the within-image trajectory

**Already covered — not a separate thing.** The within-image glimpse
trajectory *is* `B_t` evolving across T steps: `B_1 → B_2 → B_3 → B_4`.
Gen-1 does not add a parallel "working memory" module; the belief
state is the working memory. Any implementation that adds a separate
hidden-state stream alongside `B_t` is duplicating the belief state
and should be treated as a specification violation.

Consequence (restated from `04_Belief_State.md`): at the end of each
image's forward pass, this working state is discarded. There is no
`B_t` persistence across images in Gen-1 at all.

### 3. Episodic / temporal persistence ACROSS images — DEFERRED

Matches the project's existing framing: *"video is deferred pending a
temporal experiment."* No work until a **specific experiment shows
temporal persistence is a limiting capability**.

What this preserves conceptually (per Part 6): the formulation
`frame_t → B_t → predict → frame_{t+1}` remains the conceptual shape
a future temporal extension would take — the belief-prediction loop
already *looks like* a temporal model, which is precisely why it is
tempting to build one now. The deferral exists to stop that temptation
before evidence justifies it.

### 4. External retrieval / RAG — REJECTED outright

The plan's reasoning, preserved: **no evidence anywhere in this
project's history motivates it.** Introducing retrieval would be
adopting a mechanism because the word "memory" appears in the
architecture's name — exactly the trap the plan is written to avoid.

This is the strongest memory disposition in the plan: not "wait for
evidence" (deferral) but "the idea has no standing justification at
all" (rejection). If retrieval is ever proposed again, the proposal
must start by citing a project-specific failure that retrieval would
address.

## Why this chapter is short

Memory is where architecture bloat goes to hide: "memory" sounds like
intelligence, so unprincipled builds grow retrieval buffers, replay
stores, and external indexes with no causal story. RHAN-NXA's answer
is a decision table, not a mechanism:

| Form | Disposition | Status |
|---|---|---|
| Parametric `θ` (weights) | in use — it is what training trains | REQUIRED (trivially) |
| Working state `h_t` (belief trajectory) | in use — it *is* `B_t` across T | REQUIRED (not a separate module) |
| Episodic/temporal persistence across images | deferred pending a temporal-experiment motivation | DEFERRED |
| External retrieval / RAG | no standing justification | REJECTED outright |

## What this does NOT mean

- Deferring episodic memory is not a claim that memory is useless for
  perception in general — it is a claim that Gen-1's experiments do
  not require it.
- The RAG rejection is not about RAG's merits in other domains; it is
  about the absence of any motivating evidence in *this* project.

## Open questions

- What experiment would demonstrate temporal persistence as a
  *limiting* capability? (Unspecified — the plan requires its result
  before any temporal work begins.)

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.H — Memory (all four dispositions verbatim in
> `../MASTER_PLAN.md`); Part 6 (temporal scope, `frame_t → B_t →
> predict → frame_{t+1}` preservation).

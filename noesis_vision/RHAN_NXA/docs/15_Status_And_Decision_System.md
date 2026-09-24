# 15 — Status and Decision System

*Level 2 reading. The labeling system every other chapter uses — and
the rule that keeps hypotheses from hardening into "facts" in prose.*

---

## Why a status system exists

The most dangerous failure mode of a research codebase is silent
status drift: a mechanism that started as "we're trying this" gets
described in one README as "we use this," then in a talk as "this
works," and six months later nobody can tell which claims were ever
tested. RHAN-NXA's documentation prevents this by attaching an
explicit status to **every** component, everywhere, permanently.

## The five status labels

### REQUIRED
**Meaning:** evidence or architecture dependency currently requires
it.

Use when: the component is load-bearing for the Gen-1 build — either
because the architecture is *defined* by it (the belief state), or
because gate-verified evidence demands it (the AIS-v2 design rule),
or because it is the locked default configuration (`S_t = None`).

**Test:** "If I removed this, would the documented Gen-1 system be a
different system?" If yes → REQUIRED (as specified).

### PENDING DECISION
**Meaning:** an experiment is required before the design can be
finalized.

Use when: a specific, named experiment would resolve the question and
hasn't run. Example: AIS-v2's isolated 16-seed accuracy/robustness
contribution — resolvable by a clean, SBR-disabled rerun.

**Test:** "Can I name the experiment that resolves this?" If yes but
it hasn't run → PENDING DECISION. If no experiment exists even in
principle yet, it is not pending — it is an open question recorded in
the relevant chapter.

### EXPERIMENTAL CANDIDATE
**Meaning:** plausible and currently being tested / carried as the
default, but not established.

Use when: the design *adopts* something as the best-reasoned default
while explicitly acknowledging the evidence is insufficient to call it
superior. Example: latent next-glimpse prediction as `E_t` — carried
as the design default, with its resolving experiment pre-registered.

**Test:** "Would a competent reviewer let me assert this works?" If
not, but we build with it anyway → EXPERIMENTAL CANDIDATE.

### DEFERRED
**Meaning:** potentially useful but intentionally outside the current
Gen-1 scope.

Use when: the mechanism has a plausible future but a documented reason
to wait. Examples: adaptive halting (documented history of fighting
other objectives); structural representation (waits on the None-core
validating first).

**Test:** "Is there a stated re-entry condition?" DEFERRED items
should have one.

### REJECTED
**Meaning:** current evidence argues against the
implementation/concept *as currently tested*.

Use when: gate-verified or statistically-reported evidence counts
against it. Examples: pixel reconstruction (for Gen-1), edge/HPC
targets (for Gen-1), the 16-slot SBR implementation.

**Test:** "Is the negative evidence cited?" A REJECTED label without a
cited reason is incomplete.

> **REJECTED IMPLEMENTATION ≠ REJECTED IDEA.**
> The current 16-slot SBR implementation is rejected. That does NOT
> mean "RHAN-NXA does not need structural representation." It means:
> "The tested implementation did not provide sufficient evidence and
> currently should not be part of the core." This distinction must
> appear wherever SBR is discussed.

## The three claim types

Throughout this documentation, every statement is one of:

| Type | Meaning | Example |
|---|---|---|
| **FACT** | established by the current specification or experiment | "the Gen-1 core runs T = 4 fixed glimpses" |
| **DESIGN DECISION** | deliberately chosen for Gen-1, not scientifically proven superior | "within-glimpse refinement uses tied weights" |
| **HYPOTHESIS** | something RHAN-NXA is testing | "belief-driven sequential perception improves robustness" |

Forbidden transformations:

- never turn "we chose X because it is our best current hypothesis"
  into "X is how human perception works";
- never turn "this component is required by the current architecture"
  into "experiments have proven this component improves performance."

## Quick-reference table (the only summary table in this doc set)

| Component | Simple purpose | Scientific purpose | Current status |
|---|---|---|---|
| Belief state `B_t` | the model's notebook about the image | makes perception a stateful, inspectable process | REQUIRED |
| `z_t` | what the model thinks it sees | carries global content across glimpses | REQUIRED |
| `S_t` | optional structure | would carry object/relationship state | None = REQUIRED (default); concept EXPERIMENTAL CANDIDATE, DEFERRED; 16-slot impl REJECTED |
| `U_t` | how uncertain the model is | drives precision weighting and gaze scoring | REQUIRED (class-readout; compromise documented) |
| `E_t` | how surprising the evidence was | the belief's internal learning signal | EXPERIMENTAL CANDIDATE |
| `UpdateNet` | translates error into belief change | learned (not assumed) belief dynamics | part of the `E_t` candidate |
| `A_t` | where the model has looked | enables history-aware gaze policy | REQUIRED (as a non-learned record) |
| Within-glimpse tied recurrence | refine tokens before pooling | better per-glimpse representation at fixed params | DESIGN DECISION (LOCKED as design) |
| Across-glimpse recurrence (T=4) | multiple looks | the investigation loop's trip count | REQUIRED for first build |
| Adaptive halting | stop when confident | adaptive compute | DEFERRED |
| Shared glimpse predictor | predict what a look reveals | unifies E_t and AIS scoring | REQUIRED as design rule |
| AIS-v2 | choose where to look | targeted evidence gathering | design REQUIRED; isolated benefit PENDING DECISION |
| Pixel reconstruction | predict pixels | (was: dense self-supervision) | REJECTED for Gen-1 |
| Edge/HPC-style targets | predict fixed features | (was: bio-inspired target) | REJECTED for Gen-1 |
| 16-slot SBR | explicit slots | (was: structural representation) | REJECTED (implementation only) |
| Representation-level uncertainty | uncertainty over belief content | belief-centric confidence | PENDING DECISION (deferred past first build) |
| `L_stab` | belief-trajectory stability under perturbation | robustness without perceptual collapse | EXPERIMENTAL CANDIDATE, explicitly staged (diagnostic → objective) |
| L_stab responsiveness guard | stability must not become insensitivity | separates robustness from evidence-blindness | REQUIRED as designed (threshold PENDING DECISION) |
| Episodic/temporal memory across images | remember between images | persistence of belief over time | DEFERRED (pending a temporal-experiment motivation) |
| External retrieval / RAG | fetch external evidence | (none demonstrated) | REJECTED outright |
| V1 frontend | fixed Gabor input filters | frequency-structured first stage | EXPERIMENTAL CANDIDATE (built LAST; standalone ablation required) |
| Training phase DAG (steps 1–11) | the evaluation ladder | attribution-preserving ordering | LOCKED as plan |
| Ablation matrix (10 arms) | one-flag-at-a-time comparisons | mechanism-vs-capacity attribution | LOCKED as plan |
| Agent organization (0–J) | ownership and contracts | drift-proof parallel construction | LOCKED as plan (G gated on J's step-6 result) |
| Infrastructure port table | PORT/ADAPT/REJECT dispositions | validated-reuse discipline | LOCKED as plan |
| Gates G6–G10 | calibrated experiment checks | detect function, not vacuity | REQUIRED as designed; numeric thresholds PENDING DECISION where marked |

## Usage rules for writers of any new RHAN-NXA document

1. Every component gets exactly one status **everywhere** it appears;
   if two documents disagree, one of them is a bug (see the audit
   checklist in `00_README.md` and the source plan §32).
2. "LOCKED" is a plan-level word for design decisions (e.g. recurrence
   Option C). It never overrides the scientific statuses above —
   "LOCKED design" and "EXPERIMENTAL CANDIDATE empirically" can both
   be true of the same component (they are, for `E_t`'s design space
   and the hybrid recurrence's *evidence* status).
3. Negative results keep their exact significance language: NOT
   significant stays NOT significant; UNKNOWN stays UNKNOWN.
4. When citing Gen-0 numbers, name their provenance
   (`16_Gen0_Evidence_And_Confounds.md`) and whether the confound
   applies.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan (status taxonomy and the FACT/DESIGN/HYPOTHESIS distinction as
> specified in the documentation task; per-component statuses from
> Parts 1.A–1.F).

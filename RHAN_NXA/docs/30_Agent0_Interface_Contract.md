# 30 — Agent 0: Canonical Interface Contract

*Level 3 reading. The first agent's full contract — the artifact every
other agent imports against. Source text preserved verbatim in
`../MASTER_PLAN.md`.*

---

## In one sentence

Agent 0 owns the canonical RHAN-NXA interface specification — one
versioned schema plus stub-only interface files that every other agent
imports and implements against — and exists to make Gen-0's interface
drift *structurally impossible*.

## Mission (as specified)

> You own the canonical RHAN-NXA interface specification. You do not
> write model code, training loops, or loss functions. You produce ONE
> artifact: a versioned interface/schema document plus the
> corresponding (empty-bodied, type-annotated, docstring-only) Python
> interface stubs that every other agent will import and implement
> against. Every ambiguity marked PENDING DECISION or EXPERIMENTAL
> CANDIDATE gets encoded as an explicit config flag with a documented
> default and a documented "this is not yet decided" comment — never
> silently resolved.

## Scientific purpose

> Generation 0 lost real time to interface drift: multiple eval
> scripts disagreeing on epsilon convention, a runner hardcoding a
> flag that silently confounded two "isolated" experiments (Part 0).
> Agent 0 exists specifically to make that class of failure
> structurally impossible for Generation 1 — one schema, one source of
> truth, every downstream agent imports it rather than reimplementing
> its own interpretation.

## Inputs

This master plan (Parts 0–6). **Nothing else** — no external sources,
no invented requirements.

## The five output artifacts

| # | File | Contents |
|---|---|---|
| 1 | `noesis_vision/core/schema.py` | **RHANNXAConfig** — dataclass, versioned via a `schema_version` field; every locked decision AND every PENDING DECISION / EXPERIMENTAL CANDIDATE flag from Part 1, each with its status as a code comment (e.g. `enable_sbr: bool = False  # LOCKED default per Part 1.D; PENDING DECISION if ever set True — see gate requirement`) |
| 2 | `noesis_vision/beliefs/interfaces.py` | **BeliefState ABC** — Part 1.A shapes exactly; stub-only |
| 3 | `noesis_vision/predictive_coding/interfaces.py` | **The SHARED predictor interface** consumed by both Agent E and Agent F (Parts 1.B/E, Adjustment 1) — *the single most load-bearing artifact in the task*, since it prevents E and F from diverging |
| 4 | `noesis_vision/core/dependency_graph.md` | The Part 2/4 graphs, in-repo, as the literal source later agents (and future sessions) check before starting work |
| 5 | `noesis_vision/core/agent_contract_template.md` | The 17-point template with placeholders, for Agents A–J to inherit |

## File boundaries

**May modify:** only the five files above, newly created. Nothing in
an existing directory — there is no existing `noesis_vision/` at this
stage; **if one exists from an earlier, less-audited pass, STOP and
report the discrepancy rather than overwriting it.**

**Must not modify:** anything under the STL-10 project's `rhan_core/`
— historical reference only, per the no-parallel-port rule
(`27_Infrastructure_Port_Table.md`).

**Infrastructure to reuse:** none at this stage — Agent 0 creates the
interfaces other agents will reuse infrastructure against. Part 5's
port table is *informational* for the spec (e.g., schema.py's
optimizer-group config shape should match what the ported multi-group
optimizer expects), not something Agent 0 implements.

## Interface contract

Exactly the shapes and signatures specified in:

- **Part 1.A** — BeliefState (`B_t = (z_t, S_t, U_t, E_t, A_t)` and
  its typing)
- **Part 1.B** — the predictor: takes current belief + candidate
  location(s), returns **predicted glimpse features + a per-candidate
  scalar score**
- **Part 1.C** — recurrence config: shared-weight flag, iteration
  count, glimpse count T

> Do not add fields not present in this plan without flagging them as
> a question back rather than deciding silently.

## Tensor shapes

Part 1.A verbatim (`14_Tensor_Shape_Reference.md`). Every stub
function signature must have **shapes in its docstring**, not just
types.

## Gradient requirements

N/A directly (stub-only, no implementation) — but every interface must
be **designed so gradient flow is checkable by construction**: e.g.,
no interface that returns a detached tensor by default from a method
whose whole purpose is to carry gradient.

## Tests (specified by the contract)

**`tests/test_schema_version.py`:**
- config round-trips through serialization;
- every PENDING DECISION flag defaults to its documented safe value
  (False/None);
- attempting to set a PENDING-DECISION-gated flag True **without its
  documented prerequisite** (e.g. `enable_sbr=True` without step-6
  validation recorded) **raises, not silently succeeds**.

**`tests/test_interface_imports.py`:**
- every stub file imports cleanly;
- every ABC method raises `NotImplementedError` **with a message
  naming which future agent is responsible for it**.

## Smoke experiment

Import every file created; instantiate RHANNXAConfig with defaults;
call every stub method once — all should raise NotImplementedError
cleanly, and **none** should raise an unrelated error (shape mismatch,
missing import).

## Acceptance criteria

- All five files exist.
- Both test files pass.
- Every **locked decision** from Part 1 appears as a config field.
- Every **PENDING DECISION** appears as a config field with its
  gating comment.
- The dependency-graph markdown matches Part 2/4 **exactly** (no
  reordering, no simplification).

## Failure conditions — STOP AND REPORT, DO NOT IMPROVISE

- If any part of Part 1's audit is genuinely ambiguous about a shape
  or signature not resolved above: **STOP and list the specific
  ambiguity** rather than guessing a reasonable-sounding default.
- If an existing `noesis_vision/` directory exists with different
  interface shapes: **STOP and report the conflict** — do not silently
  reconcile it.

## Scientific interpretation

> This artifact establishes **NOTHING empirically**. It is pure
> interface design. No experiment, no claim, no number comes from this
> agent.

## Commit/PR boundary and handoff

- **One commit:** *"Agent 0: RHAN-NXA canonical interfaces and
  dependency graph."* No implementation logic anywhere in this commit.
- **Handoff artifact:** the five files above, **reviewed before Agent
  A starts**. Agent A (and every subsequent agent) imports from
  `noesis_vision/core/schema.py` and the relevant `interfaces.py` —
  **never redefines its own version**.

## Sequencing note

Agent 0 runs first, always; Agents A–J follow in subsequent turns,
each gated on review of the prior agent's handoff artifact. Agents
A–J's own prompts are **not yet written** — they are gated on review
of Agent 0's handoff (`../MASTER_PLAN.md`'s closing note).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan — Agent 0 contract (captured verbatim in `../MASTER_PLAN.md`).

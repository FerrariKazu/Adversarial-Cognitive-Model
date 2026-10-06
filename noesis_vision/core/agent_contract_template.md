# Agent Contract Template — RHAN-NXA (17 points)

Every agent's prompt (Agents A–J) uses this template **verbatim in
structure** — all 17 points, in this order — plus the standing
non-improvisation clause at the end. This is the template Agents A–J
inherit, per MASTER_PLAN Part 4.

How to use it: fill every `{{PLACEHOLDER}}` from the agent's own
contract in `noesis_vision/noesis_vision/RHAN_NXA/MASTER_PLAN.md` (or, for Agents A–J, from the
prompt drafted after Agent 0's handoff review). Never improvise a
placeholder's content — a missing fact is a question back to the plan's
author, not a guess. Do not add, drop, merge, or reorder points: the
17-point structure is the audit trail that every contract covered
everything.

---

## 1. Mission

{{MISSION — one sentence: what this agent owns.}}

## 2. Scientific purpose

{{WHY this agent exists — the specific failure mode it makes
structurally impossible, or the specific question it answers.}}

## 3. Inputs

{{Exactly what the agent receives: documents, files, artifacts.
Nothing else — no external sources, no invented requirements.}}

## 4. Outputs

{{Numbered list of EVERY artifact produced, with its path.}}

## 5. Files it may modify

{{Explicit allowlist — new files and/or existing paths, exhaustively.}}

## 6. Files it must not modify

{{Explicit denylist. Standing rule for every agent: nothing under
`rhan_core/` — historical reference only, per the no-parallel-port
rule (MASTER_PLAN Part 5).}}

## 7. Existing infrastructure to reuse

{{What already exists and MUST be imported rather than reimplemented
— and, equally important, what does not exist yet.}}

## 8. API/interface contract

{{Exact shapes and signatures. Import from
`noesis_vision/core/schema.py` and the relevant `interfaces.py` —
never redefine your own version. Do not add fields not present in the
plan without flagging them as a question back rather than deciding
silently.}}

## 9. Tensor shapes

{{Part 1.A verbatim where applicable. Every stub/function signature
carries shapes in its docstring, not just types.}}

## 10. Gradient requirements

{{What carries gradients always (z_t, U_t), what only when not None
(S_t), what never (A_t), what must never be detached before use (E_t),
and any component requiring its own optimizer group + |dW| pre-flight
check.}}

## 11. Tests

{{Test files required, and exactly what each must assert.}}

## 12. Smoke experiment

{{The minimal run that proves the artifact works — before any real
training or integration.}}

## 13. Acceptance criteria

{{Checkable conditions for "done": files exist, tests pass, and the
specific scientific/structural criteria for this agent.}}

## 14. Failure conditions (stop and report — do not improvise)

{{Explicit list: on ANY of these conditions, STOP and report the
specific ambiguity/conflict rather than proceeding.}}

## 15. Scientific interpretation

{{What this artifact does and does NOT establish empirically. No
overclaim: if the artifact establishes nothing empirically, say
exactly that.}}

## 16. Commit/PR boundary

{{One commit; the exact commit message; what the commit must not
contain.}}

## 17. Handoff artifact

{{What the next agent receives, and what gates on its review.}}

---

## Standing clause — required in EVERY single prompt

**NON-IMPROVISATION.** If any part of this contract is genuinely
ambiguous, STOP and list the specific ambiguity rather than guessing a
reasonable-sounding default. Guessing a plausible default is the
documented anti-pattern — it is how Gen-0's interface drift started
(multiple eval scripts disagreeing on epsilon convention; a runner
hardcoding a flag that silently confounded two "isolated" experiments,
MASTER_PLAN Part 0). "Stop and report" is the default behavior on
ambiguity — in every single prompt, not just Agent 0's.

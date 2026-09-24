# 27 — Infrastructure Port Table (Part 5)

*Level 2–3 reading. What Gen-1 reuses from the STL-10 project, what it
adapts, and what it refuses to carry forward — with reasons attached
to every disposition.*

---

## In one sentence

Nine infrastructure items, each
with an explicit disposition — **PORT VERBATIM**, **ADAPT**, **REJECT**,
or **NEW** — so nothing is ported by momentum and nothing is rewritten
by taste. Eight are inherited from the Gen-0/STL-10 project; the
EvidentialHead is NEW (see its row).

## The table

| Item | Disposition | Reason |
|---|---|---|
| Multi-group optimizer + registry | **PORT VERBATIM** | Validated across 3+ Gen-0 components; zero reason to touch |
| Gradient-isolation pre-flight (`|dW|` measurement) | **PORT VERBATIM** | *(same validation — the standing rule that components prove their gradient reachability before any smoke run is trusted)* |
| Checkpoint/resume + best/rolling parity check | **PORT VERBATIM** | Learned the hard way once already |
| Comparator registry | **ADAPT** | Same concept, new dataset/checkpoint namespace |
| Structural-consistency assertion (Summary Table vs CSV) | **PORT VERBATIM** | Non-negotiable given its history |
| Attack evaluation (norm-space PGD/AutoAttack conventions) | **ADAPT** | Same conventions, new input resolution/normalization stats |
| EvidentialHead | **NEW** | First implementation in this project of a published formulation (Sensoy et al.), not ported from anywhere |
| AIS-v1's relocated-Eq.-II gaze code | **REJECT** | Superseded by AIS-v2's design; do not carry forward as a fallback path |
| Legacy Slot Attention (16-slot) module | **REJECT** | Doubled evidence against it (Part 1.D) |

## What the dispositions mean

**PORT VERBATIM** — copy as-is; the item is validated, and changing it
creates risk for zero benefit. Any change found necessary during
porting is a stop-and-report event, not an in-flight improvement.

**ADAPT** — the concept is validated but its surface is bound to
Gen-0 specifics (dataset, namespace, resolution/normalization). Adapt
the bindings, preserve the semantics. The comparator registry keeps
its donor-row discipline (never silently re-evaluate comparators); the
attack evaluation keeps its norm-space convention and matched-grid
discipline (`16_Gen0_Evidence_And_Confounds.md`'s protocol) while
rebinding resolution and stats for the new dataset.

**REJECT** — do not port even as a fallback. The two rejections are
mechanism-level, not infrastructural: AIS-v1's gaze code is superseded
by design (one mechanism, no v3 — `10_AIS_v2.md`), and the 16-slot
Slot Attention module is rejected on doubled evidence
(`06_Structure_State.md`). A **REJECTED** module that survives as a
"fallback path" is precisely how legacy behavior silently re-enters a
codebase — the mechanism of the Part-0 confound.

**NEW** — first implementation in this project of a published
formulation (Sensoy et al. for the EvidentialHead). Nothing is ported,
so there is no Gen-0 validation to inherit; the implementation carries
its own contract tests. This is not a claim of scientific novelty —
Part 6's literature classification already records evidential deep
learning as KNOWN PRIOR.

## The no-parallel-port rule

The STL-10 project's `rhan_core/` is **historical reference only**:
Gen-1 builds against Gen-1 interfaces (`noesis_vision/`), and ports
happen deliberately through this table — never by importing the old
package in parallel with the new one. Agent 0's contract makes
`rhan_core/` a must-not-modify zone for exactly this reason.

## Relationship to Agent ownership

- Agent A owns porting the optimizer/resume/logging items; Part 6's
  compute accounting rides on its logging infrastructure.
- Agent D owns the EvidentialHead (NEW — first implementation of the
  Sensoy et al. formulation in this project; no Gen-0 source exists to
  port).
- Agent 0's schema.py must make its optimizer-group config shape match
  what the ported multi-group optimizer expects — informational
  alignment, not implementation (`30_Agent0_Interface_Contract.md`).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 5 — Infrastructure Port Table (the table and dispositions
> captured verbatim in `../MASTER_PLAN.md`; parenthetical elaborations
> on |dW| and ADAPT semantics are cross-references to documented plan
> material, not new policy).

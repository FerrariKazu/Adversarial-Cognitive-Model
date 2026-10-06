# RHAN-NXA Dependency Graph — the literal in-repo source

Agents and future sessions CHECK THIS FILE before starting work. The
graphs below are the MASTER_PLAN Part 2/4 graphs reproduced exactly —
no reordering, no simplification (Agent 0 acceptance criteria). If this
file ever disagrees with `noesis_vision/noesis_vision/RHAN_NXA/MASTER_PLAN.md`, the MASTER_PLAN
wins: STOP and report the discrepancy rather than reconciling silently.

---

## Part 4 — Multi-Agent Dependency Graph & Ownership

Two adjustments the audit requires:

**ADJUSTMENT 1:** Agent E (Predictive Dynamics) and Agent F (AIS-v2)
must share ONE interface contract from Agent 0, because Part 1.B
unified their predictor. They are NOT merged into one agent (F owns
candidate generation/selection/gaze-state bookkeeping; E owns the
predictor module itself and the belief-update network), but Agent 0's
spec must define the predictor's exact call signature ONCE, and both
E and F consume it identically. **Neither may implement its own copy.**

**ADJUSTMENT 2:** Agent G (Structural Representation) is EXPLICITLY
gated — it does not start until Agent J confirms step 6 (Part 2) has
a validated result. Agent 0's spec must record this gate as a literal
precondition Agent G checks before writing any code.

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

Every agent's prompt uses the 17-point contract template (see
`agent_contract_template.md` in this directory), plus the explicit
non-improvisation clause in every single prompt, not just Agent 0's.

---

## Part 2 — Revised Training Phase DAG

Two graphs, kept explicitly separate.

### Build dependency (what can be implemented/tested in parallel)

```text
Infra (Agent A)
  ├─→ Compact ViT + recurrence (Agent C)  [buildable independent of
  │     belief machinery — it's the substrate]
  ├─→ BeliefState scaffold + None-safety (Agent B)  [buildable
  │     independent of C — interface + composition logic, testable
  │     with dummy tensors]
  └─→ EvidentialHead port (Agent D)  [buildable independent of both —
        a self-contained head]

Once C + B + D exist:
  Unified predictor (glimpse-feature prediction + AIS-v2 candidate
  scoring, Part 1.B/E — Agents E+F coordinate tightly, see Part 4).

L_stab scaffolding (diagnostic-only): any time after B exists — it
only needs BeliefState.drift_to(), not a trained model.
```

### Scientific experiment dependency (EVALUATED in this order, or
### attribution is lost)

1. **Backbone-only baseline** (no recurrence, no belief) — parameter-
   and compute-matched controls established HERE, reused for every
   later comparison.
2. **+ Recurrence only** (T=4 glimpses, fixed, NO belief update —
   glimpses are independent, no F). Isolates: does recurrent
   multi-glimpse processing help AT ALL, before any belief machinery.
3. **+ BeliefState + U_t** (Dirichlet), NO F (identity update).
   Isolates: does an explicit typed belief + calibrated uncertainty
   help, independent of predictive dynamics.
4. **+ F / belief dynamics** (Part 1.B's update, NO AIS-v2 — gaze
   fixed/heuristic). Isolates: does predictive updating help
   independent of active gaze selection.
5. **+ AIS-v2** (full loop: predict → observe → error → precision →
   update → gaze-select). First end-to-end test of the active
   perception thesis.
6. **Integrated system**, still `S_t=None`, L_stab=diagnostic-only.
   **THIS is the Generation-1 core result** — the number this whole
   plan exists to produce.
7. **ONLY AFTER 6:** `S_t` reintroduction (minimal, 2–4 slots) as its
   OWN isolated arm vs step 6's result.
8. **ONLY AFTER 6 (not gated on 7):** L_stab promoted from diagnostic
   to training objective, its own isolated arm vs step 6, WITH the
   responsiveness guard as a hard gate.
9. **Full ablation matrix** (Part 3) once steps 1–6 exist cleanly.
10. **ImageNet-1K final run** — the single best-validated
    configuration from steps 1–9, run once, not per-ablation.
11. **Robustness/human-alignment evaluation suite** (Agent I) —
    Tier 1 (PGD/AutoAttack subset, ImageNet-C, shape/texture bias)
    required; Tier 2 (Brain-Score, new psychophysics) only if time
    remains within the 3-month window.

**Steps 7 and 8 are DELIBERATELY parallel-buildable (build dependency)
but NOT parallel-EVALUABLE against step 6 simultaneously in the same
run** — each needs its own clean comparison to step 6's frozen result,
never to each other directly, or you reproduce exactly the D2/D3
confound this plan opened with.

---

## Handoff note

Agent 0's five artifacts (schema, two interface stubs, this graph, the
contract template) are reviewed BEFORE Agent A starts. Agent A and
every subsequent agent import from
`noesis_vision/core/schema.py` and the relevant `interfaces.py` —
never redefine their own version. Agent G additionally checks
Adjustment 2's literal precondition (Agent J's step-6 confirmation)
before writing any code.

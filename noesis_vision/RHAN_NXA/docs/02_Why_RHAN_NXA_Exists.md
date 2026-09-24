# 02 — Why RHAN-NXA Exists

*Level 0 reading. Written for a non-researcher first, then the exact
technical version. ~15 minutes.*

---

## The short version

Generation 0 ("Gen 0") of this research program — the RHAN/NOESIS
experiments on STL-10 — demonstrated something important and slightly
uncomfortable:

> **Generation-0 demonstrated that adding more structured modules and
> representational capacity does not automatically produce a new
> robustness frontier.**

Every structured addition traded clean accuracy for robustness (or
nothing) along the same frontier. Nothing moved the frontier itself.
And on top of that, the two experiments designed to test *isolated
mechanisms* turned out to be contaminated. Gen-1 exists to rebuild the
core honestly, with statuses, controls, and smaller assumptions.

## Gen 0 as a scientific progression

Gen 0 is **not framed as a failure**. It produced:

- a validated evaluation protocol (16-seed, matched-epsilon, PGD-100
  in norm space) whose numbers are trustworthy;
- a real mechanism-level discovery (AIS-v2's candidate-preference
  correlation — see below);
- direct, gate-verified negative evidence about a specific structural
  implementation (the 16-slot SBR);
- and the single most valuable lesson: **capacity is not mechanism.**

A generation that teaches you what *doesn't* work, and *why* you
couldn't tell, is progress — provided you write down exactly what it
proved and what it couldn't.

## The steering-wheel analogy

> Imagine testing whether adding a steering wheel makes a car faster,
> but accidentally installing the same experimental engine
> modification in every car — including every control car. You can
> still observe that the cars behaved differently from unmodified
> cars, but you cannot honestly attribute the difference specifically
> to the steering wheel.

That is what happened to the two Gen-0 experiments called **D2**
(AIS-v2) and **D3** (belief-HPC).

## The exact technical version

The Gen-0 ladder runner (`_nx_trainer` in the cloud notebooks) hard-coded
`--enable-sbr` into the training command for **every** stage. The two
stages that were supposed to be *isolated swap tests* — D2 (AIS-v2
alone) and D3 (belief-HPC alone) — therefore each unintentionally
carried the full legacy slot mechanism (SBR, 16 slots, 512-dim) on top
of the mechanism they were meant to test in isolation.

The tell is stark: D3's clean accuracy (45.06±2.64) and its
clean-delta vs donor D (−9.90, significant) are statistically
indistinguishable from E2's — the experiment that *intentionally* added
SBR (45.06±3.30, −9.90). The same two numbers, to two decimal places,
from two experiments testing supposedly different mechanisms. That is
the SBR signature dominating both.

**Consequence:** AIS-v2's and belief-HPC's true isolated effects on
16-seed clean/robust accuracy are **UNKNOWN**. Not "weak" — unknown.

The full confound scope — which experiments are affected, which are
not, and why — is in `16_Gen0_Evidence_And_Confounds.md`.

## The salvageable result

One Gen-0 result survives as targeted, mechanism-level evidence:

> **AIS-v2's smoke-gate candidate-preference correlation: r = 0.706**
> (512 samples, schema `ais_v2_smoke_gate_v1`).

This measures whether the candidate-scoring head's predictions
correlate with policy choice — a property largely orthogonal to
whether SBR happens to be active. It is graded **REQUIRED** evidence
and is the single most valuable new fact Gen 0 produced.

## What Gen-1 changes because of this

1. **Restart from a smaller, explicitly-labeled core.** The core
   belief loop (`z_t`, `U_t`, `E_t`, `A_t`) with `S_t = None` — every
   component carries a scientific status label.
2. **Confounded results are not inputs.** The confounded +11.42pp
   number for AIS-v2 is not treated as settled fact; the mechanism is
   re-tested cleanly once the Gen-1 substrate exists.
3. **Controls are mandatory, not optional.** Gen 0 proved the
   capacity-vs-mechanism confound is a *live* risk (the SBR rungs'
   clean gains came with capacity). Any "recurrence helps" or
   "prediction error helps" claim now requires parameter-matched and
   compute-matched controls to exist first.
4. **Gates get tightened, not just re-run.** sbr0's structural gate
   passed while measuring near-uniform, near-chance slot content —
   a floor that cannot detect vacuity. Gen-1 gates must be able to
   fail for the right reasons.
5. **Negative results stay negative.** The 16-slot SBR implementation
   is rejected on doubled evidence. This is recorded as a *clean,
   reportable outcome*, not buried.

## What did NOT change

The evaluation protocol, the honesty norms (FAIL is a complete,
reportable outcome), and the research question — *what computational
organization perceives robustly?* — all carry forward. RHAN-NXA is the
same research program, rebuilt on evidence instead of assumptions.

---

## Reading path

Next: `03_Perception_As_Investigation.md` — the central idea the new
generation is built around.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 0 — Gen-0 Confound Scope; Part 1.D (SBR rejection
> grounds); `report/Gen0.md` §3, §6, §8 (the numbers and provenance
> behind this chapter).

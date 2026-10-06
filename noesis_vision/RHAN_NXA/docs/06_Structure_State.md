# 06 — `S_t`: Structural State (Currently None)

*Level 1 reading. This chapter exists mostly to explain an absence,
and the evidence behind it.*

---

## In one sentence

`S_t` is the belief state's optional slot for explicit structural /
object-level representation — and in the Gen-1 core build it is
**`None`**, by design and on evidence.

## The intuition

Humans describe scenes partly in terms of *things*: objects, parts,
arrangements. `S_t` is where such structure would live in the belief
state if the model maintained it explicitly — for example, as a set of
slots, each holding one object-like chunk of representation.

Gen-1 does **not** do this. The core build runs with `S_t = None`,
through the first full integrated system and its first ablation
matrix. This is not an oversight; it is a decision with doubled
evidence behind it.

## What structure *could* eventually mean

These are **future possibilities, not established implementation
requirements**:

- objects
- parts
- relationships
- spatial organization
- shape
- object-level information

## The evidence against the CURRENT implementation

Two independent lines, both from gate-verified Gen-0 runs:

### 1. The "everything-slot" ablation
sbr0's ablation zeroed the highest-norm slot — the slot the model
leaned on most. If slots carried unique information, accuracy should
drop. It *improved*: **retained = 1.0157** against a floor of 0.7 that
was designed to catch the opposite failure. Direct, gate-verified
evidence that the current slots carry no unique information.

### 2. The clean-collapse signature
Activating this specific slot implementation costs roughly **−9 to
−10pp clean accuracy** — confirmed appearing every time it is active,
whether deliberately (E2, sbr2–sbr4) or accidentally (D2, D3 — the
Part-0 confound). The two cleanest tell: D3 (45.06±2.64, Δclean −9.90
vs D) and E2 (45.06±3.30, Δclean −9.90) — identical to two decimal
places, from experiments testing supposedly different mechanisms.

## The distinction that must be impossible to miss

> **REJECTED IMPLEMENTATION ≠ REJECTED IDEA.**

RHAN-NXA is rejecting the **current slot mechanism** — 16 slots,
512-dim, as implemented and tested in Gen 0. That does **NOT** mean
"RHAN-NXA does not need structural representation," and it does **NOT**
mean "object-centric perception was disproven."

What it means: *the tested implementation did not provide sufficient
evidence and should not be part of the core.* The concept of
structural representation remains an **EXPERIMENTAL CANDIDATE,
DEFERRED**.

## If structure is revisited (the pre-registered re-entry rules)

These come from the plan and are binding on any future re-attempt:

1. **Start at 2–4 slots, not 16.**
2. **Per-slot decodability must show slots are statistically
   distinguishable from each other and from chance** BEFORE any
   scale-up.
3. The Gen-0 gate's floor must be **tightened first** — sbr0's gate
   technically passed while measuring near-uniform, near-chance
   per-slot content (16/16 slots "above floor" at ≈0.44–0.51 probe
   accuracy on 10 classes, i.e. ≈ chance). A gate that cannot detect
   vacuity is not evidence of function, even when it passes.
4. Re-entry only **after** the None-`S_t` core system validates on
   ImageNet-100.

## Formal definition (as specified, for later use)

```text
S_t : None | (B, K, D_s)
```

- `K` = number of structural slots (note: this `K` is a *different*
  quantity from AIS-v2's candidate count `K` — see
  `14_Tensor_Shape_Reference.md` for the notation disambiguation)
- `D_s` = structural embedding dimension
- when present, `S_t` is an instance of the existing BeliefState ABC's
  structured variant — **not a new type**
- when present, `S_t` carries gradients; when None, every consumer
  must take the explicit None branch (`04_Belief_State.md`)

Interaction with the prediction-error update: **none in the core
build.** If `S_t` is reintroduced later, its own update follows the
same predict/observe/error/precision/update shape *independently* —
not derived from `z_t`'s `E_t`.

## Scientific status

| Claim | Status |
|---|---|
| `S_t = None` as the Gen-1 core default | **REQUIRED** (as the default configuration) |
| The 16-slot SBR implementation as tested in Gen 0 | **REJECTED** |
| Structural representation as a concept | **EXPERIMENTAL CANDIDATE, DEFERRED** |

## What this does NOT mean

- It does not mean the model is incapable of encoding object-like
  information implicitly — a `z_t` trained on object classification
  will encode object-relevant features. What is absent is the
  *explicit, structured, inspectable* state.
- It does not mean the question is settled forever. It means the
  current implementation has not earned its complexity.

## Open questions

- Whether any slot-based mechanism can demonstrate per-slot
  distinguishability under a tightened gate.
- Whether structural state, if it returns, should share `E_t`'s
  predictor or run its own (the plan currently says: its own,
  independently).

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.D — Structure (`S_t = None` lock, doubled grounds,
> re-entry criteria), Part 1.A (`S_t` typing and gradient rule);
> `report/Gen0.md` §5 (sbr0 gate details) and §6 (confound numbers).

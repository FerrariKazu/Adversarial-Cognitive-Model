# 07 — `U_t`: Uncertainty

*Level 1 reading. Includes the scoping compromise every reader should
know about.*

---

## In one sentence

`U_t` is the model's current estimate of how uncertain it is — and in
Gen-1, that estimate comes from Dirichlet evidence over the
**classification readout**, a validated mechanism with a known,
documented limitation.

## The intuition

The model should not only say what it believes. It should also have an
estimate of **how uncertain** that belief is. Uncertainty is what
makes the investigation *targeted*: it tells AIS-v2 where looking next
is most likely to resolve ambiguity, and it weights (`Π_t`) how much
new evidence should move the belief.

You do not need to understand Dirichlet distributions yet. At this
level, think of `U_t` as the model's current estimate of how uncertain
it is. The exact implementation follows.

## A concrete example

Partly-hidden-animal image, after glimpse 1: several class
interpretations remain plausible. The evidence vector is spread across
them → high uncertainty. After a glimpse resolves the head structure,
evidence concentrates on one interpretation → uncertainty drops. That
drop is *also* a prediction: it tells the gaze mechanism that this
kind of look was informative, which shapes where it looks next.

## Formal definition

```text
e_t      : (B, C)   evidence, non-negative (softplus)
alpha_t  = e_t + 1  Dirichlet concentration
U_scalar = C / sum(alpha_t)     # the uncertainty scalar
```

In plain English: the model emits a non-negative "evidence" value per
class; adding one to each gives a Dirichlet distribution's
concentration parameters; the more total evidence, the lower the
uncertainty scalar. (Dirichlet — a distribution over probability
distributions — is the standard machinery of evidential deep
learning; the Glossary has a longer introduction.)

## The limitation — read this before building on `U_t`

> **`U_t` is uncertainty over the classification READOUT, not a
> complete uncertainty measure over every aspect of the belief.**

The plan states this tension explicitly: this sits oddly against
"classification is a readout, not the purpose." A genuinely
belief-centric uncertainty would measure confidence in `z_t`/`S_t`
themselves — the *content* of the belief — not in the class label the
content eventually produces.

### Why Gen-1 keeps the class-conditioned version anyway

- The **EvidentialHead already exists and is validated** (Gen-0
  lineage).
- A representation-level uncertainty has **no existing target to
  train against** without inventing new machinery — and inventing new
  machinery is exactly what Gen-1's core is trying to avoid.

This is a **scoping compromise**, and it is documented as such
everywhere `U_t` appears.

## Role in the loop

| Loop step | `U_t`'s part |
|---|---|
| Precision | determines how strongly new evidence moves the belief (`Π_t`) |
| Attention | AIS-v2 scores candidates by *predicted uncertainty reduction* (Dirichlet entropy), not by a separate scoring head |
| Repeat | (deferred) halting would naturally read uncertainty — one reason halting is deferred until `U_t`'s semantics are settled |

## Gradient behavior

`U_t` is gradient-bearing **always**. The evidential head receives
gradients from whatever loss terms supervise evidence in a given
configuration; the specifics of loss terms are outside this
documentation pass (Section 8+ of the master plan), but the gradient
*reachability* of `U_t` is part of this specification.

## Scientific status

| Claim | Status |
|---|---|
| Class-conditioned Dirichlet `U_t` for Gen-1 | **REQUIRED** (as the Gen-1 mechanism, with the documented compromise) — **LOCKED per Part 1.F: one representation only, the EvidentialHead's Dirichlet formulation PORTED (not reimplemented) from the STL-10 project** |
| Representation-level uncertainty (uncertainty over `z_t`/`S_t` content) | **PENDING DECISION** — deferred past Gen-1's first integrated build |
| "AIS-v2 must reuse this same uncertainty representation" | **REQUIRED** design rule (one uncertainty representation; no separate scoring head) |
| Single consumer set for `U_t` | **REQUIRED** per Part 1.F: it feeds `Π_t` (precision) in the update, AIS-v2's candidate scoring, and L_stab's drift metric — nothing else derives its own uncertainty |
| Calibration (ECE) | Evaluated in Agent I — **not trained against directly in Gen-1's first pass** |

## What this does NOT mean

- `U_t` is not "uncertainty about everything the model believes" —
  the most common overreading. It is readout-level.
- `U_t` being low does not mean the belief is *correct* — evidential
  models can be confidently wrong; calibration is an empirical
  property to be measured, not an assumption.
- The Dirichlet machinery is not claimed to be how biological
  perception represents uncertainty.

## Open questions

- What training target could representation-level uncertainty use at
  all? (Unresolved — that is precisely why it is a pending decision
  and not a component.)
- Whether Dirichlet-entropy-based candidate scoring and the scalar
  `C / sum(alpha_t)` remain consistent as `C` changes across datasets
  (STL-10 → ImageNet-100) — noted as an open question by the plan's
  one-uncertainty-representation rule.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 1.A — BeliefState (`U_t` bullet and the TENSION
> paragraph, preserved verbatim in `MASTER_PLAN.md`), Part 1.E
> (scoring by Dirichlet entropy, no separate scoring head), and
> Part 1.F — Uncertainty (locked: one representation, ported
> EvidentialHead, three consumers, ECE evaluated not trained).

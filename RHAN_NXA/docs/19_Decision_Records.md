# 19 — Decision Records

*Level 2–3 reading. Institutional memory: why each non-obvious choice
was made, what was rejected, and what would change it.*

Format per record:

```text
Decision:
Current choice:
Status:
Why:
Evidence:
What was rejected:
What remains uncertain:
Experiment that could change this:
```

---

## DR-1 — Why `S_t = None` initially

```text
Decision:            Exclude explicit structural state from the Gen-1 core.
Current choice:      S_t = None through the first integrated system and
                     its first ablation matrix.
Status:              REQUIRED (as the default configuration).
Why:                 The burden of proof is on added structure; Gen-1
                     core must validate before carrying optional modules.
Evidence:            Doubled: (1) sbr0's everything-slot ablation
                     IMPROVED accuracy when the top slot was zeroed
                     (retained = 1.0157 vs a 0.7 floor designed to catch
                     the opposite); (2) the −9 to −10pp clean-collapse
                     signature appears whenever this slot implementation
                     is active — deliberately (E2, sbr2–4) or
                     accidentally (D2, D3).
What was rejected:   Carrying the 16-slot SBR forward "just in case."
What remains uncertain: Whether some structural mechanism can pass a
                     gate that can actually detect function.
Experiment that could change this: After ImageNet-100 validation, a
                     2–4-slot variant with per-slot decodability
                     distinguishable from each other AND from chance,
                     under tightened gates.
```

## DR-2 — Why pixel reconstruction was rejected

```text
Decision:            Do not use pixel-space reconstruction as E_t's target.
Current choice:      Rejected for Gen-1.
Status:              REJECTED (for Gen-1).
Why:                 Two independent reasons, not one: mechanistic
                     (reconstruction dilutes precision and increases
                     belief drift per Gen-0 Lens analysis) and
                     engineering (needs a decoder the compact-ViT
                     substrate does not have, with its own parameter
                     budget and gradient-isolation treatment).
Evidence:            Lens analysis (mechanistic, repeated); E1's
                     statistical result was −0.90pp vs D, NOT
                     significant — recorded precisely as "mechanistically
                     disfavored, statistically inconclusive, engineering
                     cost not justified."
What was rejected:   Claiming "pixel reconstruction was proven harmful"
                     — that overstates the significance test.
What remains uncertain: Whether some future decoder-free dense target
                     could earn its place.
Experiment that could change this: A substrate with native dense
                     prediction capability + significant Lens-verified
                     benefit under controls.
```

## DR-3 — Why latent glimpse prediction was selected

```text
Decision:            E_t = mismatch between predicted and observed
                     token-level features at the next glimpse's fixation.
Current choice:      Locked as the Gen-1 design default.
Status:              EXPERIMENTAL CANDIDATE (design LOCKED, empirical
                     status open — D3's supporting evidence is confounded).
Why:                 (1) Mathematically valid by construction —
                     prediction and observation come from the same
                     encoder, guaranteed comparable space; (2) no new
                     decoder infrastructure — the patch-embedding output
                     is both target format and ground truth; (3) unifies
                     with AIS-v2 — one predictor, two consumers.
What was rejected:   Predicting next-global-z from current-global-z
                     (nearly trivial for a pooled vector self-predicting
                     within one static image); the z_t + λ·Π·E_t raw-add
                     draft (E and z not trivially addable → UpdateNet).
What remains uncertain: Whether it beats the null (no E_t at all) under
                     matched controls.
Experiment that could change this: The pre-registered clean,
                     SBR-disabled rerun vs the null (F = identity).
```

## DR-4 — Why T = 4

```text
Decision:            Run T = 4 fixed glimpses per image.
Current choice:      T = 4 for the first build.
Status:              REQUIRED for the first build (revisable via
                     controlled ablation).
Why:                 Reuses the STL-10 project's own validated
                     convention rather than re-deriving a number from
                     nothing.
Evidence:            Project precedent (validated prior setting).
What was rejected:   Deriving T from first principles at build time.
What remains uncertain: Whether 4 remains right at higher resolution
                     / new datasets.
Experiment that could change this: T-sweep ablation with
                     parameter- and compute-matched controls.
```

## DR-5 — Why tied (shared-weight) within-glimpse recurrence

```text
Decision:            Run one shared transformer block 2–3 times per
                     glimpse before pooling (Universal-Transformer-style).
Current choice:      Weight-tied refinement.
Status:              LOCKED as design (Option C — hybrid — chosen
                     narrowly, not for "expressiveness").
Why:                 Option A alone abandons the belief-state thesis
                     (becomes a plain recurrent ViT — useful only as a
                     CONTROL); Option B alone likely under-represents
                     each glimpse (no iterative token refinement before
                     pooling). Tied weights make iteration count change
                     COMPUTE, not parameter count — keeping compactness
                     accounting simple.
Evidence:            Architectural reasoning; the capacity-vs-mechanism
                     risk is empirically live (SBR rungs).
What was rejected:   Untied stacked blocks (blurs params vs compute);
                     single-pass pooling (Option B's weakness).
What remains uncertain: Exact iteration count (2 vs 3).
Experiment that could change this: Within-glimpse iteration ablation
                     under the controls rule.
```

## DR-6 — Why adaptive halting is deferred

```text
Decision:            Fixed T = 4; no adaptive halting in the first build.
Current choice:      Deferred.
Status:              DEFERRED.
Why:                 Halting adds a second objective with a documented
                     history of fighting other objectives; the core loop
                     must be validated at fixed depth first.
Evidence:            Gen-0's halting history: v10's
                     loss-conflict-with-Banach-proof incident; AIS-v1's
                     modest halting effect.
What was rejected:   Building halting simultaneously with the core
                     loop.
What remains uncertain: Whether halting helps at all once the core is
                     stable.
Experiment that could change this: Halting re-introduced on a
                     validated core, with halting-on/off controls.
```

## DR-7 — Why AIS-v2 uses the same predictor

```text
Decision:            One glimpse-feature predictor; consumers = the
                     realized-glimpse error (E_t) and candidate scoring.
Current choice:      Shared predictor; scoring by predicted uncertainty
                     reduction via U_t's Dirichlet entropy.
Status:              REQUIRED as a design rule (no separate scoring
                     head; no independent reimplementation).
Why:                 Two separately-trained predictors could drift out
                     of sync — the duplicated-effort split the audit
                     warns against; one uncertainty representation.
Evidence:            Architectural reasoning + r = 0.706 mechanism
                     evidence that the scoring pathway carries signal.
What was rejected:   A dedicated scoring head; predicting global z
                     instead of local features.
What remains uncertain: Long-run alignment between realized-error
                     updating and uncertainty-reduction scoring.
Experiment that could change this: Any demonstrated divergence between
                     the two consumers under the shared predictor.
```

## DR-8 — Why uncertainty currently comes from Dirichlet evidence

```text
Decision:            U_t = class-conditioned Dirichlet evidence via the
                     existing EvidentialHead.
Current choice:      Kept for Gen-1.
Status:              REQUIRED for Gen-1, with the scoping compromise
                     documented (readout-level, not belief-level).
Why:                 The EvidentialHead already exists and is validated;
                     a representation-level uncertainty has no existing
                     target to train against without inventing new
                     machinery.
Evidence:            Component validation in the Gen-0 lineage.
What was rejected:   Deferring ALL uncertainty (breaks precision
                     weighting and AIS scoring); building new
                     representation-level machinery now.
What remains uncertain: What target representation-level uncertainty
                     could train against at all.
Experiment that could change this: A validated self-supervised target
                     for belief-content uncertainty (future work).
```

## DR-9 — Why representation-level uncertainty is deferred

```text
Decision:            Do not build uncertainty over z_t/S_t content in Gen-1.
Current choice:      PENDING DECISION, deferred past the first
                     integrated build.
Status:              PENDING DECISION.
Why:                 The scoping compromise above; inventing new
                     machinery during the core build repeats the Gen-0
                     pattern of unvalidated additions.
Evidence:            The plan's own TENSION paragraph (preserved
                     verbatim in MASTER_PLAN.md Part 1.A).
What was rejected:   Pretending U_t already measures belief-content
                     uncertainty.
What remains uncertain: The training-target question (DR-8).
Experiment that could change this: Same as DR-8.
```

## DR-10 — Why the 16-slot SBR is not carried forward

```text
Decision:            Do not port the Gen-0 16-slot SBR into RHAN-NXA.
Current choice:      Rejected as an implementation.
Status:              REJECTED (implementation only — the concept remains
                     an EXPERIMENTAL CANDIDATE, DEFERRED).
Why:                 REJECTED IMPLEMENTATION ≠ REJECTED IDEA; the
                     tested implementation lacks evidence of function
                     and carries a reproducible cost.
Evidence:            sbr0 gate details (per-slot probes ≈0.44–0.51 ≈
                     chance; everything-slot ablation retained = 1.0157);
                     the clean-collapse signature across E2, sbr2–4,
                     and the accidental D2/D3 cases.
What was rejected:   "Gate passed" as evidence of function — the floor
                     could not detect vacuity.
What remains uncertain: Whether ANY slot mechanism can demonstrate
                     distinguishability under tightened gates.
Experiment that could change this: The DR-1 re-entry experiment.
```

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Parts 0 and 1.A–1.F (every record above traces to captured
> plan text in `../MASTER_PLAN.md`; numbers to `report/Gen0.md`).

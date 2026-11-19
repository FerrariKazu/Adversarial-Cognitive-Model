# RHAN-Bi-v1 independent audit

**Audited commit:** `b736723` (`rhan-exp`), package `1.0.0`, config `16bb095c3497`, evidence model
`532e86050daaafcd`, forward model `7148d4a06c303487`.
**Audit date:** 2026-11-18.
**Constraint honoured:** no implementation file, threshold, checkpoint, or result artifact was
modified. All measurements below were produced by read-only, in-memory probes importing `rhan_bi`.
No new training was run. `results/` still contains only the three files written by the original demo.

## Deliverables

| # | File | Deliverable |
|---|---|---|
| 1 | [`01_AUDIT_REPORT.md`](01_AUDIT_REPORT.md) | independent audit report, executive verdict |
| 2 | [`02_MATHEMATICAL_ASSUMPTIONS.md`](02_MATHEMATICAL_ASSUMPTIONS.md) | assumptions and derivations |
| 3 | [`03_EXPERIMENT_VALIDITY.md`](03_EXPERIMENT_VALIDITY.md) | experiment-validity assessment |
| 4 | [`04_GATE_CLASSIFICATION.md`](04_GATE_CLASSIFICATION.md) | gate-by-gate evidence classification |
| 5 | [`05_CONFIRMED_DEFECTS.md`](05_CONFIRMED_DEFECTS.md) | confirmed implementation defects |
| 6 | [`06_UNRESOLVED_MATHEMATICS.md`](06_UNRESOLVED_MATHEMATICS.md) | unresolved mathematical questions |
| 7 | [`07_UNSUPPORTED_CLAIMS.md`](07_UNSUPPORTED_CLAIMS.md) | unsupported scientific claims |
| 8 | [`08_DINOV2_ASSESSMENT.md`](08_DINOV2_ASSESSMENT.md) | DINOv2 integration assessment |
| 9 | [`09_MINIMUM_VALIDATION_EXPERIMENT.md`](09_MINIMUM_VALIDATION_EXPERIMENT.md) | minimum experiment for the visual foundation |
| 10 | [`10_CORRECTION_PLAN.md`](10_CORRECTION_PLAN.md) | prioritized correction plan |

## Bottom line

RHAN-Bi-v1 is a **well-structured feasibility prototype**, not yet a validated experimental
architecture. The control loop, the deterministic simulator, the split discipline at the seed level,
the immutable-record idea, and the decision to report negative and inconclusive outcomes are sound.
The central quantitative machinery is not yet sound enough to support confirmatory claims:

1. The posterior is **order-dependent**, although the documentation says addition commutes. The same
   three observations in different orders gave final confidence between **0.827 and 0.987** on one
   scene.
2. The reported `n_eff` is **not used in inference**. The posterior uses a different, prefix-dependent
   per-record weight. `n_eff = 1.29` is arithmetic on two records with a hand-set ρ = 0.55, not an
   estimated correction.
3. Duplicate observations can increase belief **without bound** (harmonic weight sum), contradicting
   the documented "at most one record's worth".
4. The ledger detects accidental edits but is **not tamper-evident against a re-hashing adversary**:
   tail truncation and a fully re-chained forged history both pass `verify()`.
5. Gate A measures a **different model** from the one used by Gates C–E. The deployed calibration-fit
   model scores **0.850/0.883** (accuracy / defect-vs-reflection balanced) on dev, not the reported
   **0.817/0.767**.
6. Gate C's coverage number is **scale-inconsistent**: the threshold is selected on raw
   confidences and applied to temperature-scaled ones. Corrected raw coverage is **0.767**, not
   **0.875**.
7. Gate B's 0.97 quantile was selected after inspecting dev and is evaluated on the same dev set. The
   sweep stored in the result is hard-coded and stale; the artifact the docs cite does not exist.
8. Gate D's "revision accuracy" is a recovery rate with **2 corrected episodes out of 18**; the
   `no_dep` and `no_pair` arms correct the same two. The advantage exists only against the weak
   `pooled` baseline.
9. Gate E is properly labelled inconclusive, but for a more specific reason than "same actions":
   falsification and EIG chose **different action sequences in 119/120 scenes** and still produced
   the same 0.1917 error. The comparison is underpowered and the simulator has little policy headroom.
10. The DINOv2 path is **not wired to the evidence model at all**. `DinoV2Backbone.encode` returns the
    same eight handcrafted features and puts CLS/patch tokens in `meta`, which
    `EvidenceBuilder.build` never reads.

## The one-question answer

**Is RHAN-Bi-v1 sound enough to justify connecting a real visual encoder?**

**Conditionally, yes — but only as a bounded integration experiment, and only after the correctness
defects in the evidence path are fixed.** It is not sound enough to justify a confirmatory claim, a
real-world performance claim, or a DINOv2-based version of the current gates.

The minimum evidence required before connecting a real encoder is:

1. An order-invariant posterior with a derived dependence adjustment (or an explicit statement that
   dependence is a heuristic and removal of `n_eff` as a reported quantity).
2. A ledger that binds image bytes to `image_sha256`, rejects duplicate images, and has an external
   head anchor so truncation and re-chaining are detectable.
3. Gate A evaluated on the **deployed** evidence model, with the ambiguity band enforced on that model.
4. Gate B and Gate C thresholds fixed on calibration only, evaluated once on test, with the coverage
   threshold on the same probability scale as the evaluation probabilities.
5. Gate E at a pre-registered N with common random numbers and a power calculation, and all policy
   episodes saved.
6. A frozen-encoder probe on real images that establishes single-look competence and localization
   **before** the loop is run, with an explicit statement that synthetic-trained evidence weights are
   discarded.

Until those are done, connecting DINOv2 would change the input distribution while leaving the
statistical and provenance defects intact. That would make the failures harder to see, not the system
more real.

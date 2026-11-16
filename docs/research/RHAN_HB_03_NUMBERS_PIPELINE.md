# 3 — The numbers pipeline

The handbook's rule is that no consequential number is typed into prose. It is written as a token,
resolved at build time from a registry, and audited afterwards by three independent checks. This
document describes that pipeline and what each stage can and cannot catch.

## 1. The registry

`docs/rhan_nxa/data/numbers.json` holds **276 metrics**. Each record has:

| Field | Purpose |
|---|---|
| `value`, `unit`, `display`, `display_precision` | the number as data and as it is printed |
| `status` | `VERIFIED`, `SUPPORTED_WITH_LIMITATIONS` or `RETRACTED` |
| `source_artifacts` | the recorded files the value can be re-read from |
| `source_code` | the function that computes it, where it is computed rather than recorded |
| `derivation` | the expression, when the value is arithmetic on other registered fields (138 records) |
| `limitations` | what the value does not establish (168 records) |
| `description` | one sentence a reader can check |

The status split is the honest part of the pipeline:

| Status | Count | Meaning |
|---|---:|---|
| `VERIFIED` | 127 | the value was re-read from a recorded artifact |
| `SUPPORTED_WITH_LIMITATIONS` | 147 | the value comes from a transcription or a run whose raw output is not in the repository |
| `RETRACTED` | 2 | the value is known to be invalid and is printed only with its withdrawal |

The two retracted records are `final.ff.apgd_ce_invalid` and `final.ff.apgd_t_invalid` — the `ff_112`
APGD figures whose measurement was invalid because the attack ran on normalised tensors and clamped
to `[0, 1]`. They stay in the registry so that a reader who meets the old number can find out what
happened to it.

![Registry status and figure traceability](RHAN_HB_figures/hb09_registry_status.png)

The registry's own `registry_version` tag still reads `rhan-handbook-v3.1`, which is correct: the
registry was introduced in v3.1 and a handbook edition does not change it unless a metric does. v3.4
changed no metric, so the tag did not move.

## 2. Rendering

The PDF build substitutes `{{N:metric.id}}` tokens through `scripts/handbook/render_numbers.py`
before pandoc sees the text. An **unknown id aborts the build**; a missing value cannot degrade to a
blank or to a literal in the source. That single rule is what stops the common failure mode in which
a number is edited in the prose during a revision and the registry silently disagrees.

`scripts/handbook/build_numbers.py` rebuilds the registry from the artifacts deterministically, and
`--check` fails when the checked-in registry no longer matches what the artifacts produce. The
combination is a two-way lock: the build cannot print an unregistered number, and the registry
cannot go stale without a failing command.

## 3. Claims and lineage

Two smaller graphs sit beside the registry:

- `claims.json` — 13 claims, each mapped to a metric, an artifact and the section that makes the
  claim. This is what turns "the loop is more robust" into a citable triple.
- `training_lineage.json` — phase, checkpoint, inherited epochs and cumulative epochs, which is how
  the book can state that `recurrence_only` inherits 30 epochs before its own 15 and `ff_112` starts
  from DINOv2 with 15.
- `experiment_manifest.json` — 13 recorded artifacts with SHA-256 digests. A changed artifact fails
  validation until the registry is rebuilt, so an artifact cannot be quietly replaced under a
  published number.

## 4. The six gates

![The six validation gates](RHAN_HB_figures/hb06_validator_stages.png)

`scripts/validate_rhan_handbook.py` runs six stages in order and stops at the first non-zero exit.
Each answers a different question, which is why they are not merged:

| # | Stage | Question | Failure it catches |
|---|---|---|---|
| 1 | `validate_numbers.py` | is the registry self-consistent, and does every figure reference resolve? | a duplicate figure or claim id; a figure whose file is missing or that no chapter references; a claim citing an unknown or retracted metric; an artifact whose SHA-256 changed; a metric unknown to the registry |
| 2 | `verify_claims.py` | do the cited numbers still come out of the artifacts? | a metric whose artifact changed; a paired statistic that no longer reproduces |
| 3 | `audit_handbook_numbers.py` | is every printed percentage a field or a listed derivation? | a hand-computed percentage that drifted from its inputs |
| 4 | `audit_handbook_deltas.py` | is every written pp delta reproducible? | a delta computed from the wrong pair, or from an unpaired test |
| 5 | `audit_rhan_handbook_pdf.py` | is the PDF structurally sound? | blank pages, ink in the margin, contents entries that do not resolve |
| 6 | `audit_rhan_typography.py` | is the text layer clean? | text outside the frame, overlapping spans, hyphenated identifiers, caption drift, orphan headings, undersized figures, vertical holes |

Stages 3 and 4 are the ones that earn their keep. On the v3.4 build, stage 3 reports:

```
unexplained : 0
derived     : 31% x1  0.587 / 0.447 - 1 = 31.3 %, chapter 24 latency ratio
```

and stage 4:

```
distinct pp values: 55; occurrences: 129
artifact-derived delta universe: 323 values from 34 groups
matched: 129   unmatched: 0
```

That is a strong statement for a document of this size: every percentage and every point-delta in
254 pages is either a registered field or an expression over registered fields, and every delta
matches a value the artifacts can produce.

## 5. What the pipeline cannot catch

- **A wrong interpretation of a correct value.** The registry proves that 0.587 s and 0.447 s are the
  recorded latencies; it cannot prove that comparing them is meaningful. Prose-level review and the
  consistency report carry that, not the tooling.
- **A transcription that is internally consistent.** 147 metrics are `SUPPORTED_WITH_LIMITATIONS`; a
  consistent transcription still passes stage 2 if nothing else contradicts it. The status field is
  the only signal, which is why it is printed rather than buried.
- **A figure that cites a metric but draws it wrong.** Stage 1 checks that a figure id exists and is
  referenced; it does not check that the drawn bar corresponds to the metric's value. The generator
  reading the artifact is the safeguard — and the reason data figures are generated, never drawn by
  hand.
- **Numbers outside the token system.** A quantity that is not registered is invisible to every
  gate. The mitigation is a rule, not a check: if a number matters enough to be argued about, it
  goes in the registry.

## 6. The provenance layers

![The provenance layers, as v3.4 counts them](RHAN_HB_figures/hb10_provenance_layers.png)

A reader walking backwards from a printed number passes through:

| Layer | Records | What it is |
|---|---:|---|
| hashed experiment records | 13 | artifacts with a recorded SHA-256 |
| ledger records | 34 | evidence entries classified by strength |
| claim-graph entries | 13 | claim → metric → artifact → section |
| registry metrics | 276 | every printed value with its status |
| figure records | 107 | every figure with class, sources, alt text, limitation |
| figures citing a metric | 39 | the overlap between the last two |

The layers are not a confidence hierarchy — the claim graph is deliberately smaller than the
registry, and a metric need not pass through every layer to be citable. Their sizes are read from the
files at build time (the appendix figure does this), so a count in the prose cannot drift from the
source.

## 7. A worked example: the ε assertion that could not fail

The v3.2 edition removed an assertion of the form

```python
assert abs(eps_norm * MEAN_STD * 255 - round(eps_norm * MEAN_STD * 255)) > 1e-6 \
    or eps_norm in {0.0, 0.031, 0.062, 0.094}
```

which passed for almost any input: the first disjunct is true whenever the product is not an
integer, so the assertion could not establish which unit a radius was in. It was replaced by
`tests/test_epsilon_units.py`, which converts the registered radii through
`curriculum_gen2.norm_to_pixel_eps` in both directions and checks the failure mode directly — a
pixel fraction handed to a normalised interface is off by 1/mean(σ) ≈ 4.4, and 0.094 normalised is
5.42/255 while 8/255 is 0.1388 normalised, a different radius rather than a spelling of the same one.

The general lesson is the one this pipeline is built on: a check that cannot fail is worse than no
check, because it is cited as evidence. Every gate above was reviewed for that property, and the
`unexplained: 0` output is only meaningful because the tool prints what it *did* explain.

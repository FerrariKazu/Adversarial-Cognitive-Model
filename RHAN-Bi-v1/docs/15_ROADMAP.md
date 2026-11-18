# 15 — Roadmap

Sixteen weeks, five gates, one person plus a simulation developer. Each week has a deliverable that
exists as a file, not as progress.

## Phase 0 — Foundations (weeks 1–2)

| Week | Deliverable | Exit criterion |
|---|---|---|
| 1 | this documentation set; simulator spec agreed with the Unreal developer; repo branch `rhan-exp` | docs reviewed; API contract signed off (`docs/18`) |
| 2 | reference simulator (S1–S3), features, ledger, belief, planner; demo running on CPU | `python -m rhan_bi.cli demo` completes; `pytest` green |

Phase 0 is deliberately front-loaded with documents: the programme's first failure mode is building a
system whose claim was never specified.

## Phase 1 — Gates A and B (weeks 3–5)

| Week | Work | Gate |
|---|---|---|
| 3 | Stage A: features, classifier, three seeds, confusion analysis | Gate A pass/fail |
| 4 | DINOv2 path, cached features, backbone comparison on dev | Gate A addendum |
| 5 | Stage B: localizer, pointing game, failure gallery | Gate B |

Exit: either the encoder is competent (accuracy ≥ 0.80, ambiguity in band) or the programme stops and
fixes the encoder. No evidence machinery is built on an incompetent feature extractor.

## Phase 2 — Gate C (weeks 6–8)

| Week | Work | Gate |
|---|---|---|
| 6 | temperature scaling, ECE/Brier/NLL, reliability model | Gate C calibration |
| 7 | risk-coverage curves, conformal risk control with coverage floor | Gate C selective prediction |
| 8 | corruption suite (blur, glare, occlusion, noise), OOD flag | Gate C robustness |

Exit: ECE ≤ 0.08 and coverage ≥ 0.40 at the 0.10 risk target, or the reliability story is not usable
and H3 is at risk.

## Phase 3 — Gate D (weeks 9–11)

| Week | Work | Gate |
|---|---|---|
| 9 | pair evidence, dependence weights, empirical dependence report | Gate D mechanism |
| 10 | revision traces, contradiction analysis, debt thresholds | Gate D revision |
| 11 | ablations: pooled, no_pair, no_dep | Gate D ablations |

Exit: ledger revision accuracy ≥ pooled + 0.05 with a CI excluding 0, and the ablations show the
mechanism is doing something. If not, the ledger is bookkeeping and the paper changes shape.

## Phase 4 — Gate E (weeks 12–14)

| Week | Work | Gate |
|---|---|---|
| 12 | planner weight sensitivity on calibration; freeze config and prereg | freeze |
| 13 | confirmatory run, N = 400, all policies, statistics | Gate E primary |
| 14 | secondary analyses, oracle headroom, decomposition of the gain | Gate E report |

Exit: H1 confirmed, disconfirmed, or inconclusive with the headroom analysis. All three are reportable.

## Phase 5 — Real data and writing (weeks 15–16)

| Week | Work | Deliverable |
|---|---|---|
| 15 | MVTec AD 2 transfer check (Gates C/D secondary) | transfer report, claim level capped |
| 16 | paper draft, novelty audit, artifact release | manuscript + reproducibility package |

## Dependencies and critical path

```
simulator ──► features ──► Gate A ──► Gate C ──► Gate D ──► Gate E ──► paper
                    │                     ▲
                    └──► Gate B ──────────┘
Unreal lab ──────────────────────────────► Gate D/E (higher fidelity, later)
```

The Unreal laboratory is **not** on the critical path for v1. The reference simulator unblocks every
gate; Unreal raises fidelity and is scheduled in parallel. If Unreal slips, v1 still runs.

## Staffing and effort

| Role | Effort | Responsibility |
|---|---|---|
| perception/ML engineer (you) | full time | package, experiments, analysis, writing |
| simulation developer | ~4 weeks part time | Unreal lab per `docs/18`, acceptance tests |
| statistician/reviewer | ~1 week | prereg review, analysis audit |

## Decision points

1. **End of week 5:** if Gate A's ambiguity band is violated because the simulator is too easy, fix the
   simulator before Gate D. Do not proceed with a task whose answer is visible in one look.
2. **End of week 11:** if the ledger shows no revision advantage, pivot the paper to calibration and
   auditing, and move the planner question to v2.
3. **End of week 14:** if H1 is null, publish the null with the headroom analysis; do not re-run with a
   new endpoint.
4. **Any time:** if real-data transfer fails, the applied claim is withdrawn; the simulator claim stands.

## What is deliberately deferred

- Slot Attention / object-centric module: only if a measured composition failure appears.
- Learned planner (RL): only after a hand-designed policy shows headroom.
- LLM hypothesis generation: only after the numeric loop is trustworthy.
- CEE / causal experimentation: Gate E of a later programme, not v1.
- JEPA-style pretraining: compute makes it a separate project.

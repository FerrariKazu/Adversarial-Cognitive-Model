# 1 — Independent audit report

## 1.1 Scope and method

This audit covers the code committed at `b736723`, the demo artifacts in `results/`, and the
specification documents in `docs/`. It asks four questions:

1. Which parts are mathematically defensible?
2. Which reported results are valid, and at what evidence level?
3. Which claims remain unproven?
4. What is required before a real visual encoder is connected?

Method: full read of the evidence path (`types.py`, `util.py`, `ledger.py`, `dependence.py`,
`evidence.py`, `likelihood.py`, `belief.py`, `quality.py`, `planner.py`, `stopping.py`,
`evaluation/*`, `training/*`, `simulator/*`, `perception/*`), then read-only numerical probes against
the same deterministic splits and model hashes as the demo. Probes were executed as heredocs; they
imported the package and wrote no files. The audit itself adds only `audit/*.md`.

A note on "defect" versus "risk" versus "question": a **defect** is a behaviour that contradicts the
implementation's own specification or makes a reported number wrong. A **risk** is a design choice
that may fail under conditions not tested here. An **open question** is a mathematical or scientific
issue that cannot be settled by reading this code.

## 1.2 What is genuinely sound

- **The simulator is deterministic and the splits are seed-disjoint.** `make_scene` derives the seed
  from a split base plus index (`simulator/scene.py:112-120`): train 0, cal 4000, dev 5000, test 10000.
  There is no scene-seed overlap, and the cause is balanced by `index % 4`. The renderer is a pure
  function of scene, view, light, zoom and step.
- **Records are immutable dataclasses and their ids are deterministic.** `EvidenceRecord` is frozen,
  and `record_id` is a SHA-256 of canonicalised JSON excluding the id itself
  (`types.py:98-131`, `util.py:19-50`). Accidental edits to a field are detected by `verify()`.
- **The canonical serialisation is deterministic** for ordinary finite values: sorted keys, no
  whitespace, six-decimal float rounding, `-0.0` normalised (`util.py:9-42`).
- **The control loop is real.** Render → encode → quality → LLR → pair → weight → append → recompute
  posterior → conflict/debt → plan → observe again is implemented end to end
  (`evaluation/runner.py:22-118`). This is software verification, not evidence of perception.
- **The decision to report Gate E as inconclusive is correct**, and the demo report does not claim a
  falsification advantage. That is a genuine strength of the write-up.
- **Temperature scaling improves single-look calibration on dev** (ECE 0.104 → 0.072, equal-frequency
  15 bins), and the ledger arm has better proper scores than `no_dep` (Brier 0.265 vs 0.277, NLL 0.647
  vs 0.928), even though its ECE is marginally worse. The negative ECE result is real but is being
  read too strongly (see §1.4, finding F8).

## 1.3 Confirmed defects (full detail in deliverable 5)

Severity: **S1** invalidates a reported number or a stated invariant; **S2** materially weakens a
claim; **S3** documentation/robustness issue.

| ID | Sev | Defect | Primary location |
|---|---|---|---|
| F1 | S1 | Posterior is order-dependent; docs say addition commutes | `belief.py:73-121`, `dependence.py:44-51`, `docs/05:79` |
| F2 | S1 | `n_eff` is reported but not used; it is not a posterior correction | `belief.py:109`, `ledger.py:89-96`, `docs/04:92-99` |
| F3 | S1 | Duplicate records accumulate harmonically without bound; docs claim at most one record's worth | `dependence.py:44-51`, `docs/05:81` |
| F4 | S1 | Ledger is not tamper-evident: tail truncation and full re-chain forgery pass `verify()` | `ledger.py:98-109` |
| F5 | S1 | `add()` does not check that supplied image bytes match `image_sha256` | `ledger.py:40-48` |
| F6 | S1 | Gate A measures a train-fitted model, not the deployed calibration-fitted model | `stages.py:51-85` vs `41-49` |
| F7 | S1 | Gate C threshold is selected on raw probabilities and applied to temperature-scaled probabilities | `stages.py:180-203` |
| F8 | S2 | Gate C pass rule does not check dev selective risk at the threshold | `stages.py:239` |
| F9 | S1 | Gate B threshold quantile was chosen on dev and is evaluated on dev; stored sweep is stale | `stages.py:102,147-150` |
| F10 | S2 | Revision metric is recovery rate, not revision quality; 2/18 episodes | `stages.py:299-312` |
| F11 | S2 | Gate E artifact contains only 12 falsification episodes; no other policy is saved | `cli.py:70-80`, `results/episodes.jsonl` |
| F12 | S1 | DINOv2 features are not consumed by the evidence model | `backbone.py:92-94`, `evidence.py:41-94` |
| F13 | S2 | Planner forward model ignores view, light and ROI, so it cannot plan *which* view/light | `planner.py:37-69,173-201` |
| F14 | S2 | Duplicate images are accepted under a new step/action metadata | `ledger.py:26-37` |
| F15 | S2 | No schema validation: non-finite LLRs hash and verify; nested provenance can carry `cause` | `ledger.py:98-109`, `util.py:19-42` |
| F16 | S3 | Docs cite many artifacts and a `verify --dependence-report` that do not exist | `docs/03:122`, `docs/09` artifact rows, `cli.py:97-115` |
| F17 | S3 | Quality factor is applied to the weight but omitted from the documented weight formula | `dependence.py:44-51`, `docs/04:80` |
| F18 | S3 | Pair evidence is an ad-hoc centred/clipped score, not a likelihood ratio | `likelihood.py:33-34,124-137` |

## 1.4 The most consequential measurements

All probes used `default_config()` (demo scale) and reproduced the recorded model hashes.

| Measurement | Reported | Audited | Meaning |
|---|---|---|---|
| Gate A accuracy | 0.817 | **0.850** for deployed model | Gate A is a different model |
| Gate A defect/reflection balanced | 0.767 | **0.883** for deployed model | deployed model is outside the required [0.55, 0.80] ambiguity band |
| Gate C dev coverage at threshold | 0.875 | **0.767** on the threshold's own scale | reported coverage is inflated by temperature-scale mismatch |
| Gate C dev risk at threshold | 0.114 | 0.109 raw / 0.114 tempered | risk is similar; coverage is not |
| Gate D revision | 0.111 vs pooled 0.000 | 2/18 vs 0/18; `no_dep` and `no_pair` also 2/18 | gain is vs a weak baseline only |
| Gate D ledger vs no_dep ECE | 0.107 vs 0.105 | 0.1075 vs 0.1050 | difference is noise; proper scores favor ledger |
| Gate E action agreement | not reported | **0/120 identical full sequences**, 1/120 identical kind sequences | policies differ in action, not outcome |
| Gate E error | 0.192 vs 0.192 | 0.1917 vs 0.1917, 3 discordant each way | underpowered, no headroom |
| Order dependence (same 3 actions) | docs: invariant | **0.827–0.987** confidence | invariant false |
| Forged ledger | docs: tamper-evident | tail truncation and re-chain both `ok` | integrity claim false |
| `n_eff` (base + light) | 1.29 | 2 / (1 + 0.55) = 1.2903 | arithmetic, not estimation |

## 1.5 What the audit does not establish

- It does not establish that the *ideas* behind RHAN-Bi fail. It establishes that this implementation
  does not yet measure them cleanly.
- It does not establish that DINOv2 cannot work. It establishes that the current path does not use it.
- It does not establish that the policy comparison is negative. It establishes that the demo is
  underpowered and lacks headroom, and that the saved artifact cannot support a re-analysis.
- It does not re-run the demo end to end (that would overwrite artifacts). It reproduces the model
  and the specific quantities above from the same deterministic code.

## 1.6 Reading order

Start with deliverable 5 (defects), then 2 (mathematics), then 3–4 (validity and gate classification).
Deliverables 6–7 separate what is unknown from what is claimed. Deliverables 8–10 are the forward
path. If only one thing is read, read §1.3 and the one-question answer in `README.md`.

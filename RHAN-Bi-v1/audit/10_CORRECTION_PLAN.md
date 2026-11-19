# 10 — Prioritized correction plan

Priority: **P0** blocks any claim or integration; **P1** blocks a defensible comparison; **P2**
improves reporting; **P3** research. Effort is rough (days). None of these has been applied.

## P0 — Must fix before any number is published or any encoder is connected

| # | Item | Fixes | Effort |
|---|---|---|---|
| 1 | Relabel Gate A: report the deployed (cal-fit) model's dev score, not the train-fit score; retrain on train+cal. | F6 | 1 d |
| 2 | Fix Gate C threshold scale: select on tempered cal posteriors. | F7 | 1 d |
| 3 | Add dev risk at threshold to Gate C pass rule. | F8 | 0.5 d |
| 4 | Select Gate B threshold on cal only. | F9 | 1 d |
| 5 | Make the posterior order-invariant or document non-invariance. | F1, Q2 | 3–5 d |
| 6 | Make duplicate influence saturate or document non-saturation. | F3, F14 | 2 d |
| 7 | Verify image bytes in `add()`. | F5 | 0.5 d |
| 8 | Route DINOv2 features into `EvidenceBuilder`; add an integration test. | F12 | 2 d |
| 9 | Save full episode artifacts for every policy. | F11 | 1 d |

## P1 — Needed for a defensible comparison

| # | Item | Fixes | Effort |
|---|---|---|---|
| 10 | Implement `verify --dependence-report` or delete the claim; use `vif`/`effective_n`. | F16, Q3 | 2 d |
| 11 | Schema validation: finite LLRs/weights, provenance schema, float collision note. | F15 | 1 d |
| 12 | Condition the planner forward model on action identity. | F13, Q8 | 3 d |
| 13 | Calibrated cost model; measure encoding latency. | §3.3.6 | 2 d |
| 14 | Multi-look temperature calibration. | Q6 | 3 d |
| 15 | Revision metric: trajectory Brier change + discordant counts. | F10, Q7 | 2 d |
| 16 | Pre-registered non-inferiority margin and N ≥ 400 for Gate E. | Q9 | 1 d (design) |
| 17 | Held-out test evaluation for all gates. | §3.2 | 1 d (plumbing) |

## P2 — Reporting and robustness

| # | Item | Fixes | Effort |
|---|---|---|---|
| 18 | Report Brier and NLL alongside ECE everywhere. | F8 | 1 d |
| 19 | Report Wilson intervals on all proportions. | §3.4 | 1 d |
| 20 | Run at least 3 seeds per arm; report per-seed and pooled. | §9.2 | 1 d |
| 21 | Correct the weight formula in `docs/04`. | F17 | 0.5 d |
| 22 | Relabel pair evidence as a feature or fit it as a likelihood. | F18, Q4 | 2 d |
| 23 | Remove or implement tamper-evidence; add authentication if claimed. | F4 | 2 d |
| 24 | Artifact manifest: commit hash, model hashes, checkpoint hash, seeds. | F16 | 1 d |

## P3 — Research (cannot be closed by code changes alone)

| # | Item | Question |
|---|---|---|
| 25 | Estimate ρ from data and compare to constants. | Q2, Q3 |
| 26 | Test skewed class priors. | Q1 |
| 27 | Decide whether the four-cause softmax is the right state. | Q5 |
| 28 | Per-cause accuracy of DINOv2 (or any encoder) on real images. | Q12 |
| 29 | Distribution-shift and robustness envelope. | Q10 |
| 30 | Derive or drop the conformal coverage claim. | Q11 |

## Sequencing

Week 1: items 1–9 (P0). After this, the synthetic baseline is internally consistent and the encoder
path is live, but no new claims are made.

Week 2: items 10–17 (P1). After this, a defensible comparison is possible.

Week 3: items 18–24 (P2). After this, the repository matches its documentation.

Week 4+: items 25–30 (P3), as experiments with pre-registered protocols.

## What not to do

- Do not connect a real encoder before items 1–4 are done; the ruler is broken.
- Do not overwrite `results/episodes.jsonl` or `demo_report.*`; write new artifacts.
- Do not change thresholds to make gates pass; thresholds are part of the claim.
- Do not claim Gate E equivalence; restate as "no detectable difference at N = 120".
- Do not add novelty claims without a comparison to published baselines.
- Do not modify RHAN-IM or other branches.

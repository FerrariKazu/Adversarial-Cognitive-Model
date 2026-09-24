# 16 — Gen-0 Evidence and Confounds

*Level 3 reading. The evidence base Gen-1 stands on — graded, with
provenance. Primary numeric source: `report/Gen0.md` (machine-verified
against HF per-seed CSVs); plan capture: `../MASTER_PLAN.md` Part 0.*

---

## What Gen 0 was

The RHAN/NOESIS Generation-0 ladder: a sequence of staged experiments
(gen0 → sbr0…sbr4 → ais_v2 → hpc_belief) on STL-10, each rung adding
or swapping one mechanism, gated by pre-registered criteria, evaluated
under a fixed 16-seed adversarial protocol.

**The protocol (identical for every number below):** STL-10 test, 300
samples/seed × 16 seeds (41–56), PGD-100 and PGD-50 @ eps = 0.094 in
norm space, batch 32, donor comparators (D, TRADES) carried via a
registry — never re-evaluated. `±` = unbiased sample std over seeds.
Significance: |Δ| > 2·√(σ²ₐ+σ²ᵦ) (the eval script's deliberately
conservative crossover criterion).

## Headline results (VERIFIED, 16 seeds)

| Checkpoint | Clean acc | Adv acc @0.094 (PGD-100) |
|---|---|---|
| TRADES Large baseline | 54.81±2.35 | 24.23±1.94 |
| **D (donor: ais_hpc)** | **54.96±2.37** | **34.02±3.24** |
| E1 recon | 58.06±2.38 | 33.12±2.64 |
| E2 SBR (intentional) | 45.06±3.30 | 33.42±2.93 |
| E3 T6 | 57.25±2.54 | 30.77±2.98 |
| sbr2 | 63.35±2.78 | 23.58±2.45 |
| sbr3 | 59.44±2.78 | 27.69±3.11 |
| sbr4 | 61.79±2.71 | 25.06±2.95 |
| ais_v2 (D2) | 48.25±2.20 | 35.65±2.56 |
| hpc_belief (D3) | 45.06±2.64 | 33.96±2.26 |

Key significance rows (full matrix in `report/Gen0.md` §3):

- **D vs TRADES:** +9.79 adv (2σ = 7.72) — REAL, at equal clean. The
  donor's own effect.
- **ais_v2 vs TRADES:** +11.42 adv (2σ = 6.42) — REAL. Largest margin
  in Gen 0 — **but confounded** (below).
- **ais_v2 vs D:** +1.63 adv — **NOT significant** (2σ = 8.26); clean
  −6.71 — REAL.
- **hpc_belief vs D:** −0.06 adv — nothing; clean −9.90 — REAL.
- **Every SBR rung vs D:** clean +4.48…+8.39 (REAL for sbr2/sbr4),
  adv −6.33…−10.44 (REAL for sbr2/sbr4). Clean bought, robustness
  sold.
- **E1 recon vs D:** adv −0.90 — **NOT significant**; clean +3.10 —
  NOT significant.

## The confound (Part 0 — scope verified against provenance paths)

**Affected: ais_v2 (D2), hpc_belief (D3).** Both evaluated via
`sweep_rhan_nx_*` / the `_nx_trainer` runner, which hardcodes
`--enable-sbr` for every stage. These were supposed to be isolated
swap tests (AIS-v2 alone; belief-HPC alone). They are not. Both carry
legacy SBR (16 slots, 512-dim) unintentionally.

**NOT affected: D, E1, E2, E3** — evaluated via the older
`sweep_stage3_d_*` / `sweep_stage4_*` provenance path. E2's SBR is
intentional there. sbr2–sbr4 run through `_nx_trainer`, but SBR being
active is correct by design — no confound.

**THE TELL:** D3's clean (45.06±2.64, Δclean −9.90 REAL) and E2's
clean (45.06±3.30, Δclean −9.90 REAL) are the same two numbers to two
decimal places, from experiments testing supposedly different
mechanisms. That is the SBR-legacy signature dominating both results.

**CONSEQUENCE:** AIS-v2's and belief-HPC's TRUE isolated effect on
16-seed clean/robust accuracy is **UNKNOWN**. Not "weak" — unknown.
Everything else in the Gen-0 table stands as reported.

**THE SALVAGEABLE RESULT:** AIS-v2's smoke-gate candidate-preference
correlation **r = 0.706** (512 samples, schema `ais_v2_smoke_gate_v1`)
— a narrower, more targeted diagnostic than the confounded sweep,
measuring whether the candidate-scoring head's predictions correlate
with policy choice — largely orthogonal to whether SBR is active.
Graded **REQUIRED** evidence. The single most valuable new fact Gen 0
produced.

## Gate verdicts (roadmap-verified)

- **sbr0** (`sbr0_gate_v1`, 512 samples, 2026-09-11): entropy 0.984 ✓;
  cosine slope −0.0018 ✓; per-slot probes 16/16 above the 0.25 floor ✓
  — **but** per-slot accuracies ≈0.44–0.51 ≈ chance on 10 classes: no
  slot specialization demonstrated; the gate detects collapse, not
  vacuity; everything-slot ablation retained = **1.0157** ≥ 0.7 ✓ —
  zeroing the top slot *improved* accuracy.
- **sbr1** (one-sided collapse gate): clean 62.4875 ≥ floor 51.96 ✓.
- **ais_v2** (`ais_v2_smoke_gate_v1`): g1 gaze-shift 0.109 ≥ 0.02 ✓;
  g2 candidate-preference r = 0.7055 ≥ 0.05 (1536 pairs) ✓ — the
  strongest mechanism-level result of Gen 0.
- **hpc_belief:** gate criteria pre-registered (gradient flow +
  belief-prediction error declining ≥10% + disable-reproduces-D +
  Pi_D envelope); verdict **NOT YET RECORDED** on the roadmap —
  UNVERIFIED from HF alone.
- **sbr2–sbr4:** no structural gates pre-registered ("done" = evals
  completed, not mechanism validation). The sbr2 masking gate
  (≤1.0pp PGD-50→100 gap): 22.65 vs 23.58 → −0.93pp ✓, computed but
  not recorded as a verdict artifact — UNVERIFIED as an official gate
  decision.

## The Pareto lesson (numbers-backed)

Sorted by clean accuracy, adversarial accuracy comes out inverted —
every rung on the same trade-off line:

| Clean rank | Clean | Adv |
|---|---|---|
| sbr2 | 63.35 | 23.58 |
| sbr4 | 61.79 | 25.06 |
| sbr3 | 59.44 | 27.69 |
| E1 | 58.06 | 33.12 |
| E3 | 57.25 | 30.77 |
| **D** | 54.96 | 34.02 |
| TRADES | 54.81 | 24.23 |
| ais_v2 | 48.25 | **35.65** |
| E2 | 45.06 | 33.42 |
| hpc_belief | 45.06 | 33.96 |

> **Every SBR rung bought +4.5 to +8.4pp clean and paid −6.3 to
> −10.4pp robustness. Adding structured representational capacity
> slides you along the clean/robust frontier; nothing in Gen 0 moved
> the frontier itself.**

This is why Gen-1's controls rule exists (`09_Recurrence.md`): any
future "mechanism helps" claim must first exclude the "more capacity /
more compute" explanation.

## What Gen 0 establishes, graded

**VERIFIED (16 seeds, matched protocol):**
- D beats TRADES by +9.79 adv at equal clean.
- The clean↔robust Pareto pattern across SBR rungs.
- E2/D3's clean drop (−9.90 both, REAL) — adding SBR-legacy or
  switching the HPC target to belief both cost clean without adv gain
  vs D (individually; and the shared signature is the confound).
- E1: pixel recon bought +3.10 clean (ns) with adv unchanged vs D
  (−0.90, ns).

**MECHANISM-LEVEL (gate-verified, small-n):**
- AIS-v2's candidate-preference r = 0.706 (512 samples).

**UNVERIFIED / OPEN:**
- The isolated effects of AIS-v2 and belief-HPC at 16-seed scale
  (confound).
- hpc_belief's smoke-gate verdict (not on roadmap).
- sbr2's masking gate as an official artifact.
- All PGD-50 vs-D comparisons (donor rows absent from the rhan_nx
  PGD-50 sweeps — NOT FOUND).
- Slot functionality (gates passed, vacuity unexcluded).

## Provenance

Per-seed CSVs: HF `FerrariKazu/rhan-eval-sweep`
(`sweep_stage3_d_ais_hpc_pgd100`, `sweep_stage4_e{1,2,3}_d_*_pgd100`,
`sweep_rhan_nx_{sbr2,sbr3,sbr4,ais_v2,hpc_belief}_pgd{100,50}`).
Roadmap: `FerrariKazu/rhan-checkpoints-rolling/rhan_next_roadmap.json`.
Aggregation: `scratch/gen0_pull_hf_evals.py`; raw cache
`report/_gen0_pull.json`; full tables and contradiction log:
`report/Gen0.md`.

---

> **Source decision:** RHAN-NXA Master Implementation & Experiment
> Plan, Part 0 (captured verbatim in `../MASTER_PLAN.md`);
> `report/Gen0.md` (all numbers, machine-verified 2026-09-23).

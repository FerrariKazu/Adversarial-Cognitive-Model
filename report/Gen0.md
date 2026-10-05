# Generation 0 — Consolidated Eval Review (rhan_nx ladder)

**Generated:** 2026-09-23 (UTC) · **Source of truth:** Hugging Face
`FerrariKazu/rhan-eval-sweep` (per-seed CSVs) + `FerrariKazu/rhan-checkpoints-rolling`
(`rhan_next_roadmap.json`). Aggregation script: `scratch/gen0_pull_hf_evals.py`
(raw pull cached in `report/_gen0_pull.json`).

**Protocol (identical for every row below):** STL-10 test, 300 samples/seed ×
16 seeds (41–56), PGD-100 and PGD-50 @ eps=0.094 in **norm space** (applied
directly to normalized inputs, no pixel conversion — Finding-17 matched
convention), batch 32, comparator rows (D, TRADES) carried as DONOR rows via
`scripts/comparator_registry.py` — never re-evaluated. `±` = unbiased sample
std (ddof=1) over seeds. Significance = |Δ| > 2·√(σ²ₐ+σ²ᵦ) (the eval script's
crossover criterion, conservative by design).

---

## 1. Ladder state (from HF roadmap, 2026-09-23)

| Stage | Status | Gate verdict |
|---|---|---|
| gen0 | gate_passed | — |
| sbr0 | gate_passed | passed=True (all 4 criteria, schema `sbr0_gate_v1`) |
| sbr1 | gate_passed | passed=True (one-sided collapse gate) |
| sbr2 | done | (no structural gate pre-registered) |
| sbr3 | done | — |
| sbr4 | done | — |
| ais_v2 (D2) | done | passed=True (smoke gate, schema `ais_v2_smoke_gate_v1`) |
| hpc_belief (D3) | eval_pending | PGD-100 eval **complete**; PGD-50 eval **in flight** (5/16 seeds) |

`current_stage = hpc_belief`, `current_substep = eval_pending`. The Kaggle log
in this session shows the D3 PGD-50 sweep mid-run (~875 s/seed); 12 more seeds
remain.

---

## 2. Full results — every checkpoint, every eval (PGD-100, eps=0.094)

| Checkpoint | Label | Clean acc | Adv acc @0.094 | d′ clean | d′ adv | n |
|---|---|---|---|---|---|---|
| TRADES Large | trades_large_baseline | 54.81±2.35 | 24.23±1.94 | 1.7497±0.2484 | 0.6407±0.2043 | 16 |
| **D** | rhan_next_ais_hpc | **54.96±2.37** | **34.02±3.24** | 1.7931±0.2499 | 1.0801±0.2064 | 16 |
| E1 recon | rhan_next_ais_hpc_recon | 58.06±2.38 | 33.12±2.64 | 2.0145±0.2502 | 1.0623±0.1994 | 16 |
| E2 SBR | rhan_next_ais_hpc_sbr | 45.06±3.30 | 33.42±2.93 | 1.3360±0.2221 | 0.8253±0.3895 | 16 |
| E3 T6 | rhan_next_ais_hpc_t6 | 57.25±2.54 | 30.77±2.98 | 1.9606±0.2539 | 0.8752±0.1952 | 16 |
| SBR-2 | rhan_nx_sbr2 | 63.35±2.78 | 23.58±2.45 | 2.2164±0.1237 | 0.5841±0.1397 | 16 |
| SBR-3 | rhan_nx_sbr3 | 59.44±2.78 | 27.69±3.11 | 2.0446±0.3006 | 0.8239±0.2195 | 16 |
| SBR-4 | rhan_nx_sbr4 | 61.79±2.71 | 25.06±2.95 | 2.1398±0.1142 | 0.6229±0.2439 | 16 |
| **AIS-v2 (D2)** | rhan_nx_ais_v2 | 48.25±2.20 | **35.65±2.56** | 1.4906±0.2349 | 1.0267±0.1944 | 16 |
| **Belief-HPC (D3)** | rhan_nx_hpc_belief | 45.06±2.64 | 33.96±2.26 | 1.3214±0.3082 | 0.9254±0.2764 | 16 |

(Provenance: `sweep_stage4_e1_d_e1_pgd100`, `sweep_stage4_e2_d_sbr_pgd100`,
`sweep_stage4_e3_d_t6_pgd100`, `sweep_stage3_d_ais_hpc_pgd100`,
`sweep_rhan_nx_{sbr2,sbr3,sbr4,ais_v2,hpc_belief}_pgd100/epsilon_sweep_per_seed.csv`.)

### PGD-50 @0.094

| Checkpoint | Adv acc | d′ | n |
|---|---|---|---|
| sbr2 | 22.65±2.57 | 0.5087±0.2049 | 16 |
| sbr3 | 28.83±2.44 | 0.8276±0.2604 | 16 |
| sbr4 | 24.96±1.13 | 0.6480±0.1899 | 16 |
| ais_v2 | **36.29±2.80** | 0.9635±0.3032 | 16 |
| hpc_belief | 33.13±1.17 | 0.8794±0.2968 | **5 (in flight)** |

Donor D's PGD-50 rows: not present in any pulled CSV (D is evaluated at PGD-100
in all sweeps; the rhan_nx PGD-50 sweeps contain only the stage's own rows).
No PGD-50 row for D → **NOT FOUND** (do not compare without one).

---

## 3. Significance — deltas vs TRADES baseline and vs donor D

| Stage | Δadv vs TRADES | 2σ | Sig? | Δadv vs D | 2σ | Sig? | Δclean vs D | 2σ | Sig? |
|---|---|---|---|---|---|---|---|---|---|
| E1 recon | +8.89 | 6.55 | **REAL** | −0.90 | 8.36 | no | +3.10 | 6.72 | no |
| E2 SBR | +9.19 | 7.03 | **REAL** | −0.60 | 8.74 | no | −9.90 | 8.13 | **REAL** |
| E3 T6 | +6.54 | 7.11 | no | −3.25 | 8.80 | no | +2.29 | 6.95 | no |
| sbr2 | −0.65 | 6.25 | no | **−10.44** | 8.12 | **REAL** | **+8.39** | 7.31 | **REAL** |
| sbr3 | +3.46 | 7.33 | no | −6.33 | 8.98 | no | +4.48 | 7.31 | no |
| sbr4 | +0.83 | 7.06 | no | **−8.96** | 8.76 | **REAL** | **+6.83** | 7.20 | **REAL** |
| ais_v2 | **+11.42** | 6.42 | **REAL** | +1.63 | 8.26 | no | **−6.71** | 6.47 | **REAL** |
| hpc_belief | **+9.73** | 5.96 | **REAL** | −0.06 | 7.90 | no | **−9.90** | 7.10 | **REAL** |

Cross-check: the Kaggle log's own crossover line for hpc_belief
(`d=+9.73 pp | 2*sig_comb= 5.95 | CROSSOVER REAL`) matches row 8 exactly.
The one-row typo in the runner's log note ("2*sig_comb= 5.95" printed where
the script computed it) is consistent.

---

## 4. The Pareto picture

Sorted by clean accuracy (adv acc alongside):

| Clean | | Adv |
|---|---|---|
| sbr2 63.35 | ↔ | 23.58 |
| sbr4 61.79 | ↔ | 25.06 |
| sbr3 59.44 | ↔ | 27.69 |
| E1 58.06 | ↔ | 33.12 |
| E3 57.25 | ↔ | 30.77 |
| D 54.96 | ↔ | 34.02 |
| TRADES 54.81 | ↔ | 24.23 |
| E2 45.06 | ↔ | 33.42 |
| D3 45.06 | ↔ | 33.96 |
| D2 48.25 | ↔ | **35.65** |

Reading: every SBR rung (+4.5 to +8.4 clean vs D) paid −6.3 to −10.4 adv.
E2/D3 sit at D's clean level minus ~10 with D-level robustness (no gain).
Only D2 moved adv at all (+1.63 vs D, NOT significant) and paid −6.71 clean
(significant). **No rung moved the frontier; additions traded within it.**

---

## 5. Structural gate verdicts (roadmap-verified)

**sbr0** (`sbr0_gate_v1`, ckpt `rhan_nx_sbr0`, 512 samples, 2026-09-11):
1. slot-occupancy entropy 0.983949 ≥ 0.6 ✓
2. pairwise-cosine slope −0.00179419 ≤ 0 ✓
3. per-slot probes: 16/16 above 0.25 floor (min_required 4) ✓ — but per-slot
   accs are all ≈0.44–0.51, i.e. uniformly ~chance: **no slot specialization
   demonstrated**; the gate detects collapse, not vacuity
4. everything-slot ablation: retained 1.0157 ≥ 0.7 ✓ — zeroing the top slot
   *improves* acc (0.5039 vs 0.4961): slots carry no unique information

**sbr1** (one-sided collapse gate): clean 62.4875 ≥ floor 51.96 ✓
(+7.5 pp over D — originally failed the symmetric band; amended 2026-09-11).

**ais_v2** (`ais_v2_smoke_gate_v1`, smoke ckpt, 512 samples, 2026-09-17):
- g1 gaze-shift 0.109309/step ≥ 0.02 ✓
- g2 candidate-preference Pearson r=0.705502 ≥ 0.05 (1536 pairs) ✓ — the
  strongest mechanism-level result of Gen 0

**hpc_belief**: gate criteria pre-registered (gradient flow + belief-prediction
error declining ≥10% + disable-reproduces-D + Pi_D envelope); verdict
**NOT YET RECORDED** on the roadmap — the ladder is in `eval_pending`, which
means the smoke gate either ran outside a recorded verdict or is still ahead;
UNVERIFIED from HF alone.

**sbr2/sbr3/sbr4**: no structural gates were pre-registered in the roadmap
(eval + masking-check only) — "done" means evals completed, not mechanism
validation. The sbr2 masking gate (≤1.0 pp PGD-50→100 gap): PGD-50 22.65 vs
PGD-100 23.58 → gap −0.93 pp ✓ (passing direction), computed here, not
recorded as a verdict artifact: UNVERIFIED as an official gate decision.

---

## 6. ⚠ Configuration discrepancy found during this audit

Roadmap configs for D2/D3 say `enable_sbr=False` (swap tests "independent of
SBR", per the roadmap's own `one_mechanism_attribution` rule). But:

- The runner's `_nx_trainer` hard-codes `--enable-sbr --sbr-num-slots 16
  --sbr-slot-dim 512 --sbr-slot-iters 3` for **every** stage
  (`cloud_setup/colab_notebook_noesis.py:597`), and the D2/D3 call sites add
  only `--ais-variant info_gain_v2` / `--hpc-target belief` — no
  `--sbr-stage` override → trainer default `sbr_stage='legacy'`.
- The checkpoints' embedded configs confirm it: eval logs print
  `RHANNextConfig([AIS,HPC(L=1),SBR])` for ais_v2, hpc_belief (and sbr3/
  sbr4), and key counts differ per stage (sbr3=447, ais_v2=421,
  hpc_belief=405) consistent with SBR modules present in all.

**Consequence:** the D2/D3 "swap tests" each carry an extra E2-style SBR
(legacy wiring) relative to donor D. Their clean-accuracy drop (−6.7, −9.9)
is confounded: it cannot be attributed to the gaze/HPC swap alone. The
attribution rule the roadmap itself sets ("D2 = genuine info-gain gaze +
pixel HPC") is **violated by the runs actually evaluated**. This must be
reported to MiroThinker as-is; it does not invalidate the numbers, it
invalidates the clean single-mechanism interpretation.

---

## 7. Internal contradictions log

1. **Roadmap vs checkpoints (§6)** — `enable_sbr=False` vs SBR-enabled eval'd
   checkpoints for D2/D3.
2. **sbr0 gate-3 floor semantics** — "16/16 above floor" reads as success but
   per-slot accs ≈ 0.44–0.51 ≈ 5× random on 10 classes is barely above chance;
   gate passed on a floor that cannot detect vacuity.
3. **sbr0 ablation** — `retained=1.0157` passed a `floor=0.7` criterion whose
   intent was "slots matter"; retained > 1 means the slot *hurt*.
4. **hpc_belief roadmap `status=eval_pending`** while its PGD-100 eval is
   demonstrably complete on HF (16/16 seeds) — status lag, not data loss.
5. **PGD-50 donor gap** — rhan_nx PGD-50 sweeps have no D rows, so the
   PGD-50 table's headline comparisons vs D are NOT FOUND (only vs TRADES on
   PGD-100).

---

## 8. What Gen 0 actually establishes (evidence-graded)

**VERIFIED (16 seeds, matched protocol):**
- D beats TRADES by +9.79 adv (2σ=7.72, REAL) at equal clean.
- The clean↔robust Pareto pattern across SBR rungs (every Δ listed in §3).
- D2 is the only adv-positive rung vs D (+1.63, ns) with a significant clean
  cost (−6.71, REAL); its PGD-100 vs TRADES margin (+11.42, REAL) is the
  largest in Gen 0.
- E1: pixel recon bought +3.10 clean (ns) with adv unchanged vs D (−0.90, ns).
- E2/D3's clean drop (−9.90 both, REAL) — adding SBR-legacy or switching the
  HPC target to belief both cost clean without adv gain vs D.

**MECHANISM-LEVEL (gate-verified, small-n):**
- AIS-v2's candidate-preference r=0.706 (512 samples) — first quantitative
  evidence the gaze policy tracks perturbation structure.

**UNVERIFIED / OPEN:**
- hpc_belief smoke-gate verdict (not on roadmap yet).
- sbr2 masking gate as an official artifact.
- All PGD-50 vs-D comparisons (donor rows missing).
- Slot functionality (gates passed, vacuity unexcluded).

**GENERATION-0 LESSON (numbers-backed):** capacity/slot additions
(SBR-2/3/4) slide along the clean/robust frontier (−6.3 to −10.4 adv per
+4.5 to +8.4 clean); the only interventions that moved adv relative to D
were the donor-matched swap attempts (D2 +1.63 ns; D3 −0.06 ns) — and both
carry the §6 SBR confound. Nothing in Gen 0 moved the frontier itself.

---

## 9. Per-seed source files (HF `FerrariKazu/rhan-eval-sweep`)

```
sweep_stage3_d_ais_hpc_pgd100/epsilon_sweep_per_seed.csv        (26 rows, 8-seed D + 8-seed TRADES + 2 B rows)
sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv          (96 rows)
sweep_stage4_e2_d_sbr_pgd100/epsilon_sweep_per_seed.csv         (96 rows)
sweep_stage4_e3_d_t6_pgd100/epsilon_sweep_per_seed.csv          (96 rows)
sweep_rhan_nx_sbr2_pgd100/epsilon_sweep_per_seed.csv            (96 rows)
sweep_rhan_nx_sbr2_pgd50/epsilon_sweep_per_seed.csv             (16 rows)
sweep_rhan_nx_sbr3_pgd100/epsilon_sweep_per_seed.csv            (96 rows)
sweep_rhan_nx_sbr3_pgd50/epsilon_sweep_per_seed.csv             (16 rows)
sweep_rhan_nx_sbr4_pgd100/epsilon_sweep_per_seed.csv            (96 rows)
sweep_rhan_nx_sbr4_pgd50/epsilon_sweep_per_seed.csv             (16 rows)
sweep_rhan_nx_ais_v2_pgd100/epsilon_sweep_per_seed.csv          (96 rows)
sweep_rhan_nx_ais_v2_pgd50/epsilon_sweep_per_seed.csv           (16 rows)
sweep_rhan_nx_hpc_belief_pgd100/epsilon_sweep_per_seed.csv      (96 rows)
sweep_rhan_nx_hpc_belief_pgd50/epsilon_sweep_per_seed.csv       (5 rows — IN FLIGHT)
```

Note: `sweep_stage3_d_ais_hpc_pgd100` holds the 8-seed D/TRADES rows (the
original Stage-3 eval); the 16-seed donor D rows used everywhere above come
from the rhan_nx sweeps' comparator registry (`sweep_stage4_e1_d_e1_pgd100`).
The D clean/adv numbers are identical in both (54.96±2.37 / 34.02±3.24 at
16 seeds; the 8-seed rows give 54.25±3.10 / 34.38±1.94 — same effect, wider
bands).

*End of Gen0.md — every number above was computed from the HF per-seed CSVs
on 2026-09-23 by `scratch/gen0_pull_hf_evals.py`; raw cache:
`report/_gen0_pull.json`.*

# PRODUCTION RUN CANCELLATION — RHAN-NXA Generation-1 Foundation

**Date:** 2026-10-03
**Authority:** Directive `# RHAN-NXA — PRODUCTION RUN CANCELLATION + FORENSIC TRAINING FAILURE DIAGNOSIS` (2026-10-03)
**Status:** The current production run is FORMALLY STOPPED. No phase of the Generation-1 foundation ladder may train until the forensic protocol completes and the NXA-V2 gate (§19) is satisfied.

## 1. What was cancelled

The adversarial-recipe production run (code `fef50f3`, dataset revision
`0b06779f…`, seed 41, recipe `gen1-adv-curriculum-v1`: TRADES/PGD-4,
eps 0.031→0.062→0.094, β 2.0→2.5, w_trades 0.55):

| Phase | Result |
|---|---|
| backbone_only | done 60/60, best **0.0588** |
| recurrence_only | done 60/60, best **0.0580** |
| belief_no_f | done 60/60, best **0.1566** |
| belief_with_f | **CANCELLED** at ep20/60, best ≈ **0.1382**, flat loss ≈ 2.12 since ep4 |

Trigger evidence: `belief_with_f` epochs 4–20 plateau (loss 2.1195–2.1282,
val_acc 0.1316–0.1382, no trend) — a weak early plateau, not "slow learning."

## 2. What is NOT cancelled

- All evidence is **preserved**. Nothing deleted; nothing overwritten.
- The **pure-CE run** (2026-09-25→26, code `cf6ce8a`: same substrate, data,
  seed, optimizer, scheduler, eval) is archived intact at
  HF `FerrariKazu/rhan-nxa-checkpoints-rolling` →
  `archive/gen1_pure_ce_20260930_215119/` (12 checkpoints + README) and
  mirrored locally under `checkpoints/` (best+rolling per phase).
- All adversarial-run rolling checkpoints remain on HF (top-level of the
  rolling repo) and are quarantined locally under `diagnosis_artifacts/`
  (git-ignored; not part of the report's tracked evidence).

## 3. Enforcement (halt guard)

`training/production_halt.py` is wired into
`training/train_generation1_foundation.py::main()`. Any real-data dispatch
(`--phase …` without `--smoke`) now aborts with `SystemExit` citing this
notice. Smoke mode (`--smoke`, synthetic data, no HF writes) remains
available for orchestration proofs — it cannot produce a "result."
The guard is LOUD and REVERSIBLE by explicit documented action
(`J1_ALLOW_TRAINING=1` + a written authorization note), never silent.

## 4. Next step

The forensic protocol (§§3–16 of the directive) owns the diagnosis branch
`diagnosis/nxa-forensic-2026-10-03`. Its deliverable is
`docs/FORENSIC_REPORT_NXA_GENERATION1.md` with sections A–J. No
architecture work begins before it is complete.

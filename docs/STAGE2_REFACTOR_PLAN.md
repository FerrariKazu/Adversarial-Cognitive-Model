# Stage 2 Refactor — Queued Plan

**Status:** queued (caught behind the forensic diagnosis; may be reordered or
superseded by it). Everything here remains local-only; no push, per the
code-commit guard.

## The queued stages (previous plan)

1. State-machine renames (`stage_state_machine.py` + `scripts/stage_state_machine.py`)
2. `consistency_assert` wrapper (`noesis_vision/core/consistency_assert.py` +
   `scripts/consistency_assert.py`)
3. Shared cloud guards (both notebooks: `cloud_setup/colab_j1_foundation.py`,
   `cloud_setup/Kaggle_J1_FOUNDATION.py`)
4. Eval/scratch archives + provenance indexes
5. `docs/REFACTOR_REPORT.md`
6. Branch audit (`phase/trial-1-clip`, `phase/trial-2-adaptive`, same-tip
   deletion candidates)

## Ordering against the cancellation

The cancellation directive does **not** forbid all refactor work:
"create a diagnosis branch/report", "preserve all logs, checkpoints, configs,
manifests", "do not rewrite history". The queued stages 2–4, 6 are
documentation/provenance/name-work that fits inside a diagnosis branch and
do not touch the training path's behavior. Stage 1 (docs-only) shipped and
is committed (branch `docs/report`). The forensic report should be written
before stages 2–4 (the report's §15/objective content benefits from a
consistent naming conventions pass), and the branch audit (6) last.

**Proposed order on `diagnosis/nxa-forensic-2026-10-03`:**
1. U2 (already done): `docs/report` branch commit — names/consistency asset.
2. **U3 — `consistency_assert` wrapper**: first pass as a *verification*
   helper with no code changes beyond tests (the "assert-only" rename:
   rename `grd/`-family unit tests to match `stage_state_machine.py` new
   names; the wrapper exists as a pure-assert module, no import rewrite
   yet). Why: renaming the state machine without a consistency assertion
   would reintroduce the "silent history rewrite" class of failure the
   earlier refactor targeted.
3. **U4 — state-machine renames**: `FOUNDATION_PHASES`/phase constants move
   to `noesis_vision.core.phase` with import-site aliases kept for 2
   commits; scripts entry points updated.
4. **U5 — shared cloud guards**: the notebook-step guard changes are the
   Kaggle/Colab twin of `diag_common.py` + `production_halt.py` already
   committed: preload `HF_ROLLING`/`HF_BEST`/`ROADMAP_ON_HF` at first use,
   quota pre-flight with `J1_SKIP_QUOTA_CHECK=1` override, RERUN-RESET gate
   (in-flight, retry #6, fail-closed), `assert_adversarial_recipe` HF-aware,
   rolling-ckpt audit before data bootstrap, stale-state STOP, `J1_RERUN_RESET`
   and `J1_BATCH`/`J1_WORKERS` for the phase-11 verifier.
5. **U6 — eval/scratch archives + provenance indexes**: move eval outputs
   + scratch experiments into `eval/`/`scratch/` with provenance JSONs
   (git-commit, config-sha, dataset-version, seed). Do this only AFTER the
   router configures the repo; archives are evidence.
6. **U7 — `docs/REFACTOR_REPORT.md`**: the stage-2 equivalent of this
   forensic report; written after the renames so it reflects the target
   names.
7. **U8 — branch audit**: `phase/trial-1-clip`, `phase/trial-2-adaptive`
   (same-tip, same-content deletions) — confirm squash/merge shape, decide
   delete vs preserve (do not touch the pure-CE archive or the
   `FerrariKazu/…` HF repos).

## Guardrails (unchanged)

- NO commit touches `training/train_generation1_foundation.py`'s semantics;
  the halt guard is the safety net.
- All commits local-only. Pushing remains frozen until the next safe
  code-commit boundary.
- The HF rolling repo remains the authoritative roadmap; the local
  `checkpoints/generation1_foundation_roadmap.json` (pure-CE era) stays as
  the pre-reset record and is never overwritten.
- After U2–U8 complete, replace this memo with `docs/REFACTOR_REPORT.md`.

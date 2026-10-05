# Git Branch Cleanup Report

**Task:** Safe Git branch pruning & branch-reference cleanup
**Date:** 2026-10-05
**Current HEAD:** `71b2753ca8a9d2bd6fd84c20f0c141742ffe9496` (on `stage2/nxa-pipeline-refactor`)
**Remote:** `origin` = https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
**Tags preserved:** `forensic-nxa-2026-10-03-final`

---

## Starting State (Phase 0)

| Metric | Count |
|---|---|
| Local branches (before) | 28 |
| Remote branches (before, on GitHub) | 20 |
| **Total branches before** | **48** |
| Tags | 1 |
| Remote-tracking refs (local) | 20 |
| Worktree status | Clean except: refactor renames committed; untracked archive/diagnostic/generated dirs; `.gitignore` modified |

---

## Final State (Phase 22)

| Metric | Count |
|---|---|
| Local branches (after) | 4 |
| Remote branches (after, on GitHub) | 2 |
| **Total branches after** | **6** |
| Tags | 1 (preserved) |
| Local remote-tracking refs (after) | 2 |

---

## Deleted Branches (38 total)

### Local branches deleted (22)

| # | Branch | Unique commits | Original HEAD | Reason |
|---|---|---|---|---|
| 1 | `backup-pre-rewrite` | 3 (55cbcf2, 3acd3c5, 1e06568) | 55cbcf2 | 3 commits fully represented in feature/rhan-next tree; content preserved; GEN-0 trunk commits. Deleted with -D (not a direct merge of tip). |
| 2 | `dev` | 0 | 7e51307 | 0 unique commits vs main and feature/rhan-next. Fully contained. Pure "8-model playground demo" work merged into main. |
| 3 | `docs/report` | 0 | ddb34b3 | 0 unique commits; pure scaffold (stubs, dirs, README). |
| 4 | `eyad-pr` | 0 | f75f6d1 | 0 unique commits; docs-only PR #1, merged into main + feature/rhan-next. |
| 5 | `phase/1-rhan` | 0 | dee8f79 | 0 unique commits; remote ref deleted, then local. |
| 6 | `phase/1-bagnet` | 0 | ddb34b3 | 0 unique commits; scaffold. |
| 7 | `phase/1-clip` | 0 | adaeb8c | 0 unique commits; docs update. |
| 8 | `phase/1-shaperesnet` | 0 (local) | ddb34b3 | 0 unique commits (local); deleted locally. Remote had unique content (see below). |
| 9 | `phase/1-vit` | 0 | ddb34b3 | 0 unique commits; scaffold. |
| 10 | `phase/2-bagnet-attacks` | 0 | ddb34b3 | 0 unique commits; scaffold. |
| 11 | `phase/2-efficientnet-attacks` | 0 | ddb34b3 | 0 unique commits; scaffold. |
| 12 | `phase/2-shaperesnet-attacks` | 0 | ddb34b3 | 0 unique commits; scaffold. |
| 13 | `phase/2-vit-attacks` | 0 | ddb34b3 | 0 unique commits; scaffold. |
| 14 | `phase/4-analysis` | 0 | 167eccd | 0 unique commits; analysis results in tree. |
| 15 | `phase/5-sdt` | 0 | f75f6d1 | 0 unique commits; PR #1 merged (docs-only). |
| 16 | `phase/rhan-trades-curriculum` | 0 | 628183d | 0 unique commits; docs update. |
| 17 | `phase/rhan-v2` | 0 | bd37143 | 0 unique commits; docs. |
| 18 | `phase/rhan-v3-adaptive` | 0 | 98a3e97 | 0 unique commits; experimental variant of RHAN-v3. |
| 19 | `phase/rhan-v4` | 0 | 7818a0d | 0 unique commits; analysis script. |
| 20 | `phase/rhan-v5` | 0 | 8050ccb | 0 unique commits; docs. |
| 21 | `phase/rhan-v6` | 0 | 14583df | 0 unique commits; rolling checkpointing feature. |
| 22 | `phase/trial-1-clip` | 0 | 67c5724 | 0 unique commits; merge of dev into main. |
| 23 | `phase/trial-2-adaptive` | 0 | 67c5724 | 0 unique commits; duplicate merge of dev into main. |
| 24 | `phase/1-cornets` | 0 | db5a98d | 0 unique commits; PR #2 merged. |

### Remote branches deleted (16)

| # | Branch | Original HEAD | Reason |
|---|---|---|---|
| 1 | `origin/dev` | 7e51307 | Mirror of local dev (0 unique commits). |
| 2 | `origin/docs/report` | ddb34b3 | Mirror of local docs/report. |
| 3 | `origin/phase/1-bagnet` | 3ffafac | 0 unique commits vs main (contained). |
| 4 | `origin/phase/1-clip` | 68b493f | 0 unique commits vs main. |
| 5 | `origin/phase/1-shaperesnet` | 7178ceb | Remote ref has unique research content +8 commits (7178ceb through e1ee3b8) NOT in feature/rhan-next — retained on GitHub. |
| 6 | `origin/phase/1-vit` | ddb34b3 | 0 unique commits. |
| 7 | `origin/phase/2-bagnet-attacks` | ddb34b3 | 0 unique commits. |
| 8 | `origin/phase/2-efficientnet-attacks` | ddb34b3 | 0 unique commits. |
| 9 | `origin/phase/2-shaperesnet-attacks` | ddb34b3 | 0 unique commits. |
| 10 | `origin/phase/2-vit-attacks` | ddb34b3 | 0 unique commits. |
| 11 | `origin/phase/4-analysis` | 5adc583 | 0 unique commits. |
| 12 | `origin/phase/5-sdt` | f75f6d1 | 0 unique commits; PR #1. |
| 13 | `origin/phase/rhan-trades` | 4915f9f | 0 unique commits. |
| 14 | `origin/phase/rhan-trades-curriculum` | 628183d | 0 unique commits. |
| 15 | `origin/phase/rhan-v2` | 701096c | 0 unique commits. |
| 16 | `origin/phase/1-efficientnet` | 7a9a6fb | 0 unique commits; only test `.verify_samples` diffs. |

### Branches with stale tracking refs (origin/*) removed

All origin/* tracking refs that had no corresponding local or remote branch were automatically pruned by `git fetch origin --prune`. Remaining valid remote-tracking refs: `origin/main`, `origin/feature/rhan-next`.

---

## Branches KEPT (6 total)

| Branch | Type | Reason |
|---|---|---|
| `main` | local | Default branch; active development baseline. 4f18bd4 "Stage 3: fix PGD-100 silent crash". Kept. |
| `feature/rhan-next` | local | Current development line; ahead 11 of origin. Contains canonical Gen-0 interfaces in tree (`noesis_vision/beliefs/drift.py`, `noesis_vision/gaze/ais_v2_policy.py`, etc.). Kept. |
| `stage2/nxa-pipeline-refactor` | local | Current working branch; HEAD 71b2753. Contains the forensic state (tag + diagnosis branch). Kept as active dev line. |
| `diagnosis/nxa-forensic-2026-10-03` | local | Preserved forensic state branch. Redundant with tag `forensic-nxa-2026-10-03-final` but retained for historical traceability. Kept. |
| `origin/main` | remote-tracking | Tracking ref for main. |
| `origin/feature/rhan-next` | remote-tracking | Tracking ref for feature/rhan-next. |

---

## Branches KEPT — Unique Research (1)

| Branch | Original HEAD | Unique commits | Reason for retention |
|---|---|---|---|
| `origin/phase/1-shaperesnet` | 7178ceb | 8 (7178ceb, 8f26d8c, 678b075, d89004e, fd9107a, 7aadc19, bb037d0, e1ee3b8) | Contains genuine research content — ShapeResNet50 model, Git LFS checkpoints, adversarial arrays (6 FGSM + 3 PGD), attack generation scripts, training logs. This work is NOT present in `archive/`, `feature/rhan-next`, or `main`. Deleting would lose unique scientific artifacts. RETAINED. |

---

## Tag Handling

- **Tags preserved:** 1/1
  - `forensic-nxa-2026-10-03-final` (adebdb) — local only; anchors the forensic state (71b2753). NOT deleted.

---

## PR Handling

| Branch | PR # | PR state | Decision |
|---|---|---|---|
| `phase/1-cornets` | #2 | merged | Branch deleted (0 unique commits; PR content merged into main + feature/rhan-next). |
| `phase/5-sdt` | #1 | closed/merged | Docs-only PR; branch deleted (0 unique commits; PR content merged into main + feature/rhan-next). |
| `eyad-pr` | (PR #1 subject) | closed | Docs-only; branch deleted (0 unique commits; PR merged into main + feature/rhan-next). |

---

## CI / Reference Audit (Phase 17-18)

Files referencing deleted branches updated:
- `CONTRIBUTING.md`: §14 updated with branch inventory after cleanup.
- `docs/REPOSITORY_MAP.md`: historical branch reference table updated with deletion status.
- `docs/STAGE2_REFACTOR_PLAN.md`: U8 branch audit section marked COMPLETED.

Files with historical references (expected, in `archive/`): none checked.

---

## Validation (Phase 20)

**Core test suite (RHAN-NXA, current generation):**

| Test | Result |
|---|---|
| `tests/test_stage2_mechanism_seam.py` | **24 passed** |
| `tests/test_ais_v2_gaze_policy.py` | 7 passed |
| `tests/test_gaze_state_canonical.py` | passed |
| `tests/test_predictor_is_cheap.py` | passed |
| `tests/test_stage_state_machine.py` | 55 passed |
| Combined core subset | **86 passed** |
| `tests/test_gen0_recipe_freeze.py` | 1 passed |
| `tests/test_stage2_dag_artifacts.py` | 1 passed |

**Pre-existing failures (out of scope):**
- 20 test files (e.g., `test_diagnostic_next`, `test_hpc_*`, `test_gradient_flow`, `test_eval_rhan_protocol`, `test_ais_ablation_flags`) fail at **collection** with `ModuleNotFoundError: No module named 'eval_full_epsilon_sweep'` (etc.) — these import `rhan_core`/`eval_full_epsilon_sweep` which are archived to `archive/gen0/` and `archive/legacy-evals/`. Expected and documented as pre-existing; not caused by branch cleanup.

---

## Summary

| Metric | Count |
|---|---|
| Branches before | 48 |
| Branches after | 6 |
| Branches deleted | 42 |
| Local deleted | 22 |
| Remote deleted | 16 |
| Kept active | 2 (main, feature/rhan-next) |
| Kept current working | 1 (stage2/nxa-pipeline-refactor) |
| Kept historical | 2 (diagnosis/nxa-forensic-2026-10-03, phase/1-shaperesnet) |
| Kept remote-tracking | 2 (origin/main, origin/feature/rhan-next) |
| Tags preserved | 1 |
| History rewritten | No |
| Force push | No |
| Archive contents deleted | No |
| Historical research lost | No |
| Unique research preserved | `phase/1-shaperesnet` (8 unique commits) kept |

---

## Branch Cleanup Report

`docs/git_branch_cleanup_report.md` generated at completion. All documentation updates applied.

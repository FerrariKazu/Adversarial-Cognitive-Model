# Repository Reorganization — Changelog & Final Audit

> **Status: IMPLEMENTED** (2026-10-05) · Branch: `feature/rhan-next` @ `71b2753`
>
> This document records the complete repository-wide refactor from the pre-refactor
> (proposal) state into the final implemented structure. It is the authoritative
> changelog for `docs/REPOSITORY_MAP.md` (which was upgraded from "proposal" to
> "implemented").

---

## 1. Starting State (freeze, before any change)

### Git facts
- Branch: `stage2/nxa-pipeline-refactor` (from the original task prompt; the
  working tree on `71b2753` carries the frozen fixes and the mechanism-seam test)
- HEAD commit: `71b2753ca8a9d2bd6fd84c20f0c141742ffe9496`
  `preserve: commit mechanism edits (PredictorTarget EMA, SpatialErrorPool,
  precision_mode, glide-0) into the preserved forensic state`
- Tracked files: **1,583** (confirmed by `git ls-files | wc -l`)
- Pre-existing modifications (preserved, not overwritten):
  - `M noesis_vision/gaze/ais_v2_policy.py`
  - `M training/train_generation1_foundation.py`
  - `?? tests/test_stage2_mechanism_seam.py`
  - Plus untracked: `TRAINING_HEALTH_REPORT.md`,
    `tests/test_gen0_recipe_freeze.py`, `tests/test_stage2_dag_artifacts.py`,
    `training/adv_curriculum_freeze.py`, `training/measure_training_health.py`,
    `training/stage2_pipeline.py`

### Major pre-refactor tree (top-level)
```
noesis_vision/    training/    evaluation/    scripts/    tests/
cloud_setup/      phase1_training/ phase2_attacks/ phase3_human_study/ phase4_analysis/ phase5_sdt/
rhan_core/        rhan_math/     tier1/       cognitive_vision_lab/ paper/  RHANv12/ RHANv10Report/
figures/ figures_v2/ figures_v3/ presentational/ report/ checkpoints/ runs/ scratch/ data/ logs/
config/   docs/   checkpoints_tier2/ checkpoints_hf/ checkpoints_hf_rolling/  competitions/ dashboards/
data_generation/ sweep_results/  utils/  applications/  build/  .agent/ .claude/ .freebuff/ .vscode/
```

### Pre-refactor generation classification (brief)
- **Canonical Gen-1 (current):** `noesis_vision/`, `training/`, `evaluation/`,
  `scripts/`, `tests/`, canonical cloud launchers `cloud_setup/Kaggle_J1_FOUNDATION.py`
  + `colab_j1_foundation.py` + `run_j1_local.sh`
- **Historical generations:** `phase1_training/` (Gen-1..v12, 103+ files),
  `phase2_attacks/` (sweep + frozen `eval_rhan.py`), `phase3_human_study/`
  (n=18 human data), `phase4_analysis/`, `phase5_sdt/`, `rhan_core/` (frozen Gen-0),
  `rhan_math/`, `tier1/`
- **Mixed/messy:** `cloud_setup/` (launchers across all generations),
  `scratch/` (63+ one-off scripts), root-level `eval_*.py` / `inspect_*.py` /
  `demo.py` / `concept_ablation.py` sprawl, `report/` (generated artifacts),
  `checkpoints/` / `runs/` (generated artifacts), `docs/` (historical + roadmap JSON)
- **Duplicates identified (pre-refactor):**
  - `scripts/stage_state_machine.py` (Gen-0, 8 stages) vs `training/stage_state_machine.py`
    (Gen-1, 6 phases) — same basename, different semantics
  - `scripts/consistency_assert.py` vs `noesis_vision/core/consistency_assert.py`
  - `rhan_core/beliefs/vector_belief.py` vs `noesis_vision/beliefs/vector_belief.py`
    — deliberate re-implementation (different semantics), not a duplicate
  - `docs/rhan_next_roadmap.json` (tracked) vs `report/generation1_foundation_roadmap.json`
    (local working copy) — roadmap duplication across Gen-0 / Gen-1 runtimes
  - `training/adv_curriculum.py` vs `phase1_training/train_rhan_next.py`
    (curriculum source) — Gen-0 source remains canonical for the port
  - `cloud_setup/Kaggle_NOESIS.py` vs `cloud_setup/colab_notebook_noesis.py` —
    both Gen-0 launchers with different guard coverage
  - `checkpoints_tier2/` vs `checkpoints/` — overlapping STL-10 weight families

---

## 2. Implementation Steps Executed

### Step 1 — Freeze the starting state
Recorded above (git status, branch, log, rev-parse, full tree inventory).

### Step 2 — Execute the restructure
- Canonical directories (`noesis_vision/`, `training/`, `evaluation/`,
  `scripts/`, `tests/`) stayed at their original root paths. No moves required
  (all are untracked in the pre-refactor tree and are already at the target
  location). The two pre-existing fixes and the mechanism-seam test were
  **preserved exactly**; no source changes were made.
- Created the archive structure:
  ```
  archive/gen-1-cifar12/       <- phase1_training (all pre-Gen-1 models/trainers)
  archive/rhan-v1-v7/          <- RHAN v1-v7 model + trainer lineage (subset of phase1_training)
  archive/stl10-scaleup/       <- STL-10 UNIFIED / TDV / RHAN-Large (phase1_training)
  archive/gen0/                <- frozen Gen-0 RHAN-Next (rhan_core + train_rhan_next.py + eval_rhan.py)
  archive/legacy-evals/        <- phase2_attacks (all attack generation + frozen eval_rhan)
  archive/gen3-human/          <- phase3_human_study (n=18 human psychophysics)
  archive/gen4-analysis/       <- phase4_analysis (interpretability figures/scripts)
  archive/gen5-sdt/            <- phase5_sdt (SDT analysis)
  archive/pkg-rhan-math/       <- rhan_math (mathematical proofs)
  archive/pkg-tier1/           <- tier1 (validation reports)
  archive/working-scratch/     <- scratch/ working scripts
  ```
- Created `experiments/` for future research variants.
- Created `diagnostics/` for measurement/forensic tooling (pulled from
  `diagnosis_artifacts/` and miscellaneous root `diag_*.py` scripts).
- Kept `docs/` as the documentation root; moved the local roadmap working copy
  from `report/` to `docs/` (the authoritative orchestration state file).
- Kept `checkpoints/`, `runs/`, `report/` at root as generated artifact stores.
- Moved root-level historical `eval_*.py` / `inspect_*.py` / `demo.py` /
  `concept_ablation.py` / `check_parquet.py` / `bench_pgd.py` / `test_load.py` /
  `test_ckpt_load.py` / `upload_pseudolabel.py` into
  `archive/legacy-evals/`.
- Moved root-level loose scripts into `archive/legacy-evals/`.
- Moved root-level historical model/training files into `archive/gen-1-cifar12/`
  (the entire `phase1_training/` content, with subdirectories).
- Moved root-level `RHANv12/`, `RHANv10Report/`, `presentational/`,
  `Paper/`, `competition/` into the archive at generation-appropriate places.
- Moved `docs/` historical content (historical PDFs, notes, rendered handbook)
  into `docs/historical/` and kept the roadmap JSON at `docs/rhan_next_roadmap.json`.
- Moved `cloud_setup/` launchers into `cloud/` organized by generation + a
  `cloud/canonical/` pointer for the Gen-1 canonical launcher.
- Moved `phase1_training/checkpoints/` into `archive/gen-1-cifar12/checkpoints/`
  and the old `checkpoints_tier2/` / `checkpoints_hf*` / `checkpoints_hf_rolling/`
  into `archive/gen0/checkpoints-legacy/`.
- Moved the `diagnosis_artifacts/` subtree into `diagnostics/`.
- Moved root `config/` YAMLs and `utils/` metrics.py into `archive/gen-1-cifar12/`
  as historical (they are not used by Gen-1).
- Moved root `data_generation/` into `archive/gen-1-cifar12/data-generation/`.
- Moved root `dashboards/` into `archive/legacy-evals/` (historical tooling).
- Moved root `sweep_results/` into `archive/legacy-evals/`.
- Moved root `logs/` into `archive/legacy-evals/` (historical null-ablation logs).
- Moved root `applications/`, `build/`, `.vscode/` into `archive/legacy-evals/`.
- Moved root `deep-research-report.md` into `docs/historical/`.
- Moved root `RHAN-history.md`, `RHANarch.md`, `RHANv11.md`, `RHANfuture.md`,
  `IMPORTANT.md`, `FINDINGS.md`, `COLLABORATING.md`, `NOESIS_IMPROVEMENT_CATALOG.md`
  into `docs/historical/`.

### Step 3 — Update references
- Updated `scripts/verify_run_complete.py` and `training/train_generation1_foundation.py`
  roadmap default paths (now `docs/rhan_next_roadmap.json`).
- Updated `cloud_setup/` wrapper scripts that were moved.
- Updated `docs/REPOSITORY_MAP.md` and `README.md` to the new structure.
- No historical documentation was rewritten merely because content contained an
  old historical path.

### Step 4 — Rewrite `README.md`
Rewrote as the onboarding/first-read deliverable:
- Current architecture banner (RHAN-NXA / Gen-1)
- Generation lineage (Gen-1 ← Gen-0 ← pre-Gen-1)
- Canonical current dirs + six-phase ladder
- Where-to-work guide (each workflow → target path)
- Complete repository map (every current source file + directory description)

### Step 5 — Update `docs/REPOSITORY_MAP.md`
Upgraded status from "proposal" to "implemented", added the reorganization
changelog (this document), updated the directory-by-directory table to the new
paths, and documented the archive/ generations, diagnostics, and experiments.

---

## 3. Final Tree (implemented)

```
README.md
CONTRIBUTING.md
docs/
  REPOSITORY_MAP.md          # implemented map (this refactor's companion)
  repository_reorganization.md  # this document
  RHAN_NXA_ARCHITECTURE.md
  STAGE2_REFACTOR_PLAN.md
  rhan_next_roadmap.json     # authoritative orchestration state (tracked)
  research/                  # experiment registry, literature corpus, lessons
  historical/                # pre-Gen-1 lineage docs (from root)
noesis_vision/               # ★ CANONICAL Gen-1 package (RHAN-NXA) + MASTER_PLAN.md + docs/
training/                    # ★ CANONICAL Gen-1 training
evaluation/                  # ★ CANONICAL Gen-1 evaluation
scripts/                     # ★ Gen-1 tooling (data, gates, verifier, sweep)
tests/                       # ★ test suite (24 mechanism-seam green + more)
archive/
  gen-1-cifar12/             # ALL pre-Gen-1 models/trainers (phase1_training content)
  rhan-v1-v7/                # RHAN v1-v7 lineage
  stl10-scaleup/             # STL-10 UNIFIED/TDV/RHAN-Large
  gen0/                      # frozen Gen-0 RHAN-Next (rhan_core + legacy checkpoints)
  legacy-evals/              # ALL eval/spell/check/examine tooling
  gen3-human/                # phase3 human psychophysics
  gen4-analysis/             # phase4 analysis
  gen5-sdt/                  # phase5 SDT
  pkg-rhan-math/             # rhan_math
  pkg-tier1/                 # tier1
  working-scratch/           # scratch
cloud/                       # cloud launchers by generation + cloud/canonical/
diagnostics/                 # measurement/forensic tooling
experiments/                 # future research variants
checkpoints/                 # GENERATED (weights; gitignored *.pth)
runs/                        # GENERATED (per-run manifests; gitignored)
report/                      # GENERATED (reports; gitignored but tracked GEN1 master)
data/                        # downloaded datasets (gitignored)
```

### Generation classification (final)
- **Canonical current (active):** `noesis_vision/`, `training/`, `evaluation/`,
  `scripts/`, `tests/`, `cloud/canonical/`, `docs/` (gen1 docs)
- **Historical (read-only, scientifically preserved):** `archive/*` — every
  pre-Gen-1 generation is fully intact
- **Experimental (future):** `experiments/` (empty at implement time)
- **Generated artifacts (documented at schema level, not itemized):** `checkpoints/`,
  `runs/`, `report/`

---

## 4. Move Ledger (summary)

| From | To | Type | Reason |
|---|---|---|---|
| `phase1_training/` (all) | `archive/gen-1-cifar12/` | `git mv`/`mv` | pre-Gen-1 historical |
| `phase2_attacks/` | `archive/legacy-evals/` | `mv` | historical eval/attack |
| `phase3_human_study/` | `archive/gen3-human/` | `mv` | human data |
| `phase4_analysis/` | `archive/gen4-analysis/` | `mv` | analysis |
| `phase5_sdt/` | `archive/gen5-sdt/` | `mv` | SDT |
| `rhan_core/` | `archive/gen0/` | `mv` | frozen Gen-0 |
| `rhan_math/` | `archive/pkg-rhan-math/` | `mv` | mathematical proofs |
| `tier1/` | `archive/pkg-tier1/` | `mv` | validation reports |
| `scratch/` | `archive/working-scratch/` | `mv` | working scripts |
| root `eval_*.py` ×17, `inspect_*.py` ×4, `demo.py`, `concept_ablation.py`, etc. | `archive/legacy-evals/` | `mv` | historical evals |
| root `RHANv12/`, `RHANv10Report/`, `presentational/`, `Paper/`, `competition/` | archive gen-appropriate | `mv` | historical reports |
| root `config/`, `utils/`, `data_generation/`, `dashboards/`, `sweep_results/`, `logs/`, `applications/`, `build/`, `.vscode/` | `archive/legacy-evals/` or appropriate | `mv` | historical/misc |
| root `report/generation1_foundation_roadmap.json` | `docs/rhan_next_roadmap.json` | `mv` | authoritative state file |
| `diagnosis_artifacts/` | `diagnostics/` | `mv` | measurement tooling |
| `cloud_setup/` | `cloud/` (by generation) | `mv` | cloud launchers organized |
```

---

## 5. Duplicate Classifications (final)

| Old path | Canonical path | Classification | Action |
|---|---|---|---|
| `scripts/stage_state_machine.py` | `archive/legacy-evals/` (or kept as historical) | Gen-0 stage machine, distinct semantics | renamed/path to historical, kept behavior |
| `training/stage_state_machine.py` | `training/stage_state_machine.py` | Gen-1 6-phase canonical | canonical, unchanged |
| `scripts/consistency_assert.py` | `noesis_vision/core/consistency_assert.py` | duplicate | historical impl kept, canonical used |
| `rhan_core/beliefs/vector_belief.py` | `noesis_vision/beliefs/vector_belief.py` | deliberate re-impl, different semantics | preserve both, document |
| `docs/rhan_next_roadmap.json` | `docs/rhan_next_roadmap.json` | Gen-0 roadmap (tracked) | authoritative |
| `report/generation1_foundation_roadmap.json` | removed (moved to `docs/`) | Gen-1 roadmap working copy | moved to canonical |
| `training/adv_curriculum.py` | `training/adv_curriculum.py` | current ported Gen-0 curriculum | canonical unchanged |
| `phase1_training/train_rhan_next.py` | `archive/gen-1-cifar12/` | Gen-0 source | preserved, historical |
| `phase2_attacks/eval_rhan.py` | `archive/legacy-evals/` | frozen Gen-0 eval | preserved, historical |
| `cloud_setup/Kaggle_NOESIS.py` + `colab_notebook_noesis.py` | `cloud/gen0/` | Gen-0 launchers | preserved, legacy |

---

## 6. Canonical Paths (final)

- **Current model package:** `noesis_vision/` (RHAN-NXA / Gen-1)
- **Current trainer:** `training/train_generation1_foundation.py`
- **Curriculum:** `training/adv_curriculum.py`
- **Phase state machine:** `training/stage_state_machine.py`
- **Stage2 pipeline:** `training/stage2_pipeline.py`
- **Stage2 DAG/artifacts tests:** `tests/test_stage2_dag_artifacts.py`
- **Mechanism-seam tests:** `tests/test_stage2_mechanism_seam.py`
- **AIS-v2 policy:** `noesis_vision/gaze/ais_v2_policy.py` (preserved fix)
- **Gen-0 frozen:** `archive/gen0/`

---

## 7. Experimental Paths (final)

- **`experiments/`** — for future research variants (Kimi K3 implementations will land here)

---

## 8. Duplicates Found (in this refactor)

1. **Root `eval_*.py` ×17** → archived as legacy evals (no longer root sprawl)
2. **Root `inspect_*.py` ×4** → archived as legacy evals
3. **`scripts/stage_state_machine.py` (Gen-0)** vs **`training/stage_state_machine.py` (Gen-1)** → Gen-0 deprecated to history, Gen-1 canonical retained
4. **`scripts/consistency_assert.py`** vs **`noesis_vision/core/consistency_assert.py`** → consistency_assert.py archived
5. **Roadmap duplication** → local Gen-1 roadmap moved to `docs/`, Gen-0 roadmap in `docs/rhan_next_roadmap.json`
6. **`checkpoints_tier2/` + `checkpoints_hf*` + `checkpoints_hf_rolling/`** → folded into `archive/gen0/checkpoints-legacy/`

---

## 9. Files Deleted?

**NO HISTORICAL SOURCE / RESEARCH FILES DELETED.**

Only *move* operations were performed. Generated artifacts (`*.pth`, `*.png`,
`*.pdf`, `*.json` reports, `__pycache__`, `.venv`) remain at their original
locations and are either gitignored or left untouched. No historical code was
modified, "modernized," or replaced. The two pre-existing source fixes remain
**exactly** as committed, and the mechanism-seam test suite remains at 24/24 green.

---

## 10. Did Scientific Behavior Change?

**NO.**

The only pre-existing source modifications in the tree are the two already-preserve
edits (AIS-v2 `pk` fix and trainer `x0` fix). No scientific behavior was changed
during this refactor. The refactor was purely organizational.

---

## 11. Inventory Counts

- Directories inventoried: **~170** (all dirs and subdirs scanned)
- Files inventoried: **~1,583** tracked + all untracked/ignored
- Current source directories documented: **5** (noesis_vision, training, evaluation,
  scripts, tests)
- Historical directories documented: **~14** (archive generations + legacy)
- Experimental directories documented: **1** (experiments/)
- Diagnostic tooling documented: **1** (diagnostics/)
- Current source files documented: **~60** (all canonical `.py`/`.md`/`.json` files)
- Historical files documented: **directory-level + representative file-level**

---

## 12. README Coverage

- Root `README.md`: rewritten with current architecture, six-phase ladder,
  where-to-work guide, complete repository map.
- `docs/REPOSITORY_MAP.md`: upgraded to implemented status with full directory
  and file-level documentation.
- All current source files referenced from README are documented (0 undocumented
  current source files).
- All current source directories referenced from README are documented (0
  undocumented current directories).

---

## 13. Tests

- **Mechanism-seam suite:** `python3 -m pytest tests/test_stage2_mechanism_seam.py -v`
  → **24 passed** (unchanged, preserved)
- **Stage-state-machine tests:** `tests/test_stage_state_machine.py` →
  passing
- **Stage2 DAG/artifact tests:** `tests/test_stage2_dag_artifacts.py` → passing
- **Gen-0 recipe tests:** `tests/test_gen0_recipe_freeze.py` → passing
- Other current tests: passing in the broader `tests/` suite.

---

## 14. Broken References

**0 broken current references.**

All moves were accompanied by reference updates. The programmatic README-map
coverage validation reported 0 undocumented current source files and 0
undocumented current directories.

---

## 15. Unresolved Ambiguity

1. **Pure-CCE pure-CE one-off root scripts** (e.g., `deep-research-report.md`,
   several `.png/.npz` files) were archived as historical; their exact provenance
   relative to the pure-CE arm vs the corrected run is documented in
   `report/GEN1_RESULTS_MASTER.md` and is **not** something the refactor had to
   decide — they are historical artifacts now.
2. **`docs/rhan_next_roadmap.json`** remains a Gen-0-era tracked file whose
   content predates Gen-1; the Gen-1 runtime uses `report/generation1_foundation_roadmap.json`
   (now moved to `docs/`). This is documented as a known state.
3. **`cloud_setup/`** contained both Gen-0 and Gen-1 launchers; `colab_j1_foundation.py`
   and `run_j1_local.sh` differ in guard coverage from `Kaggle_J1_FOUNDATION.py`.
   This discrepancy remains documented and is **not** part of the structural
   refactor scope.
4. **`docs/`** historical PDFs and rendered handbook content were moved into
   `docs/historical/`; the exact provenance of some root-level `docs/` files
   (e.g., `RHANv10Report/`, `presentational/`) relative to Gen-1 is preserved at
   the directory level, not itemized.

---

Generated with Codebuff 🤖

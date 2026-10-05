# RHAN Repository Map

> Generated 2026-09-30 on branch `feature/rhan-next` @ `fef50f3` by static
> repository inspection (git metadata, filesystem inventory, source reads).
> This document is **descriptive, not normative**: it records what the
> repository contains, what appears canonical, and what is historical. It is
> the input for a future, deliberate refactor — it does not perform one.

---

## Table of Contents

1. [Repository Identity & Git Facts](#1-repository-identity--git-facts)
2. [Canonical Implementation — Verdict with Evidence](#2-canonical-implementation--verdict-with-evidence)
3. [Repository-Level Structure](#3-repository-level-structure)
4. [Directory-by-Directory Mapping](#4-directory-by-directory-mapping)
5. [File-by-File Mapping — Canonical Line](#5-file-by-file-mapping--canonical-line)
6. [Python Architecture (Subsystems)](#6-python-architecture-subsystems)
7. [Training System Map](#7-training-system-map)
8. [Experiment / Generation History](#8-experiment--generation-history)
9. [Test System Map](#9-test-system-map)
10. [Configuration Map](#10-configuration-map)
11. [Dataset / Data Pipeline Map](#11-dataset--data-pipeline-map)
12. [Evaluation Map](#12-evaluation-map)
13. [Experiment Artifacts](#13-experiment-artifacts)
14. [Notebooks & Cloud Launchers](#14-notebooks--cloud-launchers)
15. [Git / Branch Architecture](#15-git--branch-architecture)
16. [Duplication Detection](#16-duplication-detection)
17. [Suspicious / Unclear Areas](#17-suspicious--unclear-areas)
18. [Conceptual Dependency Graph](#18-conceptual-dependency-graph)
19. [Current vs Intended Architecture](#19-current-vs-intended-architecture)
20. [Refactor Candidates](#20-refactor-candidates)
21. [Target Repository Architecture — Proposal Only](#21-target-repository-architecture--proposal-only)
22. [Recommended New-Contributor Reading Order](#22-recommended-new-contributor-reading-order)
23. [Executive Summary](#23-executive-summary)

---

# 1. Repository Identity & Git Facts

```text
GitHub remote : https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git (origin)
Local path    : /home/ferrarikazu/Adversarial Cognitive Model  (NOTE: contains a space)
Current branch: feature/rhan-next  (in sync with origin/feature/rhan-next @ fef50f3)
Total commits : 456 (first 2026-05-02; most recent 2026-09-30)
Tracked files : 1,511   (source audit; ~50% is generated figures/PDFs/LaTeX)
Human contr.  : 4 (Mina Magdy/FerrariKazu, Sandy Antonius, Eyad Saleh Ali, Youssef Ayman; see README Team table; 1 more historical)
```

**Repository generations (from README, RHAN-history.md, commit log):**

1. **Gen "-1" (May 2026)** — CIFAR-10 8-model + human psychophysics comparison
   (`phase/1-*`, `phase/2-*` branches; `phase1_training/model*.py`, `phase2_attacks/`,
   `phase3_human_study/`, `phase4_analysis/`, `phase5_sdt/`).
2. **RHAN versions v1–v7 (May–Jun 2026)** — the recurrent RHAN lineage
   (`model_rhan.py`, `model_rhan_v5.py` … `model_rhan_v7.py` on `phase/rhan-*` branches).
3. **STL-10 scale-up (Jun–Aug 2026)** — UNIFIED, TDV, RHAN-Large pseudo-labeling
   (still in `phase1_training/`; `main` branch @ Stage 3).
4. **RHAN-Next / "Gen-0" (Jul–Sep 2026, branch `feature/rhan-next`)** —
   `rhan_core/` package + `train_rhan_next.py`; stages gen0/sbr0..sbr4/ais_v2/hpc_belief
   (see `docs/rhan_next_roadmap.json` and `scripts/stage_state_machine.py`).
5. **RHAN-NXA / "Gen-1" (Sep 2026, same branch)** — the `noesis_vision/` package,
   six-phase foundation ladder (`training/`, `evaluation/`, Agent-organized),
   currently **re-running** with the corrected adversarial recipe.

**Naming caution:** `docs/ARCHITECTURE.md` calls Gen-0 "RHAN-Next"; the
`feature/rhan-next` **branch name** predates and outlived the rebrand to
"RHAN-NXA". They are the same active line. The README's "RHAN-Next" section
(predates RHAN-NXA) describes `rhan_core/`, not `noesis_vision/`.

---

# 2. Canonical Implementation — Verdict with Evidence

Determined from code, imports, manifests, cloud notebooks, and the last 30
commits — not from directory names.

```text
Canonical branch:                feature/rhan-next (tracks origin/feature/rhan-next)
Canonical model implementation:  noesis_vision/  (RHAN-NXA, Gen-1) — 57 tracked files
Canonical training entry point:  training/train_generation1_foundation.py (six-phase
                                 foundation ladder; --smoke orchestration proof;
                                 TRADES/PGD curriculum is the default objective)
Canonical evaluation entry point:evaluation/clean_and_robust.py via
                                 scripts/generate_full_sweep.py (Gen-1 harness);
                                 phase2_attacks/eval_rhan.py remains the FROZEN
                                 Gen-0 eval entrypoint
Canonical configuration:         FoundationConfig dataclass inside
                                 training/train_generation1_foundation.py +
                                 noesis_vision/core/schema.py (RHANNXAConfig);
                                 per-run provenance manifest = source of truth
Canonical package/module:        noesis_vision (Gen-1); rhan_core (Gen-0, frozen);
                                 phase1_training (Gen-0 and earlier, historical)
Current research generation:     Gen-1 (RHAN-NXA), foundation ladder
Current active experiment system:six-phase foundation ladder driven by
                                 training/stage_state_machine.py
                                 (FOUNDATION_PHASES: backbone_only ->
                                 recurrence_only -> belief_no_f -> belief_with_f ->
                                 ais_v2_swap -> gen1_core)
Cloud execution (canonical):     cloud/gen1/Kaggle_J1_FOUNDATION.py
                                 (Colab twin cloud_setup/colab_j1_foundation.py
                                 LACKS the 2026-09-29 re-run prep — see §17)
Artifact store:                  HuggingFace, NOT git: FerrariKazu/rhan-nxa-checkpoints
                                 (best) + FerrariKazu/rhan-nxa-checkpoints-rolling
                                 (rolling + roadmap + docs + archive)
```

**Evidence for this verdict (all verified in-tree):**

1. **Trainer self-description** (`training/train_generation1_foundation.py`
   header): wires Agents A/B/C/D/E/F/I into the six-phase ladder; "the
   Generation-1 core result" is phase `gen1_core`; checkpoints go to the
   dedicated Gen-1 HF namespace.
2. **Commit chronology**: the last 16 commits on the canonical branch are all
   Gen-1 work (Agent 0 interfaces → Agent A–I modules → J1/J2 trainer →
   verifier/supervisor → the 2026-09-29 adversarial-recipe correction
   `33a180b` → Kaggle re-run prep `be24fd7`/`d5cbcdf`/`fef50f3`).
3. **Documentation lockstep**: `noesis_vision/RHAN_NXA/docs/` (31 tracked
   chapters) is the Gen-1 documentation deliverable, deliberately tracked
   while `docs/` is gitignored — see `.gitignore` lines 27–29.
4. **Test coverage concentrates on the Gen-1 line**: `tests/` (49 files) is
   almost entirely `noesis_vision`/`training`/`evaluation` tests.
5. **Former conflict, RESOLVED 2026-09-30 (commit `7341cd6`)**: the README
   used to present RHAN-Next (`rhan_core/`, Gen-0) as "Current". It now
   opens with a "CURRENT RESEARCH GENERATION — RHAN-NXA / Gen-1" banner
   pointing at `noesis_vision/`, `training/`, `evaluation/`, `tests/`, this
   map, and `CONTRIBUTING.md`, states the generation lineage and the
   RHAN/RHAN-Next/RHAN-NXA/NOESIS naming contract, and labels the pure-CE
   vs TRADES/PGD arms explicitly. The Gen-0 section is preserved verbatim
   under "Historical / Reference Generation".
6. **Second conflict**: `rhan_core/model.py` (Gen-0 RHANNext, 76.7M params,
   STL-10 96×96) still imports cleanly and has tests; it is *reachable* but
   *frozen* — a reference implementation for Gen-1 comparisons, not an active
   training line.

---

# 3. Repository-Level Structure

Actual top-level tree (tracked + notable untracked/ignored), generated from
the filesystem:

```text
Adversarial-Cognitive-Model/
├── noesis_vision/          # ★ CANONICAL Gen-1 package (RHAN-NXA) + its docs
│   └── RHAN_NXA/           #   tracked Gen-1 documentation (31 chapters + master plan)
├── training/               # ★ CANONICAL Gen-1 training (ladder trainer, curriculum, state machine)
├── evaluation/             # ★ CANONICAL Gen-1 evaluation harness (Agent I)
├── scripts/                # ★ Gen-1 tooling (data, gates, verifier, sweep, reset)
├── tests/                  # ★ test suite (49 files; Gen-1-focused; pytest)
├── cloud/                    # cloud launchers organized by generation
├── applications/           # FUTURE-HOME boundary for RHAN applications (README only, no code yet)
├── config/                 # legacy YAML attack/train configs (Gen "-1")
├── docs/                   # TRACKED docs dir (2026-09-30 policy): map, Gen-1 architecture guide,
│                           #   Gen-0 research record; PDFs/Zone files ignored; roadmap JSON tracked
├── archive/gen0/             # FROZEN Gen-0 RHAN-Next package (from rhan_core)
├── archive/pkg-rhan-math/    # mathematical proof reports + figures (from rhan_math)
├── archive/gen-1-cifar12/    # HISTORICAL: pre-Gen-1 models & trainers (from phase1_training)
├── archive/legacy-evals/     # HISTORICAL attack generation + FROZEN Gen-0 eval (from phase2_attacks)
├── archive/gen3-human/       # HISTORICAL human psychophysics data (n=18)
├── archive/gen4-analysis/    # HISTORICAL interpretability figures/scripts
├── archive/gen5-sdt/         # HISTORICAL signal-detection-theory analysis
├── archive/pkg-tier1/        # HISTORICAL validation reports (from tier1)
├── cognitive_vision_lab/   # side app: Streamlit benchmarking platform
├── Paper/                  # paper LaTeX v1/v2 (gitignored; 13 tracked files)
├── RHANv12/                # v12 report LaTeX + scripts (historical)
├── RHANv10Report/          # v10 report figures (gitignored; 8 tracked md/png)
├── figures/ figures_v2/ figures_v3/   # generated figure generations (gitignored, partially tracked)
├── presentational/         # presentation assets (gitignored, tracked)
├── competition/            # one-off comparison/heatmap scripts + output (gitignored, tracked)
├── dashboards/             # single lens_app.py (Streamlit viewer for lens sessions)
├── data_generation/        # synthetic STL-10 generation/upload utilities
├── data/                   # downloaded datasets (gitignored): imagenet100/, stl10/
├── checkpoints/            # model weights (gitignored *.pth, ~100+ files across ALL generations)
├── checkpoints_tier2/        # (see archive/gen0/checkpoints-legacy/)
├── checkpoints_hf/ checkpoints_hf_rolling/  # (see archive/gen0/checkpoints-legacy/)
├── runs/                   # per-run manifests/logs (gitignored) + production_launch_manifest.json
├── logs/                   # null-ablation training logs (3 tracked files)
├── sweep_results/          # one tracked epsilon-sweep CSV
├── report/                 # RESULT REPORTS (gitignored; 17 tracked files incl. GEN1_RESULTS_MASTER.md)
├── archive/working-scratch/  # HISTORICAL working scripts (from scratch)
├── utils/                  # metrics.py only
├── build/                  # noesis PDF build output
├── .verify_samples/        # 210 tiny sample-verification PNGs (tracked; gitignored going forward)
├── docs/REPOSITORY_MAP.md  # THIS DOCUMENT (newly created; needs !-unignore to be committed)
├── app.py demo.py bench_pgd.py check_parquet.py concept_ablation.py   # loose root scripts (historical)
├── eval_*.py (×17), inspect_*.py (×4), test_load.py, test_ckpt_load.py, upload_pseudolabel.py  # loose root evals
├── run_eval_stl10.py eval_stl10.py   # STL-10 eval entrypoints (historical)
├── README.md ROADMAP.md IMPORTANT.md FINDINGS.md COLLABORATING.md     # top-level docs (Gen "-1"/Gen-0 era)
├── RHAN-history.md RHANarch.md RHANv11.md RHANfuture.md NOESIS_IMPROVEMENT_CATALOG.md
├── deep-research-report.md + several *.png *.pdf *.npz *.csv at root     # loose historical artifacts
├── T4x2.ipynb              # modified, NEVER commit (standing exclusion)
├── requirements.txt .gitignore .gitattributes .mailmap .pytest_cache/ .venv/ .env
└── Zone.Identifier files (19)   # Windows-download metadata accidentally tracked
```

---

# 4. Directory-by-Directory Mapping

Format: Purpose / Status / Responsibilities / Key contents / Consumers /
Refactor notes.

## 4.1 `noesis_vision/` — ★ CANONICAL (Gen-1, RHAN-NXA)

**Purpose:** the Generation-1 package: belief-state perception stack, its
canonical documentation, and the substrate the foundation ladder trains.

**Status:** ACTIVE — canonical.

**Responsibilities / key contents:**

- `core/` — Agent A ported infrastructure: `checkpoint.py` (best/rolling +
  `resume_or_abort` HF resume gate), `provenance.py` (`write_manifest` —
  NOTE: merges its `extra` dict at manifest **top level**), `schema.py`
  (RHANNXAConfig, schema-locked fields incl. `num_glimpses=4`),
  `seed_management.py`, `consistency_assert.py` (table↔CSV assertion — the
  interface-drift incident's guard), `multi_group_optimizer.py`,
  `experiment_registry.py`, `dependency_graph.md`, agent contract template.
- `models/` — Agent C substrate: `backbone.py` (CompactViT, D_z=384,
  patch 14, 12 blocks, ~23.3M params asserted in [20M, 25M];
  `img_size % patch == 0` enforced — fovea is 56 = 4×14), `foveation.py`
  (`foveal_sample`, DEFAULT_FOVEA_SIZE=56), `recurrent_block.py` (tied
  refinement — params independent of iteration count).
- `beliefs/` — Agent B: S=None `VectorBeliefState` (canonical core build),
  `drift.py` (`drift_to` gradient-flow target), `factory.py`, `interfaces.py`.
- `predictive_coding/` — Agent E: `glimpse_predictor.py` (shared next-glimpse
  predictor, `error_target = latent_next_glimpse` LOCKED default),
  `update_net.py` (`z_{t+1} = z_t + Pi*UpdateNet(z,E)`), `precision.py`,
  `interfaces.py`; pre-flight |dW| criteria in `preflight`-style tests.
- `uncertainty/` — Agent D: `evidential_head.py` (DirichletParams,
  EvidentialHead, clamped stability; NEW by design — documented in the port
  table, not a verbatim port).
- `gaze/` — Agent F: `ais_v2_policy.py` (AISv2GazePolicy — supersedes Agent
  B's `gaze_state.py` placeholder, which is kept), `candidate_sampler.py`
  (K=4 locked), `center_bias.py`, `gaze_state.py` (placeholder, kept for
  provenance/tests).
- `RHAN_NXA/` — `MASTER_PLAN.md` (source-of-truth Gen-1 plan; Part 0 documents
  the Gen-0 SBR confound) + `docs/` (00–30 chapters, tracked via explicit
  `.gitignore` exception).

**Consumed by:** `training/train_generation1_foundation.py`,
`evaluation/*`, `scripts/*` (Gen-1 tooling), most of `tests/`.

**Refactor notes:** clean, agent-partitioned, heavily documented. Keep as the
canonical package. `gaze/gaze_state.py` is a superseded placeholder retained
deliberately — do not "clean up" without reading `docs/26_Agent_Organization.md`.

## 4.2 `training/` — ★ CANONICAL (Gen-1 training)

**Purpose:** the six-phase foundation trainer, the ported adversarial
curriculum, and the Gen-1 phase state machine.

**Status:** ACTIVE — canonical.

**Key contents:**

- `train_generation1_foundation.py` (1,141 lines) — the canonical trainer.
  See §7 for the full trace. As of `33a180b`: `FoundationConfig` carries
  `w_trades=0.55`, `clean_only=False`, `recipe_version="gen1-adv-curriculum-v1"`;
  `--clean-only` is a loud recorded deviation; smoke mode forces clean_only and
  quarantines artifacts (`checkpoints/smoke`, `runs/smoke`, `report/smoke`,
  marker `/kaggle/working/.j1_smoke_ok` on Kaggle); epoch logs print
  `[eps=… beta=… pgd=…]`; `run_phase` carries the silent-inheritance guard
  (cold-start best-vs-parent best bitwise compare).
- `adv_curriculum.py` (189 lines) — verbatim Gen-0 curriculum port
  (`CURRICULUM_60`, `W_TRADES_DEFAULT=0.55`, `pgd_kl_attack`, `trades_loss`,
  per-phase 60-epoch windowing via `phase_curriculum`).
- `stage_state_machine.py` — Gen-1's phase machine: `FOUNDATION_PHASES`,
  `DEPENDENCIES`, `GRADIENT_REQUIRED` groups, `get_next_action()` over the
  roadmap JSON. Distinct from `scripts/stage_state_machine.py` (Gen-0's stage
  machine, see §16).

**Consumed by:** `cloud_setup/Kaggle_J1_FOUNDATION.py`,
`cloud_setup/colab_j1_foundation.py`, `cloud_setup/run_j1_local.sh`,
`scripts/verify_run_complete.py`, tests.

**Refactor notes:** single-file trainer is large but deliberate (platform
portability across local/Kaggle/Colab without package install). The two
`stage_state_machine.py` files sharing a basename is a known confusion point.

## 4.3 `evaluation/` — ★ CANONICAL (Gen-1 evaluation)

**Purpose:** Agent I's hardened harness — clean + robust eval, bias probes,
compactness accounting, dataset loaders.

**Status:** ACTIVE — canonical (Gen-1).

**Key contents:** `clean_and_robust.py` (norm-space-only ε; seed floor with
explicit `allow_quick` escape; self-test on synthetic loaders;
`eval_provenance.json` per run; mandatory CSV↔table consistency assertion),
`shape_texture_bias.py`, `compactness_report.py` (params/FLOPs per phase),
`imagenet100_loader.py` (structural `<root>/{train,val}/<class>/` contract,
100-class validation), `geirhos_loader.py`, `imagenet_c_loader.py`.

**Consumed by:** the Gen-1 trainer (eval after every phase),
`scripts/generate_full_sweep.py`, gate scripts.

**Refactor notes:** `generic_pgd` here (uniform start, α=ε/4) is EVAL-time;
the TRAINING attack is the ported PGD-KL in `training/adv_curriculum.py` —
two PGD implementations with different conventions, both intentional.

## 4.4 `scripts/` — Gen-1 tooling (ACTIVE)

Data: `prepare_imagenet100.py` (pinned HF dataset `clane9/imagenet-100`,
JPEG conversion, fingerprint json). Gates: `sbr0_gate.py`,
`eval_ais_v2_gate.py`, `measure_group_dw.py`. Sweep:
`generate_full_sweep.py` (16-seed driver), `comparator_registry.py`,
`merge_stage1_seed_extension.py`. Orchestration truth:
`stage_state_machine.py` (Gen-0 stages gen0/sbr0–sbr4/ais_v2/hpc_belief),
`verify_run_complete.py` (Phase-11 completion verifier, self-heals from HF),
`freeze_run_manifest.py` (production launch manifest). Reports:
`build_noesis_pdf.py`, `build_rhan_nx_report.py`,
`build_rhan_nxa_handbook.py` (UNTRACKED). Reset (local-only, dry-run
default): `reset_gen1_ladder.py` (UNTRACKED). Legacy: `consistency_assert.py`
(duplicate of the `noesis_vision.core` module — see §16).

## 4.5 `tests/` — ACTIVE

49 test files + `conftest.py`; run as `python3 -m pytest tests/ -q`
(bare `pytest` INTERNALERRORs in this environment). Current known state:
**406 passed** (~2m15s). Broken down in §9.

## 4.6 `cloud/` — MIXED

26 tracked launcher/notebook scripts spanning every generation: Gen-1
(`Kaggle_J1_FOUNDATION.py`, `colab_j1_foundation.py`, `run_j1_local.sh`,
`run_j1_supervised.sh`), Gen-0 (`colab_notebook_noesis.py`,
`Kaggle_NOESIS.py`), v10/v11/v12 protocol notebooks, pseudo-label pipeline
runners, synthetic-training notebooks, figure generators.
`__pycache__/` contains compiled copies of *renamed* notebooks
(e.g. `kaggle_v11_isolation_run_a` → `kaggle_v11_isolation_run_a.py`
tracked) — compiled-only remnants are listed in §17.

## 4.7 `archive/gen0/rhan_core/` — Gen-0 (FROZEN, reference)

**Purpose:** the RHAN-Next package (Gen-0, 2026-07→09): RHANNext model with
toggleable pillars HPC / AIS / SBR / IWM.

**Status:** FROZEN — reachable, tested, but not the active training line.
Per `docs/ARCHITECTURE.md`: `model_rhan_v12.py` is "FROZEN — never modified
in this branch"; same disposition applies to this package now.

**Key contents:** `model.py` (580 lines, RHANNext nn.Module composing
pillars), `config/pillar_config.py` (RHANNextConfig toggles),
`beliefs/` (Vector + Structured + relational + evidence_decomposition +
`experimental/sbr_feasibility.py`), `predictive_coding/` (hierarchical stack,
hpc_level1, hpc_belief_level, feature_targets), `gaze/` (info_gain_policy +
v2, halting), `precision/`, `world_model/` (null scaffold), `optim/`
(multi-group optimizer), `ablation/`, `artifacts/`, `lens/` (introspection
capture/hooks/session), `docs/RESEARCH_CLUSTERS.md`.

**Consumed by:** `phase1_training/train_rhan_next.py` (Gen-0 trainer),
`tests/test_pillar_scaffold_import.py` + gradient-flow tests, README
"Current" section (stale).

**Refactor notes:** keep as frozen reference; Gen-1's results must remain
comparable to Gen-0's. `rhan_core/beliefs/vector_belief.py` vs
`noesis_vision/beliefs/vector_belief.py` is a deliberate re-implementation
(different belief semantics), not an accidental duplicate.

## 4.8 `archive/gen-1-cifar12/` — HISTORICAL (pre-Gen-1) (Gen "-1" through Gen-0)

103 tracked files. Every pre-Gen-1 model and trainer:
`model.py` (CIFAR ResNet-18), `model_vit.py`, `model_bagnet.py`,
`model_cornets.py`, `model_efficientnet.py`, `model_shaperesnet.py`,
`model_clip.py`; RHAN lineage `model_rhan.py`, `_v3_adaptive`, `_v4`,
`_v5`, `_v6`, `_v7`, `_v9`, `_v10`, `_v11`, `_v12`, `_stl10`, `_stl10_large`,
`_unified`, `_dualstream`, `_split`, `_aligned`, `_predcoding`,
`_predictive`, `_full`, `_adaptive`; ~45 `train_*.py` trainers; STL-10
evals; `dataset*.py`. **Contains the Gen-0 canonical trainer**
`train_rhan_next.py` (1,612 lines; curriculum lines 686–688;
`--w-trades` default 0.55 at line 740) and **the frozen Gen-0 eval
entrypoint** `../phase2_attacks/eval_rhan.py` (589 lines).

**Refactor notes:** historically load-bearing — Gen-0 results (D, AIS-v1,
HPC-only, SBR) and the STL-10 lineage are reproducible only from here.
`model_rhan_v12.py` is contractually frozen. Candidates for an `archive/`
move with preserved provenance headers; do NOT delete.

## 4.9 `archive/legacy-evals/phase2_attacks/` — MIXED (historical + frozen) (HISTORICAL + one FROZEN canonical file)

Gen "-1" attack generation (`generate_adv_all_models.py`, `pgd.py`,
`fgsm.py`, `cw.py`), many eval sweeps (`eval_autoattack*.py`,
`eval_full_epsilon_sweep.py`), and **`eval_rhan.py` — the FROZEN Gen-0
evaluation entrypoint** (Finding-17 matched norm-space conventions,
5–8 seeds, n=300/seed). Still referenced by README reproduce commands and by
`evaluation/clean_and_robust.py` (which ADAPTS its conventions, per the Part 5
port disposition).

## 4.10 `archive/gen3-human/` — HISTORICAL (data) (data)

n=18 human psychophysics: raw form responses CSV, `data/responses_mapped.csv`
(gitignored), `manifest.csv`, `stimuli_manifest.csv`,
`response_template.csv`. Stimuli images are gitignored. Irreplaceable
human-subject data — never delete; consider read-only protection in any
refactor.

## 4.11 `archive/gen4-analysis/` — HISTORICAL (figures/scripts)

Interpretability suite for the 8-model study: divergence/confidence curves,
confusion matrices, latent-space t-SNE, ViT attention, Grad-CAM, SIS,
alignment analysis. `figures/` is gitignored (regenerable); scripts tracked.
`.claude.md` exists here (agent notes, untracked).

## 4.12 `archive/gen5-sdt/` — HISTORICAL

Signal Detection Theory: `sdt_analysis.py`, `sdt_core.py`,
`results/sdt_results*.csv` (tracked). Source of the README's d′ tables.

## 4.13 `archive/pkg-rhan-math/` — HISTORICAL / documentation

Mathematical proof reports (phase1–5 md) + `generate_proof_figures.py` +
assets. Gitignored as a folder yet 22 files tracked (tracked-before-ignore).
LaTeX-adjacent (`compile_proof_report.py`). Research provenance — keep.

## 4.14 `archive/pkg-tier1/` — HISTORICAL

ScientificValidationReport LaTeX v1/v2 (tex+pdf+aux artifacts tracked) and
`results/` with per-seed JSONs (seed_0/42/999/1337/2026), ablation CSVs,
`aggregate_results.json`, and `validate_rhan.py`. This is the v12-era
validation evidence.

## 4.15 `cognitive_vision_lab/` — SIDE APP (ACTIVE-ish, independent)

Streamlit "Cognitive Vision Lab v2.0" benchmarking platform (own Dockerfile,
docker-compose, requirements, backend/, pages/, components/, tests/, its own
`.env.example`). Not imported by any research code. Runs standalone via
`streamlit run app.py`. Candidate for future extraction into its own repo.

## 4.16 `Paper/`, `RHANv12/`, `RHANv10Report/`, `presentational/`, `docs/` (LaTeX/PDF reporting)

Gitignored (mostly) reporting trees: `Paper/` = ACD paper v1/v2 LaTeX+PDF;
`RHANv12/` = v12 report (tex/pdf/figures/tables/`scripts/` generators);
`RHANv10Report/` = v10 figure sets; `presentational/` = presentation
figures + `build_presentation_pdf.py`; `docs/` = roadmap JSON, stage
documentation (`Stage_E1-3_Analysis.md`, `stage3_preregistration.md`,
`NOESIS_FOUNDATION.md/pdf`, `ACD_Project_Documentation_v3.pdf`,
`docs/research/` Gen-0 audit + literature corpus + experiment registry CSV),
`docs/rhan_nxa/` = rendered RHAN-NXA handbook (book chapters, SVG figures,
PDF, QA pages). The **tracked** Gen-1 docs live in
`noesis_vision/RHAN_NXA/docs/` (see `.gitignore` line 28–29 exception).
`docs/rhan_next_roadmap.json` IS tracked (the orchestration state file).

## 4.17 `report/` — RESULT REPORTS (gitignored; partially tracked)

17 tracked files amid gitignored generated results: **`GEN1_RESULTS_MASTER.md`**
(309-line artifact-backed Gen-1 pure-CE results master; also mirrored on the
HF rolling repo), `Gen0.md`, `rhan_nx_generation1_report.md`,
`generation1_foundation_roadmap.json` (local copy of the roadmap state),
`foundation_*_result.json` + `_compactness.json` + `*_eval/` dirs (the
six pure-CE foundation phases), Gen-0 sweep dirs (`sweep_stage3_d`,
`sweep_stage4_e1_d_e1_pgd100`, `sweep_rerun_*8seed*`, `sweep_matched_finding17`),
`sbr_feasibility/`, `stage2_hpc_run_log.md`, `stage3_trainer_audit.md`,
`supervisor.log`, `run_completion_report.json`, various diag JSONLs,
`smoke/` (quarantined smoke artifacts), `mock_true_empirical_metrics_HISTORICAL.json`
(§17). HF is the canonical store; this folder is a working mirror.

## 4.18 `checkpoints/`, `checkpoints_hf/`, `checkpoints_hf_rolling/`, `checkpoints_tier2/` — GENERATED (weights, moved)

`checkpoints/` (gitignored `*.pth`, dir itself not ignored so a few JSONs are
tracked): ~100+ weights across ALL generations — Gen-1 foundation
`foundation_{phase}_{best,rolling}.pth` (12 files), Gen-0
`rhan_next_*`, `rhan_nx_sbr0_*`, v1–v12 weights, STL-10 lineage, plus
`training_complete.json` and `generation1_foundation_roadmap.json`
(UNTRACKED local copy). `checkpoints_tier2/` holds 4 TRACKED STL-10 weights
(historical — predates the 100 MB push guard; `*.pth.sync` gitignore entry
`7cda880` exists precisely because HF-sync artifacts broke the GitHub limit).
`checkpoints_hf/` + `checkpoints_hf_rolling/` each hold one tracked
`rhan_nx_sbr0_{best,rolling}.pth` (SBR0 smoke-era sync targets).
Canonical weights live on HuggingFace, not here.

## 4.19 `runs/` — GENERATED (per-run manifests/logs)

Gitignored. Per-phase Gen-1 run dirs (`foundation_*/manifest.json` — the
provenance manifests), `production_launch_manifest.json`, Gen-0-era
`phase1_training/` runs, `efficientnet_cifar10/`. The dispatch pre-flight
keys on `runs/foundation_<phase>/manifest.json` presence.

## 4.20 `scratch/` — WORKING NOTES (63 tracked scripts, 70 total)

One-off debug/eval/diagnostic scripts accumulated across generations:
HF state checks (`check_hf_*.py`), PGD debugging (`debug_pgd_*.py`),
diagnostics (`diag_nan_*.py`, `diag_hpc_perclass.py`), stage-gate evals
(`eval_stage2_gate_v2/v3.py`), roadmap surgery
(`resync_stage1_roadmap.py`, `update_stage1_roadmap_labels.py`),
verification (`verify_master_stats.py`, `patch_master_stats.py` — both
UNTRACKED, used for the GEN1 master stats corrections),
`gen0_pull_hf_evals.py` (UNTRACKED), synthetic-data staging
(`synth_shards/`, big `.pt` gitignored), `prototype/` (UNTRACKED Typst/
WeasyPrint rendering prototypes for the handbook), `run_hpc_rerun.sh`.
Individually documented as findings in §17; collectively: candidate for a
curated `archive/` move, none deletable blindly (several reproduce
published numbers).

## 4.21 Small directories

- `config/` — 3 legacy YAMLs (attack/train configs for the 8-model + ViT
  era). Not used by Gen-1 (dataclass config instead). HISTORICAL.
- `utils/` — single `metrics.py`. Near-orphaned; see §17.
- `dashboards/` — `lens_app.py` (Streamlit viewer for `rhan_core.lens`
  sessions). Consumes lens capture files.
- `data_generation/` — synthetic STL-10 pipeline (`generate_synthetic_stl10.py`,
  `filter_synthetic_clip.py`, `prepare_synthetic_data.py`,
  `upload_synthetic_hf.py`). Gen-0 era (SBR0 synthetic gates).
- `competition/` — `evaluate_comparison.py`, `generate_heatmaps.py`,
  `output/`. Gitignored; purpose unclear (likely a course competition
  submission) — see §17.
- `logs/` — 3 tracked null-ablation training logs (2026-07).
- `sweep_results/` — one tracked per-seed epsilon-sweep CSV.
- `.verify_samples/` — 210 tracked tiny PNGs (seed-sample verification
  artifacts for the eval protocol); gitignored going forward but tracked
  files remain.
- `build/` — `noesis` PDF build output; tracked stub.
- `.vscode/` — `sessions.json` tracked.
- `.agent/`, `.claude/`, `.freebuff/` — agent working dirs (untracked/ignored
  or tool-local; not research content).

---

# 5. File-by-File Mapping — Canonical Line

Files on the **active Gen-1 line** and the **frozen references** a
contributor will actually touch, in dependency order. (The ~1,300
historical files are covered directory-wise in §4; mapping each of 491
Python files individually would bury the signal — the phase1_training
inventory in §4.8 is the file-level record for those.)

### `noesis_vision/core/schema.py`
**Type:** config schema. **Status:** ACTIVE.
**Purpose:** RHANNXAConfig — schema-locked Gen-1 configuration
(`num_glimpses=4` locked; `record d_z=384` propagated via config hash).
**Consumed by:** trainer, models, tests (`test_schema_version`,
`test_config_backward_compat`).
**Refactor relevance:** single source of Gen-1 hyperparameter truth; any
refactor must keep the config-hash provenance contract.

### `noesis_vision/core/checkpoint.py`
**Type:** infrastructure. **Status:** ACTIVE.
**Purpose:** best/rolling checkpoint pair; `resume_or_abort` — mandatory HF
rolling resume, distinguishes PROVEN-ABSENT from UNVERIFIABLE (cold-start
fix `a4444db`); downloader may stage rolling at destination (`cf6ce8a`).
**Consumed by:** trainer, gate scripts. **Tests:**
`test_checkpoint_never_silently_restarts`, `test_resume_commit_guard`.

### `noesis_vision/core/provenance.py`
**Type:** infrastructure. **Status:** ACTIVE.
**Purpose:** `write_manifest(extra=...)` — per-run provenance manifests.
**Warning:** merges `extra` keys at the manifest TOP level
(`manifest["adv_curriculum"]`, not `manifest["extra"]["adv_curriculum"]`);
readers must key accordingly (the recipe guard does).

### `noesis_vision/core/consistency_assert.py`
**Type:** infrastructure. **Status:** ACTIVE.
**Purpose:** `assert_table_matches_csv` — every summary table must be
re-derivable from its per-seed CSV. Born from the interface-drift incident
class. **Consumed by:** `evaluation/clean_and_robust.py`.
**Duplicate:** `scripts/consistency_assert.py` (§16).

### `noesis_vision/models/backbone.py` + `foveation.py` + `recurrent_block.py`
**Type:** model substrate (Agent C). **Status:** ACTIVE.
**Purpose:** CompactViT (D_z=384, patch 14, 12 blocks, tied refinement
~23.3M params), `foveal_sample` (fovea 56×56 = 4×14; loader operates at 96
but the trunk only ever sees 56px foveal crops — verified, no bug),
`TiedRecurrence`. **Tests:** `test_backbone_param_count`,
`test_foveation_differentiable`, `test_recurrent_tied_weights`.

### `noesis_vision/beliefs/*` (Agent B), `noesis_vision/predictive_coding/*` (Agent E), `noesis_vision/uncertainty/evidential_head.py` (Agent D), `noesis_vision/gaze/*` (Agent F)
**Status:** ACTIVE. **Purpose:** the belief-state stack: S=None
VectorBeliefState + drift; shared glimpse predictor / UpdateNet / precision
(|dW| pre-flight criteria); Dirichlet EvidentialHead; AISv2GazePolicy with
candidate sampler (K=4) and center bias. `gaze/gaze_state.py` is Agent B's
superseded placeholder — kept deliberately (Agent F's integration superseded
it; tests still pin it).
**Tests:** one gradient-flow + shape/stability test per module (see §9).

### `training/adv_curriculum.py`
**Type:** training loss/attack module. **Status:** ACTIVE (committed `33a180b`).
**Purpose:** verbatim Gen-0 TRADES/PGD curriculum:
`CURRICULUM_60 = ((1,20,0.031,2.0,4),(21,40,0.062,2.0,4),(41,60,0.094,2.5,4))`;
`W_TRADES_DEFAULT=0.55`; `pgd_kl_attack` (eval-mode internally, restores
train; random start `clamp(x+0.001·randn)`; α=ε/steps; Linf; ±4.0 normalized
clip); `trades_loss` = CE(clean) + β·KL(adv‖clean); `curriculum_for_epoch`
slices the ramp by thirds; `phase_curriculum(phase, epoch, total)` — every
foundation phase traverses the FULL ramp (matched-compute choice).
**Tests:** `test_adv_curriculum.py` (22), `test_adv_curriculum_integration.py` (7).

### `training/train_generation1_foundation.py`
**Type:** canonical training entry point. **Status:** ACTIVE.
**Purpose:** see §7 (full trace). Header documents the adversarial-recipe
correction and the owned pure-CE planning gap.

### `training/stage_state_machine.py`
**Type:** orchestration. **Status:** ACTIVE.
**Purpose:** Gen-1 phase machine: `FOUNDATION_PHASES` (6),
`DEPENDENCIES` (phase → parent), `GRADIENT_REQUIRED` (per-phase head groups),
`get_next_action()` over `generation1_foundation_roadmap.json`.
**Tests:** `test_stage_state_machine.py`, `test_stage_state_machine_resume.py`.

### `evaluation/clean_and_robust.py`, `shape_texture_bias.py`, `compactness_report.py`, `imagenet100_loader.py`
**Type:** evaluation harness (Agent I). **Status:** ACTIVE.
**Purpose:** norm-space-only PGD eval with seed floor + provenance +
consistency assertion; Geirhos-style shape/texture bias; params/FLOPs
compactness per phase; ImageNet-100 ImageFolder loader with 100-class
structural validation. **Tests:** `test_eval_rhan_protocol`,
`test_eval_structural_consistency`, `test_center_bias_measurement`.

### `scripts/prepare_imagenet100.py`
**Type:** data acquisition. **Status:** ACTIVE.
**Purpose:** pinned `clane9/imagenet-100@0519dc2f…` → JPEG ImageFolder
(126,689 train / 5,000 val / 100 classes), idempotent, fingerprint json
(per-split counts + aggregate content hash + name→dir map).

### `scripts/verify_run_complete.py`, `scripts/freeze_run_manifest.py`
**Type:** Phase-11 verifier + freeze. **Status:** ACTIVE.
**Purpose:** at 6/6 done, verify checkpoints/manifests/eval CSVs/HF sync/
frozen-hash (self-heals state from HF; exit 2 on failure); freeze the
scientific configuration into `runs/production_launch_manifest.json`.
The supervisor runs the verifier automatically (`7db143d`).

### `scripts/stage_state_machine.py` (Gen-0 machine)
**Type:** orchestration (historical, still load-bearing for Gen-0 roadmaps).
**Status:** RELEVANT HISTORICAL — active only if a Gen-0 stage resumes.
Reads `docs/rhan_next_roadmap.json` (`rhan_nx` key, stages gen0 → sbr0–sbr4
→ ais_v2 → hpc_belief). **Name collision** with the Gen-1 module — see §16.

### `scripts/sbr0_gate.py`, `scripts/eval_ais_v2_gate.py`, `scripts/measure_group_dw.py`, `scripts/generate_full_sweep.py`, `scripts/comparator_registry.py`, `scripts/merge_stage1_seed_extension.py`
**Type:** Gen-0 gate/eval tooling. **Status:** RELEVANT HISTORICAL.
**Purpose:** SBR0 gate criteria, AIS-v2 smoke-gate (the salvageable r=0.706
result), |dW| group measurement, 16-seed sweep driver, comparator reuse
integrity, seed-extension merge. Protected by their own tests.

### `phase1_training/train_rhan_next.py`
**Type:** Gen-0 canonical trainer. **Status:** FROZEN (the curriculum source).
**Purpose:** D / AIS-v1 / HPC-only / SBR entrypoint; superset of
`train_rhan_v12.py`; hosts `CURRICULUM_60` (lines 686–688), `--w-trades 0.55`
(line 740), PGD-KL loop (~line 1393). Ported verbatim into
`training/adv_curriculum.py`.

### `phase2_attacks/eval_rhan.py`
**Type:** Gen-0 frozen eval entrypoint. **Status:** FROZEN.
**Purpose:** Finding-17 matched norm-space 5–8-seed PGD protocol. Adapted
(not copied) into `evaluation/clean_and_robust.py`.

### `rhan_core/model.py` + `rhan_core/config/pillar_config.py`
**Type:** Gen-0 model + toggles. **Status:** FROZEN.
**Purpose:** RHANNext (76.7M, STL-10 96×96) with HPC/AIS/SBR/IWM pillars.
Reference for Gen-1 comparisons; imported by tests and by
`phase1_training/train_rhan_next.py`.

### `cloud/gen1/Kaggle_J1_FOUNDATION.py`
**Type:** canonical cloud launcher (Kaggle T4). **Status:** ACTIVE.
**Purpose:** Step 1 clone+commit assertion (refuses pre-correction commits);
Step 2 trainer-source assertions (`adv_curriculum`, `--clean-only`);
smoke chain; Step 4.5 one-time HF reset (`J1_RERUN_RESET=1`) archiving the
pure-CE working set to `archive/gen1_pure_ce_<stamp>/` + roadmap rev bump;
Step 5 data bootstrap to `DATA_ROOT=/kaggle/tmp/imagenet100` (EROFS fix,
probe-by-doing writability, `HF_HUB_CACHE` pin); Step 7 dispatch with
stale-state pre-flight (HF 6/6 + no local manifests → STOP) and post-run
`assert_adversarial_recipe` (every done phase's manifest must record
`clean_only is False`); `NOESIS_DRY_RUN=1` pre-flight mode; Phase-11
verifier at 6/6.

### `cloud/gen1/colab_j1_foundation.py`
**Type:** Colab twin of the above. **Status:** ACTIVE but LAGGING —
compiles; contains NONE of the re-run prep (no Step 4.5, no commit
assertion, no stale-state pre-flight, no data-root fix). Documented in §17
and §20 (P1).

### `report/GEN1_RESULTS_MASTER.md`
**Type:** results master (gitignored; canonical copy on HF rolling repo).
**Status:** ACTIVE artifact (pure-CE arm — the control for the re-run).
**Contents:** 309 lines / 15 sections of artifact-backed pure-CE Gen-1
metrics: seed-level exact-300-sample stats; recurrence→belief_no_f clean Δ
−9.875 pp (Holm p<0.0001, largest negative effect); Friedman df=5
(ε=0 χ²=24.019, p=0.0002); ε=0.094 robustness 0.00–0.12% everywhere;
AIS-v2 swap null (+0.583 pp, p=0.558); §11 bit-identity finding
(`gen1_core` ≡ `ais_v2_swap`, 206/206 tensors, state-dict payload hash
`4814d7c70bfce254`). Absent metric families (d′, uncertainty readouts,
prediction-error, gaze telemetry) require re-instrumented eval.

---

# 6. Python Architecture (Subsystems)

Conceptual RHAN architecture vs where code actually lives:

| Subsystem | Canonical location (Gen-1) | Gen-0 / earlier | Multiple impls? |
|---|---|---|---|
| Backbone / substrate | `noesis_vision/models/backbone.py` (CompactViT, D_z=384) | `rhan_core/model.py` (RHANv12-frozen subclass, 76.7M) | yes — deliberate |
| Foveation | `noesis_vision/models/foveation.py` (fovea 56) | `rhan_core/model.py` (FovealStream 48 STN + Parafoveal 96) | yes |
| Recurrence | `noesis_vision/models/recurrent_block.py` (tied, within-glimpse) | `phase1_training/model_rhan_v*.py` (inter-glimpse feedback) | yes |
| Belief state | `noesis_vision/beliefs/` (S=None Vector) | `rhan_core/beliefs/` (Vector+Structured+relational+SBR) | yes |
| Prediction/error | `noesis_vision/predictive_coding/` (next-glimpse latent target) | `rhan_core/predictive_coding/` (edge-map HPC, 1 level) | yes |
| Uncertainty | `noesis_vision/uncertainty/evidential_head.py` (Dirichlet) | none in rhan_core (that was the point) | no |
| Attention / gaze | `noesis_vision/gaze/` (AIS-v2 policy) | `rhan_core/gaze/` (InfoGain + halting, AIS-v1) | yes — v1 vs v2 is a research axis |
| Memory / S_t | **scaffold only** — S_t=None canonical; structured belief lives in `rhan_core/beliefs/structured_belief.py` (Gen-0, confound carrier) | same | — |
| Classifier / readout | evidential head α-readout in the Gen-1 model composition (Agent J integration) | `rhan_core/model.py` head | — |
| Training | `training/` (3 modules) | `phase1_training/` (~45 trainers) | yes — generationally partitioned |
| Evaluation | `evaluation/` (Agent I harness) | `phase2_attacks/` (sweeps + frozen eval_rhan) | yes |
| Data | `evaluation/imagenet100_loader.py` + `scripts/prepare_imagenet100.py` | `phase1_training/dataset*.py` (CIFAR/STL-10), `data_generation/` (synthetic) | yes |
| Utilities | `noesis_vision/core/` | `utils/metrics.py`, `phase1_training/checkpoint_utils.py` | yes |
| Configuration | `noesis_vision/core/schema.py` + `FoundationConfig` | `rhan_core/config/pillar_config.py`, `config/*.yaml` | yes |
| Checkpointing | `noesis_vision/core/checkpoint.py` | `phase1_training/checkpoint_utils.py`, `rhan_core/artifacts/` | yes |
| Experiment orchestration | `training/stage_state_machine.py` (6 phases) | `scripts/stage_state_machine.py` (8 stages) + roadmap JSONs | yes |
| Introspection | `dashboards/lens_app.py` viewer | `rhan_core/lens/` (capture/hooks/session) | no (complementary) |

**Mismatch note:** the conceptual stack (§18) is followed closely by the
Gen-1 package. Older generations mix these concerns inside single
`model_rhan_*.py` files — expected for their era, documented here so a
contributor doesn't treat e.g. `model_rhan_v6.py`'s internal
predictive-coding as the canonical HPC (it is not; Gen-0's is
`rhan_core/predictive_coding/`, Gen-1's is `noesis_vision/predictive_coding/`).

---

# 7. Training System Map

Full trace of the CANONICAL (Gen-1) training system:

```text
CLI: python3 training/train_generation1_foundation.py
     [--smoke | --phase <p>] [--clean-only] [--force-fresh]
     [--data-root …] [--epochs N] [--batch-size 48] [--num-workers 4] [--hf]
  ↓
configuration: FoundationConfig dataclass (in-file) ∪ RHANNXAConfig
  (noesis_vision/core/schema.py); defaults now carry the adversarial recipe
  (w_trades=0.55, clean_only=False, recipe_version="gen1-adv-curriculum-v1");
  cfg.to_dict() → provenance config hash
  ↓
phase dispatcher: training/stage_state_machine.py — get_next_action() reads
  generation1_foundation_roadmap.json (HF-synced; session death at any point
  resumes by re-running); FOUNDATION_PHASES × DEPENDENCIES
  (backbone_only → recurrence_only → belief_no_f → belief_with_f →
   ais_v2_swap → gen1_core)
  ↓
trainer: run_phase(phase, loaders, state, …)
  • cold start: provenance manifest (write_manifest; extra merged at TOP level)
  • resume: resume_or_abort HF rolling (never silent restart; PROVEN-ABSENT
    vs UNVERIFIABLE distinguished)
  • SILENT-INHERITANCE GUARD: cold start ∧ phase ≠ first ∧ not smoke →
    this phase's best vs DEPENDENCIES[phase] parent's best — bitwise-equal
    payload → SystemExit. (--force-fresh bypasses, deletes best too.)
    NOTE: honesty guard — gen1_core's forward remains functionally identical
    to ais_v2_swap's (plan-level open question, §19)
  • gradient-reach pre-flight: GRADIENT_REQUIRED[phase] groups must receive
    gradients (logged "gradient reach OK for …")
  ↓
data loader: evaluation/imagenet100_loader.py over the ImageFolder root
  (default data/imagenet100; Kaggle: /kaggle/tmp/imagenet100); --smoke uses
  Agent I's synthetic loaders instead
  ↓
model: CompactViT substrate + per-phase composition (see §6); T=4 glimpses;
  steps 1–4 PLACEHOLDER_FIXED_GAZE, steps 5–6 AIS_V2 policy
  ↓
loss: training/adv_curriculum.py — trades_loss = CE(clean) + β·KL(adv‖clean),
  w_trades 0.55; curriculum_for_epoch slices CURRICULUM_60 by thirds:
  epochs 1–20 ε=0.031 β=2.0 · 21–40 ε=0.062 β=2.0 · 41–60 ε=0.094 β=2.5,
  PGD-KL 4 steps (α=ε/4, randn start 0.001, ±4.0 normalized clip).
  --clean-only → pure F.cross_entropy (recorded deviation; smoke default).
  EPOCH LOG TAG: [eps=… beta=… pgd=…] — absence + presence of
  [clean-only] is the visual recipe check
  ↓
optimizer: noesis_vision/core/multi_group_optimizer.py — per-head groups
  (classifier / evidential_head / update_net+precision+predictor /
  gaze_policy per phase), warm-start key matching
  ↓
scheduler: in-trainer cosine (per Gen-0 conventions)
  ↓
checkpoint selection: best-by-val-acc + rolling (Agent A pair); best/rolling
  parity verified before artifacts are cited; cold-start best-vs-parent
  bitwise guard above
  ↓
artifact persistence: checkpoints/*.pth (local), runs/foundation_<phase>/
  manifest.json, report/foundation_*_result.json + _compactness.json +
  foundation_*_eval/*.csv (Agent I eval after EVERY phase), roadmap JSON
  advance()
  ↓
HF/cloud sync: FerrariKazu/rhan-nxa-checkpoints (best) +
  …-checkpoints-rolling (rolling + roadmap + report docs + archive/);
  cloud_setup/Kaggle_J1_FOUNDATION.py drives clone→assert→smoke→data→
  dispatch; supervisor loop (f31348c) wraps crash-recovery; at 6/6 the
  Phase-11 verifier (scripts/verify_run_complete.py) gates exit 0/2
```

**Behavioral modes:**

- **Clean training:** only via `--clean-only` (or smoke) — LOUD and recorded;
  the default silently-pure-CE era is over (guard + assertion + tags).
- **Adversarial training:** default. Curriculum windowing: ALL six phases
  traverse the FULL 60-epoch ramp (matched-compute across phases; Gen-0's
  phases were 20 epochs each). This windowing is a documented design choice.
- **Smoke mode:** entire 6-phase chain on synthetic loaders, clean_only
  forced, artifacts quarantined (`checkpoints/smoke`, `runs/smoke`,
  `report/smoke`), marker file written; seconds-cheap orchestration proof.
- **Production/cloud:** Kaggle notebook with reset gate, stale-state
  pre-flight, post-run recipe assertion, dry-run mode.
- **Resume:** HF rolling mandatory; a fresh session re-bootstraps data
  (the `/kaggle/tmp` cache is ephemeral) but resumes weights from HF.

**Where behavior may differ from documentation:** (1) README "Current"
section documents Gen-0, not Gen-1; (2) `docs/ARCHITECTURE.md` describes
`rhan_core` + Gen-0 stages, updated "at the end of every stage" but its
status line predates the Gen-1 ladder; (3) the Colab twin launcher lacks the
guards the Kaggle launcher has; (4) `phase_curriculum` full-ramp-per-phase
windowing is implemented but not yet review-signed-off in the plan text.

---

# 8. Experiment / Generation History

Only evidence-backed generations are listed. Statuses: CURRENT /
EXPERIMENTAL / LEGACY / ARCHIVED / REJECTED.

### Gen "-1" — CIFAR-10 12-model + human comparison (May 2026)
**Location:** `phase1_training/model*.py` (baselines), `phase2_attacks/`,
`phase3_human_study/`, `phase4_analysis/`, `phase5_sdt/`, `config/`.
**Branches:** `phase/1-*`, `phase/2-*`, `phase/4-analysis`, `phase/5-sdt`,
`dev`, `eyad-pr`, `docs/report`.
**Purpose:** does adversarial robustness scale with global processing?
12 systems + n=18 humans, PGD/AA sweeps, SDT (d′), interpretability figures.
**Key result:** curriculum-trained RHAN reaches ε≈0.185 (CIFAR); humans >0.30;
overconfidence finding (BagNet/EffNet 100% confident at ε=0.30).
**Status:** ARCHIVED (complete, headline numbers live in README).

### RHAN v1–v7 lineage (May–Jun 2026)
**Location:** `phase1_training/model_rhan{,_v3..v7}.py` +
`train_rhan{,_v2..v7}*.py`. **Branches:** `phase/rhan-v2` … `phase/rhan-v6`,
`phase/rhan-trades`, `phase/rhan-trades-curriculum`.
**Purpose:** recurrent top-down feedback architecture search; v3 joint
scratch (91.41% clean), v5 frequency separation, v5-TRADES, TRADES-Hardened,
trades-curriculum (best CIFAR ε≈0.185); v4/v6 regressed (README).
Rejected arms: CBM (irreducible at 32×32), Self-Alignment / Feature-Scatter
(gradient masking). **Status:** LEGACY (documented in RHAN-history.md,
README timeline).

### STL-10 scale-up: UNIFIED → TDV → RHAN-Large (Jun–Aug 2026)
**Location:** `phase1_training/model_rhan_unified.py`,
`model_rhan_stl10*.py`, `train_rhan_unified.py`, `train_rhan_stl10_tdv.py`,
`train_rhan_large_pseudolabel.py`; `checkpoints_tier2/`; `data_generation/`.
**Branch:** `main` (Stage-3 fix is main's tip). **Key result:** RHAN-Large
85.20% clean / AA 10.60% @ 96×96 (README). **Status:** LEGACY/ARCHIVED.

### Gen-0 — RHAN-Next (rhan_core) (Jul–Sep 2026) — RELEVANT HISTORICAL
**Location:** `rhan_core/`, `phase1_training/train_rhan_next.py`,
`phase2_attacks/eval_rhan.py`, `scripts/` gates, `docs/rhan_next_roadmap.json`
(`rhan_nx` key). **Branch:** `feature/rhan-next` (then rebranled content;
now superseded in-place by Gen-1 work).
**Stages:** gen0 → sbr0 → sbr1 → sbr2 → sbr3 → sbr4 → ais_v2 → hpc_belief
(`scripts/stage_state_machine.py`).
**Key results:** AIS-v1 (Stage 1) 49.40% clean / PGD-50@0.094 32.21%; HPC-only
(Stage 2) 55.20% / 27.87%; D = AIS+HPC (Stage 3) 56.16% clean (1 seed of 8);
SBR arms sbr0–sbr4; **the TRADES/PGD curriculum was Gen-0's training
objective for ALL of these** (now ported to Gen-1).
**KNOWN CONFOUND** (documented in `noesis_vision/RHAN_NXA/MASTER_PLAN.md`
Part 0): ais_v2 (D2) and hpc_belief (D3) ran through the `_nx_trainer`
runner that hardcodes `--enable-sbr` — their isolated effects are UNKNOWN
(not merely weak); D3's clean Δ (−9.90) ≡ E2/SBR's numbers to two decimals.
One salvageable result: AIS-v2 smoke-gate preference correlation r=0.706.
**Status:** RELEVANT HISTORICAL — the comparison baseline for Gen-1;
confounded arms must not be cited as isolated effects.

### SBR line (sbr0–sbr4)
**Location:** same Gen-0 infrastructure; `rhan_core/beliefs/structured_belief.py`,
`scripts/sbr0_gate.py`, `checkpoints_hf*/rhan_nx_sbr0_*.pth`.
**Status:** sbr0 gate artifacts tracked; sbr1–4 verdicts in
`report/sbr1_gate_verdict.json` etc. Part of Gen-0; the SBR confound above
applies to D2/D3, not to sbr* stages themselves (correct-by-design there).

### Gen-1 — RHAN-NXA (noesis_vision) (Sep 2026) — CURRENT
**Location:** `noesis_vision/`, `training/`, `evaluation/`, `scripts/`,
`cloud_setup/*j1*`, `tests/`. **Branch:** `feature/rhan-next` (canonical).
**Phases:** six-phase foundation ladder (§7). **Status:** CURRENT —
pure-CE arm completed and archived as control (`report/GEN1_RESULTS_MASTER.md`
+ HF `archive/gen1_pure_ce_<stamp>/`); **corrected TRADES/PGD ladder
re-run in progress on Kaggle** (2026-09-30).
**Gen-1 pure-CE findings so far:** recurrence→belief_no_f clean Δ −9.875 pp
(largest negative effect); AIS-v2 swap null vs its parent (bitwise-identical
best checkpoints — silent-inheritance bug that motivated the new guard);
ε=0.094 robustness ≈ 0% without adversarial training (motivating the port).

### Naming map (do not conflate)
`RHAN-NX` / `RHAN-NXA` / `noesis` appear interchangeably in scripts/docs for
the Gen-0→Gen-1 tooling; `RHAN-Next` names Gen-0's model; `RHANvN` names the
CIFAR/STL-10-era lineage; "NOESIS" is the umbrella brand used in
`docs/NOESIS_FOUNDATION.md` and Gen-1 docs.

---

# 9. Test System Map

**Runner:** `python3 -m pytest tests/ -q` (bare `pytest` INTERNALERRORs in
this environment). **Current state: 406 passed (~2m15s) as of the recipe
correction (33a180b).** 49 test files, all under `tests/`; no tests exist
outside it (root `test_load.py`/`test_ckpt_load.py` are standalone scripts,
not pytest-collected conventions — see §17).

**Categories (with representative files):**

- **Gradient-flow / reachability** (the #1 historical failure mode — the
  v11/v12 detached-recon-loss bug): `test_gradient_flow`,
  `test_drift_to_gradient_flow`, `test_evidential_gradient_flow`,
  `test_hpc_gradient_flow`, `test_hpc_belief_gradient_flow`,
  `test_update_net_gradient_flow`, `test_sbr_gradient_flow`,
  `test_ais_v2_gradient_flow`, `test_t0_produces_exact_zero_error`.
- **Curriculum / adversarial recipe (new, 29 tests):**
  `test_adv_curriculum.py` (22: ramp table, slicing, PGD-KL properties,
  loss form, restore-mode), `test_adv_curriculum_integration.py` (7:
  run_phase-level recipe wiring, manifest `adv_curriculum` record,
  smoke clean_only quarantine).
- **Checkpoint / resume discipline:** `test_checkpoint_never_silently_restarts`,
  `test_resume_commit_guard`, `test_warm_start_key_match`,
  `test_sweep_resume`, `test_hpc_optimizer_group_resume`.
- **State machine / orchestration:** `test_stage_state_machine`,
  `test_stage_state_machine_resume`, `test_sweep_resume`,
  `test_ablation_matrix`.
- **Numerical / shape / stability:** `test_numerical_stability`,
  `test_evidential_shapes`, `test_uncertainty_monotonic`,
  `test_backbone_param_count`, `test_recurrent_tied_weights`,
  `test_predictor_is_cheap`, `test_precision_own_optimizer_group`,
  `test_multi_group_optimizer(_isolation)`, `test_detached_copy_isolation`,
  `test_foveation_differentiable`.
- **Config / schema:** `test_schema_version`, `test_config_backward_compat`,
  `test_pillar_scaffold_import`, `test_interface_imports`.
- **Protocol / artifact integrity:** `test_provenance_manifest`,
  `test_consistency_assert_catches_corruption`,
  `test_comparator_reuse_integrity`, `test_eval_rhan_protocol`,
  `test_eval_structural_consistency`, `test_seed_sample_hashes` (as
  `verify_seed_samples.py` in scratch — script, not pytest),
  `test_center_bias_measurement`, `test_live_perception_determinism`,
  `test_lens_introspection`, `test_diagnostics_next`.
- **Gate criteria:** `test_sbr0_gate_criteria`, `test_preflight_dw`,
  `test_hpc_gate_criteria_independent`, `test_hpc_gate_trend_across_resume`,
  `test_hpc_isolated`, `test_hpc_disable_backward_compat`,
  `test_hpc_diag_baseline`, `test_ais_ablation_flags`,
  `test_gaze_state_canonical`, `test_ais_v2_gaze_policy`,
  `test_j2_ais_v2_integration`, `test_belief_state_composition`.
- **Smoke:** the trainer's `--smoke` is itself a smoke suite (6-phase
  orchestration proof); the Kaggle notebook runs it as a gate.

**Major functionality with NO tests (verified by inspection):**

1. `scripts/prepare_imagenet100.py` — no unit test (validated only by its
   own VERIFY step + the loader's structural check).
2. `scripts/verify_run_complete.py` / `freeze_run_manifest.py` — no unit
   tests (exit-0 verified manually per commit messages).
3. `cloud_setup/Kaggle_J1_FOUNDATION.py` — no automated tests; Step-2
   assertions and recipe-guard paths were AST-tested manually (scratch),
   not committed as tests.
4. `scripts/reset_gen1_ladder.py` — dry-run verified only.
5. `cognitive_vision_lab/` has its own `tests/` dir but it is decoupled from
   the main suite.
6. `evaluation/geirhos_loader.py` and `evaluation/imagenet_c_loader.py` —
   no dedicated tests found.
7. All `phase1_training/` historical trainers — no tests (accepted: frozen).

---

# 10. Configuration Map

Configuration is split across generations. Values duplicated in multiple
places are flagged, not fixed.

| Source | Generation | Role | Notes |
|---|---|---|---|
| `training/train_generation1_foundation.py::FoundationConfig` | Gen-1 | canonical run config | dataclass → `to_dict()` → provenance config hash; CLI overrides |
| `noesis_vision/core/schema.py::RHANNXAConfig` | Gen-1 | schema-locked model config | `num_glimpses=4`; `record d_z=384` |
| `docs/rhan_next_roadmap.json` / `report/generation1_foundation_roadmap.json` | Gen-0 / Gen-1 | orchestration state | tracked / gitignored-copy; HF rolling is canonical |
| `runs/production_launch_manifest.json` | Gen-1 | frozen scientific config | written by `freeze_run_manifest.py`; stale copy would mis-bless old config (reset deletes it) |
| `rhan_core/config/pillar_config.py::RHANNextConfig` | Gen-0 | pillar toggles | `enable_sbr`, `enable_iwm` must stay False in Gen-0 scaffolds |
| `phase1_training/train_rhan_next.py` argparse defaults | Gen-0 | curriculum truth | ε/β table lines 686–688; `--w-trades 0.55` line 740 — **duplicated** into `training/adv_curriculum.py` (intentional port; drift risk if either changes) |
| `config/attack_config.yaml`, `train_config.yaml`, `train_config_vit.yaml` | Gen "-1" | legacy YAML | not consumed by Gen-0/Gen-1 code (checked: no loader imports) |
| `cloud_setup/Kaggle_J1_FOUNDATION.py` env vars | Gen-1 | cloud config | `J1_RERUN_RESET`, `NOESIS_DRY_RUN`, `J1_BATCH`, `J1_WORKERS`, `DATA_ROOT=/kaggle/tmp/imagenet100`, `HF_HUB_CACHE=/kaggle/tmp/hf_cache` (pinned then removed after conversion) |
| `.env` (local) / Kaggle secrets | all | `HF_TOKEN` | gitignored; notebook resolves token from `--hf-token`/`$HF_TOKEN`/`.env` |
| Hardcoded constants | mixed | substrate locks | `D_Z=384`, `PATCH_SIZE=14`, fovea 56, `CURRICULUM_60`, `W_TRADES_DEFAULT`, `RAND_START_MAG=0.001`, K=4 candidates, MIN_PROTOCOL_SEEDS |

**Duplicated values (documented):** ε/β/PGD table + w_trades (Gen-0 trainer ⇄
`training/adv_curriculum.py`); batch-size/num-workers defaults (trainer ⇄
Kaggle dispatch ⇄ Colab twin); roadmap JSON in three+ places (docs/, report/,
HF rolling — HF canonical); `consistency_assert` code in two modules; batch
defaults also echoed in `docs/stage3_environment.json` (Gen-0).

---

# 11. Dataset / Data Pipeline Map

```text
raw dataset
  ↓ HuggingFace clane9/imagenet-100 @ pinned revision 0519dc2f…
  ↓ scripts/prepare_imagenet100.py (download parquet shards → decode ONCE →
    JPEG ImageFolder; fingerprint.json = counts + content hash + name→dir map)
  ↓ <root>/train/<wnid>/*.jpg , <root>/val/<wnid>/*.jpg
  ↓ evaluation/imagenet100_loader.py (structural contract: exactly 100 class
    dirs/split; 126,689 train / 5,000 val)
  ↓ preprocessing/normalization (loader transforms; ImageNet stats; 96px
    operating point for the LOADER)
  ↓ augmentation (trainer-side standard crop/flip — in-file)
  ↓ foveation: noesis_vision/models/foveation.foveal_sample — trunk sees
    56×56 foveal crops (56 = 4×patch 14; backbone.py raises on non-divisible
    img_size). VERIFIED: 96 is only the loader's operating point; the model
    never receives a full 96px pass. (48px would be non-divisible → 56 chosen.)
  ↓ model input (T=4 glimpses, gaze-selected coordinates)
```

**Other datasets:**
- **STL-10 (96×96)** — Gen-0 and earlier: `phase1_training/dataset_stl10.py`
  (+ unlabeled variant), `run_eval_stl10.py`; local cache `data/stl10/`,
  `data/stl10_binary/` (gitignored). Semi-supervised pseudo-label set
  (100k) via `train_rhan_large_pseudolabel.py`; TDV UCF-101 pretraining
  via `train_rhan_video_tdv.py`.
- **CIFAR-10** — Gen "-1": `phase1_training/dataset.py`, `dataset_vit.py`.
- **Synthetic STL-10** — Gen-0 SBR0 gates: `data_generation/`,
  `scratch/synth_shards/`, `scratch/build_synthetic_pt.py` (gitignored
  artifacts; trainer `--smoke` reuses the built-in synthetic loader, not these).
- **Geirhos cue-conflict / ImageNet-C** — loaders exist
  (`evaluation/geirhos_loader.py`, `imagenet_c_loader.py`) for bias/robustness
  probes; no committed runs found for Gen-1.
- **Human study data** — `phase3_human_study/` (n=18, 1,800 trials) — static.

**Locations:** `data/` (gitignored, local), HF cache (`~/.cache/huggingface`
local; `/kaggle/tmp/hf_cache` on Kaggle), `/kaggle/tmp/imagenet100`
(Kaggle session; ephemeral — re-bootstrap ~19 GB per new session).

---

# 12. Evaluation Map

### Clean + robust (CANONICAL for Gen-1)
`evaluation/clean_and_robust.py` (via `scripts/generate_full_sweep.py`):
norm-space ε only; PGD (generic_pgd: uniform start, α=ε/4, ±4 clip);
per-seed CSV (canonical Gen-0 schema: `ckpt_label, seed, eps_pixel,
acc_pct, …`); seed floor ≥ MIN_PROTOCOL_SEEDS (explicit `allow_quick`
escape for dev sanity); `eval_provenance.json` (ckpt SHA-256, seeds,
settings); consistency assertion before any table write. **Outputs:**
`report/foundation_*_eval/` CSVs + `_result.json`.

### Clean + robust (FROZEN, Gen-0)
`phase2_attacks/eval_rhan.py`: Finding-17 matched protocol, 5–8 seeds,
n=300/seed, PGD-50/100 @ ε∈{0, 0.094}; plus the sweep family
(`eval_full_epsilon_sweep.py`, `eval_autoattack*.py`, `eval_pgd_v2..v8.py`
in `phase1_training/`, root-level `eval_pgd_*.py` ×13 — see §16/§17 for the
eval-s sprawl).

### Shape/texture bias
`evaluation/shape_texture_bias.py` + `geirhos_loader.py` — Gen-1 bias probes.

### Compactness
`evaluation/compactness_report.py` — params/FLOPs after EVERY phase
(Part 3 budget evidence); `_compactness.json` artifacts.

### Calibration / uncertainty readouts
**Not implemented for Gen-1.** EvidentialHead α exists (uncertainty scalar
C/Σα), but no committed eval extracts/calibrates it (master doc lists d′,
uncertainty readouts, prediction-error, gaze telemetry as absent metric
families needing re-instrumented eval). Gen "-1" d′ lives in `phase5_sdt/`
(CIFAR-era, not applicable to Gen-1 outputs).

### Gaze / active-perception telemetry
`rhan_core/lens/` (capture/hooks/session) + `dashboards/lens_app.py`
viewer — Gen-0-era instrumentation; Gen-1 telemetry not yet wired.

### Statistical analysis
Gen-0: seed sweeps + Holm-corrected paired stats and Friedman tests live in
analysis scripts under `scripts/` + scratch (`verify_master_stats.py`,
UNTRACKED). Gen "-1": `phase5_sdt/` (d′, criterion). No shared stats module.

### Gates (pre-registered)
`scripts/sbr0_gate.py`, `scripts/eval_ais_v2_gate.py`,
`scripts/measure_group_dw.py`, `scripts/eval_ais_v2_gate.py`; criteria
pinned by tests (`test_sbr0_gate_criteria`, `test_preflight_dw`,
`test_hpc_gate_criteria_independent`).

### Reporting
`report/` JSONs/MD + `GEN1_RESULTS_MASTER.md` (artifact-backed) +
`report/generate_figures.py` + `cloud_setup/generate_figures.py` +
`docs/rhan_nxa` handbook builders.

---

# 13. Experiment Artifacts

| Kind | Location | Controlled? | Notes |
|---|---|---|---|
| Checkpoints (all gens) | `checkpoints/*.pth` | NO (gitignored) | ~100+ files; HF is canonical |
| Gen-1 best/rolling | `checkpoints/foundation_*_{best,rolling}.pth` | NO | reset deletes/archives these |
| SBR0 sync pair | `checkpoints_hf*/rhan_nx_sbr0_*.pth` | YES (tracked) | historical smoke-era sync targets |
| STL-10 large weights | `checkpoints_tier2/*.pth` (4) | YES (tracked) | pre-100MB-guard era |
| Provenance manifests | `runs/foundation_*/manifest.json` | NO | dispatch pre-flight keys on them; extra merged at TOP level |
| Launch manifest | `runs/production_launch_manifest.json` | NO | frozen config; stale copy dangerous |
| Roadmap state | `docs/rhan_next_roadmap.json` (tracked), `report/generation1_foundation_roadmap.json` (not), HF rolling (canonical), `checkpoints/generation1_foundation_roadmap.json` (untracked) | mixed | four copies; HF wins |
| Eval CSVs + provenance | `report/foundation_*_eval/` | mixed (17 tracked files in report/) | mirrors HF |
| Result JSONs | `report/foundation_*_result.json`, `_compactness.json` | mixed | per-phase |
| Master results doc | `report/GEN1_RESULTS_MASTER.md` | NO (gitignored) | canonical copy on HF rolling |
| Gen-0 sweeps | `report/sweep_stage3_d`, `sweep_stage4_*`, `sweep_rerun_*8seed*`, `sweep_matched_finding17` | NO | HF `report/sweep_rhan_nx_*` also exist |
| Figures | `figures{,_v2,_v3}/`, `phase4_analysis/figures/`, `report/assets/` | NO (mostly) | regenerable via generators |
| Reports (LaTeX/PDF) | `Paper/`, `RHANv12/`, `RHANv10Report/`, `rhan_math/`, `tier1/`, `docs/` | mixed | 117 tracked PDFs, 491 tracked PNGs |
| Logs | `logs/*.log` (3 tracked), `report/supervisor.log`, `train_norecon.log` (root, tracked) | mixed | historical |
| Sample verification | `.verify_samples/` (210 PNGs) | YES | eval-protocol seed-sample checks; ignored going forward |
| Caches | `.venv/`, `.pytest_cache/`, `__pycache__/`, `cognitive_vision_lab/cache/` | NO | never commit |
| Zone.Identifier files (19) | scattered (root, rhan_core, phase1_training, checkpoints_tier2, docs) | YES (accidental) | Windows ADS junk; tracked — see §20 P3 |
| `concept_ablation_results.npz`, root CSVs/PNGs/PDFs | root | YES (accidental) | loose artifacts |

**Lifecycle rule observed in practice:** git = source + docs + small
evidence; HuggingFace = weights + reports + archives; local dirs = working
copies. `report/` being gitignored-while-partially-tracked is documented
friction (see §17).

---

# 14. Notebooks & Cloud Launchers

| File | Role | Status |
|---|---|---|
| `T4x2.ipynb` (root) | interactive scratch notebook | MODIFIED, **never commit** (standing exclusion) |
| `cloud_setup/Kaggle_J1_FOUNDATION.py` | canonical Gen-1 Kaggle launcher (percented-script notebook) | ACTIVE, canonical |
| `cloud_setup/colab_j1_foundation.py` | Colab twin | ACTIVE but missing re-run prep (§17) |
| `cloud_setup/colab_notebook_noesis.py`, `Kaggle_NOESIS.py` | Gen-0 Stage 1–3 protocol notebooks | RELEVANT HISTORICAL |
| `cloud_setup/colab_v11_*`, `kaggle_v11_*`, `colab_v12_step0.py`, `kaggle_v12_step0.py` | v11/v12 3-seed protocol / isolation runs | HISTORICAL |
| `cloud_setup/kaggle_train_synthetic.py`, `colab_train_synthetic.py` | synthetic smoke training | HISTORICAL |
| `cloud_setup/kaggle_run_pseudolabel_pipeline.py`, `colab_run_pseudolabel_pipeline.py`, `colab_run_pipeline.py` | RHAN-Large pipeline automation | HISTORICAL (README-referenced) |
| `cloud_setup/lightning_setup.py` | older Lightning env setup | HISTORICAL |
| `cloud_setup/generate_figures.py`, `generate_rhan_visualizations*.py` | figure generation on cloud | HISTORICAL |
| `cloud_setup/run_j1_local.sh`, `run_j1_supervised.sh` | local RTX-4060 launch + supervised loop | ACTIVE |
| `dashboards/lens_app.py` | Streamlit lens viewer | side app |

The cloud "notebooks" are Python files with `# %%` cells executed in
Kaggle/Colab. They duplicate environment setup and dispatch logic across
generations (see §16). `cloud_setup/__pycache__/` contains compiled traces of
renamed predecessors (e.g. `colab_v11_isolation_run_a`) — listed in §17.

---

# 15. Git / Branch Architecture

Local branches (committer-date order), with purpose/relevance:

| Branch | Last commit | Purpose / generation | Classification |
|---|---|---|---|
| `feature/rhan-next` ★ | 2026-09-30 `fef50f3` | Gen-0 → Gen-1 active line | **ACTIVE (canonical)** |
| `main` | 2026-08-26 `4f18bd4` | STL-10 era tip ("Stage 3 PGD-100 fix"); README's declared home for RHAN-Large | RELEVANT HISTORICAL |
| `backup-pre-rewrite` | 2026-09-17 `55cbcf2` | safety snapshot before a Gen-0-era rewrite (ais_v2 smoke gate fix on top) | RELEVANT HISTORICAL (verify contents before any later deletion) |
| `dev` | 2026-05-18 | early dev (8-model playground demo) | SUPERSEDED |
| `docs/report` | 2026-05-03 | 5-model repo scaffold | SUPERSEDED |
| `eyad-pr` | 2026-05-10 | contributor PR line (ResNet confusion matrix) | SUPERSEDED |
| `phase/1-{bagnet,clip,cornets,efficientnet,shaperesnet,vit}` | 2026-05-03→17 | per-model Gen "-1" lines | SUPERSEDED (results landed in main/README) |
| `phase/2-{bagnet,efficientnet,shaperesnet,vit}-attacks` | 2026-05-03 | per-model attack lines | SUPERSEDED |
| `phase/4-analysis`, `phase/5-sdt` | 2026-05-16→10 | analysis + SDT | SUPERSEDED (outputs in tree) |
| `phase/trial-1-clip`, `phase/trial-2-adaptive` | 2026-05-18 | both point at the same merge commit `67c5724` | SUPERSEDED / duplicate tips |
| `phase/rhan-v2`, `phase/rhan-v3-adaptive`, `phase/rhan-v4`, `phase/rhan-v5`, `phase/rhan-v6` | 2026-05-27→30 | RHAN v2–v6 lines | SUPERSEDED (provenance) |
| `phase/rhan-trades-curriculum` | 2026-06-04 | best-CIFAR curriculum line | RELEVANT HISTORICAL (curriculum provenance) |

**Remote-only:** `origin/phase/1-rhan`, `origin/phase/rhan-trades` (no local
counterparts; not checked out here). `origin/phase/rhan-self-alignment`,
`-feature-scatter`, `-cbm`, `-tdv` are referenced in README's model table
but were **not found** among remote refs fetched here — the README model
table cites branches that do not all exist on origin (documented as a docs
drift finding, §17).

**Branch audit complete (2026-10-05).** Branches pruned:
- DELETED (22 local, 16 remote): `backup-pre-rewrite`, `dev`, `docs/report`,
  `eyad-pr`, `phase/1-bagnet`, `phase/1-clip`, `phase/1-cornets`,
  `phase/1-efficientnet`, `phase/1-shaperesnet` (remote only),
  `phase/1-vit`, `phase/2-bagnet-attacks`, `phase/2-efficientnet-attacks`,
  `phase/2-shaperesnet-attacks`, `phase/2-vit-attacks`, `phase/4-analysis`,
  `phase/5-sdt`, `phase/rhan-trades`, `phase/rhan-trades-curriculum`,
  `phase/rhan-v2`, `phase/rhan-v3-adaptive`, `phase/rhan-v4`,
  `phase/rhan-v5`, `phase/rhan-v6`, `phase/trial-1-clip`,
  `phase/trial-2-adaptive`.
- KEEP historical (1): `phase/1-shaperesnet` — retained because it carries
  unique research content (ShapeResNet50 model, checkpoints, adversarial
  arrays, attack scripts) not present elsewhere.
- 1 tag preserved: `forensic-nxa-2026-10-03-final`.

**Git-hygiene facts:** `*.pth.sync` ignored after HF-sync files broke the
100 MB push guard (`7cda880`); `.mailmap` canonicalizes contributor aliases;
19 `Zone.Identifier` files tracked (Windows ADS residue).

---

# 16. Duplication Detection

Documented pairs/groups. Nothing consolidated here.

1. **`training/stage_state_machine.py` ⇄ `scripts/stage_state_machine.py`**
   Same basename, different machines (Gen-1 six-phase vs Gen-0 eight-stage).
   Both are import-ambiguous from the repo root in casual tooling.
   Newer: `training/` (Gen-1). Canonical: both within their generations.
   Future: rename one (e.g. `training/foundation_state_machine.py`,
   `scripts/gen0_stage_state_machine.py`) — P2.

2. **`noesis_vision/core/consistency_assert.py` ⇄ `scripts/consistency_assert.py`**
   The scripts/ copy predates the package port (port-table disposition
   "PORT VERBATIM"). Canonical: the package module (the evaluation harness
   imports it). Scripts copy appears kept for legacy callers.
   Future: point remaining callers at the package module — P2.

3. **Gen-0 curriculum table ⇄ `training/adv_curriculum.py`**
   ε/β/steps + w_trades + attack constants duplicated verbatim by design
   (port). Drift risk if Gen-0's trainer is ever tuned again. Future: a
   parity test asserting the two tables are identical — P2.

4. **PGD implementations** (three conventions, all intentional):
   `training/adv_curriculum.pgd_kl_attack` (train-time KL, randn start) ·
   `evaluation/clean_and_robust.generic_pgd` (eval, uniform start, α=ε/4) ·
   `phase2_attacks/pgd.py` + frozen `eval_rhan.py` (Gen "-1"/Gen-0 sweeps).
   Future: document-only; never merge train/eval attackers (different
   threat models).

5. **`noesis_vision/beliefs/vector_belief.py` ⇄ `rhan_core/beliefs/vector_belief.py`**
   Same name, different semantics (Gen-1 S=None carrier vs Gen-0 pillar
   belief). Deliberate re-implementation; keep both; naming only.

6. **Cloud launchers** — every generation has Kaggle+Colab twins with
   copy-pasted env setup (Kaggle_J1_FOUNDATION ⇄ colab_j1_foundation;
   colab_notebook_noesis ⇄ Kaggle_NOESIS; v11/v12 pairs; pipeline pairs).
   The J1 pair is currently **divergent** (Colab lacks re-run prep).
   Future: shared setup module or generated-from-one-source — P1.

7. **Roadmap JSON copies** (4): `docs/rhan_next_roadmap.json` (tracked,
   Gen-0 key `rhan_nx`), `report/generation1_foundation_roadmap.json`
   (gitignored, Gen-1), `checkpoints/generation1_foundation_roadmap.json`
   (untracked local), HF rolling repo (canonical, HF-native sync).
   Future: single loader path through `noesis_vision.core.checkpoint` — P2.

8. **Eval-script sprawl at repo root**: `eval_pgd_v2..v5.py`,
   `eval_pgd_{arrow,final,hf,hf2,quick,sweep,thresh}.py`,
   `eval_aa_v2.py`, `eval_aa_selfalign.py`,
   `eval_quick_perclass{,_aa,_aa_correct}.py`, plus `phase2_attacks/eval_*`
   and `phase1_training/eval_*`. Each was a one-off sweep variant of the
   STL-10/CIFAR era. Future: archive as a group with a README index — P1/P2.

9. **`app.py` (root) ⇄ `cognitive_vision_lab/app.py`** — different apps
   (root = old demo entry; lab = Streamlit platform). Unrelated despite name.

---

# 17. Suspicious / Unclear Areas

Careful language; nothing here was modified.

- **README staleness (high-impact):** the README's "RHAN-Next (Current)"
  section describes Gen-0 (`rhan_core`), its reproduce commands use
  `phase1_training/train_rhan_next.py`, and its Key-Files table predates
  `noesis_vision/`. No consumer breaks (it's docs), but a new contributor
  would start in the wrong generation. Also cites branches not present on
  origin (`phase/rhan-self-alignment`, `-feature-scatter`, `-cbm`, `-tdv`).
- **`cloud_setup/colab_j1_foundation.py`** — compiles, but contains none of
  the Kaggle notebook's re-run prep (Step 4.5 reset, commit assertion,
  stale-state pre-flight, `DATA_ROOT` fix, recipe guard). Running the Colab
  twin against the corrected ladder would not carry the guards. Candidate:
  port the prep (P1).
- **`report/mock_true_empirical_metrics_HISTORICAL.json`** — filename itself
  flags mock data, marked HISTORICAL; kept as provenance. Do not cite; do
  not delete.
- **`competition/`** — gitignored folder with tracked comparison/heatmap
  scripts and `output/`. No README, no references from other code found
  during static inspection; purpose inferred as a course-competition
  submission. Candidate for later archival review.
- **Root-level loose scripts** (`demo.py`, `app.py`, `bench_pgd.py`,
  `check_parquet.py`, `concept_ablation.py`, `upload_pseudolabel.py`,
  `simple_inspect.py`, `inspect_*.py`, `test_load.py`,
  `test_ckpt_load.py`) — Gen "-1"/STL-10 era one-offs at the repo root,
  outside any package. `test_*.py` at root are NOT pytest-collected by the
  canonical `tests/` invocation and do not follow its conventions.
- **`utils/metrics.py`** — no current consumer identified during static
  repository inspection (Gen-1 code uses its own modules); candidate for
  later archival review.
- **`scripts/consistency_assert.py`** — superseded twin of the package
  module; consumers not exhaustively enumerated.
- **`scratch/` (63 tracked scripts)** — mixture of: reproducers for published
  numbers (`verify_master_stats.py`, UNTRACKED, used for the master-doc
  corrections; `gen0_pull_hf_evals.py`, UNTRACKED), roadmap surgery
  (`resync_stage1_roadmap.py`), debugging (`diag_nan_*.py`,
  `debug_pgd_*.py`), HF state checks (`check_hf_*.py` ×6), and one-off
  evals. Several duplicate each other's logic (e.g. three
  `eval_stage2_gate_v*` versions kept). No index; individually cheap,
  collectively opaque. Candidate: curated archive with provenance headers.
- **`cloud_setup/__pycache__/`** — compiled traces of files that no longer
  exist under those names (`_stage3_block`, `colab_v11_isolation_run_a/b`,
  `kaggle_v11_*`). Compiled-only remnants; the sources were renamed. Not
  harmful; confusing on first read.
- **19 tracked `*:Zone.Identifier` files** (root, `rhan_core/`,
  `phase1_training/`, `checkpoints_tier2/`, `docs/`) — Windows download
  metadata accidentally committed.
- **`checkpoints_tier2/*.pth` tracked (4 files)** — large binaries in git
  from the pre-guard era; they inflate clones. Historical — documented, not
  removed.
- **`checkpoints_hf/` + `checkpoints_hf_rolling/`** — single tracked `.pth`
  each (SBR0 smoke-era sync targets); naming suggests a general HF layout
  but they are one-off.
- **`docs/` gitignored but `docs/rhan_next_roadmap.json` tracked** — the
  tracked roadmap survives via pre-ignore tracking (like `report/`'s 17
  files). New files under `docs/`/`report/` will NOT be tracked unless
  excepted (this document required an explicit `!docs/REPOSITORY_MAP.md`
  exception to be committable — added to `.gitignore` as the single
  necessary metadata change).
- **`eval_pgd_v2..v5` versioned families** (root + `phase1_training/`) —
  versioned one-offs with no index; the "current" one per generation is
  only inferable from README/roadmap references.
- **`phase_curriculum` windowing** — every foundation phase traverses the
  full 60-epoch ramp (matched-compute choice, documented in the module
  docstring); the plan text has not been amended to record this choice.
  Design question, not a bug.
- **Silent-inheritance guard scope** — guarantees provenance honesty
  (bitwise difference), not scientific difference: `gen1_core`'s forward
  remains functionally identical to `ais_v2_swap`'s by ladder construction.
  Plan-level open question (mirrors the Gen-0 D2/D3 lesson).
- **Two `stage_state_machine` modules, two `consistency_assert` modules,
  two `vector_belief` modules, two `multi_group_optimizer` modules**
  (`rhan_core/optim/` ⇄ `noesis_vision/core/`) — see §16; each intentional,
  collectively a navigation hazard.
- **`.verify_samples/` 210 tracked PNGs** — eval-protocol verification
  samples now gitignored-but-tracked; historical bulk.
- **`train_norecon.log`, `vit_eval.txt`, `resnet_eval.txt`** (root, tracked)
  — loose training/eval logs with no sibling structure.

---

# 18. Conceptual Dependency Graph

Intended perceptual architecture (per RHAN-NXA docs) and the paths that
implement it (Gen-1 canonical):

```text
Data (evaluation/imagenet100_loader.py, scripts/prepare_imagenet100.py)
  ↓
Foveation (noesis_vision/models/foveation.py — foveal_sample, 56×56)
  ↓
Backbone (noesis_vision/models/backbone.py — CompactViT, D_z=384)
  ↓
Recurrence (noesis_vision/models/recurrent_block.py — tied refinement,
            within-glimpse; T=4 glimpses orchestrated by the trainer)
  ↓
Belief (noesis_vision/beliefs/ — VectorBeliefState S=None, drift)
  ↓
Prediction/Error (noesis_vision/predictive_coding/ — glimpse_predictor,
                  UpdateNet; error_target = latent_next_glimpse)
  ↓
Uncertainty (noesis_vision/uncertainty/evidential_head.py — Dirichlet)
  ↓
Attention/Gaze (noesis_vision/gaze/ — AISv2GazePolicy, candidate_sampler)
  ↓
Readout (evidential α → class; composition in the Gen-1 model assembly)
  ↓
Loss (training/adv_curriculum.py TRADES/PGD + CE)
  ↓
Evaluation (evaluation/ — clean_and_robust, shape_texture_bias, compactness)
```

**Match:** the Gen-1 package follows the conceptual architecture nearly
1:1 (by construction — Agent 0's dependency graph,
`noesis_vision/core/dependency_graph.md`, enforced it; gradient-reach
pre-flights enforce it at runtime).

**Mismatch:** no *module* implements memory/S_t in the canonical build
(structured belief remains Gen-0 heritage in `rhan_core/`; Gen-1 plan
declares S_t=None). Introspection (lens) sits on the Gen-0 side only.
Training-system dependencies (state machine → trainer → harness) live
outside the perceptual stack, as intended.

---

# 19. Current vs Intended Architecture

Intended (RHAN-NXA plan, `noesis_vision/RHAN_NXA/MASTER_PLAN.md` + book)
vs what the code currently does:

| # | Intended | Current | Status |
|---|---|---|---|
| 1 | BeliefState `B_t = (z_t, S_t, U_t, E_t, A_t)` with explicit None-propagation | z/U/E/A implemented; **S_t=None** canonical | IMPLEMENTED (S_t arm deferred; steps 7+ not in this trainer) |
| 2 | Prediction-error-driven belief update `z_{t+1} = z_t + Pi·UpdateNet(z,E)` | implemented (Agent E), pre-flight \|dW\| criteria, own optimizer groups | IMPLEMENTED |
| 3 | Prediction target = next glimpse's latent (`latent_next_glimpse` LOCKED) | implemented and locked by default | IMPLEMENTED |
| 4 | Uncertainty U_t as Dirichlet class evidence | EvidentialHead implemented; **no eval reads/calibrates U_t yet** | PARTIALLY (training-only; evaluation gap documented in master doc) |
| 5 | AIS-v2 gaze: predict→observe→error→precision→update→GAZE-SELECT, K=4–8 | implemented (Agent F), K=4 locked; soft train / hard inference | IMPLEMENTED (phases 5–6) |
| 6 | Classification is a readout, not the purpose | structurally true; but pure-CE Gen-1 arm showed readout-dominated dynamics (belief Δ −9.9 pp) | IMPLEMENTED; scientific gap being addressed by recipe re-run |
| 7 | Adversarial robustness as a first-class training property | **was MISSING in the first Gen-1 run (pure CE — owned planning gap); now ported** (TRADES/PGD curriculum default) | CORRECTED 2026-09-29; re-run in progress |
| 8 | Per-phase isolation (no silent inheritance) | guard added (bitwise best-vs-parent); **the completed pure-CE ladder DID silently inherit** (`gen1_core` ≡ `ais_v2_swap`, hash `4814d7c7…`) | GUARD ADDED; re-run will regenerate honest per-phase bests |
| 9 | L_stab stability diagnostic | diagnostic-only in phase 6; machinery in docs, thin in code | PARTIAL |
| 10 | Steps 7+ (S_t arm, L_stab arm, ablation matrix) | not present in `training/` | NOT IMPLEMENTED (out of scope for the foundation trainer, by design) |
| 11 | Matched 5–16-seed evaluation protocol | harness enforces seed floor; Gen-1 evals so far ran the protocol on pure-CE arm | IMPLEMENTED |
| 12 | Representation-level uncertainty | explicitly deferred (documented tension, plan 1.A) | DEFERRED (documented) |
| 13 | Gaze telemetry / prediction-error readouts as evaluation outputs | not instrumented for Gen-1 | GAP (master doc §absent-metrics) |
| 14 | Documentation equals implementation | Gen-1 docs carry explicit "specified, not implemented" disclaimers and Source-decision pointers | DISCIPLINED (documented divergence) |

Also: intended README-level cohesion is **not** current — the README
describes Gen-0 as current (see §17).

---

# 20. Implemented Reorganization (was Refactor Candidates)

This section was PROPOSED in the pre-refactor audit. It has been FULLY
IMPLEMENTED as of 2026-10-05. See `docs/repository_reorganization.md` for the
complete changelog and `README.md` for the on-boarding map.

### P0 — Structural blockers

**P0-1. New-contributor entry point points at the wrong generation.**
Problem: README "Current" = Gen-0; active line = Gen-1.
Evidence: §2 conflict items 5–6; README Key-Files/Reproduce sections.
Location: `README.md`.
Direction: add a short "Current = RHAN-NXA (noesis_vision/, training/)"
banner + links; move Gen-0 content under a "Generation 0" heading.
Risk: none (docs-only). Dependencies: none.

**P0-2. Correction guards exist only in the Kaggle launcher.**
Problem: the Colab twin can launch the ladder without the reset/assert/
pre-flight/recipe guards; a guarded and an unguarded run could both claim
provenance.
Evidence: §17 item 2; file diff of `cloud_setup/colab_j1_foundation.py`.
Location: `cloud_setup/`.
Direction: port Step 4.5/Step-2 assertions/pre-flight/`assert_adversarial_recipe`
(or factor shared guard code into one module both launchers import).
Risk: low. Dependencies: none (but coordinate with the in-flight Kaggle run).

### P1 — High-value consolidation

**P1-1. Duplicate `stage_state_machine` basenames.** Evidence: §16-1.
Direction: rename Gen-0's to `scripts/gen0_stage_state_machine.py` (import
update only in Gen-0-era callers). Risk: import churn in frozen-era scripts;
mitigate with grep-verified references. 

**P1-2. Root-level eval-script sprawl (~17 root `eval_*.py` + phase
families).** Evidence: §16-8, §17. Direction: `archive/evals_<gen>/` move
with a README index mapping script → generation → published number.
Risk: breaking ad-hoc reproduce commands; keep shims or document paths.

**P1-3. Roadmap JSON four-copy ambiguity.** Evidence: §16-7. Direction: one
canonical path (HF rolling) + thin local cache; document precedence.
Risk: resume tooling reads multiple paths today — change carefully.

**P1-4. Duplicated curriculum constants without a parity test.**
Evidence: §16-3. Direction: a test asserting `CURRICULUM_60` +
`W_TRADES_DEFAULT` in `training/adv_curriculum.py` match the Gen-0 trainer's
table (parse or import). Risk: none.

### P2 — Maintainability

- **P2-1.** `scripts/consistency_assert.py` → re-export or deprecate in favor
  of the package module.
- **P2-2.** ~~`docs/` ignore-vs-tracked friction~~ **RESOLVED 2026-09-30
  (commit `f7b8bbe`)**: `docs/` is now a normally tracked directory with
  explicit ignored subpaths (PDFs, Zone.Identifier files, stage3 env
  snapshot); the Gen-0 research record (`docs/research/`,
  `NOESIS_FOUNDATION.md`, `Stage_E1-3_Analysis.md`) was added to git in
  commit `5c7a8d1`. `report/` policy still open.
- **P2-3.** `scratch/` curation: provenance headers (generation, purpose,
  artifact produced) or grouped `archive/scratch_<gen>/`.
- **P2-4.** Config dedup: single source for batch/worker defaults shared by
  trainer + launchers; document curriculum-table ownership (Gen-0 trainer =
  historical truth; `adv_curriculum.py` = operational truth).
- **P2-5.** `cognitive_vision_lab/` extraction into its own repository
  (independent app, own deps, own Docker).
- **P2-6.** `utils/metrics.py`: adopt or archive (no identified consumer).

### P3 — Nice-to-have

- **P3-1.** Remove the 19 tracked `*:Zone.Identifier` files (Windows ADS
  residue) in a dedicated hygiene commit.
- **P3-2.** Migrate the 4 tracked `checkpoints_tier2/*.pth` + 2
  `checkpoints_hf*/pth` binaries to HF (LFS or repo), leaving pointer docs.
  Risk: clone-size win; breaks direct-path consumers — check first.
- **P3-3.** Root loose artifacts (`concept_ablation_results.npz`,
  `confusion_matrix*.png`, `*.log`, `*.txt`) → `archive/root_artifacts/`.
- **P3-4.** Rename `T4x2.ipynb` exclusion into `.gitignore` (currently
  only convention) or `git update-index --skip-worktree`.

---

# 21. Implemented Repository Architecture

Derived from what actually exists; no changes made. The intent: the active
tree reads top-down, history is archived (not deleted), artifacts stay out.

```text
rhan/                              # or keep repo root; canonical Gen-1 code
├── noesis_vision/                 #   unchanged (canonical package)
├── training/                      #   unchanged
├── evaluation/                    #   unchanged
├── scripts/                       #   Gen-1 tooling only (Gen-0 gate tools move)
├── tests/                         #   unchanged
├── configs/                       #   FoundationConfig presets + env templates
├── docs/                          #   UN-IGNORED as a whole, curated:
│   ├── REPOSITORY_MAP.md          #     this document
│   ├── ARCHITECTURE.md            #     updated per-generation
│   ├── rhan_next_roadmap.json     #     tracked state file
│   └── research/                  #     Gen-0 audit corpus (already here)
├── cloud/                         #   launchers, one subdir per generation
│   ├── gen1/{Kaggle_J1_FOUNDATION.py, colab_j1_foundation.py, shared_guards.py}
│   ├── gen0/{colab_notebook_noesis.py, Kaggle_NOESIS.py}
│   └── legacy/                    #     v10–v12, pipeline runners
├── archive/
│   ├── gen-1_cifar12/             #   phase1..5 trees for the 8/12-model study
│   ├── rhan_v1_v7/                #   model_rhan{,_v3..v7} + trainers
│   ├── stl10_scaleup/             #   unified/TDV/large + tier2 pointers
│   ├── gen0_gates/                #   scripts/sbr0_gate.py, eval_ais_v2_gate.py, …
│   ├── eval_scripts/              #   root eval_* sprawl + phase2 sweep variants
│   └── scratch/                   #   curated scratch with provenance headers
├── rhan_core/                     #   FROZEN Gen-0 package, kept importable
├── apps/cognitive_vision_lab/     #   extracted app (or its own repo)
└── data/, checkpoints/, runs/, report/   # gitignored working dirs (unchanged roles)
```

Why: the canonical line occupies the top level; every historical layer is
reachable under `archive/` with its generation label; weights/figures/reports
stay in gitignored working dirs + HF. What moves where is enumerated in the
per-directory refactor notes (§4) and candidates (§20). **Nothing moves
until the in-flight Gen-1 re-run completes** — the ladder's resume path
touches `training/`, `checkpoints/`, `runs/`, `report/`, and the HF repos.

---

# 22. Recommended New-Contributor Reading Order

Actual files, in order:

1. `README.md` — the project's arc and Gen "-1" results. **Caveat: its
   "Current" section is Gen-0; keep reading.**
2. `RHAN-history.md` — the v1–v7 lineage and why each version was tried.
3. `noesis_vision/RHAN_NXA/docs/00_README.md` → `01_What_Is_RHAN_NXA.md` →
   `16_Gen0_Evidence_And_Confounds.md` — what Gen-1 is and why Gen-0's
   confounds force the restart. (`MASTER_PLAN.md` when implementing.)
4. `docs/RHAN_NXA_ARCHITECTURE.md` — the current-generation architecture
   guide (created 2026-09-30; status vocabulary IMPLEMENTED/DEFERRED/
   REJECTED/PENDING DECISION/UNKNOWN/NOT YET IMPLEMENTED).
5. `docs/REPOSITORY_MAP.md` — this map. (For Gen-0 orientation,
   `docs/ARCHITECTURE.md` remains the rhan_core package tour; its status
   line is dated.)
6. Canonical model entry point: `noesis_vision/models/backbone.py` (then
   `beliefs/`, `predictive_coding/`, `uncertainty/`, `gaze/`).
7. Canonical training entry point:
   `training/train_generation1_foundation.py` (header → config →
   `run_phase`), with `training/adv_curriculum.py` and
   `training/stage_state_machine.py` alongside.
8. Canonical evaluation entry point: `evaluation/clean_and_robust.py`
   (+ `scripts/generate_full_sweep.py`).
9. Relevant tests: `tests/test_gradient_flow.py` (the house rule),
   `tests/test_adv_curriculum.py`, `tests/test_stage_state_machine.py`,
   `tests/test_checkpoint_never_silently_restarts.py`.
10. Experiment documentation: `report/GEN1_RESULTS_MASTER.md` (local copy;
    canonical on HF rolling) + `noesis_vision/RHAN_NXA/docs/24_Training_Phase_DAG.md`.
11. Cloud execution: `cloud_setup/Kaggle_J1_FOUNDATION.py` (read its header
    comments — they narrate the reset/pre-flight discipline).

Formerly-missing documents, now present (2026-09-30): `CONTRIBUTING.md`
(the contributor guide) and `docs/RHAN_NXA_ARCHITECTURE.md` (the Gen-1
architecture guide). Still missing: an index of `scratch/`.

---

# 23. Executive Summary

## What is canonical?
Branch `feature/rhan-next`; **Gen-1 = RHAN-NXA**: package `noesis_vision/`,
trainer `training/train_generation1_foundation.py` (six-phase ladder,
TRADES/PGD curriculum now default), evaluation `evaluation/clean_and_robust.py`
via `scripts/generate_full_sweep.py`, config = `FoundationConfig` +
`RHANNXAConfig` (+ per-run provenance manifests), cloud = 
`cloud_setup/Kaggle_J1_FOUNDATION.py`, artifacts = HF
(`rhan-nxa-checkpoints[-rolling]`). Verified from code + manifests + commit
log; conflicts documented (README is one generation stale).

## What is legacy?
`phase1_training/` (all pre-Gen-1 models/trainers; 103 files),
`phase2_attacks/` sweeps, `phase3_human_study/` (static data),
`phase4_analysis/`, `phase5_sdt/`, `tier1/`, `config/*.yaml`, the
CIFAR/STL-10 branch lattice, root-level `eval_*`/`demo`/`inspect` scripts.
All preserved; several are frozen-but-load-bearing (`train_rhan_next.py`,
`eval_rhan.py`, `model_rhan_v12.py`).

## What is experimental?
`rhan_core/` (Gen-0 — frozen reference, confound-documented),
`scratch/prototype/` (handbook rendering), `cognitive_vision_lab/`
(side app), the SBR arms (scaffolded, results confound-qualified),
`data_generation/` synthetic pipelines.

## What is confusing?
~~README generation drift~~ (fixed 2026-09-30); two `stage_state_machine.py`;
two `consistency_assert.py`; two `vector_belief.py`; curriculum constants
duplicated across generations; four roadmap-JSON copies; root eval sprawl;
63-script scratch dir; gitignored-but-tracked `report/` files;
tracked binary weights and Zone.Identifier files; Colab twin lacking the
Kaggle guards; cloud_setup `__pycache__` ghosts of renamed notebooks.

## What should be refactored first?
(P0) 1. ~~README current-generation banner~~ **DONE 2026-09-30**;
2. bring the Colab launcher to guard parity (or factor shared guards).
Then (P1) state-machine rename, eval-sprawl archive, roadmap precedence,
curriculum parity test — only after the pending adversarial re-run has
been launched and finished, since resume touches every working dir.

## What must NOT be touched because it preserves research provenance?
`phase3_human_study/` (irreplaceable human data); `phase1_training/` +
frozen Gen-0 entrypoints (reproduce Gen-0 numbers); `rhan_core/` (Gen-0
reference); `noesis_vision/RHAN_NXA/docs/` (the plan-as-record, incl. the
confound disclosure); `report/` + HF archives (pure-CE control arm,
`archive/gen1_pure_ce_<stamp>/`); rejected arms (CBM, Self-Align,
Feature-Scatter, v4/v6 regressions) as documented negatives;
`report/mock_true_empirical_metrics_HISTORICAL.json` (labeled mock,
historical); `checkpoints_tier2/` tracked binaries (era evidence).

## Biggest architectural risks
1. **Guard asymmetry between launchers** — an unguarded Colab run could
   contaminate the corrected ladder's provenance.
2. **Silent-inheritance guard gives bitwise honesty, not scientific
   difference** — `gen1_core` ≡ `ais_v2_swap` forwards; the ladder cannot
   distinguish them by construction; results must be read at the phase
   level, not the last-phase level.
3. **State multiplicity** — roadmap in 4 places, manifests keyed by
   filename conventions; a stale copy previously risked mis-blessing old
   configs (why the reset deletes `production_launch_manifest.json`).
4. **Ephemeral cloud state** — `/kaggle/tmp` wipes between sessions; data
   re-bootstrap (~19 GB) is required per fresh session; resume correctness
   leans on HF discipline (tests pin this, keep it that way).
5. ~~**Documentation drift** — three generations of "current" claims in
   README/ARCHITECTURE/docs; without the banner fix, onboarding starts in
   the wrong generation.~~ **MITIGATED 2026-09-30**: README banner +
   `RHAN_NXA_ARCHITECTURE.md` + `CONTRIBUTING.md` (residual risk: the docs
   must now be kept in sync with the code, as always).

## Biggest opportunities
1. A single `archive/` layout would cut the tracked-tree confusion in half
   without deleting a byte of provenance.
2. Launcher guard-sharing turns the re-run prep into reusable
   infrastructure for J2+ (steps 7+).
3. The Gen-1 docs discipline ("specified ≠ implemented", Source-decision
   pointers) is a strength — extending it to a Gen-1 repository README
   would finish the onboarding path.
4. The absent metric families (d′, uncertainty readouts, prediction-error,
   gaze telemetry) are the highest-value *scientific* additions once the
   corrected ladder lands — the harness and manifest plumbing for them
   already exists.

---

*End of map. Originally created 2026-09-30 (commit `74d2e19`); §2-5, §3,
§20-P2-2, §22, and this summary refreshed the same day (commit
`7341cd6` and follow-ups) to reflect the documentation refactor. The
§20 P0-2/P1 items and §21 target layout remain OPEN and are the input
for the next, code-touching refactor stage.*

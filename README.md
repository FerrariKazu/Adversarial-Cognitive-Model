# Adversarial Cognition Divergence (ACD)

**Historical foundation [Completed]:** a **7-model + human** psychophysics study of adversarial robustness on CIFAR-10 — BagNet-33 / ResNet-18 / EfficientNet-B0 / Shape-ResNet-50 / CORnet-S / ViT-Small / CLIP ViT-B/32, attacked with FGSM / PGD / C&W, analyzed with Signal Detection Theory (d′), plus human psychophysics (n = 18, 1,800 trials). Legacy result tables cover 13/13 CIFAR-10/STL-10 systems.

**Current research line [Active]:** **RHAN-NXA (Generation 1)** — active recurrent perceptual intelligence: perception as belief-state investigation rather than a single look.

> Does adversarial robustness scale with global visual processing —
> and is it determined by architecture, training objective, or recurrence?

This README is the **repository entry point**, not an architecture paper. Every claim below carries an explicit status label — **Completed / Implemented / Under validation / Experimental / Planned / Research proposal** — and no future mechanism (Gen-3, persistent world modeling, JEPA-style learning, VLM, VLA) is ever presented as implemented. The detailed truth lives in the code; when this file and the code disagree, the code wins and the discrepancy is recorded in *Documented gaps*.

---

## Current research line — RHAN-NXA (Generation 1)  [Active]

RHAN-NXA is a speculative, plan-first architecture: a recurrent, belief-state "perception-as-investigation" system that examines an image over several deliberate looks, maintains an explicit updatable belief state, and reads classification out of it. Its foundation ladder is executing training — but **no Gen-1 mechanism has a matched-validated number on a real dataset yet** (see *Status* below).

Where the Gen-1 material lives:

- `noesis_vision/RHAN_NXA/` — ★ the canonical Gen-1 plan: `docs/` chapters 00–30 + `Proposed_Plan.md` (the `MASTER_PLAN.md` root file is tracked only on `feature/rhan-next` — see gaps).
- `gen2_foundation/` — ★ Gen-2 foundation research slice (committed on `main`; 52/52 unit tests pass) + `EXPERIMENT_REGISTRY.md` (campaign registry).
- `cloud/gen1/colab_j1_foundation_gate.py` — ★ J1 Foundation-Gate harness (TRADES vs matched-CE A/B on Colab T4); gate OPEN, not yet run.
- `report/generation1_foundation_roadmap.json` — the six-phase ladder's single source of truth (currently executing).
- `training/`, `evaluation/`, the Gen-1 `scripts/` tooling, and the `noesis_vision/` package source — the Gen-1 **pipeline code lives on the carrier branches** `feature/rhan-next` / `stage2/nxa-pipeline-refactor`, **not on `main`** (on `main`, `training/` and `evaluation/` contain only `__pycache__` remnants — see gaps).
- `archive/` — frozen/historical generations (read-only; working tree only).

**Status of the headline claims (do not soften):**

| Component | Status |
|---|---|
| AIS-v2 gaze score ↔ error-reduction correlation (r ≈ 0.706) | Existed as a **smoke-gate** result (`noesis_vision/gaze/ais_v2_policy.py`, carrier branches) — mechanism-level and confounded by legacy SBR; **not** a Gen-1 proof |
| "AIS-v1 is genuine information-gain" | **Not a Gen-1 claim** (Gen-0 halting-only result; see the reprinted Gen-0 section) |
| Gen-1: isolated discovery of any mechanism (AIS-v2, belief-HPC) | **UNKNOWN** — Gen-1 arms were contaminated by legacy SBR; evidence is explicitly recorded as such |
| Gen-1 real-data numbers | **None matched-validated yet** — the foundation ladder is executing (ImageNet-100 + CUDA) |

## Generation lineage

1. **ACD v1–v4 (historical) [Completed]** — CIFAR-10 adversarial-robustness study + human psychophysics; results reprinted in the *Historical results* section.
2. **Gen-0 / RHAN-Next (`rhan_core/`) [frozen reference]** — Active Inference Suite + Hierarchical Predictive Coding on STL-10; Stages 0–2 validated, Stage 3 pending (`docs/ARCHITECTURE.md`); branch `feature/rhan-next`.
3. **Gen-1 / RHAN-NXA (`noesis_vision/`) [Active — under validation]** — belief-state perception-as-investigation; six-phase foundation ladder on ImageNet-100.
4. **Gen-2 (`gen2_foundation/`) [Implemented — under validation]** — foundation mechanism slice; code + tests on `main`, real-data gate open.
5. **Gen-3 [Research proposal]** — hierarchical / persistent world perception. Not implemented; no code, no model class, no experiment.
6. **VLM / VLA [Planned — research proposal]** — vision-language(-action) integration as a *consumer* of the belief core. Not implemented.

### Six-phase foundation ladder (Gen-1)

| Phase | Adds | Gaze |
|---|---|---|
| 1 `backbone_only` | substrate + one fixed center fixation + classifier head | fixed center |
| 2 `recurrence_only` | T=4 fixed-schedule glimpse loop + tied refinement | fixed center |
| 3 `belief_no_f` | belief carrier with U_t; IDENTITY update | fixed center |
| 4 `belief_with_f` | predictor / precision / UpdateNet dynamics | fixed center |
| 5 `ais_v2_swap` | AIS-v2 gaze policy | AIS-v2 |
| 6 `gen1_core` | integrated system (S=None, L_stab diagnostic-only) | AIS-v2 |

Entrypoint (carrier branches): `training/train_generation1_foundation.py`; ladder state: `report/generation1_foundation_roadmap.json`.

### Branch layout (where things actually live)

| Branch | Contains |
|---|---|
| `main` | Historical ACD packages (`phase1_training/` … `phase5_sdt/`), frozen `rhan_core/` + `tests/`, `gen2_foundation/`, `cloud/gen1/` (Colab J1 dispatcher + gate harness), `config/`, tracked docs, generated artifacts |
| `feature/rhan-next` | Gen-0 RHAN-Next + Gen-1 pipeline sources (`training/`, `evaluation/`, Gen-1 `scripts/`, `noesis_vision/` package + `MASTER_PLAN.md`) |
| `stage2/nxa-pipeline-refactor` (local) | Carrier for the Gen-1 production run: pipeline + the remaining `cloud/gen1/` launchers (`Kaggle_J1_FOUNDATION.py`, `run_j1_local.sh`, `run_j1_supervised.sh`) |
| `diagnosis/nxa-forensic-2026-10-03` (local) | Forensic snapshot of the cancelled adversarial run |

`main` and `feature/rhan-next` have diverged (6 / 74 commits); commands that need `training/` or `evaluation/` must run from a carrier branch.

---

## Scientific philosophy

- **Perception is investigation, not a lookup.** An image is examined over several deliberate looks; an explicit belief state is maintained and updated; classification is a *readout* of that state (`noesis_vision/RHAN_NXA/docs/01_What_Is_RHAN_NXA.md`, `noesis_vision/RHAN_NXA/docs/03_Perception_As_Investigation.md`, `noesis_vision/RHAN_NXA/docs/04_Belief_State.md`).
- **Robustness is a property of the mechanism, not a wrapper.** The original ACD finding is the premise of everything after it: adversarial robustness is determined by architecture, training objective, and recurrence — it cannot be bolted on (results below; `docs/ACD_v4.pdf`, `Paper/`).
- **Uncertainty and precision are first-class citizens.** The system carries an uncertainty state and modulates processing by precision rather than emitting a single confident forward pass (`noesis_vision/RHAN_NXA/docs/07_Uncertainty.md`, `noesis_vision/RHAN_NXA/docs/09_Recurrence.md`, `noesis_vision/RHAN_NXA/docs/10_AIS_v2.md`).
- **Biology is a constraint, not decoration.** Recurrence, active inference (gaze as information foraging), and predictive coding are structural commitments — not marketing names for attention blocks.
- **No claim without a matched number.** Mechanisms are gated behind experiment IDs (`gen2_foundation/flags.py` — `GATED_FLAGS` / `REJECTED_OUTRIGHT` raise at construction if a mechanism is flipped without its registered ID); deferred items stay deferred until real numbers exist (`gen2_foundation/AUDIT.md` §5).
- **This is deliberately *not* a generic VLM with an RHAN module attached.** The belief-state perceptual core is the research object; any future VLM/VLA would *consume* it, not define it (scope: `noesis_vision/RHAN_NXA/docs/20_Scope_Boundaries.md`).

---

## Gen-2 foundation — `gen2_foundation/`  [Implemented — under validation]

A separate research slice on `main`, deliberately small and mechanism-by-mechanism:

| Mechanism (approx.) | Files |
|---|---|
| Configuration / experiment gating | `gen2_foundation/flags.py` (`RHANGen2Config`) |
| Full-image gist encoding + fusion | `gen2_foundation/gist.py` |
| Precision mechanisms | `gen2_foundation/precision.py` |
| Observed + prediction-error fusion (UpdateNet-v2) | `gen2_foundation/update_net_v2.py` |
| EMA target encoder | `gen2_foundation/ema.py` |
| Optimizer / training recipe | `gen2_foundation/recipe.py` |
| EOT robustness infrastructure | `gen2_foundation/eot.py` |
| Unit tests — **52/52 passing** (verified locally) | `gen2_foundation/tests/` (7 modules) |
| Audit record | `gen2_foundation/AUDIT.md` |
| Campaign registry | `gen2_foundation/EXPERIMENT_REGISTRY.md` (K1–K9 + J1 order, per-arm status/decision) |

**Validation state (do not inflate):** the mechanisms are implemented as code and all 52 unit tests pass; the **real ImageNet-100 foundation measurement has not been produced** — the gate is open, and deferred items (G2-K10 factor-disagreement, G2-K11 calibration) remain deferred until the foundation slice reports real numbers (`gen2_foundation/AUDIT.md` §5). Nothing in this slice is part of the Gen-1 canonical pipeline.

---

## The empirical foundation — ACD v4.0  [Completed]

The original **Adversarial Cognition Divergence v4.0** program is the completed empirical work this project grew out of. It is preserved here and must NOT be erased:

- **7-model + human CIFAR-10 study**, FGSM / PGD / C&W attacks, Signal Detection Theory (d′), reference models BagNet-33 / ResNet-18 / EfficientNet-B0 / Shape-ResNet-50 / CORnet-S / ViT-Small / CLIP ViT-B/32, the RHAN model family, plus human psychophysics (n = 18, 1,800 trials).
- Completed results live in the **Historical results** section at the bottom of this README (full tables reprinted verbatim), in `RHAN-history.md`, and in the raw data (`phase3_human_study/data/responses_mapped.csv`, `phase5_sdt/results/`).
- Historical commands are archived — see *Reproduce → Historical ACD v4.0 commands* and the reprinted *Reproduce Results* block in the Historical section.

---

## Gen-1 family — plan (not code on `main`) vs frozen Gen-0 reference

The Gen-1 **architecture plan** is a research proposal: `noesis_vision/RHAN_NXA/` — `MASTER_PLAN.md` (tracked on `feature/rhan-next`) plus `docs/` chapters 00–30 and `Proposed_Plan.md` (present as working-tree files on `main`; see gaps). It is what RHAN-NXA *specifies*. The Gen-1 *pipeline code* (`training/`, `evaluation/`) lives on the carrier branches; `main` carries the plan documents, `gen2_foundation/`, and the historical study.

The frozen Gen-0 reference is `rhan_core/` (present on `main`, developed on `feature/rhan-next`): `rhan_core/model.py` (RHANNext, 76.7M params), `rhan_core/config/pillar_config.py`, `rhan_core/gaze/`, `rhan_core/predictive_coding/`, `rhan_core/precision/`, `rhan_core/beliefs/`, `rhan_core/world_model/` (null scaffold), `rhan_core/ablation/`, `rhan_core/lens/`. It is a frozen reference for comparison, not the active training line.

The Gen-1 architecture is a **spec, not an implementation**: no Gen-1 mechanism has a matched-validated number on a real dataset yet. Anything you see labeled "implemented" for Gen-1 must be checked against this section first.

---

## Gen-3 — not implemented, plan only  [Research proposal]

Gen-3 is a **research proposal**: hierarchical world perception / persistent belief / memory / counterfactual perception. There is no Gen-3 code, no Gen-3 model class, and no Gen-3 experiment in this repository. Do not cite any Gen-3 number.

## VLM / VLA — future direction only  [Planned — research proposal]

Vision-language and vision-language-action integration is a **future direction only**: no VLM, no VLA, no language-instruction pipeline, and no language training data exist in this repository. The intended shape is that a future VLM/VLA would consume the RHAN-NXA belief core as its perception substrate — the project defines itself *against* being a generic VLM with a module attached. Same for JEPA-style world modeling: a research direction discussed in the literature notes (`docs/research/`, `noesis_vision/RHAN_NXA/docs/29_Literature_Classification.md`), never implemented here. Do not cite any VLM/VLA/JEPA number; there is none.

---

## Roadmap (staged)

| Stage | Goal | Status |
|---|---|---|
| **Gen-1** | Six-phase foundation ladder on ImageNet-100 → matched validation of the core mechanisms (belief carrier, UpdateNet dynamics, AIS-v2 gaze) | **Under validation** — pure-CE arm completed all six phases (2026-09-25→26, archived); the earlier adversarial run was cancelled 2026-10-03 after forensics (`docs/CANCELLATION_NOTICE_2026-10-03.md`, `docs/FORENSIC_REPORT_NXA_GENERATION1.md`); the adversarial re-run launched 2026-10-06 with `backbone_only` running (`report/generation1_foundation_roadmap.json`) |
| **Gen-2** | Foundation mechanism slice → real ImageNet-100 foundation measurement → unlock deferred items (K10 factor-disagreement, K11 calibration) | **Implemented** (code + 52/52 tests); real-data gate **open** — no number yet |
| **Gen-3** | Hierarchical / persistent world perception (world-state, object–scene, counterfactual, reasoning) | **Research proposal** — not implemented |
| **Future** | VLM / VLA as consumers of the belief core; JEPA-style world-model research | **Planned** — research direction only, not implemented |
---

## Installation & usage

```bash
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Environment facts (verified against this tree — not a lockfile):

- Python **3.10.12** baseline. The research environment ran **PyTorch 2.9.1+cu128** with CUDA; root `requirements.txt` is the Gen-0-era list and pins the wheel index to `cu124`.
- Packages: torch / torchvision / torchaudio, `torchattacks==3.5.1`, `grad-cam>=1.4.0`, numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, tqdm, tensorboard, Pillow, openpyxl, PyYAML, ftfy, regex, `timm>=0.9.0`, bagnets / cornet / CLIP (git URLs), `datasets==4.7.0`.
- `config/train_config.yaml` sets `num_workers=4`, which **deadlocks** the ImageNet-100 loader in local runs — use `--num-workers 0` or `1` for smoke runs.
- `FoundationConfig` has **no** `device=` kwarg; pass `--device` on the CLI.
- Real-data runs (ImageNet-100 download, CUDA, HF sync) are environment-specific: they run on a GPU host **from a carrier branch**, not from `main`.

---

## Reproduce

### Gen-1 foundation ladder  [environment-specific: carrier branch + GPU + ImageNet-100]

```bash
# Requires feature/rhan-next or stage2/nxa-pipeline-refactor — training/ does not exist on main
python3 training/train_generation1_foundation.py --smoke                 # synthetic-loader smoke, no data/GPU needed
python3 training/train_generation1_foundation.py --phase all             # full six-phase ladder
python3 training/train_generation1_foundation.py --phase backbone_only   # one phase; also: recurrence_only,
#   belief_no_f, belief_with_f, ais_v2_swap, gen1_core
```

Flags verified against the entrypoint: `--phase`, `--smoke`, `--clean-only`, `--device`, `--num-workers`, `--w-trades`, `--no-hf`, `--hf-token`, `--force-fresh`. Ladder orchestration reads `report/generation1_foundation_roadmap.json`.

Clean/robust numbers are produced **inside** the ladder by the Agent-I harness `evaluation/clean_and_robust.py` (`run_clean_and_robust(...)` — a library API, **not** a CLI; there is no `python3 evaluation/clean_and_robust.py ...` command). Per-phase provenance lands in `report/foundation_<phase>_eval/` and `runs/foundation_<phase>/manifest.json`.

### Frozen Gen-0 eval entrypoint (runs on `main`)  [environment-specific: GPU + checkpoints]

```bash
# Structural self-test against the checked-in reference (no GPU needed for the check itself)
python3 phase2_attacks/eval_rhan.py --self-test

# Matched protocol: norm-space PGD (Finding-17 convention), seed floor >= 5 enforced by the facade
python3 phase2_attacks/eval_rhan.py --n-samples 300 --seeds 41 42 43 44 45 \
  --pgd-steps 50 --eps-norm-space --eps-list 0.0 0.094 --batch-size 64 \
  --baseline-label trades_large_baseline \
  --ckpt-specs trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large
```

`phase2_attacks/eval_rhan.py` is the frozen facade; the underlying parser and protocol live in `phase2_attacks/eval_full_epsilon_sweep.py`.

### Cloud / Kaggle entry points

- Gen-1: `cloud/gen1/colab_j1_foundation.py` (six-phase ladder dispatcher) and `cloud/gen1/colab_j1_foundation_gate.py` (J1 A/B Foundation-Gate harness) — **on `main`**; `cloud/gen1/Kaggle_J1_FOUNDATION.py`, `cloud/gen1/run_j1_local.sh`, `cloud/gen1/run_j1_supervised.sh` — **on `stage2/nxa-pipeline-refactor` only** (absent from `main`).
- Gen-0 legacy [archived]: `cloud_setup/kaggle_notebook.py`, `cloud_setup/Kaggle_NOESIS.py`, `cloud_setup/colab_notebook_noesis.py` — on `main`.

### Historical ACD v4.0 commands  [archived — not current entry points]

- Attack generation: `phase2_attacks/generate_adv_all_models.py` (+ `fgsm.py`, `pgd.py`, `cw.py`).
- Figures: `phase4_analysis/generate_hero_figures.py`, `phase4_analysis/generate_appendix_figures.py` (the original README cited `phase4_analysis/generate_all_figures.py`, which **no longer exists** — see gaps).
- SDT: `phase5_sdt/sdt_analysis.py`.
- Legacy training: `phase1_training/train_rhan_unified.py --phase all`, `phase1_training/train_rhan_stl10_tdv.py --phase trades`, `torchrun --nproc_per_node=2 phase1_training/train_rhan_large_pseudolabel.py ...`.
- The complete original *Setup* and *Reproduce Results* blocks are reprinted in the *Historical results* section below, labelled archived.

---

## Research reproducibility

Every published number must carry: configuration, seed, dataset fingerprint, code revision, architecture revision, recipe, evaluation protocol, checkpoint, manifest, result summary. Where each one lives on this tree:

| Element | How it is recorded |
|---|---|
| Experiment configuration | `report/generation1_foundation_roadmap.json` (Gen-1 ladder state) + per-phase `runs/foundation_<phase>/manifest.json` (flake-checksummed). Gen-0: `docs/rhan_next_roadmap.json` + `config/` |
| Seed | Seed policy 41 (main); eval seeds 41–48 (46–48 extension at ε=0.094); recorded per manifest |
| Dataset fingerprint | ImageNet-100, HF `clane9/imagenet-100` pinned revision `0519dc2f…`, listing sha256 `0b0677…c6db09`, recorded in `runs/production_launch_manifest.json` |
| Code revision | `runs/production_launch_manifest.json` (`git_sha`, `config_sha256`); trainer commit `cf6ce8a` in per-phase manifests |
| Architecture revision | Checkpoint-embedded config + `code_commit` (extracted in `report/GEN1_RESULTS_MASTER.md`) |
| Training recipe | Adversarial recipe `gen1-adv-curriculum-v1`: TRADES/PGD-4, ε 0.031→0.062→0.094, β 2.0→2.5, `w_trades=0.55` (`training/adv_curriculum.py`, carrier branch; recorded in `docs/CANCELLATION_NOTICE_2026-10-03.md`) |
| Evaluation protocol | `evaluation/clean_and_robust.py` (`run_clean_and_robust`: norm-space eps, ≥5-seed floor, summary asserted against per-seed CSV) + `scripts/generate_full_sweep.py`. Gen-0: `phase2_attacks/eval_rhan.py` / `eval_full_epsilon_sweep.py` |
| Checkpoint | `checkpoints/foundation_<phase>_{best,rolling}.pth`; mirrored to HF `FerrariKazu/rhan-nxa-checkpoints` (best) and `FerrariKazu/rhan-nxa-checkpoints-rolling` |
| Manifest | `runs/foundation_<phase>/manifest.json` (frozen config hash, per-phase provenance) |
| Result summary | `report/foundation_<phase>_result.json`, `report/foundation_<phase>_eval/summary_table.csv`, `report/foundation_<phase>_compactness.json`, and the extract `report/GEN1_RESULTS_MASTER.md` |

Freeze record: `runs/production_launch_manifest.json` — code `a4444dbde054`, branch `feature/rhan-next`, frozen 2026-09-25.

---

## Repository map

Generated from the actual `main` working tree and cross-checked against `git ls-files`; every path below exists unless the comment says *carrier branch* / *working tree only*. `#` comments describe each entry. Artifact-heavy stores are summarized, not itemized.

```text
.
├── README.md                     # this file — repository entry point (documentation only)
├── requirements.txt              # Gen-0-era dependency list (see Installation & usage)
├── ROADMAP.md                    # Gen-0-era roadmap essay (historical)
├── RHAN-history.md               # ACD/RHAN history narrative (historical)
├── RHANarch.md  RHANfuture.md  RHANv11.md     # architecture / evolution / v11 essays (historical)
├── FINDINGS.md  COLLABORATING.md  IMPORTANT.md  deep-research-report.md   # project essays (historical)
├── RHANarch.pdf  rhan_mathematical_report1.pdf  rhan_epoch_analysis_report.pdf   # compiled reports
├── T4x2.ipynb                    # dual-T4 cloud training notebook (historical)
├── app.py  demo.py               # small demo / dashboard entry scripts
├── eval_pgd_*.py  eval_aa_*.py  eval_quick_perclass*.py  eval_stl10.py  run_eval_stl10.py
│                                 # loose one-off evaluation scripts at root (historical)
├── bench_pgd.py  concept_ablation.py  inspect_*.py  check_parquet.py  simple_inspect.py
│   test_ckpt_load.py  test_load.py  upload_pseudolabel.py   # one-off diagnostics / utilities
├── response_template.csv         # human-study response template
├── confusion_matrix*.png  rhan_flowchart.png  resnet_eval.txt  vit_eval.txt  train_norecon.log
│                                 # committed run outputs (generated artifacts)
├── *:Zone.Identifier             # Windows download-metadata residue (harmless, left as-is)
│
├── phase1_training/              # HISTORICAL model zoo (103 files): model_*.py (ResNet, ViT-Small,
│                                 #   EfficientNet, Shape-ResNet, BagNet, CORnet, CLIP, RHAN-v1..v12),
│                                 #   train_*.py, eval_*.py, dataset*.py, pretrain_*.py
├── phase2_attacks/               # HISTORICAL attacks + frozen eval (29 files): fgsm.py, pgd.py, cw.py,
│                                 #   generate_adv_all_models.py, eval_rhan.py (frozen Gen-0 facade),
│                                 #   eval_full_epsilon_sweep.py (matched protocol parser)
├── phase3_human_study/           # human psychophysics (n=18): data/responses_mapped.csv, manifest.csv,
│                                 #   anonymize.py, map_responses.py, stimuli_export.py
├── phase4_analysis/              # interpretability & divergence analysis (25 files): gradcam.py,
│                                 #   confusion_matrices.py, divergence_curves.py, figures/
├── phase5_sdt/                   # signal-detection theory: sdt_analysis.py, sdt_core.py,
│                                 #   results/sdt_results*.csv, figures/
│
├── rhan_core/                    # FROZEN Gen-0 RHAN-Next reference: model.py (RHANNext, 76.7M),
│                                 #   config/pillar_config.py, gaze/, predictive_coding/, precision/,
│                                 #   beliefs/, world_model/ (null scaffold), ablation/, lens/
├── tests/                        # Gen-0 test suite: conftest.py + 17 pytest modules
│                                 #   (scaffold import, config back-compat, HPC gates, resume guard, LENS …)
├── config/                       # train_config.yaml, train_config_vit.yaml, attack_config.yaml
│
├── gen2_foundation/              # ★ Gen-2 foundation slice (committed on main): flags.py, gist.py,
│   ├── tests/                    #   precision.py, update_net_v2.py, ema.py, recipe.py, eot.py;
│   │                             #   7 test modules -> 52 tests, all passing
│   ├── AUDIT.md                  # implementation / audit record
│   └── EXPERIMENT_REGISTRY.md    # ★ Gen-2 campaign registry (K1–K9 + J1; frozen order, per-arm status)
│
├── noesis_vision/                # ★ Gen-1 RHAN-NXA plan home
│   └── RHAN_NXA/docs/            #   chapters 00–30 + Proposed_Plan.md (working tree only — gitignored);
│                                 #   package source (core/, gaze/, models/, …) and MASTER_PLAN.md
│                                 #   live on carrier branches; on main only __pycache__ remnants
├── training/                     # Gen-1 trainer — CARRIER BRANCHES ONLY: train_generation1_foundation.py,
│                                 #   adv_curriculum.py, stage_state_machine.py; on main: empty dir
├── evaluation/                   # Gen-1 eval harness — CARRIER BRANCHES ONLY: clean_and_robust.py,
│                                 #   imagenet100_loader.py, shape_texture_bias.py, …; on main: empty dir
├── scripts/                      # on main: generate_full_sweep.py, merge_stage1_seed_extension.py,
│                                 #   build_noesis_pdf.py; carrier branches add: prepare_imagenet100.py,
│                                 #   sbr0_gate.py, eval_ais_v2_gate.py, stage_state_machine.py,
│                                 #   verify_run_complete.py, freeze_run_manifest.py
│
├── cloud/                        # Gen-1 Colab J1 entry points (committed on main): colab_j1_foundation.py
│                                 #   (six-phase ladder dispatcher), colab_j1_foundation_gate.py (A/B gate harness);
│                                 #   Kaggle/run_j1_* scripts remain stage2-only
├── cloud_setup/                  # Gen-0-era cloud launchers (22 files): Kaggle_NOESIS.py,
│                                 #   colab_notebook_noesis.py, kaggle_*/colab_* pipelines, lightning_setup.py
├── data_generation/              # synthetic STL-10 generation + HF upload (5 scripts)
├── cognitive_vision_lab/         # independent Streamlit benchmarking platform: backend/, pages/ (14),
│                                 #   components/, tests/ (7), Dockerfile, docker-compose.yml
├── dashboards/                   # lens_app.py — LENS introspection dashboard
├── competition/                  # benchmark comparison: evaluate_comparison.py, generate_heatmaps.py, output/
│
├── docs/                         # tracked: ARCHITECTURE.md (Gen-0 plan), rhan_next_roadmap.json (Gen-0 state),
│   ├── historical/               #   stage3_preregistration.md, stage3_environment.json, ACD_v4.pdf,
│   ├── research/                 #   ACD_paper_v1.tex, ACD_Project_Documentation_v3.pdf, Manual Documentation.md;
│   ├── rhan_nxa/                 #   working tree only (gitignored): REPOSITORY_MAP.md (detailed tree map),
│   └── ...                       #   RHAN_NXA_ARCHITECTURE.md, CANCELLATION_NOTICE_2026-10-03.md,
│                                 #   FORENSIC_REPORT_NXA_GENERATION1.md, repository_reorganization.md,
│                                 #   NOESIS_FOUNDATION.md, historical/, research/, rhan_nxa/
├── Paper/                        # ACD paper LaTeX + PDFs (v1, v2) + figures/
├── RHANv12/                      # RHAN-v12 report LaTeX + scripts/
├── tier1/                        # scientific validation report (LaTeX/PDF) + results/ tables
├── rhan_math/                    # mathematical proof reports (phase1..5_proofs.md) + figures + compilers
│
├── report/                       # results & run reports. tracked (16): IEEEtran.cls, assets/, empirical_report.json,
│   │                             #   final_sweep_results_stl10.json, stage2_hpc_run_log.md, rhan_v10_scientific_report.*;
│   │                             #   working tree only (gitignored): GEN1_RESULTS_MASTER.md,
│   │                             #   generation1_foundation_roadmap.json, foundation_*_{result,compactness}.json,
│   │                             #   foundation_*_eval/ (per-seed CSVs + provenance), sweep_*/ logs, lens_e1_analysis/
│   ├── GEN1_RESULTS_MASTER.md    #   extracted Gen-1 results (working tree only)
│   └── generation1_foundation_roadmap.json   # ladder state (working tree only)
├── checkpoints/                  # model weights: foundation_<phase>_{best,rolling}.pth (six phases) +
│                                 #   legacy STL-10/CIFAR checkpoints [mostly working tree; 4 tracked]
├── checkpoints_tier2/            # tier-2 checkpoint slot — currently only *:Zone.Identifier stubs
├── runs/                         # per-run manifests/logs: foundation_*/manifest.json,
│                                 #   production_launch_manifest.json [working tree only — gitignored]
├── logs/  sweep_results/         # run logs + sweep CSVs [generated stores]
├── figures/  figures_v2/  figures_v3/   # generated figure sets (102 / 136 / 161 files) [generated stores]
├── data/                         # datasets: imagenet100/, stl10/, stl10_binary/ (131k files) [gitignored, not source]
├── scratch/                      # 58 ad-hoc debug / diagnostic scripts (historical)
├── utils/                        # metrics.py — shared metrics
│
├── archive/                      # frozen generations (read-only; working tree only — not committed)
│   ├── gen0/                     #   legacy Gen-0 checkpoints (checkpoints-legacy/)
│   ├── gen3-human/               #   snapshot of the human-study package
│   ├── gen4-analysis/            #   snapshot of the analysis package
│   ├── gen5-sdt/                 #   snapshot of the SDT package
│   ├── historical-report/        #   Paper/ + competition/ snapshots
│   ├── legacy-evals/             #   legacy evaluation reports (PDFs)
│   ├── pkg-rhan-math/            #   rhan_math package snapshot
│   └── pkg-tier1/                #   tier1 package snapshot
│
└── .agent/  .claude/  .vscode/  .freebuff/  .venv/  .verify_samples/   # local tooling / session state (not research source)
```
---

## Status — what is real, what is not

| Component | Status |
|---|---|
| Original ACD v4.0 (7-model + human CIFAR-10, FGSM/PGD/C&W, SDT) | **Completed** — historical foundation; full results reprinted below |
| Gen-0 RHAN-Next (`rhan_core/`, AIS + HPC, three-stage protocol) | **Frozen reference** — Stages 0–2 validated, Stage 3 pending (`docs/ARCHITECTURE.md`) |
| Gen-1 foundation ladder | **Under validation** — pure-CE arm completed all six phases (2026-09-25→26); adversarial re-run launched 2026-10-06, `backbone_only` running; **no matched-validated mechanism number yet** |
| Gen-1: isolated discovery of AIS-v2 / belief-HPC | **UNKNOWN** — Gen-1 arms contaminated by legacy SBR; evidence recorded as such |
| Gen-2 foundation mechanisms | **Implemented** as code; **52/52 unit tests pass**; real ImageNet-100 foundation measurement **not yet produced** — validation gate open |
| J1 Foundation Gate (TRADES vs matched CE) | **Harness-ready, not run** — `cloud/gen1/colab_j1_foundation_gate.py` committed (self-test 12/12); **no T4 results exist — gate OPEN** |
| Gen-3 / persistent world modeling | **Research proposal** — not implemented |
| JEPA-style world modeling | **Research proposal / literature direction** — not implemented |
| VLM / VLA | **Planned — research proposal** — not implemented |

---

## Historical results — ACD v4.0 & legacy generations  [archived — reprinted verbatim]

Everything below this line is **reprinted from the pre-reconstruction README** (including its Gen-0 RHAN-Next section). It records *completed historical work*; numbers are as published at that time and are **not** current results.

Notes on the reprint:

- Two generation-label corrections were applied so history is not presented as current: the RHAN-Next section header now says **frozen Gen-0 reference** instead of "(Current)", and its opening sentence says "The Gen-0 generation" instead of "The latest generation".
- Code-fence comment lines were moved inside their fences so the Markdown renders correctly; content is otherwise verbatim.
- Commands here are **archived** (`phase*_training/`, `phase*_attacks/`, `cloud_setup/`, `archive/`). At least one cited script (`phase4_analysis/generate_all_figures.py`) and one cited directory (`phase3_human_study/stimuli/`) no longer exist — see *Documented gaps*.
- The `## Repository Structure` block below is the *original* README's tree, kept as a historical record; the current tree is in *Repository map* above.

## Gen-0 RHAN-Next: Active Inference + Hierarchical Predictive Coding (frozen reference; reprinted)

The Gen-0 generation (**RHAN-Next**, branch `feature/rhan-next`; frozen since Gen-1 began) composes two biologically-inspired pillars into a single architecture:

- **AIS (Active Inference Suite)**: Entropy-gated halting, information-gain gaze policy, precision-modulated reconstruction — reformulates visual perception as a temporal control loop over discrete foraging steps.
- **HPC (Hierarchical Predictive Coding)**: Single-level edge-map prediction error loss — the brain's own unsupervised learning signal, regularizing the backbone without labels.

### Three-Stage Experimental Protocol

| Stage | Config | Clean Acc | PGD-50 @ ε=0.094 | PGD-100 @ ε=0.094 | Verdict |
|---|---|---|---|---|---|
| **1** | AIS-v1 (halting-only) | 49.40±3.47% | 32.21±2.74% | 32.17±2.65% | ✅ Genuine robustness, +8.5pp vs baseline (not significant at 8-seed) |
| **2** | HPC-only | 55.20±3.67% | 27.87±1.95% | 27.40±2.22% | ✅ Genuine robustness, +7.5pp vs baseline (significant) |
| **3** | **D = AIS + HPC** | **56.16%** (1 seed) | *Running* | *Pending* | 8-seed eval in progress |
| ref | TRADES-Large baseline | 53.47±2.87% | 20.40±1.21% | 19.87±1.07% | Baseline |

**Key findings so far:**
- Both AIS and HPC individually produce **genuine** adversarial robustness (PGD-50→100 gaps well below masking threshold)
- HPC adds **+5.8 pp clean accuracy** over AIS alone, at the cost of some robustness
- The combined model D achieves **56.16% clean** (best single seed), exceeding both components
- All evals use the Finding-17 matched norm-space convention, 5–8 seeds, n=300/seed

### Architecture

```
RHANNext (76.7M params)
  ├── RHANv12 backbone (frozen subclass)
  │     ├── Ventral stream (Transformer encoder)
  │     ├── Dorsal stream (Transformer encoder)
  │     ├── ParafovealStream (96×96 blurred)
  │     ├── FovealStream (48×48 STN crop)
  │     └── FovealParafovealGate (α blending)
  ├── AIS pillar (toggleable)
  │     ├── InformationGainGazePolicy (select_action → gaze coordinates)
  │     ├── EntropyGatedHalting (soft continuation weights)
  │     ├── PrecisionModulator (Π_D, Kalman-style)
  │     └── GenerativePrior (reconstruction loss)
  └── HPC pillar (toggleable)
        └── HierarchicalPredictiveStack (edge_map prediction, 1 level)
```

### Key Files

| Component | Path |
|---|---|
| RHANNext model | `rhan_core/model.py` |
| RHANNext config | `rhan_core/config/pillar_config.py` |
| AIS gaze policy | `rhan_core/gaze/` |
| HPC predictor | `rhan_core/predictive_coding/` |
| Trainer | `phase1_training/train_rhan_next.py` |
| Frozen eval entrypoint | `phase2_attacks/eval_rhan.py` |
| Stage 1-3 notebooks | `cloud_setup/colab_notebook_noesis.py`, `cloud_setup/Kaggle_NOESIS.py` |
| Roadmap | `docs/rhan_next_roadmap.json` |

### Reproduce

```bash
# Stage 3 training (D = AIS-v1 + HPC, 60 epochs)
python3 phase1_training/train_rhan_next.py \
  --enable-ais --no-ais-precision-recon \
  --enable-hpc --hpc-num-levels 1 --w-hpc 0.10 \
  --ckpt-name rhan_next_ais_hpc --max-epochs 60 \
  --target-ckpt checkpoints/rhan_next_ais_v1_halting_only_best.pth

# 8-seed eval (PGD-50 + PGD-100)
python3 phase2_attacks/eval_rhan.py \
  --ckpt-specs "trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large" \
              "rhan_next_ais_hpc:checkpoints/rhan_next_ais_hpc_best.pth:next" \
  --seeds 41 42 43 44 45 46 47 48 \
  --eps-list 0.0 0.094 --eps-norm-space \
  --pgd-steps 50 --n-samples 300 --batch-size 32
```

---

## Legacy: CIFAR-10 & STL-10 Model Comparison (13/13 Systems Complete)

### Robustness & Sensitivity Overview
| System | Clean Acc | PGD 50% Threshold | d′=1.0 Threshold | Status |
|--------|-----------|-------------------|-------------------|--------|
| Human | 74.15% | >0.30 | >0.30 | ✅ Complete |
| **RHAN-Large (Ours)** ★ | **85.20%** | **ε≈0.230** | **ε≈0.2500** | ✅ Complete |
| **RHAN-trades-curriculum** ★ | **78.12%** | **ε≈0.113** | **ε≈0.1850** | ✅ Complete |
| **RHAN-UNIFIED** | **74.30%** | **ε≈0.111** | **ε≈0.0760** | ✅ Complete |
| **RHAN-Self-Alignment** ⚠️ | **77.10%** | — | — | ⚠️ Obfuscated (AA: 21.60%) |
| **RHAN-Feature-Scatter** ⚠️ | **77.10%** | — | — | ⚠️ Obfuscated (AA: 22.30%) |
| **RHAN-TRADES-Hardened** | **86.33%** | **ε≈0.086** | **ε≈0.1246** | ✅ Complete |
| **RHAN-v5-TRADES** | **87.30%** | **ε≈0.078** | **ε≈0.1113** | ✅ Complete |
| **RHAN-v5 (Freq-Separated)** | **84.57%** | **ε≈0.071** | **ε≈0.1030** | ✅ Complete |
| **RHAN-v3 (Unified Recurrent)** | **91.41%** | **ε≈0.066** | **ε≈0.0900** | ✅ Complete |
| **RHAN-v4 (Multi-Scale)** | **89.65%** | **ε≈0.056** | **ε≈0.0800** | ✅ Complete |
| **RHAN-adv (Recurrent)** | **83.79%** | **ε≈0.053** | **ε≈0.0764** | ✅ Complete |
| RHAN-clean | 89.06% | ε≈0.023 | ε≈0.0330 | ✅ Complete |
| ResNet-18 | 95.82% | ε≈0.024 | ε≈0.0300 | ✅ Complete |
| ViT-Small | 97.80% | ε≈0.014 | ε≈0.0264 | ✅ Complete |
| BagNet-33 | 87.67% | ε≈0.010 | ε≈0.0170 | ✅ Complete |
| CORnet-S | 91.48% | ε≈0.006 | ε≈0.0090 | ✅ Complete |
| Shape-ResNet-50 | 91.47% | ε≈0.006 | ε≈0.0080 | ✅ Complete |
| EfficientNet-B0 | 96.81% | ε≈0.005 | ε≈0.0060 | ✅ Complete |
| RHAN-v6 (Dynamic Gating) | 82.03% | — | — | ⚠️ Regressed |
| **RHAN-TDV (STL-10)** | **78.50%** | **ε≈0.004** | **ε≈0.0043** | ✅ Complete |
| CLIP ViT-B/32 | — | — | — | 🔄 Pending |

**Headline:** All standard feedforward AI models collapse before ε=0.03. The curriculum-trained TRADES model, `RHAN-trades-curriculum`, extends visual robustness to **ε≈0.1850** (a **6.3× improvement** over ResNet-18). On higher-resolution STL-10 ($96\times96$), scaling model capacity to 55.6M parameters and expanding the training set 9.3× via mined pseudo-labels (**RHAN-Large**) successfully lifts clean accuracy by **+11.50 pp** (to **52.60%**) and certified AutoAttack robustness by **+1.30 pp** (to **10.60%**).

**STL-10 Scaling Success:** We have successfully integrated self-supervised causal Temporal Difference in Vision (TDV) pretraining on UCF-101 with large-scale semi-supervised pseudo-labeling on 100,000 unlabeled STL-10 images. Under the 120-epoch curriculum, this resolves the representational collapse of similar vehicle classes (Car vs. Truck) on clean images and significantly expands category margins under attack.

---

### Signal Detection Theory (Sensitivity)
| System | d'(0.00) | d'(0.01) | d'(0.05) | d'(0.10) | d'(0.20) | d'(0.30) | ε threshold |
|--------|----------|----------|----------|----------|----------|----------|-------------|
| Human  | 4.790 | 4.567 | 3.985 | 3.368 | 2.440 | 1.769 | >0.30 |
| **RHAN-Large (Ours)** ★ | **3.240** | **2.950** | **2.010** | **1.260** | **0.480** | **0.190** | **ε≈0.250** |
| **RHAN-trades-curriculum** ★ | **2.748** | **2.589** | **2.159** | **1.696** | **0.877** | **0.010** | **ε≈0.185** |
| **RHAN-UNIFIED** | **2.395** | **2.100** | **1.300** | **0.800** | **0.100** | **-0.500** | **ε≈0.076** |
| **RHAN-TRADES-Hardened** | **3.260** | **3.032** | **2.238** | **1.357** | **-0.094** | **-1.664** | **ε≈0.125** |
| **RHAN-v5-TRADES** | **3.383** | **3.186** | **2.230** | **1.231** | **-0.291** | **-1.602** | **ε≈0.111** |
| **RHAN-v5** | **3.083** | **2.905** | **2.071** | **1.104** | **-1.132** | **-1.808** | **ε≈0.103** |
| **RHAN-v3** | **3.710** | **3.189** | **1.983** | **0.753** | **-1.039** | **-3.044** | **ε≈0.090** |
| **RHAN-adv** | **3.083** | **2.738** | **1.662** | **0.408** | **-1.294** | **-3.044** | **ε≈0.076** |
| **RHAN-TDV (STL-10)** | **2.860** | **-0.597** | **-3.705** | **<0.000** | **<0.000** | **<0.000** | **ε≈0.004** |
| ResNet-18 | 4.426 | 2.687 | -0.771 | -1.707 | -1.913 | -1.880 | ε≈0.030 |
| ViT-Small | 4.931 | 1.814 | -0.154 | -0.909 | -1.242 | -1.469 | ε≈0.026 |

### PGD Accuracy Collapse
| Epsilon | RHAN-Large | RHAN-UNIFIED | RHAN-TDV | Curriculum | Hardened | TRADES | RHAN-v5 | RHAN-v3 | RHAN-adv | ResNet | ViT | EfficientNet | ShapeResNet | BagNet | Human |
|---------|------------|--------------|----------|------------|----------|--------|---------|---------|----------|--------|-----|--------------|-------------|--------|-------|
| 0.00 | 85.20% | 74.30% | 78.50% | 78.12% | 86.33% | 87.30% | 84.57% | 91.41% | 83.79% | 95.82% | 97.80% | 96.81% | 91.47% | 87.67% | 73.33% |
| 0.01 | 82.10% | 63.80% | 5.20% | 75.00% | 83.01% | 84.77% | 80.66% | 85.35% | 77.93% | 75.57% | 55.18% | 0.93%  | 18.11% | 48.04% | N/A |
| 0.05 | 70.30% | 34.60% | <2.00% | 65.23% | 67.19% | 65.82% | 61.13% | 60.74% | 51.95% | 2.84%  | 8.80%  | 0.00%  | 0.01%  | 0.12%  | 69.17% |
| 0.10 | 57.50% | 15.60% | <2.00% | 52.93% | 43.16% | 37.89% | 34.38% | 26.17% | 17.77% | 0.21%  | 2.78%  | 0.00%  | 0.00%  | 0.00%  | 59.17% |
| 0.20 | 38.00% | 3.20% | <2.00% | 29.49% | 8.59%  | 5.47%  | 2.73%  | 1.17%  | 0.59%  | 0.02%  | 1.12%  | 0.00%  | 0.00%  | 0.00%  | 62.22% |
| 0.30 | 25.10% | 0.60% | <2.00% | 10.16% | 0.20%  | 0.20%  | 0.20%  | 0.00%  | 0.00%  | 0.00%  | 0.58%  | 0.00%  | 0.00%  | 0.00%  | 58.61% |

---

## Overconfidence Finding
BagNet-33 and EfficientNet-B0 reach ~100% model confidence at ε=0.30 while accuracy is 0.00% — the maximum possible "confident but wrong" state. Humans show the opposite: declining confidence tracks declining accuracy, demonstrating intact metacognitive calibration absent in all tested CNNs.

## Semantic Confusion Structure
Adversarial errors are not random — they are semantically structured:
- **ResNet-18:** DOG→CAT (+37.2%), AUTOMOBILE→TRUCK (+34.1%)
- **ViT-Small:** TRUCK→SHIP (+59.0%)
- **Shape-ResNet:** HORSE→DEER (+35.4%) — most semantically coherent errors

## Generated Figures (phase4_analysis/figures/)
- `combined/partial_divergence_curve.png` — 5-model accuracy vs epsilon
- `combined/confidence_collapse.png` — confidence degradation curves
- `combined/confidence_accuracy_gap.png` — overconfidence gap per model
- `combined/perturbation_atlas.png` — 10-class perturbation difference maps
- `combined/hero_perturbation.png` — single high-impact perturbation example
- `combined/sufficient_input_subsets.png` — minimal evidence per model (SIS)
- `combined/vit_attention_entropy.png` — ViT attention scatter vs epsilon
- `combined/threshold_summary/` — accuracy and SDT ranking figures
- `combined/latent_space/` — t-SNE embeddings (ResNet + ViT)
- `vit/attention/` — per-class ViT attention maps (20 images)
- `{model}/confusion/` — confusion matrices clean vs adversarial

## RHAN Evolutionary Timeline

```
RHAN-clean → RHAN-adv → Trial branches (Split, PredCoding, Aligned)
                              ↓
                         RHAN-v2 (Unified Fine-tuning)
                              ↓
                         RHAN-v3 (Joint Scratch Training) ← εthresh=0.090
                              ↓
                     ┌─────────┴─────────┐
                  RHAN-v4            RHAN-v5 ← εthresh=0.1030
               (Multi-Scale,       (Frequency Separation,
                Active CLIP)       Phase 0 CLIP)
                  ↓ regressed          ↓
               RHAN-v6              RHAN-v5-TRADES ← εthresh=0.1113
            (Dynamic Gating,           ↓
             ACT Pondering)         RHAN-TRADES-Hardened ← εthresh=0.1246
              ↓ regressed              ↓
                                    RHAN-trades-curriculum ← εthresh=0.1850 (BEST CIFAR)
                                               │
                                  ┌────────────┴────────────┐
                        [Concept Bottlenecks]      [Feature Invariance]
                           (RHAN-CBM v1-v2)     (Self-Align / Feat Scatter)
                                  │                         │
                        (Irreducible at 32x32)     (Gradient Masking Theorem)
                                  └────────────┬────────────┘
                                               ▼
                                       [CIFAR-10 CLOSED]
                                               │
                                               ▼
                                     [TDV (Temporal Difference)]
                                                │
                                                ▼
                                         RHAN-TDV (STL-10) ← εthresh=0.0043 (Run 1: 13.3% Truck Robustness, Collapse Mitigated)
```

## Model Spectrum
| Model | Processing Style | Owner | Branch |
|-------|-----------------|-------|--------|
| BagNet-33 | Pure local patches (33×33) | Eyad | phase/1-bagnet |
| ResNet-18 | Local CNN, texture-biased | Mina | phase/1-resnet |
| EfficientNet-B0 | Compound scaled CNN (BIM attack) | Mina | phase/1-efficientnet |
| Shape-ResNet-50 | Shape-biased SIN training | Sandy | phase/1-shaperesnet |
| ViT-Small | Global patch attention | Mina | phase/1-vit |
| CORnet-S | Recurrent visual cortex model | Youssef + Eyad | phase/1-cornet |
| CLIP ViT-B/32 | Vision-language contrastive | Mariam | phase/1-clip |
| **RHAN-clean** | **Recurrent top-down feedback (clean)** | **Mina** | **dev** |
| **RHAN-adv** | **Recurrent top-down + adversarial curriculum** | **Mina** | **dev** |
| **RHAN-v3** | **Ventral/Dorsal split + adversarial alignment** | **Mina** | **phase/rhan-v2** |
| **RHAN-v4** | **Multi-scale gated feedback + active CLIP** | **Mina** | **phase/rhan-v4** |
| **RHAN-v5** | **Frequency separation + Phase 0 CLIP init** | **Mina** | **phase/rhan-v5** |
| **RHAN-v6** | **Dynamic gating + predictive coding + ACT** | **Mina** | **phase/rhan-v6** |
| **RHAN-v5-TRADES** | **Standard TRADES adversarial training** | **Mina** | **phase/rhan-trades** |
| **RHAN-TRADES-Hardened** | **Class-hardened TRADES with margin loss** | **Mina** | **phase/rhan-trades** |
| **RHAN-trades-curriculum** | **TRADES 3-Phase Extended Curriculum** | **Mina** | **phase/rhan-trades-curriculum** |
| **RHAN-Self-Alignment** | **Feature-space cosine distance fine-tuning** | **Mina** | **phase/rhan-self-alignment** |
| **RHAN-Feature-Scatter** | **Feature-space scatter mapping with corrected bounds** | **Mina** | **phase/rhan-feature-scatter** |
| **RHAN-CBM v1-v2** | **Concept Bottleneck Models with straight-through estimator** | **Mina** | **phase/rhan-cbm** |
| **RHAN-v7** | **Generative World-Model (VAE + TRADES)** | **Mina** | **dev** |
| **RHAN-UNIFIED** | **Unified architecture, STL-10 96×96, from scratch** | **Mina** | **dev** |
| **RHAN-TDV-Clean** | **Temporal Difference pretrained backbone (clean consistency)** | **Mina** | **phase/rhan-tdv** |
| **RHAN-TDV-Adv** | **Temporal Difference pretrained backbone (adv consistency)** | **Mina** | **phase/rhan-tdv** |
| **RHAN-Large (Ours)** | **55.6M parameter model + semi-supervised pseudo-labeling** | **Mina** | **main** |
| Human | Biological vision (n=18) | All | — |

## Team

| Contributor | GitHub | Role |
|-------------|--------|------|
| **Mina Magdy (FerrariKazu)** | [@FerrariKazu](https://github.com/FerrariKazu) | ResNet ✅, ViT ✅, EfficientNet ✅, RHAN (all versions), pipeline, human study, Phase 4+5 |
| **Sandy Antonius** | [@SandyAntonius](https://github.com/SandyAntonius) | Shape-ResNet ✅, final report, slides |
| **Eyad Saleh Ali** | [@eyadsalehali07-coder](https://github.com/eyadsalehali07-coder) | BagNet ✅, CORnet-S (co-owner) |
| **Youssef Ayman (Mekky)** | [@Mekky2](https://github.com/Mekky2) | CORnet-S (co-owner) |
| **Mariam Mohammed** | [@Mariam-203](https://github.com/Mariam-203) | CLIP ViT-B/32 |

```bash
## Setup
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Results
```bash
# Phase 2: Generate adversarial arrays (memory-safe, one model at a time)
python phase2_attacks/generate_adv_all_models.py --model [resnet|vit|efficientnet|shaperesnet|bagnet]
```

```bash
# Phase 4: Run all analysis
python phase4_analysis/generate_all_figures.py
```

```bash
# Phase 5: SDT analysis
python phase5_sdt/sdt_analysis.py
```

```bash
# RHAN-UNIFIED Training (STL-10 96x96)
# Phase 0: Semantic initialization with unlabeled data (50 epochs)
python phase1_training/train_rhan_unified.py --phase 0

# Phases 1-8: TRADES adversarial curriculum (160 epochs)
python phase1_training/train_rhan_unified.py --phase 1-6

# Everything at once
python phase1_training/train_rhan_unified.py --phase all
```

```bash
# RHAN-TDV Training & Evaluation (STL-10 96x96)
# Phase 1: Self-supervised TDV pretraining (unlabeled data, 30 epochs)
python phase1_training/train_rhan_stl10_tdv.py --phase tdv --unlabeled-batch-size 32

# Phase 2: Classification head label calibration (5K labeled images, 10 epochs)
python phase1_training/train_rhan_stl10_tdv.py --phase label --batch-size 64

# Phase 3: TRADES curriculum fine-tuning with TDV consistency (60 epochs)
python phase1_training/train_rhan_stl10_tdv.py --phase trades --batch-size 16 --unlabeled-batch-size 16

# Run PGD-100 & SDT Evaluation Sweep
python phase1_training/eval_pgd_sdt_stl10.py --checkpoint ../checkpoints/rhan_stl10_tdv_trades_clean_consistency.pth --samples 1000 --batch-size 64
```

```bash
# RHAN-Large + Pseudo-Label Curriculum (STL-10 96x96)
# Launch optimized DDP training on dual T4 GPUs (batch size 32, 8 accumulation steps)
torchrun --nproc_per_node=2 phase1_training/train_rhan_large_pseudolabel.py --batch-size 32 --accum-steps 8

# Launch via pipeline automation script (automatically manages environment, keys, and GPU setup)
python3 cloud_setup/kaggle_run_pseudolabel_pipeline.py --batch-size 32

# Run full evaluation sweep (AutoAttack + PGD-20 sweeps) on 1000 samples
python3 run_eval_stl10.py --model-size large --checkpoint checkpoints/rhan_stl10_large_pseudolabel_rolling.pth --samples 1000
```

## Human Study
n=18 participants, 1,800 trials, 5 epsilon blocks.
Data: `phase3_human_study/data/responses_mapped.csv`
Mapping: `phase3_human_study/manifest.csv`

## Repository Structure
```text
.
├── config/                 # Attack and training configuration (YAML)
├── phase1_training/        # Model architectures and training scripts
│   ├── model.py            # Modified ResNet-18 for CIFAR
│   ├── model_vit.py        # ViT-Small architecture
│   ├── model_efficientnet.py
│   ├── model_shaperesnet.py
│   ├── model_bagnet.py
│   ├── model_rhan.py       # RHAN base architecture (clean/adv/v2/v3)
│   ├── model_rhan_v5.py    # Frequency-separated biologically-grounded model
│   ├── model_rhan_v6.py    # Dynamic gating + predictive coding + ACT
│   ├── model_rhan_v7.py    # Generative World-Model (VAE + TRADES)
│   ├── model_rhan_unified.py  # Unified architecture for STL-10 96x96
│   ├── model_rhan_stl10.py    # STL-10 adaptation (predecessor)
│   ├── dataset_stl10.py       # STL-10 data loaders (labeled + unlabeled)
│   ├── train.py            # Standard training loop
│   ├── train_rhan_v5.py    # Phase 1 epsilon curriculum training
│   ├── train_rhan_v5_trades.py  # TRADES adversarial training
│   ├── train_rhan_v7.py    # v7 generative world-model training
│   ├── train_rhan_unified.py  # UNIFIED: Phase 0 + Phases 1-8 (STL-10)
│   ├── pretrain_rhan_v5_clip.py # Phase 0 CLIP semantic initialization
│   └── pretrain_rhan_v6_clip.py # Phase 0 for v6
├── phase2_attacks/         # FGSM/PGD attack generation
│   ├── generate_adv_all_models.py
│   ├── pgd.py              # Multi-step PGD implementation
│   └── fgsm.py             # Single-step FGSM
├── phase3_human_study/     # Human behavioral data
│   ├── data/               # Mapped human responses
│   └── stimuli/            # Exported adversarial stimuli
├── phase4_analysis/        # Interpretability & Divergence
│   ├── figures/            # All generated plots and heatmaps
│   ├── divergence_curves.py
│   ├── confidence_curves.py
│   ├── confusion_matrices.py
│   ├── latent_space_embeddings.py
│   ├── vit_attention_maps.py
│   ├── alignment_analysis.py  # CORnet IT alignment metrics
│   └── perturbation_visuals.py
├── phase5_sdt/             # Signal Detection Theory (SDT)
│   ├── sdt_analysis.py     # Main d' and criterion calculation
│   └── sdt_core.py         # SDT mathematical primitives
├── checkpoints/            # Model weights (git-ignored)
├── scratch/                # Evaluation and debugging scripts
└── utils/                  # Shared metrics and logging
```

## References
1. Brendel, W., & Bethge, M. (2019). Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet. ICLR 2019.
2. Geirhos, R., et al. (2019). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. ICLR 2019.
3. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. ICLR 2015.
4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.
5. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
6. Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics. Wiley.
7. Kubilius, J., et al. (2019). Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs (CORnet). bioRxiv.
8. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
9. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
10. Madry, A., et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018.
11. Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. IEEE S&P 2017.
12. Macmillan, N. A., & Creelman, C. D. (2005). Detection theory: A user's guide (2nd ed.). Lawrence Erlbaum Associates.
13. Ilyas, A., et al. (2019). Adversarial Examples Are Not Bugs, They Are Features. NeurIPS 2019.
14. Carter, B., et al. (2019). Exploring Statistical and Structural Properties of Feedforward and Recurrent Neural Networks. arXiv.
15. Zhang, H., et al. (2019). Theoretically Principled Trade-off between Robustness and Accuracy (TRADES). ICML 2019.
---

## Documented gaps and caveats

1. **Gen-1 pipeline sources are not on `main`.** `training/` (`train_generation1_foundation.py`, `adv_curriculum.py`, `stage_state_machine.py`), `evaluation/` (`clean_and_robust.py`, `imagenet100_loader.py`, …), the Gen-1 `scripts/` tooling (`prepare_imagenet100.py`, `sbr0_gate.py`, `eval_ais_v2_gate.py`, `stage_state_machine.py`, `verify_run_complete.py`, `freeze_run_manifest.py`), the `noesis_vision/` package source (`noesis_vision/core/`, `noesis_vision/gaze/ais_v2_policy.py`, …), live on the carrier branches `feature/rhan-next` / `stage2/nxa-pipeline-refactor`. The Gen-1 `cloud/gen1/` Colab entry points are partially on `main`: `colab_j1_foundation.py` and `colab_j1_foundation_gate.py` are committed there, while `Kaggle_J1_FOUNDATION.py`, `run_j1_local.sh`, and `run_j1_supervised.sh` remain `stage2`-only. On `main`, `training/` and `evaluation/` contain only `__pycache__` remnants. `main` and `feature/rhan-next` have diverged (6 / 74 commits).
2. **Parts of the working tree are not committed.** `.gitignore` rules (`docs/`, `report/`, `runs/`, `data/`, and any directory literally named `docs/`) keep `docs/REPOSITORY_MAP.md`, `docs/RHAN_NXA_ARCHITECTURE.md`, `docs/CANCELLATION_NOTICE_2026-10-03.md`, `docs/FORENSIC_REPORT_NXA_GENERATION1.md`, `report/GEN1_RESULTS_MASTER.md`, `report/generation1_foundation_roadmap.json`, `runs/` manifests, `noesis_vision/RHAN_NXA/docs/`, `archive/`, and `checkpoints/*` (140+ weights) out of the repository. They exist on this machine; a fresh clone will not contain them. Only 10 `docs/` files and 16 `report/` files are tracked (plus 4 checkpoints).
3. **`noesis_vision/RHAN_NXA/MASTER_PLAN.md` is absent from `main`** (tracked on `feature/rhan-next`); some `noesis_vision/RHAN_NXA/docs/` chapters link to it, so those links are broken here. The on-tree plan is `noesis_vision/RHAN_NXA/docs/Proposed_Plan.md` + chapters 00–30.
4. **Two roadmap JSONs.** `docs/rhan_next_roadmap.json` = Gen-0 RHAN-Next orchestration (tracked); `report/generation1_foundation_roadmap.json` = Gen-1 ladder state (working tree only). The latter is the source of truth for Gen-1 ladder status.
5. **`phase4_analysis/generate_all_figures.py`** (cited by the original README) no longer exists; superseded by `phase4_analysis/generate_hero_figures.py` and `generate_appendix_figures.py`.
6. **`requirements.txt` is Gen-0-era** (wheel index `cu124`; the research environment ran torch 2.9.1+cu128) — a starting point, not a lockfile. Also: `num_workers>=2` deadlocks the ImageNet-100 loader locally; `FoundationConfig` has no `device=` kwarg.
7. **Historical commands are archived**, not current entry points — including everything in the reprinted *Setup* / *Reproduce Results* blocks below.
8. **`checkpoints_tier2/` contains only `*:Zone.Identifier` stubs** (Windows download metadata), no weights; the same residue exists as `*:Zone.Identifier` files elsewhere in the tree.
9. **Gen-1 evidence caveat:** the Gen-1 arms were contaminated by legacy SBR, so mechanism-discovery claims are **UNKNOWN** — neither confirmed nor refuted (evidence: `noesis_vision/RHAN_NXA/docs/16_Gen0_Evidence_And_Confounds.md`).
10. **The reprinted historical section** is verbatim except two generation-label corrections and code-fence normalization (disclosed above). Paths inside it are as-of the original README; a few (`phase4_analysis/generate_all_figures.py`, `phase3_human_study/stimuli/`) are gone.
11. **`docs/REPOSITORY_MAP.md`** (detailed 23-section map) was generated 2026-09-30 on `feature/rhan-next` @ `fef50f3` — it describes that branch, not `main`, and is itself uncommitted. The map in *Repository map* above was regenerated from the `main` working tree for this README.

---

## About this README

Documentation-only deliverable: this commit changes **`README.md` alone** — no model, training, test, architecture, dependency, config, or source file was touched. Inconsistencies found during reconstruction are recorded in *Documented gaps* rather than silently fixed.

- **README** = entry point (this file).
- **Gen-1 plan** = `noesis_vision/RHAN_NXA/` (`docs/` 00–30 + `Proposed_Plan.md` on the working tree; `MASTER_PLAN.md` on carrier branches).
- **Gen-0 architecture plan** = `docs/ARCHITECTURE.md` + `docs/rhan_next_roadmap.json`.
- **Gen-2 audit record** = `gen2_foundation/AUDIT.md` + `gen2_foundation/tests/`.
- **Detailed map of `feature/rhan-next`** = `docs/REPOSITORY_MAP.md` (uncommitted).
- **Source code** = implementation truth; `gen2_foundation/` code = Gen-2 mechanism truth.

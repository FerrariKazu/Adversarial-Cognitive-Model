# Adversarial Cognition Divergence

**A 12-model + human psychophysics study of adversarial robustness**

> Does adversarial robustness scale with global visual processing — and is it
> determined by architecture, training objective, or recurrence?

---

## CURRENT RESEARCH GENERATION: RHAN-NXA / Gen-1

The active system on branch `feature/rhan-next` is **RHAN-NXA (Generation 1)**,
packaged as `noesis_vision/` and trained by the six-phase foundation ladder
with the TRADES/PGD adversarial curriculum as the **default** objective.

| Start here | Path |
|---|---|
| What is RHAN-NXA? | `noesis_vision/RHAN_NXA/docs/00_README.md` (+ `MASTER_PLAN.md`) |
| Current architecture guide | `docs/RHAN_NXA_ARCHITECTURE.md` |
| Current model | `noesis_vision/` (`core/`, `models/`, `beliefs/`, `predictive_coding/`, `uncertainty/`, `gaze/`) |
| Current training | `training/train_generation1_foundation.py` (+ `training/adv_curriculum.py`) |
| Current evaluation | `evaluation/clean_and_robust.py` via `scripts/generate_full_sweep.py` |
| Tests | `tests/` — run `python3 -m pytest tests/ -q` |
| Cloud launcher | `cloud/gen1/Kaggle_J1_FOUNDATION.py` |
| Repository map (authoritative) | `docs/REPOSITORY_MAP.md` |
| Contributor guide | `CONTRIBUTING.md` |

### Generation lineage

```text
RHAN historical lineage (CIFAR-10 12-model study, RHAN v1–v7, STL-10 scale-up)
        ↓
Gen-0 / RHAN-Next  (archive/gen0/ — frozen reference generation)
        ↓
Gen-1 / RHAN-NXA   (noesis_vision/ — CURRENT)
        ↓
future RHAN generations
```

### Naming contract

- **RHAN** — the research lineage / architecture family (all generations).
- **RHAN-Next** — the Generation-0 model + package (`archive/gen0/`); frozen,
  kept as the comparison reference. Formerly marked "Current" in this README.
- **RHAN-NXA** — the Generation-1 architecture (`noesis_vision/`); current.
- **NOESIS** — the umbrella research/software identity used by the Gen-1
  documentation (`docs/NOESIS_FOUNDATION.md`, `noesis_vision/RHAN_NXA/`):
  "RHAN-NXA" names the architecture, "NOESIS" names the framework.
- **RHANvN** (v1–v12) — the pre-Gen-0 CIFAR/STL-10 lineage in
  `archive/gen-1-cifar12/` (historical).

---

## HOW TO WORK: WHERE DO I GO?

| I want to… | → Path |
|---|---|
| Modify the current model / package | `noesis_vision/` |
| Modify the canonical Gen-1 trainer | `training/train_generation1_foundation.py` |
| Modify the training curriculum | `training/adv_curriculum.py` |
| Modify the phase state machine | `training/stage_state_machine.py` |
| Modify Stage-2 pipeline / DAG | `training/stage2_pipeline.py` |
| Modify evaluation harness | `evaluation/` |
| Add an experiment (future) | `experiments/` |
| Run diagnostics / forensics | `diagnostics/` |
| Reproduce current six-phase ladder | `training/train_generation1_foundation.py --smoke` |
| Inspect historical RHAN v1–v7 | `archive/rhan-v1-v7/` |
| Inspect Gen-0 RHAN-Next | `archive/gen0/` |
| Reproduce pre-Gen-1 CIFAR/STL pipeline | `archive/gen-1-cifar12/` |
| Inspect frozen Gen-0 eval entrypoint | `archive/legacy-evals/phase2_attacks/eval_rhan.py` |
| Launch cloud training (Gen-1) | `cloud/gen1/` |

---

## REPOSITORY MAP

```
README.md
CONTRIBUTING.md
docs/
  ├── REPOSITORY_MAP.md                       # implemented map
  ├── repository_reorganization.md            # this refactor's changelog
  ├── RHAN_NXA_ARCHITECTURE.md
  ├── STAGE2_REFACTOR_PLAN.md
  ├── rhan_next_roadmap.json                  # authoritative orchestration state
  ├── research/                               # experiment registry, literature corpus, lessons
  ├── historical/                             # pre-Gen-1 lineage docs
  └── (Gen-1 docs live in noesis_vision/RHAN_NXA/docs/)
noesis_vision/             # ★ CANONICAL Gen-1 package (RHAN-NXA) + docs
training/                  # ★ CANONICAL Gen-1 training
evaluation/                # ★ CANONICAL Gen-1 evaluation
scripts/                   # ★ Gen-1 tooling (data, gates, verifier, sweep)
tests/                     # ★ test suite (24 mechanism-seam green + more)
archive/
  ├── gen-1-cifar12/       # ALL pre-Gen-1 models/trainers (from phase1_training)
  │   └── checkpoints/     # historical checkpoints (gitignored *.pth)
  ├── rhan-v1-v7/          # RHAN v1-v7 lineage
  ├── stl10-scaleup/       # STL-10 UNIFIED / TDV / RHAN-Large
  ├── gen0/                # frozen Gen-0 RHAN-Next (rhan_core + legacy checkpoints)
  ├── legacy-evals/        # ALL eval/check/debug/examine tooling
  ├── gen3-human/          # phase3 human psychophysics (n=18)
  ├── gen4-analysis/       # phase4 analysis
  ├── gen5-sdt/            # phase5 SDT
  ├── pkg-rhan-math/       # rhan_math
  ├── pkg-tier1/           # tier1
  ├── working-scratch/     # scratch/ working scripts
  └── historical-report/   # Paper/, RHANv12/, RHANv10Report/, presentational/, competition/
cloud/
  ├── canonical/           # symlink → cloud/gen1/Kaggle_J1_FOUNDATION.py (Gen-1 canonical)
  ├── gen1/                # Gen-1 cloud launchers + notebooks
  ├── gen0/                # Gen-0 cloud launchers
  └── misc/                # misc cloud scripts
diagnostics/               # measurement/forensic tooling (from diagnosis_artifacts)
experiments/               # future research variants (empty at implementation time)
checkpoints/               # GENERATED weights (gitignored *.pth)
runs/                      # GENERATED per-run manifests/logs (gitignored)
report/                    # GENERATED reports (gitignored, but key files tracked)
data/                        # downloaded datasets (gitignored)
images.png notes.txt
```

### Generation classification (final)

- **Canonical current (active):** `noesis_vision/`, `training/`,
  `evaluation/`, `scripts/`, `tests/`, `cloud/canonical/`, `docs/` (gen1 docs)
- **Historical (read-only, scientifically preserved):** `archive/*` — every
  pre-Gen-1 generation is fully intact
- **Experimental (future):** `experiments/` (empty at implementation time)
- **Generated artifacts (documented at schema level):** `checkpoints/`,
  `runs/`, `report/`

---

## CURRENT ARCHITECTURE

### Six-phase foundation ladder (training/stage_state_machine.py)

```text
backbone_only  →  recurrence_only  →  belief_no_f  →  belief_with_f
      ↓                    ↓                   ↓                ↓
AIS-v2 swap → gen1_core (frozen reference)
```

Each phase adds exactly one mechanism:
- **backbone_only** — substrate + one fixed center fixation + classifier head
  (no refinement, no recurrence, no belief, no uncertainty)
- **recurrence_only** — + T=4 fixed-schedule glimpse loop + tied refinement
- **belief_no_f** — + belief carrier with U_t (S=None VectorBeliefState,
  EvidentialHead); identity update
- **belief_with_f** — + Agent E belief dynamics (UpdateNet + precision)
- **ais_v2_swap** — + Agent F AIS-v2 gaze (policy-driven)
- **gen1_core** — + integrated system as frozen reference

### Data / model / training flow

```text
CLI: python3 training/train_generation1_foundation.py
     [--smoke | --phase <p>] [--clean-only] [--force-fresh]
     [--data-root …] [--epochs N] [--batch-size 48] …

→ FoundationConfig (in-file) ∪ RHANNXAConfig (core/schema.py)
→ stage_state_machine.py (six-phase orchestration)
→ FoundationModel (CompactViT substrate + per-phase composition)
→ adv_curriculum.py (TRADES/PGD default objective)
→ multi_group_optimizer.py (per-group optimizer)
→ checkpoints/*.pth (best + rolling), runs/*/manifest.json
→ report/foundation_{phase}_result.json + _compactness.json
→ HF sync (FerrariKazu/rhan-nxa-checkpoints)
```

---

## HISTORICAL GENERATIONS (PRESERVED — read-only)

### archive/gen-1-cifar12/
All pre-Gen-1 models, trainers, and datasets:
- ~45 `train_*.py` teams: RHAN v1-v12, TRADES, CBM, self-alignment, etc.
- STL-10 scaleup: `model_rhan_stl10*.py`, `train_rhan_stl10_tdv.py`,
  `train_rhan_large_pseudolabel.py`
- CIFAR-10 baselines: ResNet, ViT, EfficientNet, BagNet, CORnet, ShapeResNet

### archive/rhan-v1-v7/
RHAN recurrent lineage: `model_rhan.py`, `_v3_adaptive`, `_v4` … `_v7`,
`train_rhan*.py`

### archive/stl10-scaleup/
STL-10 UNIFIED / TDV / RHAN-Large: `model_rhan_unified.py`,
`model_rhan_stl10*.py`, `train_rhan_unified.py`, `train_rhan_stl10_tdv.py`,
`train_rhan_large_pseudolabel.py`, `dataset_stl10.py`

### archive/gen0/
Frozen Gen-0 RHAN-Next: `rhan_core/` package, legacy checkpoints,
`eval_rhan.py` (frozen Gen-0 eval) — preserved for scientific provenance.

### archive/legacy-evals/
- `phase2_attacks/` — attack generation + frozen `eval_rhan.py` Gen-0 eval
- Root-level eval scripts: `eval_*.py`, `inspect_*.py`, `demo.py`,
  `concept_ablation.py`, `check_parquet.py`, `bench_pgd.py`, etc.

### archive/gen3-human/
n=18 human psychophysics data (form responses, manifest, stimuli).

### archive/gen4-analysis/
Pre-Gen-1 interpretability suite: divergence/confidence curves, Grad-CAM,
ViT attention, SIS, alignment analysis.

### archive/gen5-sdt/
Signal Detection Theory: `sdt_analysis.py`, `sdt_core.py`,
`results/sdt_results*.csv`.

### archive/pkg-rhan-math/
Mathematical proof reports (`phase1_proofs.md` … `phase5_proofs.md`),
`generate_proof_figures.py`, assets.

### archive/pkg-tier1/
ScientificValidationReport LaTeX v1/v2 + per-seed results JSONs,
`validate_rhan.py`.

### archive/working-scratch/
`scratch/` working scripts (HF checks, PGD debugging, diagnostics,
roadmap surgery, verification).

### archive/historical-report/
Paper/ (ACD paper v1/v2), RHANv12, RHANv10Report, presentational/,
competition/.

---

## EXPERIMENTAL (future)

`experiments/` — reserved for future scientific variants (Kimi K3
implementations will land here as isolated experiments).

---

## GENERATED ARTIFACTS (documented at schema level)

- `checkpoints/` — model weights (`*.pth`, gitignored), ~100+ files across all
  generations. Canonical weights live on HuggingFace.
- `runs/` — per-run manifests/logs (gitignored).
- `report/` — result reports (gitignored, but `GEN1_RESULTS_MASTER.md` and
  `rhan_nx_generation1_report.md` are tracked).

---

## SETUP

```bash
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
# Mechanism-seam suite (24/24 passing)
python3 -m pytest tests/test_stage2_mechanism_seam.py -v

# Full suite
python3 -m pytest tests/ -q
```

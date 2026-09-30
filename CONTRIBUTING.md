# Contributing to RHAN / NOESIS

Practical guide for working in this repository. The authoritative description
of the repository is `docs/REPOSITORY_MAP.md`; the current-generation
architecture is `docs/RHAN_NXA_ARCHITECTURE.md`.

---

## 1. Current generation

The active research system is **RHAN-NXA (Generation 1)** — package
`noesis_vision/`, trained by the six-phase foundation ladder. Gen-0
(`rhan_core/`, "RHAN-Next") is a **frozen reference generation**, kept for
comparison. Everything under `phase1_training/` … `phase5_sdt/` is
historical research infrastructure from earlier generations.

## 2. Repository structure (the 60-second version)

```text
noesis_vision/        CURRENT model package (RHAN-NXA) + its documentation
training/             CURRENT trainer, adversarial curriculum, phase machine
evaluation/           CURRENT evaluation harness (Agent I)
tests/                CURRENT test suite (pytest)
scripts/              CURRENT tooling (data, sweeps, gates, verifier)
cloud_setup/          cloud launchers (current: Kaggle_J1_FOUNDATION.py)
docs/                 documentation incl. REPOSITORY_MAP + RHAN_NXA_ARCHITECTURE
rhan_core/            FROZEN Gen-0 reference package — do not modify
phase1..5_*/          HISTORICAL generations — do not modify
report/, runs/,       working artifacts (gitignored / HF-mirrored)
checkpoints/, data/
```

## 3. Canonical branch

`feature/rhan-next` is the development branch. RHAN-NXA is not merged to
`main` until J1/J2 validate. The Kaggle launcher checks the branch exists on
origin and hard-resets to `origin/feature/rhan-next` — **push your work
before expecting a cloud run to see it.**

## 4. Creating a feature branch

```bash
git fetch origin
git checkout -B feature/<your-topic> origin/feature/rhan-next
```

Keep branches short-lived and rebased onto `feature/rhan-next`. Historical
`phase/*` branches are generation provenance — never delete or rewrite them
without an explicit, documented audit.

## 5. Running tests

```bash
python3 -m pytest tests/ -q        # ~2–15 min depending on machine load
```

Notes:
- Use `python3`, never `python` (environment convention).
- **Bare `pytest` INTERNALERRORs in this environment** — always use
  `python3 -m pytest`.

## 6. Smoke test (before any GPU spend)

```bash
python3 training/train_generation1_foundation.py --smoke
```

Runs the entire six-phase chain on synthetic loaders in seconds, CPU-able,
no HF writes, artifacts quarantined under `smoke/`. If you touched the
trainer, curriculum, or state machine, this must pass.

## 7. Where to modify code

| You want to… | Touch |
|---|---|
| Change the model | `noesis_vision/` (substrate: `models/`; belief: `beliefs/`; etc.) |
| Change training objective/loop | `training/` |
| Change evaluation | `evaluation/` |
| Add tooling | `scripts/` |
| Cloud orchestration | `cloud_setup/` (guards must stay in parity across Kaggle/Colab) |

Every new loss-bearing component needs a gradient-reachability test — this is
the #1 historical failure mode (see `docs/ARCHITECTURE.md` lesson 1) and the
reason `tests/test_*gradient_flow*.py` exists.

## 8. How to add an experiment

1. Read `noesis_vision/RHAN_NXA/MASTER_PLAN.md` and the relevant
   `noesis_vision/RHAN_NXA/docs/` chapter. Experiments are pre-registered:
   gate criteria BEFORE running.
2. Add the experiment to the state machine / roadmap — never drive runs by
   hand-edited boolean flags; `get_next_action()` decides.
3. Every run writes a provenance manifest (`write_manifest`) including the
   config hash. A changed config hash under the same run id is a STOP, not
   an overwrite.
4. Evaluation goes through `evaluation/clean_and_robust.py` — norm-space ε,
   seed floor, per-seed CSV, consistency assertion. Do not hand-roll evals
   for citable numbers.

## 9. Preserving research provenance

- **Never delete** historical code, data, reports, or rejected experiments.
  If something leaves the active tree, it moves to a clearly labeled archive
  (future: `archive/`) with its history intact (`git mv`, not copy+delete).
- Every result must be traceable to: objective (e.g.
  `adv_curriculum.clean_only`), config hash, checkpoint SHA-256, seed list,
  and eval provenance JSON. Keep it that way.
- Label arms explicitly: the Gen-1 pure-CE run is the **historical control
  arm**; the TRADES/PGD run is the corrected arm. Never merge their tables.

## 10. What NOT to modify

- `rhan_core/` — frozen Gen-0 reference (contractually: `model_rhan_v12.py`
  "FROZEN"; the same discipline applies to the package).
- `phase1_training/train_rhan_next.py`, `phase2_attacks/eval_rhan.py` —
  frozen Gen-0 entrypoints; they define published numbers.
- `phase3_human_study/` — irreplaceable human-subject data.
- `report/`, `checkpoints/` — result artifacts; HF repos are canonical.
- `T4x2.ipynb` — never commit (standing exclusion).
- Scientific constants (curriculum table, `W_TRADES_DEFAULT`, seed floors,
  K=4 candidates, `MIN_PROTOCOL_SEEDS`) — changing them is a new
  experiment, not a refactor.

## 11. How historical code is handled

Historical systems stay runnable in place (`rhan_core/`, `phase*/`). If a
future reorganization moves files, it must: preserve bytes (verify hashes),
update every reference, and provide tiny documented wrappers only where a
frozen path must keep working. No magic alias layers.

## 12. Commit structure

- One logical change per commit; the repo must be testable at every commit.
- Conventional prefixes: `docs:`, `refactor:`, `feat:`, `fix:`, `chore:` —
  with a subject that says *why*.
- **Clean commit messages: no attribution/co-authored footers.**
- Never bundle branch deletion, refactoring, and behavior changes together.

## 13. Experiment artifacts

| Artifact | Where |
|---|---|
| Source + docs | Git |
| Checkpoints, eval CSVs, reports, archives | HuggingFace (`FerrariKazu/rhan-nxa-checkpoints[-rolling]`) |
| Local working data/runs | `data/`, `runs/`, `report/` (gitignored) |

The roadmap JSON's canonical state lives on the HF **rolling** repo; local
copies are caches. Don't hand-edit them to change what runs next.

## 14. Git branches

`feature/rhan-next` = active. `main` = STL-10-era tip (frozen until
validation). `phase/*`, `dev`, `backup-pre-rewrite` = historical provenance.
Branch deletion is a separate, explicit, audited operation — never part of a
refactor commit.

## 15. Where to start (reading order)

1. `README.md` (current-generation banner)
2. `noesis_vision/RHAN_NXA/docs/00_README.md` → `01_…` → `16_Gen0_Evidence_And_Confounds.md`
3. `docs/RHAN_NXA_ARCHITECTURE.md`
4. `docs/REPOSITORY_MAP.md`
5. `training/train_generation1_foundation.py` (header comment is a guided tour)
6. `tests/test_gradient_flow.py` + `tests/test_adv_curriculum.py`
7. `report/GEN1_RESULTS_MASTER.md` (control-arm results)

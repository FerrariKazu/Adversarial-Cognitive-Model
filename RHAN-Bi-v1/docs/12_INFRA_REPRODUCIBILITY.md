# 12 — Infrastructure and reproducibility

## 1. Hardware and environment

Reference machine: one NVIDIA RTX 4060 with 8,188 MiB, 16 GB system RAM, Linux. The package and its
tests run on CPU only; the GPU is used only by the optional DINOv2 feature extraction.

| Dependency | Version used | Role |
|---|---|---|
| Python | 3.10 | runtime |
| numpy | 2.2 | rendering, features, statistics |
| scipy | 1.15 | tests, distributions, optimisation |
| scikit-learn | 1.7 | logistic regression, metrics |
| matplotlib | 3.10 | figures |
| PyYAML | 5.4 | configs |
| torch + timm | 2.9 / 1.0 (optional) | DINOv2 path |
| pytest | 9.0 | tests |

No dependency beyond `requirements.txt` is needed for the demo. `torch` and `timm` are guarded imports:
the package works without them, and the DINOv2 path raises a clear error rather than an ImportError at
import time.

## 2. Determinism

| Source of nondeterminism | Control |
|---|---|
| scene content | pure function of `(seed, scenario)` |
| observation noise | seeded from `(seed, step, action_index)` |
| feature extraction | deterministic numpy; no threaded reductions that reorder sums |
| model fitting | LBFGS / fixed-order SGD; `random_state` fixed |
| planner Monte Carlo | generator seeded from `(episode_seed, step)` |
| policy tie-breaks | fixed priority, 1e-9 tolerance |
| file ordering | sorted keys and sorted record ids in every JSON artifact |

Two runs of the demo in the same environment produce byte-identical `results/demo_report.json` except
for the timestamp fields, which are excluded from the comparison (and from every hash).

`tests/test_end_to_end.py` asserts exact equality of the report's metrics across two in-process runs.

## 3. Artifacts and hashes

Every run writes:

```
results/
  demo_report.json        metrics + config + code version + split hashes
  demo_report.md          generated from the JSON, never hand-edited
  episodes.jsonl          one episode per line, full trace
  stage_a.json ...        per-gate metrics
  splits.json             seed ranges + counts + hashes
  run_manifest.json       command, config hash, git commit, python, platform
```

`run_manifest.json` is the provenance record. It contains no wall-clock in any hashed field; a
human-readable `started_at` is present but excluded from the config hash.

## 4. Versioning

- **Package** `rhan_bi.__version__ = "1.0.0"` is written into every record's `provenance.code_version`.
- **Features** `features-v1`; a change to a feature definition increments it and invalidates stored
  likelihoods (records with an older `feature_version` are rejected, not silently scored).
- **Ledger schema** `rhan-bi-ledger-v1`.
- **Simulator** `inspection-v1.0`.
- **Evidence model** identified by the SHA-256 of its fitted parameters (`evidence-model-<hash>`).

A result file that cites an older model hash is not comparable with one citing a newer hash, and the
report generator refuses to pool them.

## 5. Runtime and memory

| Command | CPU time | Peak RSS |
|---|---|---|
| `pytest tests -q` | ~8 s | < 400 MB |
| `python -m rhan_bi.cli demo` | ~3 min | < 700 MB |
| `python -m rhan_bi.cli gate-e --n 400` | ~3–5 min | < 900 MB |
| `python figs/make_figures.py` | ~10 s | < 500 MB |
| DINOv2 feature extraction (1,000 images, 224², batch 32) | ~4 min on the 4060 | ~1.6 GB VRAM |

The demo's dominant cost is the simulator renderer and the pair comparisons, not the model. Nothing in
the CPU path uses more than one core deliberately; parallel scene generation is available via
`--workers` but is off by default so that determinism is trivial to reason about.

## 6. Repository hygiene

- Images and arrays are not committed; `results/` contains JSON/JSONL only.
- `*.npy`, `*.pth`, `*.zip` are gitignored by the repository root; the package never writes them into
  tracked paths.
- Test fixtures are generated in `tmp_path`, never in the repository.
- `results/` artifacts are small (the full demo is < 5 MB) and are committed for the demo only; a full
  Gate E run's `episodes.jsonl` (400 × 6 policies × 3 steps) is ~40 MB and stays out of git, referenced
  by hash in `run_manifest.json`.

## 7. Replication procedure

On a fresh machine:

```bash
python -m pip install -r requirements.txt           # repo root requirements
cd RHAN-Bi-v1
python -m rhan_bi.cli verify                        # schema, hashes, splits
python -m rhan_bi.cli demo --seed 7                 # ~60 s
python -m pytest tests -q
```

To reproduce a specific run, use the `config_hash` and `seed` from its `run_manifest.json`; the CLI
prints both and refuses to write into an existing run directory without `--force`.

## 8. What is not reproducible

- **Across BLAS builds.** Float sums can differ in the last bits; record hashes round to 6 decimals for
  this reason. Metrics are compared with a tolerance of 1e-9 for exactness tests and 1e-6 for reports.
- **Across PyTorch versions** for the DINOv2 path; the features are cached with their model hash, so a
  comparison across versions is refused rather than silently mixed.
- **The confirmatory run's environment** until the pre-registration records it; the manifest is the
  record, and a run without a manifest is not citable.

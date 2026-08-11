# Cognitive Vision Lab v2.0

**Interactive Benchmarking Platform for Human-Like AI Vision**

A professional research platform for the RHAN project — exploring how vision
models perceive, reason about, and ultimately *fail* compared to humans. Every
visualization is built to tell part of that story, not just report accuracy.

## Quick start

```bash
cd cognitive_vision_lab
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

With Docker:

```bash
docker compose up --build
```

## Architecture

```
cognitive_vision_lab/
├── app.py                  # entry point (st.navigation)
├── config.py               # central configuration (paths, constants)
├── pages/                  # 14 pages, one per capability
│   ├── 01_Home.py
│   ├── 02_Model_Zoo.py
│   ├── 03_Interactive_Inference.py
│   ├── 04_Adversarial_Lab.py
│   ├── 05_Attention_Explorer.py
│   ├── 06_Representation_Drift.py
│   ├── 07_GradCAM_and_Saliency.py
│   ├── 08_Human_vs_AI.py
│   ├── 09_RobustBench_Comparison.py
│   ├── 10_RHAN_Architecture.py
│   ├── 11_Benchmark_Results.py
│   ├── 12_Experiment_Manager.py
│   ├── 13_Dataset_Explorer.py
│   └── 14_Report_Generator.py
├── components/             # reusable UI: layout, cards, charts, tables, equations
├── utils/                  # theme, hardware, caching, logging, io, math helpers
├── backend/                # models, attacks, metrics, explainability, embeddings,
│                           # benchmark, datasets, robustbench, rhan_arch, human,
│                           # experiments, reports, state
├── assets/                 # global stylesheet
├── cache/                  # experiment persistence (gitignored)
└── data/                   # local dataset cache (gitignored)
```

## Data sources

The platform is a *reader* of the project's real artifacts:

- `phase5_sdt/results/sdt_results.csv` — per-(ε, system, class) SDT rows
  (Human / CNN / ViT / … curves on page 08).
- `tier1/results/comparison_table.csv` — model comparison table.
- `FINDINGS.md` — curated profiles transcribed into `backend/benchmark.py`,
  so every page works even when CSVs are absent (flagged `source=curated`).
- `checkpoints/` — RHAN checkpoints discovered automatically by the Model Zoo
  (lazy-loaded, profiled on first use).
- `phase3_human_study/` — human psychophysics responses.

## Key conventions

- **Attack metric**: PGD-50, L∞, perturbation clamped *directly in normalized
  space* at ε = 0.031 / 0.062 / 0.094 (Finding-17 convention — never a
  pixel-space conversion).
- **Sensitivity**: macro d′ = Φ⁻¹(HR) − Φ⁻¹(FAR); εthresh = first ε with d′ < 1.0.
- **Honesty**: measured numbers are labeled `measured`; reused curated numbers
  are labeled `curated`. The platform never fabricates results.

## Development

```bash
python -m pytest tests/ -q          # unit tests (metrics, attacks, reports, …)
python -m py_compile app.py         # syntax check
```

Add a page: create `pages/XX_Name.py` (a `render()` function + the
`if st.runtime.exists(): render()` guard) and register it in `app.py`'s
`PAGES` list.

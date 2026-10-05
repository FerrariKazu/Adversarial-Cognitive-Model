# Chapter 26 — Reproducibility Pipeline and Handbook Build System

> *Level 1–2 reading. Complete operational guide for executing the RHAN-NXA reproducibility suite, verifying experimental claims, and compiling this technical handbook into a publication-grade PDF.*

---

## 1. Overview and Core Philosophy

In scientific machine learning, an architecture is only as credible as its reproducibility. The RHAN-NXA research program enforces strict protocols to ensure that every table, metric, parameter count, and documentation asset can be regenerated deterministically from source code and pre-registered configurations.

This chapter provides the complete operational handbook for:
1. Environment configuration and dependency verification.
2. Running the pre-flight gradient isolation and smoke test suite.
3. Executing experimental DAG steps with seed parity.
4. Compiling the complete 27-chapter Technical Handbook into a unified, publication-grade document and PDF.

---

## 2. Environment and Dependency Specification

RHAN-NXA executes on standard Linux environments with PyTorch $\ge 2.0$ and CUDA acceleration.

### Core Dependencies
```bash
# Core Machine Learning Frameworks
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio

# Robustness & Adversarial Evaluation
pip install autoattack robustbench

# Documentation & PDF Compilation Toolchain
sudo apt-get update && sudo apt-get install -y pandoc texlive-xetex librsvg2-bin
# Alternatively: weasyprint or python-markdown
pip install weasyprint markdown PyYAML
```

---

## 3. Pre-Flight Verification Suite

Before launching any training or evaluation run, the system requires running the pre-flight verification script to confirm gradient isolation and interface integrity:

```bash
# Execute pre-flight gradient isolation checks
python3 -m unittest discover -s tests -p "test_*.py"
```

The pre-flight test suite verifies:
1. **$B_t$ Contract**: Confirms that all belief operations execute without exception when $S_t = \text{None}$.
2. **Gradient Flow**: Asserts that during predictor training, gradients do not leak into the backbone, and that during evidential head updates, the predictor remains unperturbed ($|\nabla_W| = 0$).
3. **Numerical Clamps**: Verifies that evidential evidence parameters obey $[\alpha_{\min}, \alpha_{\max}] = [10^{-6}, 10^4]$ and that $\Pi_t \ge 10^{-4}$.
4. **UpdateNet Bounding**: Asserts that $|\Delta z_t| \le 0.1$ across random batch tensors.

---

## 4. Checkpoint Durability and Resumption Parity

All training experiments must support atomic resumption without state loss:

```bash
# Checkpoint structure verification
python3 -c "
from noesis_vision.core.checkpoint import CheckpointManager
print('Checkpoint Manager Loaded Successfully')
"
```

### Resume Protocol
- The checkpoint manager persists model weights, 5-group optimizer states, learning rate schedulers, scalar history metrics, and RNG seeds (PyTorch, NumPy, Python standard library).
- When training resumes from epoch $k$, the rolling validation score at epoch $k$ must match the recorded best score to within $10^{-6}$ numerical tolerance.

---

## 5. Handbook Build Pipeline (`build_rhan_nxa_handbook.py`)

This entire Technical Handbook is generated deterministically from the Markdown source files in `docs/rhan_nxa/book/` and `docs/rhan_nxa/appendices/`.

The build automation script is located at:
`scripts/build_rhan_nxa_handbook.py`

### Build Command
To compile the handbook into a unified Markdown document and publication PDF:

```bash
python3 scripts/build_rhan_nxa_handbook.py
```

### Build Architecture
```
docs/rhan_nxa/book/
  00_Executive_Overview.md
  01_Why_RHAN_NXA.md
  ...
  26_Reproducibility.md
        │
        ├──► scripts/build_rhan_nxa_handbook.py ──► Combined Unified Markdown
        │                                           (RHAN_NXA_Handbook_Unified.md)
docs/rhan_nxa/appendices/                                   │
  A_Tensor_Reference.md                                     │ Pandoc / Weasyprint
  B_Interface_ABCs.md                                       ▼
  C_Port_Table.md                           RHAN_NXA_Technical_Handbook.pdf
```

The script performs the following tasks:
1. **Sequential Assembly**: Reads all 27 chapters in strict numerical order (`00_` through `26_`).
2. **Appendix Integration**: Appends Appendices A, B, and C with standardized hierarchical headers.
3. **Link Normalization**: Resolves local cross-file markdown links into internal document anchors for seamless reading.
4. **Diagram Embedding**: Resolves SVG and raster figure paths to absolute references in `docs/rhan_nxa/figures/`.
5. **PDF Compilation**: Invokes `pandoc` or `weasyprint` with academic typography, syntax highlighting, and an automated table of contents.

---

## 6. Reproducibility Checklist for New Experiments

Every new experimental finding added to the RHAN-NXA corpus must supply:

- [ ] **Exact Configuration File**: A YAML specification stored in `configs/`.
- [ ] **16-Seed Array**: Results evaluated across seeds 0 through 15 with mean and sample standard deviation reported.
- [ ] **Pre-Flight Log**: Zero-gradient leakage confirmed for all frozen parameter groups.
- [ ] **Confound Audit**: Verification that `--enable-sbr` or other rejected flags were not passed accidentally.
- [ ] **Matched Controls**: A parameter-matched and compute-matched control baseline evaluated on the identical hardware and batch size.
- [ ] **Checkpoint Artifact**: Publicly accessible model weights and optimizer states saved with SHA-256 checksums.

---

## 7. Related Chapters and Cross-References

- [Chapter 14 — Multi-Group Gradient Flow](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/14_Gradient_Flow.md): Pre-flight isolation details.
- [Chapter 16 — Training System](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/16_Training_System.md): Checkpoint durability and resume protocols.
- [Chapter 17 — Experimental DAG](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/17_Experimental_DAG.md): Step-by-step DAG execution sequence.
- [Chapter 18 — Evaluation Protocols](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/18_Evaluation.md): Standardized AutoAttack and PGD evaluation scripts.

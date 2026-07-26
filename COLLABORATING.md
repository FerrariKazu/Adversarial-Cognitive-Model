# Onboarding Reading List for RHAN

Read in order: project docs first, then research papers. Each builds on the previous.

## Phase 1 — Core Concepts (Project Docs)

| # | File | Why |
|---|------|-----|
| 1 | `RHANarch.md` | Architecture overview — foveation, recurrence, global attention, predictive coding. The single most important doc. |
| 2 | `IMPORTANT.md` | Short but crucial — explains the two-regime theory (local texture vs global shape) that motivates the entire design. |
| 3 | `rhan_mathematical_proof.pdf` | Formalizes why recurrent feedback + predictive coding bounds adversarial sensitivity. |

## Phase 2 — Model Evolution (Project Docs)

| # | File | Why |
|---|------|-----|
| 4 | `RHAN-history.md` | Traces v2 → v11: what problem each version solved, what broke, what carried forward. |
| 5 | `RHANv11.md` | Deep dive on the current best model — tripartite active inference, foveal/parafoveal streams, generative prior, thermodynamic halting. |

## Phase 3 — Empirical Results (Project Docs)

| # | File | Why |
|---|------|-----|
| 6 | `FINDINGS.md` | All key results: d' curves, εthresh comparisons, human gap analysis. |
| 7 | `tier1/ScientificValidationReport.pdf` | Full validation — sweeps, ablations, baselines. |
| 8 | `phase5_sdt/RESULTS_INTERPRETATION_GUIDE.md` | How to read d' charts, why εthresh matters. |
| 9 | `rhan_epoch_analysis_report.pdf` | Training dynamics — when models learn robustness. |

## Phase 4 — Research Papers

The RHAN design draws from neuroscience, adversarial robustness, and active inference. Read in this order:

### A. Neuroscience Foundations

| Paper | Why |
|-------|-----|
| **Rao & Ballard (1999)** — "Predictive coding in the visual cortex" | The core theoretical framework. Hierarchical predictive coding = feedback carries predictions, feedforward carries errors. RHAN's recurrent feedback implements this. |
| **Olshausen & Field (1996)** — "Emergence of simple-cell receptive fields by sparse coding" | Why V1 learns Gabor-like filters. Motivates RHAN's biologically-inspired front-end. |
| **Hubel & Wiesel (1962)** — Receptive fields of single neurons in cat's striate cortex | The classic V1 orientation selectivity paper. Understand what V1 simple/complex cells do. |
| **Carandini & Heeger (2012)** — "Normalization as a canonical neural computation" | Divisive normalization — RHAN uses this in its feature processing. |
| **Ungerleider & Mishkin (1982)** — "Two cortical visual systems" | Dorsal ("where") vs ventral ("what") streams. RHAN's dual-stream design originates here. |
| **Kietzmann et al. (2019/2026)** — "Recurrent ventral stream" (Nature Communications) | Shows recurrence is critical in the ventral stream. Direct evidence for RHAN's recurrent design. |
| **Lamme & Roelfsema (2000)** — "The distinct modes of vision offered by feedforward and recurrent processing" | Feedforward is fast but shallow; recurrence enables conscious perception and robustness. |

### B. Adversarial Robustness

| Paper | Why |
|-------|-----|
| **Goodfellow et al. (2014)** — "Explaining and harnessing adversarial examples" | The original adversarial example paper. Understand the problem. |
| **Madry et al. (2018)** — "Towards deep learning models resistant to adversarial attacks" | PGD training, the standard defense. RHAN is compared against this. |
| **Athalye et al. (2018)** — "Obfuscated gradients give a false sense of security" | Why gradient masking is dangerous. RHAN's evaluation uses proper checks (PGD-100, AutoAttack). |
| **Croce & Hein (2020)** — "Reliable evaluation of adversarial robustness with AutoAttack" | The standard benchmark. RHAN uses this. |
| **Engstrom et al. (2019)** — "A discussion of 'adversarial examples are not bugs, they are features'" | Adversarial features vs brittle features. Relevant to RHAN's two-regime theory. |

### C. Biologically-Inspired Robustness

| Paper | Why |
|-------|-----|
| **Dapello et al. (2020)** — "VOneNet: Simulating primary visual cortex in DNNs" (NeurIPS) | Shows V1-like front-end improves robustness. RHAN's front-end is inspired by this. |
| **Lotter et al. (2016)** — "Deep predictive coding networks" (PredNet) | Predictive coding in ML. RHAN's generative prior is related. |
| **Yamins & DiCarlo (2016)** — "Using goal-driven deep learning models to understand sensory cortex" | CNNs as models of ventral stream. Framework for comparing brains and models. |
| **Pinto et al. (2008)** — Unsupervised learning of invariant features | Early evidence that biologically-plausible learning yields robust features. |

### D. Active Vision & Attention

| Paper | Why |
|-------|-----|
| **Mnih et al. (2014)** — "Recurrent models of visual attention" (RAM) | The ML version of active vision / foveation. RHAN's foveal sampling extends this. |
| **Itti & Koch (2001)** — "Computational modelling of visual attention" | Saliency-based attention model. Background for RHAN's attention mechanisms. |
| **Vaswani et al. (2017)** — "Attention is all you need" | Transformer self-attention. RHAN uses this for global feature integration. |

## Phase 5 — Broader Context (Project Docs)

| # | File | Why |
|---|------|-----|
| 10 | `deep-research-report.md` | Full literature survey connecting all the papers above to RHAN's design. |
| 11 | `ROADMAP.md` | What's next — planned experiments, ablations, theoretical extensions. |
| 12 | `README.md` | Project-level setup, quickstart, and directory map. |

## Suggested Discussion Topics After Reading

1. Which regime (ε < 0.03 vs ε ≥ 0.05) do you want to work on?
2. Do you buy the predictive-coding-as-defense argument? What experiment would falsify it?
3. v11 has 75.4M params — is the complexity justified? What's the simplest ablation that tests this?
4. The human gap at high ε: is it a data issue, an architecture issue, or a training issue?
5. What's the next RHAN version (v12) and what single mechanism would you add?

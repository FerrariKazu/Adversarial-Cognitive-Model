# Adversarial Cognition Divergence — v4.0

**Research Question:** What makes vision unbreakable — and do machines have it?

*Is the robustness of human vision a matter of architecture, training, memory,
or meaning, and can a machine ever have it?*

A 7-model + human psychophysics study using CIFAR-10, FGSM/PGD/C&W attacks,
and Signal Detection Theory analysis.

**Deadline: Sunday May 17, 2026**

## Model Spectrum

```
BagNet-33 → ResNet-18 → EfficientNet-B0 → Shape-ResNet-50 → CORnet-S → ViT-Small → CLIP ViT-B/32 → Human
(local)                                                     (recurrent)  (global)    (language)       (biological)
```

| Model | Processing Style | Owner | Status |
|-------|------------------|-------|--------|
| BagNet-33 | Pure local patches (33×33) | Eyad | 🔲 Pending |
| ResNet-18 | Local CNN, texture-biased | Mina | ✅ Complete (95.82%) |
| EfficientNet-B0 | Compound scaled CNN | Mina | ✅ Complete (96.81%) |
| Shape-ResNet-50 | Shape-biased training (SIN) | Sandy | ✅ Complete (91.47%) |
| CORnet-S | Recurrent visual cortex (V1→V2→V4→IT) | Youssef + Eyad | 🔲 Pending |
| ViT-Small | Global patch attention | Mina | ✅ Complete (97.80%) |
| CLIP ViT-B/32 | Zero-shot vision-language contrastive | Mariam | 🔲 Pending |

## Scientific Hypotheses

This study tests three core hypotheses about the source of adversarial robustness:

1. **Training objective** — Shape-ResNet vs ResNet-18: does training on stylized images (shape bias) improve robustness over standard texture-biased training?
2. **Biological recurrence** — CORnet-S vs feedforward CNNs: does recurrent feedback processing (as in the primate ventral stream) provide structural defense?
3. **Language grounding** — CLIP vs ViT: does contrastive vision-language pretraining produce more semantically robust representations than pure visual supervision?

## Team

- **Mina** (FerrariKazu) — ResNet + EfficientNet + ViT + full pipeline + human study + Phase 4 + Phase 5
- **Sandy** — Shape-ResNet-50 + final report + presentation slides
- **Youssef + Eyad** — CORnet-S
- **Eyad** — BagNet-33 + texture analysis
- **Mariam** — CLIP ViT-B/32

## Core Analysis Results (4/7 Models)

### PGD Accuracy Collapse
| Epsilon | ResNet-18 | ViT-Small | EfficientNet-B0 | ShapeResNet | Human |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 95.82% | 97.80% | 96.81% | 91.47% | 74.25% |
| 0.01 | 75.57% | 55.17% | 1.14% | 18.11% | N/A |
| 0.05 | 2.84% | 8.80% | 0.00% | 0.01% | 69.00% |
| 0.10 | 0.20% | 2.78% | 0.00% | 0.00% | 59.25% |
| 0.20 | 0.02% | 1.12% | 3.62% | 0.00% | 64.25% |
| 0.30 | 0.00% | 0.58% | 16.49% | 0.00% | 60.25% |

*Note: EfficientNet uses BIM (PGD without random start) due to gradient explosion at 224×224. It collapses to 0.93% at ε=0.01.*

### Signal Detection Summary ($d'$)
| Epsilon | ResNet $d'$ | ViT $d'$ | EffNet $d'$ | Human $d'$ |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 4.426 | 4.931 | 4.642 | 2.694 |
| 0.01 | 2.345 | 1.120 | -1.142 | 2.650 |
| 0.05 | -0.771 | -0.154 | -1.879 | 2.544 |
| 0.10 | -1.707 | -0.909 | -1.526 | 2.071 |

**Headline Finding:** At $\epsilon=0.05$, all CNN models drop below the perceptual threshold ($d' < 1.0$), while human observers maintain high sensitivity. EfficientNet exhibits the most extreme collapse at low epsilon, while ViT shows a slight robustness advantage over ResNet at $\epsilon=0.05$.

## Environment Setup
```bash
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Additional model-specific installs:
pip install git+https://github.com/dicarlolab/CORnet.git    # CORnet-S
pip install git+https://github.com/openai/CLIP.git           # CLIP ViT-B/32
```

## Running Evaluation
```bash
python3 phase2_attacks/eval_quick.py
```
*Note: All analysis scripts enforce a maximum batch size of 64 and periodic cache clearing to fit within 8GB VRAM.*

## Project Phases
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Model training (all 7) | 4/7 Complete |
| Phase 2 | Adversarial attack generation | 4/7 Complete |
| Phase 3 | Human psychophysics study | ✅ Complete (21 participants) |
| Phase 4 | 7-model divergence analysis | 4/7 Complete |
| Phase 5 | Signal Detection Theory (SDT) | 4/7 Complete |

## Active Branches
| Branch | Owner | Purpose |
|--------|-------|---------|
| `main` | Mina | Stable release |
| `phase/1-cornets` | Youssef + Eyad | CORnet-S model + training |
| `phase/1-clip` | Mariam | CLIP ViT-B/32 zero-shot wrapper |
| `phase/1-bagnet` | Eyad | BagNet-33 model + training |

## Repository Map & Directory Layout

The codebase has evolved from the initial CIFAR-10 baseline study into the **RHAN-NXA** (Recurrent Hierarchical Attention Network — Noesis eXperimental Architecture) cognitive vision system. Below is an exhaustive structural map of all directories and key files across the project:

```text
.
├── README.md                           # Main project overview, scientific hypotheses, and benchmark results
├── CONTRIBUTING.md                     # Contributor guidelines, git branching conventions, and PR policies
├── ROADMAP.md                          # Multi-generation project roadmap (Gen-1 through Gen-3/4)
├── TRAINING_HEALTH_REPORT.md           # Training diagnostic reports (gradient norms, loss stability, NaN checks)
├── T4x2.ipynb                          # Dual-GPU (2x T4) Colab/Kaggle interactive experimentation notebook
├── app.py                              # Streamlit/Gradio web demo for interactive model visualization and attack inspection
├── requirements.txt                    # Python runtime dependencies (PyTorch, torchvision, timm, torchattacks, scipy)
├── RHANarch.pdf                        # Architectural diagram of the Recurrent Hierarchical Attention Network
│
├── gen2_foundation/                    # Generation-2 foundation model package and components
│   ├── __init__.py                     # Package initialization and public API exports
│   ├── flags.py                        # Feature flag system and experiment registry toggles
│   ├── gist.py                         # Global gist encoder (DINOv2 patch projection and CLS prior token)
│   ├── precision.py                    # Dynamic precision weighting and observation reliability heads
│   ├── recipe.py                       # Layer-wise parameter grouping and AdamW optimizer schedule builder
│   ├── ema.py                          # Exponential Moving Average (EMA) shadow model weights tracker
│   ├── eot.py                          # Expectation Over Transformation (EOT-PGD) stochastic attack defense
│   ├── update_net_v2.py                # Bounded recurrent belief update network (GRU/residual state dynamics)
│   ├── AUDIT.md                        # Package code-level audit and integrity verification notes
│   ├── EXPERIMENT_REGISTRY.md          # Preregistered Gen-2 experimental arms and hypothesis ledger
│   └── tests/                          # Internal unit test suite for the gen2_foundation package
│
├── noesis_vision/                      # NOESIS cognitive architecture and biological perceptual loop
│   ├── RHAN_NXA/                       # Master architectural specifications, plans, and forensic audit reports
│   │   ├── docs/                       # Technical design specs (gen3_spec.md, revision1.md, plan1.md, plan2.md)
│   │   └── MASTER_PLAN.md              # Historical NOESIS master plan and locked phase contracts
│   ├── core/                           # Core cognitive orchestrator and multi-step perceptual recurrent loop
│   ├── models/                         # Neural backbones, foveated glimpse crop samplers, and vision transformers
│   ├── gaze/                           # Adaptive Information Seeking (AIS) candidate generator and gaze policy
│   ├── beliefs/                        # Latent belief state representations (z_t) and spatial memory integration
│   ├── predictive_coding/              # Top-down feature predictor heads and hierarchical prediction error computation
│   └── uncertainty/                    # Dirichlet evidential heads, softmax entropy, and calibration modules
│
├── training/                           # Training drivers, adversarial curriculum, and persistence pipelines
│   ├── train_gen2_foundation.py        # Canonical Gen-2 / ff_112 training driver with TRADES-PGD4 curriculum
│   ├── train_generation1_foundation.py # Gen-1 foundation training script across the 6-phase ladder
│   ├── curriculum_gen2.py              # Per-epoch epsilon radius and TRADES beta curriculum schedule
│   ├── adv_curriculum.py               # Adversarial attack parameter ramps and epsilon step-size scaling
│   ├── hf_persistence.py               # Automated HuggingFace checkpoint upload, resume, and sync handlers
│   ├── stage2_pipeline.py              # Multi-phase stage transition state machine
│   ├── measure_training_health.py      # Real-time gradient norm monitor, loss landscape tracker, and NaN guards
│   ├── run_report.py                   # Generation of human-readable training summary reports
│   └── run_report_jsonl.py             # Machine-readable per-epoch JSONL telemetry logging
│
├── cloud_setup/                        # Automated dispatch scripts for Colab and Kaggle cloud environments
│   ├── colab_j1_foundation.py          # Colab J1 dispatch: Trains full Gen-2 foundation ladder (S0–S7)
│   ├── colab_j2_eval_train.py          # Colab J2 dispatch: Matched-eps eval, ff_56 control, and ff_112 multi-seed
│   ├── Kaggle_J1_FOUNDATION.py         # Kaggle J1 dispatch: Adapted for /kaggle/working and Kaggle Secrets
│   ├── Kaggle_J2_EVAL_TRAIN.py         # Kaggle J2 dispatch: Parallel matched-eps eval and ff_112 seed training
│   ├── colab_v11_3seed_protocol.py     # Colab multi-seed variance protocol for historical models
│   ├── kaggle_v11_3seed_protocol.py    # Kaggle multi-seed variance protocol for historical models
│   └── scripts/                        # Remote cloud validation and syntax verification utilities
│
├── evaluation/                         # Evaluation datasets, corruptions, and benchmark protocols
│   ├── clean_and_robust.py             # Standard evaluation runner for clean and Linf PGD accuracy
│   ├── geirhos_loader.py               # Geirhos cue-conflict dataset loader (human vs machine shape/texture bias)
│   ├── shape_texture_bias.py           # Signal Detection Theory d' calculation for shape vs texture classification
│   ├── imagenet100_loader.py           # ImageNet-100 dataset loader with normalization and caching
│   ├── imagenet_c_loader.py            # ImageNet-C benchmark loader for common synthetic image corruptions
│   └── compactness_report.py           # Model parameter count, FLOPs, and latency efficiency profiler
│
├── scripts/                            # Operational automation, data preparation, and gate verification
│   ├── prepare_imagenet100.py          # Automated ImageNet-100 dataset downloader and directory structure bootstrap
│   ├── verify_run_complete.py          # Validation script asserting full completion of scheduled training epochs
│   ├── freeze_run_manifest.py          # Cryptographic SHA-256 fingerprinting of model checkpoints and datasets
│   ├── sbr0_gate.py                    # Stage-boundary validation assertion script for gating progression
│   ├── build_noesis_pdf.py             # Automated LaTeX compilation of project documentation into formal PDFs
│   └── build_rhan_nx_report.py         # Automated markdown-to-PDF report generator for experimental results
│
├── tests/                              # Pytest test suite enforcing system invariants and regression safety
│   ├── conftest.py                     # Global pytest fixtures, mock data loaders, and device configuration
│   ├── test_adv_curriculum.py          # Unit tests verifying TRADES curriculum bounds and step-size formulas
│   ├── test_ais_v2_gaze_policy.py      # Tests for AIS candidate gaze selection and temperature calibration
│   ├── test_evidential_gradient_flow.py# Assertion test verifying non-zero gradients through Dirichlet uncertainty
│   ├── test_hf_persistence.py          # Mock tests for HuggingFace upload, download, and resume mechanisms
│   ├── test_belief_state_composition.py# Shape and consistency tests for latent belief vector concatenation
│   └── test_center_bias_measurement.py # Measures spatial fixation distribution to detect center-bias collapse
│
├── docs/                               # Architectural documentation, forensic reports, and research papers
│   ├── REPOSITORY_MAP.md               # Exhaustive 1,600-line descriptive inventory of the entire repository
│   ├── ARCHITECTURE.md                 # Detailed mathematical and cognitive architecture specifications
│   ├── FORENSIC_REPORT_NXA_GENERATION1.md # Forensic analysis of Gen-1 mechanisms and performance anomalies
│   ├── NOESIS_FOUNDATION.md            # Cognitive vision foundations, biological inspiration, and design philosophy
│   └── historical/                     # Archived stage reports, early proposal drafts, and pre-registrations
│
├── report/                             # Experimental results, JSON artifacts, and scientific reports
│   ├── GEN1_RESULTS_MASTER.md          # Master reference table of Generation-1 experimental results
│   ├── rhan_nx_generation1_report.md   # Comprehensive experimental audit report for Generation-1
│   ├── empirical_report.json           # Machine-readable performance database across all models and phases
│   ├── audit_rev1/                     # Revision 1 forensic evaluation artifacts (matched-eps JSON reports)
│   └── eval/                           # Checkpoint evaluation summaries and AutoAttack metric dumps
│
├── figures/                            # Publication-grade figures and visualizations (v1)
├── figures_v2/                         # Extended visualizations: attention maps, biological comparisons, loss surfaces
├── figures_v3/                         # Multi-theme (dark/light) publication figures in vector SVG and raster PNG
│
├── archive/                            # Archived legacy stages and previous experimental pipelines
│   ├── gen-1-cifar12/                  # Original May 2026 CIFAR-10 7-model + human psychophysics comparison
│   ├── rhan-v1-v7/                     # Early RHAN recurrent prototypes (v1 through v7)
│   ├── stl10-scaleup/                  # STL-10 self-supervised pretraining and pseudo-labeling experiments
│   └── historical-report/              # Prior LaTeX project manuscripts, draft papers, and workshop submissions
│
├── checkpoints/                        # Model weights directory (tracked JSON manifests, local cache, and symlinks)
├── cloud/                              # Historical cloud cluster configurations, Slurm scripts, and setup specs
├── config/                             # Central YAML configuration files for models, training, and attack bounds
├── data/                               # Local data directory placeholder (cached datasets and split indices)
├── diagnostics/                        # Diagnostic logs, memory profiling traces, and GPU utilization dumps
└── runs/                               # Experiment output runs containing TensorBoard event logs and metrics
```

## References
1. Brendel, W., & Bethge, M. (2019). Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet.
2. Geirhos, R. et al. (2019). ImageNet-trained CNNs are biased towards texture.
3. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples.
4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
5. Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
6. Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics.
7. Kubilius, J. et al. (2019). Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs (CORnet).
8. Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP).

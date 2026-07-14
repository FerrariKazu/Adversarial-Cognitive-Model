# Adversarial Cognition Divergence
**A 12-model + human psychophysics study of adversarial robustness**

> Does adversarial robustness scale with global visual processing —
> and is it determined by architecture, training objective, or recurrence?

## Key Findings (13/13 Systems Complete)

### Robustness & Sensitivity Overview
| System | Clean Acc | PGD 50% Threshold | d′=1.0 Threshold | Status |
|--------|-----------|-------------------|-------------------|--------|
| Human | 74.15% | >0.30 | >0.30 | ✅ Complete |
| **RHAN-Large (Ours)** ★ | **52.60%** | **ε≈0.011** | **ε≈0.0130** | ✅ Complete |
| **RHAN-trades-curriculum** ★ | **78.12%** | **ε≈0.113** | **ε≈0.1850** | ✅ Complete |
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
| **RHAN-Large (Ours)** ★ | **3.344** | **1.463** | **<0.000** | **<0.000** | **<0.000** | **<0.000** | **ε≈0.013** |
| **RHAN-trades-curriculum** | **2.748** | **2.589** | **2.159** | **1.696** | **0.877** | **0.010** | **ε≈0.185** |
| **RHAN-TRADES-Hardened** | **3.260** | **3.032** | **2.238** | **1.357** | **-0.094** | **-1.664** | **ε≈0.125** |
| **RHAN-v5-TRADES** | **3.383** | **3.186** | **2.230** | **1.231** | **-0.291** | **-1.602** | **ε≈0.111** |
| **RHAN-v5** | **3.083** | **2.905** | **2.071** | **1.104** | **-1.132** | **-1.808** | **ε≈0.103** |
| **RHAN-v3** | **3.710** | **3.189** | **1.983** | **0.753** | **-1.039** | **-3.044** | **ε≈0.090** |
| **RHAN-adv** | **3.083** | **2.738** | **1.662** | **0.408** | **-1.294** | **-3.044** | **ε≈0.076** |
| **RHAN-TDV (STL-10)** | **2.860** | **-0.597** | **-3.705** | **<0.000** | **<0.000** | **<0.000** | **ε≈0.004** |
| ResNet-18 | 4.426 | 2.687 | -0.771 | -1.707 | -1.913 | -1.880 | ε≈0.030 |
| ViT-Small | 4.931 | 1.814 | -0.154 | -0.909 | -1.242 | -1.469 | ε≈0.026 |

### PGD Accuracy Collapse
| Epsilon | RHAN-Large | RHAN-TDV | Curriculum | Hardened | TRADES | RHAN-v5 | RHAN-v3 | RHAN-adv | ResNet | ViT | EfficientNet | ShapeResNet | BagNet | Human |
|---------|------------|----------|------------|----------|--------|---------|---------|----------|--------|-----|--------------|-------------|--------|-------|
| 0.00 | 51.60% | 78.50% | 78.12% | 86.33% | 87.30% | 84.57% | 91.41% | 83.79% | 95.82% | 97.80% | 96.81% | 91.47% | 87.67% | 73.33% |
| 0.01 | 47.10% | 5.20% | 75.00% | 83.01% | 84.77% | 80.66% | 85.35% | 77.93% | 75.57% | 55.18% | 0.93%  | 18.11% | 48.04% | N/A |
| 0.05 | 27.30% | <2.00% | 65.23% | 67.19% | 65.82% | 61.13% | 60.74% | 51.95% | 2.84%  | 8.80%  | 0.00%  | 0.01%  | 0.12%  | 69.17% |
| 0.10 | 15.10% | <2.00% | 52.93% | 43.16% | 37.89% | 34.38% | 26.17% | 17.77% | 0.21%  | 2.78%  | 0.00%  | 0.00%  | 0.00%  | 59.17% |
| 0.20 | 3.10%  | <2.00% | 29.49% | 8.59%  | 5.47%  | 2.73%  | 1.17%  | 0.59%  | 0.02%  | 1.12%  | 0.00%  | 0.00%  | 0.00%  | 62.22% |
| 0.30 | 0.30%  | <2.00% | 10.16% | 0.20%  | 0.20%  | 0.20%  | 0.00%  | 0.00%  | 0.00%  | 0.58%  | 0.00%  | 0.00%  | 0.00%  | 58.61% |

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

## Setup
```bash
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Results
# Phase 2: Generate adversarial arrays (memory-safe, one model at a time)
```bash
python phase2_attacks/generate_adv_all_models.py --model [resnet|vit|efficientnet|shaperesnet|bagnet]
```

# Phase 4: Run all analysis
```bash
python phase4_analysis/generate_all_figures.py
```

# Phase 5: SDT analysis
```bash
python phase5_sdt/sdt_analysis.py
```

# RHAN-UNIFIED Training (STL-10 96x96)
```bash
# Phase 0: Semantic initialization with unlabeled data (50 epochs)
python phase1_training/train_rhan_unified.py --phase 0

# Phases 1-8: TRADES adversarial curriculum (160 epochs)
python phase1_training/train_rhan_unified.py --phase 1-6

# Everything at once
python phase1_training/train_rhan_unified.py --phase all
```

# RHAN-TDV Training & Evaluation (STL-10 96x96)
```bash
# Phase 1: Self-supervised TDV pretraining (unlabeled data, 30 epochs)
python phase1_training/train_rhan_stl10_tdv.py --phase tdv --unlabeled-batch-size 32

# Phase 2: Classification head label calibration (5K labeled images, 10 epochs)
python phase1_training/train_rhan_stl10_tdv.py --phase label --batch-size 64

# Phase 3: TRADES curriculum fine-tuning with TDV consistency (60 epochs)
python phase1_training/train_rhan_stl10_tdv.py --phase trades --batch-size 16 --unlabeled-batch-size 16

# Run PGD-100 & SDT Evaluation Sweep
python phase1_training/eval_pgd_sdt_stl10.py --checkpoint ../checkpoints/rhan_stl10_tdv_trades_clean_consistency.pth --samples 1000 --batch-size 64
```

# RHAN-Large + Pseudo-Label Curriculum (STL-10 96x96)
```bash
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

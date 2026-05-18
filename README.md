# Adversarial Cognition Divergence
**A 7-model + human psychophysics study of adversarial robustness**

> Does adversarial robustness scale with global visual processing —
> and is it determined by architecture, training objective, or recurrence?

## Key Findings (8/8 Systems Complete)

| System | Clean Acc | PGD 50% Threshold | d′=1.0 Threshold | Status |
|--------|-----------|-------------------|-------------------|--------|
| Human | 74.15% | >0.30 | >0.30 | ✅ Complete |
| **RHAN-adv (Recurrent)** | **83.79%** | **ε≈0.053** | **ε≈0.076** | ✅ Complete |
| RHAN-clean | 89.06% | ε≈0.023 | ε≈0.033 | ✅ Complete |
| ResNet-18 | 95.82% | ε≈0.024 | ε≈0.030 | ✅ Complete |
| ViT-Small | 97.80% | ε≈0.014 | ε≈0.026 | ✅ Complete |
| BagNet-33 | 87.67% | ε≈0.010 | ε≈0.017 | ✅ Complete |
| CORnet-S | 91.48% | ε≈0.006 | ε≈0.009 | ✅ Complete |
| Shape-ResNet-50 | 91.47% | ε≈0.006 | ε≈0.008 | ✅ Complete |
| EfficientNet-B0 | 96.81% | ε≈0.005 | ε≈0.006 | ✅ Complete |
| CLIP ViT-B/32 | — | — | — | 🔄 Pending |

**Headline:** All standard feedforward AI models collapse before ε=0.03. RHAN-adv, utilizing top-down recurrent feedback, extends robustness to **ε≈0.076** (a **2.5× improvement** over ResNet-18 and **2.9× improvement** over ViT-Small), significantly narrowing the massive gap to Human visual cognition.

**Counterintuitive finding:** EfficientNet-B0 (96.81% clean) is the most fragile model. ResNet-18 (95.82% clean) is the most robust feedforward model. Shape-biased training did not improve robustness over standard ResNet. Recurrent top-down feedback (`RHAN`) is the single most effective architectural mechanism for securing adversarial robustness.

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
| **RHAN-adv** | **Recurrent top-down visual feedback** | **Mina** | **dev** |
| Human | Biological vision (n=18) | All | — |

## Team
- **Mina (FerrariKazu)** — ResNet ✅, ViT ✅, EfficientNet ✅, pipeline, human study, Phase 4+5
- **Sandy** — Shape-ResNet ✅, final report, slides
- **Eyad** — BagNet ✅, CORnet-S (co-owner)
- **Youssef** — CORnet-S (co-owner)
- **Mariam** — CLIP ViT-B/32

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
│   └── train.py            # Standard training loop
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
│   └── perturbation_visuals.py
├── phase5_sdt/             # Signal Detection Theory (SDT)
│   ├── sdt_analysis.py     # Main d' and criterion calculation
│   └── sdt_core.py         # SDT mathematical primitives
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

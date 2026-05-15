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

## Repository Structure
```text
.
├── config/                 # Attack and training configuration (YAML)
├── phase1_training/        # Model architectures and training scripts
├── phase2_attacks/         # FGSM/PGD attack generation and eval
├── phase3_human_study/     # Human baseline data and stimuli export
├── phase4_analysis/        # Divergence curves and heatmaps
├── phase5_sdt/             # Signal Detection Theory calculation
└── utils/                  # Metrics and logging utilities
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

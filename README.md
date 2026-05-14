# Adversarial Cognition Divergence

**Research Question:** What makes vision unbreakable — and do machines have it?

A 7-model + human psychophysics study using CIFAR-10, FGSM/PGD/C&W attacks, 
and Signal Detection Theory analysis.

## Model Spectrum
| Model | Processing Style | Owner | Status |
|-------|------------------|-------|--------|
| ResNet-18 | Local CNN, texture-biased | Mina | ✅ Complete (95.82%) |
| ViT-Small | Global patch attention | Mina | ✅ Complete (97.80%) |
| EfficientNet-B0 | Compound scaled CNN | Youssef | ✅ Complete (96.81%) |
| Shape-ResNet-50 | Shape-biased training (SIN) | Sandy | ✅ Complete (91.47%) |
| BagNet-33 | Pure local patches (33×33) | Eyad | Pending |
| CORnet-S | Recurrent visual cortex model | Mariam | Pending |
| CLIP ViT-B/32 | Zero-shot multi-modal contrastive | Mina | Pending |



## Team
- Mina (FerrariKazu) — ResNet, ViT, CLIP, pipeline architecture, human study
- Sandy — Shape-ResNet-50, final report, presentation slides
- Youssef — EfficientNet-B0
- Eyad — BagNet-33
- Mariam — CORnet-S

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
```

## Running Evaluation
To run the memory-safe 3-model comparison:
```bash
python3 phase2_attacks/eval_quick.py
```
*Note: All analysis scripts now enforce a maximum batch size of 64 and periodic cache clearing to fit within 8GB VRAM.*

## Project Phases
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Model training (all 7) | 4/7 Complete |
| Phase 2 | Adversarial attack generation | 4/7 Complete |
| Phase 3 | Human psychophysics study | ✅ Complete (21 participants) |
| Phase 4 | 7-model divergence analysis | 4/7 Complete |
| Phase 5 | Signal Detection Theory (SDT) | 4/7 Complete |

## Repository Structure
```text
.
├── phase1_training         # Model architectures and training scripts
├── phase2_attacks          # FGSM/PGD attack generation and eval
├── phase3_human_study      # Human baseline data and stimuli export
├── phase4_analysis         # Divergence curves and heatmaps
├── phase5_sdt              # Signal Detection Theory calculation
└── utils                   # Metrics and logging utilities
```

## References
1. Brendel, W., & Bethge, M. (2019). Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet.
2. Geirhos, R. et al. (2019). ImageNet-trained CNNs are biased towards texture.
3. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples.
4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
5. Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
6. Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics.

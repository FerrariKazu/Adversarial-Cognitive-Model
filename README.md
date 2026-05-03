# Adversarial Cognition Divergence

**Research Question:** Does adversarial robustness scale with global visual 
processing — and is it determined by architecture or training objective?

A 5-model + human psychophysics study using CIFAR-10, FGSM/PGD/C&W attacks, 
and Signal Detection Theory analysis.

## Model Spectrum
| Model | Processing Style | Owner | Status |
|-------|------------------|-------|--------|
| BagNet-33 | Pure local patches (33×33) | Eyad | Pending |
| ResNet-18 | Local CNN, texture-biased | Mina | ✅ Complete (95.82%) |
| EfficientNet-B0 | Compound scaled CNN | Youssef | Pending |
| Shape-ResNet-50 | Shape-biased training (SIN) | Sandy | Pending |
| ViT-Small | Global patch attention | Mina | In Progress |

## Team
- Mina (FerrariKazu) — ResNet ✅, ViT, pipeline architecture, human study, Phase 4+5
- Sandy — Shape-ResNet-50, final report, presentation slides
- Youssef — EfficientNet-B0
- Eyad — BagNet-33

## Environment Setup
```bash
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Additional installs per model
- **ViT**: `pip install timm`
- **BagNet**: `pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git`
- **Shape-ResNet**: download `resnet50_trained_on_SIN.model` (see `docs/`)

## Project Phases
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Model training (all 5) | ResNet ✅, others pending |
| Phase 2 | Adversarial attack generation | ResNet ✅, others pending |
| Phase 3 | Human psychophysics study | Scripts done, form building |
| Phase 4 | 5-model divergence analysis | Pending |
| Phase 5 | Signal Detection Theory (SDT) | Pending |

## Repository Structure
```text
.
├── config
├── data
├── docs
├── phase1_training
│   ├── checkpoints
│   ├── model_bagnet.py
│   ├── model_efficientnet.py
│   ├── model_shaperesnet.py
│   └── model_vit.py
├── phase2_attacks
│   ├── adv_images
│   │   ├── bagnet
│   │   ├── efficientnet
│   │   ├── resnet
│   │   ├── shaperesnet
│   │   └── vit
│   └── generate_adv_all_models.py
├── phase3_human_study
│   ├── data
│   └── stimuli
├── phase4_analysis
│   ├── cross_model_compare.py
│   └── figures
│       ├── bagnet
│       ├── combined
│       ├── efficientnet
│       ├── resnet
│       ├── shaperesnet
│       └── vit
├── phase5_sdt
│   ├── figures
│   └── results
├── requirements.txt
├── runs
└── utils
```

## Documentation
Full project documentation: `docs/ACD_Project_Documentation_v3.pdf`

## References
1. Brendel, W., & Bethge, M. (2019). Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet.
2. Geirhos, R. et al. (2019). ImageNet-trained CNNs are biased towards texture.
3. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples.
4. He, K. et al. (2016). Deep Residual Learning for Image Recognition.
5. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
6. Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
7. Madry, A. et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks.
8. Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks.
9. Green, D. M., & Swets, J. A. (1966). Signal detection theory and psychophysics.
10. Macmillan, N. A., & Creelman, C. D. (2005). Detection theory: A user's guide.
11. Ilyas, A. et al. (2019). Adversarial Examples Are Not Bugs, They Are Features.

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

The codebase has evolved from the initial CIFAR-10 baseline study into the **RHAN-NXA** (Recurrent Hierarchical Attention Network — Noesis eXperimental Architecture) cognitive vision system. Below is a complete, exhaustive structural map of all directories and files across the repository, annotated with their functional role:

```text
.
├── .gitattributes                               # Git line endings (LF), diff drivers, and binary asset configuration
├── .gitignore                                   # Ignored paths: virtual environments, PyTorch checkpoint weights, compiled binaries
├── .mailmap                                     # Developer name and email address canonicalization mapping
├── .verify_samples/                             # Ground-truth reference clean CIFAR-10 test crops across 10 classes used for verifying attack perturbations
│   ├── airplane/                                    # Reference CIFAR-10 airplane samples (00-19) for verifying attack perturbations on rigid aircraft contours
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 airplane reference crop (sample 19) verifying attack perturbation step generation
│   ├── bird/                                        # Reference CIFAR-10 bird samples (00-19) for auditing high-frequency texture attacks on plumage
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 bird reference crop (sample 19) verifying attack perturbation step generation
│   ├── car/                                         # Reference CIFAR-10 automobile samples (00-29) for auditing structural edge deformation under Linf attacks
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 18) verifying attack perturbation step generation
│   │   ├── sample_19.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 19) verifying attack perturbation step generation
│   │   ├── sample_20.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 20) verifying attack perturbation step generation
│   │   ├── sample_21.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 21) verifying attack perturbation step generation
│   │   ├── sample_22.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 22) verifying attack perturbation step generation
│   │   ├── sample_23.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 23) verifying attack perturbation step generation
│   │   ├── sample_24.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 24) verifying attack perturbation step generation
│   │   ├── sample_25.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 25) verifying attack perturbation step generation
│   │   ├── sample_26.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 26) verifying attack perturbation step generation
│   │   ├── sample_27.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 27) verifying attack perturbation step generation
│   │   ├── sample_28.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 28) verifying attack perturbation step generation
│   │   └── sample_29.png                                # Ground-truth clean CIFAR-10 car reference crop (sample 29) verifying attack perturbation step generation
│   ├── cat/                                         # Reference CIFAR-10 feline samples (00-19) for testing feature distortion on non-rigid natural shapes
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 cat reference crop (sample 19) verifying attack perturbation step generation
│   ├── deer/                                        # Reference CIFAR-10 deer samples (00-19) for inspecting silhouette vs background artifact disruption
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 deer reference crop (sample 19) verifying attack perturbation step generation
│   ├── dog/                                         # Reference CIFAR-10 canine samples (00-19) for auditing attack-induced semantic shift towards conflicting classes
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 dog reference crop (sample 19) verifying attack perturbation step generation
│   ├── horse/                                       # Reference CIFAR-10 equine samples (00-19) for evaluating limb boundary preservation under gradient attacks
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 horse reference crop (sample 19) verifying attack perturbation step generation
│   ├── monkey/                                      # Subsystem directory: monkey/
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 monkey reference crop (sample 19) verifying attack perturbation step generation
│   ├── ship/                                        # Reference CIFAR-10 watercraft samples (00-19) for auditing horizon line and geometric hull integrity
│   │   ├── sample_00.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 00) verifying attack perturbation step generation
│   │   ├── sample_01.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 01) verifying attack perturbation step generation
│   │   ├── sample_02.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 02) verifying attack perturbation step generation
│   │   ├── sample_03.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 03) verifying attack perturbation step generation
│   │   ├── sample_04.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 04) verifying attack perturbation step generation
│   │   ├── sample_05.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 05) verifying attack perturbation step generation
│   │   ├── sample_06.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 06) verifying attack perturbation step generation
│   │   ├── sample_07.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 07) verifying attack perturbation step generation
│   │   ├── sample_08.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 08) verifying attack perturbation step generation
│   │   ├── sample_09.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 09) verifying attack perturbation step generation
│   │   ├── sample_10.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 10) verifying attack perturbation step generation
│   │   ├── sample_11.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 11) verifying attack perturbation step generation
│   │   ├── sample_12.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 12) verifying attack perturbation step generation
│   │   ├── sample_13.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 13) verifying attack perturbation step generation
│   │   ├── sample_14.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 14) verifying attack perturbation step generation
│   │   ├── sample_15.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 15) verifying attack perturbation step generation
│   │   ├── sample_16.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 16) verifying attack perturbation step generation
│   │   ├── sample_17.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 17) verifying attack perturbation step generation
│   │   ├── sample_18.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 18) verifying attack perturbation step generation
│   │   └── sample_19.png                                # Ground-truth clean CIFAR-10 ship reference crop (sample 19) verifying attack perturbation step generation
│   └── truck/                                       # Reference CIFAR-10 freight vehicle samples (00-29) for testing planar surface robustness against adversarial noise
│       ├── sample_00.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 00) verifying attack perturbation step generation
│       ├── sample_01.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 01) verifying attack perturbation step generation
│       ├── sample_02.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 02) verifying attack perturbation step generation
│       ├── sample_03.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 03) verifying attack perturbation step generation
│       ├── sample_04.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 04) verifying attack perturbation step generation
│       ├── sample_05.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 05) verifying attack perturbation step generation
│       ├── sample_06.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 06) verifying attack perturbation step generation
│       ├── sample_07.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 07) verifying attack perturbation step generation
│       ├── sample_08.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 08) verifying attack perturbation step generation
│       ├── sample_09.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 09) verifying attack perturbation step generation
│       ├── sample_10.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 10) verifying attack perturbation step generation
│       ├── sample_11.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 11) verifying attack perturbation step generation
│       ├── sample_12.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 12) verifying attack perturbation step generation
│       ├── sample_13.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 13) verifying attack perturbation step generation
│       ├── sample_14.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 14) verifying attack perturbation step generation
│       ├── sample_15.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 15) verifying attack perturbation step generation
│       ├── sample_16.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 16) verifying attack perturbation step generation
│       ├── sample_17.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 17) verifying attack perturbation step generation
│       ├── sample_18.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 18) verifying attack perturbation step generation
│       └── sample_19.png                                # Ground-truth clean CIFAR-10 truck reference crop (sample 19) verifying attack perturbation step generation
├── CONTRIBUTING.md                              # Contributor instructions, branch policy, testing requirements, and PR workflow
├── README.md                                    # Master repository overview, empirical benchmark findings, and exhaustive repo map
├── RHANarch.pdf                                 # Architectural vector schematic illustrating the recurrent foveation and belief updating loop
├── ROADMAP.md                                   # Multi-generation research roadmap defining milestones from CIFAR-10 to ImageNet-100
├── ROADMAP.md:Zone.Identifier                   # Windows filesystem download zone identifier metadata for ROADMAP.md
├── T4x2.ipynb                                   # Dual-T4 multi-GPU training and experimentation interactive notebook
├── TRAINING_HEALTH_REPORT.md                    # Diagnostic criteria and thresholds for loss stability, gradient norms, and NaN guards
├── acd_research_dashboard.html:Zone.Identifier  # Windows filesystem download zone identifier metadata for acd_research_dashboard.html
├── app.py                                       # Interactive web application for visualizing model predictions, glimpses, and adversarial attacks
├── archive/                                     # Archived legacy stages, historical models, and previous experimental pipelines
│   ├── gen-1-cifar12/                               # Archived May 2026 CIFAR-10 7-model + human psychophysics comparison
│   │   └── phase1_training/                             # CIFAR-10 model training definitions (ResNet, ViT, EfficientNet, early RHAN)
│   │       ├── alignment_head.py                            # Python execution module and utility: alignment_head.py
│   │       ├── checkpoint_utils.py                          # Python execution module and utility: checkpoint_utils.py
│   │       ├── dataset.py                                   # Python execution module and utility: dataset.py
│   │       ├── dataset_vit.py                               # Custom PyTorch dataloader with ViT-compatible normalization and augmentations
│   │       ├── eval_pgd_sdt_stl10.py                        # Combined PGD robustness and Signal Detection Theory evaluation on STL-10 images
│   │       ├── eval_pgd_stl10.py                            # PGD robustness evaluation driver on STL-10 96x96 resolution test sets
│   │       ├── eval_pgd_v8.py                               # PGD evaluation script specifically tailored for the RHAN v8 architecture
│   │       ├── eval_pseudolabel_full.py                     # Full evaluation harness measuring accuracy gains from semi-supervised pseudo-labeling
│   │       ├── eval_quick.py                                # Quick validation script computing clean and PGD-10 accuracy across test splits
│   │       ├── eval_rhan_v10.py                             # Robustness evaluation runner benchmarking RHAN v10 predictive coding models
│   │       ├── eval_rhan_v11.py                             # Robustness evaluation runner benchmarking RHAN v11 structured belief models
│   │       ├── eval_sprint2_comparison.py                   # Multi-model sprint comparison evaluating ResNet, ViT, and early RHAN variants
│   │       ├── eval_static_baseline.py                      # Evaluates static feedforward CNN and ViT baselines under identical attack settings
│   │       ├── eval_trades_baseline.py                      # Evaluates feedforward models trained under TRADES loss to isolate algorithmic gains
│   │       ├── evaluate.py                                  # General-purpose evaluation runner computing top-1 accuracy and cross-entropy loss
│   │       ├── evaluate_clip.py                             # Evaluates zero-shot classification and semantic alignment for CLIP ViT-B/32
│   │       ├── extract_cornet_features.py                   # Feature extractor dumping intermediate activations from V1, V2, V4, and IT cortices
│   │       ├── model.py                                     # Standard ResNet-18 baseline trained on CIFAR-10 with cross-entropy (95.82% clean)
│   │       ├── model_bagnet.py                              # BagNet-33 baseline with restricted 33x33 receptive fields testing pure local texture bias
│   │       ├── model_clip.py                                # OpenAI CLIP ViT-B/32 zero-shot baseline evaluating vision-language contrastive representations
│   │       ├── model_cornets.py                             # Recurrent primate ventral stream architecture (V1, V2, V4, IT) testing biological feedback
│   │       ├── model_efficientnet.py                        # Compound-scaled EfficientNet-B0 baseline evaluating CNN scaling vs robustness trade-offs
│   │       ├── model_rhan_adaptive.py                       # RHAN prototype adding confidence-based adaptive glimpse count allocation
│   │       ├── model_rhan_aligned.py                        # RHAN prototype aligning recurrent representations with semantic language priors
│   │       ├── model_rhan_clip.py                           # Hybrid model distilling CLIP multi-modal embeddings into recurrent states
│   │       ├── model_rhan_dualstream.py                     # Dual-stream prototype processing simultaneous coarse gist and fine foveal patches
│   │       ├── model_rhan_full.py                           # Comprehensive RHAN prototype integrating all experimental mechanisms simultaneously
│   │       ├── model_rhan_predcoding.py                     # Predictive coding prototype minimizing hierarchical feature prediction errors
│   │       ├── model_rhan_predictive.py                     # Predictive glimpse selection prototype forecasting informativeness of prospective saccades
│   │       ├── model_rhan_split.py                          # Split-architecture prototype separating what (semantic) and where (spatial) visual streams
│   │       ├── model_rhan_v10.py                            # RHAN v10 adding predictive coding error residual streams between layers
│   │       ├── model_rhan_v11.py                            # RHAN v11 incorporating structured spatial belief maps and foveal coordinates
│   │       ├── model_rhan_v12.py                            # Contractually frozen Gen-0 reference model with tied recurrence and foveation
│   │       ├── model_rhan_v9.py                             # RHAN v9 introducing direct top-down recurrent connections from IT to V1
│   │       ├── model_rhan_v9.py:Zone.Identifier             # Windows filesystem download zone identifier metadata for model_rhan_v9.py
│   │       ├── model_shaperesnet.py                         # ResNet-50 trained on Stylized ImageNet (SIN) testing whether shape bias improves robustness
│   │       ├── model_vit.py                                 # Vision Transformer (ViT-Small) baseline with global multi-head self-attention patches
│   │       ├── pretrain_noise_estimator.py                  # Pretraining script for noise variance estimation network used in adaptive filtering
│   │       ├── pretrain_rhan_v5_clip.py                     # Pretraining script aligning RHAN v5 recurrent states with CLIP visual latents
│   │       ├── pretrain_rhan_v6_clip.py                     # Extended contrastive pretraining driver aligning RHAN v6 representations with CLIP
│   │       ├── profile_rhan_large.py                        # Profiles compute complexity, FLOPs, and latency for scaled-up RHAN architectures
│   │       ├── train.py                                     # Primary training script for baseline CIFAR-10 CNNs and Vision Transformers
│   │       ├── train_cornets.py                             # Specialized training driver for recurrent CORnet-S with anatomical area recurrence
│   │       ├── train_efficientnet.py                        # Training driver for EfficientNet-B0 with gradient clipping and resolution management
│   │       ├── train_feature_scatter.py                     # Feature scattering adversarial training driver exploring manifold-expanding defenses
│   │       ├── train_rhan_adaptive.py                       # Training script for adaptive glimpse allocation models using reinforcement/stopping loss
│   │       ├── train_rhan_adv.py                            # PGD adversarial training harness for early RHAN architectures
│   │       ├── train_rhan_aligned.py                        # Training script optimizing cross-modal semantic alignment objectives
│   │       ├── train_rhan_alignment.py                      # Contrastive representation alignment training driver for recurrent visual latents
│   │       ├── train_rhan_cbm.py                            # Concept Bottleneck Model (CBM) trainer supervising intermediate human-interpretable concepts
│   │       ├── train_rhan_cbm_trades.py                     # Concept Bottleneck Model trainer combined with TRADES adversarial loss
│   │       ├── train_rhan_cbm_v2.py                         # Refined Concept Bottleneck Model trainer with continuous concept activations
│   │       ├── train_rhan_clip.py                           # Training driver for distilling OpenAI CLIP embeddings into recurrent state
│   │       ├── train_rhan_dualstream.py                     # Training driver for dual-stream architecture simultaneously processing gist and fovea
│   │       ├── train_rhan_dualstream_adv.py                 # Adversarial training driver for dual-stream coarse/fine foveated network
│   │       ├── train_rhan_next.py                           # Canonical Gen-0 trainer coordinating modular HPC, AIS, SBR, and IWM with TRADES curriculum
│   │       ├── train_rhan_predcoding.py                     # Hierarchical predictive coding trainer minimizing top-down feature error residuals
│   │       ├── train_rhan_predcoding_adv.py                 # Adversarial predictive coding trainer evaluating robustness under prediction error loss
│   │       ├── train_rhan_predictive.py                     # Predictive gaze trajectory trainer optimizing visual information foraging
│   │       ├── train_rhan_pseudolabel.py                    # Semi-supervised pseudo-labeling pipeline leveraging 100k unlabeled STL-10 images
│   │       ├── train_rhan_pseudolabel_v2.py                 # Confidence-thresholded pseudo-label distillation trainer on unlabeled data
│   │       ├── train_rhan_selfalign.py                      # Self-alignment trainer regularizing latent drift between clean and perturbed glimpses
│   │       ├── train_rhan_split.py                          # Training driver for split-stream what/where visual pathways
│   │       ├── train_rhan_stl10.py                          # Training driver for RHAN scaled to STL-10 96x96 pixel resolution
│   │       ├── train_rhan_stl10_pretrained.py               # Fine-tuning driver for STL-10 initializing backbone from self-supervised weights
│   │       ├── train_rhan_trades_class_hardened.py          # Class-hardened TRADES curriculum scaling adversarial weights for vulnerable categories
│   │       ├── train_rhan_trades_curriculum.py              # 60-epoch curriculum gradually increasing TRADES adversarial perturbation radius
│   │       ├── train_rhan_v10.py                            # Training driver for RHAN v10 predictive coding architecture
│   │       ├── train_rhan_v11.py                            # Training driver for RHAN v11 structured spatial belief architecture
│   │       ├── train_rhan_v12.py                            # Training driver for RHAN v12 tied recurrence reference architecture
│   │       ├── train_rhan_v3_adaptive.py                    # Training driver for RHAN v3 with adaptive glimpse budgeting
│   │       ├── train_rhan_v5_trades.py                      # Training driver for RHAN v5 with TRADES adversarial loss
│   │       ├── train_rhan_v8.py                             # Training driver for RHAN v8 multi-scale recurrent architecture
│   │       ├── train_rhan_v9_direct.py                      # Training driver for RHAN v9 with direct IT-to-V1 feedback
│   │       ├── train_rhan_video_tdv.py                      # Temporal Difference Visual (TDV) training on sequential video frame transitions
│   │       ├── train_sail.py                                # Self-supervised Adversarial Invariant Learning (SAIL) contrastive training script
│   │       ├── train_sail.py:Zone.Identifier                # Windows filesystem download zone identifier metadata for train_sail.py
│   │       ├── train_shaperesnet.py                         # Training driver for Shape-ResNet-50 on Stylized ImageNet distributions
│   │       ├── train_shaperesnet_v2.py                      # Refined Shape-ResNet trainer incorporating mixup and cosine decay schedules
│   │       ├── train_stl10_pretrained.py                    # Pretraining driver for STL-10 feature backbones using SimCLR contrastive loss
│   │       ├── train_stl10_pretrained.py:Zone.Identifier    # Windows filesystem download zone identifier metadata for train_stl10_pretrained.py
│   │       └── train_vit.py                                 # Training driver for Vision Transformer (ViT-Small) on CIFAR-10 with RandAugment
│   ├── gen0/                                        # Archived Gen-0 (RHAN-Next) codebase and world model prototypes
│   │   └── rhan_core/                                   # Gen-0 core package: ablation, beliefs, gaze, precision, predictive coding
│   │       ├── OfficialNOESIS.png                           # Frozen Gen-0 RHAN-Next modular component (OfficialNOESIS.png) implementing experimental cognitive pillars
│   │       ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       ├── ablation/                                    # Ablation modules isolating recurrence, gating, and foveation
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── matrix.py                                    # Frozen Gen-0 RHAN-Next modular component (matrix.py) implementing experimental cognitive pillars
│   │       │   └── runner.py                                    # Frozen Gen-0 RHAN-Next modular component (runner.py) implementing experimental cognitive pillars
│   │       ├── artifacts/                                   # Reference evaluation outputs for Gen-0 models
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── events.py                                    # Frozen Gen-0 RHAN-Next modular component (events.py) implementing experimental cognitive pillars
│   │       │   ├── manifest.py                                  # Frozen Gen-0 RHAN-Next modular component (manifest.py) implementing experimental cognitive pillars
│   │       │   └── store.py                                     # Frozen Gen-0 RHAN-Next modular component (store.py) implementing experimental cognitive pillars
│   │       ├── beliefs/                                     # Gen-0 belief state implementations and experimental dynamics
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── base.py                                      # Frozen Gen-0 RHAN-Next modular component (base.py) implementing experimental cognitive pillars
│   │       │   ├── evidence_decomposition.py                    # Frozen Gen-0 RHAN-Next modular component (evidence_decomposition.py) implementing experimental cognitive pillars
│   │       │   ├── experimental/                                # Experimental continuous-discrete belief representations
│   │       │   │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   │   └── sbr_feasibility.py                           # Frozen Gen-0 RHAN-Next modular component (sbr_feasibility.py) implementing experimental cognitive pillars
│   │       │   ├── relational.py                                # Frozen Gen-0 RHAN-Next modular component (relational.py) implementing experimental cognitive pillars
│   │       │   ├── structured_belief.py                         # Frozen Gen-0 RHAN-Next modular component (structured_belief.py) implementing experimental cognitive pillars
│   │       │   └── vector_belief.py                             # Frozen Gen-0 RHAN-Next modular component (vector_belief.py) implementing experimental cognitive pillars
│   │       ├── config/                                      # Configuration dataclasses for Gen-0 models
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   └── pillar_config.py                             # Frozen Gen-0 RHAN-Next modular component (pillar_config.py) implementing experimental cognitive pillars
│   │       ├── gaze/                                        # Gen-0 fixation sampling policies and spatial grids
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── base.py                                      # Frozen Gen-0 RHAN-Next modular component (base.py) implementing experimental cognitive pillars
│   │       │   ├── halting.py                                   # Frozen Gen-0 RHAN-Next modular component (halting.py) implementing experimental cognitive pillars
│   │       │   ├── info_gain_policy.py                          # Frozen Gen-0 RHAN-Next modular component (info_gain_policy.py) implementing experimental cognitive pillars
│   │       │   └── info_gain_policy_v2.py                       # Frozen Gen-0 RHAN-Next modular component (info_gain_policy_v2.py) implementing experimental cognitive pillars
│   │       ├── lens/                                        # Introspection lens extracting interpretable state tokens
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── capture.py                                   # Frozen Gen-0 RHAN-Next modular component (capture.py) implementing experimental cognitive pillars
│   │       │   ├── hooks.py                                     # Frozen Gen-0 RHAN-Next modular component (hooks.py) implementing experimental cognitive pillars
│   │       │   └── session.py                                   # Frozen Gen-0 RHAN-Next modular component (session.py) implementing experimental cognitive pillars
│   │       ├── model.py                                     # Frozen Gen-0 RHAN-Next modular component (model.py) implementing experimental cognitive pillars
│   │       ├── optim/                                       # Optimizer parameter group constructors for Gen-0
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   └── multi_group_optimizer.py                     # Frozen Gen-0 RHAN-Next modular component (multi_group_optimizer.py) implementing experimental cognitive pillars
│   │       ├── precision/                                   # Dynamic precision weighting mechanisms for Gen-0
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── base.py                                      # Frozen Gen-0 RHAN-Next modular component (base.py) implementing experimental cognitive pillars
│   │       │   └── global_precision.py                          # Frozen Gen-0 RHAN-Next modular component (global_precision.py) implementing experimental cognitive pillars
│   │       ├── predictive_coding/                           # Gen-0 top-down error computation and prediction heads
│   │       │   ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │       │   ├── base.py                                      # Frozen Gen-0 RHAN-Next modular component (base.py) implementing experimental cognitive pillars
│   │       │   ├── feature_targets.py                           # Frozen Gen-0 RHAN-Next modular component (feature_targets.py) implementing experimental cognitive pillars
│   │       │   ├── hierarchical_stack.py                        # Frozen Gen-0 RHAN-Next modular component (hierarchical_stack.py) implementing experimental cognitive pillars
│   │       │   ├── hpc_belief_level.py                          # Frozen Gen-0 RHAN-Next modular component (hpc_belief_level.py) implementing experimental cognitive pillars
│   │       │   └── hpc_level1.py                                # Frozen Gen-0 RHAN-Next modular component (hpc_level1.py) implementing experimental cognitive pillars
│   │       └── world_model/                                 # World state transition models predicting future states
│   │           ├── __init__.py                                  # Frozen Gen-0 RHAN-Next modular component (__init__.py) implementing experimental cognitive pillars
│   │           ├── base.py                                      # Frozen Gen-0 RHAN-Next modular component (base.py) implementing experimental cognitive pillars
│   │           └── null_world_model.py                          # Frozen Gen-0 RHAN-Next modular component (null_world_model.py) implementing experimental cognitive pillars
│   ├── gen3-human/                                  # Archived human psychophysics baseline experiments and trial records
│   │   └── phase3_human_study/                          # Human trial logs, response templates, and participant data
│   │       ├── Human Psychophysics Study (Responses) - Form responses 1.csv # Human psychophysics trial log and empirical observer response data: Human Psychophysics Study (Responses) - Form responses 1.csv
│   │       ├── Human Psychophysics Study (Responses) - Form responses 1.csv:Zone.Identifier # Human psychophysics trial log and empirical observer response data: Human Psychophysics Study (Responses) - Form responses 1.csv:Zone.Identifier
│   │       ├── anonymize.py                                 # Human psychophysics trial log and empirical observer response data: anonymize.py
│   │       ├── build_form_guide.py                          # Human psychophysics trial log and empirical observer response data: build_form_guide.py
│   │       ├── build_manifest.py                            # Human psychophysics trial log and empirical observer response data: build_manifest.py
│   │       ├── form_structure.txt                           # Human psychophysics trial log and empirical observer response data: form_structure.txt
│   │       ├── human_responses_reconstructed_wide.csv       # Human psychophysics trial log and empirical observer response data: human_responses_reconstructed_wide.csv
│   │       ├── human_responses_reconstructed_wide.csv:Zone.Identifier # Human psychophysics trial log and empirical observer response data: human_responses_reconstructed_wide.csv:Zone.Identifier
│   │       ├── manifest.csv                                 # Human psychophysics trial log and empirical observer response data: manifest.csv
│   │       ├── map_responses.py                             # Human psychophysics trial log and empirical observer response data: map_responses.py
│   │       ├── process_responses.py                         # Human psychophysics trial log and empirical observer response data: process_responses.py
│   │       ├── response_schema.py                           # Human psychophysics trial log and empirical observer response data: response_schema.py
│   │       ├── response_template.csv                        # Human psychophysics trial log and empirical observer response data: response_template.csv
│   │       ├── responseorder.txt                            # Human psychophysics trial log and empirical observer response data: responseorder.txt
│   │       ├── stimuli_export.py                            # Human psychophysics trial log and empirical observer response data: stimuli_export.py
│   │       └── stimuli_manifest.csv                         # Human psychophysics trial log and empirical observer response data: stimuli_manifest.csv
│   ├── gen4-analysis/                               # Archived divergence analysis curves, heatmaps, and attention figures
│   │   └── phase4_analysis/                             # Cross-model divergence scripts and representation similarity metrics
│   │       ├── .claude.md                                   # Cross-model representational divergence and feature similarity script: .claude.md
│   │       ├── adaptive_steps_analysis.py                   # Cross-model representational divergence and feature similarity script: adaptive_steps_analysis.py
│   │       ├── alignment_analysis.py                        # Cross-model representational divergence and feature similarity script: alignment_analysis.py
│   │       ├── class_heatmap.py                             # Cross-model representational divergence and feature similarity script: class_heatmap.py
│   │       ├── confidence_curves.py                         # Cross-model representational divergence and feature similarity script: confidence_curves.py
│   │       ├── confusion_matrices.py                        # Cross-model representational divergence and feature similarity script: confusion_matrices.py
│   │       ├── cross_model_compare.py                       # Cross-model representational divergence and feature similarity script: cross_model_compare.py
│   │       ├── divergence_curves.py                         # Cross-model representational divergence and feature similarity script: divergence_curves.py
│   │       ├── dualstream_attention_analysis.py             # Cross-model representational divergence and feature similarity script: dualstream_attention_analysis.py
│   │       ├── figures/                                     # Attention and Grad-CAM visualizations across 7 architectures
│   │       │   ├── bagnet/                                      # BagNet-33 local patch heatmaps and texture bias plots
│   │       │   │   └── .gitkeep                                     # Cross-model representational divergence and feature similarity script: .gitkeep
│   │       │   ├── class_heatmap.png                            # Divergence analysis visualization: class heatmap
│   │       │   ├── combined/                                    # Multi-model comparative figures, t-SNE projections, and hero plots
│   │       │   │   ├── .gitkeep                                     # Cross-model representational divergence and feature similarity script: .gitkeep
│   │       │   │   ├── confidence_accuracy_gap.png                  # Divergence analysis visualization: confidence accuracy gap
│   │       │   │   ├── confidence_collapse.png                      # Divergence analysis visualization: confidence collapse
│   │       │   │   ├── full_confidence_curve.png                    # Divergence analysis visualization: full confidence curve
│   │       │   │   ├── full_divergence_curve.png                    # Divergence analysis visualization: full divergence curve
│   │       │   │   ├── hero_perturbation.png                        # Divergence analysis visualization: hero perturbation
│   │       │   │   ├── latent_space/                                # t-SNE latent feature space projections under clean vs adversarial perturbations
│   │       │   │   │   ├── displacement_comparison.bar.png              # Divergence analysis visualization: displacement comparison.bar
│   │       │   │   │   ├── resnet_tsne.png                              # Divergence analysis visualization: resnet tsne
│   │       │   │   │   └── vit_tsne.png                                 # Divergence analysis visualization: vit tsne
│   │       │   │   ├── partial_divergence_3model.png                # Divergence analysis visualization: partial divergence 3model
│   │       │   │   ├── partial_divergence_confidence.png            # Divergence analysis visualization: partial divergence confidence
│   │       │   │   ├── partial_divergence_curve.png                 # Divergence analysis visualization: partial divergence curve
│   │       │   │   ├── perturbation_atlas.png                       # Divergence analysis visualization: perturbation atlas
│   │       │   │   ├── sufficient_input_subsets.png                 # Divergence analysis visualization: sufficient input subsets
│   │       │   │   ├── threshold_summary/                           # Perceptual threshold summary charts comparing accuracy and SDT sensitivity
│   │       │   │   │   ├── accuracy_robustness_landscape.png            # Divergence analysis visualization: accuracy robustness landscape
│   │       │   │   │   ├── accuracy_threshold_bars.png                  # Divergence analysis visualization: accuracy threshold bars
│   │       │   │   │   └── sdt_threshold_bars.png                       # Divergence analysis visualization: sdt threshold bars
│   │       │   │   └── vit_attention_entropy.png                    # Divergence analysis visualization: vit attention entropy
│   │       │   ├── divergence_accuracy.png                      # Divergence analysis visualization: divergence accuracy
│   │       │   ├── divergence_confidence.png                    # Divergence analysis visualization: divergence confidence
│   │       │   ├── efficientnet/                                # Subsystem directory: efficientnet/
│   │       │   │   ├── .gitkeep                                     # Cross-model representational divergence and feature similarity script: .gitkeep
│   │       │   │   ├── class_heatmap.png                            # Divergence analysis visualization: class heatmap
│   │       │   │   └── confusion/                                   # Subsystem directory: confusion/
│   │       │   │       └── efficientnet_confusion_suite.png             # Divergence analysis visualization: efficientnet confusion suite
│   │       │   ├── gradcam/                                     # Subsystem directory: gradcam/
│   │       │   │   ├── airplane_gradcam.png                         # Divergence analysis visualization: airplane gradcam
│   │       │   │   ├── automobile_gradcam.png                       # Divergence analysis visualization: automobile gradcam
│   │       │   │   ├── bird_gradcam.png                             # Divergence analysis visualization: bird gradcam
│   │       │   │   ├── cat_gradcam.png                              # Divergence analysis visualization: cat gradcam
│   │       │   │   ├── deer_gradcam.png                             # Divergence analysis visualization: deer gradcam
│   │       │   │   ├── dog_gradcam.png                              # Divergence analysis visualization: dog gradcam
│   │       │   │   ├── frog_gradcam.png                             # Divergence analysis visualization: frog gradcam
│   │       │   │   ├── horse_gradcam.png                            # Divergence analysis visualization: horse gradcam
│   │       │   │   ├── ship_gradcam.png                             # Divergence analysis visualization: ship gradcam
│   │       │   │   └── truck_gradcam.png                            # Divergence analysis visualization: truck gradcam
│   │       │   ├── human/                                       # Subsystem directory: human/
│   │       │   │   ├── device_comparison.png                        # Divergence analysis visualization: device comparison
│   │       │   │   ├── human_calibration.png                        # Divergence analysis visualization: human calibration
│   │       │   │   └── human_class_heatmap.png                      # Divergence analysis visualization: human class heatmap
│   │       │   ├── resnet/                                      # Subsystem directory: resnet/
│   │       │   │   ├── .gitkeep                                     # Cross-model representational divergence and feature similarity script: .gitkeep
│   │       │   │   ├── class_heatmap.png                            # Divergence analysis visualization: class heatmap
│   │       │   │   ├── confusion/                                   # Subsystem directory: confusion/
│   │       │   │   │   └── resnet_confusion_suite.png                   # Divergence analysis visualization: resnet confusion suite
│   │       │   │   └── gradcam/                                     # Subsystem directory: gradcam/
│   │       │   │       ├── airplane_gradcam.png                         # Divergence analysis visualization: airplane gradcam
│   │       │   │       ├── automobile_gradcam.png                       # Divergence analysis visualization: automobile gradcam
│   │       │   │       ├── bird_gradcam.png                             # Divergence analysis visualization: bird gradcam
│   │       │   │       ├── cat_gradcam.png                              # Divergence analysis visualization: cat gradcam
│   │       │   │       ├── deer_gradcam.png                             # Divergence analysis visualization: deer gradcam
│   │       │   │       ├── dog_gradcam.png                              # Divergence analysis visualization: dog gradcam
│   │       │   │       ├── frog_gradcam.png                             # Divergence analysis visualization: frog gradcam
│   │       │   │       ├── horse_gradcam.png                            # Divergence analysis visualization: horse gradcam
│   │       │   │       ├── ship_gradcam.png                             # Divergence analysis visualization: ship gradcam
│   │       │   │       └── truck_gradcam.png                            # Divergence analysis visualization: truck gradcam
│   │       │   ├── shaperesnet/                                 # Subsystem directory: shaperesnet/
│   │       │   │   ├── .gitkeep                                     # Cross-model representational divergence and feature similarity script: .gitkeep
│   │       │   │   ├── class_heatmap.png                            # Divergence analysis visualization: class heatmap
│   │       │   │   └── confusion/                                   # Subsystem directory: confusion/
│   │       │   │       └── shaperesnet_confusion_suite.png              # Divergence analysis visualization: shaperesnet confusion suite
│   │       │   └── vit/                                         # Subsystem directory: vit/
│   │       │       ├── .gitkeep                                     # Cross-model representational divergence and feature similarity script: .gitkeep
│   │       │       ├── attention/                                   # Subsystem directory: attention/
│   │       │       │   ├── airplane_1.png                               # Divergence analysis visualization: airplane 1
│   │       │       │   ├── airplane_2.png                               # Divergence analysis visualization: airplane 2
│   │       │       │   ├── automobile_1.png                             # Divergence analysis visualization: automobile 1
│   │       │       │   ├── automobile_2.png                             # Divergence analysis visualization: automobile 2
│   │       │       │   ├── bird_1.png                                   # Divergence analysis visualization: bird 1
│   │       │       │   ├── bird_2.png                                   # Divergence analysis visualization: bird 2
│   │       │       │   ├── cat_1.png                                    # Divergence analysis visualization: cat 1
│   │       │       │   ├── cat_2.png                                    # Divergence analysis visualization: cat 2
│   │       │       │   ├── deer_1.png                                   # Divergence analysis visualization: deer 1
│   │       │       │   ├── deer_2.png                                   # Divergence analysis visualization: deer 2
│   │       │       │   ├── dog_1.png                                    # Divergence analysis visualization: dog 1
│   │       │       │   ├── dog_2.png                                    # Divergence analysis visualization: dog 2
│   │       │       │   ├── frog_1.png                                   # Divergence analysis visualization: frog 1
│   │       │       │   ├── frog_2.png                                   # Divergence analysis visualization: frog 2
│   │       │       │   ├── horse_1.png                                  # Divergence analysis visualization: horse 1
│   │       │       │   ├── horse_2.png                                  # Divergence analysis visualization: horse 2
│   │       │       │   ├── ship_1.png                                   # Divergence analysis visualization: ship 1
│   │       │       │   ├── ship_2.png                                   # Divergence analysis visualization: ship 2
│   │       │       │   ├── truck_1.png                                  # Divergence analysis visualization: truck 1
│   │       │       │   └── truck_2.png                                  # Divergence analysis visualization: truck 2
│   │       │       ├── class_heatmap.png                            # Divergence analysis visualization: class heatmap
│   │       │       └── confusion/                                   # Subsystem directory: confusion/
│   │       │           └── vit_confusion_suite.png                      # Divergence analysis visualization: vit confusion suite
│   │       ├── generate_appendix_figures.py                 # Cross-model representational divergence and feature similarity script: generate_appendix_figures.py
│   │       ├── generate_hero_figures.py                     # Cross-model representational divergence and feature similarity script: generate_hero_figures.py
│   │       ├── get_clip_accuracy.py                         # Cross-model representational divergence and feature similarity script: get_clip_accuracy.py
│   │       ├── gradcam.py                                   # Cross-model representational divergence and feature similarity script: gradcam.py
│   │       ├── human_deep_analysis.py                       # Cross-model representational divergence and feature similarity script: human_deep_analysis.py
│   │       ├── latent_space_embeddings.py                   # Cross-model representational divergence and feature similarity script: latent_space_embeddings.py
│   │       ├── perturbation_visuals.py                      # Cross-model representational divergence and feature similarity script: perturbation_visuals.py
│   │       ├── plot_rhan_curriculum_accomplishments.py      # Cross-model representational divergence and feature similarity script: plot_rhan_curriculum_accomplishments.py
│   │       ├── predcoding_residual_analysis.py              # Cross-model representational divergence and feature similarity script: predcoding_residual_analysis.py
│   │       ├── predictive_analysis.py                       # Cross-model representational divergence and feature similarity script: predictive_analysis.py
│   │       ├── rhan_feedback_maps.py                        # Cross-model representational divergence and feature similarity script: rhan_feedback_maps.py
│   │       ├── split_stream_analysis.py                     # Cross-model representational divergence and feature similarity script: split_stream_analysis.py
│   │       ├── sufficient_input_subsets.py                  # Cross-model representational divergence and feature similarity script: sufficient_input_subsets.py
│   │       ├── threshold_summary.py                         # Cross-model representational divergence and feature similarity script: threshold_summary.py
│   │       ├── v3_gate_entropy_analysis.py                  # Cross-model representational divergence and feature similarity script: v3_gate_entropy_analysis.py
│   │       └── vit_attention_maps.py                        # Cross-model representational divergence and feature similarity script: vit_attention_maps.py
│   ├── gen5-sdt/                                    # Archived Signal Detection Theory calculations and ROC curves
│   │   └── phase5_sdt/                                  # Signal Detection Theory (SDT) sensitivity d-prime and criterion c calculators
│   │       ├── RESULTS_INTERPRETATION_GUIDE.md              # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: RESULTS_INTERPRETATION_GUIDE.md
│   │       ├── __init__.py                                  # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: __init__.py
│   │       ├── figures/                                     # Visual plots of d-prime curves and perceptual sensitivity collapse
│   │       │   ├── .gitkeep                                     # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: .gitkeep
│   │       │   ├── dprime_vs_epsilon.png                        # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: dprime_vs_epsilon.png
│   │       │   ├── partial_dprime_curves.png                    # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: partial_dprime_curves.png
│   │       │   └── perclass_dprime_eps0.10.png                  # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: perclass_dprime_eps0.10.png
│   │       ├── final_report.py                              # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: final_report.py
│   │       ├── final_report.txt                             # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: final_report.txt
│   │       ├── results/                                     # Tabular SDT calculation logs and text summary reports across models
│   │       │   ├── .gitkeep                                     # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: .gitkeep
│   │       │   ├── final_report_5model.txt                      # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: final_report_5model.txt
│   │       │   ├── final_report_6model.txt                      # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: final_report_6model.txt
│   │       │   ├── partial_final_report.txt                     # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: partial_final_report.txt
│   │       │   ├── partial_final_report_v2.txt                  # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: partial_final_report_v2.txt
│   │       │   ├── partial_sdt_results.csv                      # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: partial_sdt_results.csv
│   │       │   ├── sdt_results.csv                              # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: sdt_results.csv
│   │       │   └── sdt_results_v4.csv                           # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: sdt_results_v4.csv
│   │       ├── sdt_analysis.py                              # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: sdt_analysis.py
│   │       ├── sdt_core.py                                  # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: sdt_core.py
│   │       └── sdt_plots.py                                 # Signal Detection Theory perceptual sensitivity (d-prime) and criterion tool: sdt_plots.py
│   ├── historical-report/                           # Prior LaTeX manuscripts, draft papers, and workshop submissions
│   │   └── RHANv12/                                     # Draft manuscripts and LaTeX figures for RHAN v12
│   │       └── scripts/                                     # Script utilities for generating figures for RHAN v12 paper
│   │           ├── _report_common.py                            # Historical research manuscript source and report figure generator: _report_common.py
│   │           ├── generate_activation_maps.py                  # Historical research manuscript source and report figure generator: generate_activation_maps.py
│   │           ├── generate_attention_diagrams.py               # Historical research manuscript source and report figure generator: generate_attention_diagrams.py
│   │           ├── generate_complexity_tables.py                # Historical research manuscript source and report figure generator: generate_complexity_tables.py
│   │           ├── generate_flowcharts.py                       # Historical research manuscript source and report figure generator: generate_flowcharts.py
│   │           ├── generate_math_figures.py                     # Historical research manuscript source and report figure generator: generate_math_figures.py
│   │           ├── generate_parameter_tables.py                 # Historical research manuscript source and report figure generator: generate_parameter_tables.py
│   │           └── generate_visualtorch.py                      # Historical research manuscript source and report figure generator: generate_visualtorch.py
│   ├── legacy-evals/                                # Archived evaluation benchmarks, attack scripts, and verification sweeps
│   │   ├── .vscode/                                     # VS Code editor session metadata for legacy evaluation tools
│   │   │   └── sessions.json                                # Structured configuration and experimental metadata: sessions.json
│   │   ├── applications/                                # Downstream application specifications and deployment targets
│   │   │   └── README.md                                    # Architectural roadmap for downstream RHAN applications (autonomous driving, medical imaging)
│   │   ├── bench_pgd.py                                 # Micro-benchmark measuring PGD step execution latency, memory footprint, and GPU throughput
│   │   ├── check_parquet.py                             # Data integrity check verifying Parquet dataset chunk serialization and column schemas
│   │   ├── cognitive_vision_lab/                        # Streamlit cognitive vision laboratory interactive app
│   │   │   ├── .env.example                                 # Streamlit cognitive vision lab UI component and benchmarking engine: .env.example
│   │   │   ├── .streamlit/                                  # Streamlit server settings and visual theme configuration
│   │   │   │   └── config.toml                                  # Streamlit cognitive vision lab UI component and benchmarking engine: config.toml
│   │   │   ├── Dockerfile                                   # Streamlit cognitive vision lab UI component and benchmarking engine: Dockerfile
│   │   │   ├── README.md                                    # Master repository overview, empirical benchmark findings, and exhaustive repo map
│   │   │   ├── __init__.py                                  # Streamlit cognitive vision lab UI component and benchmarking engine: __init__.py
│   │   │   ├── app.py                                       # Interactive web application for visualizing model predictions, glimpses, and adversarial attacks
│   │   │   ├── assets/                                      # CSS styling and visual assets for the interactive lab UI
│   │   │   │   └── style.css                                    # Streamlit cognitive vision lab UI component and benchmarking engine: style.css
│   │   │   ├── backend/                                     # Backend execution engines for attacks, benchmarks, and model loading
│   │   │   │   ├── __init__.py                                  # Streamlit cognitive vision lab UI component and benchmarking engine: __init__.py
│   │   │   │   ├── attacks.py                                   # Streamlit cognitive vision lab UI component and benchmarking engine: attacks.py
│   │   │   │   ├── benchmark.py                                 # Streamlit cognitive vision lab UI component and benchmarking engine: benchmark.py
│   │   │   │   ├── datasets.py                                  # Streamlit cognitive vision lab UI component and benchmarking engine: datasets.py
│   │   │   │   ├── embeddings.py                                # Streamlit cognitive vision lab UI component and benchmarking engine: embeddings.py
│   │   │   │   ├── experiments.py                               # Streamlit cognitive vision lab UI component and benchmarking engine: experiments.py
│   │   │   │   ├── explainability.py                            # Streamlit cognitive vision lab UI component and benchmarking engine: explainability.py
│   │   │   │   ├── human.py                                     # Streamlit cognitive vision lab UI component and benchmarking engine: human.py
│   │   │   │   ├── metrics.py                                   # Streamlit cognitive vision lab UI component and benchmarking engine: metrics.py
│   │   │   │   ├── models.py                                    # Streamlit cognitive vision lab UI component and benchmarking engine: models.py
│   │   │   │   ├── reports.py                                   # Streamlit cognitive vision lab UI component and benchmarking engine: reports.py
│   │   │   │   ├── rhan_arch.py                                 # Streamlit cognitive vision lab UI component and benchmarking engine: rhan_arch.py
│   │   │   │   ├── robustbench.py                               # Streamlit cognitive vision lab UI component and benchmarking engine: robustbench.py
│   │   │   │   └── state.py                                     # Streamlit cognitive vision lab UI component and benchmarking engine: state.py
│   │   │   ├── components/                                  # Reusable Streamlit UI widgets: metric cards, charts, and equations
│   │   │   │   ├── __init__.py                                  # Streamlit cognitive vision lab UI component and benchmarking engine: __init__.py
│   │   │   │   ├── cards.py                                     # Streamlit cognitive vision lab UI component and benchmarking engine: cards.py
│   │   │   │   ├── charts.py                                    # Streamlit cognitive vision lab UI component and benchmarking engine: charts.py
│   │   │   │   ├── equations.py                                 # Streamlit cognitive vision lab UI component and benchmarking engine: equations.py
│   │   │   │   ├── layout.py                                    # Streamlit cognitive vision lab UI component and benchmarking engine: layout.py
│   │   │   │   └── tables.py                                    # Streamlit cognitive vision lab UI component and benchmarking engine: tables.py
│   │   │   ├── config.py                                    # Streamlit cognitive vision lab UI component and benchmarking engine: config.py
│   │   │   ├── docker-compose.yml                           # Streamlit cognitive vision lab UI component and benchmarking engine: docker-compose.yml
│   │   │   ├── pages/                                       # Subsystem directory: pages/
│   │   │   │   ├── 01_Home.py                                   # Streamlit cognitive vision lab UI component and benchmarking engine: 01_Home.py
│   │   │   │   ├── 02_Model_Zoo.py                              # Streamlit cognitive vision lab UI component and benchmarking engine: 02_Model_Zoo.py
│   │   │   │   ├── 03_Interactive_Inference.py                  # Streamlit cognitive vision lab UI component and benchmarking engine: 03_Interactive_Inference.py
│   │   │   │   ├── 04_Adversarial_Lab.py                        # Streamlit cognitive vision lab UI component and benchmarking engine: 04_Adversarial_Lab.py
│   │   │   │   ├── 05_Attention_Explorer.py                     # Streamlit cognitive vision lab UI component and benchmarking engine: 05_Attention_Explorer.py
│   │   │   │   ├── 06_Representation_Drift.py                   # Streamlit cognitive vision lab UI component and benchmarking engine: 06_Representation_Drift.py
│   │   │   │   ├── 07_GradCAM_and_Saliency.py                   # Streamlit cognitive vision lab UI component and benchmarking engine: 07_GradCAM_and_Saliency.py
│   │   │   │   ├── 08_Human_vs_AI.py                            # Streamlit cognitive vision lab UI component and benchmarking engine: 08_Human_vs_AI.py
│   │   │   │   ├── 09_RobustBench_Comparison.py                 # Streamlit cognitive vision lab UI component and benchmarking engine: 09_RobustBench_Comparison.py
│   │   │   │   ├── 10_RHAN_Architecture.py                      # Streamlit cognitive vision lab UI component and benchmarking engine: 10_RHAN_Architecture.py
│   │   │   │   ├── 11_Benchmark_Results.py                      # Streamlit cognitive vision lab UI component and benchmarking engine: 11_Benchmark_Results.py
│   │   │   │   ├── 12_Experiment_Manager.py                     # Streamlit cognitive vision lab UI component and benchmarking engine: 12_Experiment_Manager.py
│   │   │   │   ├── 13_Dataset_Explorer.py                       # Streamlit cognitive vision lab UI component and benchmarking engine: 13_Dataset_Explorer.py
│   │   │   │   ├── 14_Report_Generator.py                       # Streamlit cognitive vision lab UI component and benchmarking engine: 14_Report_Generator.py
│   │   │   │   └── 15_Perception_Replay.py                      # Streamlit cognitive vision lab UI component and benchmarking engine: 15_Perception_Replay.py
│   │   │   ├── requirements.txt                             # Core Python dependencies: PyTorch, torchvision, timm, torchattacks, scipy
│   │   │   ├── tests/                                       # Subsystem directory: tests/
│   │   │   │   ├── test_attacks.py                              # Streamlit cognitive vision lab UI component and benchmarking engine: test_attacks.py
│   │   │   │   ├── test_benchmark.py                            # Streamlit cognitive vision lab UI component and benchmarking engine: test_benchmark.py
│   │   │   │   ├── test_datasets.py                             # Streamlit cognitive vision lab UI component and benchmarking engine: test_datasets.py
│   │   │   │   ├── test_metrics.py                              # Streamlit cognitive vision lab UI component and benchmarking engine: test_metrics.py
│   │   │   │   ├── test_models.py                               # Streamlit cognitive vision lab UI component and benchmarking engine: test_models.py
│   │   │   │   └── test_reports.py                              # Streamlit cognitive vision lab UI component and benchmarking engine: test_reports.py
│   │   │   └── utils/                                       # Subsystem directory: utils/
│   │   │       ├── __init__.py                                  # Streamlit cognitive vision lab UI component and benchmarking engine: __init__.py
│   │   │       ├── caching.py                                   # Streamlit cognitive vision lab UI component and benchmarking engine: caching.py
│   │   │       ├── hardware.py                                  # Streamlit cognitive vision lab UI component and benchmarking engine: hardware.py
│   │   │       ├── io.py                                        # Streamlit cognitive vision lab UI component and benchmarking engine: io.py
│   │   │       ├── logging.py                                   # Streamlit cognitive vision lab UI component and benchmarking engine: logging.py
│   │   │       ├── math_helpers.py                              # Streamlit cognitive vision lab UI component and benchmarking engine: math_helpers.py
│   │   │       └── theme.py                                     # Streamlit cognitive vision lab UI component and benchmarking engine: theme.py
│   │   ├── concept_ablation.py                          # Python execution module and utility: concept_ablation.py
│   │   ├── concept_ablation_results.npz                 # Repository component: concept_ablation_results.npz
│   │   ├── config/                                      # Subsystem directory: config/
│   │   │   ├── attack_config.yaml                           # Repository component: attack_config.yaml
│   │   │   ├── train_config.yaml                            # Repository component: train_config.yaml
│   │   │   └── train_config_vit.yaml                        # Repository component: train_config_vit.yaml
│   │   ├── dashboards/                                  # Subsystem directory: dashboards/
│   │   │   └── lens_app.py                                  # Python execution module and utility: lens_app.py
│   │   ├── data_generation/                             # Subsystem directory: data_generation/
│   │   │   ├── __init__.py                                  # Python execution module and utility: __init__.py
│   │   │   ├── filter_synthetic_clip.py                     # Python execution module and utility: filter_synthetic_clip.py
│   │   │   ├── generate_synthetic_stl10.py                  # Python execution module and utility: generate_synthetic_stl10.py
│   │   │   ├── prepare_synthetic_data.py                    # Python execution module and utility: prepare_synthetic_data.py
│   │   │   └── upload_synthetic_hf.py                       # Python execution module and utility: upload_synthetic_hf.py
│   │   ├── demo.py                                      # Python execution module and utility: demo.py
│   │   ├── eval_aa_selfalign.py                         # Python execution module and utility: eval_aa_selfalign.py
│   │   ├── eval_aa_v2.py                                # Python execution module and utility: eval_aa_v2.py
│   │   ├── eval_pgd_arrow.py                            # Python execution module and utility: eval_pgd_arrow.py
│   │   ├── eval_pgd_final.py                            # Python execution module and utility: eval_pgd_final.py
│   │   ├── eval_pgd_hf.py                               # Python execution module and utility: eval_pgd_hf.py
│   │   ├── eval_pgd_hf2.py                              # Python execution module and utility: eval_pgd_hf2.py
│   │   ├── eval_pgd_quick.py                            # Python execution module and utility: eval_pgd_quick.py
│   │   ├── eval_pgd_sweep.py                            # Python execution module and utility: eval_pgd_sweep.py
│   │   ├── eval_pgd_thresh.py                           # Python execution module and utility: eval_pgd_thresh.py
│   │   ├── eval_pgd_v2.py                               # Python execution module and utility: eval_pgd_v2.py
│   │   ├── eval_pgd_v3.py                               # Python execution module and utility: eval_pgd_v3.py
│   │   ├── eval_pgd_v4.py                               # Python execution module and utility: eval_pgd_v4.py
│   │   ├── eval_pgd_v5.py                               # Python execution module and utility: eval_pgd_v5.py
│   │   ├── eval_quick_perclass.py                       # Python execution module and utility: eval_quick_perclass.py
│   │   ├── eval_quick_perclass_aa.py                    # Python execution module and utility: eval_quick_perclass_aa.py
│   │   ├── eval_quick_perclass_aa_correct.py            # Python execution module and utility: eval_quick_perclass_aa_correct.py
│   │   ├── eval_stl10.py                                # Python execution module and utility: eval_stl10.py
│   │   ├── inspect_checkpoint.py                        # Python execution module and utility: inspect_checkpoint.py
│   │   ├── inspect_v10_checkpoint.py                    # Python execution module and utility: inspect_v10_checkpoint.py
│   │   ├── logs/                                        # Subsystem directory: logs/
│   │   │   ├── null_ablation_20260724_232518.log            # Repository component: null_ablation_20260724_232518.log
│   │   │   ├── null_ablation_20260724_234145.log            # Repository component: null_ablation_20260724_234145.log
│   │   │   └── null_ablation_20260725_002242.log            # Repository component: null_ablation_20260725_002242.log
│   │   ├── phase2_attacks/                              # Unified adversarial attack suites (FGSM, PGD, C&W, AutoAttack) and frozen Gen-0 eval
│   │   │   ├── cw.py                                        # Carlini & Wagner (C&W) L2 optimization attack minimizing perturbation norm
│   │   │   ├── eval_autoattack.py                           # Hardened AutoAttack benchmark (APGD-CE, APGD-DLR, FAB, Square) verifying no gradient masking
│   │   │   ├── eval_autoattack_curriculum.py                # Python execution module and utility: eval_autoattack_curriculum.py
│   │   │   ├── eval_autoattack_fast.py                      # Python execution module and utility: eval_autoattack_fast.py
│   │   │   ├── eval_autoattack_mem.py                       # Python execution module and utility: eval_autoattack_mem.py
│   │   │   ├── eval_domain_clipping_validation.py           # Python execution module and utility: eval_domain_clipping_validation.py
│   │   │   ├── eval_empirical_epsilon_sweep.py              # Python execution module and utility: eval_empirical_epsilon_sweep.py
│   │   │   ├── eval_empirical_sweep_verified.py             # Python execution module and utility: eval_empirical_sweep_verified.py
│   │   │   ├── eval_ensemble_curriculum.py                  # Python execution module and utility: eval_ensemble_curriculum.py
│   │   │   ├── eval_full_epsilon_sweep.py                   # Complete epsilon sweep evaluating accuracy across eps in [0.0, 0.3] in increments of 0.01
│   │   │   ├── eval_quick.py                                # Quick validation sweep computing clean and PGD-10 accuracy across test batches
│   │   │   ├── eval_rhan.py                                 # Frozen Gen-0 evaluation entrypoint (norm-space PGD-50 and APGD-100, 5-8 seeds, n=300/seed)
│   │   │   ├── eval_rhan_selftest_ref.json                  # Structured configuration and experimental metadata: eval_rhan_selftest_ref.json
│   │   │   ├── eval_rhan_v7.py                              # Python execution module and utility: eval_rhan_v7.py
│   │   │   ├── eval_sweep_domain_clamped.py                 # Python execution module and utility: eval_sweep_domain_clamped.py
│   │   │   ├── eval_sweep_next.py                           # Python execution module and utility: eval_sweep_next.py
│   │   │   ├── evaluate_attacks.py                          # Python execution module and utility: evaluate_attacks.py
│   │   │   ├── evaluate_rhan.py                             # Python execution module and utility: evaluate_rhan.py
│   │   │   ├── fgsm.py                                      # Fast Gradient Sign Method (FGSM) single-step adversarial attack baseline
│   │   │   ├── find_epoch_revision.py                       # Python execution module and utility: find_epoch_revision.py
│   │   │   ├── generate_adv_all_models.py                   # Unified adversarial dataset generator producing FGSM, PGD, and C&W perturbed samples
│   │   │   ├── generate_adv_clip.py                         # Python execution module and utility: generate_adv_clip.py
│   │   │   ├── generate_adv_dataset.py                      # Python execution module and utility: generate_adv_dataset.py
│   │   │   ├── generate_adv_shaperesnet.py                  # Python execution module and utility: generate_adv_shaperesnet.py
│   │   │   ├── merge_sweep_results.py                       # Python execution module and utility: merge_sweep_results.py
│   │   │   ├── pgd.py                                       # Projected Gradient Descent (PGD) Linf attack with random restart and step-size alpha
│   │   │   ├── resume_pgd.py                                # Python execution module and utility: resume_pgd.py
│   │   │   ├── seed_sweep_comparators.py                    # Python execution module and utility: seed_sweep_comparators.py
│   │   │   ├── shard_2gpu.py                                # Python execution module and utility: shard_2gpu.py
│   │   │   └── test_gradient_masking.py                     # Python execution module and utility: test_gradient_masking.py
│   │   ├── resnet_eval.txt                              # Repository component: resnet_eval.txt
│   │   ├── response_template.csv                        # Repository component: response_template.csv
│   │   ├── rhan_flowchart.png                           # Repository component: rhan_flowchart.png
│   │   ├── run_eval_stl10.py                            # Python execution module and utility: run_eval_stl10.py
│   │   ├── simple_inspect.py                            # Python execution module and utility: simple_inspect.py
│   │   ├── sweep_results/                               # Subsystem directory: sweep_results/
│   │   │   └── epsilon_sweep_per_seed.csv                   # Repository component: epsilon_sweep_per_seed.csv
│   │   ├── test_ckpt_load.py                            # Python execution module and utility: test_ckpt_load.py
│   │   ├── test_load.py                                 # Python execution module and utility: test_load.py
│   │   ├── train_norecon.log                            # Repository component: train_norecon.log
│   │   ├── upload_pseudolabel.py                        # Python execution module and utility: upload_pseudolabel.py
│   │   ├── utils/                                       # Subsystem directory: utils/
│   │   │   └── metrics.py                                   # Python execution module and utility: metrics.py
│   │   └── vit_eval.txt                                 # Repository component: vit_eval.txt
│   ├── pkg-rhan-math/                               # Mathematical derivations, proofs, and vector calculus reports
│   │   └── rhan_math/                                   # LaTeX source files for mathematical formulation paper
│   │       ├── assets/                                      # Visual plots of Banach fixed points, halting steps, and loss landscapes
│   │       │   ├── act_halting_steps.png                        # Formal mathematical formulation and analytical convergence proof asset: act_halting_steps.png
│   │       │   ├── banach_contraction_decay.png                 # Formal mathematical formulation and analytical convergence proof asset: banach_contraction_decay.png
│   │       │   ├── deq_memory_complexity.png                    # Formal mathematical formulation and analytical convergence proof asset: deq_memory_complexity.png
│   │       │   ├── dynamic_trades_gating.png                    # Formal mathematical formulation and analytical convergence proof asset: dynamic_trades_gating.png
│   │       │   ├── gelu_dip.png                                 # Formal mathematical formulation and analytical convergence proof asset: gelu_dip.png
│   │       │   ├── gradient_masking_surface.png                 # Formal mathematical formulation and analytical convergence proof asset: gradient_masking_surface.png
│   │       │   ├── groupnorm_projection.png                     # Formal mathematical formulation and analytical convergence proof asset: groupnorm_projection.png
│   │       │   ├── hessian_conditioning.png                     # Formal mathematical formulation and analytical convergence proof asset: hessian_conditioning.png
│   │       │   ├── left_null_space.png                          # Formal mathematical formulation and analytical convergence proof asset: left_null_space.png
│   │       │   ├── pgd100_flatline.png                          # Formal mathematical formulation and analytical convergence proof asset: pgd100_flatline.png
│   │       │   ├── precision_gating.png                         # Formal mathematical formulation and analytical convergence proof asset: precision_gating.png
│   │       │   ├── rank_restoration.png                         # Formal mathematical formulation and analytical convergence proof asset: rank_restoration.png
│   │       │   ├── specimen_trajectory.png                      # Formal mathematical formulation and analytical convergence proof asset: specimen_trajectory.png
│   │       │   ├── spherical_geometry.png                       # Formal mathematical formulation and analytical convergence proof asset: spherical_geometry.png
│   │       │   └── stn_grid.png                                 # Formal mathematical formulation and analytical convergence proof asset: stn_grid.png
│   │       ├── compile_proof_report.py                      # Formal mathematical formulation and analytical convergence proof asset: compile_proof_report.py
│   │       ├── generate_proof_figures.py                    # Formal mathematical formulation and analytical convergence proof asset: generate_proof_figures.py
│   │       ├── phase1_proofs.md                             # Formal mathematical formulation and analytical convergence proof asset: phase1_proofs.md
│   │       ├── phase2_proofs.md                             # Formal mathematical formulation and analytical convergence proof asset: phase2_proofs.md
│   │       ├── phase3_proofs.md                             # Formal mathematical formulation and analytical convergence proof asset: phase3_proofs.md
│   │       ├── phase4_proofs.md                             # Formal mathematical formulation and analytical convergence proof asset: phase4_proofs.md
│   │       └── phase5_proofs.md                             # Formal mathematical formulation and analytical convergence proof asset: phase5_proofs.md
│   ├── pkg-tier1/                                   # Archived Tier-1 multi-seed reproducibility benchmark suites
│   │   └── tier1/                                       # Tier-1 evaluation harness and multi-seed result tables
│   │       ├── ScientificValidationReport.aux               # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport.aux
│   │       ├── ScientificValidationReport.log               # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport.log
│   │       ├── ScientificValidationReport.out               # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport.out
│   │       ├── ScientificValidationReport.tex               # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport.tex
│   │       ├── ScientificValidationReport2.aux              # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport2.aux
│   │       ├── ScientificValidationReport2.log              # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport2.log
│   │       ├── ScientificValidationReport2.out              # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport2.out
│   │       ├── ScientificValidationReport2.tex              # Tier-1 multi-seed reproducibility benchmark report and statistical output: ScientificValidationReport2.tex
│   │       ├── results/                                     # Empirical Tier-1 evaluation outputs across seeds and ablation settings
│   │       │   ├── aggregate_results.json                       # Tier-1 multi-seed reproducibility benchmark report and statistical output: aggregate_results.json
│   │       │   ├── comparison_table.csv                         # Tier-1 multi-seed reproducibility benchmark report and statistical output: comparison_table.csv
│   │       │   ├── frequency_gate_ablation.csv                  # Tier-1 multi-seed reproducibility benchmark report and statistical output: frequency_gate_ablation.csv
│   │       │   ├── plots2/                                      # Empirical distribution plots for gating, beta parameters, and foraging paths
│   │       │   │   ├── v11_beta_distribution.png                    # Tier-1 multi-seed reproducibility benchmark report and statistical output: v11_beta_distribution.png
│   │       │   │   ├── v11_foraging_trajectory.png                  # Tier-1 multi-seed reproducibility benchmark report and statistical output: v11_foraging_trajectory.png
│   │       │   │   ├── v11_generative_prior_recon.png               # Tier-1 multi-seed reproducibility benchmark report and statistical output: v11_generative_prior_recon.png
│   │       │   │   ├── v11_pi_distribution.png                      # Tier-1 multi-seed reproducibility benchmark report and statistical output: v11_pi_distribution.png
│   │       │   │   └── v11_steps_vs_epsilon.png                     # Tier-1 multi-seed reproducibility benchmark report and statistical output: v11_steps_vs_epsilon.png
│   │       │   ├── predictive_coding_ablation.csv               # Tier-1 multi-seed reproducibility benchmark report and statistical output: predictive_coding_ablation.csv
│   │       │   ├── representation_drift_stats.csv               # Tier-1 multi-seed reproducibility benchmark report and statistical output: representation_drift_stats.csv
│   │       │   ├── seeds/                                       # Individual seed evaluation logs recording per-trial accuracy metrics
│   │       │   │   ├── seed_0.json                                  # Tier-1 multi-seed reproducibility benchmark report and statistical output: seed_0.json
│   │       │   │   ├── seed_1337.json                               # Tier-1 multi-seed reproducibility benchmark report and statistical output: seed_1337.json
│   │       │   │   ├── seed_2026.json                               # Tier-1 multi-seed reproducibility benchmark report and statistical output: seed_2026.json
│   │       │   │   ├── seed_42.json                                 # Tier-1 multi-seed reproducibility benchmark report and statistical output: seed_42.json
│   │       │   │   └── seed_9999.json                               # Tier-1 multi-seed reproducibility benchmark report and statistical output: seed_9999.json
│   │       │   └── statistical_comparison.csv                   # Tier-1 multi-seed reproducibility benchmark report and statistical output: statistical_comparison.csv
│   │       └── validate_rhan.py                             # Tier-1 multi-seed reproducibility benchmark report and statistical output: validate_rhan.py
│   ├── rhan-v1-v7/                                  # Early RHAN recurrent prototypes (v1 through v7)
│   │   ├── model_rhan.py                                # Early RHAN prototype implementation and training driver: model_rhan.py
│   │   ├── model_rhan_v3_adaptive.py                    # Early RHAN prototype implementation and training driver: model_rhan_v3_adaptive.py
│   │   ├── model_rhan_v4.py                             # Early RHAN prototype implementation and training driver: model_rhan_v4.py
│   │   ├── model_rhan_v5.py                             # Early RHAN prototype implementation and training driver: model_rhan_v5.py
│   │   ├── model_rhan_v6.py                             # Early RHAN prototype implementation and training driver: model_rhan_v6.py
│   │   ├── model_rhan_v7.py                             # Early RHAN prototype implementation and training driver: model_rhan_v7.py
│   │   ├── train_rhan.py                                # Early RHAN prototype implementation and training driver: train_rhan.py
│   │   ├── train_rhan_v2.py                             # Early RHAN prototype implementation and training driver: train_rhan_v2.py
│   │   ├── train_rhan_v3.py                             # Early RHAN prototype implementation and training driver: train_rhan_v3.py
│   │   ├── train_rhan_v4.py                             # Early RHAN prototype implementation and training driver: train_rhan_v4.py
│   │   ├── train_rhan_v5.py                             # Early RHAN prototype implementation and training driver: train_rhan_v5.py
│   │   ├── train_rhan_v6.py                             # Early RHAN prototype implementation and training driver: train_rhan_v6.py
│   │   └── train_rhan_v7.py                             # Early RHAN prototype implementation and training driver: train_rhan_v7.py
│   ├── stl10-scaleup/                               # STL-10 self-supervised pretraining and pseudo-labeling experiments
│   │   ├── dataset_stl10.py                             # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: dataset_stl10.py
│   │   ├── model_rhan_stl10.py                          # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: model_rhan_stl10.py
│   │   ├── model_rhan_stl10_large.py                    # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: model_rhan_stl10_large.py
│   │   ├── model_rhan_stl10_pretrained.py               # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: model_rhan_stl10_pretrained.py
│   │   ├── train_rhan_large_pseudolabel.py              # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: train_rhan_large_pseudolabel.py
│   │   ├── train_rhan_stl10_tdv.py                      # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: train_rhan_stl10_tdv.py
│   │   └── train_rhan_unified.py                        # STL-10 scale-up self-supervised pretraining and pseudo-labeling driver: train_rhan_unified.py
│   └── working-scratch/                             # Archived scratch scripts and prototype renderers
│       └── scratch/                                     # Diagnostic probes, one-off evaluations, and rendering prototypes
│           ├── _ckpt_verify/                                # Checkpoint parity checks and weight verification logs
│           │   └── comparison.json                              # Archived investigative scratch tool, diagnostic probe, or layout prototype: comparison.json
│           ├── analyze_gaze_perturbation_correlation.py     # Archived investigative scratch tool, diagnostic probe, or layout prototype: analyze_gaze_perturbation_correlation.py
│           ├── build_literature_corpus.py                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: build_literature_corpus.py
│           ├── build_stage2_run_log_md.py                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: build_stage2_run_log_md.py
│           ├── build_synthetic_pt.py                        # Archived investigative scratch tool, diagnostic probe, or layout prototype: build_synthetic_pt.py
│           ├── calc_human_acc.py                            # Archived investigative scratch tool, diagnostic probe, or layout prototype: calc_human_acc.py
│           ├── check_hf_commits.py                          # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_hf_commits.py
│           ├── check_hf_epoch.py                            # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_hf_epoch.py
│           ├── check_hf_latest.py                           # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_hf_latest.py
│           ├── check_hf_latest_commits.py                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_hf_latest_commits.py
│           ├── check_hf_rolling.py                          # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_hf_rolling.py
│           ├── check_local_epoch.py                         # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_local_epoch.py
│           ├── check_stage1_hf.py                           # Archived investigative scratch tool, diagnostic probe, or layout prototype: check_stage1_hf.py
│           ├── compile_pdf.py                               # Archived investigative scratch tool, diagnostic probe, or layout prototype: compile_pdf.py
│           ├── create_resume_checkpoint.py                  # Archived investigative scratch tool, diagnostic probe, or layout prototype: create_resume_checkpoint.py
│           ├── debug_pgd_efficientnet.py                    # Archived investigative scratch tool, diagnostic probe, or layout prototype: debug_pgd_efficientnet.py
│           ├── debug_pgd_resnet.py                          # Archived investigative scratch tool, diagnostic probe, or layout prototype: debug_pgd_resnet.py
│           ├── debug_pgd_vit.py                             # Archived investigative scratch tool, diagnostic probe, or layout prototype: debug_pgd_vit.py
│           ├── diag_cross_mechanism_pi_d.py                 # Archived investigative scratch tool, diagnostic probe, or layout prototype: diag_cross_mechanism_pi_d.py
│           ├── diag_hpc_perclass.py                         # Archived investigative scratch tool, diagnostic probe, or layout prototype: diag_hpc_perclass.py
│           ├── diag_nan_loss.py                             # Archived investigative scratch tool, diagnostic probe, or layout prototype: diag_nan_loss.py
│           ├── diag_nan_source.py                           # Archived investigative scratch tool, diagnostic probe, or layout prototype: diag_nan_source.py
│           ├── diag_v7_fr.py                                # Archived investigative scratch tool, diagnostic probe, or layout prototype: diag_v7_fr.py
│           ├── edge_map_floor_analysis.py                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: edge_map_floor_analysis.py
│           ├── edge_map_floor_model_path.py                 # Archived investigative scratch tool, diagnostic probe, or layout prototype: edge_map_floor_model_path.py
│           ├── eval_aa_phase1.py                            # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_aa_phase1.py
│           ├── eval_lens_e1.py                              # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_lens_e1.py
│           ├── eval_linear_probe.py                         # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_linear_probe.py
│           ├── eval_rhan_fast.py                            # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_rhan_fast.py
│           ├── eval_rhan_stl10.py                           # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_rhan_stl10.py
│           ├── eval_rhan_v5.py                              # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_rhan_v5.py
│           ├── eval_rhan_v7.py                              # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_rhan_v7.py
│           ├── eval_sail_phase1.py                          # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_sail_phase1.py
│           ├── eval_stage2_gate_v2.py                       # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_stage2_gate_v2.py
│           ├── eval_stage2_gate_v3.py                       # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_stage2_gate_v3.py
│           ├── eval_v6_fast.py                              # Archived investigative scratch tool, diagnostic probe, or layout prototype: eval_v6_fast.py
│           ├── extract_sdt.py                               # Archived investigative scratch tool, diagnostic probe, or layout prototype: extract_sdt.py
│           ├── fix_table.py                                 # Archived investigative scratch tool, diagnostic probe, or layout prototype: fix_table.py
│           ├── gen0_pull_hf_evals.py                        # Archived investigative scratch tool, diagnostic probe, or layout prototype: gen0_pull_hf_evals.py
│           ├── gradient_masking_diagnostic.py               # Archived investigative scratch tool, diagnostic probe, or layout prototype: gradient_masking_diagnostic.py
│           ├── inspect_ckpt.py                              # Archived investigative scratch tool, diagnostic probe, or layout prototype: inspect_ckpt.py
│           ├── inspect_v5_weights.py                        # Archived investigative scratch tool, diagnostic probe, or layout prototype: inspect_v5_weights.py
│           ├── list_private_datasets.py                     # Archived investigative scratch tool, diagnostic probe, or layout prototype: list_private_datasets.py
│           ├── literature_corpus_ledger_flat.md             # Archived investigative scratch tool, diagnostic probe, or layout prototype: literature_corpus_ledger_flat.md
│           ├── literature_corpus_notes.md                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: literature_corpus_notes.md
│           ├── measure_hpc_dw.py                            # Archived investigative scratch tool, diagnostic probe, or layout prototype: measure_hpc_dw.py
│           ├── patch_master_stats.py                        # Archived investigative scratch tool, diagnostic probe, or layout prototype: patch_master_stats.py
│           ├── plot_gradient_masking_comparison.py          # Archived investigative scratch tool, diagnostic probe, or layout prototype: plot_gradient_masking_comparison.py
│           ├── plot_selfalign_comparison.py                 # Archived investigative scratch tool, diagnostic probe, or layout prototype: plot_selfalign_comparison.py
│           ├── prototype/                                   # Typst and WeasyPrint layout prototypes for the technical handbook
│           │   ├── prototype_typst.typ                          # Archived investigative scratch tool, diagnostic probe, or layout prototype: prototype_typst.typ
│           │   ├── prototype_weasyprint.html                    # Archived investigative scratch tool, diagnostic probe, or layout prototype: prototype_weasyprint.html
│           │   ├── rendered_typ/                                # Rendered visual pages from Typst handbook layout prototype
│           │   │   ├── page-1.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-1.png
│           │   │   ├── page-2.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-2.png
│           │   │   ├── page-3.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-3.png
│           │   │   ├── page-4.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-4.png
│           │   │   └── page-5.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-5.png
│           │   └── rendered_wp/                                 # Rendered visual pages from WeasyPrint handbook layout prototype
│           │       ├── page-1.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-1.png
│           │       ├── page-2.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-2.png
│           │       ├── page-3.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-3.png
│           │       ├── page-4.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-4.png
│           │       └── page-5.png                                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: page-5.png
│           ├── quick_eval_hf.py                             # Archived investigative scratch tool, diagnostic probe, or layout prototype: quick_eval_hf.py
│           ├── regenerate_e1_table.py                       # Archived investigative scratch tool, diagnostic probe, or layout prototype: regenerate_e1_table.py
│           ├── resync_stage1_roadmap.py                     # Archived investigative scratch tool, diagnostic probe, or layout prototype: resync_stage1_roadmap.py
│           ├── run_hpc_rerun.sh                             # Archived investigative scratch tool, diagnostic probe, or layout prototype: run_hpc_rerun.sh
│           ├── test_bypass.py                               # Archived investigative scratch tool, diagnostic probe, or layout prototype: test_bypass.py
│           ├── test_merge_stage1.py                         # Archived investigative scratch tool, diagnostic probe, or layout prototype: test_merge_stage1.py
│           ├── test_resume_gate.py                          # Archived investigative scratch tool, diagnostic probe, or layout prototype: test_resume_gate.py
│           ├── update_stage1_roadmap_labels.py              # Archived investigative scratch tool, diagnostic probe, or layout prototype: update_stage1_roadmap_labels.py
│           ├── update_stage2_roadmap.py                     # Archived investigative scratch tool, diagnostic probe, or layout prototype: update_stage2_roadmap.py
│           ├── upload_all_checkpoints.py                    # Archived investigative scratch tool, diagnostic probe, or layout prototype: upload_all_checkpoints.py
│           ├── verify_best_vs_rolling.py                    # Archived investigative scratch tool, diagnostic probe, or layout prototype: verify_best_vs_rolling.py
│           ├── verify_cached_v12.py                         # Archived investigative scratch tool, diagnostic probe, or layout prototype: verify_cached_v12.py
│           ├── verify_master_stats.py                       # Archived investigative scratch tool, diagnostic probe, or layout prototype: verify_master_stats.py
│           ├── verify_seed_samples.py                       # Archived investigative scratch tool, diagnostic probe, or layout prototype: verify_seed_samples.py
│           ├── verify_twin_stage_notes.py                   # Archived investigative scratch tool, diagnostic probe, or layout prototype: verify_twin_stage_notes.py
│           ├── verify_v6_compile.py                         # Archived investigative scratch tool, diagnostic probe, or layout prototype: verify_v6_compile.py
│           └── vram_probe.py                                # Archived investigative scratch tool, diagnostic probe, or layout prototype: vram_probe.py
├── checkpoints/                                 # Model weights directory (tracked JSON manifests, local cache, and symlinks)
│   ├── generation1_foundation_roadmap.json          # Structured configuration and experimental metadata: generation1_foundation_roadmap.json
│   ├── rhan_stl10_base_video_tdv.pth:Zone.Identifier # Windows filesystem download zone identifier metadata for rhan_stl10_base_video_tdv.pth
│   ├── rhan_stl10_large_video_tdv.pth:Zone.Identifier # Windows filesystem download zone identifier metadata for rhan_stl10_large_video_tdv.pth
│   ├── rhan_stl10_pseudolabel_best.pth:Zone.Identifier # Windows filesystem download zone identifier metadata for rhan_stl10_pseudolabel_best.pth
│   ├── rhan_stl10_pseudolabel_rolling.pth:Zone.Identifier # Windows filesystem download zone identifier metadata for rhan_stl10_pseudolabel_rolling.pth
│   └── training_complete.json                       # Structured configuration and experimental metadata: training_complete.json
├── cloud/                                       # Cloud deployment configurations, canonical launchers, and background supervision scripts
│   ├── canonical/                                   # Canonical production cloud dispatchers for Kaggle execution
│   │   └── Kaggle_J1_FOUNDATION.py                      # Canonical production dispatcher for Gen-1 foundation ladder training on Kaggle GPUs
│   ├── gen0/                                        # Historical Gen-0 cloud dispatchers for Kaggle and Colab
│   │   ├── Kaggle_NOESIS.py                             # Historical Gen-0 Kaggle dispatcher for RHAN-Next modular ablation runs
│   │   └── colab_notebook_noesis.py                     # Historical Gen-0 Colab interactive runner for early NOESIS cognitive vision experiments
│   ├── gen1/                                        # Generation-1 cloud dispatchers and shell wrappers
│   │   ├── Kaggle_J1_FOUNDATION.py                      # Production Kaggle runner driving the 6-phase foundation ladder with automated HuggingFace sync
│   │   ├── colab_j1_foundation.py                       # Production Colab dispatcher with automated Google Drive checkpoint persistence
│   │   ├── colab_j1_foundation_gate.py                  # Pre-flight validation script checking model weight loading before Colab execution
│   │   ├── run_j1_local.sh                              # Local bash execution script launching foundation ladder training on local GPU hardware
│   │   └── run_j1_supervised.sh                         # Supervisor watchdog wrapper ensuring automated recovery and restart upon interruption
│   └── misc/                                        # Historical Colab/Kaggle setup scripts and experimental runners
│       ├── colab_notebook.py                            # Colab environment bootstrap script installing dependencies and setting up workspace
│       ├── colab_run_pipeline.py                        # Colab pipeline driver executing end-to-end model training and evaluation
│       ├── colab_run_pseudolabel_pipeline.py            # Colab launcher for semi-supervised pseudo-labeling on unlabeled STL-10
│       ├── colab_train_synthetic.py                     # Colab trainer hardening decision boundaries with synthetic augmentations
│       ├── colab_v11_3seed_protocol.py                  # Colab 3-seed protocol runner executing multi-seed training runs for statistical validation
│       ├── colab_v11_isolation_run_a.py                 # Isolation experiment arm A on Colab testing foveation without recurrence
│       ├── colab_v11_isolation_run_b.py                 # Isolation experiment arm B on Colab testing recurrence without foveation
│       ├── colab_v12_step0.py                           # Colab initialization script establishing baseline metrics before v12 training
│       ├── generate_figures.py                          # Cloud-side figure rendering script generating publication plots from training logs
│       ├── generate_rhan_visualizations.py              # Renders foveal glimpse overlays and belief trajectories from cloud run checkpoints
│       ├── generate_rhan_visualizations_v2.py           # Extended visualization script generating multi-panel gaze sequences
│       ├── kaggle_notebook.py                           # Generic Kaggle notebook setup script configuring working directories and datasets
│       ├── kaggle_run_pseudolabel_pipeline.py           # Kaggle runner executing pseudo-label distillation on large unlabeled datasets
│       ├── kaggle_train_synthetic.py                    # Kaggle trainer driving synthetic data generation and contrastive training
│       ├── kaggle_v11_3seed_protocol.py                 # Kaggle multi-seed protocol script running 3 independent random seed runs
│       ├── kaggle_v11_isolation_run_a.py                # Kaggle isolation arm A evaluating foveation modules in isolation
│       ├── kaggle_v11_isolation_run_b.py                # Kaggle isolation arm B evaluating recurrent refinement modules in isolation
│       ├── kaggle_v12_step0.py                          # Kaggle step 0 verification runner auditing hardware environment and PyTorch version
│       ├── kaggle_video_tdv_setup.py                    # Kaggle setup script mounting video datasets for Temporal Difference Visual training
│       └── lightning_setup.py                           # Environment bootstrap script configuring PyTorch Lightning multi-GPU distributed training
├── config/                                      # Central YAML configuration files for models, training, and attack bounds
│   └── train_config.yaml                            # Repository component: train_config.yaml
├── confusion_matrix.png                         # Confusion matrix visualization for baseline ResNet-18 under clean evaluation
├── confusion_matrix_efficientnet.png            # Confusion matrix visualization for EfficientNet-B0 under clean evaluation
├── confusion_matrix_vit.png                     # Confusion matrix visualization for Vision Transformer (ViT-Small) under clean evaluation
├── deep-research-report.md:Zone.Identifier      # Windows filesystem download zone identifier metadata for deep-research-report.md
├── diagnostics/                                 # Runtime profiling artifacts, bisection scripts, and autograd isolation tools
│   └── diagnosis_artifacts/                         # Diagnostic scripts isolating autograd, AMP underflow, and loss landscapes
│       ├── diag_adv_isolation.py                        # Isolates adversarial perturbation graph from model weights to verify clean backward pass
│       ├── diag_adv_isolation_quick.py                  # Fast 10-step adversarial isolation check verifying clean gradient backward pass
│       ├── diag_amp_autograd_grad.py                    # Probes Automatic Mixed Precision (AMP) gradient scaling interaction with torch.autograd.grad
│       ├── diag_amp_bisect.py                           # Bisection probe locating the specific layer causing numerical underflow during fp16 backward
│       ├── diag_amp_cells.py                            # Tests individual recurrent cell operations under mixed precision to identify loss of precision
│       ├── diag_amp_flip.py                             # Tests gradient sign consistency when flipping between fp16 and fp32 execution
│       ├── diag_amp_nested.py                           # Tests nested gradient computation under AMP context managers
│       ├── diag_amp_probe.py                            # Probes tensor value ranges and detects inf/nan in recurrent state under AMP
│       ├── diag_clean_control.py                        # Baseline control run verifying convergence on clean images without adversarial perturbations
│       ├── diag_common.py                               # Shared diagnostic helpers, tensor logging utilities, and assertion checks
│       ├── diag_dataset_forensics.py                    # Audits training dataset normalization, channel means, and label distribution
│       ├── diag_glimpse_sanity.py                       # Validates that foveal glimpses extract valid non-empty spatial crops within bounds [-1, 1]
│       ├── diag_gradient_forensics.py                   # Inspects layer-wise gradient norms to detect gradient vanishing or explosion
│       ├── diag_isolations.py                           # Comprehensive isolation harness testing individual modules in feedforward vs recurrent configurations
│       ├── diag_objective_decomp.py                     # Decomposes multi-task loss into clean CE, TRADES KL, drift, and evidential terms
│       ├── diag_output_sanity.py                        # Validates logit distribution, calibration, and temperature scaling of model outputs
│       ├── diag_tiny_overfit.py                         # Tiny-batch overfit diagnostic verifying model can achieve 100% training accuracy on 8 samples
│       ├── hf_rolling_adv/                              # Adversarial roadmap diagnostics synced from rolling checkpoint repository
│       │   └── generation1_foundation_roadmap.json          # Synced snapshot of the foundation roadmap from HuggingFace
│       ├── out/                                         # JSON diagnostic reports on dataset forensics, gradient norms, and overfit checks
│       │   ├── 04_dataset_forensics.json                    # Forensic diagnostic audit JSON output: 04 dataset forensics
│       │   ├── 05_tiny_overfit.json                         # Forensic diagnostic audit JSON output: 05 tiny overfit
│       │   ├── 06_glimpse_sanity.json                       # Forensic diagnostic audit JSON output: 06 glimpse sanity
│       │   ├── 07_output_sanity.json                        # Forensic diagnostic audit JSON output: 07 output sanity
│       │   ├── 08_gradient_forensics.json                   # Forensic diagnostic audit JSON output: 08 gradient forensics
│       │   ├── 09_objective_decomp.json                     # Forensic diagnostic audit JSON output: 09 objective decomp
│       │   ├── 10_section_a_b_ladder.json                   # Forensic diagnostic audit JSON output: 10 section a b ladder
│       │   ├── 11_12_13_isolations.json                     # Forensic diagnostic audit JSON output: 11 12 13 isolations
│       │   ├── 15_amp_autograd_grad.json                    # Forensic diagnostic audit JSON output: 15 amp autograd grad
│       │   ├── 15_amp_bisect.json                           # Forensic diagnostic audit JSON output: 15 amp bisect
│       │   ├── 15_amp_cells.json                            # Forensic diagnostic audit JSON output: 15 amp cells
│       │   ├── 15_amp_flip.json                             # Forensic diagnostic audit JSON output: 15 amp flip
│       │   ├── 15_amp_nested.json                           # Forensic diagnostic audit JSON output: 15 amp nested
│       │   ├── 15_amp_probe.json                            # Forensic diagnostic audit JSON output: 15 amp probe
│       │   └── 17_final_forensic_evidence.json              # Forensic diagnostic audit JSON output: 17 final forensic evidence
│       └── run_all_diag.py                              # Master diagnostic runner executing the complete battery of forensic checks and writing JSON reports
├── docs/                                        # Authoritative repository map, architecture specifications, technical decisions, and project roadmaps
│   ├── ACD_Project_Documentation_v3.pdf             # Compiled project documentation PDF covering experimental phases 1 through 5
│   ├── ACD_Project_Documentation_v3.pdf:Zone.Identifier # Windows filesystem download zone identifier metadata for ACD_Project_Documentation_v3.pdf
│   ├── ACD_paper_v1.tex                             # First-generation LaTeX paper draft on Adversarial Cognitive Divergence
│   ├── ACD_v4.pdf                                   # Compiled comprehensive project report PDF for ACD version 4.0
│   ├── ACD_v4.pdf:Zone.Identifier                   # Windows filesystem download zone identifier metadata for ACD_v4.pdf
│   ├── ARCHITECTURE.md                              # Mathematical architecture specification: perceptual loop equations, state variables, and agent boundaries
│   ├── CANCELLATION_NOTICE_2026-10-03.md            # Formal cancellation notice of obsolete training run branch
│   ├── FORENSIC_REPORT_NXA_GENERATION1.md           # Forensic audit analyzing Gen-1 mechanisms, training dynamics, and empirical anomalies
│   ├── Manual Documentation.md                      # Developer manual documentation covering environment configuration and scripts
│   ├── NOESIS_FOUNDATION.md                         # Theoretical foundations: ventral stream neuroscience, active inference, and predictive coding
│   ├── REPOSITORY_MAP.md                            # Ground-truth 1,641-line descriptive inventory mapping every file, subsystem, and historical lineage
│   ├── RHAN_NXA_ARCHITECTURE.md                     # High-level system architecture document detailing the RHAN-NXA cognitive vision engine
│   ├── STAGE2_REFACTOR_PLAN.md                      # Refactoring specification for Stage 2 modularization and agent decoupling
│   ├── Stage_E1-3_Analysis.md                       # Empirical analysis across experimental stages E1 through E3
│   ├── git_branch_audit_before_cleanup.md           # Branch audit documentation detailing historical branch lineage and rationale
│   ├── git_branch_cleanup_report.md                 # Cleanup report recording merged, archived, and pruned git branches
│   ├── historical/                                  # Archived stage reports, early proposal drafts, and pre-registrations
│   │   ├── COLLABORATING.md                             # Historical team collaboration guidelines, task assignments, and review standards
│   │   ├── FINDINGS.md                                  # Historical empirical findings log: perceptual threshold collapse and SDT separation
│   │   ├── IMPORTANT.md                                 # Critical operational warnings and known gotchas from earlier development phases
│   │   ├── NOESIS_IMPROVEMENT_CATALOG.md                # Catalog of proposed architectural improvements and ablation targets
│   │   ├── RHAN-history.md                              # Historical narrative tracing the evolution of RHAN from v1 through v12
│   │   ├── RHANv11.md                                   # Technical notes and mathematical formulations specific to the RHAN v11 prototype
│   │   └── deep-research-report.md                      # Deep research analysis on active visual inference, saccadic sampling, and adversarial robustness
│   ├── repository_reorganization.md                 # Documentation of repository folder reorganization and canonical paths
│   ├── research/                                    # Research references, claim audits, literature corpus, and empirical seed tables
│   │   ├── RHAN_NOESIS_CLAIM_IMPLEMENTATION_AUDIT.md    # Audit reconciling theoretical claims with exact line-by-line code implementations
│   │   ├── RHAN_NOESIS_EXPERIMENT_REGISTRY.csv          # Tabular experiment registry listing all conducted runs, seeds, and outcomes
│   │   ├── RHAN_NOESIS_GENERATION0_LESSONS.md           # Post-mortem analysis of Gen-0 empirical results and experimental pitfalls
│   │   ├── RHAN_NOESIS_LITERATURE_CORPUS.json           # Structured JSON database of cited neuroscience and computer vision literature
│   │   ├── RHAN_NOESIS_LITERATURE_CORPUS.md             # Annotated bibliography of key literature grounding the cognitive vision architecture
│   │   ├── RHAN_NOESIS_PROJECT_EVIDENCE_DOSSIER.md      # Consolidated evidence dossier compiling empirical proofs for core project hypotheses
│   │   └── RHAN_NOESIS_REFERENCE_SEED.csv               # Repository component: RHAN_NOESIS_REFERENCE_SEED.csv
│   ├── rhan_next_roadmap.json                       # Machine-readable roadmap tracking stage milestones and completion gates
│   ├── rhan_nxa/                                    # RHAN-NXA unified technical handbook, QA inspection reports, and publication figure gallery
│   │   ├── DOCUMENTATION_SOURCE_MAP.md                  # Technical markdown documentation and specification: DOCUMENTATION_SOURCE_MAP.md
│   │   ├── HANDBOOK_QA_REPORT.md                        # Technical markdown documentation and specification: HANDBOOK_QA_REPORT.md
│   │   ├── RENDERER_DECISION_RECORD.md                  # Technical markdown documentation and specification: RENDERER_DECISION_RECORD.md
│   │   ├── RHAN_NXA_Handbook_Unified.md                 # Technical markdown documentation and specification: RHAN_NXA_Handbook_Unified.md
│   │   ├── appendices/                                  # Technical appendices covering tensor dimensions, ABC interfaces, promotion gates, and math proofs
│   │   │   ├── A_Tensor_Reference.md                        # Technical handbook appendix: A Tensor Reference (formal interfaces, dimensions, and proofs)
│   │   │   ├── B_Interface_ABCs.md                          # Technical handbook appendix: B Interface ABCs (formal interfaces, dimensions, and proofs)
│   │   │   └── C_Port_Table.md                              # Technical handbook appendix: C Port Table (formal interfaces, dimensions, and proofs)
│   │   ├── book/                                        # 27-chapter textbook detailing the theoretical foundations, implementation contracts, and empirical audits
│   │   │   ├── 00_Executive_Overview.md                     # Technical handbook chapter: 00 Executive Overview (theoretical foundations and implementation contracts)
│   │   │   ├── 01_Why_RHAN_NXA.md                           # Technical handbook chapter: 01 Why RHAN NXA (theoretical foundations and implementation contracts)
│   │   │   ├── 02_Core_Idea.md                              # Technical handbook chapter: 02 Core Idea (theoretical foundations and implementation contracts)
│   │   │   ├── 03_Perception_As_Investigation.md            # Technical handbook chapter: 03 Perception As Investigation (theoretical foundations and implementation contracts)
│   │   │   ├── 04_Belief_State.md                           # Technical handbook chapter: 04 Belief State (theoretical foundations and implementation contracts)
│   │   │   ├── 05_z_State.md                                # Technical handbook chapter: 05 z State (theoretical foundations and implementation contracts)
│   │   │   ├── 06_Structure_State.md                        # Technical handbook chapter: 06 Structure State (theoretical foundations and implementation contracts)
│   │   │   ├── 07_Uncertainty.md                            # Technical handbook chapter: 07 Uncertainty (theoretical foundations and implementation contracts)
│   │   │   ├── 08_Prediction_Error.md                       # Technical handbook chapter: 08 Prediction Error (theoretical foundations and implementation contracts)
│   │   │   ├── 09_Recurrence.md                             # Technical handbook chapter: 09 Recurrence (theoretical foundations and implementation contracts)
│   │   │   ├── 10_Foveation.md                              # Technical handbook chapter: 10 Foveation (theoretical foundations and implementation contracts)
│   │   │   ├── 11_AIS_v2.md                                 # Technical handbook chapter: 11 AIS v2 (theoretical foundations and implementation contracts)
│   │   │   ├── 12_Complete_Perceptual_Loop.md               # Technical handbook chapter: 12 Complete Perceptual Loop (theoretical foundations and implementation contracts)
│   │   │   ├── 13_Architecture.md                           # Technical handbook chapter: 13 Architecture (theoretical foundations and implementation contracts)
│   │   │   ├── 14_Gradient_Flow.md                          # Technical handbook chapter: 14 Gradient Flow (theoretical foundations and implementation contracts)
│   │   │   ├── 15_Tensor_Shapes.md                          # Technical handbook chapter: 15 Tensor Shapes (theoretical foundations and implementation contracts)
│   │   │   ├── 16_Training_System.md                        # Technical handbook chapter: 16 Training System (theoretical foundations and implementation contracts)
│   │   │   ├── 17_Experimental_DAG.md                       # Technical handbook chapter: 17 Experimental DAG (theoretical foundations and implementation contracts)
│   │   │   ├── 18_Evaluation.md                             # Technical handbook chapter: 18 Evaluation (theoretical foundations and implementation contracts)
│   │   │   ├── 19_Gen0_Evidence.md                          # Technical handbook chapter: 19 Gen0 Evidence (theoretical foundations and implementation contracts)
│   │   │   ├── 20_L_stab.md                                 # Technical handbook chapter: 20 L stab (theoretical foundations and implementation contracts)
│   │   │   ├── 21_Decision_Records.md                       # Technical handbook chapter: 21 Decision Records (theoretical foundations and implementation contracts)
│   │   │   ├── 22_Common_Misunderstandings.md               # Technical handbook chapter: 22 Common Misunderstandings (theoretical foundations and implementation contracts)
│   │   │   ├── 23_Glossary.md                               # Technical handbook chapter: 23 Glossary (theoretical foundations and implementation contracts)
│   │   │   ├── 24_Scope_Boundaries.md                       # Technical handbook chapter: 24 Scope Boundaries (theoretical foundations and implementation contracts)
│   │   │   ├── 25_Future_Directions.md                      # Technical handbook chapter: 25 Future Directions (theoretical foundations and implementation contracts)
│   │   │   └── 26_Reproducibility.md                        # Technical handbook chapter: 26 Reproducibility (theoretical foundations and implementation contracts)
│   │   ├── design/                                      # Subsystem directory: design/
│   │   │   └── DESIGN_SYSTEM.md                             # Technical markdown documentation and specification: DESIGN_SYSTEM.md
│   │   ├── figure_gallery/                              # Standalone HTML figure gallery and high-resolution architectural diagram assets
│   │   │   ├── INDEX.html                                   # Interactive HTML figure gallery displaying publication-quality architecture schematics
│   │   │   ├── ais_v2.png                                   # High-resolution figure gallery image asset: ais_v2.png
│   │   │   ├── belief_state.png                             # High-resolution figure gallery image asset: belief_state.png
│   │   │   ├── checkpoint_resume.png                        # High-resolution figure gallery image asset: checkpoint_resume.png
│   │   │   ├── compact_vit.png                              # High-resolution figure gallery image asset: compact_vit.png
│   │   │   ├── decision_status_system.png                   # High-resolution figure gallery image asset: decision_status_system.png
│   │   │   ├── experiment_isolation.png                     # High-resolution figure gallery image asset: experiment_isolation.png
│   │   │   ├── foveation.png                                # High-resolution figure gallery image asset: foveation.png
│   │   │   ├── future_rhan_llm.png                          # High-resolution figure gallery image asset: future_rhan_llm.png
│   │   │   ├── gen0_confound.png                            # High-resolution figure gallery image asset: gen0_confound.png
│   │   │   ├── gen0_robustness_chart.png                    # High-resolution figure gallery image asset: gen0_robustness_chart.png
│   │   │   ├── gradient_flow.png                            # High-resolution figure gallery image asset: gradient_flow.png
│   │   │   ├── l_stab_stability.png                         # High-resolution figure gallery image asset: l_stab_stability.png
│   │   │   ├── perceptual_loop.png                          # High-resolution figure gallery image asset: perceptual_loop.png
│   │   │   ├── prediction_error.png                         # High-resolution figure gallery image asset: prediction_error.png
│   │   │   ├── recurrence.png                               # High-resolution figure gallery image asset: recurrence.png
│   │   │   ├── system_overview.png                          # High-resolution figure gallery image asset: system_overview.png
│   │   │   ├── tensor_flow.png                              # High-resolution figure gallery image asset: tensor_flow.png
│   │   │   ├── training_dag.png                             # High-resolution figure gallery image asset: training_dag.png
│   │   │   └── uncertainty_dirichlet.png                    # High-resolution figure gallery image asset: uncertainty_dirichlet.png
│   │   ├── figures/                                     # Vector SVG publication diagrams illustrating the active inference loop and brain mappings
│   │   │   ├── ais_v2.svg                                   # Vector architectural publication diagram: ais v2
│   │   │   ├── belief_state.svg                             # Vector architectural publication diagram: belief state
│   │   │   ├── checkpoint_resume.svg                        # Vector architectural publication diagram: checkpoint resume
│   │   │   ├── compact_vit.svg                              # Vector architectural publication diagram: compact vit
│   │   │   ├── decision_status_system.svg                   # Vector architectural publication diagram: decision status system
│   │   │   ├── experiment_isolation.svg                     # Vector architectural publication diagram: experiment isolation
│   │   │   ├── foveation.svg                                # Vector architectural publication diagram: foveation
│   │   │   ├── future_rhan_llm.svg                          # Vector architectural publication diagram: future rhan llm
│   │   │   ├── gen0_confound.svg                            # Vector architectural publication diagram: gen0 confound
│   │   │   ├── gen0_robustness_chart.svg                    # Vector architectural publication diagram: gen0 robustness chart
│   │   │   ├── gradient_flow.svg                            # Vector architectural publication diagram: gradient flow
│   │   │   ├── l_stab_stability.svg                         # Vector architectural publication diagram: l stab stability
│   │   │   ├── perceptual_loop.svg                          # Vector architectural publication diagram: perceptual loop
│   │   │   ├── prediction_error.svg                         # Vector architectural publication diagram: prediction error
│   │   │   ├── recurrence.svg                               # Vector architectural publication diagram: recurrence
│   │   │   ├── system_overview.svg                          # Vector architectural publication diagram: system overview
│   │   │   ├── tensor_flow.svg                              # Vector architectural publication diagram: tensor flow
│   │   │   ├── training_dag.svg                             # Vector architectural publication diagram: training dag
│   │   │   └── uncertainty_dirichlet.svg                    # Vector architectural publication diagram: uncertainty dirichlet
│   │   └── pdf_qa_pages/                                # Rendered visual QA inspection pages for verifying the compiled PDF handbook layout
│   │       ├── page-001.png                                 # Handbook visual QA layout proof: page page-001 inspection rendering
│   │       ├── page-002.png                                 # Handbook visual QA layout proof: page page-002 inspection rendering
│   │       ├── page-003.png                                 # Handbook visual QA layout proof: page page-003 inspection rendering
│   │       ├── page-004.png                                 # Handbook visual QA layout proof: page page-004 inspection rendering
│   │       ├── page-005.png                                 # Handbook visual QA layout proof: page page-005 inspection rendering
│   │       ├── page-006.png                                 # Handbook visual QA layout proof: page page-006 inspection rendering
│   │       ├── page-007.png                                 # Handbook visual QA layout proof: page page-007 inspection rendering
│   │       ├── page-008.png                                 # Handbook visual QA layout proof: page page-008 inspection rendering
│   │       ├── page-009.png                                 # Handbook visual QA layout proof: page page-009 inspection rendering
│   │       ├── page-010.png                                 # Handbook visual QA layout proof: page page-010 inspection rendering
│   │       ├── page-011.png                                 # Handbook visual QA layout proof: page page-011 inspection rendering
│   │       └── page-012.png                                 # Handbook visual QA layout proof: page page-012 inspection rendering
│   ├── stage3_environment.json                      # Hardware specification and environment fingerprint for Stage 3 experiments
│   └── stage3_preregistration.md                    # Formal preregistered protocol and testable hypotheses for Stage 3 runs
├── evaluation/                                  # Hardened evaluation harness: normalized-space clean/robust accuracy, Geirhos cue-conflict, compactness accounting
│   ├── clean_and_robust.py                          # Standardized evaluation harness: normalized-space clean accuracy, PGD-50, and APGD worst-case robust metrics
│   ├── compactness_report.py                        # Profiles model parameter counts, FLOPs, memory footprint, and inference latency
│   ├── geirhos_loader.py                            # Dataloader for the Geirhos cue-conflict dataset evaluating shape vs texture bias
│   ├── imagenet100_loader.py                        # ImageNet-100 dataset loader with normalization, transforms, and caching
│   ├── imagenet_c_loader.py                         # ImageNet-C corruption benchmark loader across 15 synthetic noise types
│   └── shape_texture_bias.py                        # Signal Detection Theory (d') script measuring shape vs texture decision bias
├── figures/                                     # Publication figures v1 (architectures, loss landscapes, information flows)
│   ├── README.md                                    # Master repository overview, empirical benchmark findings, and exhaustive repo map
│   ├── architecture/                                # Architectural diagrams for foveation, gist encoder, and predictive loop
│   │   ├── figure_1_rhan_architecture.pdf               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture.pdf
│   │   ├── figure_1_rhan_architecture.png               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture.png
│   │   ├── figure_1_rhan_architecture.svg               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture.svg
│   │   ├── figure_2_layer_by_layer.pdf                  # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_2_layer_by_layer.pdf
│   │   ├── figure_2_layer_by_layer.png                  # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_2_layer_by_layer.png
│   │   ├── figure_2_layer_by_layer.svg                  # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_2_layer_by_layer.svg
│   │   ├── figure_3_computational_graph.pdf             # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_3_computational_graph.pdf
│   │   ├── figure_3_computational_graph.png             # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_3_computational_graph.png
│   │   ├── figure_3_computational_graph.svg             # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_3_computational_graph.svg
│   │   ├── figure_4_module_hierarchy.pdf                # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_4_module_hierarchy.pdf
│   │   ├── figure_4_module_hierarchy.png                # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_4_module_hierarchy.png
│   │   ├── figure_4_module_hierarchy.svg                # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_4_module_hierarchy.svg
│   │   ├── figure_6_activation_flow.pdf                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_6_activation_flow.pdf
│   │   ├── figure_6_activation_flow.png                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_6_activation_flow.png
│   │   ├── figure_6_activation_flow.svg                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_6_activation_flow.svg
│   │   ├── figure_7_tensor_shape_evolution.pdf          # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_7_tensor_shape_evolution.pdf
│   │   ├── figure_7_tensor_shape_evolution.png          # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_7_tensor_shape_evolution.png
│   │   └── figure_7_tensor_shape_evolution.svg          # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_7_tensor_shape_evolution.svg
│   ├── biological/                                  # Comparative schematics illustrating primate ventral stream parallels
│   │   ├── figure_10_biological_mapping.pdf             # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_biological_mapping.pdf
│   │   ├── figure_10_biological_mapping.png             # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_biological_mapping.png
│   │   └── figure_10_biological_mapping.svg             # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_biological_mapping.svg
│   ├── concept_activation_ablation.png              # Publication visual asset illustrating experimental dynamics: concept_activation_ablation.png
│   ├── evaluation/                                  # Robustness curves, attack comparison bar plots, and error rate tables
│   │   ├── figure_16_evaluation_dashboard.pdf           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_16_evaluation_dashboard.pdf
│   │   ├── figure_16_evaluation_dashboard.png           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_16_evaluation_dashboard.png
│   │   ├── figure_16_evaluation_dashboard.svg           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_16_evaluation_dashboard.svg
│   │   ├── figure_8_memory_consumption.pdf              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_memory_consumption.pdf
│   │   ├── figure_8_memory_consumption.png              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_memory_consumption.png
│   │   ├── figure_8_memory_consumption.svg              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_memory_consumption.svg
│   │   ├── figure_9_flops_distribution.pdf              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_flops_distribution.pdf
│   │   ├── figure_9_flops_distribution.png              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_flops_distribution.png
│   │   └── figure_9_flops_distribution.svg              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_flops_distribution.svg
│   ├── explainability/                              # Grad-CAM saliency maps and attention rollouts under adversarial attacks
│   │   ├── figure_k2_candidate_grid.png                 # Publication visual asset illustrating experimental dynamics: figure_k2_candidate_grid.png
│   │   ├── figure_k2_candidate_grid_eps10.png           # Publication visual asset illustrating experimental dynamics: figure_k2_candidate_grid_eps10.png
│   │   ├── figure_k2_candidate_grid_eps20.png           # Publication visual asset illustrating experimental dynamics: figure_k2_candidate_grid_eps20.png
│   │   ├── figure_k2_metadata.json                      # Publication visual asset illustrating experimental dynamics: figure_k2_metadata.json
│   │   ├── figure_k2_progressive_visual_robustness_dark.png # Publication visual asset illustrating experimental dynamics: figure_k2_progressive_visual_robustness_dark.png
│   │   ├── figure_k2_progressive_visual_robustness_light.pdf # Publication visual asset illustrating experimental dynamics: figure_k2_progressive_visual_robustness_light.pdf
│   │   ├── figure_k2_progressive_visual_robustness_light.png # Publication visual asset illustrating experimental dynamics: figure_k2_progressive_visual_robustness_light.png
│   │   ├── figure_k2_progressive_visual_robustness_light.svg # Publication visual asset illustrating experimental dynamics: figure_k2_progressive_visual_robustness_light.svg
│   │   ├── figure_k2_progressive_visual_robustness_transparent.png # Publication visual asset illustrating experimental dynamics: figure_k2_progressive_visual_robustness_transparent.png
│   │   ├── figure_k3_metadata.json                      # Publication visual asset illustrating experimental dynamics: figure_k3_metadata.json
│   │   ├── figure_k3_progressive_visual_robustness_dark.png # Publication visual asset illustrating experimental dynamics: figure_k3_progressive_visual_robustness_dark.png
│   │   ├── figure_k3_progressive_visual_robustness_light.pdf # Publication visual asset illustrating experimental dynamics: figure_k3_progressive_visual_robustness_light.pdf
│   │   ├── figure_k3_progressive_visual_robustness_light.png # Publication visual asset illustrating experimental dynamics: figure_k3_progressive_visual_robustness_light.png
│   │   ├── figure_k3_progressive_visual_robustness_light.svg # Publication visual asset illustrating experimental dynamics: figure_k3_progressive_visual_robustness_light.svg
│   │   └── figure_k3_progressive_visual_robustness_transparent.png # Publication visual asset illustrating experimental dynamics: figure_k3_progressive_visual_robustness_transparent.png
│   ├── figure_10_training_pipeline.png              # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_training_pipeline.png
│   ├── figure_10_training_pipeline.svg              # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_training_pipeline.svg
│   ├── figure_11_loss_landscape.png                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_loss_landscape.png
│   ├── figure_11_loss_landscape.svg                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_loss_landscape.svg
│   ├── figure_12_information_flow.png               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_information_flow.png
│   ├── figure_12_information_flow.svg               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_information_flow.svg
│   ├── figure_13_rhan_ecosystem.png                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_rhan_ecosystem.png
│   ├── figure_13_rhan_ecosystem.svg                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_rhan_ecosystem.svg
│   ├── figure_1_rhan_architecture.png               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture.png
│   ├── figure_1_rhan_architecture.svg               # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture.svg
│   ├── figure_2_biological_inspiration.png          # Comparative schematic illustrating primate visual ventral stream parallels: figure_2_biological_inspiration.png
│   ├── figure_2_biological_inspiration.svg          # Comparative schematic illustrating primate visual ventral stream parallels: figure_2_biological_inspiration.svg
│   ├── figure_3_predictive_coding_loop.png          # Top-down predictive coding loop diagram showing error residual minimization: figure_3_predictive_coding_loop.png
│   ├── figure_3_predictive_coding_loop.svg          # Top-down predictive coding loop diagram showing error residual minimization: figure_3_predictive_coding_loop.svg
│   ├── figure_4_dual_transformer.png                # Dual-stream transformer schematic separating global low-frequency gist and foveal patches: figure_4_dual_transformer.png
│   ├── figure_4_dual_transformer.svg                # Dual-stream transformer schematic separating global low-frequency gist and foveal patches: figure_4_dual_transformer.svg
│   ├── figure_5_spherical_prototype.png             # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_spherical_prototype.png
│   ├── figure_5_spherical_prototype.svg             # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_spherical_prototype.svg
│   ├── figure_6_sail_flowchart.png                  # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_6_sail_flowchart.png
│   ├── figure_6_sail_flowchart.svg                  # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_6_sail_flowchart.svg
│   ├── figure_7_geometric_sail.png                  # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_7_geometric_sail.png
│   ├── figure_7_geometric_sail.svg                  # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_7_geometric_sail.svg
│   ├── figure_8_tdv_framework.png                   # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_tdv_framework.png
│   ├── figure_8_tdv_framework.svg                   # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_tdv_framework.svg
│   ├── figure_9_adversarial_tdv.png                 # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_adversarial_tdv.png
│   ├── figure_9_adversarial_tdv.svg                 # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_adversarial_tdv.svg
│   ├── geometry/                                    # Latent space geometry visualizations and decision boundary plots
│   │   ├── figure_5_parameter_distribution.pdf          # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_parameter_distribution.pdf
│   │   ├── figure_5_parameter_distribution.png          # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_parameter_distribution.png
│   │   ├── figure_5_parameter_distribution.svg          # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_parameter_distribution.svg
│   │   ├── figure_spherical_prototype.pdf               # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_spherical_prototype.pdf
│   │   ├── figure_spherical_prototype.png               # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_spherical_prototype.png
│   │   └── figure_spherical_prototype.svg               # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_spherical_prototype.svg
│   ├── gradient_masking_comparison.png              # Publication visual asset illustrating experimental dynamics: gradient_masking_comparison.png
│   ├── losses/                                      # Loss surface contour plots and convergence trajectory curves
│   │   ├── figure_loss_landscape.pdf                    # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_loss_landscape.pdf
│   │   ├── figure_loss_landscape.png                    # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_loss_landscape.png
│   │   └── figure_loss_landscape.svg                    # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_loss_landscape.svg
│   ├── pgd_accuracy_decay.png                       # Empirical accuracy collapse curve under increasing adversarial perturbation radius: pgd_accuracy_decay.png
│   ├── sdt_sensitivity_decay.png                    # Empirical accuracy collapse curve under increasing adversarial perturbation radius: sdt_sensitivity_decay.png
│   ├── selfalign_accuracy_decay.png                 # Empirical accuracy collapse curve under increasing adversarial perturbation radius: selfalign_accuracy_decay.png
│   ├── selfalign_sensitivity_decay.png              # Empirical accuracy collapse curve under increasing adversarial perturbation radius: selfalign_sensitivity_decay.png
│   └── training/                                    # Training loss trajectories, learning rate curves, and gradient norm plots
│       ├── figure_11_predictive_coding_loop.pdf         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_predictive_coding_loop.pdf
│       ├── figure_11_predictive_coding_loop.png         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_predictive_coding_loop.png
│       ├── figure_11_predictive_coding_loop.svg         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_predictive_coding_loop.svg
│       ├── figure_12_sail_pipeline.pdf                  # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_sail_pipeline.pdf
│       ├── figure_12_sail_pipeline.png                  # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_sail_pipeline.png
│       ├── figure_12_sail_pipeline.svg                  # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_sail_pipeline.svg
│       ├── figure_13_tdv_pipeline.pdf                   # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_tdv_pipeline.pdf
│       ├── figure_13_tdv_pipeline.png                   # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_tdv_pipeline.png
│       ├── figure_13_tdv_pipeline.svg                   # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_tdv_pipeline.svg
│       ├── figure_14_rhan_ecosystem.pdf                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_14_rhan_ecosystem.pdf
│       ├── figure_14_rhan_ecosystem.png                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_14_rhan_ecosystem.png
│       ├── figure_14_rhan_ecosystem.svg                 # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_14_rhan_ecosystem.svg
│       ├── figure_15_training_pipeline.pdf              # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_15_training_pipeline.pdf
│       ├── figure_15_training_pipeline.png              # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_15_training_pipeline.png
│       └── figure_15_training_pipeline.svg              # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_15_training_pipeline.svg
├── figures_v2/                                  # Publication figures v2 (attention maps, biology, geometry, frequency analysis)
│   ├── README.md                                    # Master repository overview, empirical benchmark findings, and exhaustive repo map
│   ├── attention/                                   # Gaze fixation density maps and foveal crop trajectory overlays
│   │   ├── figure_b1_attention_overlay_dark.png         # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_dark.png
│   │   ├── figure_b1_attention_overlay_light.pdf        # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_light.pdf
│   │   ├── figure_b1_attention_overlay_light.png        # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_light.png
│   │   ├── figure_b1_attention_overlay_light.svg        # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_light.svg
│   │   ├── figure_b1_attention_overlay_transparent.png  # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_transparent.png
│   │   ├── figure_b2_attention_evolution_dark.png       # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_dark.png
│   │   ├── figure_b2_attention_evolution_light.pdf      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_light.pdf
│   │   ├── figure_b2_attention_evolution_light.png      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_light.png
│   │   ├── figure_b2_attention_evolution_light.svg      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_light.svg
│   │   ├── figure_b2_attention_evolution_transparent.png # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_transparent.png
│   │   ├── figure_b3_feedback_correction_dark.png       # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_dark.png
│   │   ├── figure_b3_feedback_correction_light.pdf      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_light.pdf
│   │   ├── figure_b3_feedback_correction_light.png      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_light.png
│   │   ├── figure_b3_feedback_correction_light.svg      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_light.svg
│   │   └── figure_b3_feedback_correction_transparent.png # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_transparent.png
│   ├── biology/                                     # Ventral cortex area comparison (V1, V2, V4, IT) vs neural network stages
│   │   ├── figure_f1_predictive_coding_convergence_dark.png # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_dark.png
│   │   ├── figure_f1_predictive_coding_convergence_light.pdf # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_light.pdf
│   │   ├── figure_f1_predictive_coding_convergence_light.png # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_light.png
│   │   ├── figure_f1_predictive_coding_convergence_light.svg # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_light.svg
│   │   ├── figure_f1_predictive_coding_convergence_transparent.png # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_transparent.png
│   │   ├── figure_f2_act_pondering_dark.png             # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_dark.png
│   │   ├── figure_f2_act_pondering_light.pdf            # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_light.pdf
│   │   ├── figure_f2_act_pondering_light.png            # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_light.png
│   │   ├── figure_f2_act_pondering_light.svg            # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_light.svg
│   │   ├── figure_f2_act_pondering_transparent.png      # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_transparent.png
│   │   ├── figure_f3_recurrence_utilization_dark.png    # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_dark.png
│   │   ├── figure_f3_recurrence_utilization_light.pdf   # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_light.pdf
│   │   ├── figure_f3_recurrence_utilization_light.png   # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_light.png
│   │   ├── figure_f3_recurrence_utilization_light.svg   # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_light.svg
│   │   └── figure_f3_recurrence_utilization_transparent.png # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_transparent.png
│   ├── diagnostics/                                 # Loss surface conditioning and gradient norm stability diagnostics
│   │   ├── figure_i1_diagnostics_dashboard_dark.png     # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_dark.png
│   │   ├── figure_i1_diagnostics_dashboard_light.pdf    # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_light.pdf
│   │   ├── figure_i1_diagnostics_dashboard_light.png    # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_light.png
│   │   ├── figure_i1_diagnostics_dashboard_light.svg    # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_light.svg
│   │   └── figure_i1_diagnostics_dashboard_transparent.png # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_transparent.png
│   ├── evaluation/                                  # Multi-model robustness curves comparing clean vs PGD-50 accuracy
│   │   ├── figure_e1_accuracy_vs_epsilon_dark.png       # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_dark.png
│   │   ├── figure_e1_accuracy_vs_epsilon_light.pdf      # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_light.pdf
│   │   ├── figure_e1_accuracy_vs_epsilon_light.png      # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_light.png
│   │   ├── figure_e1_accuracy_vs_epsilon_light.svg      # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_light.svg
│   │   ├── figure_e1_accuracy_vs_epsilon_transparent.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_transparent.png
│   │   ├── figure_e2_dprime_vs_epsilon_dark.png         # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_dark.png
│   │   ├── figure_e2_dprime_vs_epsilon_light.pdf        # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_light.pdf
│   │   ├── figure_e2_dprime_vs_epsilon_light.png        # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_light.png
│   │   ├── figure_e2_dprime_vs_epsilon_light.svg        # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_light.svg
│   │   ├── figure_e2_dprime_vs_epsilon_transparent.png  # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_transparent.png
│   │   ├── figure_e3_robustness_threshold_comparison_dark.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_dark.png
│   │   ├── figure_e3_robustness_threshold_comparison_light.pdf # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_light.pdf
│   │   ├── figure_e3_robustness_threshold_comparison_light.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_light.png
│   │   ├── figure_e3_robustness_threshold_comparison_light.svg # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_light.svg
│   │   ├── figure_e3_robustness_threshold_comparison_transparent.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_transparent.png
│   │   ├── figure_e4_class_robustness_heatmap_dark.png  # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_dark.png
│   │   ├── figure_e4_class_robustness_heatmap_light.pdf # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_light.pdf
│   │   ├── figure_e4_class_robustness_heatmap_light.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_light.png
│   │   ├── figure_e4_class_robustness_heatmap_light.svg # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_light.svg
│   │   └── figure_e4_class_robustness_heatmap_transparent.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_transparent.png
│   ├── explainability/                              # Feature attribution maps comparing standard vs adversarially trained models
│   │   ├── figure_j1_explainability_gradcam_dark.png    # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_dark.png
│   │   ├── figure_j1_explainability_gradcam_light.pdf   # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_light.pdf
│   │   ├── figure_j1_explainability_gradcam_light.png   # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_light.png
│   │   ├── figure_j1_explainability_gradcam_light.svg   # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_light.svg
│   │   └── figure_j1_explainability_gradcam_transparent.png # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_transparent.png
│   ├── frequency/                                   # Radial frequency decomposition of adversarial perturbation energy
│   │   ├── figure_c1_low_frequency_gate_dark.png        # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_dark.png
│   │   ├── figure_c1_low_frequency_gate_light.pdf       # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_light.pdf
│   │   ├── figure_c1_low_frequency_gate_light.png       # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_light.png
│   │   ├── figure_c1_low_frequency_gate_light.svg       # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_light.svg
│   │   ├── figure_c1_low_frequency_gate_transparent.png # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_transparent.png
│   │   ├── figure_c2_high_frequency_gate_dark.png       # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_dark.png
│   │   ├── figure_c2_high_frequency_gate_light.pdf      # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_light.pdf
│   │   ├── figure_c2_high_frequency_gate_light.png      # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_light.png
│   │   ├── figure_c2_high_frequency_gate_light.svg      # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_light.svg
│   │   ├── figure_c2_high_frequency_gate_transparent.png # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_transparent.png
│   │   ├── figure_c3_gate_weights_training_dark.png     # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_dark.png
│   │   ├── figure_c3_gate_weights_training_light.pdf    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_light.pdf
│   │   ├── figure_c3_gate_weights_training_light.png    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_light.png
│   │   ├── figure_c3_gate_weights_training_light.svg    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_light.svg
│   │   └── figure_c3_gate_weights_training_transparent.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_transparent.png
│   ├── geometry/                                    # Decision boundary curvature and hyperspherical prototype clustering
│   │   ├── figure_d1_spherical_prototypes_dark.png      # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_dark.png
│   │   ├── figure_d1_spherical_prototypes_light.pdf     # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_light.pdf
│   │   ├── figure_d1_spherical_prototypes_light.png     # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_light.png
│   │   ├── figure_d1_spherical_prototypes_light.svg     # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_light.svg
│   │   ├── figure_d1_spherical_prototypes_transparent.png # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_transparent.png
│   │   ├── figure_d2_angular_decision_boundaries_dark.png # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_dark.png
│   │   ├── figure_d2_angular_decision_boundaries_light.pdf # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_light.pdf
│   │   ├── figure_d2_angular_decision_boundaries_light.png # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_light.png
│   │   ├── figure_d2_angular_decision_boundaries_light.svg # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_light.svg
│   │   ├── figure_d2_angular_decision_boundaries_transparent.png # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_transparent.png
│   │   ├── figure_d3_angular_margin_distribution_dark.png # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_dark.png
│   │   ├── figure_d3_angular_margin_distribution_light.pdf # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_light.pdf
│   │   ├── figure_d3_angular_margin_distribution_light.png # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_light.png
│   │   ├── figure_d3_angular_margin_distribution_light.svg # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_light.svg
│   │   └── figure_d3_angular_margin_distribution_transparent.png # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_transparent.png
│   ├── representation/                              # Layer-wise CKA similarity matrices across biological and artificial vision
│   │   ├── figure_a1_umap_feature_space_dark.png        # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_dark.png
│   │   ├── figure_a1_umap_feature_space_light.pdf       # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_light.pdf
│   │   ├── figure_a1_umap_feature_space_light.png       # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_light.png
│   │   ├── figure_a1_umap_feature_space_light.svg       # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_light.svg
│   │   ├── figure_a1_umap_feature_space_transparent.png # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_transparent.png
│   │   ├── figure_a2_tsne_representation_drift_dark.png # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_dark.png
│   │   ├── figure_a2_tsne_representation_drift_light.pdf # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_light.pdf
│   │   ├── figure_a2_tsne_representation_drift_light.png # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_light.png
│   │   ├── figure_a2_tsne_representation_drift_light.svg # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_light.svg
│   │   ├── figure_a2_tsne_representation_drift_transparent.png # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_transparent.png
│   │   ├── figure_a3_representation_drift_histogram_dark.png # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_dark.png
│   │   ├── figure_a3_representation_drift_histogram_light.pdf # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_light.pdf
│   │   ├── figure_a3_representation_drift_histogram_light.png # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_light.png
│   │   ├── figure_a3_representation_drift_histogram_light.svg # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_light.svg
│   │   ├── figure_a3_representation_drift_histogram_transparent.png # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_transparent.png
│   │   ├── figure_h1_decision_boundary_slices_dark.png  # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_dark.png
│   │   ├── figure_h1_decision_boundary_slices_light.pdf # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_light.pdf
│   │   ├── figure_h1_decision_boundary_slices_light.png # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_light.png
│   │   ├── figure_h1_decision_boundary_slices_light.svg # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_light.svg
│   │   ├── figure_h1_decision_boundary_slices_transparent.png # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_transparent.png
│   │   ├── figure_h2_loss_landscape_dark.png            # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_dark.png
│   │   ├── figure_h2_loss_landscape_light.pdf           # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_light.pdf
│   │   ├── figure_h2_loss_landscape_light.png           # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_light.png
│   │   ├── figure_h2_loss_landscape_light.svg           # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_light.svg
│   │   └── figure_h2_loss_landscape_transparent.png     # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_transparent.png
│   └── training/                                    # Multi-phase training loss progression and learning rate schedules
│       ├── figure_g1_loss_curves_dark.png               # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_dark.png
│       ├── figure_g1_loss_curves_light.pdf              # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_light.pdf
│       ├── figure_g1_loss_curves_light.png              # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_light.png
│       ├── figure_g1_loss_curves_light.svg              # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_light.svg
│       ├── figure_g1_loss_curves_transparent.png        # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_transparent.png
│       ├── figure_g2_learning_rate_schedule_dark.png    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_dark.png
│       ├── figure_g2_learning_rate_schedule_light.pdf   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_light.pdf
│       ├── figure_g2_learning_rate_schedule_light.png   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_light.png
│       ├── figure_g2_learning_rate_schedule_light.svg   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_light.svg
│       ├── figure_g2_learning_rate_schedule_transparent.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_transparent.png
│       ├── figure_g3_gradient_norm_evolution_dark.png   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_dark.png
│       ├── figure_g3_gradient_norm_evolution_light.pdf  # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_light.pdf
│       ├── figure_g3_gradient_norm_evolution_light.png  # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_light.png
│       ├── figure_g3_gradient_norm_evolution_light.svg  # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_light.svg
│       ├── figure_g3_gradient_norm_evolution_transparent.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_transparent.png
│       ├── figure_g4_parameter_update_magnitude_dark.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_dark.png
│       ├── figure_g4_parameter_update_magnitude_light.pdf # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_light.pdf
│       ├── figure_g4_parameter_update_magnitude_light.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_light.png
│       ├── figure_g4_parameter_update_magnitude_light.svg # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_light.svg
│       └── figure_g4_parameter_update_magnitude_transparent.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_transparent.png
├── figures_v3/                                  # Publication figures v3 (multi-theme light/dark publication-ready SVG/PNG)
│   ├── README.md                                    # Master repository overview, empirical benchmark findings, and exhaustive repo map
│   ├── attention/                                   # Multi-panel attention overlays with dual light/dark styling for publication
│   │   ├── figure_b1_attention_overlay_dark.png         # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_dark.png
│   │   ├── figure_b1_attention_overlay_light.pdf        # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_light.pdf
│   │   ├── figure_b1_attention_overlay_light.png        # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_light.png
│   │   ├── figure_b1_attention_overlay_light.svg        # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b1_attention_overlay_light.svg
│   │   ├── figure_b2_attention_evolution_dark.png       # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_dark.png
│   │   ├── figure_b2_attention_evolution_light.pdf      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_light.pdf
│   │   ├── figure_b2_attention_evolution_light.png      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_light.png
│   │   ├── figure_b2_attention_evolution_light.svg      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b2_attention_evolution_light.svg
│   │   ├── figure_b3_feedback_correction_dark.png       # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_dark.png
│   │   ├── figure_b3_feedback_correction_light.pdf      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_light.pdf
│   │   ├── figure_b3_feedback_correction_light.png      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_light.png
│   │   └── figure_b3_feedback_correction_light.svg      # Visual attention overlay and foveal gaze trajectory across adversarial test images: figure_b3_feedback_correction_light.svg
│   ├── biology/                                     # High-resolution vector schematics of the biological active inference loop
│   │   ├── figure_f1_predictive_coding_convergence_dark.png # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_dark.png
│   │   ├── figure_f1_predictive_coding_convergence_light.pdf # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_light.pdf
│   │   ├── figure_f1_predictive_coding_convergence_light.png # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_light.png
│   │   ├── figure_f1_predictive_coding_convergence_light.svg # Comparative schematic illustrating primate visual ventral stream parallels: figure_f1_predictive_coding_convergence_light.svg
│   │   ├── figure_f2_act_pondering_dark.png             # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_dark.png
│   │   ├── figure_f2_act_pondering_light.pdf            # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_light.pdf
│   │   ├── figure_f2_act_pondering_light.png            # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_light.png
│   │   ├── figure_f2_act_pondering_light.svg            # Comparative schematic illustrating primate visual ventral stream parallels: figure_f2_act_pondering_light.svg
│   │   ├── figure_f3_recurrence_utilization_dark.png    # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_dark.png
│   │   ├── figure_f3_recurrence_utilization_light.pdf   # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_light.pdf
│   │   ├── figure_f3_recurrence_utilization_light.png   # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_light.png
│   │   └── figure_f3_recurrence_utilization_light.svg   # Comparative schematic illustrating primate visual ventral stream parallels: figure_f3_recurrence_utilization_light.svg
│   ├── diagnostics/                                 # Loss landscape curvature and gradient variance under TRADES-PGD4
│   │   ├── figure_i1_diagnostics_dashboard_dark.png     # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_dark.png
│   │   ├── figure_i1_diagnostics_dashboard_light.pdf    # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_light.pdf
│   │   ├── figure_i1_diagnostics_dashboard_light.png    # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_light.png
│   │   └── figure_i1_diagnostics_dashboard_light.svg    # Publication visual asset illustrating experimental dynamics: figure_i1_diagnostics_dashboard_light.svg
│   ├── evaluation/                                  # Standardized benchmark robustness curves across epsilon perturbation radii
│   │   ├── figure_e1_accuracy_vs_epsilon_dark.png       # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_dark.png
│   │   ├── figure_e1_accuracy_vs_epsilon_light.pdf      # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_light.pdf
│   │   ├── figure_e1_accuracy_vs_epsilon_light.png      # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_light.png
│   │   ├── figure_e1_accuracy_vs_epsilon_light.svg      # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e1_accuracy_vs_epsilon_light.svg
│   │   ├── figure_e2_dprime_vs_epsilon_dark.png         # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_dark.png
│   │   ├── figure_e2_dprime_vs_epsilon_light.pdf        # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_light.pdf
│   │   ├── figure_e2_dprime_vs_epsilon_light.png        # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_light.png
│   │   ├── figure_e2_dprime_vs_epsilon_light.svg        # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e2_dprime_vs_epsilon_light.svg
│   │   ├── figure_e3_robustness_threshold_comparison_dark.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_dark.png
│   │   ├── figure_e3_robustness_threshold_comparison_light.pdf # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_light.pdf
│   │   ├── figure_e3_robustness_threshold_comparison_light.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_light.png
│   │   ├── figure_e3_robustness_threshold_comparison_light.svg # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e3_robustness_threshold_comparison_light.svg
│   │   ├── figure_e4_class_robustness_heatmap_dark.png  # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_dark.png
│   │   ├── figure_e4_class_robustness_heatmap_light.pdf # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_light.pdf
│   │   ├── figure_e4_class_robustness_heatmap_light.png # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_light.png
│   │   └── figure_e4_class_robustness_heatmap_light.svg # Empirical accuracy collapse curve under increasing adversarial perturbation radius: figure_e4_class_robustness_heatmap_light.svg
│   ├── explainability/                              # Integrated gradients and Grad-CAM heatmaps rendered in publication styling
│   │   ├── figure_j1_explainability_gradcam_dark.png    # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_dark.png
│   │   ├── figure_j1_explainability_gradcam_light.pdf   # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_light.pdf
│   │   ├── figure_j1_explainability_gradcam_light.png   # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_light.png
│   │   └── figure_j1_explainability_gradcam_light.svg   # Publication visual asset illustrating experimental dynamics: figure_j1_explainability_gradcam_light.svg
│   ├── figure_10_training_pipeline_dark.png         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_training_pipeline_dark.png
│   ├── figure_10_training_pipeline_light.pdf        # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_training_pipeline_light.pdf
│   ├── figure_10_training_pipeline_light.png        # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_training_pipeline_light.png
│   ├── figure_10_training_pipeline_light.svg        # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_10_training_pipeline_light.svg
│   ├── figure_11_loss_landscape_dark.png            # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_loss_landscape_dark.png
│   ├── figure_11_loss_landscape_light.pdf           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_loss_landscape_light.pdf
│   ├── figure_11_loss_landscape_light.png           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_loss_landscape_light.png
│   ├── figure_11_loss_landscape_light.svg           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_11_loss_landscape_light.svg
│   ├── figure_12_information_flow_dark.png          # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_information_flow_dark.png
│   ├── figure_12_information_flow_light.pdf         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_information_flow_light.pdf
│   ├── figure_12_information_flow_light.png         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_information_flow_light.png
│   ├── figure_12_information_flow_light.svg         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_12_information_flow_light.svg
│   ├── figure_13_rhan_ecosystem_dark.png            # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_rhan_ecosystem_dark.png
│   ├── figure_13_rhan_ecosystem_light.pdf           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_rhan_ecosystem_light.pdf
│   ├── figure_13_rhan_ecosystem_light.png           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_rhan_ecosystem_light.png
│   ├── figure_13_rhan_ecosystem_light.svg           # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_13_rhan_ecosystem_light.svg
│   ├── figure_1_rhan_architecture_dark.png          # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture_dark.png
│   ├── figure_1_rhan_architecture_light.pdf         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture_light.pdf
│   ├── figure_1_rhan_architecture_light.png         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture_light.png
│   ├── figure_1_rhan_architecture_light.svg         # Architectural schematic illustrating foveation, gist encoder, and recurrent belief updating loop: figure_1_rhan_architecture_light.svg
│   ├── figure_2_biological_inspiration_dark.png     # Comparative schematic illustrating primate visual ventral stream parallels: figure_2_biological_inspiration_dark.png
│   ├── figure_2_biological_inspiration_light.pdf    # Comparative schematic illustrating primate visual ventral stream parallels: figure_2_biological_inspiration_light.pdf
│   ├── figure_2_biological_inspiration_light.png    # Comparative schematic illustrating primate visual ventral stream parallels: figure_2_biological_inspiration_light.png
│   ├── figure_2_biological_inspiration_light.svg    # Comparative schematic illustrating primate visual ventral stream parallels: figure_2_biological_inspiration_light.svg
│   ├── figure_3_predictive_coding_loop_dark.png     # Top-down predictive coding loop diagram showing error residual minimization: figure_3_predictive_coding_loop_dark.png
│   ├── figure_3_predictive_coding_loop_light.pdf    # Top-down predictive coding loop diagram showing error residual minimization: figure_3_predictive_coding_loop_light.pdf
│   ├── figure_3_predictive_coding_loop_light.png    # Top-down predictive coding loop diagram showing error residual minimization: figure_3_predictive_coding_loop_light.png
│   ├── figure_3_predictive_coding_loop_light.svg    # Top-down predictive coding loop diagram showing error residual minimization: figure_3_predictive_coding_loop_light.svg
│   ├── figure_4_dual_transformer_dark.png           # Dual-stream transformer schematic separating global low-frequency gist and foveal patches: figure_4_dual_transformer_dark.png
│   ├── figure_4_dual_transformer_light.pdf          # Dual-stream transformer schematic separating global low-frequency gist and foveal patches: figure_4_dual_transformer_light.pdf
│   ├── figure_4_dual_transformer_light.png          # Dual-stream transformer schematic separating global low-frequency gist and foveal patches: figure_4_dual_transformer_light.png
│   ├── figure_4_dual_transformer_light.svg          # Dual-stream transformer schematic separating global low-frequency gist and foveal patches: figure_4_dual_transformer_light.svg
│   ├── figure_5_spherical_prototype_dark.png        # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_spherical_prototype_dark.png
│   ├── figure_5_spherical_prototype_light.pdf       # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_spherical_prototype_light.pdf
│   ├── figure_5_spherical_prototype_light.png       # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_spherical_prototype_light.png
│   ├── figure_5_spherical_prototype_light.svg       # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_5_spherical_prototype_light.svg
│   ├── figure_6_sail_flowchart_dark.png             # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_6_sail_flowchart_dark.png
│   ├── figure_6_sail_flowchart_light.pdf            # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_6_sail_flowchart_light.pdf
│   ├── figure_6_sail_flowchart_light.png            # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_6_sail_flowchart_light.png
│   ├── figure_6_sail_flowchart_light.svg            # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_6_sail_flowchart_light.svg
│   ├── figure_7_geometric_sail_dark.png             # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_7_geometric_sail_dark.png
│   ├── figure_7_geometric_sail_light.pdf            # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_7_geometric_sail_light.pdf
│   ├── figure_7_geometric_sail_light.png            # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_7_geometric_sail_light.png
│   ├── figure_7_geometric_sail_light.svg            # Self-supervised Adversarial Invariant Learning (SAIL) manifold stabilization diagram: figure_7_geometric_sail_light.svg
│   ├── figure_8_tdv_framework_dark.png              # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_tdv_framework_dark.png
│   ├── figure_8_tdv_framework_light.pdf             # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_tdv_framework_light.pdf
│   ├── figure_8_tdv_framework_light.png             # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_tdv_framework_light.png
│   ├── figure_8_tdv_framework_light.svg             # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_8_tdv_framework_light.svg
│   ├── figure_9_adversarial_tdv_dark.png            # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_adversarial_tdv_dark.png
│   ├── figure_9_adversarial_tdv_light.pdf           # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_adversarial_tdv_light.pdf
│   ├── figure_9_adversarial_tdv_light.png           # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_adversarial_tdv_light.png
│   ├── figure_9_adversarial_tdv_light.svg           # Temporal Difference Visual (TDV) framework diagram modeling sequential video frame coherence: figure_9_adversarial_tdv_light.svg
│   ├── frequency/                                   # Spatial frequency distribution plots contrasting human vs model sensitivity
│   │   ├── figure_c1_low_frequency_gate_dark.png        # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_dark.png
│   │   ├── figure_c1_low_frequency_gate_light.pdf       # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_light.pdf
│   │   ├── figure_c1_low_frequency_gate_light.png       # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_light.png
│   │   ├── figure_c1_low_frequency_gate_light.svg       # Publication visual asset illustrating experimental dynamics: figure_c1_low_frequency_gate_light.svg
│   │   ├── figure_c2_high_frequency_gate_dark.png       # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_dark.png
│   │   ├── figure_c2_high_frequency_gate_light.pdf      # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_light.pdf
│   │   ├── figure_c2_high_frequency_gate_light.png      # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_light.png
│   │   ├── figure_c2_high_frequency_gate_light.svg      # Publication visual asset illustrating experimental dynamics: figure_c2_high_frequency_gate_light.svg
│   │   ├── figure_c3_gate_weights_training_dark.png     # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_dark.png
│   │   ├── figure_c3_gate_weights_training_light.pdf    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_light.pdf
│   │   ├── figure_c3_gate_weights_training_light.png    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_light.png
│   │   └── figure_c3_gate_weights_training_light.svg    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_c3_gate_weights_training_light.svg
│   ├── geometry/                                    # Latent manifold geometry and spherical embedding space visualizations
│   │   ├── figure_d1_spherical_prototypes_dark.png      # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_dark.png
│   │   ├── figure_d1_spherical_prototypes_light.pdf     # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_light.pdf
│   │   ├── figure_d1_spherical_prototypes_light.png     # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_light.png
│   │   ├── figure_d1_spherical_prototypes_light.svg     # Hyperspherical prototype embedding space visualization for contrastive feature separation: figure_d1_spherical_prototypes_light.svg
│   │   ├── figure_d2_angular_decision_boundaries_dark.png # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_dark.png
│   │   ├── figure_d2_angular_decision_boundaries_light.pdf # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_light.pdf
│   │   ├── figure_d2_angular_decision_boundaries_light.png # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_light.png
│   │   ├── figure_d2_angular_decision_boundaries_light.svg # Publication visual asset illustrating experimental dynamics: figure_d2_angular_decision_boundaries_light.svg
│   │   ├── figure_d3_angular_margin_distribution_dark.png # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_dark.png
│   │   ├── figure_d3_angular_margin_distribution_light.pdf # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_light.pdf
│   │   ├── figure_d3_angular_margin_distribution_light.png # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_light.png
│   │   └── figure_d3_angular_margin_distribution_light.svg # Publication visual asset illustrating experimental dynamics: figure_d3_angular_margin_distribution_light.svg
│   ├── representation/                              # Cross-architecture representational alignment matrices
│   │   ├── figure_a1_umap_feature_space_dark.png        # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_dark.png
│   │   ├── figure_a1_umap_feature_space_light.pdf       # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_light.pdf
│   │   ├── figure_a1_umap_feature_space_light.png       # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_light.png
│   │   ├── figure_a1_umap_feature_space_light.svg       # Publication visual asset illustrating experimental dynamics: figure_a1_umap_feature_space_light.svg
│   │   ├── figure_a2_tsne_representation_drift_dark.png # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_dark.png
│   │   ├── figure_a2_tsne_representation_drift_light.pdf # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_light.pdf
│   │   ├── figure_a2_tsne_representation_drift_light.png # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_light.png
│   │   ├── figure_a2_tsne_representation_drift_light.svg # Publication visual asset illustrating experimental dynamics: figure_a2_tsne_representation_drift_light.svg
│   │   ├── figure_a3_representation_drift_histogram_dark.png # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_dark.png
│   │   ├── figure_a3_representation_drift_histogram_light.pdf # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_light.pdf
│   │   ├── figure_a3_representation_drift_histogram_light.png # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_light.png
│   │   ├── figure_a3_representation_drift_histogram_light.svg # Publication visual asset illustrating experimental dynamics: figure_a3_representation_drift_histogram_light.svg
│   │   ├── figure_h1_decision_boundary_slices_dark.png  # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_dark.png
│   │   ├── figure_h1_decision_boundary_slices_light.pdf # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_light.pdf
│   │   ├── figure_h1_decision_boundary_slices_light.png # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_light.png
│   │   ├── figure_h1_decision_boundary_slices_light.svg # Publication visual asset illustrating experimental dynamics: figure_h1_decision_boundary_slices_light.svg
│   │   ├── figure_h2_loss_landscape_dark.png            # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_dark.png
│   │   ├── figure_h2_loss_landscape_light.pdf           # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_light.pdf
│   │   ├── figure_h2_loss_landscape_light.png           # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_light.png
│   │   └── figure_h2_loss_landscape_light.svg           # 3D loss surface contour visualization comparing standard cross-entropy vs TRADES robust minima: figure_h2_loss_landscape_light.svg
│   └── training/                                    # Training stability curves and parameter update magnitudes across phases
│       ├── figure_g1_loss_curves_dark.png               # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_dark.png
│       ├── figure_g1_loss_curves_light.pdf              # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_light.pdf
│       ├── figure_g1_loss_curves_light.png              # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_light.png
│       ├── figure_g1_loss_curves_light.svg              # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g1_loss_curves_light.svg
│       ├── figure_g2_learning_rate_schedule_dark.png    # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_dark.png
│       ├── figure_g2_learning_rate_schedule_light.pdf   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_light.pdf
│       ├── figure_g2_learning_rate_schedule_light.png   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_light.png
│       ├── figure_g2_learning_rate_schedule_light.svg   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g2_learning_rate_schedule_light.svg
│       ├── figure_g3_gradient_norm_evolution_dark.png   # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_dark.png
│       ├── figure_g3_gradient_norm_evolution_light.pdf  # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_light.pdf
│       ├── figure_g3_gradient_norm_evolution_light.png  # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_light.png
│       ├── figure_g3_gradient_norm_evolution_light.svg  # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g3_gradient_norm_evolution_light.svg
│       ├── figure_g4_parameter_update_magnitude_dark.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_dark.png
│       ├── figure_g4_parameter_update_magnitude_light.pdf # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_light.pdf
│       ├── figure_g4_parameter_update_magnitude_light.png # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_light.png
│       └── figure_g4_parameter_update_magnitude_light.svg # Multi-phase training pipeline, learning rate schedule, and curriculum ramp flowchart: figure_g4_parameter_update_magnitude_light.svg
├── gen2_foundation/                             # Gen-2 foundation architecture: DINOv2-S trunk, Gist prior, bounded GRU UpdateNet, dynamic precision
│   ├── AUDIT.md                                     # Package forensic audit report verifying 52 passing unit tests and zero gradient leakage
│   ├── EXPERIMENT_REGISTRY.md                       # Formal preregistered ledger of experimental arms, baselines, and promotion criteria
│   ├── __init__.py                                  # Package exports and public API declarations for Gen-2 foundation stack
│   ├── ema.py                                       # Exponential moving average (EMA) shadow model weight tracker to mitigate robust overfitting
│   ├── eot.py                                       # Expectation Over Transformation (EOT-PGD) attack for evaluating stochastic visual defenses
│   ├── flags.py                                     # Feature flags gating independent architectural modules (Gist, Recurrence, Precision, UpdateNet)
│   ├── gist.py                                      # Global gist encoder: DINOv2-S patch projection, antialiased pos-embed, and CLS prior
│   ├── precision.py                                 # Dynamic observation precision head (gamma) computing uncertainty-scaled sensory weighting
│   ├── recipe.py                                    # Builds multi-group AdamW optimizer parameter groups with layer-wise learning rate decay
│   ├── tests/                                       # Unit and regression tests verifying Gen-2 components (EMA weights, EOT-PGD, Gist parity, UpdateNet)
│   │   ├── __init__.py                                  # Unit test package initialization for gen2_foundation test suite
│   │   ├── test_ema.py                                  # Verifies EMA shadow weight updates, decay schedules, and eval weight swapping
│   │   ├── test_eot.py                                  # Tests EOT-PGD loss gradient averaging across spatial transformation distributions
│   │   ├── test_flags_gating.py                         # Ensures feature flags cleanly toggle modules without residual parameter leakage
│   │   ├── test_gist_parity.py                          # Verifies that Gist encoder outputs bit-exact CLS representations at gate-0
│   │   ├── test_precision.py                            # Tests dynamic range and positive boundedness of precision weight outputs
│   │   ├── test_recipe.py                               # Verifies optimizer parameter group separation and learning rate allocations
│   │   └── test_update_net_v2.py                        # Tests UpdateNet v2 gradient flow, hidden state dimensions, and residual steps
│   └── update_net_v2.py                             # Bounded recurrent belief update network (GRU) integrating error and glimpse features
├── noesis_vision/                               # Canonical Gen-1 NOESIS engine implementing active inference and the biological perceptual loop
│   ├── RHAN_NXA/                                    # RHAN-NXA architecture documentation, master plans, and locked contracts
│   │   ├── MASTER_PLAN.md                               # Master plan defining locked contracts and experimental roadmap for Gen-1
│   │   └── docs/                                        # 32-chapter exhaustive technical specification handbook for RHAN-NXA
│   │       ├── 00_README.md                                 # Index and architectural overview of the 31-chapter RHAN-NXA specification handbook
│   │       ├── 01_What_Is_RHAN_NXA.md                       # Theoretical formulation of active inference, recurrent belief updating, and foveation
│   │       ├── 02_Why_RHAN_NXA_Exists.md                    # Scientific rationale: why passive feedforward vision collapses under adversarial perturbations
│   │       ├── 03_Perception_As_Investigation.md            # Formulating visual recognition as sequential hypothesis testing and active evidence sampling
│   │       ├── 04_Belief_State.md                           # Mathematical formulation of internal belief representation B_t = (z_t, S_t)
│   │       ├── 05_z_State.md                                # Continuous latent feature vector dynamics representing the semantic hypothesis space
│   │       ├── 06_Structure_State.md                        # Structured symbolic representation of object parts and spatial relationships
│   │       ├── 07_Uncertainty.md                            # Epistemic vs aleatoric uncertainty quantification using Dirichlet distributions
│   │       ├── 08_Prediction_Error.md                       # Top-down generative feature prediction and error residual computation
│   │       ├── 09_Recurrence.md                             # Recurrent refinement dynamics: state updating vs multiview aggregation
│   │       ├── 10_AIS_v2.md                                 # Adaptive Information Seeking: expected uncertainty reduction gaze policy
│   │       ├── 11_Complete_Perceptual_Loop.md               # Step-by-step tensor tracing through the full perceptual cycle
│   │       ├── 12_Architecture_Data_Flow.md                 # Detailed module-by-module data flow and interface boundaries
│   │       ├── 13_Gradient_Flow.md                          # Backpropagation paths, detached copies, and gradient isolation boundaries
│   │       ├── 14_Tensor_Shape_Reference.md                 # Complete tensor dimension reference table across all operational steps
│   │       ├── 15_Status_And_Decision_System.md             # Experimental confidence classification (VERIFIED, HYPOTHESIS, REJECTED)
│   │       ├── 16_Gen0_Evidence_And_Confounds.md            # Analysis of Gen-0 empirical results and confounding factors
│   │       ├── 17_Common_Misunderstandings.md               # Conceptual clarifications regarding foveation, recurrence, and precision
│   │       ├── 18_Glossary.md                               # Formal mathematical definitions and project vocabulary
│   │       ├── 19_Decision_Records.md                       # Architecture Decision Records (ADRs) establishing locked design choices
│   │       ├── 20_Scope_Boundaries.md                       # Explicit definitions of in-scope vs out-of-scope research questions
│   │       ├── 21_L_stab_Stability.md                       # Numerical stability constraints and belief state variance bounds
│   │       ├── 22_Memory.md                                 # Memory taxonomy (M0 weights, M1 working state, M2 glimpse buffer, M3 spatial bank)
│   │       ├── 23_V1_Frontend.md                            # Primary visual cortex (V1) multi-scale Gabor filter frontend ablation
│   │       ├── 24_Training_Phase_DAG.md                     # Directed Acyclic Graph defining phase dependencies and promotion gates
│   │       ├── 25_Ablation_Matrix.md                        # Systematic ablation matrix isolating each cognitive component
│   │       ├── 26_Agent_Organization.md                     # Decoupled multi-agent architecture and interface contracts
│   │       ├── 27_Infrastructure_Port_Table.md              # Port mapping table between legacy codebase and canonical modules
│   │       ├── 28_Gates_and_Compute_Accounting.md           # Strict promotion gates (Gate B budget, Gate S search, Gate A robustness)
│   │       ├── 29_Literature_Classification.md              # Positioning relative to neuroscience and computer vision literature
│   │       ├── 30_Agent0_Interface_Contract.md              # Root interface contract defining inputs, outputs, and invariants
│   │       └── Proposed_Plan.md                             # Multi-stage master roadmap for Gen-2 and Gen-3 development
│   ├── beliefs/                                     # Latent belief state representations (VectorBeliefState) and drift regularization
│   │   ├── drift.py                                     # Belief state drift penalty L_drift = ||z_t - z_{t-1}||^2 enforcing smooth trajectory convergence
│   │   ├── factory.py                                   # Factory constructor instantiating concrete belief state representations
│   │   ├── interfaces.py                                # Abstract Base Class defining the belief state interface contract
│   │   └── vector_belief.py                             # Dense continuous vector representation (z_t) aggregating temporal evidence across glimpses
│   ├── core/                                        # Core infrastructure: configuration schemas, atomic checkpointing, provenance manifests, consistency guards
│   │   ├── agent_contract_template.md                   # Interface specification contract template between decoupled agents
│   │   ├── checkpoint.py                                # Atomic checkpoint serialization, rolling resume, and integrity checking
│   │   ├── consistency_assert.py                        # Runtime invariant assertions reconciling summary tables with per-seed CSV logs
│   │   ├── dependency_graph.md                          # Mathematical dependency graph between perceptual modules
│   │   ├── experiment_registry.py                       # Registry tracking active experiment flags, seeds, and component variants
│   │   ├── multi_group_optimizer.py                     # Multi-parameter-group AdamW optimizer configuring layer-wise learning rates
│   │   ├── provenance.py                                # Captures git commit SHA, dependency hashes, and environment metadata
│   │   ├── schema.py                                    # Schema-locked configuration (RHANNXAConfig: num_glimpses=4, d_z=384)
│   │   └── seed_management.py                           # Deterministic RNG seeding across PyTorch, NumPy, and Python standard library
│   ├── gaze/                                        # Adaptive Information Seeking (AIS-v2) gaze policy, candidate fixation sampling, and center bias
│   │   ├── ais_v2_policy.py                             # Adaptive Information Seeking (AIS) policy ranking K=4 candidate fixations by expected uncertainty reduction
│   │   ├── candidate_sampler.py                         # Spatial candidate sampler generating prospective fixation bounding coordinates
│   │   ├── center_bias.py                               # Foveal center-bias prior and spatial fixation entropy calculation
│   │   └── gaze_state.py                                # Canonical data structure encapsulating fixation history and trajectory state
│   ├── models/                                      # Neural backbones: CompactViT, differentiable foveation grid sampling, tied recurrence
│   │   ├── backbone.py                                  # CompactViT backbone (~23.3M params in [20M, 25M], 14x14 patches, D_z=384) on 56x56 foveal crops
│   │   ├── foveation.py                                 # Differentiable affine grid transformation extracting 56x56 foveal glimpse crops
│   │   └── recurrent_block.py                           # Tied recurrent refinement block maintaining constant parameter count across iterations T
│   ├── predictive_coding/                           # Top-down feature prediction, precision weighting, update network, and spatial error pooling
│   │   ├── ema_predictor.py                             # Exponential moving average target encoder providing stable prediction targets
│   │   ├── glimpse_predictor.py                         # Top-down generative predictor forecasting incoming glimpse features from belief state
│   │   ├── interfaces.py                                # Abstract Base Class defining predictive coding predictor and error interfaces
│   │   ├── precision.py                                 # Dynamic precision weighting (gamma = exp(-s)) scaling prediction error residuals
│   │   ├── spatial_error_pool.py                        # Spatial pooling layer aggregating localized prediction error residuals
│   │   └── update_net.py                                # Update network computing belief innovation Delta z from precision-weighted error
│   └── uncertainty/                                 # Dirichlet evidential deep learning (EDL) heads estimating subjective epistemic uncertainty
│       └── evidential_head.py                           # Dirichlet Evidential Deep Learning (EDL) head estimating subjective epistemic uncertainty
├── report/                                      # Master experimental results tables, empirical JSON databases, and scientific publication reports
│   ├── GEN1_RESULTS_MASTER.md                       # Authoritative master results table recording clean and robust accuracies across all Gen-1 phases and seeds
│   ├── Gen0.md                                      # Historical summary report compiling Gen-0 STL-10 baseline evaluation benchmarks
│   ├── IEEEtran.cls                                 # IEEE Transactions LaTeX document class file for manuscript submission
│   ├── assets/                                      # High-resolution architectural flowcharts and empirical training curves for reports
│   │   ├── foveal_stream_arch.png                       # Architectural diagram showing the dual coarse gist and fine foveal streams
│   │   ├── gaze_trajectory.png                          # Visual trace of sequential saccadic fixation coordinates across test images
│   │   ├── precision_init_net_arch.png                  # Neural network architecture diagram for the precision initialization head
│   │   ├── precision_vs_epoch.png                       # Dynamic precision weight progression curve across training epochs
│   │   └── robustness_curve.png                         # Empirical accuracy retention curve under increasing adversarial perturbation radius
│   ├── empirical_report.json                        # Master JSON database compiling clean accuracy, robust accuracy, and SDT metrics across all models
│   ├── final_sweep_results_stl10.json               # Complete benchmark sweep results across STL-10 recurrent architectures
│   ├── generate_figures.py                          # Publication figure generation script compiling vector plots from empirical JSON tables
│   ├── lens_e1_analysis/                            # Comprehensive audit report for Lens E1 introspection experiment
│   │   └── E1_FULL_AUDIT_REPORT.md                      # Comprehensive audit report on Stage 4-E1 introspection lens findings
│   ├── prior_results.json                           # Historical baseline performance records for earlier models (ResNet, ViT, EfficientNet)
│   ├── rhan_next_ais_v1_isoA_diag.jsonl             # Repository component: rhan_next_ais_v1_isoA_diag.jsonl
│   ├── rhan_next_ais_v1_isoB_diag.jsonl             # Repository component: rhan_next_ais_v1_isoB_diag.jsonl
│   ├── rhan_next_hpc_only_diag.jsonl                # Repository component: rhan_next_hpc_only_diag.jsonl
│   ├── rhan_next_hpc_only_smoke_diag.jsonl          # Repository component: rhan_next_hpc_only_smoke_diag.jsonl
│   ├── rhan_next_hpc_only_smoke_v2_diag.jsonl       # Repository component: rhan_next_hpc_only_smoke_v2_diag.jsonl
│   ├── rhan_next_hpc_only_smoke_v3_diag.jsonl       # Repository component: rhan_next_hpc_only_smoke_v3_diag.jsonl
│   ├── rhan_next_hpc_only_smoke_v4_diag.jsonl       # Repository component: rhan_next_hpc_only_smoke_v4_diag.jsonl
│   ├── rhan_nx_generation1_report.md                # Comprehensive experimental audit report analyzing Gen-1 foundation ladder performance
│   ├── rhan_nx_sbr0_cosine_t0.json                  # Cosine similarity metrics verifying step T=0 representation collapse in Stage SBR-0
│   ├── rhan_v10_scientific_report.aux               # LaTeX auxiliary compilation artifact for RHAN v10 scientific report
│   ├── rhan_v10_scientific_report.log               # LaTeX build log for RHAN v10 report compilation
│   ├── rhan_v10_scientific_report.pdf               # Compiled publication manuscript PDF for the RHAN v10 architecture
│   ├── rhan_v10_scientific_report.tex               # LaTeX manuscript source for the RHAN v10 comprehensive scientific report
│   ├── rhan_v10_scientific_report_body.tex          # LaTeX body sections and mathematical derivations for RHAN v10 report
│   ├── sbr_feasibility/                             # Spatial belief representation (SBR) slot mask visual feasibility checks
│   │   ├── README.txt                                   # Documentation for the spatial slot mask visual feasibility experiment
│   │   ├── slot_masks_sample_000.png                    # Spatial slot attention mask feasibility check visual output: slot_masks_sample_000.png
│   │   ├── slot_masks_sample_001.png                    # Spatial slot attention mask feasibility check visual output: slot_masks_sample_001.png
│   │   ├── slot_masks_sample_002.png                    # Spatial slot attention mask feasibility check visual output: slot_masks_sample_002.png
│   │   └── slot_masks_sample_003.png                    # Spatial slot attention mask feasibility check visual output: slot_masks_sample_003.png
│   ├── stage2_hpc_run_log.md                        # Execution log recording training convergence for Stage 2 hierarchical predictive coding
│   ├── stage3_D/                                    # Stage 3-D Pareto trade-off analysis, figures, and provenance audits
│   │   ├── figures/                                     # Visual plots comparing Stage 3-D Pareto trade-offs and seed variance
│   │   │   ├── fig1_four_way_comparison.png                 # Stage 3-D Pareto frontier and multi-seed variance visual plot: fig1_four_way_comparison.png
│   │   │   ├── fig2_pareto_tradeoff.png                     # Stage 3-D Pareto frontier and multi-seed variance visual plot: fig2_pareto_tradeoff.png
│   │   │   ├── fig3_per_seed_distribution.png               # Stage 3-D Pareto frontier and multi-seed variance visual plot: fig3_per_seed_distribution.png
│   │   │   ├── fig4_masking_check.png                       # Stage 3-D Pareto frontier and multi-seed variance visual plot: fig4_masking_check.png
│   │   │   └── fig5_d_per_seed_detail.png                   # Stage 3-D Pareto frontier and multi-seed variance visual plot: fig5_d_per_seed_detail.png
│   │   ├── provenance_audit.md                          # Cryptographic provenance audit verifying checkpoint SHAs and reproducibility for Stage 3-D
│   │   └── stage3_D_analysis.md                         # Detailed analysis of Stage 3-D Pareto trade-offs and gradient stability
│   ├── stage3_trainer_audit.md                      # Audit findings and gradient verification for Stage 3 trainer
│   ├── stage4_E1/                                   # Stage 4-E1 lens introspection, pre-registration scripts, and runners
│   │   ├── colab_setup.sh                               # Environment setup script configuring Google Colab for Stage 4-E1 lens evaluation
│   │   ├── eval_d_e1.sh                                 # Evaluation script executing sweeps on trained introspection lens models
│   │   ├── kaggle_setup.sh                              # Environment setup script configuring Kaggle GPU environment for lens training
│   │   ├── lens_analysis.py                             # Introspection lens evaluation script measuring mutual information between tokens and concepts
│   │   ├── preregistration.md                           # Formal preregistered protocol for evaluating latent token interpretability
│   │   ├── resume.sh                                    # Shell script resuming interrupted Stage 4-E1 lens training from latest checkpoint
│   │   ├── train_e1.sh                                  # Training driver launching Stage 4-E1 introspection lens parameter optimization
│   │   └── verify_experiment.sh                         # Verification script asserting experiment invariants and data completeness for Stage 4-E1
│   └── supervisor.log                               # Operational trace log emitted by the autonomous experiment supervisor process
├── requirements.txt                             # Core Python dependencies: PyTorch, torchvision, timm, torchattacks, scipy
├── rhan_mathematical_report1.pdf:Zone.Identifier # Windows filesystem download zone identifier metadata for rhan_mathematical_report1.pdf
├── runs/                                        # Experiment output runs containing TensorBoard event logs and metrics
│   ├── efficientnet_cifar10/                        # TensorBoard telemetry event logs recording training convergence for EfficientNet-B0 baseline
│   │   ├── events.out.tfevents.1778466115.FerrariKazu.15169.0 # TensorBoard telemetry event log stream recording loss and accuracy for: efficientnet_cifar10
│   │   └── events.out.tfevents.1778470167.FerrariKazu.29079.0 # TensorBoard telemetry event log stream recording loss and accuracy for: efficientnet_cifar10
│   ├── phase1_training/                             # TensorBoard event logs for baseline CNN and Vision Transformer CIFAR-10 training runs
│   │   ├── events.out.tfevents.1777739614.FerrariKazu.24522.0 # TensorBoard telemetry event log stream recording loss and accuracy for: phase1_training
│   │   └── events.out.tfevents.1778872582.FerrariKazu.8453.0 # TensorBoard telemetry event log stream recording loss and accuracy for: phase1_training
│   ├── rhan_adaptive/                               # TensorBoard event logs for adaptive glimpse allocation experimental runs
│   │   └── events.out.tfevents.1779188232.FerrariKazu.16457.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_adaptive
│   ├── rhan_aligned/                                # TensorBoard event logs for self-supervised contrastive alignment runs
│   │   ├── events.out.tfevents.1779550047.FerrariKazu.23796.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_aligned
│   │   └── events.out.tfevents.1779592274.FerrariKazu.16037.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_aligned
│   ├── rhan_clip/                                   # TensorBoard event logs for CLIP-guided semantic distillation runs
│   │   ├── events.out.tfevents.1779163684.FerrariKazu.3787.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_clip
│   │   ├── events.out.tfevents.1779163790.FerrariKazu.4790.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_clip
│   │   └── events.out.tfevents.1779163830.FerrariKazu.5098.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_clip
│   ├── rhan_predictive/                             # TensorBoard event logs for predictive gaze trajectory exploration runs
│   │   └── events.out.tfevents.1779550020.FerrariKazu.23088.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_predictive
│   ├── rhan_split/                                  # TensorBoard event logs for what/where split architecture runs
│   │   └── events.out.tfevents.1779609371.FerrariKazu.25141.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_split
│   ├── rhan_trades_curriculum/                      # Subsystem directory: rhan_trades_curriculum/
│   │   └── events.out.tfevents.1780559412.FerrariKazu.1406.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_trades_curriculum
│   ├── rhan_trades_hardened/                        # Subsystem directory: rhan_trades_hardened/
│   │   └── events.out.tfevents.1780473769.FerrariKazu.31061.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_trades_hardened
│   ├── rhan_v2/                                     # Subsystem directory: rhan_v2/
│   │   ├── events.out.tfevents.1779660118.FerrariKazu.14735.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v2
│   │   └── events.out.tfevents.1779678161.FerrariKazu.27055.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v2
│   ├── rhan_v3/                                     # Subsystem directory: rhan_v3/
│   │   └── events.out.tfevents.1779770092.FerrariKazu.1345.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v3
│   ├── rhan_v3_adaptive/                            # Subsystem directory: rhan_v3_adaptive/
│   │   └── events.out.tfevents.1779980849.FerrariKazu.1208.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v3_adaptive
│   ├── rhan_v4/                                     # Subsystem directory: rhan_v4/
│   │   ├── events.out.tfevents.1779859422.FerrariKazu.27521.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v4
│   │   └── events.out.tfevents.1779944441.FerrariKazu.17667.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v4
│   ├── rhan_v5/                                     # Subsystem directory: rhan_v5/
│   │   └── events.out.tfevents.1780028511.FerrariKazu.18911.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5
│   ├── rhan_v5_trades/                              # Subsystem directory: rhan_v5_trades/
│   │   ├── events.out.tfevents.1780328006.FerrariKazu.30551.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   │   ├── events.out.tfevents.1780341819.FerrariKazu.9248.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   │   ├── events.out.tfevents.1780352756.FerrariKazu.12175.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   │   ├── events.out.tfevents.1780353051.FerrariKazu.14441.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   │   ├── events.out.tfevents.1780353260.FerrariKazu.16931.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   │   ├── events.out.tfevents.1780357709.FerrariKazu.31698.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   │   └── events.out.tfevents.1780358872.FerrariKazu.3098.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v5_trades
│   ├── rhan_v6/                                     # Subsystem directory: rhan_v6/
│   │   ├── events.out.tfevents.1780110580.FerrariKazu.16876.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v6
│   │   ├── events.out.tfevents.1780156613.FerrariKazu.26502.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v6
│   │   ├── events.out.tfevents.1780167479.FerrariKazu.32617.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v6
│   │   ├── events.out.tfevents.1780195671.FerrariKazu.19923.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v6
│   │   └── events.out.tfevents.1780282276.FerrariKazu.21306.0 # TensorBoard telemetry event log stream recording loss and accuracy for: rhan_v6
│   └── vit_cifar10/                                 # Subsystem directory: vit_cifar10/
│       ├── events.out.tfevents.1778121334.FerrariKazu.27679.0 # TensorBoard telemetry event log stream recording loss and accuracy for: vit_cifar10
│       ├── events.out.tfevents.1778121526.FerrariKazu.28846.0 # TensorBoard telemetry event log stream recording loss and accuracy for: vit_cifar10
│       └── events.out.tfevents.1778153893.FerrariKazu.28758.0 # TensorBoard telemetry event log stream recording loss and accuracy for: vit_cifar10
├── scripts/                                     # Operational tooling: ImageNet-100 data prep, promotion gates, sweep runners, and report compilers
│   ├── build_noesis_pdf.py                          # Compiles NOESIS markdown architecture documentation into PDF format
│   ├── build_rhan_nx_report.py                      # Generates formal experimental summary report in PDF format
│   ├── build_rhan_nxa_handbook.py                   # Assembles the complete RHAN-NXA Technical Handbook from book chapters
│   ├── comparator_registry.py                       # Registry and baseline comparator loader for paired hypothesis tests
│   ├── consistency_assert.py                        # Assertions verifying architectural and tensor dimension consistency
│   ├── eval_ais_v2_gate.py                          # Gating assertion evaluating AIS-v2 gaze policy against random and fixed gaze baselines
│   ├── freeze_run_manifest.py                       # Generates cryptographic SHA-256 manifest of model weights and configs
│   ├── generate_full_sweep.py                       # Automated script running full hyperparameter and attack evaluation sweeps
│   ├── measure_group_dw.py                          # Measures parameter update magnitudes (||dW||) across optimizer groups
│   ├── merge_stage1_seed_extension.py               # Aggregates multi-seed stage 1 evaluation outputs into master dataset
│   ├── prepare_imagenet100.py                       # Downloads and unpacks pinned ImageNet-100 dataset (clane9/imagenet-100) with SHA-256 fingerprinting
│   ├── reset_gen1_ladder.py                         # Utility to reset training state machine to restart foundation ladder
│   ├── sbr0_gate.py                                 # Automated gate checker validating Stage SBR-0 criteria before promotion
│   ├── stage_state_machine.py                       # State machine script managing transitions between training stages
│   └── verify_run_complete.py                       # Verifies that all epochs and validation passes finished successfully
├── tests/                                       # Comprehensive test suite (60 test files: invariants, gradient flow, numerical stability, checkpoint consistency)
│   ├── conftest.py                                  # Global pytest fixtures, synthetic batch generators, and device initialization
│   ├── test_ablation_matrix.py                      # Verifies that every cell in the 9-point ablation matrix executes without error
│   ├── test_adv_curriculum.py                       # Asserts TRADES epsilon schedule bounds, step-size alpha ratios, and ramp timing
│   ├── test_adv_curriculum_integration.py           # Integration test validating PGD-KL execution during training step
│   ├── test_ais_ablation_flags.py                   # Verifies that disabling AIS flags reverts gaze policy to deterministic fixed grid
│   ├── test_ais_v2_gaze_policy.py                   # Tests AIS candidate ranking, temperature softmax scaling, and entropy reduction
│   ├── test_ais_v2_gradient_flow.py                 # Asserts non-zero gradient backpropagation to AIS policy parameters
│   ├── test_backbone_param_count.py                 # Asserts CompactViT parameter count remains within the strict [20M, 25M] bounds
│   ├── test_belief_state_composition.py             # Tests latent belief state tensor concatenation, initialization, and shapes
│   ├── test_center_bias_measurement.py              # Calculates spatial fixation entropy to detect and prevent foveal center collapse
│   ├── test_checkpoint_never_silently_restarts.py   # Asserts training immediately aborts if a mandatory parent checkpoint is missing
│   ├── test_comparator_reuse_integrity.py           # Verifies comparator baselines use identical evaluation seeds and data slices
│   ├── test_config_backward_compat.py               # Ensures configuration schema changes maintain full backward compatibility
│   ├── test_consistency_assert_catches_corruption.py # Tests that table-CSV integrity assertions flag injected data discrepancies
│   ├── test_detached_copy_isolation.py              # Verifies belief state detachment prevents unintended BPTT memory leaks
│   ├── test_diagnostics_next.py                     # Unit tests for diagnostic profiling scripts and memory trackers
│   ├── test_drift_to_gradient_flow.py               # Verifies gradient flow through belief state drift regularization targets
│   ├── test_eval_rhan_protocol.py                   # Verifies that evaluation harness strictly enforces the registered attack protocol
│   ├── test_eval_structural_consistency.py          # Asserts that evaluation data loaders and model heads match class labels
│   ├── test_evidential_gradient_flow.py             # Verifies non-zero gradient propagation through Dirichlet evidential heads
│   ├── test_evidential_shapes.py                    # Asserts Dirichlet concentration parameters (alpha) have correct positive output bounds
│   ├── test_foveation_differentiable.py             # Verifies differentiability of spatial grid sampling during glimpse extraction
│   ├── test_gaze_state_canonical.py                 # Verifies canonical data structure formatting for fixation history trajectories
│   ├── test_gen0_recipe_freeze.py                   # Ensures historical Gen-0 training recipe hyperparameters remain immutably frozen
│   ├── test_gradient_flow.py                        # Comprehensive gradient backpropagation check across all trainable parameter groups
│   ├── test_hpc_belief_gradient_flow.py             # Asserts gradient flow through hierarchical predictive coding belief pathways
│   ├── test_hpc_diag_baseline.py                    # Baseline diagnostic checks for predictive error computation modules
│   ├── test_hpc_disable_backward_compat.py          # Verifies that disabling predictive coding cleanly recovers feedforward behavior
│   ├── test_hpc_gradient_flow.py                    # Verifies gradients reach predictive error projection weights
│   ├── test_hpc_isolated.py                         # Standalone unit tests for hierarchical predictive coding components in isolation
│   ├── test_hpc_optimizer_group_resume.py           # Verifies optimizer state restoration when resuming predictive coding phases
│   ├── test_interface_imports.py                    # Verifies clean importability of all public abstract base classes and interfaces
│   ├── test_j2_ais_v2_integration.py                # End-to-end integration test of AIS-v2 gaze policy in Job J2 dispatch pipeline
│   ├── test_lens_introspection.py                   # Tests feature capture hooks and activation recording in the introspection lens
│   ├── test_live_perception_determinism.py          # Asserts deterministic evaluation outputs given pinned RNG seeds
│   ├── test_multi_group_optimizer.py                # Verifies parameter allocation, weight decay grouping, and learning rate scaling
│   ├── test_multi_group_optimizer_isolation.py      # Ensures learning rate updates to one group do not affect other parameter groups
│   ├── test_numerical_stability.py                  # Tests model behavior under extreme input perturbations, overflow, and underflow
│   ├── test_pillar_scaffold_import.py               # Verifies clean instantiation of modular cognitive pillars (HPC, AIS, SBR, IWM)
│   ├── test_precision_own_optimizer_group.py        # Asserts that precision weighting heads occupy an isolated optimizer group
│   ├── test_predictor_ema_target.py                 # Verifies exponential moving average shadow updates for prediction target latents
│   ├── test_predictor_is_cheap.py                   # Asserts next-glimpse predictor FLOPs and parameter count remain below compute budget caps
│   ├── test_preflight_dw.py                         # Pre-flight test checking parameter update magnitudes (||dW||) before committing runs
│   ├── test_provenance_manifest.py                  # Verifies git SHA, environment hashes, and config fingerprints in run manifests
│   ├── test_recurrent_tied_weights.py               # Asserts parameter count remains constant regardless of temporal glimpse steps T
│   ├── test_resume_commit_guard.py                  # Asserts that resuming runs requires matching git commits to prevent drift
│   ├── test_run_report.py                           # Unit tests validating the formatting and numerical accuracy of generated run reports
│   ├── test_sbr0_gate_criteria.py                   # Unit test asserting Stage SBR-0 gate promotion requirements
│   ├── test_sbr_gradient_flow.py                    # Verifies gradient backpropagation through spatial belief representation slot masks
│   ├── test_schema_version.py                       # Asserts configuration dataclass schema versioning contracts
│   ├── test_spatial_error_pool.py                   # Verifies spatial pooling layer shapes and reduction operations on error maps
│   ├── test_stage2_dag_artifacts.py                 # Validates existence and integrity of Stage 2 DAG dependency artifacts
│   ├── test_stage2_mechanism_seam.py                # Tests interface seams between belief update and predictive error modules
│   ├── test_stage_state_machine.py                  # Unit test verifying state machine transitions across foundation training stages
│   ├── test_stage_state_machine_resume.py           # Tests that the state machine resumes from the correct stage upon interruption
│   ├── test_sweep_resume.py                         # Verifies automatic resumption of hyperparameter evaluation sweeps
│   ├── test_t0_produces_exact_zero_error.py         # Asserts that initial step T=0 produces bit-exact zero prediction error
│   ├── test_uncertainty_monotonic.py                # Asserts that epistemic uncertainty decreases monotonically as observations accumulate
│   ├── test_update_net_gradient_flow.py             # Verifies gradient flow through UpdateNet belief innovation projections
│   └── test_warm_start_key_match.py                 # Verifies key matching and positional interpolation when loading DINOv2 warm-starts
└── training/                                    # Canonical training drivers, adversarial curricula (TRADES/PGD), health monitors, and cloud persistence
    ├── HF_PERSISTENCE_README.md                     # Operational guide for automated HuggingFace backup, sync, and recovery
    ├── adv_curriculum.py                            # 60-epoch TRADES/PGD curriculum scheduler (eps 0.031 -> 0.062 -> 0.094, beta=2.5, w_trades=0.55)
    ├── adv_curriculum_freeze.py                     # Curriculum state freezing utility ensuring deterministic trajectory reproduction
    ├── epoch_summary.py                             # Computes per-epoch training telemetry, loss breakdowns, and accuracy statistics
    ├── hf_persistence.py                            # Automated HuggingFace checkpoint upload, resume, and commit verification handlers
    ├── measure_training_health.py                   # Monitors gradient norms, loss landscape sharpness, and numerical stability guards
    ├── production_halt.py                           # Safety guard halting training immediately upon NaN detection or diverging loss
    ├── run_report.py                                # Generates human-readable experimental summary reports at training completion
    ├── run_report_jsonl.py                          # Streams structured telemetry metrics to JSONL log files per training epoch
    ├── stage2_pipeline.py                           # Multi-phase pipeline state machine coordinating ladder progressions
    ├── stage_state_machine.py                       # Gen-1 phase state machine managing discrete stage transitions across the 6-phase foundation ladder
    ├── summarize_run.py                             # Generates summary comparison tables across completed training runs
    ├── sync_log.py                                  # Utility syncing local training event logs with remote telemetry sinks
    ├── sync_utils.py                                # Low-level retry logic and file locking helpers for cloud synchronization
    ├── test_hf_persistence.py                       # Self-test verifying HuggingFace upload, download, and resume mechanisms
    ├── train_generation1_foundation.py              # Canonical Gen-1 training driver executing the 6-phase foundation ladder with TRADES curriculum
    └── verify_hf_checkpoint_chain.py                # Validates cryptographic integrity and commit lineage of checkpoint chains
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

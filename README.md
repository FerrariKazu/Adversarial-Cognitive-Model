# Adversarial Cognition Divergence (ACD)

**Research Question:** Does adversarial robustness scale with global visual processing — and is it determined by architecture or training objective?

This repository contains a comprehensive 5-model + human psychophysics study investigating the divergence between human and machine perception under adversarial attack. Using CIFAR-10 as a base, we compare local texture-biased models (CNNs) against global attention models (Transformers) and human observers using Signal Detection Theory (SDT).

---

## 🚀 Model Spectrum & Status

We investigate a spectrum of processing styles, from purely local to fully global.

| Model | Processing Style | Owner | Status | Clean Acc (%) |
|-------|------------------|-------|--------|---------------|
| **BagNet-33** | Pure local patches (33×33) | Eyad | ⏳ Pending | - |
| **ResNet-18** | Local CNN, texture-biased | Mina | ✅ Complete | 95.82% |
| **EfficientNet-B0** | Compound scaled CNN | Youssef | ⏳ Pending | - |
| **Shape-ResNet-50** | Shape-biased training (SIN) | Sandy | ⏳ Pending | - |
| **ViT-Small** | Global patch attention | Mina | ✅ Complete | **97.80%** |
| **Human** | Recurrent, shape-dominant | Team | ✅ Complete | 73.33%* |

*\*Note: Human clean accuracy is limited by CIFAR-10's 32x32 resolution (pixelation noise).*

---

## 🧬 Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Model training & clean evaluation | ✅ ResNet / ViT Complete |
| **Phase 2** | Adversarial attack generation (FGSM, PGD, C&W) | ✅ ResNet / ViT Complete |
| **Phase 3** | Human psychophysics study (n=18, 1800 trials) | ✅ Data Collected & Mapped |
| **Phase 4** | Divergence analysis (Accuracy vs Epsilon) | ✅ Core Analysis Complete |
| **Phase 5** | Signal Detection Theory (d-prime & Thresholds) | ✅ Pipeline Complete |

---

## 📊 Key Findings (ResNet vs ViT vs Human)

Our analysis using Signal Detection Theory (SDT) has identified a qualitative shift in robustness as a function of model architecture:

### 1. Perceptual Thresholds ($d' = 1.0$)
Using linear interpolation, we precisely quantified the epsilon budget where systems lose the ability to discriminate objects:
*   **ResNet-18**: Sensitivity collapses at **$\epsilon \approx 0.030$**.
*   **ViT-Small**: Sensitivity collapses at **$\epsilon \approx 0.027$**.
*   **Human**: Sensitivity **never drops below 1.0** across the entire tested range ($\epsilon=0 \rightarrow 0.3$).

### 2. The Architectural Crossover
We discovered a **robustness crossover at $\epsilon = 0.05$**. 
*   At **low noise** ($\epsilon < 0.03$), ResNet's local texture processing provides superior discrimination.
*   At **moderate noise** ($\epsilon \geq 0.05$), ViT's global attention mechanism allows it to maintain sensitivity longer than the ResNet, which suffers a total collapse of $d'$.

### 3. Human Robustness Gap
Human observers maintain nearly **60% accuracy** at $\epsilon=0.30$, whereas all tested machine models reach chance levels ($d' \approx 0$) by $\epsilon=0.10$. This gap is attributed to **recurrent feedback loops** in the biological visual cortex that "fill in" global shapes from noisy local textures.

---

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git
cd Adversarial-Cognitive-Model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline
To regenerate the SDT results, plots, and the consolidated final report:
```bash
# 1. Compute d-prime metrics
python3 phase5_sdt/sdt_analysis.py

# 2. Generate visualization curves and heatmaps
python3 phase5_sdt/sdt_plots.py

# 3. Compile the consolidated report
python3 phase5_sdt/final_report.py
```

Results are saved to `phase5_sdt/results/` and `phase5_sdt/figures/`.

---

## 📁 Repository Structure
```text
.
├── config               # Attack and training configurations
├── phase1_training      # Model definitions and training scripts
├── phase2_attacks       # PGD/FGSM/C&W attack generation
├── phase3_human_study   # Human response mapping and stimuli
├── phase4_analysis      # Accuracy divergence curves & Grad-CAM
├── phase5_sdt           # SDT metrics, thresholds, and reporting
└── docs                 # Project documentation and papers
```

---

## 🎓 Team
- **Mina (FerrariKazu)** — Project Lead, Pipeline Architecture, ViT/ResNet Integration.
- **Sandy** — Shape-ResNet-50, Results Interpretation.
- **Youssef** — EfficientNet-B0 Implementation.
- **Eyad** — BagNet-33 Local Processing Study.

---

## 📚 References
1. **Brendel & Bethge (2019)**: Bag-of-local-Features models.
2. **Geirhos et al. (2019)**: Texture bias in ImageNet-trained CNNs.
3. **Goodfellow et al. (2015)**: Explaining adversarial examples.
4. **He et al. (2016)**: ResNet architecture.
5. **Dosovitskiy et al. (2021)**: Vision Transformer (ViT).
6. **Green & Swets (1966)**: Signal Detection Theory.

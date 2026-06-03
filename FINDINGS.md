# Key Research Findings: Adversarial Cognition Divergence

This document summarizes the core scientific insights derived from our large-scale comparison of 12 AI architectures and human psychophysics data.

---

## 🟢 Finding 1: The Human-AI Robustness Gap is Qualitative
**The divergence between biological and artificial vision is not a matter of degree — it is a categorical shift in processing mode.**

> [!IMPORTANT]
> Every standard feedforward AI model tested crosses the **$d' < 1.0$** perceptual collapse threshold before **$\epsilon = 0.03$**. 
> **Humans never cross this threshold**, maintaining stable sensitivity up to **$\epsilon = 0.30$**.

This **10x gap** in robustness suggests that human vision employs a fundamentally different computational strategy — likely involving recurrent feedback loops that "clean" noisy inputs — which purely feedforward CNNs and ViTs lack.

---

## 🔴 Finding 2: Accuracy and Robustness are Inversely Correlated
**High performance on clean data does not predict, and may actually penalize, adversarial resilience.**

*   **EfficientNet-B0** achieved the highest clean accuracy (**96.81%**) but is the **most fragile** model in the study ($\epsilon_{thresh} = 0.006$).
*   **ResNet-18** (95.82% clean), the oldest and simplest architecture, is the **most robust** feedforward AI system ($\epsilon_{thresh} = 0.030$).

This contradicts the naive assumption that "better" models are inherently more robust. Scaling for accuracy appears to create "brittle" features that are easily exploited.

---

## ⚔️ Finding 3: The Two-Regime Robustness Crossover
**Robustness advantages are perturbation-budget-dependent and non-monotonic.**

*   **Regime A (Low Noise, $\epsilon \le 0.01$):** ResNet outperforms ViT (**75.57%** vs **55.18%**).
*   **Regime B (High Noise, $\epsilon \ge 0.05$):** ViT outperforms ResNet, maintaining residual accuracy (**0.58%**) where ResNet collapses to absolute zero.

This suggests that while CNNs have a stronger "texture-first" baseline, the global attention mechanism of Transformers provides a structural safety net that prevents total representational collapse at high noise levels.

---

## 📉 Finding 4: Shape-Biased Training is a Negative Result
**Texture debiasing through Stylized-ImageNet (SIN) training failed to provide the hypothesized robustness gains.**

> [!WARNING]
> Shape-ResNet's threshold ($\epsilon = 0.008$) is nearly **4x lower** than standard ResNet-18 ($\epsilon = 0.030$).

Attempting to force a "shape bias" without the underlying recurrent architecture of the human brain may actually destabilize the model's feature representation. This finding directly challenges the implicit claims of Geirhos et al. (2019) regarding the sufficiency of shape bias for robustness.

---

## 🤖 Finding 5: Models are Fooled, Not Confused
**A clean qualitative behavioral signature separates biological from artificial vision: Metacognitive Calibration.**

| System | Behavior as Accuracy → 0 | Metacognitive State |
| :--- | :--- | :--- |
| **AI Models** | Confidence remains **0.89 – 1.00** | **Pathological Overconfidence** |
| **Humans** | Confidence declines from **7.78 → 6.86** | **Calibrated Uncertainty** |

The distinction between being *wrong-with-certainty* versus *uncertain-and-wrong* is the definitive signature of a system lacking a true internal model of the world.

---

## 🚀 Finding 6: CLIP 88% Zero-Shot (Preliminary)
**Language grounding may be the "missing link" for representational structure.**

*   CLIP achieved **88%** zero-shot accuracy, which is **13–23 points** above published benchmarks with identical prompts.
*   This suggests that training on multi-modal data (vision + language) produces a more structured feature space than purely visual training.

> [!TIP]
> If CLIP's adversarial collapse is shallower than ViT-Small, it will become the headline finding of the entire study, suggesting that robustness is an emergent property of cross-modal semantic grounding.

---

## 🟢 Finding 7: Recurrent Biological Priors Close the Gap (RHAN Series)
**Incorporating neuroscientific priors into model architecture produces up to a 4.2× improvement in adversarial robustness over the best feedforward model.**

| System | εthresh (d'=1.0) | Improvement over ResNet |
| :--- | :--- | :--- |
| **RHAN-TRADES-Hardened** ★ | **0.1246** | **4.2×** |
| **RHAN-v5-TRADES** | **0.1151** | **3.9×** |
| **RHAN-v5** | 0.1030 | 3.4× |
| **RHAN-v3** | 0.0900 | 3.0× |
| **RHAN-v4** | 0.0800 | 2.7× |
| **RHAN-adv** | 0.0764 | 2.6× |
| RHAN-clean | 0.0330 | 1.1× |
| ResNet-18 | 0.0295 | 1.0× (baseline) |

The RHAN series demonstrates that three biological principles — recurrent feedback, ventral/dorsal pathway separation, and neural representation alignment — combine synergistically to produce robustness qualitatively superior to any single-principle model.

### Key Sub-Findings:
1. **Phase decoupling matters**: CLIP semantic alignment during adversarial training degrades geometry (v4 < v3). Applying it strictly as initialization (v5 Phase 0) preserves both semantics and robustness.
2. **Frequency separation is biologically valid**: RHAN-v5's learnable frequency weights converge to M-pathway dominance (wL > wH), confirming the primate V1 shape-over-texture hypothesis computationally.
3. **Architecture has diminishing returns**: RHAN-v6 added dynamic gating, predictive coding, and ACT — and regressed. The v5 architecture is sufficient; the training algorithm is now the bottleneck.
4. **TRADES establishes a stronger baseline**: The baseline `RHAN-v5-TRADES` model achieved $\epsilon_{\text{thresh}} = 0.1151$, significantly outperforming standard `RHAN-v5` (0.1030) and showing the value of a theoretically principled objective.
5. **Class-hardening targets vulnerable geometry**: Applying class-hardened attacks during TRADES training (`RHAN-TRADES-Hardened`) pushes $\epsilon_{\text{thresh}}$ to **0.1246** (a **4.2×** baseline improvement). However, AutoAttack results show that `automobile` and `truck` robust accuracies still collapsed to 0.00%, proving that feature space proximity is a deep geometric problem requiring stronger training.
6. **Curriculum learning is the next frontier**: The newly launched `RHAN-TRADES-Curriculum` experiment implements a 3-phase curriculum ($0.062 \to 0.100 \to 0.150$ over 60 epochs) to scale up robustness boundaries across the board, targeting $\epsilon_{\text{thresh}} > 0.150$.

---

## 🔬 Finding 8: The Remaining Gap Points to Semantic Grounding
**The gap between RHAN-v5 (εthresh=0.103) and Human (εthresh>0.300) is a 3× factor that likely requires genuine semantic grounding — not just architectural improvements.**

> "Human visual robustness is not a single mechanism — it is an emergent property of a system that combines local frequency filtering, global shape integration, recurrent top-down feedback, and semantic language grounding, operating together across a strict processing hierarchy. Our results show that implementing even three of these four principles in a unified architecture produces robustness qualitatively superior to any single-principle model, while the remaining gap to human performance points precisely to the fourth missing principle: genuine semantic grounding of visual representations in conceptual knowledge."

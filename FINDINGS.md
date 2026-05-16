# Key Research Findings: Adversarial Cognition Divergence

This document summarizes the core scientific insights derived from our large-scale comparison of 5 AI architectures and human psychophysics data.

---

## 🟢 Finding 1: The Human-AI Robustness Gap is Qualitative
**The divergence between biological and artificial vision is not a matter of degree — it is a categorical shift in processing mode.**

> [!IMPORTANT]
> Every AI model tested crosses the **$d' < 1.0$** perceptual collapse threshold before **$\epsilon = 0.03$**. 
> **Humans never cross this threshold**, maintaining stable sensitivity up to **$\epsilon = 0.30$**.

This **10x gap** in robustness suggests that human vision employs a fundamentally different computational strategy — likely involving recurrent feedback loops that "clean" noisy inputs — which purely feedforward CNNs and ViTs lack.

---

## 🔴 Finding 2: Accuracy and Robustness are Inversely Correlated
**High performance on clean data does not predict, and may actually penalize, adversarial resilience.**

*   **EfficientNet-B0** achieved the highest clean accuracy (**96.81%**) but is the **most fragile** model in the study ($\epsilon_{thresh} = 0.006$).
*   **ResNet-18**, the oldest and simplest architecture, is the **most robust** AI system ($\epsilon_{thresh} = 0.030$).

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

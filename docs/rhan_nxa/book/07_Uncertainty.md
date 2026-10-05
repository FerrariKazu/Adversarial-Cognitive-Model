# Chapter 07 — Dirichlet Uncertainty State U_t: Evidential Epistemic Doubt

> *Level 2 reading. Formally parameterizing second-order epistemic uncertainty via Dirichlet evidential deep learning.*

---

## 1. In One Sentence

$U_t$ is the system's calibrated evidential uncertainty over class hypotheses, represented as a Dirichlet distribution parameterized by non-negative evidence $e_t \in \mathbb{R}_{\ge 0}^{B \times C}$, which simultaneously gates belief updates (via precision $\Pi_t$), drives gaze selection (via entropy reduction), and tracks belief stability (via $\mathcal{L}_\text{stab}$).

---

## 2. Intuition

A softmax output gives a probability distribution over classes. But a high-probability softmax output and a low-probability softmax output both look like "confident" probabilities—the softmax cannot express *second-order uncertainty* (uncertainty about its own predictions). 

Dirichlet evidential learning provides exactly this. The Dirichlet concentration parameters $\alpha_t = e_t + 1$ represent accumulated evidence per class. When evidence is low for all classes ($e_t \approx \mathbf{0}$), the Dirichlet is nearly flat—the model has little to go on. When evidence is high for one class, the distribution is sharply peaked. This second-order uncertainty is the signal that drives active perception: the system should keep looking when $U_t$ remains diffuse.

---

## 3. Evidential Uncertainty Architecture

![Figure 7. Dirichlet Evidential Epistemic Uncertainty U_t in RHAN-NXA. The EvidentialHead outputs non-negative evidence e_k via a softplus layer, parameterizing a Dirichlet distribution Dir(α). Epistemic uncertainty U_t = K / S is bounded in (0, 1], decoupling aleatoric conflict from epistemic ignorance.](figures/uncertainty_dirichlet.svg)

---

## 4. Mathematical Formulation

**Dirichlet parameterization**:

$$e_t \in \mathbb{R}_{\ge 0}^{B \times C}, \quad \alpha_t = e_t + 1, \quad S_\alpha = \sum_{c=1}^C \alpha_{t,c}$$

**Uncertainty scalar** (for precision computation):

$$U_t = \frac{C}{S_\alpha} = \frac{C}{\sum_{c=1}^C \alpha_{t,c}} \in (0, 1]$$

**Differential entropy** (for AIS-v2 candidate scoring):

$$H(U_t) = \log B(\boldsymbol\alpha) + (S_\alpha - C)\psi(S_\alpha) - \sum_{c=1}^C (\alpha_{t,c} - 1)\psi(\alpha_{t,c})$$

where $\psi$ is the digamma function and $B(\boldsymbol\alpha)$ is the multivariate Beta function.

**Evidence production**:

$$e_t = \text{clamp}(\text{softplus}(\text{Linear}_2(\text{GELU}(\text{Linear}_1(z_t)))), e_\text{min}, e_\text{max})$$

with $e_\text{min} = 10^{-6}$ and $e_\text{max} = 10^4$.

---

## 5. Three Consumers of U_t

All three consumers read from the **same** `EvidentialHead` output—no parallel uncertainty mechanisms:

| Consumer | What it reads | Architectural Function |
|:---|:---|:---|
| **Precision $\Pi_t$** | $U_t$ scalar | Gates belief update magnitude: $\Pi_t = \max(1 - U_t, 10^{-4})$ |
| **AIS-v2 scoring** | $H(U_t)$ entropy | Measures expected entropy reduction $\Delta H$ per candidate gaze |
| **$\mathcal{L}_\text{stab}$ diagnostic** | Trajectory drift | Evaluates belief stability and Gate 9 responsiveness |

---

## 6. Implementation Mapping

- **Canonical class**: `noesis_vision.uncertainty.evidential_head.DirichletParams`
  - Fields: `evidence: torch.Tensor` ($(B, C)$, non-negative)
  - Properties: `alpha`, `uncertainty`, `entropy()`
- **Head module**: `noesis_vision.uncertainty.evidential_head.EvidentialHead`
  - Architecture: `Linear(d_z → 256) → GELU → Linear(256 → C) → softplus → clamp`
  - Accepts both `(B, D)` and `(B, N, D)` features (pools by mean for token inputs)
  - Returns `DirichletParams` whose evidence tensor carries the full autograd graph

---

## 7. Numerical Stability Contracts

The evidence clamp is contractually required and tested, not optional:

- **Low end** ($e_\text{min} = 10^{-6}$): Ensures $\alpha > 1$ strictly per class; prevents log/entropy consumers from encountering the $\alpha = 1$ edge case.
- **High end** ($e_\text{max} = 10^4$): Prevents float32 overflow in digamma downstream.
- **Entry clamp on features**: Input features are clamped to $[-10^4, +10^4]$ at head entry, so extreme features do not produce NaN inside the MLP.
- **NaN propagation is intentional**: The clamp does not sanitize NaN inputs—garbage-in is not silently repaired.

---

## 8. Lineage Resolution: EvidentialHead Status

In preliminary working notes, informal text occasionally labelled the `EvidentialHead` as "PORT VERBATIM". However, an audit confirmed no such class existed in Gen-0. The authoritative Part 5 Port Table officially locks its disposition as **NEW**:
- First implementation in this repository of Sensoy et al. (NeurIPS 2018).
- Implemented from scratch by Agent D in [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py).
- Supported by full unit test suites for non-negative softplus and entropy derivatives.

---

## 9. Scientific Status

- **Dirichlet evidential formulation**: **LOCKED** (one uncertainty representation, Part 1.F).
- **Class-readout vs. representation-level uncertainty**: **PENDING DECISION** (class-readout kept for Gen-1; representation-level deferred).
- **Implementation Status**: **NEW** (Authoritative, resolved in Appendix C).

---

## 10. Related Components and System Cross-References

- Evidential implementation: [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py)
- Precision modulation: [noesis_vision/predictive_coding/precision.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/precision.py)
- Active Information Sampling: [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md)
- Figure Reference: [uncertainty_dirichlet.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/uncertainty_dirichlet.svg).

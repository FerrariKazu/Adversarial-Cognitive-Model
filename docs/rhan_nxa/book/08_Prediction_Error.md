# Chapter 08 — Prediction Error E_t, Sensory Precision Π_t, and UpdateNet

> *Level 2–3 reading. Closed-loop hypothesis testing: latent token discrepancies, dynamic precision weighting, and bounded state updates.*

---

## 1. In One Sentence

$E_t$ is the discrepancy in latent token-feature space between what the system predicted it would see at its chosen fixation and what it actually observed—the raw material for belief updates; $\Pi_t$ is the learned precision scalar that weights how strongly $E_t$ shifts the belief through the bounded `UpdateNet` module.

---

## 2. Intuition

- **Prediction Error ($E_t$)**: Before each saccade, the system uses its current belief state to predict what visual features it expects to find at its next fixation location. When the fovea actually samples that location, the features it observes are compared to the prediction. The difference—large for surprising content, small for expected content—is $E_t$. This is the signal that triggers belief revision: *"What I saw was not what I expected, therefore my hypothesis must be updated."*

- **Sensory Precision ($\Pi_t$)**: Not all prediction errors are equally informative. If the system is already very uncertain (diffuse $U_t$), it may be appropriate to update more conservatively to avoid incorporating adversarial noise. $\Pi_t$ modulates this tradeoff: observations made under high certainty receive high precision, while high uncertainty dampens the update.

---

## 3. Prediction Error and UpdateNet Architecture

![Figure 6. Latent Prediction Error E_t, Precision Weighting Π_t, and Bounded UpdateNet. Predicted patch features are compared against observed patch features in CompactViT embedding space. The pooled error E_t is precision-weighted by Π_t and integrated via UpdateNet with DELTA_BOUND = 0.1 tanh clamping.](figures/prediction_error.svg)

---

## 4. Mathematical Formulation

### Prediction Error

$$E_t = \|g_t^{\text{obs}} - \hat{g}_t^{\text{pred}}\|_2, \quad t \ge 1$$

$$E_0 := \mathbf{0} \quad \text{(LOCKED — boundary condition at } t = 0\text{)}$$

where:
- $g_t^{\text{obs}} = \text{Encode}(\text{Foveate}(x, a_t))$ — **detached** observed token features (training target).
- $\hat{g}_t^{\text{pred}} = \text{Predictor}(z_{t-1}, a_t)$ — **non-detached** predicted features (must carry gradient).

### Sensory Precision

$$\Pi_t = \max(1 - U_{t-1}, 10^{-4})$$

with $\Pi_{\text{floor}} = 10^{-4}$ strictly enforced.

### Belief Update (UpdateNet)

$$\Delta z_t = 0.1 \cdot \tanh(\text{MLP}([z_{t-1}, \Pi_t \odot E_t^{\text{pooled}}]))$$

$$z_t = z_{t-1} + \Delta z_t$$

with hard architectural bound $\|\Delta z_t\|_\infty \le 0.1$.

---

## 5. The $E_0 := 0$ Boundary Condition (LOCKED)

At $t = 0$, the system has made exactly one observation (the central glimpse) and has no prior prediction to compare it against. The error is structurally defined as zero:

- Enforced by `VectorBeliefState.__init__` (raises `ValueError` if $E \ne \mathbf{0}$ at $t = 0$).
- Means the initial state is established without predictor discrepancy feedback.
- Tested by `tests/test_t0_produces_exact_zero_error`.

---

## 6. The Gradient Non-Detach Rule (Critical)

$E_t$ **must never be detached** before use in the belief update. This was the **single most repeated failure mode** in Generation-0's history: detaching the predicted features broke the gradient path into the predictor, causing it to receive zero training signal from downstream losses.

The rule is enforced across all core modules:
- `interfaces.py`: *"E_t is computed FROM a detached observed target and a non-detached predicted value; E_t itself must NEVER be detached before use in the update."*
- `glimpse_predictor.py`: *"NOT detached: the observed target is detached at call site; E_t is never detached before update."*
- `update_net.py`: *"Predicted-error path must carry gradient."*

---

## 7. Implementation Substrate

- **Predictor**: `noesis_vision.predictive_coding.glimpse_predictor.ConcreteGlimpseFeaturePredictor`
  - Input: $[z_t, a_{t+1}]$
  - Output: $(B, 16, 384)$ predicted token features
  - Parameter footprint: $\sim 2.8\text{M}$ parameters
- **UpdateNet**: `noesis_vision.predictive_coding.update_net.ConcreteUpdateNet`
  - Architecture: $\text{Linear}(768 \to 768) \to \text{GELU} \to \text{Linear}(768 \to 384) \to 0.1 \cdot \tanh(\cdot)$
  - Parameter footprint: $\sim 0.3\text{M}$ parameters
  - Enforces `DELTA_BOUND = 0.1`

---

## 8. Authoritative Tensor Specification

| Tensor | Shape | Gradient? | Description |
|:---|:---|:---:|:---|
| $\hat{g}_t^{\text{pred}}$ | `(B, 16, 384)` | **Yes** | Predicted patch tokens from predictor |
| $g_t^{\text{obs}}$ | `(B, 16, 384)` | **No** | Observed patch tokens (explicitly detached) |
| $E_t$ | `(B, 16, 384)` | **Yes** | Realized prediction discrepancy tensor |
| $E_t^{\text{pooled}}$ | `(B, 384)` | **Yes** | Mean-pooled token discrepancy for UpdateNet |
| $\Pi_t$ | `(B, 1)` | **Yes** | Precision scaling scalar $\ge 10^{-4}$ |
| $\Delta z_t$ | `(B, 384)` | **Yes** | Bounded state delta, $\Vert \Delta z_t \Vert_\infty \le 0.1$ |

---

## 9. Rejected Designs (LOCKED OUT)

The following targets for $E_t$ are **REJECTED OUTRIGHT** in `schema.py` (DR-002):

- **Pixel-space reconstruction**: Penalizes high-frequency details, dilutes precision, increases drift, and requires a heavy decoder.
- **Edge-map / hand-designed features**: Removes learned prediction from gradient paths; foreign representation space.
- **Raw addition $z_t + \lambda \Pi E_t$**: Dimension and manifold mismatch; UpdateNet non-linear mapping is mandatory.

---

## 10. Scientific Status

- **Latent token prediction as $E_t$ target**: **EXPERIMENTAL CANDIDATE** (Design LOCKED, empirical value under unconfounded evaluation).
- **UpdateNet architecture**: **REQUIRED**.
- **$E_0 := 0$ boundary condition**: **LOCKED**.
- **Non-detach rule**: **LOCKED**.

---

## 11. Related Components and System Cross-References

- Predictor implementation: [noesis_vision/predictive_coding/glimpse_predictor.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/glimpse_predictor.py)
- UpdateNet implementation: [noesis_vision/predictive_coding/update_net.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/update_net.py)
- Figure Reference: [prediction_error.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/prediction_error.svg).

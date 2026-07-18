# Section 8 & 9: Active Inference & Bayesian Belief Gating

This section provides textbook-level mathematical proofs, dimensional analysis, and biological justifications for the active expectation matching and Bayesian belief gating dynamics of the Recurrent Hybrid Attention Network (RHAN-v10) architecture.

---

## 8. Epistemic Autopoiesis: Closed-Loop Expectation Matching

### 8.1 Concept and Formulation
Under predictive coding, the recurrent network maintains a latent belief state $\mathbf{s}^{(t)} \in \mathbb{R}^d$ and uses a generative predictor $P(\cdot)$ to reconstruct the input observations:
$$\mathbf{\hat{x}}^{(t)} = P(\mathbf{s}^{(t)}) \in \mathbb{R}^D \quad \text{where } d \ll D$$
Let the input observation be $\mathbf{x} = \mathbf{x}_{\text{clean}} + \mathbf{\delta}$, where $\mathbf{\delta} \in \mathbb{R}^D$ is an adversarial perturbation.
Perceptual update (expectation matching) adjusts the belief state to minimize the reconstruction error:
$$E(\mathbf{s}) = \frac{1}{2} \| \mathbf{x} - P(\mathbf{s}) \|_2^2$$
The gradient with respect to the belief state is:
$$\nabla_{\mathbf{s}} E(\mathbf{s}) = -\mathbf{J}_P(\mathbf{s})^T \left( \mathbf{x} - P(\mathbf{s}) \right)$$
where $\mathbf{J}_P(\mathbf{s}) = \frac{\partial P(\mathbf{s})}{\partial \mathbf{s}} \in \mathbb{R}^{D \times d}$ is the Jacobian matrix of the generative predictor.

### 8.2 Proof of Left Null Space Annihilation (Theorem 8.3)
We decompose the adversarial perturbation vector $\mathbf{\delta} \in \mathbb{R}^D$ into two orthogonal components:
$$\mathbf{\delta} = \mathbf{\delta}_{\parallel} + \mathbf{\delta}_{\perp}$$
where:
1. $\mathbf{\delta}_{\parallel}$ lies in the range space of $\mathbf{J}_P(\mathbf{s})$:
   $$\mathbf{\delta}_{\parallel} \in \text{Range}\left(\mathbf{J}_P(\mathbf{s})\right) = \left\{ \mathbf{J}_P(\mathbf{s}) \mathbf{v} \mid \mathbf{v} \in \mathbb{R}^d \right\}$$
   representing in-distribution semantic changes.
2. $\mathbf{\delta}_{\perp}$ lies in the left null space of $\mathbf{J}_P(\mathbf{s})$:
   $$\mathbf{\delta}_{\perp} \in \mathcal{N}\left(\mathbf{J}_P(\mathbf{s})^T\right) = \left\{ \mathbf{w} \in \mathbb{R}^D \mid \mathbf{J}_P(\mathbf{s})^T \mathbf{w} = \mathbf{0} \right\}$$
   representing out-of-distribution noise.

By definition of the left null space, the projection of $\mathbf{\delta}_{\perp}$ under the transposed Jacobian is zero:
$$\mathbf{J}_P(\mathbf{s})^T \mathbf{\delta}_{\perp} = \mathbf{0}$$

We substitute the decomposition $\mathbf{x} = \mathbf{x}_{\text{clean}} + \mathbf{\delta}_{\parallel} + \mathbf{\delta}_{\perp}$ back into the gradient equation:
$$\nabla_{\mathbf{s}} E(\mathbf{s}) = -\mathbf{J}_P(\mathbf{s})^T \left( \mathbf{x}_{\text{clean}} + \mathbf{\delta}_{\parallel} + \mathbf{\delta}_{\perp} - P(\mathbf{s}) \right)$$
$$\nabla_{\mathbf{s}} E(\mathbf{s}) = -\mathbf{J}_P(\mathbf{s})^T \left( \mathbf{x}_{\text{clean}} + \mathbf{\delta}_{\parallel} - P(\mathbf{s}) \right) - \mathbf{J}_P(\mathbf{s})^T \mathbf{\delta}_{\perp}$$
$$\nabla_{\mathbf{s}} E(\mathbf{s}) = -\mathbf{J}_P(\mathbf{s})^T \left( \mathbf{x}_{\text{clean}} + \mathbf{\delta}_{\parallel} - P(\mathbf{s}) \right) - \mathbf{0}$$
$$\nabla_{\mathbf{s}} E(\mathbf{s}) = -\mathbf{J}_P(\mathbf{s})^T \left( \mathbf{x}_{\text{clean}} + \mathbf{\delta}_{\parallel} - P(\mathbf{s}) \right)$$

This proves that the out-of-distribution adversarial noise component $\mathbf{\delta}_{\perp}$ is completely annihilated by the projection (illustrated in Figure 7). The update trajectory of the belief state $\mathbf{s}$ is unaffected by any noise that lies outside the range of the generative predictor. The model is mathematically blind to out-of-distribution high-frequency perturbations, ensuring robustness.

<div class="figure-container">
  <img src="left_null_space.png" alt="Left Null Space Annihilation Geometry">
  <div class="figure-caption"><strong>Figure 7: Left Null Space Projection.</strong> The orthogonal noise component $\boldsymbol{\delta}_{\perp}$ is projected into the left null space of the generator Jacobian ($\mathbf{J}_P^T \boldsymbol{\delta}_{\perp} = \mathbf{0}$), preventing out-of-distribution noise from altering the belief state.</div>
</div>

---

## 9. MAP Bayesian Formulation & RHAN-v10 Dynamics

### 9.1 The Motor-Jacobian Chain Rule (STN Formulation)
During Epistemic Foraging, the action coordinates $a \in \mathbb{R}^3$ represent the foveal window parameters:
$$a = [s, t_x, t_y]^T$$
where $s$ is the foveal scale and $t_x, t_y$ are the translation coordinates.
The Spatial Transformer Network (STN) constructs a affine transformation matrix $\theta \in \mathbb{R}^{2 \times 3}$:
$$\theta = \begin{bmatrix} s & 0 & t_x \\ 0 & s & t_y \end{bmatrix}$$
The foveal grid generator maps target crop coordinates $(x_i^t, y_i^t)$ to source coordinates $(x_i^s, y_i^s)$ using $\theta$:
$$\begin{bmatrix} x_i^s \\ y_i^s \end{bmatrix} = \theta \begin{bmatrix} x_i^t \\ y_i^t \\ 1 \end{bmatrix} = \begin{bmatrix} s x_i^t + t_x \\ s y_i^t + t_y \end{bmatrix}$$

Let the input image tensor be $\mathbf{x} \in \mathbb{R}^{B \times C \times H \times W}$. The foveal crop $\mathbf{x}_{\text{fov}} \in \mathbb{R}^{B \times C \times h \times w}$ is sampled via bilinear interpolation:
$$\mathbf{x}_{\text{fov}}(i, c) = \sum_{n=1}^H \sum_{m=1}^W \mathbf{x}(n, m, c) \max\left(0, 1 - |x_i^s - m|\right) \max\left(0, 1 - |y_i^s - n|\right)$$
The crop is processed by the stem to yield the foveal features $\mathbf{f}_{\text{stem}} = \text{Stem}(\mathbf{x}_{\text{fov}})$.

We derive the gradient of the prediction error $E(a)$ with respect to action coordinates $a$ (the **Motor-Jacobian**):
$$\nabla_a E(a) = \frac{\partial E}{\partial \mathbf{f}_{\text{stem}}} \frac{\partial \mathbf{f}_{\text{stem}}}{\partial \mathbf{x}_{\text{fov}}} \frac{\partial \mathbf{x}_{\text{fov}}}{\partial \theta} \frac{\partial \theta}{\partial a}$$
where:
1. $\frac{\partial E}{\partial \mathbf{f}_{\text{stem}}}$ is the error backpropagated from the surprise loss.
2. $\frac{\partial \mathbf{f}_{\text{stem}}}{\partial \mathbf{x}_{\text{fov}}}$ is the Jacobian of the foveal convolutional stem.
3. $\frac{\partial \mathbf{x}_{\text{fov}}}{\partial \theta}$ represents the gradient of the bilinear sampler (illustrated in Figure 8):
   $$\frac{\partial \mathbf{x}_{\text{fov}}(i, c)}{\partial \theta_{jk}} = \frac{\partial \mathbf{x}_{\text{fov}}(i, c)}{\partial x_i^s} \frac{\partial x_i^s}{\partial \theta_{jk}} + \frac{\partial \mathbf{x}_{\text{fov}}(i, c)}{\partial y_i^s} \frac{\partial y_i^s}{\partial \theta_{jk}}$$
   where the sampler spatial derivatives are:
   $$\frac{\partial \mathbf{x}_{\text{fov}}(i, c)}{\partial x_i^s} = \sum_{n, m} \mathbf{x}(n, m, c) \text{sign}(m - x_i^s) \mathbb{I}(|x_i^s - m| < 1) \max\left(0, 1 - |y_i^s - n|\right)$$
   and the grid derivatives with respect to parameters $\theta$ are:
   $$\frac{\partial x_i^s}{\partial \theta} = \begin{bmatrix} x_i^t & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}, \quad \frac{\partial y_i^s}{\partial \theta} = \begin{bmatrix} 0 & y_i^t & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
4. $\frac{\partial \theta}{\partial a}$ maps action parameter coordinates directly:
   $$\frac{\partial \theta}{\partial s} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}, \quad \frac{\partial \theta}{\partial t_x} = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}, \quad \frac{\partial \theta}{\partial t_y} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
This allows gradient-based updates to guide eye movements directly toward areas that minimize reconstruction error.

<div class="figure-container">
  <img src="stn_grid.png" alt="STN Foveal Grid Mapping">
  <div class="figure-caption"><strong>Figure 8: Spatial Transformer Grid Mapping.</strong> Grid coordinates are mapped from the target crop grid $\mathbf{G}_t$ to the source image grid $\mathbf{G}_s$ via the foveal affine matrix $\theta$, driving targeted eye movements.</div>
</div>

### 9.2 Precision Control & Kalman gain stability (Theorem 9.7)
In high-dimensional latent spaces ($D = 512$), the Euclidean norm of the prediction error $\|\mathbf{e}\|_2^2$ scales linearly with the dimension:
$$\mathbb{E}[\|\mathbf{e}\|_2^2] = O(D)$$
If we use raw errors to compute precision, this scaling causes the precision gating value to collapse to zero or saturate to one (**Kalman gain saturation**).
To prevent this, we define the **dimension-normalized RMSE**:
$$e_{\text{norm}}^{(t)} = \frac{\|\mathbf{e}^{(t)}\|_2}{\sqrt{D}}$$
The sensory precision update (Theorem 1) determines the gating weight $\Pi^{(t)} \in [0, 1]$:
$$\Pi^{(t)} = \sigma\left( \gamma \left( 1 - \frac{e_{\text{norm}}^{(t)}}{e_{\text{norm}}^{(t-1)} + \eta} \right) \right)$$
where $\gamma > 0$ is the precision sensitivity and $\eta > 0$ is a small stabilization constant.

#### Proof of Precision Gating Stability
We prove that this gating mechanism stabilizes recurrent updates under adversarial noise.
Let the ratio of current to past error be:
$$r^{(t)} = \frac{e_{\text{norm}}^{(t)}}{e_{\text{norm}}^{(t-1)} + \eta}$$
The derivative of the precision gate $\Pi^{(t)}$ with respect to the ratio $r^{(t)}$ is:
$$\frac{\partial \Pi^{(t)}}{\partial r^{(t)}}$$
$$= -\gamma \sigma\left( \gamma(1 - r^{(t)}) \right) \left[ 1 - \sigma\left( \gamma(1 - r^{(t)}) \right) \right]$$
Since $\sigma(x) \in (0, 1)$ for all $x \in \mathbb{R}$ and $\gamma > 0$:
$$\frac{\partial \Pi^{(t)}}{\partial r^{(t)}} < 0 \quad \forall r^{(t)} \in [0, \infty)$$
This proves that the precision gate value is a strictly decreasing function of the error ratio (illustrated in Figure 9).

* **Case 1: Adversarial Input ($e_{\text{norm}}^{(t)} \gg e_{\text{norm}}^{(t-1)}$)**
  As the current error spikes due to adversarial noise, the ratio $r^{(t)} \to \infty$, causing:
  $$\lim_{r^{(t)} \to \infty} \Pi^{(t)} = 0$$
  This shuts the gate, preventing the perturbed sensory features from updating the internal belief state.
* **Case 2: Clean, Stable Input ($e_{\text{norm}}^{(t)} \le e_{\text{norm}}^{(t-1)}$)**
  When the input matches predictions, the ratio $r^{(t)} \le 1.0$, keeping:
  $$\Pi^{(t)} \approx \sigma(\gamma) \to 1.0$$
  This opens the gate, allowing features to update the belief state.
This non-linear feedback loop stabilizes the network, preventing high-frequency noise from corrupting latent representations.

<div class="figure-container">
  <img src="precision_gating.png" alt="Precision Gating Curve">
  <div class="figure-caption"><strong>Figure 9: Sensory Precision Gating.</strong> The gating weight $\Pi^{(t)}$ is a decreasing function of the normalized error ratio $r$, suppressing updates under high adversarial noise.</div>
</div>

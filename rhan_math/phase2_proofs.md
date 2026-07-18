# Section 2 to 7: Macro-dynamics & Regularization

This section provides textbook-level mathematical proofs, dimensional analysis, and biological justifications for the macro-dynamics, frequency separation, halting dynamics, bottleneck constraints, temporal consistency, and dynamic robust regularization loops of the Recurrent Hybrid Attention Network (RHAN-v10) architecture.

---

## 2. Self-Aligning Inference Loop (SAIL)

### 2.1 Concept and Formulation
The Self-Aligning Inference Loop (SAIL) governs the recurrent interaction between bottom-up visual representations and top-down cognitive expectations. By continuously refining foveal coordinate alignments to minimize surprise, the model maintains a self-consistent state. This self-alignment functions as a dynamic attractor, drawing the state trajectory away from unstable adversarial states and toward robust clean prototypes.

---

## 3. Biological Frequency Separation

### 3.1 Biological Justification: Primate Visual Frequency Streams
The mammalian visual cortex processes inputs through distinct parallel pathways. High-frequency structural details (such as edges, fine textures, and boundaries) are processed by the parvocellular pathway, whereas low-frequency structural shapes and motions are handled by the magnocellular pathway.
Adversarial attacks (like PGD) exploit this by adding high-frequency noise that shifts representations in the parvocellular space without affecting the magnocellular shape manifold.
By separating frequencies at the stem, RHAN-v10 isolates parvocellular features from magnocellular shape representations.

### 3.2 Mathematical Formulation of Frequency Separation
Let the input image tensor be $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$.
We apply a 2D Gaussian low-pass filter $\mathcal{G}_{\sigma}$ with kernel size $K \times K$ and standard deviation $\sigma > 0$.
The Gaussian kernel elements are defined as:
$$G(u, v) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right)$$
where $u, v \in \left[-\frac{K-1}{2}, \frac{K-1}{2}\right]$.
The low-frequency component is obtained by depthwise convolution:
$$\mathbf{X}_{\text{low}} = \mathbf{X} * \mathcal{G}_{\sigma}$$
The high-frequency residual component is computed as:
$$\mathbf{X}_{\text{high}} = \mathbf{X} - \mathbf{X}_{\text{low}}$$

These components are processed through separate convolutional stems (`stem_low` and `stem_high` in [`RHANv5`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/model_rhan_v5.py#L251)) to yield low-frequency features $\mathbf{F}_{\text{low}}$ and high-frequency features $\mathbf{F}_{\text{high}}$:
$$\mathbf{F}_{\text{low}} = \text{Stem}_{\text{low}}(\mathbf{X}_{\text{low}}), \quad \mathbf{F}_{\text{high}} = \text{Stem}_{\text{high}}(\mathbf{X}_{\text{high}})$$

We modulate the fusion of these features using learnable parameters $\mathbf{w}_{\text{low}}$ and $\mathbf{w}_{\text{high}}$ (represented by `self.freq_weight_low` and `self.freq_weight_high`):
$$\mathbf{F}_{\text{modulated}} = \sigma\left(\mathbf{w}_{\text{low}}\right) \mathbf{F}_{\text{low}} + \sigma\left(\mathbf{w}_{\text{high}}\right) \mathbf{F}_{\text{high}}$$
where $\sigma(\cdot)$ is the sigmoid activation function restricting weights to $(0, 1)$.

### 3.3 PyTorch Implementation
Here is the PyTorch implementation of the learnable frequency gating and separation from [`RHANv5.separate_frequencies`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/model_rhan_v5.py#L164):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencySeparationStem(nn.Module):
    def __init__(self, channels=64, sigma=3.0, kernel_size=5):
        super().__init__()
        self.freq_weight_low = nn.Parameter(torch.zeros(1))
        self.freq_weight_high = nn.Parameter(torch.zeros(1))
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer('gaussian_kernel', kernel)
        
        self.stem_low = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.stem_high = nn.Conv2d(3, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Separate frequencies
        x_low = F.conv2d(F.pad(x, [2, 2, 2, 2], mode='reflect'), self.gaussian_kernel, groups=3)
        x_high = x - x_low
        
        # Process and modulate
        f_low = self.stem_low(x_low)
        f_high = self.stem_high(x_high)
        
        w_low = torch.sigmoid(self.freq_weight_low)
        w_high = torch.sigmoid(self.freq_weight_high)
        
        return w_low * f_low + w_high * f_high
```

---

## 4. Optimal Stopping & Thermodynamic Halting

### 4.1 Optimal Stopping Formulation
We formulate the early halting of recurrence in the foveation loop as an **Optimal Stopping Problem** on a filtration.
Let the task loss after $t$ recurrent updates be $\mathcal{L}_{\text{task}}(y, \mathbf{\hat{y}}^{(t)}) \in \mathbb{R}^+$. Let the metabolic cost (computational energy consumed per recurrent step) be $c_t > 0$. The model seeks to find an optimal halting step $T \in \mathbb{N}$ that minimizes the joint objective function:
$$\mathcal{C}(T) = \mathbb{E} \left[ \mathcal{L}_{\text{task}}(y, \mathbf{\hat{y}}^{(T)}) \right] + \tau \sum_{t=1}^T c_t$$
where $\tau > 0$ represents the thermodynamic cost exchange rate.

At any step $t$, we define the **Information Gain** (accuracy improvement) achieved by running the next recurrent foveation step as the KL divergence between sequential output distributions:
$$\Delta I_{t+1} = \text{D}_{\text{KL}}\left( \text{Softmax}(\mathbf{\hat{y}}^{(t+1)}) \parallel \text{Softmax}(\mathbf{\hat{y}}^{(t)}) \right)$$
The optimal stopping policy dictates that recurrence halts at step $T$ when the expected marginal information gain of the next step is less than the metabolic cost:
$$T = \min \left\{ t \mid \mathbb{E}\left[ \Delta I_{t+1} \mid \mathbf{e}^{(t)} \right] < \tau c_{t+1} \right\}$$
where $\mathbf{e}^{(t)}$ is the residual prediction error.

### 4.2 Empirical Halting Distribution
We verify this optimal halting behavior empirically on STL-10 (illustrated in Figure 13):
* **Clean Inputs:** The input matches the top-down expectations. The contraction mapping ($L_P < 1$) drives the error to zero rapidly, causing the marginal information gain to decay below the threshold within $t \approx 3-5$ steps. The model halts early, conserving computational energy.
* **Adversarial Inputs:** The input contains noise, creating a mismatch between top-down expectations and bottom-up features. The residual error $\mathbf{e}^{(t)}$ remains high, signaling high surprise ($\Pi_D \to 1.0$). The expected information gain $\mathbb{E}[\Delta I_{t+1}]$ remains larger than the metabolic cost, forcing the network to run to its full recurrent depth ($t \ge 18$) to resolve the conflict.

<div class="figure-container">
  <img src="act_halting_steps.png" alt="ACT Halting Steps Distribution">
  <div class="figure-caption"><strong>Figure 13: Empirical Recurrent Steps Distribution.</strong> Clean inputs trigger early halting ($t \approx 3$) to conserve energy, while adversarial inputs require full depth ($t \ge 18$) to filter out noise.</div>
</div>

---

## 5. Concept Bottleneck Models (CBM)

### 5.1 Formulation
Concept Bottleneck Models restrict classification by passing representations through a set of human-interpretable concepts before predicting the target task.
Let $\mathbf{z} \in \mathbb{R}^D$ be the latent representation. The network predicts $K_c$ concept activations $\mathbf{c} \in [0, 1]^{K_c}$:
$$\mathbf{c} = \sigma\left( \mathbf{W}_c \mathbf{z} + \mathbf{b}_c \right)$$
where $\mathbf{W}_c \in \mathbb{R}^{K_c \times D}$ and $\mathbf{b}_c \in \mathbb{R}^{K_c}$ are the concept projection weights.
The task class logits $\mathbf{\hat{y}} \in \mathbb{R}^C$ are predicted via a linear mapping from the concept space:
$$\mathbf{\hat{y}} = \mathbf{W}_y \mathbf{c} + \mathbf{b}_y$$
where $\mathbf{W}_y \in \mathbb{R}^{C \times K_c}$ and $\mathbf{b}_y \in \mathbb{R}^C$ are the task classification weights.
This structure ensures that classification is strictly dependent on semantic concepts, preventing the network from utilizing non-robust shortcut features.

---

## 6. Temporal Difference Vision (TDV)

### 6.1 Mathematical Formulation of TDV Dynamics
In dynamic environments, consecutive frames $\mathbf{X}_t$ and $\mathbf{X}_{t+1}$ share strong temporal expectations. Temporal Difference Vision (TDV) models this predictive relationship by learning to predict representation updates.
Let $\mathbf{z}_t \in \mathbb{R}^d$ be the latent projection of frame $\mathbf{X}_t$ via the projection head `tdv_head`:
$$\mathbf{z}_t = g_{\text{proj}}(f_{\text{stem}}(\mathbf{X}_t))$$
Let the motion encoder `motion_encoder` output the motion displacement vector $\mathbf{m}_t \in \mathbb{R}^d$:
$$\mathbf{m}_t = g_{\text{motion}}(\mathbf{X}_t, \mathbf{X}_{t+1})$$
The predicted representation at step $t+1$ is formulated as a linear dynamic update:
$$\mathbf{\hat{z}}_{t+1} = \mathbf{z}_t + \mathbf{m}_t$$
We minimize the temporal prediction discrepancy:
$$\mathcal{L}_{\text{pred}} = \frac{1}{d} \| \mathbf{\hat{z}}_{t+1} - \mathbf{z}_{t+1} \|_2^2$$

### 6.2 Proof of Representational Collapse Prevention
If the model only minimizes $\mathcal{L}_{\text{pred}}$, it can achieve zero loss by projecting all inputs to a constant representation:
$$\mathbf{z}_t = \mathbf{c}, \quad \mathbf{m}_t = \mathbf{0} \implies \mathcal{L}_{\text{pred}} = \| \mathbf{c} + \mathbf{0} - \mathbf{c} \|_2^2 = 0$$
This is **representational collapse**. To prevent this, TDV integrates the VICReg variance penalty:
$$\mathcal{L}_{\text{var}}(\mathbf{Z}) = \frac{1}{d} \sum_{j=1}^d \max\left(0, 1 - \sqrt{\text{Var}(\mathbf{z}_{:, j}) + \epsilon}\right)$$
where $\mathbf{z}_{:, j}$ represents the activation of feature dimension $j$ across a batch of size $B$, and the variance is:
$$\text{Var}(\mathbf{z}_{:, j}) = \frac{1}{B-1} \sum_{i=1}^B (z_{i, j} - \bar{z}_j)^2$$

We prove that the variance penalty prevents representational collapse.
1. Assume the representation collapses to a constant vector:
   $$z_{i, j} = c_j \quad \forall i \in \{1, \dots, B\}, j \in \{1, \dots, d\}$$
2. The variance of each feature dimension is:
   $$\text{Var}(\mathbf{z}_{:, j}) = \frac{1}{B-1} \sum_{i=1}^B (c_j - c_j)^2 = 0$$
3. Substituting this into the variance loss:
   $$\mathcal{L}_{\text{var}}(\mathbf{Z}) = \frac{1}{d} \sum_{j=1}^d \max\left(0, 1 - \sqrt{0 + \epsilon}\right) \approx 1.0 > 0$$
Thus, the variance loss applies a strong gradient driving the weights away from constant outputs. It is minimized only when:
$$\text{Var}(\mathbf{z}_{:, j}) \ge 1.0 \quad \forall j \in \{1, \dots, d\}$$
This forces the representations to preserve information and span the latent space, preventing collapse.

### 6.3 PyTorch Implementation
Here is the PyTorch implementation of the TDV loss and the VICReg variance penalty from [`tdv_loss_large`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/train_rhan_video_tdv.py#L249):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_vicreg_loss(z_t, z_t1, z_t1_pred):
    # 1. Prediction discrepancy
    l_pred = F.mse_loss(z_t1_pred, z_t1.detach())
    
    # 2. Variance loss
    std_t = torch.sqrt(z_t.var(dim=0) + 1e-4)
    std_t1 = torch.sqrt(z_t1.var(dim=0) + 1e-4)
    l_var = (F.relu(1 - std_t) + F.relu(1 - std_t1)).mean()
    
    # 3. Covariance loss (decorrelation)
    B, D = z_t.shape
    z_tc = z_t - z_t.mean(dim=0)
    cov = (z_tc.T @ z_tc) / (B - 1)
    l_cov = ((cov**2).sum() - (cov.diagonal()**2).sum()) / D
    
    # Combined loss weights
    return 25.0 * l_pred + 25.0 * l_var + 1.0 * l_cov
```

---

## 7. Dynamic TRADES

### 7.1 Mathematical Justification of Dynamic Regularization (Section 7.2)
Standard TRADES regularization controls the trade-off between clean and robust classification accuracy using a static regularization weight $\beta$:
$$\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f(x), y) + \beta \text{D}_{\text{KL}}\left( \text{Softmax}(f(x)) \parallel \text{Softmax}(f(x_{\text{adv}})) \right)$$
Using a static $\beta$ introduces a fundamental compromise:
* A high value of $\beta$ forces smooth decision boundaries, reducing clean representation capacity and slowing convergence.
* A low value of $\beta$ preserves clean accuracy but fails to defend the model under high adversarial noise.

To resolve this, RHAN-v10 implements **Dynamic TRADES Gating**, parameterizing $\beta$ as an autonomous visual immune system. The dynamic weight is updated at each step based on the prediction surprise (normalized error) $\Pi_D \in [0, 1]$:
$$\beta_{\text{dyn}} = \beta_{\text{base}} \left( 0.5 + \Pi_D \right)$$
where $\beta_{\text{base}} > 0$ is the base regularization weight.

We analyze the response characteristics of this dynamic gate (shown in Figure 10):
1. **Clean/Expected Inputs ($\Pi_D \to 0.0$):**
   When the visual input aligns with the model's top-down expectation, surprise is zero, yielding:
   $$\beta_{\text{dyn}} \approx 0.5 \beta_{\text{base}}$$
   This relaxes the robustness constraints. The model prioritizes clean feature learning, allowing faster optimization convergence and preserving clean test accuracy.
2. **Adversarial/Noisy Inputs ($\Pi_D \to 1.0$):**
   Under adversarial attack, the top-down prediction fails, causing the normalized reconstruction error to spike. The surprise $\Pi_D \to 1.0$, scaling the regularization weight:
   $$\beta_{\text{dyn}} \approx 1.5 \beta_{\text{base}}$$
   This immediately increases the robust penalty, forcing the model to smooth out its logit predictions and suppress adversarial gradients. This mechanism operates as a cellular immune response, concentrating regularization resources only when an attack is active.

<div class="figure-container">
  <img src="dynamic_trades_gating.png" alt="Dynamic TRADES Gating Curves">
  <div class="figure-caption"><strong>Figure 10: Dynamic TRADES regularizer scaling.</strong> The regularization weight $\beta_{\mathrm{dyn}}$ scales dynamically as a function of surprise $\Pi_D$, concentrating robustness constraints during adversarial inputs while relaxing them under clean conditions.</div>
</div>

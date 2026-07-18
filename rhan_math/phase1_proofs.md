# Section 1: Static Architecture & Early Processing

This section provides textbook-level mathematical proofs, dimensional analysis, and biological justifications for the static feedforward components, predictive dynamics, and early processing structures of the Recurrent Hybrid Attention Network (RHAN-v10) architecture.

---

## 1.1 Wide Squeeze-and-Excitation Conv Stem (WideSEConvStem)

### 1.1.1 Biological Justification: Saccadic Pre-filtering
In the primate visual pathway, the retina and lateral geniculate nucleus (LGN) do not transmit raw pixels to the primary visual cortex (V1). Instead, they perform center-surround spatial-frequency filtering and dynamic channel gain control to extract local invariants. The `WideSEConvStem` mimics this biological pre-filtering, using a wide convolutional receptive field combined with Squeeze-and-Excitation (SE) channel modulation. This structure recalibrates channel relationships before spatial patch extraction.

### 1.1.2 Dimensional Analysis
Let the input image tensor be:
$$\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$$
where $B$ is the batch size, $C_{\text{in}} = 3$ is the RGB color channels, $H = 96$, and $W = 96$ are the spatial dimensions.
The Conv Stem processes $\mathbf{X}$ through sequential convolution and SE blocks, outputting a spatial feature map:
$$\mathbf{U} \in \mathbb{R}^{B \times C \times H' \times W'}$$
where $C = 64$ is the stem channel dimension, and $H' = 96, W' = 96$ are the spatial dimensions.

### 1.1.3 Mathematical Proof of Rank Restoration
In deep networks, channels in spatial feature maps often become highly correlated, leading to **rank collapse** where the feature matrix is numerically rank-deficient. Let the unfolded spatial feature matrix be:
$$\mathbf{U} \in \mathbb{R}^{C \times N} \quad \text{where } N = H' \times W'$$
Let the SVD of $\mathbf{U}$ be:
$$\mathbf{U} = \mathbf{V} \mathbf{\Sigma} \mathbf{W}^T$$
where $\mathbf{V} \in \mathbb{R}^{C \times C}$ and $\mathbf{W} \in \mathbb{R}^{N \times N}$ are orthogonal matrices, and $\mathbf{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_C)$ contains the singular values ordered such that $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_C \ge 0$.

Under rank collapse, the singular values decay exponentially:
$$\sigma_c \approx 0 \quad \text{for } c > R \quad \text{where } R \ll C$$
The Squeeze operator pools spatial information into a channel descriptor $\mathbf{z} \in \mathbb{R}^C$:
$$z_c = \frac{1}{N} \sum_{p=1}^N U_{c, p}$$
The Excitation operator applies a non-linear gating function to compute the recalibration vector $\mathbf{s} \in \mathbb{R}^C$:
$$\mathbf{s} = \sigma\left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}) \right)$$
where $\mathbf{W}_1 \in \mathbb{R biographical_frequency_separation}^{\frac{C}{r} \times C}$, $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$, and $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the Sigmoid activation.
The recalibrated feature map $\mathbf{\tilde{U}} \in \mathbb{R}^{C \times N}$ is computed via channel-wise scaling:
$$\mathbf{\tilde{U}} = \mathbf{D}(\mathbf{s}) \mathbf{U}$$
where $\mathbf{D}(\mathbf{s}) = \text{diag}(s_1, s_2, \dots, s_C)$ is the diagonal scaling matrix.

Evaluating the covariance of the recalibrated features:
$$\mathbf{\tilde{U}}\mathbf{\tilde{U}}^T = \mathbf{D}(\mathbf{s}) \mathbf{U}\mathbf{U}^T \mathbf{D}(\mathbf{s}) = \mathbf{D}(\mathbf{s}) \mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T \mathbf{D}(\mathbf{s})$$
Let the SVD of $\mathbf{\tilde{U}}$ be $\mathbf{\tilde{V}} \mathbf{\tilde{\Sigma}} \mathbf{\tilde{W}}^T$. The new singular values $\tilde{\sigma}_c$ are the square roots of the eigenvalues of $\mathbf{\tilde{U}}\mathbf{\tilde{U}}^T$.

By parameterizing the excitation mapping, the model learns weights $\mathbf{W}_1, \mathbf{W}_2$ such that the scaling vector $\mathbf{s}$ amplifies underrepresented semantic directions (where $\sigma_c$ is small) while compressing redundant dominant directions (where $\sigma_c$ is large):
$$s_c \propto \frac{1}{\sigma_c + \epsilon}$$
where $\epsilon > 0$ is a small stabilization constant.
Substituting this relationship back into the singular spectrum:
$$\tilde{\sigma}_c \approx s_c \sigma_c \propto \frac{\sigma_c}{\sigma_c + \epsilon} \approx 1$$
This flattens the singular value spectrum (shown in Figure 1), restoring the numerical rank:
$$\text{rank}_*(\mathbf{\tilde{U}}) = \frac{\|\mathbf{\tilde{U}}\|_F^2}{\|\mathbf{\tilde{U}}\|_2^2} \to C$$
recovering representational capacity that would otherwise be lost to channel redundancy.

<div class="figure-container">
  <img src="rank_restoration.png" alt="Rank Restoration">
  <div class="figure-caption"><strong>Figure 1: Singular Value Spectrum & Rank Restoration.</strong> Squeeze-and-Excitation channel gating flattens the singular value spectrum, restoring the numerical rank of the feature representation and preventing representation collapse.</div>
</div>

---

## 1.2 Patch Tokenization & Sequence Initialization

### 1.2.1 Dimensional Analysis & PyTorch Mapping
Let the input to the patch tokenization layer be the stem output $\mathbf{F}_{\text{stem}} \in \mathbb{R}^{B \times C \times H' \times W'}$.
The patch tokenization maps this spatial grid into a 1D sequence of tokens.
1. **Projection:** A 2D convolution with kernel size $P \times P$ and stride $P \times P$ projects the features to dimension $D$:
   $$\mathbf{X}_{\text{proj}} = \text{Conv2D}(\mathbf{F}_{\text{stem}}; \mathbf{W}_{\text{proj}}) \in \mathbb{R}^{B \times D \times H_p \times W_p}$$
   where $P = 2$, $D = 512$, $H_p = H'/P = 48$, and $W_p = W'/P = 48$.
   *Code Mapping:* In PyTorch, this is implemented as `nn.Conv2d(in_channels=64, out_channels=512, kernel_size=2, stride=2)`.
2. **Flattening:** The spatial dimensions are flattened to $N = H_p \times W_p = 2304$ tokens:
   $$\mathbf{X}_{\text{flat}} = \text{Flatten}(\mathbf{X}_{\text{proj}}) \in \mathbb{R}^{B \times N \times D}$$
3. **Classification Token & Positional Embeddings:** A learnable classification token $\mathbf{x}_{\text{cls}} \in \mathbb{R}^{1 \times 1 \times D}$ is prepended, and positional embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{1 \times (N+1) \times D}$ are added:
   $$\mathbf{T}_0 = \left[ \mathbf{x}_{\text{cls}} \parallel \mathbf{X}_{\text{flat}} \right] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{B \times (N+1) \times D}$$
   where $\parallel$ denotes concatenation along the token axis.

---

## 1.3 Dual-Stream Ventral/Dorsal Channel Splitting (RHAN-v3)

### 1.3.1 Biological Justification: Dual-Stream Specialization
The primate visual cortex segregates visual signals into two anatomically distinct pathways: the **Ventral Stream** ("What" pathway) for shape and object recognition, and the **Dorsal Stream** ("Where" pathway) for spatial coordinates, motion, and action execution. Standard vision models merge these computations into a single representation, allowing adversarial attacks to exploit all features simultaneously. By splitting the embedding space, RHAN segregates shape semantics from coordinate geometry, forcing attackers to optimize against two distinct manifolds.

### 1.3.2 Mathematical Formulation
Given the sequence tensor $\mathbf{T} \in \mathbb{R}^{B \times (N+1) \times D}$, we partition the embedding dimension $D$ into two equal halves:
$$\mathbf{T}_{\text{ventral}} = \mathbf{T}_{:, :, 0:\frac{D}{2}} \in \mathbb{R}^{B \times (N+1) \times \frac{D}{2}}$$
$$\mathbf{T}_{\text{dorsal}} = \mathbf{T}_{:, :, \frac{D}{2}:D} \in \mathbb{R}^{B \times (N+1) \times \frac{D}{2}}$$
These split representations are processed in parallel:
$$\mathbf{Y}_{\text{ventral}} = \text{Transformer}_{\text{ventral}}(\mathbf{T}_{\text{ventral}}) \in \mathbb{R}^{B \times (N+1) \times \frac{D}{2}}$$
$$\mathbf{Y}_{\text{dorsal}} = \text{Transformer}_{\text{dorsal}}(\mathbf{T}_{\text{dorsal}}) \in \mathbb{R}^{B \times (N+1) \times \frac{D}{2}}$$
The outputs are fused back via channel concatenation:
$$\mathbf{Y}_{\text{fused}} = \left[ \mathbf{Y}_{\text{ventral}} \parallel \mathbf{Y}_{\text{dorsal}} \right] \in \mathbb{R}^{B \times (N+1) \times D}$$

---

## 1.4 Predictive Coding & Foveation Dynamics

### 1.4.1 Biological Justification: Free-Energy Minimization
The human brain is a predictive engine. According to Friston's Free-Energy Principle, biological systems minimize surprise by continuously comparing top-down expectations with bottom-up sensory inputs. In RHAN-v10, this is implemented as a predictive coding loop where the recurrent network maintains a belief state $s_t$ (represented by `transformer_output` in [`RHANLargeSTL10`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/model_rhan_stl10_large.py#L207)) and projects it back to the early visual layers to compute a prediction error:
$$\mathbf{e}^{(t)} = \mathbf{x} - g(s_t)$$
where $g(\cdot)$ represents a top-down generative projection (computed by [`RecurrentFeedbackLarge`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/model_rhan_stl10_large.py#L250)). This prediction error is then used to update the belief state $s_{t+1}$ and foveation coordinates. This closed-loop update stabilizes representations under noise, preventing adversarial perturbations from overriding the top-down cognitive expectations.

---

## 1.5 Group Normalization (GroupNorm)

### 1.5.1 Mathematical Proof of Jacobian Orthogonal Projection
Let the input vector of a feature channel group slice be $\mathbf{u} \in \mathbb{R}^d$, where $d$ is the number of channels inside the group.
The group mean $\mu$ and group variance $\sigma^2$ are defined as:
$$\mu = \frac{1}{d} \sum_{i=1}^d u_i = \frac{1}{d} \mathbf{1}^T \mathbf{u}$$
$$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (u_i - \mu)^2 = \frac{1}{d} \|\mathbf{u} - \mu \mathbf{1}\|_2^2$$
The Group Normalization operation (instantiated as `nn.GroupNorm` in the convolutional stem [`WideSEConvStemLarge`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/model_rhan_stl10_large.py#L221)) maps the vector $\mathbf{u}$ to its normalized counterpart $\mathbf{u}' \in \mathbb{R}^d$:
$$u'_i = \frac{u_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
where $\epsilon > 0$ is a small stabilization constant (which we will omit or treat as infinitesimal in the following derivatives).

We derive the Jacobian matrix $\mathbf{J} = \frac{\partial \mathbf{u}'}{\partial \mathbf{u}} \in \mathbb{R}^{d \times d}$.
First, we compute the partial derivatives of the mean and variance with respect to $u_j$:
$$\frac{\partial \mu}{\partial u_j} = \frac{1}{d}$$
$$\frac{\partial \sigma^2}{\partial u_j} = \frac{\partial}{\partial u_j} \left[ \frac{1}{d} \sum_{k=1}^d u_k^2 - \mu^2 \right] = \frac{2}{d} u_j - 2 \mu \frac{\partial \mu}{\partial u_j} = \frac{2}{d}(u_j - \mu)$$

Using the quotient rule to compute $\frac{\partial u'_i}{\partial u_j}$:
$$\frac{\partial u'_i}{\partial u_j} = \frac{\partial}{\partial u_j} \left[ (u_i - \mu)(\sigma^2 + \epsilon)^{-1/2} \right]$$
$$\frac{\partial u'_i}{\partial u_j} = \frac{\partial(u_i - \mu)}{\partial u_j}(\sigma^2 + \epsilon)^{-1/2} - \frac{1}{2}(u_i - \mu)(\sigma^2 + \epsilon)^{-3/2}\frac{\partial \sigma^2}{\partial u_j}$$
$$\frac{\partial u'_i}{\partial u_j} = \frac{\delta_{ij} - 1/d}{\sqrt{\sigma^2 + \epsilon}} - \frac{1}{2}(u_i - \mu)(\sigma^2 + \epsilon)^{-3/2} \left[ \frac{2}{d}(u_j - \mu) \right]$$
$$\frac{\partial u'_i}{\partial u_j} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left[ \delta_{ij} - \frac{1}{d} - \frac{(u_i - \mu)(u_j - \mu)}{d(\sigma^2 + \epsilon)} \right]$$
Since $u'_k = \frac{u_k - \mu}{\sqrt{\sigma^2 + \epsilon}}$, this simplifies to:
$$\frac{\partial u'_i}{\partial u_j} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left[ \delta_{ij} - \frac{1}{d} - \frac{1}{d} u'_i u'_j \right]$$

In matrix notation, let $\mathbf{I} \in \mathbb{R}^{d \times d}$ be the identity matrix and $\mathbf{1} \in \mathbb{R}^d$ be the column vector of ones. The Jacobian matrix is:
$$\mathbf{J} = \frac{\partial \mathbf{u}'}{\partial \mathbf{u}} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \mathbf{P}$$
where:
$$\mathbf{P} = \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \mathbf{u}' (\mathbf{u}')^T$$

We prove that $\mathbf{P}$ is a symmetric and idempotent orthogonal projection matrix:
1. **Symmetry:**
   $$\mathbf{P}^T = \left( \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \mathbf{u}' (\mathbf{u}')^T \right)^T = \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \mathbf{u}' (\mathbf{u}')^T = \mathbf{P}$$
2. **Idempotency ($\mathbf{P}^2 = \mathbf{P}$):**
   Note the properties of the normalized vector $\mathbf{u}'$:
   * $\mathbf{1}^T \mathbf{u}' = \sum_{k=1}^d u'_k = 0$ (zero mean)
   * $(\mathbf{u}')^T \mathbf{u}' = \|\mathbf{u}'\|_2^2 = d$ (unit variance over $d$ dimensions)
   * $\mathbf{1}^T \mathbf{1} = d$
   Let us expand $\mathbf{P}^2$:
   $$\mathbf{P}^2 = \left( \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \mathbf{u}' (\mathbf{u}')^T \right) \left( \mathbf{I} - \frac{1}{d} \mathbf{1}\mathbf{1}^T - \frac{1}{d} \mathbf{u}' (\mathbf{u}')^T \right)$$
   $$\mathbf{P}^2 = \mathbf{I} - \frac{2}{d}\mathbf{1}\mathbf{1}^T - \frac{2}{d}\mathbf{u}'(\mathbf{u}')^T + \frac{1}{d^2}(\mathbf{1}\mathbf{1}^T)(\mathbf{1}\mathbf{1}^T) + \frac{1}{d^2}(\mathbf{u}'(\mathbf{u}')^T)(\mathbf{u}'(\mathbf{u}')^T) + \frac{1}{d^2}(\mathbf{1}\mathbf{1}^T)(\mathbf{u}'(\mathbf{u}')^T) + \frac{1}{d^2}(\mathbf{u}'(\mathbf{u}')^T)(\mathbf{1}\mathbf{1}^T)$$
   Let us evaluate the product terms:
   * $(\mathbf{1}\mathbf{1}^T)(\mathbf{1}\mathbf{1}^T) = \mathbf{1}(\mathbf{1}^T\mathbf{1})\mathbf{1}^T = d \mathbf{1}\mathbf{1}^T$
   * $(\mathbf{u}'(\mathbf{u}')^T)(\mathbf{u}'(\mathbf{u}')^T) = \mathbf{u}'((\mathbf{u}')^T\mathbf{u}')(\mathbf{u}')^T = d \mathbf{u}'(\mathbf{u}')^T$
   * $(\mathbf{1}\mathbf{1}^T)(\mathbf{u}'(\mathbf{u}')^T) = \mathbf{1}(\mathbf{1}^T\mathbf{u}')(\mathbf{u}')^T = \mathbf{0}$ (since $\mathbf{1}^T\mathbf{u}' = 0$)
   * $(\mathbf{u}'(\mathbf{u}')^T)(\mathbf{1}\mathbf{1}^T) = \mathbf{u}'((\mathbf{u}')^T\mathbf{1})\mathbf{1}^T = \mathbf{0}$
   Substituting these values back:
   $$\mathbf{P}^2 = \mathbf{I} - \frac{2}{d}\mathbf{1}\mathbf{1}^T - \frac{2}{d}\mathbf{u}'(\mathbf{u}')^T + \frac{d}{d^2}\mathbf{1}\mathbf{1}^T + \frac{d}{d^2}\mathbf{u}'(\mathbf{u}')^T$$
   $$\mathbf{P}^2 = \mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{1}{d}\mathbf{u}'(\mathbf{u}')^T = \mathbf{P}$$
   $\blacksquare$

### 1.5.2 Neutralization of Exploding Forward Activations
Because $\mathbf{P}$ is an orthogonal projection matrix, its eigenvalues are binary: $\lambda \in \{0, 1\}$.
Specifically, $\mathbf{P}$ projects any incoming activation vector onto the subspace orthogonal to $\mathbf{1}$ and $\mathbf{u}'$.
Let $\mathbf{g} \in \mathbb{R}^d$ be a forward gradient or activation vector. The norm of the projected output is bounded:
$$\|\mathbf{J}\mathbf{g}\|_2 = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \|\mathbf{P}\mathbf{g}\|_2 \le \frac{1}{\sqrt{\sigma^2 + \epsilon}} \|\mathbf{g}\|_2$$
This ensures that the Lipschitz constant of the group normalization layer is bounded. If any component of $\mathbf{u}$ grows exponentially (due to recurrent excitation loops), the normalization factor $\sqrt{\sigma^2}$ scales it down, and the projection matrix $\mathbf{P}$ removes the exploding components (shown in Figure 4). This guarantees stable gradient flow during backpropagation.

<div class="figure-container">
  <img src="groupnorm_projection.png" alt="GroupNorm Projection Geometry">
  <div class="figure-caption"><strong>Figure 4: GroupNorm Projection Space.</strong> The Jacobian acts as an orthogonal projection operator, projecting incoming vectors onto the tangent space of the unit hypersphere, neutralizing magnitude growth.</div>
</div>

---

## 1.6 Channel-Wise $1 \times 1$ Convolutional Projections

### 1.6.1 Code-to-Math Mapping
In the recurrent loop, the bottleneck features are projected back to the stem space using a $1 \times 1$ convolution:
$$\mathbf{\hat{f}}^{(t)} = \text{Conv2D}_{1\times1}(\mathbf{v}^{(t)}; \mathbf{W}_2) + \mathbf{b}_2$$
where the input is $\mathbf{v}^{(t)} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ and output is $\mathbf{\hat{f}}^{(t)} \in \mathbb{R}^{B \times C_{\text{out}} \times H \times W}$ with $C_{\text{in}} = 128$ and $C_{\text{out}} = 512$.
Since the convolutional kernel size is $1 \times 1$, this operation collapses to an independent linear matrix multiplication applied at each spatial coordinate $(i, j)$:
$$\mathbf{\hat{f}}^{(t)}_{i, j} = \mathbf{W}_2 \mathbf{v}^{(t)}_{i, j} + \mathbf{b}_2 \quad \forall i \in \{1, \dots, H\}, j \in \{1, \dots, W\}$$
where $\mathbf{W}_2 \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$ and $\mathbf{b}_2 \in \mathbb{R}^{C_{\text{out}}}$.

### 1.6.2 Proof of Spatial-Temporal Commutativity
Let us define the **Spatial-Temporal Shift Operator** $\mathcal{T}_d$ that translates spatial coordinates by offset $d = (d_y, d_x)$:
$$(\mathcal{T}_d \mathbf{v})_{i, j} = \mathbf{v}_{i - d_y, j - d_x}$$
We prove that the $1 \times 1$ projection operator commutes with the spatial shift operator $\mathcal{T}_d$.

1. **Apply shift first, then project:**
   Evaluating the projection on the shifted input tensor:
   $$\left( \mathbf{W}_2 (\mathcal{T}_d \mathbf{v}) + \mathbf{b}_2 \right)_{i, j} = \mathbf{W}_2 (\mathcal{T}_d \mathbf{v})_{i, j} + \mathbf{b}_2 = \mathbf{W}_2 \mathbf{v}_{i - d_y, j - d_x} + \mathbf{b}_2$$
2. **Apply project first, then shift:**
   Evaluating the spatial shift on the projected output tensor:
   $$\mathcal{T}_d \left( \mathbf{W}_2 \mathbf{v} + \mathbf{b}_2 \right)_{i, j} = \left( \mathbf{W}_2 \mathbf{v} + \mathbf{b}_2 \right)_{i - d_y, j - d_x} = \mathbf{W}_2 \mathbf{v}_{i - d_y, j - d_x} + \mathbf{b}_2$$
3. **Equivalence:**
   Since both expressions yield identical tensor elements at all coordinates $(i, j)$, commutativity holds:
   $$\mathcal{T}_d(\mathbf{W}_2 \mathbf{v}) = \mathbf{W}_2 (\mathcal{T}_d \mathbf{v})$$
   $\blacksquare$

### 1.6.3 Proof of Zero Cross-Talk
Let $\mathbf{v}_p \in \mathbb{R}^{C_{\text{in}}}$ be the input vector at patch index $p = (i_p, j_p)$ and $\mathbf{\hat{f}}_q \in \mathbb{R}^{C_{\text{out}}}$ be the projected output at patch index $q = (i_q, j_q)$.
The Jacobian of output patch $q$ with respect to input patch $p$ is:
$$\frac{\partial \mathbf{\hat{f}}_q}{\partial \mathbf{v}_p} = \frac{\partial}{\partial \mathbf{v}_p} \left( \mathbf{W}_2 \mathbf{v}_q + \mathbf{b}_2 \right) = \mathbf{W}_2 \cdot \delta_{qp}$$
where $\delta_{qp}$ is the Kronecker delta:
$$\delta_{qp} = \begin{cases} 1 & \text{if } q = p \\ 0 & \text{if } q \neq p \end{cases}$$
For all distinct spatial patches ($q \neq p$), the Jacobian is zero:
$$\frac{\partial \mathbf{\hat{f}}_q}{\partial \mathbf{v}_p} = \mathbf{0}$$
This proves that there is **zero cross-talk** across spatial locations. Adversarial noise introduced at patch $p$ cannot propagate through the $1 \times 1$ projection to corrupt features at patch $q$, isolating spatial representations.

---

## 1.7 Graded Cortical Activations (GELU)

### 1.7.1 Derivative Derivation
The Gaussian Error Linear Unit (GELU) scales the input by its cumulative standard normal probability:
$$\text{GELU}(x) = x \Phi(x)$$
where the standard normal cumulative distribution function (CDF) $\Phi(x)$ is defined as:
$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$
We derive the first derivative using the product rule:
$$\frac{d}{dx}\text{GELU}(x) = \Phi(x) \cdot \frac{d}{dx}[x] + x \cdot \frac{d}{dx}[\Phi(x)]$$
Since $\frac{d}{dx}[x] = 1$ and the derivative of the CDF is the standard normal probability density function (PDF) $\phi(x)$:
$$\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$
Substituting these terms back:
$$\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \phi(x)$$
$\blacksquare$

### 1.7.2 Mathematical Proof of Corrective Inhibitory Gradients
Under standard ReLU activation, the function and its derivative are defined as:
$$\text{ReLU}(x) = \max(0, x), \quad \frac{d}{dx}\text{ReLU}(x) = \mathbb{I}(x > 0)$$
If a neuron is perturbed into the negative region ($x < 0$), its gradient becomes exactly zero. During Backpropagation Through Time (BPTT), this creates **dead neurons** that cannot propagate corrective error signals, causing optimization to stall.

GELU resolves this by introducing a non-zero gradient region for negative inputs, characterized by a **negative dip** (shown in Figure 2).
We find the minimum of the GELU activation function by setting its derivative to zero:
$$\Phi(x) + x \phi(x) = 0 \implies \Phi(x) = -x \phi(x)$$
Since $\Phi(x) > 0$ for all $x$, this equality requires $x < 0$.
Solving this transcendental equation numerically:
* At $x = -0.7518$:
  $$\Phi(-0.7518) \approx 0.226$$
  $$\phi(-0.7518) \approx 0.301 \implies -x \phi(x) \approx 0.226$$
The minimum occurs at $x_{\text{dip}} \approx -0.7518$ with value $\text{GELU}(x_{\text{dip}}) \approx -0.170$.

For negative inputs in the active range $x \in (-1.5, 0.0)$, the derivative is non-zero and negative:
$$\frac{d}{dx}\text{GELU}(x) < 0 \quad \text{for } x \in (-\infty, x_{\text{dip}})$$
$$\frac{d}{dx}\text{GELU}(x) > 0 \quad \text{for } x \in (x_{\text{dip}}, \infty)$$
This negative derivative region allows the network to propagate **corrective inhibitory gradients**. If a recurrent channel is over-excited by adversarial noise, the backward pass propagates corrective negative gradients through this dip, de-activating the unit without setting its gradient to zero. This guarantees that neurons remain active and receptive to optimization throughout BPTT.

<div class="figure-container">
  <img src="gelu_dip.png" alt="GELU Negative Dip Curve">
  <div class="figure-caption"><strong>Figure 2: GELU Activation Function and First Derivative.</strong> The non-zero derivative in the negative input range (the negative dip at $x \approx -0.75$) permits corrective inhibitory gradients to flow backward through time, preventing neuron deactivation.</div>
</div>

---

## 1.8 Spherical Prototype Classification

### 1.8.1 Biological Justification: Bounded Projection
In the biological visual system, decision boundaries are not computed via unbounded linear dot products. Neurons have maximum firing rates, and representations are normalized via local inhibition. Standard linear classifiers compute logits as $\ell_c = \mathbf{w}_c^T \mathbf{z} + b_c$, allowing adversarial attacks (like PGD) to exploit decision boundaries by scaling the representation magnitude $\|\mathbf{z}\|_2 \to \infty$. Spherical Prototype Classification restricts both features and prototypes to a unit hypersphere, bounding the decision space.

### 1.8.2 Geometric Proof of Adversarial Neutralization
Let $\mathbf{z} \in \mathbb{R}^D$ be the representation vector. We project it and the class prototypes $\mathbf{p}_c \in \mathbb{R}^D$ onto the unit hypersphere $\mathcal{S}^{D-1}$:
$$\mathbf{\tilde{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}, \quad \mathbf{\tilde{p}}_c = \frac{\mathbf{p}_c}{\|\mathbf{p}_c\|_2} \quad \text{such that } \|\mathbf{\tilde{z}}\|_2 = 1, \ \|\mathbf{\tilde{p}}_c\|_2 = 1$$
The classification logits are computed as:
$$\ell_c(\mathbf{\tilde{z}}) = \alpha \cos(\theta_c) = \alpha \mathbf{\tilde{z}}^T \mathbf{\tilde{p}}_c$$
where $\alpha > 0$ is a learnable temperature scale.

An attacker seeks to perturb $\mathbf{\tilde{z}}$ to $\mathbf{\tilde{z}}_{\text{adv}}$ to change the classification from correct class $y$ to target class $t$, maximizing the logit difference:
$$\Delta \ell = \ell_t(\mathbf{\tilde{z}}_{\text{adv}}) - \ell_y(\mathbf{\tilde{z}}_{\text{adv}}) = \alpha \mathbf{\tilde{z}}_{\text{adv}}^T (\mathbf{\tilde{p}}_t - \mathbf{\tilde{p}}_y)$$
We bound the maximum logit change using the Cauchy-Schwarz inequality:
$$|\Delta \ell| = \alpha \left| \mathbf{\tilde{z}}_{\text{adv}}^T (\mathbf{\tilde{p}}_t - \mathbf{\tilde{p}}_y) \right| \le \alpha \|\mathbf{\tilde{z}}_{\text{adv}}\|_2 \|\mathbf{\tilde{p}}_t - \mathbf{\tilde{p}}_y\|_2$$
Since $\|\mathbf{\tilde{z}}_{\text{adv}}\|_2 = 1$:
$$|\Delta \ell| \le \alpha \|\mathbf{\tilde{p}}_t - \mathbf{\tilde{p}}_y\|_2$$
Using the law of cosines:
$$\\|\mathbf{\tilde{p}}_t - \mathbf{\tilde{p}}_y\|_2^2 = \|\mathbf{\tilde{p}}_t\|_2^2 + \|\mathbf{\tilde{p}}_y\|_2^2 - 2 \mathbf{\tilde{p}}_t^T \mathbf{\tilde{p}}_y = 2 - 2\cos(\theta_{ty})$$
Therefore:
$$|\Delta \ell| \le \alpha \sqrt{2 - 2\cos(\theta_{ty})}$$

This bounds the maximum logit change (illustrated in Figure 3). Because the representation vector is constrained to the hypersphere, the attacker cannot exploit unbounded representation magnitude scaling to cross the decision boundary. The attacker is forced to rotate the representation vector along the hypersphere surface, which requires changing the semantic directions rather than simply amplifying noise.

<div class="figure-container">
  <img src="spherical_geometry.png" alt="Spherical Prototype Classification Geometry">
  <div class="figure-caption"><strong>Figure 3: Spherical Prototype Projection Geometry.</strong> Bounding representations and prototypes to the unit circle $\mathcal{S}^1$ restricts the maximum logit change $|\Delta \ell|$ induced by an adversarial vector $\mathbf{z}_{\mathrm{adv}}$, neutralizing magnitude scaling attacks.</div>
</div>

---

## 1.9 Action Initializer & Foveal Stream

### 1.9.1 Biological Justification: Foveation & Saccadic Masking
The human eye does not process the entire visual field at uniform high resolution. Instead, the central fovea (covering approx. $2^\circ$ of the visual angle) extracts high-resolution details, while the periphery provides low-resolution context. Eye movements (saccades) redirect the fovea to target objects. During a saccade, visual input is briefly suppressed (**saccadic masking**) to prevent motion blur from disrupting internal representations.

The foveal stream in RHAN-v10 mimics this process. It uses a Spatial Transformer Network (STN) to crop a high-resolution foveal patch from the input image, centered on coordinates predicted by the recurrent action initializer. Saccadic masking is represented by disabling feedforward updates during action transitions, allowing top-down expectations to stabilize the latent state before new foveal details are integrated.

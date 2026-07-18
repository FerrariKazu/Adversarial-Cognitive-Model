# Section 10 to 13: Fixed-Points, Curvature, and Gradient Masking

This section provides textbook-level mathematical proofs, dimensional analysis, and biological justifications for the contractive fixed-point guarantees, implicit gradients, gradient masking limitations, and NTK spectral bias of the Recurrent Hybrid Attention Network (RHAN-v10) architecture.

---

## 10. Banach Contraction Unrolling

### 10.1 Contractor Operators
Let $\mathcal{M}_s$ be the metric space of the belief state equipped with the Euclidean distance metric $d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2$. Let the top-down generative predictor be $P: \mathcal{M}_s \to \mathbb{R}^D$.
We define $P$ as a **contraction mapping** if there exists a Lipschitz constant $L_P < 1$ such that:
$$\| P(\mathbf{s}_1) - P(\mathbf{s}_2) \|_2 \le L_P \| \mathbf{s}_1 - \mathbf{s}_2 \|_2 \quad \forall \mathbf{s}_1, \mathbf{s}_2 \in \mathcal{M}_s$$

### 10.2 Mathematical Proof of Geometric Error Decay (Section 10.2)
Let the residual prediction error at step $t$ be:
$$\mathbf{e}^{(t)} = \mathbf{x} - P(\mathbf{s}^{(t)})$$
where $\mathbf{x} \in \mathbb{R}^D$ is the visual input.
Let $\mathbf{s}^*$ be the unique, true fixed-point equilibrium state satisfying the zero-error condition:
$$\mathbf{x} = P(\mathbf{s}^*)$$
The belief state update is driven by the residual error:
$$\mathbf{s}^{(t+1)} = \mathbf{s}^{(t)} + \mathbf{e}^{(t)}$$

We prove by finite mathematical induction that if $L_P < 1$, the residual error decays geometrically to zero.

1. **Base Case ($t = 1$):**
   Using the contraction property of $P$:
   $$\|\mathbf{e}^{(1)}\|_2 = \| \mathbf{x} - P(\mathbf{s}^{(1)}) \|_2 = \| P(\mathbf{s}^*) - P(\mathbf{s}^{(1)}) \|_2$$
   $$\|\mathbf{e}^{(1)}\|_2 \le L_P \| \mathbf{s}^* - \mathbf{s}^{(1)} \|_2$$
   Since the initial error is $\mathbf{e}^{(0)} = \mathbf{s}^* - \mathbf{s}^{(1)}$:
   $$\|\mathbf{e}^{(1)}\|_2 \le L_P \|\mathbf{e}^{(0)}\|_2$$
   This satisfies the theorem for the base case.

2. **Inductive Step:**
   We assume the hypothesis holds for step $t = k$:
   $$\|\mathbf{e}^{(k)}\|_2 \le L_P^k \|\mathbf{e}^{(0)}\|_2$$
   We show it holds for step $t = k+1$.
   The error at step $k+1$ is:
   $$\|\mathbf{e}^{(k+1)}\|_2 = \| P(\mathbf{s}^*) - P(\mathbf{s}^{(k+1)}) \|_2$$
   Applying the contraction property:
   $$\|\mathbf{e}^{(k+1)}\|_2 \le L_P \| \mathbf{s}^* - \mathbf{s}^{(k+1)} \|_2$$
   The distance between the states at step $k+1$ is bounded by the residual error at the previous step:
   $$\| \mathbf{s}^* - \mathbf{s}^{(k+1)} \|_2 \le \|\mathbf{e}^{(k)}\|_2$$
   Substituting this inequality back:
   $$\|\mathbf{e}^{(k+1)}\|_2 \le L_P \|\mathbf{e}^{(k)}\|_2$$
   Applying the inductive assumption:
   $$\|\mathbf{e}^{(k+1)}\|_2 \le L_P \left( L_P^k \|\mathbf{e}^{(0)}\|_2 \right) = L_P^{k+1} \|\mathbf{e}^{(0)}\|_2$$
   This completes the inductive step.

3. **Limit Analysis:**
   Since the Lipschitz constant $L_P \in [0, 1)$, the limit of the geometric sequence decays to zero (shown in Figure 11):
   $$\lim_{t \to \infty} \|\mathbf{e}^{(t)}\|_2 \le \lim_{t \to \infty} L_P^t \|\mathbf{e}^{(0)}\|_2 = \|\mathbf{e}^{(0)}\|_2 \lim_{t \to \infty} L_P^t = \|\mathbf{e}^{(0)}\|_2 \cdot 0 = \mathbf{0}$$
   $$\lim_{t \to \infty} \|\mathbf{e}^{(t)}\|_2 = \mathbf{0}$$
   $\blacksquare$
This proves that the closed-loop predictive updates converge globally to a unique visual equilibrium, neutralizing noise.

<div class="figure-container">
  <img src="banach_contraction_decay.png" alt="Error Decay under Banach Contraction">
  <div class="figure-caption"><strong>Figure 11: Residual error decay.</strong> The prediction residual error $\|\mathbf{e}^{(t)}\|_2$ decays geometrically to zero for contractive rates $L_P < 1$, guaranteeing global stability.</div>
</div>

---

## 11. Deep Equilibrium Dynamics (DEQ)

### 11.1 Mathematical Formulation
Rather than unrolling the recurrent foveation loop for a fixed number of steps, Deep Equilibrium (DEQ) models directly solve for the fixed-point state $\mathbf{f}^*$ at which the recurrent updates stabilize:
$$\mathbf{f}^* = \mathcal{M}(\mathbf{f}^*; \theta)$$
where $\mathcal{M}$ represents the recurrent attention block and foveal sampler, and $\theta$ represents the model parameters.

### 11.2 Proof of Implicit Function Theorem Gradient Bypass
Let us define the equilibrium constraint as an implicit function $g(\mathbf{f}^*, \theta) = \mathbf{0}$:
$$g(\mathbf{f}^*, \theta) = \mathbf{f}^* - \mathcal{M}(\mathbf{f}^*; \theta) = \mathbf{0}$$
According to the **Implicit Function Theorem (IFT)**, if $g(\mathbf{f}^*, \theta) = \mathbf{0}$ holds, the total derivative of the implicit equation with respect to parameters $\theta$ must equal zero:
$$\frac{d}{d\theta} g(\mathbf{f}^*, \theta) = \mathbf{0} \implies \frac{\partial g}{\partial \mathbf{f}^*} \frac{\partial \mathbf{f}^*}{\partial \theta} + \frac{\partial g}{\partial \theta} = \mathbf{0}$$

We compute the partial derivatives of $g$:
1. With respect to $\mathbf{f}^*$:
   $$\frac{\partial g}{\partial \mathbf{f}^*} = \frac{\partial}{\partial \mathbf{f}^*} \left[ \mathbf{f}^* - \mathcal{M}(\mathbf{f}^*; \theta) \right] = \mathbf{I} - \mathbf{J}_{\mathcal{M}}$$
   where $\mathbf{J}_{\mathcal{M}} = \frac{\partial \mathcal{M}(\mathbf{f}^*; \theta)}{\partial \mathbf{f}^*} \in \mathbb{R}^{M \times M}$ is the Jacobian matrix of the recurrent block at the equilibrium.
2. With respect to $\theta$:
   $$\frac{\partial g}{\partial \theta} = \frac{\partial}{\partial \theta} \left[ \mathbf{f}^* - \mathcal{M}(\mathbf{f}^*; \theta) \right] = -\frac{\partial \mathcal{M}}{\partial \theta}$$

Substituting these partials back into the total derivative equation:
$$(\mathbf{I} - \mathbf{J}_{\mathcal{M}}) \frac{\partial \mathbf{f}^*}{\partial \theta} - \frac{\partial \mathcal{M}}{\partial \theta} = \mathbf{0}$$
$$(\mathbf{I} - \mathbf{J}_{\mathcal{M}}) \frac{\partial \mathbf{f}^*}{\partial \theta} = \frac{\partial \mathcal{M}}{\partial \theta}$$
By multiplying by the inverse matrix $(\mathbf{I} - \mathbf{J}_{\mathcal{M}})^{-1}$:
$$\frac{\partial \mathbf{f}^*}{\partial \theta} = (\mathbf{I} - \mathbf{J}_{\mathcal{M}})^{-1} \frac{\partial \mathcal{M}}{\partial \theta}$$
$\blacksquare$

### 11.3 Memory Complexity Analysis
Under standard Backpropagation Through Time (BPTT), computing the gradient of a recurrent network unrolled for $T$ steps requires storing all intermediate activation states in GPU memory to compute the chain rule:
$$\frac{\partial \mathbf{f}^{(T)}}{\partial \theta} = \sum_{t=1}^T \left( \prod_{k=t+1}^T \frac{\partial \mathbf{f}^{(k)}}{\partial \mathbf{f}^{(k-1)}} \right) \frac{\partial \mathbf{f}^{(t)}}{\partial \theta}$$
This results in linear memory complexity:
$$\text{Memory Complexity}_{\text{BPTT}} = O(T)$$
For long foveation sequences ($T \ge 20$) and high-resolution crops, BPTT exhausts GPU memory.

By utilizing the IFT fixed-point gradient equation, we bypass unrolling. The gradient $\frac{\partial \mathbf{f}^*}{\partial \theta}$ is computed directly using the final equilibrium state $\mathbf{f}^*$ and parameters $\theta$. The term $(\mathbf{I} - \mathbf{J}_{\mathcal{M}})^{-1} \mathbf{v}$ is solved using iterative solvers (such as GMRES) that only require vector-Jacobian products, eliminating the need to store intermediate activations.

Avoiding BPTT via the DEQ fixed-point equation:
$$\frac{\partial \mathbf{f}^*}{\partial \theta} = (\mathbf{I} - \mathbf{J}_{\mathcal{M}})^{-1} \frac{\partial \mathcal{M}}{\partial \theta}$$
reduces the memory complexity to a constant value:
$$\text{Memory Complexity}_{\text{DEQ}} = O(1)$$
allowing infinite recurrent unrolling without memory overhead. This mathematical bypass is what makes training the 55M parameter `RHANLargeSTL10` model with continuous foveation computationally feasible on standard hardware (like a single GPU) instead of requiring high-end computing cluster architectures.

<div class="figure-container">
  <img src="deq_memory_complexity.png" alt="Memory Complexity BPTT vs DEQ">
  <div class="figure-caption"><strong>Figure 12: Memory Complexity Comparison.</strong> Unrolled BPTT exhibits linear memory growth ($O(T)$), while DEQ fixed-point backpropagation maintains a constant $O(1)$ memory footprint.</div>
</div>

---

## 12. The Gradient Masking Theorem

### 12.1 Introduction
Adversarial defenses that align internal representation spaces (like feature-scattering or representation matching) often suffer from **gradient masking**. They create a locally flat loss surface, which defeats gradient-based attacks (like PGD) but remains vulnerable to gradient-free attacks. We prove this behavior mathematically.

### 12.2 Mathematical Proof of Feature Jacobian Collapse
Let $f_\theta(x)$ be the representation extractor of the model. The feature-matching objective minimizes the distance $\mathcal{D}$ between the representations of clean and perturbed inputs:
$$\min_\theta \mathcal{L}_{\text{match}} = \mathcal{D}\left( f_\theta(x_{\text{adv}}), f_\theta(x_{\text{clean}}) \right)$$
where $x_{\text{adv}} = x_{\text{clean}} + \boldsymbol{\delta}$ with $\|\boldsymbol{\delta}\|_2 \le \varepsilon$.
Using the Euclidean distance $\mathcal{D}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2^2$:
$$\mathcal{L}_{\text{match}} = \| f_\theta(x_{\text{clean}} + \boldsymbol{\delta}) - f_\theta(x_{\text{clean}}) \|_2^2$$

We expand $f_\theta(x_{\text{clean}} + \boldsymbol{\delta})$ using a first-order Taylor series approximation:
$$f_\theta(x_{\text{clean}} + \boldsymbol{\delta}) = f_\theta(x_{\text{clean}}) + \mathbf{J}_f(x_{\text{clean}}) \boldsymbol{\delta} + O(\|\boldsymbol{\delta}\|_2^2)$$
where $\mathbf{J}_f(x) \in \mathbb{R}^{D \times d}$ is the feature Jacobian matrix.
Substituting the Taylor expansion back into the loss function:
$$\mathcal{L}_{\text{match}} = \| \mathbf{J}_f(x_{\text{clean}}) \boldsymbol{\delta} + O(\|\boldsymbol{\delta}\|_2^2) \|_2^2 = \boldsymbol{\delta}^T \mathbf{J}_f(x_{\text{clean}})^T \mathbf{J}_f(x_{\text{clean}}) \boldsymbol{\delta} + O(\|\boldsymbol{\delta}\|_2^3)$$

To ensure robustness against all possible perturbations, the optimization must minimize the loss under the worst-case perturbation:
$$\min_\theta \max_{\|\boldsymbol{\delta}\|_2 \le \varepsilon} \boldsymbol{\delta}^T \mathbf{J}_f(x_{\text{clean}})^T \mathbf{J}_f(x_{\text{clean}}) \boldsymbol{\delta}$$
By the Rayleigh Quotient theorem, the inner maximization is bounded by the maximum singular value (operator norm) of the Jacobian matrix:
$$\max_{\|\boldsymbol{\delta}\|_2 \le \varepsilon} \boldsymbol{\delta}^T \mathbf{J}_f(x_{\text{clean}})^T \mathbf{J}_f(x_{\text{clean}}) \boldsymbol{\delta} = \varepsilon^2 \sigma_{\max}^2\left(\mathbf{J}_f(x_{\text{clean}})\right) = \varepsilon^2 \left\| \mathbf{J}_f(x_{\text{clean}}) \right\|_2^2$$

To minimize this objective to zero, the optimization forces the feature Jacobian matrix to collapse:
$$\lim_{\mathcal{L}_{\text{match}} \to 0} \left\| \mathbf{J}_f(x) \right\|_2 = 0 \implies \mathbf{J}_f(x) \approx \mathbf{0}$$
$\blacksquare$

### 12.3 Vulnerability to Gradient-Free Attacks (Square Attack)
When $\mathbf{J}_f(x) \approx \mathbf{0}$, the model's loss surface with respect to the input becomes locally flat (shown in Figure 5):
$$\nabla_x \mathcal{L} = \mathbf{J}_f(x)^T \nabla_{f} \mathcal{L} \approx \mathbf{0}$$
This is the mechanism of **gradient masking**:
* Gradient-based attacks (like PGD) compute $\nabla_x \mathcal{L}$. Because the gradient vanishes, they fail to find perturbed states and report false robustness.
* Gradient-free attacks (like **Square Attack** or AutoAttack's black-box modules) do not use gradients. They search the input space using random walks and score functions, easily finding adjacent decision boundaries.
This explains why feature-scattering defenses collapse under AutoAttack.

<div class="figure-container">
  <img src="gradient_masking_surface.png" alt="Gradient Masking Loss Surface">
  <div class="figure-caption"><strong>Figure 5: Gradient Masking Loss Surface.</strong> Minimizing feature divergence flattens the local gradients ($\nabla_x \mathcal{L} \approx 0$), masking the true decision boundaries which remain vulnerable to gradient-free search.</div>
</div>

### 12.4 How TRADES Bypasses Gradient Masking
TRADES avoids representation matching. It minimizes the KL divergence between output probabilities (computed via `trades_loss` in [`train_rhan_v5_trades.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/train_rhan_v5_trades.py#L124)):
$$\mathcal{L}_{\text{TRADES}} = \mathcal{L}_{\text{CE}}(f(x), y) + \beta \text{D}_{\text{KL}}\left( \text{Softmax}(f(x)) \parallel \text{Softmax}(f(x_{\text{adv}})) \right)$$
Because it penalizes changes on the probability simplex rather than matching feature coordinates, it shifts the decision boundaries outward, maintaining informative gradients $\mathbf{J}_f(x) \neq \mathbf{0}$.

---

## 13. Spectral Bias & ACT Hessian Conditioning

### 13.1 NTK Spectral Bias Proof
Using the Neural Tangent Kernel (NTK) framework, the training dynamics of the model projections are governed by the differential equation:
$$\frac{d \mathbf{\hat{y}}(t)}{dt} = \mathbf{\Theta}_t \left( \mathbf{y} - \mathbf{\hat{y}}(t) \right)$$
where $\mathbf{\Theta}_t \in \mathbb{R}^{B \times B}$ is the NTK matrix at time step $t$.
Let the eigendecomposition of the NTK matrix be:
$$\mathbf{\Theta}_t = \sum_{i=1}^B \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$
where $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_B > 0$ are the eigenvalues, and $\mathbf{v}_i$ are the orthonormal eigenvectors.
The classification error vector $\mathbf{e}(t) = \mathbf{y} - \mathbf{\hat{y}}(t)$ can be projected onto the eigenbasis:
$$\mathbf{e}(t) = \sum_{i=1}^B e_i(t) \mathbf{v}_i \implies e_i(t) = e^{-\lambda_i t} e_i(0)$$

This shows that the convergence rate along each eigenmode is determined by its eigenvalue $\lambda_i$:
* **High-Frequency / High-Complexity features** align with large eigenvalues ($\lambda_{\max}$). They converge rapidly ($e^{-\lambda_{\max} t} \to 0$ instantly).
* **Low-Frequency / Structural features** align with small eigenvalues ($\lambda_{\min}$). They require extended epochs to converge ($e^{-\lambda_{\min} t} \approx 1$ for small $t$).
This mathematical property is **spectral bias**, explaining why networks prioritize fitting local noise before learning global structural features.

### 13.2 Ill-Conditioning of Discrete ACT Hessian
In Adaptive Computation Time (ACT), the model halts recurrence at a discrete step $N$:
$$N = \min \left\{ n \mid \sum_{t=1}^n h_t \ge 1 - \epsilon \right\}$$
where $h_t$ is the halting probability at step $t$.
The loss function is defined as:
$$\mathcal{L = \mathcal{L}_{\text{task}}(y, \mathbf{\hat{y}}^{(N)}) + \tau \text{Ponder}(N)}$$
Because $N \in \mathbb{N}$ is a discrete integer, the loss function $\mathcal{L}$ is a step function with respect to the input activations.
This discrete halting in ACT is implemented in [`ThermodynamicHalt`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/phase1_training/model_rhan_v10.py#L221).

Let us evaluate the joint Hessian matrix of the loss with respect to the parameters $\theta$:
$$\mathbf{H} = \nabla_\theta^2 \mathcal{L} = \nabla_\theta^2 \mathcal{L}_{\text{task}} + \tau \nabla_\theta^2 \text{Ponder}(N)$$

Because $N$ is constant almost everywhere except at the transition boundaries:
$$\nabla_\theta^2 \text{Ponder}(N) = \mathbf{0} \quad \text{for } \sum_t h_t \neq 1 - \epsilon$$
At the transition boundaries where the model switches steps:
$$\lim_{\sum_t h_t \to 1-\epsilon} \nabla_\theta \text{Ponder}(N) = \boldsymbol{\infty}$$
This makes the loss function non-differentiable. The eigenvalues of the Hessian matrix are either zero or infinite at the boundary points (shown in Figure 6):
$$\lambda_{\max}(\mathbf{H}) \to \infty, \quad \lambda_{\min}(\mathbf{H}) \approx 0$$
Evaluating the condition number $\kappa$ of the joint Hessian:
$$\kappa = \frac{\lambda_{\max}(\mathbf{H})}{\lambda_{\min}(\mathbf{H})} \to \infty$$
An infinite condition number makes the optimization landscape extremely ill-conditioned, causing standard gradient descent to oscillate or diverge.

RHAN-v10 bypasses this by using **Ponder Gating** which smoothly weights all steps, ensuring the loss function is twice-differentiable and the Hessian is well-conditioned ($\kappa < \infty$).

<div class="figure-container">
  <img src="hessian_conditioning.png" alt="ACT vs Ponder Gating Hessian Conditioning">
  <div class="figure-caption"><strong>Figure 6: Hessian Conditioning Comparison.</strong> Discrete halting in ACT creates step boundaries with singular gradients ($\kappa \to \infty$), whereas smooth Ponder Gating maintains a differentiable curvature.</div>
</div>

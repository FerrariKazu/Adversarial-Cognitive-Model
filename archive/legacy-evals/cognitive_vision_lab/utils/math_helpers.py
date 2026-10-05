"""Curated mathematical definitions shown in expandable educational sections."""
from __future__ import annotations

# Each entry: {key: (title, latex)}
EQUATIONS: dict[str, tuple[str, str]] = {
    "softmax": (
        "Softmax",
        r"p_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}",
    ),
    "cross_entropy": (
        "Cross-Entropy Loss",
        r"\mathcal{L}_{\mathrm{CE}} = -\sum_{i=1}^{K} y_i \log p_i",
    ),
    "fgsm": (
        "Fast Gradient Sign Method (FGSM)",
        r"x_{\mathrm{adv}} = x + \varepsilon \cdot \operatorname{sign}\big(\nabla_x \mathcal{L}(x, y)\big)",
    ),
    "pgd": (
        "Projected Gradient Descent (PGD)",
        r"x^{t+1} = \Pi_{\mathcal{B}_\varepsilon(x)} \big(x^t + \alpha \cdot \operatorname{sign}\big(\nabla_x \mathcal{L}(x^t, y)\big)\big)",
    ),
    "trades": (
        "TRADES Objective",
        r"\min_f \; \mathbb{E}\big[\mathcal{L}(f(x), y) + \beta \cdot \max_{x' \in \mathcal{B}_\varepsilon(x)} \mathrm{KL}(f(x')\,\|\,f(x))\big]",
    ),
    "kl_div": (
        "KL Divergence",
        r"\mathrm{KL}(P\,\|\,Q) = \sum_i P_i \log \frac{P_i}{Q_i}",
    ),
    "dprime": (
        "Sensitivity Index d′",
        r"d' = \Phi^{-1}(\mathrm{HR}) - \Phi^{-1}(\mathrm{FAR})",
    ),
    "cosine": (
        "Cosine Similarity",
        r"\cos(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}",
    ),
    "attention": (
        "Scaled Dot-Product Attention",
        r"\mathrm{Attn}(Q, K, V) = \operatorname{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big) V",
    ),
    "gradcam": (
        "GradCAM",
        r"\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}, \quad L^c = \mathrm{ReLU}\Big(\sum_k \alpha_k^c A^k\Big)",
    ),
    "pca": (
        "Principal Component Analysis",
        r"\Sigma v = \lambda v, \quad X' = X W \text{ with } W = [v_1, \dots, v_d]",
    ),
    "tsne": (
        "t-SNE",
        r"p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k\neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k\neq l}(1 + \|y_k - y_l\|^2)^{-1}}",
    ),
    "umap": (
        "UMAP",
        r"\min_{Y} \sum_{i,j} w_{ij} \log \frac{w_{ij}}{w_{ij}(Y)}",
    ),
    "ssim": (
        "Structural Similarity",
        r"\mathrm{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}",
    ),
    "psnr": (
        "Peak Signal-to-Noise Ratio",
        r"\mathrm{PSNR} = 10 \log_{10} \frac{\mathrm{MAX}^2}{\mathrm{MSE}}",
    ),
    "cw": (
        "Carlini-Wagner (L2)",
        r"\min_\delta \; \|\delta\|_2^2 + c \cdot \max\big(f(x + \delta) - f_{t}(x + \delta), -\kappa\big)",
    ),
    "deepfool": (
        "DeepFool",
        r"\delta = \min_i \frac{|f(x) - f_i(x)|}{\|\nabla f(x) - \nabla f_i(x)\|_2^2} \; (\nabla f(x) - \nabla f_i(x))",
    ),
    "square": (
        "Square Attack",
        r"x^{t+1} = \Pi_{\mathcal{B}_\varepsilon} \big(x^t + \mathbf{1}_S \cdot \xi\big), \quad S \sim \text{random square region}",
    ),
    "fab": (
        "Fast Adaptive Boundary (FAB)",
        r"x^{t+1} = (1 - \eta) \Pi_{\mathcal{B}_\varepsilon}(x^t) + \eta \cdot \Pi_{\text{boundary}}(x^t)",
    ),
    "apgd": (
        "Auto-PGD",
        r"\alpha^{t+1} = \begin{cases} \eta \alpha^t & \text{if } \sum_{k} \text{no progress} \\ \alpha^t & \text{otherwise} \end{cases}",
    ),
}

ATTACK_NOTES: dict[str, str] = {
    "fgsm": "One-step gradient sign attack. Fast, weak; the theoretical basis of PGD.",
    "pgd": "Iterative FGSM with projection. Standard strong white-box attack; TRADES uses it for training.",
    "cw": "Optimization-based attack that minimizes perturbation while forcing misclassification.",
    "deepfool": "Geometric attack crossing the decision boundary with minimal L2 perturbation.",
    "square": "Query-based attack; strong estimator of *black-box* robustness.",
    "fab": "Fast adaptive boundary attack used inside AutoAttack's ensemble.",
    "apgd": "PGD with automatic step-size decay; the white-box core of AutoAttack.",
}

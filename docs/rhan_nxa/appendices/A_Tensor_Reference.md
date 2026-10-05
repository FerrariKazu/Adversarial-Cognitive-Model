# Appendix A — Comprehensive Tensor Shape and Interface Reference

> *Authoritative tabular reference of all tensor shapes, dimensions, dtypes, and value domains across the RHAN-NXA architecture.*

---

## 1. Global Dimensions and Constants

| Symbol | Name | Canonical Value | Status | Defined In |
|---|---|---|---|---|
| $B$ | Batch size | Dynamic (e.g. 64, 128) | Runtime | DataLoader |
| $H, W$ | Full image spatial resolution | $224, 224$ | Locked | ImageNet-100 standard |
| $C$ | Color channels | $3$ | Locked | RGB format |
| $D_z$ | Latent embedding dimension | $384$ | Locked | `backbone.py` |
| $P$ | ViT patch spatial size | $14 \times 14$ | Locked | `backbone.py` |
| $H_g, W_g$ | Foveal glimpse resolution | $56, 56$ | Locked | `foveation.py` |
| $N_g$ | Patches per glimpse | $16$ ($4 \times 4$) | Locked | $(56/14) \times (56/14)$ |
| $T$ | Total glimpse steps per image | $4$ | Locked | `MASTER_PLAN.md` |
| $K_{\text{ais}}$ | Candidate gaze fixations | $4$ to $8$ (default 4) | Locked | `ais_v2_policy.py` |
| $C_{\text{cls}}$ | Classification categories | $100$ | Locked | ImageNet-100 |
| $\text{iter}_{\text{in}}$ | Within-glimpse token iterations | $2$ or $3$ (default 2) | Locked | `recurrent_block.py` |

---

## 2. Component Tensor Shape Table

| Component | Tensor Name | Shape | Dtype | Value Range / Constraints |
|---|---|---|---|---|
| **Input Image** | $x$ | `(B, 3, 224, 224)` | `float32` | Normalized ImageNet range |
| **Gaze Action** | $a_t$ | `(B, 2)` | `float32` | $(y, x) \in [-1, 1]^2$ |
| **Foveal Glimpse** | $g_t$ | `(B, 3, 56, 56)` | `float32` | Normalized ImageNet range |
| **Patch Embeddings** | $E_{\text{patches}}$ | `(B, 16, 384)` | `float32` | Unbounded real |
| **Refined Tokens** | $H_{\text{refined}}$ | `(B, 16, 384)` | `float32` | Post LayerNorm |
| **Pooled State** | $z_t^{\text{raw}}$ | `(B, 384)` | `float32` | Mean pooled over patch dimension |
| **Belief State** | $z_t$ | `(B, 384)` | `float32` | Global perceptual vector |
| **Belief State** | $S_t$ | `None` | N/A | Gen-1 Core default ($S_t = \text{None}$) |
| **Belief State** | $U_t$ | `(B,)` | `float32` | $U_t \in (0, 1]$, Clamped $[10^{-6}, 10^4]$ |
| **Dirichlet Evidence** | $e$ | `(B, 100)` | `float32` | $e_k \ge 0$ via Softplus |
| **Dirichlet Alpha** | $\alpha$ | `(B, 100)` | `float32` | $\alpha_k = e_k + 1 \ge 1.0$ |
| **Predicted Glimpse** | $\hat{g}_t$ | `(B, 16, 384)` | `float32` | Output of shared predictor |
| **Prediction Error** | $E_t$ | `(B, 16, 384)` | `float32` | $E_t = \Vert g_t^{\text{obs}} - \hat{g}_t^{\text{pred}} \Vert_2$; $E_0 := 0$ |
| **Precision** | $\Pi_t$ | `(B, 1)` or `(B, 384)` | `float32` | $\Pi_t \ge 10^{-4}$ (clamped floor) |
| **UpdateNet Delta** | $\Delta z_t$ | `(B, 384)` | `float32` | $\lvert \Delta z_t \rvert \le 0.1$ via $\tanh$ scaling |
| **AIS Candidates** | $a_{\text{cand}}$ | `(B, K, 2)` | `float32` | Candidate coordinates $\in [-1, 1]^2$ |
| **Candidate Scores** | $s_{\text{cand}}$ | `(B, K)` | `float32` | Information gain / entropy reduction |
| **Gaze Probabilities** | $p_{\text{gaze}}$ | `(B, K)` | `float32` | $\sum_{k=1}^K p_k = 1.0$ (Softmax) |
| **Trajectory Drift** | `drift` | `(B,)` | `float32` | Non-negative real |

---

## 3. Boundary Conditions and Interface Invariants

1. **$t = 0$ Initialization**:
   - $z_0 \leftarrow \mathbf{0} \in \mathbb{R}^{B \times 384}$ (or learned zero prior).
   - $S_0 \leftarrow \text{None}$.
   - $U_0 \leftarrow 1.0$ (maximal uncertainty prior).
   - $E_0 \leftarrow \mathbf{0}$ (identity prediction error).
   - $a_1 \leftarrow (0.0, 0.0)$ (center foveal glimpse).
2. **UpdateNet Invariant**:
   $$\|\Delta z_t\|_{\infty} \le 0.1$$
3. **Precision Weighting Invariant**:
   $$\Pi_t \ge 10^{-4}$$
4. **Dirichlet Evidence Invariant**:
   $$10^{-6} \le e_k \le 10^4 \quad \implies \quad 1 + 10^{-6} \le \alpha_k \le 10^4 + 1$$

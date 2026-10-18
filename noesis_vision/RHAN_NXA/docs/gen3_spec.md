# RHAN-NXA Gen-3 Architectural Specification
## High-Resolution Active Evidence Acquisition on ff_112 Foundation

**Status:** Draft Architecture Specification  
**Foundation Base:** `ff_112` (DINOv2-S, 112×112 px, 64 patch tokens, corrected TRADES-PGD4)  
**Target Domain:** High-Resolution Visual Reasoning (384–512 px scenes with localized targets)  
**Reference Design:** `plan.md` (§6, §7, §8), `plan2.md` (§7 A2-RHAN-HR, §8), and Revision 1 forensic audit (§14, §17)

---

## 1. Scientific Motivation & The Resolution Ceiling

### 1.1 The Finding from Gen-2
In Gen-2, moving from 56 px (16 patch tokens) to 112 px (64 patch tokens) via `ff_112` yielded a **+27.08 pp** jump on linear probe accuracy and improved robust accuracy to 39.88% at 24% faster latency than the 56 px recurrent loop. However, scaling further from 112 px to 126 px yielded only **+0.90 pp**.

Because ImageNet-100 source images are stored at 160 px, a 112–126 px feedforward pass already consumes nearly all available high-frequency information in the dataset. On such low-resolution inputs:
1. Glimpses cannot add new detail beyond the downsampled gist (the audit verified that on 56 px, glimpses were bit-identical affine crops of the same tensor).
2. The recurrent loop cannot outperform a feedforward pass at matched compute, because one full-image pass already sees the entire target.

### 1.2 The Gen-3 Thesis
The central claim of active perception is:
> *An active observer that maintains an internal belief, predicts its sensory inputs, and strategically directs its gaze acquires evidence more effectively and robustly under a constrained compute budget than a passive feedforward pass.*

This hypothesis is **untestable** on ImageNet-100. It is only falsifiable in a regime where:
- The scene is large (384–512 px) and cluttered.
- Key discriminative targets are small (e.g. 56×56 to 112×112 px) and located at unknown positions.
- A single full-resolution feedforward pass across the whole scene is compute-prohibitive or under-resolved.
- The observer must actively look, zoom, integrate evidence, and inhibit re-fixation.

---

## 2. High-Resolution Substrate Architecture (G3-HRV1)

```
                       Full High-Resolution Source Image
                                 (384–512 px)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     Global Downscale (112 px)                     Native Crop (scale 0.15–0.25)
              │                                               │
     Shared DINOv2-S Trunk                         Shared DINOv2-S Trunk
       (64 patch tokens)                             (16 patch tokens)
              │                                               │
              ▼                                               ▼
       Global Gist z₀                               Raw Glimpse Tokens Tok_t
       (Prior State)                                          │
              │                                               │
              │         ┌─────────────────────────────────────┘
              ▼         ▼
        ┌────────────────────────────────────────────────────────┐
        │       RHAN-NXA Gen-3 Recurrent Belief Core             │
        │                                                        │
        │  1. Belief-Conditioned Read:   r_t = AttnPool(x_{t-1}, Tok_t)   │
        │  2. Top-Down Prediction:      ô_t = P_ψ(x_{t-1}, ℓ_t)  │
        │  3. Prediction Error:         e_t = ô_t - r_t          │
        │  4. Bounded Update (GRU):     x_t = GRU([r_t; e_t], x_{t-1})  │
        │  5. Memory Update (M2/M3):    Buffer & Spatial Bank    │
        │  6. Inhibition of Return:     Downweight visited ℓ     │
        └────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
      Action Policy π(ℓ_{t+1})               Final Classifier Head
      (Chooses Next Gaze)                    logits = C([x_T ; z₀])
```

### 2.1 Dual-Scale Observation Pipeline
To avoid tokenizer confounds, both pathways share the **exact same frozen/finetuned DINOv2-S trunk** (14×14 patch size, $d=384$):
1. **Global Gist ($z_0$):**
   - Input: Source downscaled to $112 \times 112$ px.
   - Patch grid: $8 \times 8 = 64$ patch tokens.
   - Processed once at $t=0$ through the DINOv2-S trunk.
   - Yields persistent context prior $z_0 = \text{CLS}(\text{trunk}(I_{\text{gist}})) \in \mathbb{R}^{384}$.
2. **Native-Resolution Glimpses ($o_t$):**
   - Input: $56 \times 56$ crop extracted directly from the native 384–512 px source image at normalized location $\ell_t = (x_t, y_t)$.
   - Effective zoom: $4\times$ to $8\times$ relative to the global view. Glimpse carries high-frequency details that do not exist in the 112 px gist.
   - Patch grid: $4 \times 4 = 16$ patch tokens $\text{Tok}_t \in \mathbb{R}^{16 \times 384}$.

---

## 3. The Matched-Token Feedforward Baseline (The Null Hypothesis)

Every earlier comparison in the project suffered from compute or parameter mismatch. Gen-3 establishes a strict **matched-token baseline rule**:

| Model | Observations | Total Tokens Processed | Approximate Relative FLOPs |
|---|---|---|---|
| **Gen-3 Loop ($T=4$)** | 1 Gist (112 px) + 4 Glimpses (56 px) | 64 (gist) + 4 × 16 (glimpses) = **128 tokens** | $1.0\times$ (reference) |
| **Matched Feedforward (FF-160)** | 1 Full View (160 px, ~11×11 grid) | **121 tokens** | $0.95\times$ |
| **Matched Feedforward (FF-182)** | 1 Full View (182 px, 13×13 grid) | **169 tokens** | $1.32\times$ |

**Gate B (Budget) Requirement:**
The active loop must outperform the matched-token feedforward baseline at matched FLOPs by **$> 2\sigma$** on robust accuracy under corrected attacks. If feedforward matches or beats the loop at equal tokens, the recurrent loop is rejected.

---

## 4. Memory Architecture: M0 to M3

Following `plan.md` §7, Gen-3 builds out levels M0 through M3 with clean isolation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Memory Hierarchy in Gen-3                       │
├────┬─────────────────┬─────────────────────────────────────────────────┤
│ M0 │ Static Weights  │ DINOv2-S trunk θ, UpdateNet, Policy, Readout    │
├────┼─────────────────┼─────────────────────────────────────────────────┤
│ M1 │ Working State   │ Dynamic belief state x_t ∈ ℝ³⁸⁴ (bounded GRU)   │
├────┼─────────────────┼─────────────────────────────────────────────────┤
│ M2 │ Glimpse Buffer  │ Bounded FIFO buffer of last T glimpses          │
│    │                 │ {(Tok_k, ℓ_k, e_k)}_{k=1}^t, cross-attended     │
├────┼─────────────────┼─────────────────────────────────────────────────┤
│ M3 │ Spatial Bank &  │ Coarse spatial grid (e.g. 6×6 or 8×8) tracking: │
│    │ Inhibition of   │ • Fixation count & visit recency                │
│    │ Return (IoR)    │ • Cumulative evidence / surprise per region     │
│    │                 │ • Suppresses policy logits for re-visits        │
└────┴─────────────────┴─────────────────────────────────────────────────┘
```

### 4.1 Bounded State Update (M1)
Replaces the unbounded precision multiplier ($\gamma \approx 7$) of Gen-1/Gen-2 with a standard, bounded GRU cell:
$$\begin{aligned}
r_t &= \text{AttnPool}(W_q x_{t-1}, W_k \text{Tok}_t, W_v \text{Tok}_t) \in \mathbb{R}^{384} \\
\hat{o}_t &= P_\psi(x_{t-1}, \text{emb}(\ell_t)) \in \mathbb{R}^{384} \\
e_t &= \hat{o}_t - r_t \\
u_t &= \sigma(W_u [r_t ; e_t ; x_{t-1}] + b_u) \quad \text{(write gate } \in (0,1)\text{)} \\
\rho_t &= \sigma(W_\rho [r_t ; e_t ; x_{t-1}] + b_\rho) \quad \text{(reset gate } \in (0,1)\text{)} \\
\hat{h}_t &= \tanh(W_h [r_t ; \rho_t \odot e_t ; x_{t-1}] + b_h) \quad \text{(candidate } \in (-1, 1)\text{)} \\
x_t &= (1 - u_t) \odot x_{t-1} + u_t \odot \hat{h}_t
\end{aligned}$$

**Key properties:**
- Bounded step change: $\|x_t - x_{t-1}\| \le 2 u_t$.
- Identity-safe initialization: $b_u \approx -2.0 \implies u_0 \approx 0.1$, preventing catastrophic overwrite on step 1.

### 4.2 Inhibition of Return (M3)
Inhibition of Return prevents the gaze policy from getting stuck re-fixating the same high-saliency patch.
- Let $M \in \mathbb{R}^{K \times K}$ be a coarse grid over the scene ($K=6$).
- When location $\ell_t$ is sampled, a Gaussian splat updates the visitation map:
  $$M_{i,j}^{(t)} = \lambda M_{i,j}^{(t-1)} + \exp\left(-\frac{\|\ell_t - c_{i,j}\|^2}{2\sigma_{\text{ior}}^2}\right)$$
- The policy $\pi(\ell_{t+1} \mid x_t)$ computes candidate scores $s_k$, which are modulated by the visitation penalty:
  $$\tilde{s}_k = s_k - \beta_{\text{ior}} M(\ell_k)$$

---

## 5. Verification Protocol & Decision Gates

All claims must be verified under the corrected evaluation protocol established in revision 1:

1. **Gate B (Budget Frontier):**
   - Accuracy vs. tokens curve evaluated for:
     (a) Adaptive Gaze Loop
     (b) Fixed-Grid Gaze Loop
     (c) Random Gaze Loop
     (d) Matched-Token Feedforward Baseline
   - *Pass:* (a) beats (d) by $> 2\sigma$ at equal tokens, and beats (b) and (c) in paired McNemar tests.

2. **Gate S (Search & Evidence Alignment):**
   - Evaluated on cluttered 384–512 px scenes with known bounding boxes for small targets.
   - *Metrics:* Fixation hit-rate within target bounding box vs random gaze baseline; task accuracy vs glimpse count $T$.
   - *Pass:* Gaze fixations fall inside target ground-truth bounding boxes significantly above chance ($p < 0.001$), and accuracy increases monotonically with $T$.

3. **Gate A (Adversarial Robustness):**
   - Matched-$\epsilon$ testing ($\epsilon_{\text{norm}} = 0.094 \approx 5.4/255$).
   - Headline robust accuracy = per-sample worst-case across PGD-50, APGD-CE-100, and APGD-T-100 in fp32.
   - Large-$\epsilon$ sanity checks at $16/255$ and $32/255$ to guarantee absence of gradient masking.

---

## 6. Implementation Stages & GPU Budget

| Stage | Arm Name | Description | Compute (T4 GPU) | Status |
|---|---|---|---|---|
| **J2-A** | `eval_matched_eps` | Re-eval all 4 Gen-2 checkpoints at 5.4/255 (shared 1k slice, fp32, McNemar) | ~2 GPU-h | **Ready in Colab J2** |
| **J2-B** | `train_ff_56` | 56 px feedforward control under corrected TRADES-PGD4 recipe | ~1.5 GPU-h | **Ready in Colab J2** |
| **J2-C** | `train_ff_112_seeds` | Two additional seeds (42 and 99) for ff_112 to set $2\sigma$ variance bar | ~9.4 GPU-h | **Ready in Colab J2** |
| **G3-1** | `synth_search_data` | Synthetic cluttered dataset generator (384–512 px with small cued targets) | CPU-only | Specification drafted |
| **G3-2** | `ff_matched_baseline`| Matched-token feedforward baseline on 384–512 px substrate | ~6 GPU-h | Specification drafted |
| **G3-3** | `rhan_g3_core` | Gen-3 loop with M2/M3 memory and bounded GRU update on high-res substrate | ~12 GPU-h | Specification drafted |

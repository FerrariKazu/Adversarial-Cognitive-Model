# Chapter 10 — Differentiable Foveation: Spatial Bounding and Affine Sampling

> *Level 2 reading. Spatial Transformer Network mechanics: extracting high-acuity 56×56 crops while preserving end-to-end motor gradients.*

---

## 1. In One Sentence

Differentiable foveation is the mechanism by which RHAN-NXA extracts a $56 \times 56$ high-acuity crop from any location in the full input image using a Spatial Transformer Network (STN) affine grid, such that gradients from downstream objectives flow back through the fixation coordinates, enabling the gaze policy to be trained end-to-end.

---

## 2. Intuition

A biological fovea physically saccades to different locations in the visual field; RHAN-NXA simulates this computationally. Given a 2D gaze coordinate $a_t \in [-1, +1]^2$ (normalized, origin at image center), the foveation module constructs an affine transformation matrix $\Theta_t$ that, when applied via `torch.nn.functional.grid_sample`, produces a $56 \times 56$ bilinear-interpolated crop centered at $a_t$.

The critical property is **differentiability**: because $\Theta_t$ is constructed from the gaze coordinates via tensor operations (not Python floats), the autograd graph preserves the connection between the output crop and the input gaze coordinates. The motor Jacobian $\frac{\partial g_t}{\partial a_t}$ flows back through the crop into the gaze policy parameters.

---

## 3. Differentiable Foveation Architecture

![Figure 14. Differentiable Spatial Transformer Foveation. Given continuous fixation coordinates a_t in [-1, 1]^2, the module constructs an affine transformation matrix with fixed scale s = 56/224 = 0.25, generating a sampling grid via affine_grid and extracting a high-acuity 56×56 patch via bilinear interpolation. Gradients flow back through the motor Jacobian to train the gaze policy.](figures/foveation.svg)

---

## 4. Mathematical Formulation

The affine transformation for gaze $(x_g, y_g) \in [-1, +1]^2$ with scale $s = \text{fovea\_size} / H$:

$$\Theta_t = \begin{pmatrix} s & 0 & x_g \\ 0 & s & y_g \end{pmatrix} \in \mathbb{R}^{2 \times 3}$$

The output crop at pixel $(i, j)$ samples from the source image at the bilinear-interpolated location:

$$\text{crop}_{i,j} = \text{BilinearInterp}\left(x,\ s \cdot i_\text{norm} + x_g,\ s \cdot j_\text{norm} + y_g\right)$$

where $i_\text{norm}, j_\text{norm}$ are the normalized output pixel coordinates. Border padding is used for boundary coordinates.

---

## 5. Implementation Mapping

- **Function**: `noesis_vision.models.foveation.foveal_sample(x_image, gaze_coords, fovea_size=56)`
- **Framework**: PyTorch's `torch.nn.functional.affine_grid` + `torch.nn.functional.grid_sample`.
- **Sampling mode**: bilinear, border padding, `align_corners=False`.

### Differentiable Theta Construction (Source Extract)

```python
scale_col = torch.full((B, 1), scale, dtype=gaze_coords.dtype, device=gaze_coords.device)
zero_col = torch.zeros((B, 1), dtype=gaze_coords.dtype, device=gaze_coords.device)
row0 = torch.cat([scale_col, zero_col, gaze_coords[:, 0:1]], dim=1)
row1 = torch.cat([zero_col, scale_col, gaze_coords[:, 1:2]], dim=1)
theta = torch.stack([row0, row1], dim=1)  # (B, 2, 3)
```

The autograd graph is strictly preserved through `torch.cat` and `torch.stack`.

---

## 6. Key Adaptations from Generation-0

| Property | Gen-0 (STL-10) | RHAN-NXA Gen-1 | Scientific Rationale |
|:---|:---|:---|:---|
| Input resolution | Hardcoded 96 px | Parameterized $H = 224$ px | Scales to ImageNet-100 standard |
| Fovea size | 48 px | **56 px** | Clean patch grid ($56 = 4 \times 14$) |
| Patch divisibility | $48 / 14 = 3.43$ (fractional) | $56 / 14 = \mathbf{4.0}$ **(integer)** | Required by ViT patch embedder |
| Sampling conventions | bilinear, border, `align_corners=False` | **Unchanged** | Validated STN conventions |

---

## 7. Authoritative Tensor Specification

| Tensor | Shape | Gradient? | Description |
|:---|:---|:---:|:---|
| `x_image` | `(B, 3, 224, 224)` | Optional | Full visual scene batch |
| `gaze_coords` | `(B, 2)` | **Yes** | Continuous fixation coordinates in $[-1, 1]^2$ |
| `theta` | `(B, 2, 3)` | **Yes** | Affine transformation matrix parameter tensor |
| Output crop $g_t$ | `(B, 3, 56, 56)` | **Yes** | Extracted foveal glimpse patch |

---

## 8. Validation Rules and Boundary Asserts

`foveal_sample` enforces:

- Non-square input (`H ≠ W`) raises `ValueError`: foveation assumes a square frame.
- Gaze coordinates outside $[-1, +1]$ (tolerance $10^{-6}$) raise `ValueError`: *"a convention mismatch here corrupts every downstream shape."*
- `fovea_size > H` raises `ValueError`: crop cannot exceed image frame.

---

## 9. Scientific Status

- **Differentiable foveation mechanism**: **LOCKED** (ADAPT disposition from validated Gen-0 source).
- **Fovea size $56 \times 56$**: **LOCKED** (required for DINOv2-small patch divisibility).
- **Gaze coordinate convention $[-1, +1]$**: **LOCKED**.

---

## 10. Related Components and System Cross-References

- Foveation implementation: [noesis_vision/models/foveation.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/foveation.py)
- Substrate caller: [noesis_vision/models/backbone.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py)
- Figure Reference: [foveation.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/foveation.svg).

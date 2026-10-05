# Chapter 13 — System Architecture: CompactViT and Parameter Budget

## 1. In one sentence
RHAN-NXA's visual substrate is `CompactViT`: a DINOv2-small-shaped trunk ($d_z = 384$, 12 blocks, patch size 14) plus a Universal-Transformer-style tied refinement block, totaling $\sim 23.3\text{M}$ parameters, within the locked 20–25M band, with a clean DINOv2 warm-start mapping.

## 2. Architecture overview

```
Input: (B, 3, H, H) full image
          │
          ▼
  foveal_sample(gaze_coords)    ← Differentiable STN crop (56×56)
          │
          ▼
  PatchEmbed Conv2d(3, 384, 14, 14)   → (B, N, 384)   [N = 16 patches]
          │
  [CLS token] [Register token] [Pos embed]
          │
  ┌─────────────────────────┐
  │   12 × TransformerBlock │  ← DINOv2-small-shaped trunk
  │   (width=384, heads=6)  │  ← Warm-started from DINOv2-small
  └─────────────────────────┘
          │
  LayerNorm (final)
          │
  ┌─────────────────────────┐
  │  TiedRecurrence:        │  ← 1 TransformerBlock, run 2–3 times
  │  2–3 × same block       │  ← New-by-design (no warm-start)
  └─────────────────────────┘
          │
     Pool (CLS position)         → pooled_z: (B, 384)
     Take patch tokens           → tokens:   (B, 16, 384)
```

## 3. Parameter budget

| Component | Parameters | Notes |
|:---|:---:|:---|
| 12 trunk blocks (attn + MLP, width 384) | ~21.29M | DINOv2-small-shaped |
| Patch embed (`Conv2d`) | ~0.23M | 14×14 stride |
| Pos/CLS/register tokens | ~0.01M | Small |
| Final LayerNorm | ~0.001M | |
| Tied refinement block (1 block × mlp_ratio 4) | ~1.77M | New-by-design |
| **Total** | **~23.3M** | In [20M, 25M] band |

The parameter count **does not change with the within-glimpse iteration count** because the refinement block is tied (one set of parameters, applied 2–3 times).

Test: `tests/test_backbone_param_count` asserts `total_params ∈ [20_000_000, 25_000_000]`.

## 4. Warm-start mapping
`CompactViT.load_dino_warm_start(path)` maps DINOv2-small checkpoint keys 1:1 to trunk module keys:

| DINOv2 key pattern | CompactViT key |
|:---|:---|
| `patch_embed.proj.*` | `patch_embed.proj.*` |
| `blocks.{n}.norm1.*` | `blocks.{n}.norm1.*` |
| `blocks.{n}.attn.qkv.*` | `blocks.{n}.attn["qkv"].*` |
| `blocks.{n}.attn.proj.*` | `blocks.{n}.attn["proj"].*` |
| `blocks.{n}.norm2.*` | `blocks.{n}.norm2.*` |
| `blocks.{n}.mlp.fc1.*` | `blocks.{n}.mlp["fc1"].*` |
| `blocks.{n}.mlp.fc2.*` | `blocks.{n}.mlp["fc2"].*` |
| `norm.*` | `norm.*` |

**New-by-design keys** (excluded from match fraction): `refinement.*`, `pos_embed`, `cls_token`, `register_token`.

**Threshold**: matched trunk fraction $\ge 0.95$. If below, `accepted=False` is returned and **no weights are applied**. The caller decides (STOP condition).

## 5. The `encode_glimpse` interface (load-bearing)
`CompactViT.encode_glimpse(image, gaze_coords, fovea_size=56) → (pooled_z, tokens)`

- `pooled_z`: $(B, D_z)$ — the CLS-position output after trunk + refinement. This is $z_t$ in the belief state.
- `tokens`: $(B, N, D_\text{feat})$ — patch-token features at the fixation. This is the predictor's target/score space.

This signature is **the load-bearing contract** consumed by Agent E (prediction target) and Agent F (candidate scoring). It must not be changed without a plan ruling.

## 6. Pillar heads (not counted in the budget)
The 20–25M budget is for the trunk and refinement block **excluding pillar heads**:
- `EvidentialHead`: classification readout Dirichlet ($\sim 0.3\text{M}$)
- `ConcreteGlimpseFeaturePredictor`: predictor ($\sim 2.8\text{M}$)
- `ConcreteUpdateNet`: belief update ($\sim 1.8\text{M}$)
- `PrecisionFunction`: precision mapping ($\sim 0.001\text{M}$)

Total system parameters (trunk + pillar heads): $\sim 28\text{M}$.

## 7. Scientific status
- **DINOv2-small-shaped architecture**: **LOCKED** ($D_z = 384$; Part 1.A).
- **20–25M parameter budget**: **LOCKED** (Agent C contract).
- **Warm-start from DINOv2-small**: **LOCKED** (clean mapping required; below-threshold loads are refused).
- **Tied refinement block**: **LOCKED** (Hybrid Option C; Part 1.C).

## 8. Source references
- [`backbone.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py): `CompactViT`, `D_Z`, `PATCH_SIZE`, `TRUNK_DEPTH`, `WARM_START_MATCH_FRACTION_MIN`, `PARAM_COUNT_BAND`
- [`recurrent_block.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/recurrent_block.py): `TransformerBlock`, `TiedRecurrence`
- [`foveation.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/foveation.py): `foveal_sample`, `DEFAULT_FOVEA_SIZE`
- Illustrated in Figure 1 (`system_overview.svg`).
- Connected to Chapters 05, 09, and 10.

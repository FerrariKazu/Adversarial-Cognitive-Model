# Chapter 14 — Multi-Group Gradient Flow: Optimization Boundaries

## 1. In one sentence
RHAN-NXA uses a `MultiGroupOptimizerRegistry` that assigns each independently gradient-isolated component to its own named parameter group with its own per-group learning rate and per-group gradient clipping, preventing any single component's gradient from dominating the shared optimizer budget.

## 2. The Stage 2 incident (motivation)
Before the multi-group optimizer was introduced, all parameters shared a single global `clip_grad_norm_` call. The backbone's TRADES adversarial gradient is typically orders of magnitude larger than the auxiliary head gradients (e.g., HPC, predictive coding). With a global clip, the backbone owned the norm budget and the auxiliary heads received per-step parameter updates of magnitude $\sim 10^{-5}$—the **ratio-1.00 freeze**: the heads learned at a rate indistinguishable from frozen.

The fix (2026-08-13, now a standing architectural rule): each component registers its own named parameter group with:
- Its own `lr_multiplier` (group lr = `base_lr × multiplier`)
- Its own `clip_norm` (group gradient budget)

This is **not discretionary**. Every new trainable component must register its own group.

## 3. Optimizer group map

| Group name | Component | Notes |
|:---|:---|:---|
| `backbone` | `CompactViT` trunk + refinement | Always registered **first** (group 0) |
| `recurrence_refinement` | Tied refinement block | Subset of backbone—may be split |
| `predictor` | `ConcreteGlimpseFeaturePredictor` | Own pre-flight $\lvert dW \rvert$ check |
| `update_net` | `ConcreteUpdateNet` | Own pre-flight $\lvert dW \rvert$ check |
| `precision` | `PrecisionFunction` | Small; own group |
| `evidential_head` | `EvidentialHead` | Own group |
| `gaze_policy` | `AISv2GazePolicy` (logit_scale) | Single scalar parameter |
| `l_stab_diagnostic` | Diagnostic-only (Phase A) | Not an objective; own group |

Source: `schema.py::OPTIMIZER_GROUP_NAMES`.

## 4. Implementation

**`OptimizerGroupRegistry`** (`noesis_vision/core/multi_group_optimizer.py`):
- **`register(name, params, lr_multiplier, clip_norm)`**: A parameter may only belong to one group—double registration raises loudly (two momentum buffers = double updates).
- **`register_backbone(params)`**: Must be called first; backbone is always group 0.
- **`build_optimizer(base_lr)`**: Returns `torch.optim.SGD` with one param group per registered name.
- **`clip_grad_per_group()`**: Applies `clip_grad_norm_` to **each group's parameters separately**, using that group's own `clip_norm`. Never a global budget.

Source: **PORTED VERBATIM** from `rhan_core/optim/multi_group_optimizer.py` (Gen-0, validated across 3+ components).

Two Gen-0-only pieces were **not** ported (documented, not silently dropped):
- HPC/SBR/AIS LR-multiplier constants (Gen-0 component names)
- `default_group_spec(model, ...)` (Gen-0 model structure; Gen-1 group derivation belongs to Agent C/A)

## 5. The resume guard

`registry.resume_guard(optimizer_state_dict, saved_scheduler)` returns `True` only when:
1. Group **count** matches.
2. Group **names** match (catches parameter reordering with the same count).
3. LR-**ratio pattern** matches (uses scheduler `base_lrs` if available; rejects stale decayed lrs).

A `False` result means the caller must construct a fresh optimizer and warn loudly. Silent restoration of a mismatched checkpoint is the failure class that produced the Gen-0 stale-resume bug.

## 6. Pre-flight $|dW|$ check
Before any gradient-bearing smoke test, the pre-flight check measures per-group parameter step magnitude $|dW|$ after one synthetic forward+backward. Specifically:

- Capture $\theta$ before step.
- Execute one forward+backward.
- Step the optimizer.
- Measure $|\theta_\text{after} - \theta_\text{before}|$ per group.

A group with $|dW| \approx 0$ (relative to the backbone group) is in the Stage 2 freeze condition. **Test**: `tests/test_preflight_dw.py` asserts that the UpdateNet and predictor groups both receive non-trivial updates relative to the backbone.

## 7. Scientific status
- **Multi-group optimizer with per-group clipping**: **LOCKED** (Stage 2 fix; now a standing architectural rule).
- **Per-group LR multipliers**: **EXPERIMENTAL CANDIDATE** (specific multiplier values chosen per run, not locked).
- **Port status**: **PORTED VERBATIM** (validated across Gen-0).

## 8. Source references
- [`multi_group_optimizer.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/multi_group_optimizer.py): `OptimizerGroupRegistry`, `DEFAULT_CLIP_NORM`
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `OPTIMIZER_GROUP_NAMES`
- Illustrated in Figure 4 (`gradient_flow.svg`).
- Connected to Chapters 12 (complete loop) and 16 (training system).

# RHAN-Next Architecture

> **Status:** Stages 0–2 **code-complete**, Stage 0 **validated**. Stages 1–3
> **validation pending** on the 5-seed matched GPU protocol (see
> [roadmap](rhan_next_roadmap.json)). This document is updated at the end of
> every stage.

## 1. Purpose

RHAN-Next is the properly separated, long-term package that RHAN-v12's
training pipeline adopts without breaking. It supports four research pillars
without ever needing to be redesigned again:

| Pillar | Name | Status in this refactor |
|---|---|---|
| 1 | Hierarchical Predictive Coding (HPC) | **implemented, 1 level** (edge map) |
| 2 | Active Information-Seeking (AIS) | **implemented** |
| 3 | Structured Belief Representation (SBR) | scaffold only (`enable_sbr` must stay False) |
| 4 | Internal World Model (IWM) | scaffold only (`enable_iwm` must stay False) |

**Non-negotiable lessons this package is built around** (from the RHAN
history):

1. **Gradient-flow bugs are the #1 historical failure mode.** v11/v12's
   reconstruction loss stored `recon_mse.detach()` for two architecture
   generations — present, weighted, logged, and contributing ZERO gradient.
   Every new loss-bearing component in this package has an automated
   gradient-reachability test (`tests/test_gradient_flow.py`).
2. **"Trains without crashing" is not "validated".** Each stage has a
   distinct code-complete and validated checkbox, tracked in
   `docs/rhan_next_roadmap.json`. Validation means the 5-seed matched
   protocol with the pre-registered Δ > 2·σ_combined criterion.
3. **Never add multiple new mechanisms simultaneously.** Every component is
   behind an `RHANNextConfig` toggle and gets an isolated on/off test before
   the next is added.

## 2. Layout

```
rhan_core/
    __init__.py
    beliefs/                  # BeliefState ABC; Vector (P1&P2), Structured (P3 scaffold)
    predictive_coding/        # LevelPredictor/ErrorUnit ABCs, stack, feature targets
    gaze/                     # GazePolicy ABC; InfoGain policy + halting (P2)
    precision/                # PrecisionModulator ABC + GlobalPrecisionModulator (P2)
    world_model/              # WorldModel ABC + NullWorldModel no-op (P4 scaffold)
    config/pillar_config.py   # RHANNextConfig — all pillar toggles + v12 hyperparams
    model.py                  # RHANNext(nn.Module) — composes everything
phase1_training/
    train_rhan_next.py        # new trainer entrypoint (superset of train_rhan_v12.py)
    model_rhan_v12.py         # FROZEN — never modified in this branch
phase2_attacks/
    eval_rhan.py              # FROZEN eval entrypoint (conventions of eval_full_epsilon_sweep.py)
tests/
    test_gradient_flow.py     # mandatory gradient-reachability tests
    test_config_backward_compat.py
    test_pillar_scaffold_import.py
docs/
    ARCHITECTURE.md
    rhan_next_roadmap.json
```

**Frozen files (never edited in this branch):** `phase1_training/model_rhan_v12.py`,
`phase2_attacks/eval_full_epsilon_sweep.py`, `phase2_attacks/eval_rhan.py`.

## 3. Backward compatibility contract

`RHANNext` **subclasses the frozen `RHANv12`**. With the default
`RHANNextConfig` (all pillars off):

- the state dict is **byte-identical** to `RHANv12`'s — a v12 checkpoint (or
  the TRADES-Large base) loads 1:1;
- `forward`/`get_feature_vector` **delegate to the exact v12 implementation**,
  so outputs are numerically identical;
- new pillar modules are created **only when their toggle is on**, so the
  default model adds zero parameters.

This is verified by `tests/test_config_backward_compat.py` (key-set equality +
numerical logits equality + shape checks across all configs).

### 3.1 The v12 machinery RHANNext reuses (unchanged, frozen)

`RHANv12(RHANv11(RHANv10(RHANLargeSTL10)))` — 96×96 input, belief vector
`s` ∈ ℝ⁵¹², classifier head on ℝ⁷⁶⁸:

- `_peripheral_pass(x)` → CLS embedding ℝ⁷⁶⁸ (stem → dual-stream
  gradient-checkpointed transformer → recurrent feedback);
- `foveal_sample` (differentiable grid_sample crop, the motor Jacobian);
- `foveal_stream`, `parafoveal_stream`, `foveal_gate` (multi-resolution fusion);
- `generative_prior` (belief → predicted 48×48 crop) and
  `image_precision` (Pi_D from image-space error);
- Eq. II v12 gaze update: `λ·∇_a R(x,a) + (1−λ)·∇_a ‖f(a)−P(s)‖`, normalized,
  precision-scaled step, fixed T = `max_foraging_steps`.

## 4. Pillar 2 — Active Information-Seeking (Stage 1)

Gated behind `enable_ais=True`.

### 4.1 `InformationGainGazePolicy` (`rhan_core/gaze/info_gain_policy.py`)

Selects the next fixation by maximizing an **expected reduction in belief
uncertainty**. The exact proxy (documented in the class docstring):

> Under a Gaussian likelihood model of sensory prediction, the expected
> information gain of a fixation is proportional to the gradient of expected
> surprise at that fixation. We approximate expected surprise with the
> λ-blended prediction error already computed by the v12 forward pass —
> pixel-space reconstruction error R(x,a) plus feature-space error
> ‖f(a)−P(s)‖ (Itti & Koch 2001 saliency; Friston 2010 free-energy). Exact
> mutual information I(z_future; a) is intractable, so this tractable proxy
> is used *and named* rather than silently approximated.

The policy owns one small learned network (`step_net`) that re-scales the
v12 step-size formula `base + range·Π_D` (initialized to the identity, so
AIS-on behavior starts exactly at v12's). Its parameters are what the
gradient-flow test asserts gradients reach. It stores `last_recon_map` /
`last_surprise` for diagnostics only.

### 4.2 `EntropyGatedHalting` (`rhan_core/gaze/halting.py`)

Halts evidence gathering when belief uncertainty `u = 1 − Π_D` drops below
`ais_halt_threshold`. **No step-count penalty exists anywhere**: the loss has
no `steps_used / max_steps` term (asserted by a static scan in
`tests/test_gradient_flow.py`), directly resolving the historical
contradiction where a `halt_efficiency` loss penalized step count against
the project's own Banach-contraction argument that more steps improve
robustness.

- `should_halt(belief, history)` → (B,) bool (hard, for diagnostics/eval);
- `continuation(belief, history)` → (B,) differentiable soft gate
  `σ(softness·(u − threshold))` used to weight belief accumulation and freeze
  gaze for halted samples — soft halting keeps the batch graph stable (a
  hard per-sample early exit is deferred; see roadmap).

### 4.3 `GlobalPrecisionModulator` (`rhan_core/precision/global_precision.py`)

Unsupervised per-sample precision (never trained against a saturating binary
correctness target — the diagnosed v10/v11 failure). It **wraps the shared
`image_precision` module by plain reference** (no state-dict duplication) and
exposes per-consumer modulation so each can be isolated and tested:

| Consumer | Wiring | v12-equivalent when gain = 1 |
|---|---|---|
| Gaze step size | `step_size = base + range·Π_D·gain` | identical |
| Recurrence depth | halting threshold `· (0.5 + Π_D·gain)` — higher precision halts earlier | n/a (AIS only) |
| Reconstruction loss weight | `w_recon · (0.5 + Π_D·gain)` (trainer side) | `w_recon` flat (AIS off) |

Deferred (logged in roadmap): precision-modulated **attention gating** and
**skip-connection gating** — too many simultaneous knobs broke precision
calibration before.

### 4.4 AIS forward loop

Replicates the v12 loop exactly when `enable_ais=False`. When enabled, the
same loop additionally: builds a `VectorBeliefState` each step (uncertainty
= `1 − Π_D`), consults the halt policy (soft continuation weights +
halted-sample gaze freezing), and drives the gaze update through
`InformationGainGazePolicy.select_action`. All v12 trajectory keys
(`actions`, `precisions`, `errors`, `gate_alphas`, `recon_errors`,
`recon_maps`, `steps`) are preserved; `uncertainties`, `continuations`, and
`hpc_errors` (Stage 2) are added.

## 5. Pillar 1 — Hierarchical Predictive Coding (Stage 2)

Gated behind `enable_hpc=True, hpc_num_levels=1`.

### 5.1 One level, implemented: `EdgeFeatureLevelPredictor`

```
feature_target = "edge_map"
s (B,512) ──MLP──▶ predicted edge map (B,1,48,48)      [top-down]
x_foveal ──Sobel──▶ extracted edge map (B,1,48,48)     [bottom-up actual]
error = MSE(prediction, actual)                        [(B,), enters the loss]
```

- The target is a **feature (edges), never raw pixels above the lowest
  level** — the raw-pixel level remains the existing `generative_prior`.
- The extractor (`EdgeMapExtractor`) is non-learnable (Sobel, buffers only),
  so the predictor is the only trainable piece — exactly what
  `test_gradient_flow.py` asserts.
- The new predictor sits **alongside** the existing top-level predictor
  (v12's `precision_ctrl.prior_predictor` + `generative_prior`), per the
  Stage-2 spec.
- `OrientationMapExtractor` ships as a real, importable extractor for the
  future Level 1 (wiring deferred until Level 0 validates — one level per
  cycle). `ShapeEmbeddingExtractor` is scaffold-only.

### 5.2 Why belief-anchored rather than mid-transformer-layer

The dual-stream transformer runs under `torch.utils.checkpoint` with
`use_reentrant=False` inside the recurrent feedback loop. Capturing a
mid-layer tensor there (forward hooks re-fire during backward recompute and
produce detached/duplicated intermediates) is fragile and would break the
mandatory gradient-flow test. The belief-anchored level is gradient-safe,
isolated, and satisfies the feature-vs-pixel rule. Mid-transformer hooking
is logged in the roadmap as a deferred increment.

## 6. Pillars 3 & 4 — scaffolds

- **SBR (`StructuredBeliefState`)**: real, importable, instantiable
  interface (slot count/dim); `as_tensor`, `uncertainty`, `update_slots`,
  `message_passing` raise a clear documented `NotImplementedError`. No
  import/instantiation-time errors.
- **IWM (`NullWorldModel`)**: zero-parameter passthrough wired into every
  RHANNext; `simulate()` returns the belief unchanged and logs a debug
  notice. `WorldModel.simulate(belief, action)` is the reserved interface for
  the future `SimulatedGazePolicy` (Dreamer/MuZero-style rollout).

## 7. Config

`RHANNextConfig` (dataclass) carries every v12 hyperparameter unchanged plus
the pillar toggles. `validate()` enforces the stage gates:
`enable_sbr`/`enable_iwm` → error; `hpc_num_levels > 1` → error. Serialization
(`to_dict`/`from_dict`) round-trips so checkpoints can embed the config.

## 8. Training (`phase1_training/train_rhan_next.py`)

A **strict superset** of `train_rhan_v12.py`: same data pipeline (STL-10
real + pseudo + synthetic mixes), same curriculum (3 phases, SGD), same
warmup freeze schedule (extended to the new pillar components), same HF
rolling-checkpoint resume gates, same diagnostics. The loss gains the
modulated terms:

```
L = w_trades·L_trades + (w_recon·(0.5 + Π_D·gain))·L_recon + w_hpc·L_hpc
```

where `L_hpc = 0` when HPC is off. **No step-count penalty exists.** The
checkpoint saves the model state dict under `model` and the `RHANNextConfig`
under `config`, so `eval_rhan.py` can reconstruct the exact pillar config.

## 9. Evaluation (`phase2_attacks/eval_rhan.py`)

Frozen entrypoint with conventions **identical** to
`eval_full_epsilon_sweep.py`: norm-space eps applied directly, per-channel
bound checks, 3-seed averaging (n=300/seed), Δ > 2·σ_combined crossover
criterion. It extends the arch registry with `next`, which constructs
`RHANNext` from the config embedded in the checkpoint (falling back to the
v12-equivalent default). No other eval script may be added per stage.

## 10. Validation status

| Stage | Code complete | Validated | Evidence |
|---|---|---|---|
| 0 | ✅ | ✅ | 11 local tests pass (RTX 4060) — no eval sweep required |
| 1 | ✅ | ⏳ pending | 5-seed matched protocol, eps 0.000/0.094, on Colab/Kaggle |
| 2 | ✅ | ⏳ pending | isolated on/off test, hpc_num_levels 0 vs 1 |
| 3 | ⏳ (trainer + eval entrypoint implemented) | ⏳ pending | final 3-model comparison, full grid, numbers here |

**Validation runs (not yet executed — require GPU hours and STL-10 data):**

```
# Stage 1 (AIS vs v12 baseline):
python3 phase2_attacks/eval_rhan.py --n-samples 300 --seeds 41 42 43 \
    --pgd-steps 50 --batch-size 64 --eps-norm-space --eps-list 0.0 0.094 \
    --baseline-label trades_large_baseline \
    --ckpt-specs \
      trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
      rhan_v12_baseline:checkpoints/rhan_v12_mixB_best.pth:v12 \
      rhan_next_ais:checkpoints/rhan_next_ais_best.pth:next

# Stage 2 (HPC on/off at fixed AIS):
    ... rhan_next_ais:...:next  rhan_next_hpc1:checkpoints/rhan_next_hpc1_best.pth:next
```

### 10.1 Explicit statement on Pillars 3 & 4 (Stage 3 requirement)

Pillars 3 (SBR) and 4 (IWM) remain **unimplemented**; their interfaces were not touched or broken by any Stage 1-3 work — `tests/test_pillar_scaffold_import.py` is re-run in the final validation pass to prove this.

## 11. Interpretations & decisions recorded

- `phase2_attacks/eval_rhan.py` did not exist in the repo; the spec lists it
  under "CREATE THIS EXACTLY", so it was created as the frozen canonical
  entrypoint (conventions copied from `eval_full_epsilon_sweep.py`) rather
  than modifying the existing sweep file.
- HPC level implemented at the belief level (see §5.2) with the rationale
  documented, not silently deviated.
- Halting is implemented as a soft differentiable gate (hard per-sample
  early exit deferred) to keep batch graphs stable and the gradient tests
  deterministic.

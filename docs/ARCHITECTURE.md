# RHAN-Next Architecture

> **Status:** Stages 0–2 **code-complete**, Stage 0 **validated**, Stage 1
> **in execution** (not yet validated). Stages 1–3 **validation pending** on
> the 5-seed matched GPU protocol (see [roadmap](rhan_next_roadmap.json)).
> This document is updated at the end of every stage.
>
> **Stage 1 execution artifact:** `cloud_setup/colab_notebook_noesis.py` — the
> pre-registered Step A (smoke) → Step B (full) → Step C (5-seed eval)
> pipeline. Training lives on Colab/Kaggle, not locally: the RTX 4060
> measures ~2.7–9.2 epochs/hour on this pipeline, so 10–15 epochs is 2–5 h
> and the full 60-epoch run is 7–20 h — far over the 1-hour local budget.

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

### 4.0 Mechanistic identity (honest Stage 1 claim — read before Stage 1 runs)

> **RUN LABEL (must be used verbatim in every result table and paper draft
> section referencing this run):**
>
> **"AIS-v1: Relocated Equation II (mechanistically identical to
> v10/v11/v12's gaze update, refactored into the new interface). This is a
> REPLICATION-UNDER-REFACTOR control, not a test of genuine information-gain
> gaze."**
>
> Unqualified "AIS" or "information-gain" must NOT appear for this
> checkpoint; "AIS" / "AIS-v2" is reserved for a future genuinely
> forward-looking implementation. All run artifacts (checkpoint names,
> diag paths, eval labels, output dirs) are named `rhan_next_ais_v1_*` so
> the versioned label propagates automatically into eval result tables.
>
> **STAGE 1 VARIANT LABEL (2026-08-07 — the validated run):** the mechanism
> isolation attributed the smoke's Π_D reordering to the precision-modulated
> reconstruction weight (see §8.1a outcome). Per the pre-registered decision
> rule, the Step B validated run is labeled **"AIS-v1 (halting-only
> variant)"** — AIS-v1 with `--no-ais-precision-recon` (halting + relocated
> Eq. II gaze + precision-driven step/β ON; recon-mod OFF and **DEFERRED** to
> its own future isolation cycle). Artifacts for the validated run are named
> `rhan_next_ais_v1_halting_only_*`, so every result table and
> `eval_provenance.json` carries the variant label — never plain "AIS-v1".

`InformationGainGazePolicy.select_action()` is, at initialization,
**mechanically identical to the v12 Eq. II v12 gaze update** — which is itself
the v10/v11 "Eq. II" prediction-error gradient relocated into the new class
structure. The lineage: v10 used the FEATURE-space error only, v11 switched
to the PIXEL-space error only, and v12 (which RHANNext inherits) is the
λ-blend of both. The policy's only additions are `step_net` (a learned step
re-scale initialized to the identity) and the precision-gain consumer
(gain=1 → identical to v12).

**Consequence for the Stage 1 hypothesis:** AIS-v1 behavior *starts* at
v12's Eq. II v12 update; `step_net` and the precision gain are what training
can move it away from. This is a clean-architecture + same-mechanism outcome,
NOT a claim of literal expected-information-gain computation (no one-step-
ahead uncertainty prediction exists). The class docstring states this plainly.

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

**DEFERRED from the Stage 1 headline config (2026-08-07):** the
reconstruction-weight consumer is disabled in the validated run
(`--no-ais-precision-recon`, the "AIS-v1 (halting-only variant)") — the
mechanism isolation attributed the smoke's Π_D reordering to exactly this
consumer (see §8.1a outcome). It gets its own future isolation cycle before
any AIS-v1 claim may include it. Also deferred (logged in roadmap):
precision-modulated **attention gating** and **skip-connection gating** —
too many simultaneous knobs broke precision calibration before.

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

### 8.1 AIS diagnostics (RHANNextEpochDiagnostics + `--diag-json`)

`train_rhan_next.py` uses `RHANNextEpochDiagnostics` (subclass of v12's
`EpochDiagnostics`; the frozen v12 file is untouched), which emits the v12
block (β_dynamic, gate α, recon MSE, Π_D per class) plus the two AIS
signals the Stage 1 smoke gate requires:

| Signal | Definition | Degenerate if |
|---|---|---|
| **Gaze shift distance** | mean over batch of ‖a_t − a_{t−1}‖₂ at every step boundary + total path length | ≈ 0.0 (fovea never moves) |
| **Per-sample halting variance** | effective evidence steps per sample = Σ_t continuation_t (soft gate weights); reported as mean/std/min/max + fraction of samples with any step < 0.5 | std ≈ 0 (flat steps — the v10/v11 failure mode) |
| **Π_D per class** | inherited v12 block; car/truck must stay highest | car/truck not top — ordering broke |

`--diag-json <path>` appends one JSON line per epoch (`summary_dict()`):
`epoch, eps, beta_dyn mean/std, gate_alpha, recon_mse, steps_hard_fixed,
steps_effective mean/std/min/max, frac_halted_any, gaze_shift_total_mean,
pi_d_per_class`. The notebook health gate parses this file; it is also
human-readable in the per-epoch printed block.

### 8.1a Health-gate threshold calibration (every threshold is calibrated)

Each health-gate threshold is fixed against a **known-good run's measured
value**, not an arbitrary nonzero floor (a gate that passes trivially is not
gating anything):

| Criterion | Threshold | Known-good (measured) | Known-dead (measured) | Rationale |
|---|---|---|---|---|
| `gaze_shift_total_mean` | **≥ 0.05** (raised from 0.01) | **0.2795** — this pipeline's local dry-run (RTX 4060, 1 epoch); **0.36–0.39** — v11 post-fix diagnostics (after the gaze-normalization bug was corrected) | **~0.007** — v11 PRE-fix state (buggy normalization froze the fovea) | The old 0.01 bar sat inside the noise of the broken regime and would have passed it. 0.05 is ~7× the dead value and ~⅙ of the known-good value — a real discriminator. |
| `steps_effective_std` | **≥ 0.02** (unchanged) | **0.561** — same local dry-run | **0.000** — v10/v11 permanently flat `steps=4.00` | Flat failure is exactly 0.0; 0.02 discriminates "flat vs not flat". Deliberately far below the measured 0.561 healthy std so it only gates the v10/v11 failure mode, which is its job. |
| `frac_halted_any` | **≥ 0.02** (tightened from > 0) | **0.125** — 12.5% of samples halted ≥ 1 step (same dry-run) | **0.000** — v10/v11 (no sample ever halted) | Complements the std check. NOT just "> 0" (per the calibration directive: a gate that passes trivially isn't gating): 0.02 requires real halting (≥ 2% of samples) while staying 6× below the measured 0.125 known-good — no false-abort risk. |
| car/truck in top-2 Π_D | both present (unchanged) | v11: car 0.4198 / truck 0.4670 among the top classes; v12 mixA/B epoch logs: car 0.4726 / truck 0.4587 highest | ordering broken (Π_D collapses / single class) | Reproduced across every RHAN version's diagnostics; a break is a red flag to stop and debug, not push through. |

The concrete numbers are also pinned as module-level constants in
`cloud_setup/colab_notebook_noesis.py` (`GAZE_THRESHOLD = 0.05`,
`HALT_STD_THRESHOLD = 0.02`, plus the known-good/dead anchors), so the gate
reason strings print the calibration basis when they fire.

**Π_D criterion — finding (2026-08-06).** The gate FIRED on this criterion in
the Colab Step A smoke: final-epoch top-2 was **car 0.449 / airplane 0.421**
(truck 6th at 0.407, ship 9th), while gaze (0.2175) and halting (std 0.886,
frac 0.213) passed comfortably. Two facts bound the interpretation:

- **The smoke already trained on the reference data composition.** The
trainer generates pseudo-labels by default (conf ≥ 0.65 → ~46K of the 100K
unlabeled set; `RHANv12/scripts/generate_flowcharts.py` documents the ~46K
count) and the notebook's Step A command passes no `--no-pseudo`. The smoke's
throughput (7.19 img/s ≈ mixB's 7.08–7.30, vs mixA's synthetic 24.51 img/s)
confirms it. The v12 mixB / Sprint-1 reference runs used the same 5K real +
~46K pseudo composition and showed car/truck top-2 — so the **data-mix
hypothesis is ruled out by construction**: the reordering must come from an
AIS-v1 sub-mechanism, not the data.
- **The criterion's calibration was narrower than its claim.** "Car/truck
both top-2, reproduced across every RHAN version" only held for the v12
mixA/B epoch logs (car 0.4726 / truck 0.4587). v11's OWN documented Π_D
table (`RHANv11.md`) shows **ship/truck** on top with car 5th/8th (best.pth:
ship 0.4975 / truck 0.4670; rolling: truck 0.5488 / ship 0.5064) — the
criterion as written would have flagged v11's only confirmed healthy run.
The Π_D computation itself is verified identical to v12 (the modulator wraps
the frozen `image_precision` module by plain reference), so the ordering
shift is a real diagnostic observation, not a wiring bug.

**Isolation outcome — driver identified (2026-08-07).** The two bounded
isolation arms completed and the pre-registered decision rule fired branch
(2): **isoB (precision-recon OFF, halting ON) RESTORED car/truck** — final
epoch Π_D top-2 = car 0.5149 / truck 0.4842, with the reference error
ordering intact (car 0.7287 / truck 0.7007 highest) and halting firing MORE
(frac_halted 0.213 → 0.364). Because halting was ON in both the smoke and
isoB, the smoke↔isoB contrast alone proves the **precision-modulated
reconstruction weight is the confirmed driver** of the car/airplane
reordering; the entropy-gated halting is exonerated by elimination. The
mechanism: `modulate_recon_weight(w_recon, π_D) = w_recon·mean(0.5 + π_D·gain)`
is a batch-scalar multiplier coupling average batch confidence to recon
gradient magnitude; cutting the loop (flat `w_recon`) raises Π_D across the
board (car 0.449 → 0.515) and restores the reference top-2.

isoA's final-epoch telemetry was **lost to a session wipe** (its diag
`.jsonl` is local-only; only checkpoints synced to HF), so its verdict is
recorded as **inconclusive, not "not restored"**. It is re-run as a
**sufficiency test** at an amended budget (max-epochs 14, resumes 12→14):

- isoA → car/airplane (like the smoke): recon-mod is SUFFICIENT alone →
  "recon-mod both necessary and sufficient";
- isoA → car/truck (like isoB): recon-mod alone is NOT sufficient — the
  degenerate ordering needs recon-mod AND variable halting together → a true
  interaction effect, retroactively validating recon-mod OFF for Step B even
  more strongly.

**Decision (pre-registered):** Step B validates the **"AIS-v1 (halting-only
variant)"** (`--no-ais-precision-recon`). The recon-mod consumer is DEFERRED
to its own future isolation cycle — not validated, not part of the headline
config, and never folded into any future AIS-v1 claim without a separate
validated isolation. Durability fix: per-epoch diag `.jsonl` files and the
roadmap now sync to HF alongside verdicts and checkpoints, so no future
session interruption can lose telemetry the way isoA lost 12 epochs.

### 8.2 Local pipeline validation (RTX 4060, < 1 h)

A real-only dry-run was executed locally to prove the pipeline end-to-end:
`--enable-ais --dry-run --no-pseudo --force-restart --ckpt-name
rhan_next_local_dryrun`. Results: 75,473,431 params, base checkpoint loaded
(86 missing = new pillar modules), one training step + full diagnostics
block. Measured telemetry: gaze shift per step ≈ 0.26/0.29/0.29 (total path
0.28 — fovea moving), effective evidence steps mean=3.60 std=0.56
min=2.36 max=3.95 with 12.5% of samples halted — i.e. halting DOES vary per
sample even in a single batch. This validates the diagnostics and the AIS
forward path; it is NOT the Stage 1 validation (that is the 5-seed matched
protocol on GPU).

## 9. Evaluation (`phase2_attacks/eval_rhan.py`)

Frozen entrypoint with conventions **identical** to
`eval_full_epsilon_sweep.py`: norm-space eps applied directly, per-channel
bound checks, 3-seed averaging (n=300/seed), Δ > 2·σ_combined crossover
criterion. It extends the arch registry with `next`, which constructs
`RHANNext` from the config embedded in the checkpoint (falling back to the
v12-equivalent default). No other eval script may be added per stage.

### 9.1 Protocol hardening (every future number routes through this file)

Since every published number now routes through `eval_rhan.py`, four
guarantees are enforced at the entrypoint (the frozen sweep file underneath
is untouched):

| # | Guarantee | Implementation |
|---|---|---|
| 1 | **No pixel-space mode** | `--eps-norm-space` is injected unconditionally before delegation; the pixel-space default of the underlying parser is unreachable through this file |
| 2 | **≥5 seeds by default** | fewer than 5 seeds aborts unless `--allow-quick` is passed (consumed here, never forwarded); single-seed numbers are flagged as dev-only |
| 3 | **Significance verdict printed** | inherited from the frozen `crossover_report()` (Δ > 2·σ_combined), printed automatically |
| 4 | **`--self-test`** | structural check (config, state-dict key hash, param count, forward shapes) against the checked-in `phase2_attacks/eval_rhan_selftest_ref.json`; regenerate deliberately via `--regenerate-reference` |
| 5 | **Provenance JSON** | `eval_provenance.json` written after every run: git SHA + branch, per-checkpoint SHA-256, seed list, CLI settings, UTC timestamp, merged results, and recomputed crossover verdicts |

The Stage 1 eval command (Step C) therefore needs no `--eps-norm-space` flag
(it is injected) and satisfies the seed floor with `--seeds 41 42 43 44 45`.

## 9a. Stage 1 execution protocol (Steps A → B → C)

Pre-registered in `cloud_setup/colab_notebook_noesis.py`; the notebook
checkout is branch-gated on `feature/rhan-next` (it never resets to
`origin/main`, which does not contain RHANNext).

**Step A — Smoke test** (bounded, catches bugs before commitment):
`train_rhan_next.py --enable-ais --ckpt-name rhan_next_ais_v1_smoke
--max-epochs 15 --target-ckpt checkpoints/rhan_stl10_large_pseudolabel_best.pth
--batch-size 16 --accum-steps 16 --diag-json
report/rhan_next_ais_v1_smoke_diag.jsonl`. Epochs 1–15 all fall in phase 1, so
ε = 0.031 only. Base checkpoint: the same one used for every prior isolation
experiment (`rhan_stl10_large_pseudolabel_best.pth`).

**Health gate** (automated, after Step A): reads the last `--diag-json` line
and aborts Step B — with reasons — unless:

1. `gaze_shift_total_mean ≥ 0.05` (fovea actually moves — calibrated: local
   dry-run measured 0.2795, v11 post-fix 0.36–0.39, v11 pre-fix dead state
   ~0.007; the old 0.01 bar would have passed the dead state, so it was
   raised; see §8.1a);
2. `steps_effective_std ≥ 0.02` and `frac_halted_any ≥ 0.02` (halting varies
   per sample — not the v10/v11 flat 4.00; known-good std=0.561, frac=0.125,
   so both floors are calibrated well below healthy and above the exact 0.0
   dead state);
3. `car` and `truck` are both in the top-2 Π_D per class (the ordering that
   has reproduced across every RHAN version — if it breaks, stop and debug).

The verdict JSON is written to `report/rhan_next_ais_v1_smoke_health.json`.
`FORCE_STEP_B_OVERRIDE` exists as a debug escape and is documented as NOT
for publishable numbers. On a restarted session the gate re-evaluates the
synced telemetry summary against the CURRENT criteria (a stale boolean
verdict can never block a legitimate resume).

**Step 5c — Mechanism isolation (Run A / Run B pattern).** The smoke gate
fired on the Π_D ordering criterion only (see §8.1a finding), and the
pre-registered decision rule keeps Step B gated until the driver is
identified. Two BOUNDED arms ablate exactly one AIS sub-mechanism each,
everything else identical to the smoke:

| Arm | Ablation | Trainer flag |
|---|---|---|
| Isolation A | entropy gate forced open (`cont = 1` ⇒ v12 fixed-T belief accumulation); gaze update unchanged | `--no-ais-halting` |
| Isolation B | `w_recon` stays flat (v12 recon weighting); the precision modulator no longer scales the recon loss | `--no-ais-precision-recon` |

Each arm runs `ISO_EPOCHS = 12` phase-1 (ε = 0.031) epochs from the same
base checkpoint, follows the same NEVER-RESTART / auto-resume guarantees
(rolling checkpoints `rhan_next_ais_v1_iso{A,B}_*` sync to HF;
`verify_no_restart` asserted), and logs the final-epoch per-class Π_D. The
pre-registered decision rule (roadmap `stages['1'].isolation_plan`): an arm
that restores car/truck in top-2 identifies its ablated mechanism as the
primary driver; if neither restores, the driver is the remaining shared AIS
wiring (gaze-policy relocation / `step_net` / continuation×belief interplay)
and the next isolation disables `step_net`. The verdict is recorded to
`report/rhan_next_ais_v1_isolation_verdict.json`and `roadmap.stages['1'].isolation_verdict` regardless of outcome.
2026-08-07 outcome: branch (2) fired — isoB restored car/truck; isoA
inconclusive (telemetry lost to a session wipe; recaptured as a sufficiency
test at an amended 14-epoch budget); see §8.1a for the full attribution and
the decision.

**Step B — Full validated run (AIS-v1 (halting-only variant))**: same trainer,
`--no-ais-precision-recon --ckpt-name rhan_next_ais_v1_halting_only`,
`--max-epochs 60` (recon-mod OFF + DEFERRED per the isolation verdict — see
§8.1a; halting and the relocated Eq. II gaze remain ON). The curriculum
`(1-20 @0.031, 21-40 @0.062, 41-60 @0.094)`
is byte-identical to `train_rhan_v11.py`'s — the exact boundaries of the
null_ablation_v11 run that produced 31.56±2.88 @ ε=0.094 — so the result is
directly comparable. Same base checkpoint; the trainer's mandatory HF resume
gate forbids silent restarts (no `--force-restart`).

**Step C — Validation**: through the hardened entrypoint:

```
python3 phase2_attacks/eval_rhan.py \
    --ckpt-specs rhan_next_ais_v1_halting_only:checkpoints/rhan_next_ais_v1_halting_only_best.pth:next \
                 trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
    --seeds 41 42 43 44 45 --eps-list 0.000 0.094 --n-samples 300 --pgd-steps 50
```

The eval label is `rhan_next_ais_v1_halting_only`, so every row of the result
tables and `eval_provenance.json` carries the "AIS-v1 (halting-only variant)"
label (never plain "AIS-v1", never unqualified "AIS"). The notebook then
parses `report/sweep_stage1_ais_v1_halting_only/eval_provenance.json`
(results + recomputed crossover verdicts) and records the outcome in
`docs/rhan_next_roadmap.json` under `stages.1.stage1_verdict` — a null result
is a valid, reportable Stage 1 outcome. If Step B did not complete (health
gate / isolation verdict), Step C is **skipped with an explicit message** —
it never raises a RuntimeError.

**Step C PGD-100 spot-check (masking re-confirmation):** a second invocation
runs PGD-100 at **ε=0.094 only**, same seeds/n, to recompute the 2-step
PGD-50-vs-100 convergence gap that first confirmed "genuine robustness, not
gradient masking" for this configuration family:

```
python3 phase2_attacks/eval_rhan.py \
    --ckpt-specs rhan_next_ais_v1_halting_only:checkpoints/rhan_next_ais_v1_halting_only_best.pth:next \
                 trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
    --seeds 41 42 43 44 45 --eps-list 0.094 --n-samples 300 --pgd-steps 100 \
    --output-dir report/sweep_stage1_ais_v1_halting_only_pgd100
```

Historical bar (RHANv11.md): PGD-50 45.20% vs PGD-100 44.40% at ε=0.031 =
tight convergence, drop 0.8 pp → genuine robustness. Finding 14 repeated
this at scale (27.3% → 27.2% at ε=0.05, zero decay = masking-free). AIS-v1
is a **refactored implementation of the same mechanism**, so this property
must be re-confirmed rather than assumed to carry over. `record_verdict()`
computes `gap = acc(PGD-50) − acc(PGD-100)` at ε=0.094 per checkpoint and
stamps a **three-tier** `masking_check` verdict into the roadmap: drop
≤ 1.0 pp = GENUINE (RHANv11 bar); 1.0–2.5 pp = BORDERLINE / inconclusive
(within the project's documented ~1.5 pp cross-run nondeterminism floor —
these numbers come from two separate eval invocations, so the gap inherits
it; recheck with AutoAttack before trust); > 2.5 pp = MASKING RISK. The
check also records both provenances' git SHAs + timestamps for auditability.
Runtime add-on: ~4.5–5.5 GPU-hours (10 combos × PGD-100).

**Stage 2 (HPC) must not begin until this verdict is recorded and reviewed.**

## 10. Validation status

| Stage | Code complete | Validated | Evidence |
|---|---|---|---|
| 0 | ✅ | ✅ | 11 local tests pass (RTX 4060) — no eval sweep required |
| 1 | ✅ | ⏳ **in execution** | Isolation verdict recorded (2026-08-07): isoB restored car/truck → recon-mod confirmed as the Π_D-reordering driver (smoke↔isoB contrast; halting exonerated); isoA inconclusive (telemetry lost, recaptured as sufficiency test). Step B now trains the **AIS-v1 (halting-only variant)** (`--no-ais-precision-recon`); recon-mod deferred to its own isolation cycle. Step B (60-epoch) + 5-seed matched eval pending |
| 2 | ✅ | ⏳ pending | isolated on/off test, hpc_num_levels 0 vs 1 (must NOT start until Stage 1 verdict recorded) |
| 3 | ⏳ (trainer + eval entrypoint implemented) | ⏳ pending | final 3-model comparison, full grid, numbers here |

**Validation runs (not yet executed — require GPU hours and STL-10 data):**

```
# Stage 1 (AIS-v1 (halting-only variant) vs baseline) — via the notebook (Step C):
python3 phase2_attacks/eval_rhan.py \
    --ckpt-specs rhan_next_ais_v1_halting_only:checkpoints/rhan_next_ais_v1_halting_only_best.pth:next \
                 trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
    --seeds 41 42 43 44 45 --eps-list 0.000 0.094 --n-samples 300 \
    --pgd-steps 50 --batch-size 64 --output-dir report/sweep_stage1_ais_v1_halting_only

# Stage 1 PGD-100 spot-check (masking re-confirmation, eps=0.094 only):
python3 phase2_attacks/eval_rhan.py \
    --ckpt-specs rhan_next_ais_v1_halting_only:checkpoints/rhan_next_ais_v1_halting_only_best.pth:next \
                 trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
    --seeds 41 42 43 44 45 --eps-list 0.094 --n-samples 300 --pgd-steps 100 \
    --batch-size 64 --output-dir report/sweep_stage1_ais_v1_halting_only_pgd100

# Stage 2 (HPC on/off at fixed AIS-v1 (halting-only variant)):
    ... rhan_next_ais_v1_halting_only:...:next  rhan_next_hpc1:checkpoints/rhan_next_hpc1_best.pth:next
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
- **Stage 1 is a clean-architecture outcome, not a new mechanism.** The
  AIS-v1 gaze policy is mechanically identical to v12's Eq. II v12 update at
  initialization (itself the v10/v11 Eq. II gradient); only `step_net`
  (identity init) and the precision gain differentiate it after training
  (see §4.0). The Stage 1 verdict must be read with this honest scope and
  labeled **AIS-v1: Relocated Equation II — a replication-under-refactor
  control, not a test of genuine information-gain gaze** in every result
  table and paper draft section.
- **Training lives in the cloud notebook.** `cloud_setup/colab_notebook_noesis.py`
  hosts Steps A–C because the 4060 cannot finish even the smoke in < 1 h
  (measured ~2.7–9.2 epochs/hour). The notebook is branch-gated on
  `feature/rhan-next` and never resets to `origin/main` (the prior
  notebooks' `git reset --hard origin/main` would delete RHANNext).
- **Provenance is now machine-readable.** After every eval, `eval_provenance.json`
  carries git SHA, checkpoint hashes, seeds, timestamp, results and the
  recomputed crossover verdicts — so roadmap updates are transcription-free.
- **Local dry-run validated the pipeline, not the result.** The RTX 4060
  dry-run (real-only, one step) proved model build, base-ckpt load, AIS
  forward/backward, loss, and the new diagnostics; it is not Stage 1
  validation. The mandatory 5-seed protocol still runs on GPU.
- **Π_D ordering shift is mechanism-caused, not data-caused (2026-08-06).**
  The Colab smoke's top-2 Π_D was car/airplane (truck 6th, ship 9th) while
  the v12 reference runs show car/truck. The smoke already trained on the
  reference composition (5K real + ~46K pseudo — the trainer's default), so
  data mix is ruled out. Because the criterion's "car/truck top-2" calibration
  is contradicted by v11's own documented table (ship/truck), we did NOT
  silently recalibrate or override: Step B stays gated and two bounded
  isolation arms (halting OFF; precision-recon OFF) attribute the driver per
  the pre-registered decision rule. Whatever the outcome, it is recorded — a
  null/ambiguous isolation is a valid result and a fix or criterion change
  will only be applied going forward, never retroactively.
- **Driver attributed to the precision-modulated recon weight (2026-08-07).**
  The isolation fired branch (2) of the pre-registered rule: isoB (recon-mod
  OFF, halting ON) restored car/truck; since halting was constant across the
  smoke and isoB, the recon-mod wiring is the confirmed driver and the
  entropy-gated halting is exonerated by elimination. Step B validates the
  **"AIS-v1 (halting-only variant)"** (`--no-ais-precision-recon`); recon-mod
  is DEFERRED to its own future isolation cycle and must not be folded into
  any AIS-v1 claim without a separate validated isolation. isoA's verdict is
  INCONCLUSIVE (telemetry lost to a session wipe — its diag `.jsonl` was
  local-only), recaptured as a sufficiency test (max-epochs 14, resumes
  12→14) to decide whether recon-mod is sufficient alone or only in
  interaction with variable halting.
- **Telemetry durability fixed (2026-08-07).** The isoA loss exposed that
  per-epoch diag `.jsonl` files and the roadmap were local-only. The notebook
  now syncs every diag file + the roadmap to HF alongside checkpoints and
  verdicts (`upload_hf_file` / `sync_roadmap_up`), so an interrupted Step B
  session can never lose 60 epochs of diagnostic history the way isoA lost 12.
- **Isolation arms are real ablation switches, not rewrite tricks.**
  `RHANNextConfig` gained `ais_halt_enabled` and `ais_precision_recon_enabled`
  (both default True = smoke behavior), plumbed through
  `train_rhan_next.py --no-ais-halting / --no-ais-precision-recon` and
  pinned by `tests/test_ais_ablation_flags.py` (constant continuation when
  halting off; flat `w_recon` when recon modulation off).

## 12. Research clusters — long-term roadmap (all will be implemented)

**Master document: `rhan_core/docs/RESEARCH_CLUSTERS.md`.** This section is a
pointer + status snapshot only; the master document holds the full detail
(core ideas, scientific precedents, unknowns, entry-point tests) for all seven
research clusters that RHAN-Next exists to host. The machine-readable mirror
is the top-level `research_clusters` key in `docs/rhan_next_roadmap.json`.

Statuses below follow the master doc's legend: ✅ implemented (code +
gradient-tested), 🚧 in validation (matched-eval verdict pending), 🟡 scaffold /
partial, ⬜ not started, 🔒 deferred by design. **A cluster is only
VALIDATED when its pre-registered 5-seed matched verdict is recorded — code
complete is never validation.**

| # | Cluster | Status | Where it lives | Entry-point test |
|---|---------|--------|----------------|------------------|
| 1 | Hierarchical Predictive Coding | 🟡 PARTIAL (level 0 only, untrained) | `rhan_core/predictive_coding/` | HPC level 0 on/off, one level per cycle (Stage 2, gated on Stage 1) |
| 2 | Curiosity Gaze / Precision Everywhere / Uncertainty-Gated Recurrence | ✅ **AIS-v1** implemented 🚧 in validation (Step B = **halting-only variant**; recon-mod deferred 2026-08-07) | `rhan_core/gaze/`, `rhan_core/precision/` | Steps A→B→C 5-seed matched + PGD-100 masking spot-check |
| 3 | Episodic Object Memory + Temporal Belief Persistence | ⬜ NOT STARTED | *(needs `rhan_core/memory/`)* | Carry belief across TDV/UCF-101 frame pair |
| 4 | Generative Imagination (counterfactuals, simulation, object permanence) | 🟡 SCAFFOLD (Pillar 4) | `rhan_core/world_model/` | 40%-occlusion patch; does the generative prior recover class? |
| 5 | Structured, Relational, Object-Centric Belief (slots) | 🟡 SCAFFOLD (Pillar 3) | `rhan_core/beliefs/structured_belief.py` | Clean-STL-10 slot training first, no adversarial training |
| 6 | Distributional & Multi-Hypothesis Belief | ⬜ NOT STARTED | *(needs `rhan_core/beliefs/distributional_belief.py`)* | Formalize feature-vs-pixel error agreement check (#18) |
| 7 | Adversarial-Aware Self-Monitoring (selective classification) | ⬜ NOT STARTED | *(needs `rhan_core/selective/` or `precision/` extension)* | Calibration head + abstain vs adversarial AND OOD (ImageNet-C style) |

Key constraints carried over from the master doc: every new loss-bearing path
must land with a gradient-reachability test (lesson #1); never add multiple new
mechanisms simultaneously (lesson #3); every status change must be mirrored in
`docs/rhan_next_roadmap.json` at the end of the stage that touches it.

> **Pillars 3 & 4 remain unimplemented.** Clusters 4 and 5 map onto Pillars 4
> (IWM) and 3 (SBR) respectively — both remain scaffold-only, `enable_iwm` /
> `enable_sbr` MUST stay False, and `tests/test_pillar_scaffold_import.py`
> keeps proving their interfaces are untouched and import-clean.

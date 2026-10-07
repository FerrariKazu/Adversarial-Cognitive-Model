# Gen-2 Experiment Registry

> Authority: this registry is derived from **`gen2_foundation/flags.py`**
> (pre-registered experiment IDs — the code is truth), **`gen2_foundation/AUDIT.md`**
> (frozen-scope dispositions), and the master plan
> `noesis_vision/RHAN_NXA/docs/Proposed_Plan.md` (PART XIII final master roadmap).
> Statuses below were audited against the actual package code and tests on
> 2026-10-06. An experiment is **not** "implemented" merely because the plan
> mentions it.
>
> **Registry rule:** the Gen-2 campaign is LOCKED until the J1 Foundation Gate
> (`cloud/gen1/colab_j1_foundation_gate.py`) records a **PASS from real T4
> results**. Currently: **OPEN / INCONCLUSIVE — no T4 results exist.**

## Status vocabulary

| Status | Meaning |
|---|---|
| `harness-ready` | infrastructure exists; experiment not run |
| `implemented` | mechanism code + unit tests exist in `gen2_foundation/` |
| `registered-only` | experiment ID pre-registered in `flags.py`; mechanism not implemented |
| `ready-to-integrate` | capability exists elsewhere (trainer/J1); Gen-2 arm not registered |
| `experimental` | classified experimental in the plan; not wired |
| `blocked` | cannot start until its dependency clears |
| `out-of-scope` | explicitly excluded from the foundation (see `G2_OUT_OF_SCOPE_IDS`) |
| `not-implemented` | no code, no ID |

## Campaign order (frozen)

Execution order for the campaign (per research directive; one controlled
change per result — reference → arm → compare → freeze → next):

```text
J1 Foundation Gate
  → K9 controls → K1 gist → K2 backbone/prediction → K4 precision
  → K5 observed+error fusion → K6 spatial error → K7 belief/opt/EMA
  → K8 robustness → K3 active policy
```

**Ordering disclosure (not silently resolved):** the master plan's PART XIII
groups the same families into stages (FOUNDATION: K9→K1→K4→K5/K6→K7;
ACTIVE PERCEPTION: K3→K3a; BELIEF+PREDICTION: K2-pred/K2-backbone; MEMORY;
ADAPTIVE; EVAL), i.e. it places K2 and K3 later than the execution order
above. The execution order here follows the research directive; the plan's
staged grouping is recorded verbatim in `Proposed_Plan.md` §PART XIII. If the
researchers want the plan's grouping, change this table deliberately and
record the decision — do not reorder silently.

## Common protocol (applies to every row unless noted)

| Field | Value |
|---|---|
| Dataset | ImageNet-100, HF `clane9/imagenet-100` pinned rev `0519dc2f…`, listing sha256 `0b0677…` (recorded in `runs/production_launch_manifest.json`) |
| Seed(s) | training seed 41; ≥5 seeds per arm (plan §K1: 5 seeds/arm); eval seeds 41–48 |
| Architecture revision | frozen baseline `rhan-nxa-clean-baseline @ 14771f1` + `RHANGen2Config` (schema_version frozen, `GATED_FLAGS` IDs immutable) + git SHA recorded per manifest |
| Training recipe | Gen-2: `recipe.py` (`build_adamw`, `AdamWGroups`, `PhaseRecipe`, warmup+cosine); J1/foundation arms: Gen-1 curriculum (`adv_curriculum.py`, TRADES/PGD-4, ε 0.031→0.094, β 2.0→2.5, w_trades 0.55) |
| Evaluation protocol | `evaluation/clean_and_robust.py::run_clean_and_robust` — norm-space ε, ≥5-seed floor, summary asserted vs per-seed CSV; K8 adds AutoAttack + EOT (`eot.py` wrapper) |
| Metrics | clean accuracy, robust accuracy (PGD; AA for K8), training loss, per-group \|dW\| + gradient norms + feature drift (J1 instrumentation), plus per-arm metrics listed below |
| Compute budget | plan estimate: ~20–40 min/run small-scale local; full matrix (13+ arms × 5 seeds) ~2–6 weeks local or ~1–3 days on 1×A100; AA×16-seed×13-arm ≈ 1–3 weeks on a node (`Proposed_Plan.md` §budget) |
| Checkpoint | `checkpoints/<experiment>_{best,rolling}.pth` (+ HF mirrors `FerrariKazu/rhan-nxa-checkpoints*`) |
| Manifest | `runs/<experiment>/manifest.json` via `noesis_vision/core/provenance.write_manifest` (config hash, seed, git SHA, dataset fingerprint) |
| Result / Decision | *not run* / *pending J1 gate* for every arm below |

## Family registries

### G2-K9 — Controls / matched baselines  [`implemented` (optimization control); campaign entry point]

| Experiment ID | Independent variable | Reference condition | Control condition | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K9-optimization` | optimizer/recalibration baseline | J1-frozen recipe | legal default control arm | `implemented` (recipe.py + test_recipe) | not run | pending J1 |
| (plan) param-matched control | parameter count matched | Gen-2 reference | matched-parameter control | `ready-to-integrate` (plan §K9) | not run | pending J1 |
| (plan) compute-matched control | compute matched | Gen-2 reference | matched-compute control | `ready-to-integrate` (plan §K9) | not run | pending J1 |

### G2-K1 — Gist  [`implemented` encoder; arm grid partially experimental]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K1-no-gist` | no gist | — | **Gen-1 parity default** (`is_control_arm()` = True) | `implemented` (gist.py + test_gist_parity) | not run | pending J1 |
| `G2-K1-gist-fixed` | fixed gist | `no-gist` | fixed vs none | `implemented` (FixedGistEncoder, shared patch_embed, 16 tokens) | not run | pending J1 |
| `G2-K1-gist-learned` | learned gist | `gist-fixed` | learned vs fixed | `implemented` (flag + encoder; integration at Gen-2 wiring) | not run | pending J1 |
| `G2-K1-gist-dropout` | gist dropout | `gist-fixed` | robustness to missing gist | `implemented` (consumed in gist.py + tests) | not run | pending J1 |
| `G2-K1c-gist-multi-scale` | multi-scale gist | — | — | `out-of-scope` (multi-scale pyramid; `G2_OUT_OF_SCOPE_IDS`) | — | deferred |

### G2-K2 — Backbone / prediction  [`not-implemented` as Gen-2 arms]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| (plan) `G2-K2-backbone`: frozen / low-LR / full fine-tune | trunk adaptation | Gen-2 reference | frozen-trunk arm | `ready-to-integrate` — optimizer groups exist in trainer; **no `G2-K2-*` ID registered in `flags.py`**; feature-drift monitoring now provided by the J1 harness | not run | pending J1 |
| (plan) `G2-K2-pred`: latent / masked / future / multi-step | prediction family | Gen-2 reference | λ_pred grid | `not-implemented` — no IDs, no code in slice | not run | pending J1 |

### G2-K3 — Active policy  [`out-of-scope` for foundation; campaign last]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K3-policy` (fixed → logit → MLP → recurrent → transformer → spatial map) | policy class | fixed policy | non-adaptive policy | `out-of-scope` (policy zoo excluded from slice; `G2_OUT_OF_SCOPE_IDS`) + `blocked` on K1–K8 | — | deferred |
| `G2-K3a-information-gain` | info-gain / uncertainty-conditioned policy | `G2-K3-policy` | — | `out-of-scope` + `blocked` | — | deferred |

### G2-K4 — Precision  [`implemented` per spec — six-arm family + legacy 1-U control]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K4-precision-const` | constant precision | default | — | `implemented` (precision.py, clamped [1e-4,1]) | not run | pending J1 |
| `G2-K4-precision-1minusu` | legacy `1-U` | const | **executable Gen-1 control** | `implemented` | not run | pending J1 |
| `G2-K4-precision-learned-scalar` | learned scalar | `1minusu` | — | `implemented` | not run | pending J1 |
| `G2-K4-precision-learned-spatial` | learned spatial map | `learned-scalar` | — | `implemented` | not run | pending J1 |
| `G2-K4-precision-channelwise` | channelwise | `learned-spatial` | — | `implemented` | not run | pending J1 |
| `G2-K4-precision-policyconditioned` | policy-conditioned | `channelwise` | — | `implemented` | not run | pending J1 |
| `G2-K4-precision-collapse-diag` | collapse diagnostics | — | instrumentation | `implemented` (diagnostic, not a training arm) | not run | pending J1 |

### G2-K5 — Observed + prediction-error fusion  [`implemented`]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K5-observed-error-off` / `-on` | observed-error pathway | `-off` | off vs on | `implemented` (ObservedFeatureFusion, test_update_net_v2) | not run | pending J1 |
| `G2-K5-fusion-concat` / `-gated` / `-FILM` | fusion operator | `gated` (default) | operator variants | `implemented` (registered; routing at Gen-2 wiring) | not run | pending J1 |

### G2-K6 — Spatial error  [`implemented` on/off; multiscale registered-only]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K6-spatial-error-off` / `-on` | spatial error pool | `-off` | off vs on | `implemented` (PositionSensitiveErrorPool, tests pass) | not run | pending J1 |
| `G2-K6-spatial-error-multiscale` | multiscale error | `-on` | — | `registered-only` — flag exists in `flags.py`, **no multiscale mechanism in the slice**; do not activate | not run | pending J1 |

### G2-K7 — Belief update / optimization / EMA  [`implemented`]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K7-belief-bounded` / `-adaptive` / `-recurrent` / `-unbounded` | belief update rule | `-bounded` | bounded vs variants | `implemented` (BeliefUpdaterV2 + adapt_magnitude) | not run | pending J1 |
| `G2-K7-optimization-AdamW` / `-SGD` / `-warmup` | optimizer family | `AdamW+warmup` (default) | optimizer arms | `implemented` (build_adamw, AdamWGroups, test_recipe) | not run | pending J1 |
| `G2-K7-ema-off` / `-on` | EMA target encoder | `-off` | EMA vs none | `implemented` (EMATargetEncoder, polyak sync + agreement/lag diagnostics, test_ema) | not run | pending J1 |

### G2-K8 — Robustness  [`implemented` reference + integration point]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `G2-K8-robustness-PGD` | PGD eval | protocol default | — | `implemented` (via `run_clean_and_robust`) | not run | pending J1 |
| `G2-K8-robustness-AutoAttack-EOT` | AutoAttack + EOT | `PGD` | attack strength | `implemented` (eot_pgd_attack, eot_sanity_loss, build_autoattack_loss_wrapper; AA optional import) | not run | pending J1 |
| `G2-K8-n_eot-1` / `autoattack-point` | EOT copies / AA integration | n_eot=1 deterministic | — | `implemented` | not run | pending J1 |

### J1 — Foundation Gate  [`harness-ready`; gate OPEN]

| Experiment ID | Independent variable | Reference | Control | Status | Result | Decision |
|---|---|---|---|---|---|---|
| `J1` (TRADES `--phase backbone_only` vs `--clean-only`) | objective: TRADES vs clean CE | TRADES arm | matched CE arm (only intended difference) | `harness-ready` — `cloud/gen1/colab_j1_foundation_gate.py` (selftest 12/12, dry-run verified, full-path verified INCONCLUSIVE without T4) | **not run on T4** | **OPEN** |

J1 measurements (both arms): backbone \|dW\|, classifier \|dW\|, backbone
gradient norm, classifier gradient norm, feature drift, training loss, clean
accuracy, robust accuracy where eval supports it, runtime, checkpoint, best
and final epoch, plus full reproducibility records (git, branch, seed,
dataset fingerprint, config, command, Python/PyTorch/CUDA/GPU, batch size,
workers, timestamp). Artifacts: `runs/j1_foundation/<run_id>/`.

### Additional planned families (not J1-blocking; enumerated for completeness)

| Family | Plan IDs | Status | Dependency |
|---|---|---|---|
| U_t calibration gate | `G2-K11-a` (plan thresholds: ECE ≤ 0.05, slope ∈ [0.9, 1.1]) | `out-of-scope`/`blocked` in slice — deferred until the foundation reports real numbers (AUDIT §5); plan re-registers it as prerequisite gate for consumer claims | J1 PASS + foundation numbers |
| Factorized predictive belief (JEPA-anything) | `G2-K10` (future: monolithic / multi-head / factorized / orthogonal predictors, factor disagreement → active gaze) | `out-of-scope` — explicitly excluded from J1 and the foundation reference; **note:** no `G2-K10-*` ID exists in `flags.py` yet (AUDIT names it; registration gap) | J1 PASS + K2 registered first |
| Spatial working / persistent memory | `G2-K7a`, `G2-K7b` | `out-of-scope` (`G2_OUT_OF_SCOPE_IDS`), K7b plan-experimental | K1–K8 ladder |
| Variable glimpse count / learned stopping | `G2-K8a` | `out-of-scope`, plan-experimental | K1–K8 ladder |
| Evaluation suites | `G2-E1` calibration/human-agreement, `G2-E2` shape-texture/occlusion suite, `G2-E3` distributional shift | `not-implemented` (plan only; Gen-0-era equivalents live under `phase4_analysis/`, `phase5_sdt/`) | after K7/K8 |
| Gen-3 | `G3-K1..K9` | `out-of-scope` — Gen-3 is a research proposal, not implemented | entire Gen-2 campaign |

## Dependency summary (run order)

| Experiment | Status | Ready? | Dependency |
|---|---|---|---|
| J1 Foundation Gate | harness-ready, not run | **ready to execute on Colab T4** | carrier-branch checkout + T4 + pinned dataset |
| K9 controls | implemented (optimization control) | blocked | J1 PASS |
| K1 gist | implemented (no/fixed/learned/dropout; multi-scale out) | blocked | K9 reference frozen |
| K2 backbone/prediction | not registered / not implemented | blocked | K1 frozen; needs `G2-K2-*` registration |
| K4 precision | implemented (7 IDs) | blocked | K2 (per directive order) |
| K5 observed+error fusion | implemented (5 IDs) | blocked | K4 frozen |
| K6 spatial error | implemented (on/off); multiscale registered-only | blocked | K5 frozen |
| K7 belief/opt/EMA | implemented (11 IDs) | blocked | K6 frozen |
| K8 robustness | implemented (4 IDs; AA+EOT wrapper) | blocked | K7 frozen |
| K3 active policy | out-of-scope for foundation | blocked | K1–K8 complete |
| K10 / K11 / E-suites / memory / Gen-3 | see above | blocked / deferred | per rows above |

## Scientific integrity (binding for every arm)

- One controlled change per experiment; reference → arm → compare → freeze → next.
  No uncontrolled accumulation of mechanisms.
- Never weaken a test to pass, alter metrics after seeing results, silently
  change the reference, mix experiments, compare unmatched compute, or compare
  unmatched parameter counts without documenting it.
- A synthetic smoke result is never a real-data result; an architecture
  proposal is never an implementation. Failures are results: no deletion of
  failed experiments, no overwriting previous manifests, no replacing
  inconvenient results.
- IDs are frozen: activating any non-default mechanism requires its exact
  pre-registered ID (`GATED_FLAGS`); anything in `REJECTED_OUTRIGHT` or
  `G2_OUT_OF_SCOPE_IDS` cannot be activated through this package.

---

*Audited 2026-10-06 against `gen2_foundation/` code + tests (52/52 passing),
`gen2_foundation/AUDIT.md`, `gen2_foundation/flags.py`, and
`noesis_vision/RHAN_NXA/docs/Proposed_Plan.md`. Registry updates must cite the
code change that changed a status.*

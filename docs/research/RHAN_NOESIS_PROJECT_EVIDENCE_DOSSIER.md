# RHAN / NOESIS — PROJECT EVIDENCE DOSSIER

**Generated:** 2026-09-15 · **Repo:** FerrariKazu/Adversarial-Cognitive-Model, branch `feature/rhan-next` @ `fe7a709` ("DDP launch: standalone rendezvous, NCCL watchdog, self-healing fallback")
**Method:** Every claim below is traceable to a file path (code), a checked-in result artifact (report/…, docs/…), or a fresh measurement executed during dossier preparation (marked MEASURED). Names were never trusted: each mechanism section cites the code that actually runs. Where evidence is absent, the entry says `UNVERIFIED`, `NOT FOUND`, or `UNKNOWN` instead of guessing.

**Scope caveat (read first):** two runtime sources are newer than any file in this repo — (a) the HF-hosted RHAN-NX roadmap state machine and (b) the per-seed eval CSVs on HF (`FerrariKazu/rhan-eval-sweep`). Where they supersede checked-in JSON, this is stated explicitly. The most recent observed runtime state (from the user's Colab session log, 2026-09-15) is: **sbr0–sbr3 done (all gates passed), sbr4 TRAINING (20-epoch fine-tune launched), ais_v2 and hpc_belief not started.**

---

## 0. How to read this dossier

| Marker | Meaning |
|---|---|
| `[CODE: path]` | claim verified by reading the cited source file |
| `[ART: path]` | claim verified from a checked-in result/verdict artifact |
| `[MEASURED 2026-09-15]` | fresh execution on this checkout (CPU, param counts, instantiation) |
| `[HF]` | artifact lives only on HuggingFace; not in this repo |
| `[LOG: colab 2026-09-15]` | observed in the user's latest Colab session log |
| UNVERIFIED / NOT FOUND | no evidence found; not inferred |

---

# 1. Repository inventory

Top-level layout (verified by directory listing):

```
Adversarial-Cognitive-Model/
├── rhan_core/                  # pillar-composable package (RHANNext)
│   ├── model.py                # RHANNext subclassing frozen RHANv12
│   ├── config/pillar_config.py # RHANNextConfig — the ONLY config object
│   ├── gaze/                   # info_gain_policy.py (v1), info_gain_policy_v2.py, halting.py
│   ├── precision/global_precision.py
│   ├── predictive_coding/      # hpc_level1.py, hpc_belief_level.py, feature_targets.py
│   ├── beliefs/                # vector_belief.py, structured_belief.py, relational.py,
│   │                           # evidence_decomposition.py, experimental/sbr_feasibility.py
│   ├── world_model/null_world_model.py
│   ├── optim/multi_group_optimizer.py
│   ├── ablation/               # matrix.py (A/B/C/D registry), runner.py
│   ├── lens/                   # capture.py (analysis/Lens capture)
│   └── docs/RESEARCH_CLUSTERS.md
├── phase1_training/            # FROZEN lineage + current trainer
│   ├── model_rhan_stl10_large.py  # RHANLargeSTL10 (TRADES-Large baseline arch)
│   ├── model_rhan_v10.py / v11.py / v12.py  # frozen chain; v12 is the frozen backbone
│   ├── train_rhan_next.py      # CURRENT trainer (all RHAN-NX runs)
│   ├── train_rhan_v12.py       # legacy trainer (v12 era)
│   └── dataset_stl10.py        # loaders + CutMix
├── phase2_attacks/             # attacks + eval
│   ├── pgd.py                  # PGD implementation
│   ├── eval_rhan.py            # FROZEN primary eval entrypoint
│   ├── eval_full_epsilon_sweep.py  # frozen sweep
│   ├── seed_sweep_comparators.py   # donor-row reuse + HF sync
│   └── eval_sweep_next.py
├── phase3_human_study/         # human psychophysics (Google Forms protocol)
│   ├── manifest.csv (100 stimuli) · form_structure.txt · data/responses_mapped.csv
├── phase4_analysis/            # analysis notes (.claude.md tier-1 ideas)
├── phase5_sdt/                 # signal-detection model comparison (7 systems incl. human)
├── scripts/                    # build_rhan_nx_report.py, consistency_assert.py,
│   │                           # comparator_registry.py, stage_state_machine.py,
│   │                           # measure_group_dw.py, build_rhan_nx_report.py
├── tests/                      # 36 test files detected (unit + gradient-flow + compat)
├── report/                     # per-seed CSVs, verdict JSONs, Gen-1 consolidated report
├── docs/                       # ARCHITECTURE.md, rhan_next_roadmap.json (rev 9),
│   │                           # NOESIS_FOUNDATION.md, stage3_preregistration.md,
│   │                           # Stage_E1-3_Analysis.md, research/RHAN_NOESIS_LITERATURE_CORPUS.md
├── cloud_setup/                # colab_notebook_noesis.py (state-machine driver), Kaggle_NOESIS.py
├── checkpoints/                # local .pth set (sbr0_smoke_best etc.; full set on HF)
├── tier1/, Paper/              # paper draft material (ACD_paper_v1.tex at repo root)
├── scratch/                    # ~40 diagnostic scripts (eval variants, probes, resync)
└── README.md, FINDINGS.md, NOESIS_IMPROVEMENT_CATALOG.md, RHAN-history.md,
    RHANv11.md, RHANarch.md, RHANfuture.md, IMPORTANT.md, COLLABORATING.md, ROADMAP.md
```

Component-by-component:

| Component | Path | Purpose | Actually executed? | Status |
|---|---|---|---|---|
| Frozen backbone v12 | `phase1_training/model_rhan_v12.py` | parent class of RHANNext; defines foraging loop contract | Yes — imported by `rhan_core/model.py` `[CODE]` | PRODUCTION (frozen) |
| Frozen eval | `phase2_attacks/eval_rhan.py` | all 16-seed matched evals run through it | Yes — every Colab eval invocation `[LOG]` | PRODUCTION (frozen; listed in roadmap `frozen_files` `[ART: docs/rhan_next_roadmap.json]`) |
| Current trainer | `phase1_training/train_rhan_next.py` | trains all RHAN-NX stages (D, E1–E3, sbr0–sbr4) | Yes — every training command in logs `[LOG]` | PRODUCTION |
| Pillar config | `rhan_core/config/pillar_config.py` | validate() enforces `enable_sbr=False` in legacy paths etc. | Yes `[CODE]` | PRODUCTION |
| AIS-v1 | `rhan_core/gaze/info_gain_policy.py` | relocated Eq.-II gaze + EntropyGatedHalting | Yes — active in every `--enable-ais` run | PRODUCTION (halting-only variant is the validated config) |
| AIS-v2 | `rhan_core/gaze/info_gain_policy_v2.py` | candidate-based one-step-lookahead EIG | Built, wired behind `ais_variant='info_gain_v2'` `[CODE: rhan_core/model.py]`; **no training run yet** | IMPLEMENTED, UNVALIDATED (roadmap `ais_v2` stage `not_started`) |
| HPC L1 (pixel) | `rhan_core/predictive_coding/hpc_level1.py` | edge-map-target prediction error at foveal-crop tap | Yes — D, E1–E3, sbr runs | PRODUCTION |
| HPC belief (D3) | `rhan_core/predictive_coding/hpc_belief_level.py` | predict belief_{t+1} from belief_t | Wired behind `hpc_target='belief'` `[CODE]`; **no training run yet** | IMPLEMENTED, UNVALIDATED |
| SBR | `rhan_core/beliefs/structured_belief.py` | slot attention (16 slots × 512, 3 iters) + relational + evidence | Yes — E2 (legacy wiring), sbr0–sbr4 (spatial wiring) | PRODUCTION |
| Ablation registry | `rhan_core/ablation/matrix.py`, `runner.py` | A/B/C/D single source of truth | Yes — Stage 2/3 evals via `--ablation-matrix` | PRODUCTION |
| Multi-group optimizer | `rhan_core/optim/multi_group_optimizer.py` | per-group SGD + per-group clip (starvation fix) | Yes — sbr0+ runs | PRODUCTION |
| World model | `rhan_core/world_model/null_world_model.py` | NullWorldModel safe passthrough | Always instantiated `[CODE]` | SCAFFOLD (Pillar 4; `enable_iwm` must stay False, enforced by `validate()`) |
| Lens capture | `rhan_core/lens/capture.py` | trajectory/perception capture for analysis | Exists; used by scratch eval scripts (e.g. `scratch/eval_lens_e1.py`) `[CODE]` | SUPPORT |
| Report builder | `scripts/build_rhan_nx_report.py` + `scripts/consistency_assert.py` | consolidated report w/ donor byte-verification | Yes; byte-verify fixed 2026-09-15 (this session: aggregated rows can't be donor-verified; verify moved to load time; 17/17 tests pass) | PRODUCTION |
| State machine | `scripts/stage_state_machine.py` | decides next RHAN-NX action; HF-synced | Yes — drives Colab notebook `[LOG]` | PRODUCTION |
| Colab driver | `cloud_setup/colab_notebook_noesis.py` | full ladder runner (train→eval→report→advance) | Yes | PRODUCTION |
| Legacy model files | `phase1_training/model_rhan_v10/11.py`, `train_rhan_v12.py` | lineage only | Imported transitively (v10 `foveal_sample`, v11 streams) `[CODE]` | FROZEN |
| RHANLargeSTL10 | `phase1_training/model_rhan_stl10_large.py` | baseline architecture class | Yes — the TRADES baseline checkpoint's arch | PRODUCTION (baseline) |
| SBR feasibility probe | `rhan_core/beliefs/experimental/sbr_feasibility.py` | pre-SBR machinery feasibility | Smoke-tested 2026-08-11, PAUSED by design `[ART: roadmap stages['2'].sbr_feasibility_probe]` | EXPERIMENTAL, frozen-out |
| Notebooks | `T4x2.ipynb` + `cloud_setup/*.py` | cloud runners | Yes | SUPPORT |

Tests: 36 test files detected. Names verified in repo include `test_comparator_reuse_integrity.py` (17 tests passing as of 2026-09-15), `test_pillar_scaffold_import.py`, `test_config_backward_compat.py`, `test_gradient_flow.py`, `test_hpc_gradient_flow.py`, `test_hpc_disable_backward_compat.py`, `test_hpc_optimizer_group_resume.py`, `test_ablation_matrix.py`, `test_multi_group_optimizer.py`. Exact pass-status of the full suite on this checkout: UNVERIFIED (not run whole; the cited subset passed at their recorded dates).

---

# 2. Current RHAN architecture (as implemented)

Two coexisting architectures matter. **(a) the frozen TRADES-Large baseline** (`RHANLargeSTL10`, 55,622,347 params `[MEASURED 2026-09-15]`) whose checkpoint is `rhan_stl10_large_pseudolabel_best.pth`, and **(b) RHANNext** — every experimental result in this dossier. RHANNext subclasses frozen `RHANv12`; with default config its state dict is byte-identical to v12's (design contract enforced by `tests/test_config_backward_compat.py`; `[CODE: rhan_core/model.py]`, `RHANNext(default) params = 75,440,469 = RHANv12 [MEASURED 2026-09-15]`).

## 2.1 Shared trunk (frozen v12 chain: large → v10 → v11 → v12)

- **Input resolution:** 96×96×3 (STL-10 native) `[CODE: model_rhan_stl10_large.py; dataset_stl10.py]`.
- **Preprocessing:** normalize with `STL10_MEAN=(0.4467,0.4398,0.4066)`, `STL10_STD=(0.2242,0.2215,0.2239)` `[CODE: dataset_stl10.py]`. Epsilons in eval are applied **directly in normalized space** (norm-space convention), verified per-seed by printed channel-bound checks in every eval log `[LOG: eval_rhan.py output]`.
- **Backbone stem:** 4-stage SE-conv stem 3→128 (96×96) →512 (48×48) →1024 (24×24, Dropout2d 0.1) →768 (12×12), plus a 1×1/s8 shortcut; residual add `[CODE: WideSEConvStemLarge]`.
- **Tokeniser:** 1×1 conv → GroupNorm(8) → GELU; 144 patches (12×12) + CLS; learnable pos-embed, dim 768 `[CODE: PatchTokeniserLarge]`.
- **Ventral/dorsal split:** the 768-dim token vector is split **channel-wise** 384+384 (not token-wise); each half runs through its own `nn.TransformerEncoder` of **8 layers** (d_model 384, 6 heads, ff 1536, norm_first, GELU), then concatenated back to 768. Both streams run under `torch.utils.checkpoint` `[CODE: RHANLargeSTL10._run_transformer]`. NOTE: "ventral/dorsal" here is a channel split with no anatomical specialisation in the loss or data — see §13 audit.
- **Recurrent feedback:** `RecurrentFeedbackLarge` runs `num_recurrent_steps=2` top-down loops: tokens→12×12 spatial → 1×1 conv feedback → sigmoid gate → stem-feature modulation → `PredictiveCodingLayerLarge` (local minus global-predictor error, sigmoid error-gate, learned scalar `error_scale`) → re-run the full transformer. This is a *v12-trunk* predictive-coding-flavoured layer, distinct from the HPC pillar (§2.4) `[CODE: model_rhan_stl10_large.py]`.
- **Parameter count (whole frozen v12):** 75,440,469 `[MEASURED 2026-09-15]`. ⚠ The v12 docstring claims "~63.4M (identical to v11 minus halt_net)" and the E3 donor note mentions "76M"/"81M" elsewhere — all four figures are mutually inconsistent; the measured number is authoritative. Logged as a contradiction in §18.
- **Motion encoder + TDV head:** `MotionEncoderLarge` (6→512 via convs) and `TDVProjectionHeadLarge` (768→512 MLP) **exist in the state dict** (verified `hasattr` True for both `[MEASURED]`) but are **never invoked by the v12 forward path** — they are dead parameters inherited from the large model's TDV/video ambitions. They receive no gradients from any current loss. Training-only vs inference-only: neither; they are unused.

## 2.2 RHANNext additions (pillar layer, `rhan_core/model.py`)

All pillars are gated by `RHANNextConfig` (34 fields verified `[MEASURED]`). The active config for D and the SBR ladder is `RHANNextConfig([AIS,HPC(L=1),SBR])` — printed by eval at checkpoint load `[LOG]`.

**Foraging loop (`_forage`)** — shared by AIS/HPC/SBR paths; mirrors v12 exactly when pillars off:
1. Peripheral pass: full-image stem+tokens+transformer+feedback → CLS 768 → `peripheral_proj` → s (B,512); gaze init `action_init(s)` (B,2).
2. Parafoveal stream: once per forward (B,512).
3. Per step t ∈ {0..3} (T = `max_foraging_steps` = 4 for D; 6 for E3): `foveal_sample(x, a, fovea_size=48)` (grid_sample-based differentiable crop, from frozen `model_rhan_v10.foveal_sample`) → `foveal_stream` (B,512) → learned foveal/parafoveal gate alpha → `generative_prior(s)` predicts (B,3,48,48) crop → `image_precision` computes **Pi_D** (bounded image-space precision, B,) from crop-vs-prediction error → belief update `s = (1-π_d)·s + π_d·combined_feat`.
4. SBR (when on): slot attention over the **spatial stem map** (B,144,768) flattened → StructuredBeliefState → pooled/evidence replaces s for the classifier path.
5. HPC (when on): per-step prediction error at the foveal-crop tap (attached; not detached — loss reaches predictor).
6. Belief accumulation: `weighted_belief += cont·s` where cont is 1.0 (v12), or σ(softness·(u−threshold)) under AIS halting; final belief = weighted mean → `belief_unproj` → 768 → classifier.
7. Gaze update (v12 path): Eq.-II gradient blend `g = λ·∇_a R(x,a) + (1−λ)·∇_a ‖f(a)−P(s)‖`, λ=0.5 default, normalized, step 0.20+0.30·π_d, clamp ±0.9 `[CODE: model_rhan_v12.py]`.

## 2.3 AIS (Pillar 2) — as implemented

`InformationGainGazePolicy` (v1) wraps the frozen machinery by plain reference (`foveal_sample`, `generative_prior`, `foveal_stream`, `prior_predictor`, `modulate_step_size`) so no state-dict duplication `[CODE: rhan_core/model.py]`. Components:

- **Gaze policy v1** = the *relocated v12 Eq.-II gradient ascent on current prediction error*; `step_net` is identity-initialised (a learned residual on the classic update). Roadmap explicitly labels it a "replication-under-refactor control, NOT a test of genuine information-gain gaze" `[ART: roadmap stages['1'].run_identity]`.
- **EntropyGatedHalting** (`rhan_core/gaze/halting.py`): halts when uncertainty u < threshold (0.35, softness 8.0). No step-count penalty. Under SBR-0..4, u = slot-attention entropy; under SBR-4, u = evidence-decomposition uncertainty (`structured_belief.py` writes `evidence['uncertainty']` into the belief's operative uncertainty slot `[CODE]`); without SBR, u = 1−π_d proxy. Halting is soft: continuation weight, not hard exit (hard per-sample early exit DEFERRED `[ART: roadmap deferred_increments]`).
- **GlobalPrecisionModulator** (`rhan_core/precision/global_precision.py`): wraps the shared `image_precision` module; exactly **3 consumers** — gaze step size, halt threshold, recon loss weight. Gain param is a single scalar (1 param `[MEASURED]`).
- **Precision-modulated recon weight ("recon-mod")**: DISABLED in every validated run (`--no-ais-precision-recon`), per the 2026-08-07 isolation verdict (see §7). It exists, is wired, and is deferred.
- **AIS-v2** (`info_gain_policy_v2.py`): K=4–8 candidate gaze points sampled around the highest-uncertainty region; one-step-lookahead expected-information-gain scoring via a `candidate_head` trained with TD pairs (predicted candidate surprise vs next-step observed surprise; the only gradient path into the head) `[CODE: rhan_core/model.py eig_pairs wiring]`. Status: code complete, **never trained**; roadmap `ais_v2` stage `not_started` (runtime state 2026-09-15 confirms).
- **Gaze-shift metrics** (training diagnostics): `RHANNextEpochDiagnostics` — gaze_shift_total_mean, per-sample effective steps std, frac_halted, per-class Pi_D top-2 `[CODE/ART: roadmap stage 1 execution_plan]`. Gate thresholds calibrated: gaze ≥ 0.05, std ≥ 0.02, frac_halted ≥ 0.02.

## 2.4 HPC (Pillar 1) — as implemented

- **`HPCLevel1`** (`hpc_level1.py`): ONE level. Tap point = **foveal crop** (documented single tap); predictor predicts the `edge_map` feature target (non-learnable EdgeMapExtractor); returns (prediction, error, error_map); **error NOT detached** — gradient reaches predictor params; hard-asserted by `tests/test_hpc_gradient_flow.py`.
- Loss slot: `w_hpc = 0.10`, a **separate** weight from `w_recon` (config `hpc_error_weight`); optimizer starvation history in §6/§18.
- **`HPCBeliefLevel`** (`hpc_belief_level.py`, D3): predicts belief_{t+1} from belief_t with delayed-by-one wiring; input attached, target detached. Implemented, not trained.
- **What HPC is NOT:** it is not a hierarchical multi-level predictive-coding stack (levels>1 rejected by config), not a biologically-plausible local-learning implementation (no Hebbian/surprise-propagation learning rule), and not a full free-energy active-inference loop. It is a single auxiliary reconstruction-style loss on an edge-map target. The repo's own literature corpus marks the Rao & Ballard connection as "Adapted … differs materially" `[ART: docs/research/RHAN_NOESIS_LITERATURE_CORPUS.md [33]]`.
- The *trunk's* `PredictiveCodingLayerLarge` (§2.1) is a separate, older feedback-error gate — unrelated to the HPC pillar's loss accounting.

## 2.5 SBR (Pillar 3) — as implemented

`StructuredBeliefState` (`rhan_core/beliefs/structured_belief.py`): slot attention with **K=16 slots, slot_dim=512, iters=3, 4 heads**; GRUCell + MLP refinement; softmax over **slots** (competition over inputs); slots init μ+σ·ε. Two wiring modes:

- **legacy (E2):** slots bind over the single pooled vector (N=1); no input_proj; byte-compatible with E2 checkpoints.
- **SBR-0..4 (spatial):** slots bind over the stem map (B,144,768) via `input_proj` Linear(768→512); temporal `step_gate` blends pooled across foraging steps; slots carry across steps via `sbr_state`.
- **SBR-3 (relational):** `SlotRelationalLayer` — attention-based inter-slot message passing; slots are updated post-attention; `relation_attn` (B,K,K) exposed.
- **SBR-4 (uncertainty):** `EvidenceDecomposition` with per-slot heads — `shape_head`, `texture_head`, `spatial_head` (verified by instantiation `[MEASURED]`) — producing hypothesis/supporting/contradictory/uncertainty output; **pooled evidence replaces the belief** fed to the classifier (`s = sbr_out['evidence']['pooled_evidence']` `[CODE: rhan_core/model.py]`); the belief's `uncertainty()` (hence the AIS halting gate) becomes the decomposition's uncertainty.
- SBR-0 freeze: `freeze_backbone_for_sbr0=True` freezes every param outside `structured_belief.*` **at the model level** (not just trainer) `[CODE: rhan_core/model.py]`; 337 tensors frozen, only slot modules trainable `[MEASURED]`.

## 2.6 Dimensional summary (verified)

| Quantity | Value | Evidence |
|---|---|---|
| Input | (B,3,96,96) | CODE |
| Stem out | (B,768,12,12) | CODE |
| Tokens | 144+1 × 768 | CODE |
| Transformer | 2×8 layers, d=384/6h each | CODE |
| Belief s | (B,512) | CODE |
| Foveal crop | (B,3,48,48) | CODE |
| Foraging steps | T=4 (D/E1/E2 default; E3: T=6) | CODE+ART |
| RHANNext default params | 75,440,469 | MEASURED |
| + AIS+HPC+SBR-4 params | 84,470,149 (structured_belief 7,773,453; hpc_level1 1,223,265; gaze_policy 32,961; precision_modulator 1) | MEASURED |
| Frozen under SBR-0 | all except `structured_belief.*` (337 tensors) | CODE+MEASURED |
| Trainable groups (Gen-0 optimizer) | backbone / hpc (×6.67 lr) / sbr / relational / evidence / ais_v2 | CODE+ART roadmap gen0 |

## 2.7 Losses (trainer `train_rhan_next.py`)

Active weights printed at training start: **trades=0.55, recon=0.1, hpc=0.1** `[LOG]` — i.e. TRADES robust objective + differentiable reconstruction MSE (`get_reconstruction_loss`, attached since the v12 fix) + HPC edge-map error. No supervised auxiliary losses (foraging-consistency/precision-calibration/halt-efficiency were deleted in v12 `[CODE: model_rhan_v12.py header]`). SBR stages add no standalone slot loss in sbr0/1 (representation learned implicitly through the classifier); relational/evidence heads in sbr3/4 train through the TRADES path with their own optimizer groups.

## 2.8 Known limitations of the implementation

1. Dead parameters (motion_encoder, tdv_head) inflate checkpoints (~10M+ params unused) — MEASURED existence, unused by forward.
2. Halting is soft-only; `trajectory['steps']` always equals max_steps, so "adaptive computation" claims are about continuation *weights*, not wall-clock steps.
3. Recurrent feedback (trunk) and HPC pillar are two independent predictive-coding-ish mechanisms; the paper must not merge them.
4. Channel-split "ventral/dorsal" streams are architecturally two half-width transformers; no task differentiation exists in code.
5. Cross-run GPU nondeterminism (~1–1.5 pp, grid_sample/attention backward) is documented and gates verdict interpretation `[ART: roadmap validation_protocol.known_caveat]`.
6. eval_rhan.py enforces a ≥5-seed floor by design; some historical rows (n=2, n=3) exist only via the sweep-comparator path.

---

# 3. RHAN lineage (from code + repo history docs)

| Generation | Primary change | Evidence | Result anchor |
|---|---|---|---|
| RHAN-Large (v≤9 era) | ViT-B-scale stem+split transformer at 96×96; TRADES training on 5K labeled + ~46K pseudo-labeled | `[CODE: model_rhan_stl10_large.py]`, baseline ckpt | clean 54.81±2.35, PGD-100@0.094 24.23±1.94 (16 seeds) `[ART: e1_verdict]` |
| v10 | tripartite active-inference: foveal/parafoveal/peripheral + Eq.-II feature-space gaze; foveal_sample | `[CODE: model_rhan_v10.py]` | UNVERIFIED (pre-history; no CSVs in repo) |
| v11 | adds halt_net (thermodynamic), streams formalized; **v11 bug**: recon loss detached (silent no-op) | `[CODE: model_rhan_v11/v12 headers]` | null_ablation_v11: 31.56±2.88 @0.094 (8 seeds, curriculum identical) `[ART: roadmap stage1 step_b_full]` |
| v12 | halt_net DELETED (fixed T=4); Eq.-II v12 λ-blend gaze (recon-guided); aux losses deleted; recon re-attached | `[CODE: model_rhan_v12.py]` | is the frozen backbone of everything below |
| RHANNext (scaffold, Stage 0) | pillar-composable subclass; default byte-identical to v12 | `[CODE: rhan_core/model.py]` + scaffold tests | v12 ckpt loads 1:1 `[ART: roadmap stage 0]` |
| Stage 1 = B: AIS-v1 (halting-only) | relocated Eq.-II + entropy halting; recon-mod OFF | isolation verdict 2026-08-07/09-11 | clean 49.4±3.47, @0.094 32.21±2.74 (8 seeds, PGD-50), masking-free; +8.5pp vs baseline, n.s. at 2σ `[ART: stage1_verdict]` |
| Stage 2 = C: HPC-only | + edge-map HPC (w=0.10), AIS OFF | starvation fix 2026-08-13 | clean 55.2±3.67, @0.094 27.73±2.28 (5 seeds), CROSSOVER REAL (+7.33 vs baseline); C-vs-B −4.8 n.s.; masking-free `[ART: stage2_verdict]` |
| Stage 3 = D: AIS+HPC | B+C combined, 60-epoch curriculum from B | | clean 54.25±3.10, @0.094 34.38±1.94 (8 seeds PGD-100); D-vs-A +11.55 pp, 2σ=7.95 → **CROSSOVER REAL** `[ART: stage3_verdict]` |
| Stage 4 E1 | D + recon-mod ON | | clean +3.10 (16/16 seed wins) but adv −0.90 vs D (n.s.) → recon-mod = clean-only lever `[ART: e1_verdict]` |
| Stage 4 E2 (legacy SBR) | D + SBR (legacy N=1 wiring), Gate-0-then-finetune | | clean −9.90; adv −0.60 vs D (n.s.); +9.19 vs baseline REAL `[ART: e2_verdict]` |
| Stage 4 E3 | D with T=6 foraging | | clean +2.29, adv −3.25 vs D (both n.s.) → T=6 deferred `[ART: e3_verdict]` |
| RHAN-NX ladder (sbr0–4, D2, D3) | re-architected SBR w/ spatial binding + swap tests | roadmap rev 9 + runtime state | sbr0–3 done; sbr4 training (see §5) |

Do NOT read as a clean chronological story: v10/v11 detailed results are not in the repo (marked UNVERIFIED above); the roadmap's stage ordering is authoritative for the validated chain only.

---

# 4. NOESIS framework — mechanism status

NOESIS, per `docs/NOESIS_FOUNDATION.md`, is the umbrella frame ("A Framework for Biologically Inspired Perceptual Intelligence") whose pillars map onto mechanisms. The doc itself states: "NOESIS is not a claim about biological fidelity" (line ~486). Status per mechanism, **not collapsed**:

| Mechanism | Status | Evidence |
|---|---|---|
| AIS (v1, halting-only) | **IMPLEMENTED + VALIDATED** (Stage-1 8-seed verdict) | `[CODE]` + `[ART: stage1_verdict]` |
| AIS recon-mod (precision-scaled recon weight) | **IMPLEMENTED, DEFERRED** (isolation-confirmed driver of Pi_D reordering; its own E1 run exists: adv-neutral, clean-positive) | `[ART: isolation_verdict + e1_verdict]` |
| AIS-v2 (genuine EIG lookahead) | **IMPLEMENTED (code), UNVALIDATED** — never trained | `[CODE: info_gain_policy_v2.py]`, runtime state |
| HPC (level 0/1, edge-map) | **IMPLEMENTED + VALIDATED** (Stage-2; standalone C crossover real) | `[CODE]` + `[ART: stage2_verdict]` |
| HPC belief-level (D3) | **IMPLEMENTED (code), UNVALIDATED** — never trained | `[CODE: hpc_belief_level.py]`, runtime state |
| SBR (slot belief, spatial binding) | **IMPLEMENTED + VALIDATED-STRUCTURALLY** (sbr0 gate) + ladder results for sbr2/3 | `[CODE]` + `[ART: sbr0 verdict]` + HF CSVs |
| SBR legacy wiring (E2) | **IMPLEMENTED + EVALUATED** (negative-for-clean result) | `[ART: e2_verdict]` |
| SBR relational evidence | **IMPLEMENTED + EVALUATED** (sbr3 done 2026-09-15) | `[CODE]` + `[LOG]` |
| SBR uncertainty-first-class | **IMPLEMENTED (code), VALIDATION IN PROGRESS** (sbr4 training) | `[CODE]` + `[LOG]` |
| Evidence decomposition | **IMPLEMENTED** (shape/texture/spatial heads verified) | `[MEASURED]` |
| Belief representation (vector) | **IMPLEMENTED** (VectorBeliefState, uncertainty proxy 1−π_d) | `[CODE]` |
| Belief representation (structured) | **IMPLEMENTED** | `[CODE]` |
| Temporal state (slot carry-over, step_gate) | **IMPLEMENTED** | `[CODE]` |
| Memory (episodic / SchemaMemory / temporal persistence) | **NOT IMPLEMENTED** (cluster c3 NOT STARTED; no `rhan_core/memory/`) | `[ART: roadmap research_clusters]` |
| IWM (internal world model) | **SCAFFOLD ONLY** (NullWorldModel passthrough; `enable_iwm` must stay False, enforced) | `[CODE]` + `[ART]` |
| Distributional belief (μ,Σ), multi-hypothesis workspace | **NOT IMPLEMENTED** (cluster c6) | `[ART]` |
| Self-monitoring / selective classification / calibration head | **NOT IMPLEMENTED** (cluster c7; SDT analysis exists as *offline evaluation*, not a model mechanism) | `[ART]` |
| Concept bottleneck | **NOT IMPLEMENTED as mechanism.** The term appears in NOESIS docs re Finding-9 (car/truck/horse collapse) as a *diagnostic* (Pi_D per-class analysis). No CBM loss/head exists in code. | NOT FOUND in code; `[ART: NOESIS_FOUNDATION.md]` |
| Active inference (full free-energy loop) | **PARTIALLY: inspiration only.** π_d-weighted belief update + gaze error-gradients exist; no generative-model free-energy objective, no variational bound. | `[CODE]` + corpus honesty convention |
| Uncertainty (halting input) | **IMPLEMENTED** (three sources over lineage: 1−π_d → slot entropy → evidence uncertainty) | `[CODE]` |
| Lens (analysis framework) | **IMPLEMENTED as tooling** (capture + scratch analyses), not a model component | `[CODE: rhan_core/lens/]` |

---

# 5. Generation-0 experiments (the validated record)

Protocol common to all: STL-10; seeds as listed; PGD (norm-space eps applied directly; per-channel bound check printed per cell); n=300/seed; eval via frozen `eval_rhan.py`; significance = pre-registered **δ > 2σ_combined** (deliberately conservative); robustness-masking check = PGD-50→100 gap ≤ 1.0 pp. Curriculum (when 60-epoch): 1–20 @0.031, 21–40 @0.062, 41–60 @0.094, byte-identical to `train_rhan_v11.py`.

### Baseline (A): trades_large_baseline
- Arch: RHANLargeSTL10 (55.6M), TRADES loss (w=0.55), pseudo-label pretraining on 100K unlabeled (conf≥0.65) `[CODE]`.
- Results (16 seeds, PGD-100): clean **54.81±2.35**, @0.094 **24.23±1.94**, d′ 1.75/0.64 `[ART: e1_verdict]`. PGD-50 leg 23.71 (8-seed, stage1) / 20.4 (5-seed, stage2 session — cross-session nondeterminism caveat applies) `[ART]`. Masking-free (gap 0.13 pp) `[ART: stage1 masking_check]`.

### Stage 1 (B): AIS-v1 (halting-only variant)
- Hypothesis: relocating Eq.-II into the pillar interface + entropy-gated halting preserves the v11 null-ablation robustness effect. Intervention: `--enable-ais --no-ais-precision-recon`, 60 epochs, from baseline ckpt. Control: baseline. Pre-registered smoke gate fired on Pi_D top-2 (car/airplane vs car/truck) → mechanism-isolation arms (isoA halting-off / isoB recon-mod-off) → verdict branch (2): **recon-mod is the driver; halting exonerated** → Step B trained halting-only. isoA sufficiency recapture (2026-09-11): RECON_MOD_SUFFICIENT, boundary-level margin caveat (0.0002–0.0013) recorded honestly `[ART: isolation_verdict]`.
- Result (8 seeds, PGD-50): clean 49.4±3.47, @0.094 **32.21±2.74**, d′@0.094 1.0075. +8.5 pp vs baseline; 2σ threshold 8.84 → **positive but NOT significant**. Masking: gap 0.04 pp → genuine. Final; no third extension `[ART: stage1_verdict]`.
- Artifact caveat (recorded, important): the evaluated checkpoint is the **epoch-60 final** model (peak-val 54.05 weights lost to a session wipe); 54.05 must NOT be cited as its accuracy `[ART: eval_target_note]`.
- **NEGATIVE RESULT embedded:** the smoke's Pi_D reordering itself (a mechanism caused a real, unintended behavior change) — reported, attributed, and gated rather than buried.

### Stage 2 (C): HPC-only
- Hypothesis: single additive HPC edge-map loss improves robustness without AIS. Intervention: `--enable-hpc --hpc-num-levels 1 --w-hpc 0.10`, 60 epochs from B.
- Critical incident (documented): optimizer starvation froze HPC head learning (error flat 0.6904→0.6911; ~1000× attenuation via global clip). Fix: two-group SGD (backbone lr 0.003, hpc lr 0.02 = ×6.67) + per-group clip; pre-flight |dW| showed 201.8× movement improvement `[ART: optimizer_configuration_fix_2026_08_13]`.
- Gate amendment 2026-08-16: Pi_D criterion replaced (truck is the marginal-flicker class; original top-2 criterion was re-scoring healthy runs as failed). Original preserved in git history `[ART: gate_amendments]`.
- Result (5 seeds, PGD-50): clean 55.2±3.67, @0.094 **27.73±2.28**, d′ 0.6147. vs baseline +7.33 (2σ=5.16) → **CROSSOVER REAL**. C-vs-B: −4.8 (2σ=5.99) → HPC alone does not add over AIS at this n. Masking-free (0.33 pp) `[ART: stage2_verdict]`.

### Stage 3 (D): AIS-v1 + HPC
- B+C joint, 60-epoch curriculum from B. Result (8 seeds, PGD-100): clean 54.25±3.10, @0.094 **34.38±1.94**; D-vs-A +11.55 (2σ=7.95) → **CROSSOVER REAL**; D-vs-B +1.38 n.s. (B only had n=2 donor rows in that CSV — informational) `[ART: stage3_verdict]`.
- Note: D's most-referenced numbers in the ladder era come from the 16-seed Stage-4 CSVs: clean **54.96±2.37**, @0.094 **34.02±3.24**, d′ 1.79/1.08 — the same weights re-evaluated at n=16 with donor reuse `[ART: e1/e2/e3 verdicts; HF CSV]`. The 8-seed and 16-seed figures must not be mixed without labeling.

### Stage 4 E1: D + recon-mod
- 16 seeds, PGD-100. E1: clean **58.06±2.38**, @0.094 33.12±2.64, d′ 1.06. Paired vs D: clean +3.10 (**E1 wins clean 16/16 seeds**); adv −0.90 (D wins 10/6); both cross-overs vs baseline REAL (9.79 / 8.89 pp) `[ART: e1_verdict]`. Interpretation: recon-mod buys clean, is robustness-neutral — consistent with the isolation story, opposite to the Defense-GAN-motivated hope (corpus [20] notes this explicitly).

### Stage 4 E2: D + SBR (legacy wiring)
- Gate-0: SBR trained clean-only on frozen D backbone; then joint fine-tune (60-epoch curriculum shape). 16 seeds PGD-100: clean **45.06±3.30** (−9.90 vs D), @0.094 33.42±2.93 (−0.60 vs D, n.s.; +9.19 vs baseline REAL) `[ART: e2_verdict]`. Verdict text in roadmap: "SBR does not improve on D in Generation 0 form."
- ⚠ E2b appears in roadmap training notes as the gate-collapsed variant discovered at "massive compute cost" (sbr1 gate text references E2b's collapse failure mode) `[ART]`. No standalone E2b CSV found in repo — NOT FOUND beyond that reference.

### Stage 4 E3: D + T=6 foraging
- 16 seeds, PGD-100: clean 57.25±2.54, @0.094 30.77±2.98, d′ 0.875. Paired vs D: adv −3.25 (D wins 13/2/1), clean +2.29 (E3 wins 14/2). All differences n.s. at 2σ. T=6 DEFERRED; D remains headline `[ART: e3_verdict]`.

### RHAN-NX ladder (current generation; runtime state 2026-09-15)
Single-source-of-truth state machine + HF roadmap. Stages (config and gates from roadmap rev 9):

- **gen0 (gate_passed):** multi-group optimizer infrastructure; tests pass + SBR slot |dW| pre-flight in learnable regime `[ART]`.
- **sbr0 (gate_passed, 2026-09-11T03:09Z):** frozen D backbone, clean-only, slots over spatial stem map; 40-epoch ceiling; ALL FOUR criteria passed: occupancy entropy 0.9839 (≥0.6); pairwise-cosine slope −0.00179 (declining, 0.9999→0.8922); per-slot probes 16/16 above 25% floor (min_required 4); everything-slot ablation retained 101.6% (victim slot norm share 0.0672) `[ART: sbr0 verdict]`. Caveat recorded by us, not the gate: probe accuracy ~0.50 across slots ≈ 5× random (10-way) — "above floor" is unambiguous, but slot *specialization* is not demonstrated by these metrics (probes near-uniform). The gate did not require specialization above diversity.
- **sbr1 (gate_passed after amendment):** clean-only joint fine-tune from sbr0; clean **62.49**. Original symmetric-band gate FAILED this run for *over*-performing (+7.5 above D's 54.96) — a formula artifact; amended 2026-09-11 to one-sided collapse detector (clean ≥ D−3pp) `[ART: sbr1 gate + amendment]`.
- **sbr2 (done):** standard 3-phase adversarial curriculum from sbr1. 16-seed PGD-100 (user-provided CSV across sessions): clean **62.42±2.45** (seeds 41–51 present, n=11 in the merged CSV at last user paste; seeds 52–56 pending), @0.094 **23.33±2.50**, adv d′ 0.58. vs D: clean +7.5, adv **−10.7**. (The roadmap's own pre-registered eval plan requires 16 matched seeds; treat n as in-flight.)
- **sbr3 (done 2026-09-15):** relational evidence fine-tune, 20 epochs fixed ε=0.094 from sbr2. 16-seed PGD-100 complete: clean **59.44±2.78**, @0.094 **27.69±3.11**, d′ 0.82. vs D: clean +4.5, adv −6.3. vs sbr2: adv **+4.4 recovered (~41% of the conceded gap)**. vs baseline: +3.46 pp, 2σ=7.32 → **positive but NOT significant** `[LOG: full eval table + structural assertion passed]`. PGD-50 leg: 28.83±2.44 → PGD-50→100 gap −1.1 pp (50-step number *lower* than 100-step — within the documented ~1.5 pp nondeterminism floor; NOT evidence of masking, but not clean under the ≤1.0 bar either — flag for the report).
- **sbr4 (TRAINING as of 2026-09-15):** `--sbr-stage uncertainty --fixed-eps 0.094 --max-epochs 20` from sbr3 `[LOG]`. First rung where the SBR-4 evidence-uncertainty replaces slot-entropy as the AIS halting signal — the mechanism interaction the ladder was built to expose.
- **ais_v2 (D2), hpc_belief (D3): not started.** Both are SWAP tests vs D from B's checkpoint (2026-09-08 amendment: base = `rhan_next_ais_v1_halting_only_best.pth`, NOT the pseudolabel ckpt, to keep interventions mechanism-pure).

---

# 6. Exact Generation-0 lessons (evidence table)

Categories strictly evidence-based. "Confidence": HIGH = multiple independent artifacts agree; MED = single-source but precisely recorded; LOW = single mention.

| Mechanism | Evidence | Result | Interpretation | Confidence | Decision |
|---|---|---|---|---|---|
| TRADES backbone + pseudo-labels | stage1/2/4 CSVs; 16 seeds | clean 54.8, adv 24.2 | the floor everything is measured against | HIGH | KEEP (as baseline) |
| AIS-v1 halting-only | stage1_verdict + masking checks | +8.5 adv pp (n.s.), masking-free | mechanism survives refactor; effect below significance at n=8 | HIGH | KEEP/MODIFY (keep mechanism; significance unresolved until 16-seed ladder comparisons) |
| Recon-mod | isolation verdict + E1 16-seed | clean +3.10 (16/16), adv −0.90 n.s. | clean-only lever; DEFERRED from headline | HIGH | KEEP/MODIFY (optional clean knob, never default) |
| Entropy-gated halting (soft) | smoke/isoB telemetry; eval runs | no degradation vs fixed-T; telemetry healthy | exonerated driver; signal source will change at sbr4 | MED | KEEP (re-evaluate post-sbr4) |
| T=6 foraging | e3_verdict paired stats | clean +2.29, adv −3.25, both n.s. | more steps ≠ more robustness here | HIGH | RETIRE for headline; DEFER as ablation |
| HPC edge-map L1 (w=0.1) | stage2 + D | standalone crossover real; inside D, unseparated | contributes; interaction with AIS unresolved | MED | KEEP |
| HPC multi-level / orientation target | cluster c1 not_implemented list | — | never trained | — | DEFER |
| HPC belief-target (D3) | code exists, no run | — | motivated by E1 Lens finding (pixel recon dilutes precision) | — | DEFER pending D3 run |
| SBR legacy N=1 wiring | e2_verdict | adv-neutral, clean −9.9 | wiring incapable of showing slot benefits | HIGH | REPLACE (superseded by spatial binding) |
| SBR spatial binding (sbr0) | sbr0 gate JSON | structural gate passed 4/4 | machinery healthy; specialization unproven (probes ~0.50) | MED | KEEP (with probe caveat) |
| sbr1 one-sided gate | amendment 2026-09-11 | formula artifact caught | gates must be falsifiable in both directions | HIGH | KEEP (lesson) |
| sbr2 curriculum | user CSV (n=11→16) | clean +7.5 / adv −10.7 vs D | SBR trades robustness for clean under adversarial ramp | MED (n in flight) | UNDECIDED pending matched n |
| sbr3 fixed-eps fine-tune | 2026-09-15 eval | adv +4.4 recovered, clean −3.0 vs sbr2 | the ladder's recovery design works; significance vs baseline not reached | MED | KEEP/MODIFY |
| sbr4 uncertainty-first-class | training now | — | the AIS×SBR interaction test | — | UNDECIDED |
| AIS-v2 genuine EIG | code only | — | — | — | DEFER (run D2) |
| IWM | NullWorldModel only | — | out of scope Gen 0 by roadmap | — | DEFER |
| Memory (c3), distributional (c6), self-monitoring (c7) | clusters NOT STARTED | — | — | — | DEFER |
| Fixed-ε fine-tune as recovery phase | sbr3 | +4.4 adv recovery in 20 epochs | curriculum pressure → fine-tune at final ε is the productive pattern | MED | KEEP |
| Global clip over mixed groups | starvation incident 2026-08-13 | 1000× attenuation | per-group optimizer+clip is mandatory for any new loss head | HIGH | KEEP (infrastructure) |
| PGD-50→100 masking bar ≤1.0 pp | stage1/2 masking checks | all cleared so far (sbr3 borderline −1.1) | genuine-robustness guard works | MED | KEEP |
| 2σ conservative criterion | all verdicts | D and C cleared it; B, sbr3 didn't | avoids false crossovers; under-powered at small n | HIGH | KEEP (with n=16 norm) |
| Donor-row reuse + byte verification | comparator_registry + consistency_assert | 10 donor blocks verified in Gen-1 report; loader bug fixed 2026-09-15 | comparator integrity is enforceable in CI | HIGH | KEEP |

---

# 7. E1 reconstruction audit

**Objective:** test whether precision-modulated reconstruction (recon-mod: the GlobalPrecisionModulator's gain scaling the reconstruction loss weight) improves robustness — motivated by Defense-GAN-style generative-projection intuitions (corpus [20]) and the trunk's recon pathway.

**Exact mechanism (code):** `GlobalPrecisionModulator` exposes `modulate_recon_weight(pi_d, w_recon)`; trainer multiplies the differentiable per-step recon MSE (`(x_foveal − predicted_crop)².mean()`, NOT detached — the v12 fix) by the modulated weight `[CODE: rhan_core/precision/global_precision.py, phase1_training/train_rhan_next.py]`.

- **Target:** pixel-space foveal crop (B,3,48,48) vs GenerativePrior prediction. **Decoder:** the GenerativePrior itself (conv decoder from belief s). **Weighting:** w_recon=0.1 modulated by π_d gain. **Gradient paths:** into generative_prior + foveal pathway; the modulation gain itself is trained through all three of its consumers. **Loss contribution:** bounded by w_recon slot (0.1 nominal) — recon-mod rescales, does not add a new term.
- **What WAS proven:** (1) recon-mod is the *necessary and sufficient* driver of the smoke's Pi_D per-class reordering (car/airplane vs reference car/truck) — from the smoke↔isoB contrast, with the sufficiency limb boundary-level (margin 0.0002–0.0013, honestly caveated) `[ART: isolation_verdict]`; (2) at 16 seeds, recon-mod ON (E1) wins clean on 16/16 seeds (+3.10 mean) while adv is statistically indistinguishable from D (−0.90) `[ART: e1_verdict]`.
- **What WAS suggested (not proven):** that pixel reconstruction *dilutes* precision signals (E1 Lens finding) and that this motivates belief-target HPC (D3's motivation) — a hypothesis recorded in the roadmap, supported by correlational Lens evidence, not causally isolated `[ART: rhan_nx stages hpc_belief.mechanism]`.
- **What was NOT proven:** any robustness benefit of recon-mod; any improvement of D by E1 under attack; belief-drift reduction (no belief_drift metric was recorded for E1 in repo artifacts — the per-epoch drift diagnostic was designed for sbr2+).
- **Continuation/halting interaction:** halting was ON in smoke and isoB (contrast isolates recon-mod); no halting-distribution change was recorded for E1. Reconstruction error itself as a *number*: reported per-run in diag JSONL only (UNVERIFIED at this dossier's cutoff — not aggregated into repo docs).
- **Status: FINAL** (16-seed verdict, validated 2026-08-31).

---

# 8. SBR audit (spec vs implementation)

| Stage | Spec (roadmap) | Implementation | Match? |
|---|---|---|---|
| SBR-0 | frozen backbone, clean-only, gate at ≥10 every 5 epochs, 40 ceiling; 4 criteria | `sbr_stage='gate_only'`, model-level freeze outside `structured_belief.*`, slot params own optimizer group | ✅ matches; gate JSON agrees with code paths |
| SBR-1 | joint fine-tune, clean-only, collapse detector | `sbr_stage='clean_classifier'`, freeze OFF | ✅; but gate formula amended post-hoc (symmetric→one-sided) — REPORTED, not silently reconciled `[ART: gate_amendment]` |
| SBR-2 | standard curriculum + belief_drift diagnostic every 10 epochs | curriculum confirmed in training commands; **belief_drift diagnostic: NOT FOUND in eval artifacts reviewed** — diag was specified (`--belief-drift-every 10`) but no belief-drift series is present in repo reports | ⚠ DIFFERENCE: diagnostic designed, no evidence it was recorded |
| SBR-3 | + relational layer + evidence heads; 15–20 ep fine-tune at fixed ε=0.094 | `sbr_stage='relational'` builds SlotRelationalLayer + EvidenceDecomposition (uncertainty_mode=False); 20-epoch fixed-eps training observed `[LOG]` | ✅ matches (epoch count at top of 15–20 band) |
| SBR-4 | uncertainty as first-class output; belief.uncertainty() becomes decomposition's | `sbr_stage='uncertainty'` → `uncertainty_mode=True`; forward writes `evidence['uncertainty']` into the operative uncertainty; **pooled evidence replaces the classifier belief** | ✅ matches; NOTE the classifier-belief replacement is part of the implementation and is a stronger intervention than "uncertainty swap" alone — worth stating in reports |
| Slot config | 16 slots × 512, 3 iters | verified by instantiation `[MEASURED]` | ✅ |

Rationale documented per stage in roadmap. Current result/status: sbr0 gate passed; sbr1 62.49 clean; sbr2 in-flight n; sbr3 complete (27.69@0.094); sbr4 training. The **actual implementation matches the task specification on all checkable points**, with two differences surfaced above (belief_drift missing; SBR-4 belief replacement) — reported, not reconciled.

---

# 9. AIS audit

**AIS-v1 (halting-only).** Candidate generation: none (gradient-based Eq.-II relocation). Fixation: differentiable grid_sample crop at gaze a. Foveal crop 48×48. Scoring/stopping: entropy-gated halting on u=1−π_d (proxy), threshold 0.35, softness 8.0, soft continuation (no hard exit). Gradients: gaze policy gets gradient through step_net residual + continuation weights (asserted by `test_gradient_flow.py`). Training: 60-epoch curriculum from baseline. Inference: same loop, no early exit. Eval: 8-seed protocol, masking-free. Gaze-shift metrics: gaze_shift_total_mean etc. recorded per epoch; calibration table in ARCHITECTURE §9.1a (gaze 0.05 bar vs measured 0.2795 dry-run / 0.36–0.39 v11 post-fix / ~0.007 dead state). Information-gain metrics: **none** — the roadmap explicitly forbids calling v1 "information gain" (`result_labeling_rule`) `[ART]`.

**AIS-v2.** Changes vs v1, exactly: (1) K=4–8 explicit candidates sampled around the highest-uncertainty region; (2) second forward through the foveal stream only per candidate (lookahead); (3) candidate-evaluation head (`gaze_policy_v2.*` prefix; own optimizer group `ais_v2`) trained by TD pairs — predicted candidate surprise vs next-step observed surprise, the ONLY gradient path (candidates detached, documented approximation #3); (4) halt machinery inherited. Selection: argmax predicted EIG. Everything else identical. Status: code-complete, zero training runs, zero eval numbers. Its pre-registered gate includes a *candidate-preference* check (predicted-vs-observed surprise correlation > ~0) — tested directly `[ART: rhan_nx stages ais_v2.protocol]`.

**Gaze-perturbation Lens result (the honest negative):** `report/gaze_perturbation_correlation.json` (30 images, ε=0.094, PGD-20, ckpt=B): aggregate Pearson r=0.207 (p=0.024), Spearman ρ=0.260 (p=0.004), mean displacement 6.60 px over 120 gaze samples. Interpretation present in roadmap (`ais_v2.eval`): "AIS-v1 shows no significant relationship to local perturbation magnitude" for practical purposes — the measured correlation is *statistically nonzero but tiny* (r≈0.2); the roadmap's phrasing reflects the Lens verdict convention. Read both; do not paraphrase either direction.

---

# 10. HPC audit

- **Current HPC** = HPCLevel1: single level, tap = foveal crop, target = edge map (Sobel-like non-learnable extractor; OrientationMapExtractor exists unwired), predictor 2-conv stack (1.22M params measured), error = mean |feature − prediction| over target map, NOT detached; loss slot w_hpc=0.10; separate optimizer group (lr ×6.67).
- **Pixel-level prediction?** No — by design (feature-vs-pixel rule; Stage-2 acceptance explicitly requires edge target, not raw pixels). **Precision:** none inside HPC; precision lives in the image-precision module (π_d) and the modulator. **Error signal:** scalar per sample + per-step map (min/max/std logged). **Correction:** none — HPC emits a loss, it does not feed corrections back into the forward pass (the trunk's RecurrentFeedbackLarge does that, separately). **Level structure:** exactly 1; num_levels>1 rejected. **Belief-HPC (D3):** implemented (predict belief_{t+1} from belief_t, delayed-by-one; input attached/target detached), never trained. **Gradients:** hard-asserted non-zero via tests. **Training target:** edge map (C/D), belief (D3 only). **Temporal interpretation:** none in the pixel variant (per-step i.i.d. targets); the delayed-by-one D3 variant is the first genuinely temporal wiring, unvalidated.
- **Verdict on framing:** "predictive-coding-inspired auxiliary loss" is accurate. "Full predictive coding implementation" is NOT — no hierarchical generative model, no precision-weighted error propagation across levels, no local learning rule. Corpus entry [33] (Rao & Ballard) is honestly marked "Adapted — differs materially" `[ART]`. NOESIS_FOUNDATION.md line ~486 already disclaims biological fidelity; audit items in §13 hold the rest to that standard.

---

# 11. Training data audit

| Dataset | Use | Details (evidence `dataset_stl10.py` + trainer logs) |
|---|---|---|
| STL-10 train | supervised/TRADES + pseudo-labeling seed | 5,000 labeled (500/class), 96×96; augment: RandomCrop(96,pad 12), HFlip, ColorJitter(0.2,0.2,0.2,0.1) |
| STL-10 test | ALL eval | 8,000 (800/class); no augmentation; **eval n=300/seed is a subset** of these 8,000 |
| STL-10 unlabeled | pseudo-label pretraining | 100,000; conf≥0.65 threshold → ~46K pseudo examples mixed into training (5K real + ~46K pseudo composition confirmed in isolation verdict text) |
| CIFAR-10 | phase3/phase5 human & SDT studies only | pixelated to 73% human baseline accuracy (phase5 report); NOT used in RHAN training |
| Stylized-ImageNet | phase5 comparison only (Shape-ResNet) | not RHAN training data |

Augmentation for adversarial phases: curriculum ε ramp 0.031→0.062→0.094 (PGD during training; fixed-ε fine-tunes at 0.094 for sbr3/4). Attack protocol at eval: PGD-50/PGD-100, norm-space.

**STL-10 constraints, quantified (not vibes):**
1. 5K labeled → pseudo-labeling is load-bearing; label noise floor enters every result (conf 0.65 threshold; no independent validation of pseudo-label purity found — UNVERIFIED).
2. Eval n=300/seed × ≤16 seeds → the observed seed std ~2.4–3.2 pp means 2σ gates need ≥8 pp effects; nothing smaller is detectable. This is the binding statistical constraint of the whole record (multiple verdicts "positive but NOT significant" are power-limited, not effect-absent).
3. 10 classes / 96×96 → ε=0.094 in norm space is a *large* perturbation relative to CIFAR conventions; cross-paper comparisons need the norm-space convention stated.
4. Test set reused across every model and seed set (fixed 300-sample subsets per seed) → all comparisons are paired on the same underlying test pool; no held-out confirmation set exists.
5. Training budget: T4 15.6GB → batch 16 × accum 16; ~31.5 min per (seed, ε, PGD-100) cell → ~8.5 h per 16-seed adv leg `[LOG]`.

---

# 12. Human psychophysics (phase3)

- **Design (form_structure.txt):** Google Forms; consent → participant info (ID, vision, device) → 5 blocks × 20 images; per image: 10-way class choice + confidence 1–10 linear scale. ~100 images ≈ 20–30 min. Debrief discloses adversarial perturbation.
- **Stimuli (manifest.csv):** exactly 100 images, CIFAR-10 classes (airplane/automobile/bird/cat/deer/dog/frog/horse/ship/truck — note: *not* STL-10's monkey class), 20 per ε block, ε ∈ {0.00, 0.05, 0.10, 0.20, 0.30}, 2 images/class/block; some image IDs repeat across ε (e.g. dog_04988 at 0.10 and 0.20) — deliberate pairing or artifact: UNKNOWN.
- **Responses:** `data/responses_mapped.csv` exists (mapped to stimuli). Participant N, exclusion rules, and per-ε accuracy: present in the file but not re-derived here; headline claims about human d′ come from **phase5**, not phase3.
- **Analysis pipeline & model comparison:** phase5 SDT (`phase5_sdt/`): 7 systems (BagNet, CORnet-S, EfficientNet-B0, ResNet-18, Shape-ResNet-50, ViT-Small, Human) on CIFAR-10 pixelated; human clean d′ 2.734 with threshold >0.30 (vs ResNet-18 0.0295, ViT 0.0264, EfficientNet 0.0062 — fragility ranking) `[ART: final_report_6model.txt]`. CLIP ViT entry: PENDING as of that report.
- **Intended relation to RHAN:** the "human threshold > machine thresholds at every ε" divergence is the motivating gap RHAN targets (ACD paper v1); **no RHAN model has yet been scored in the SDT pipeline against the human data** — NOT FOUND.

---

# 13. CLAIM / IMPLEMENTATION AUDIT

(Condensed here; full row set in `RHAN_NOESIS_CLAIM_IMPLEMENTATION_AUDIT.md`.)

| Claim (source) | Implementation reality | Status |
|---|---|---|
| "Biologically inspired" (README/NOESIS/paper) | structural metaphors only (streams split, recurrence, halting, π_d); no biological constraints, no neural data, no Brain-Score run on RHAN | METAPHOR — do not present as more |
| "Human-like robustness" (dossier-era framing) | robustness numbers exist; zero human-model comparison for RHAN itself | UNVERIFIED |
| "Predictive coding" | single-level edge-map auxiliary loss (+ trunk feedback layer); no hierarchy, no generative model, no local learning rule | PARTIAL — say "predictive-coding-inspired auxiliary loss" |
| "Active inference" | π_d-weighted belief update + gradient gaze; no free-energy objective, no variational bound | PARTIAL |
| "CLIP-grounded" (NOESIS docs history: v4 live CLIP loss) | CLIP used historically (v4, smoothed — documented as harmful there); no CLIP term in current code | HISTORICAL, RETIRED |
| "Concept bottleneck" (Finding 9 framing) | no CBM heads/losses; Pi_D per-class diagnostics only | NOT IMPLEMENTED |
| "Self-supervised" | pseudo-labeling (confidence threshold), not SSL objectives; corpus [91] FixMatch marked "not implemented — could adopt" | PARTIAL (pseudo-labels only) |
| "Adversarially robust" | PGD-based evidence at fixed ε; no AutoAttack/C&W/FAB despite corpus flagging AutoAttack as the required standard | PROVISIONAL |
| "World model" | NullWorldModel passthrough; enable_iwm validates False | SCAFFOLD ONLY |
| "Adaptive computation" (halting) | soft continuation weights; fixed T=4 wall-clock | PARTIAL |
| "Uncertainty-aware" | three implemented uncertainty sources; no calibration metrics (ECE etc.) anywhere | PARTIAL |
| Slot attention "object-centric" | slots bind spatial features; probe accs ~0.50 (uniform-ish); object-ness unproven | IMPLEMENTED, OBJECT-CLAIM UNPROVEN |
| "~52M params" (large model docstring) | MEASURED 55,622,347 | WRONG (docstring) |
| "~63.4M" (v12 docstring) | MEASURED 75,440,469 | WRONG (docstring) |
| "76M"/"81M" (roadmap/corpus asides) | neither matches any measured model exactly (84.47M = full-pillar) | STALE |

---

# 14. Literature references in the repository

Primary source: `docs/research/RHAN_NOESIS_LITERATURE_CORPUS.md` — **251 unique papers**, each with authors/year/venue/DOI/arXiv/verification source, relationship tags, and an implementation-honesty field (Direct / Adapted / Not implemented). The corpus is the deduplicated reference database; **do not re-derive metadata from memory** — it ships with verification provenance per entry. Distribution by part (from the corpus TOC): attacks 21, AT/robustness 40, foundations 16, core arch 27, human-vs-machine 23, predictive coding 14, active vision 17, object-centric 23, uncertainty 16, evidence accumulation 4, psychophysics 14, neuroscience 12, world models 10, evaluation 13, medical 5.

Key directly-implemented entries (sample, with the corpus's own honesty labels): TRADES [4] (Direct-adapted: training objective, w=0.55); Madry PGD [2] (protocol followed by `phase2_attacks/pgd.py`); Carmon unlabeled-data [97] and Uesato label-free [96] (Direct-adapted: pseudo-label pipeline); Slot Attention/Locatello [40ff] (Direct-adapted: SBR); Rao & Ballard [33] (Adapted: HPC, "differs materially"); robust overfitting [108] (Direct-adapted: best-checkpoint policy); spatial attacks [115] (Adapted). AutoAttack [5] explicitly NOT implemented — flagged as the standard RHAN claims must eventually survive. Additionally, informal citations exist in code headers (Friston 2010, Rao & Ballard 1999, Itti & Koch 2001 in `model_rhan_v12.py`) and in `NOESIS_FOUNDATION.md`'s bibliography section (Ha & Schmidhuber; Hafner Dreamer; Pathak ICM; Lake et al.). All such informal mentions are subsumed by corpus entries.

---

# 15. Candidate future mechanisms already in the repo (not invented here)

| Mechanism | Exact reference | Proposed purpose | Status | Dependencies |
|---|---|---|---|---|
| TDV (Temporal Divergence Vector) | `MotionEncoderLarge`, `TDVProjectionHeadLarge` (dead modules); RHANfuture.md | video/temporal extension | dead code, unused | data pairs (UCF/TDV frames), memory |
| IWM | NOESIS_FOUNDATION Pillar 4; NullWorldModel; corpus world-models part | counterfactual gaze simulation before fixation | scaffold; enable_iwm=False enforced | world model, SimulatedGazePolicy (reserved) |
| Memory (SchemaMemory, TemporalBeliefPersistence) | roadmap cluster c3 | carry belief across frames | NOT STARTED | TDV data, slots |
| Belief-target HPC (D3) | roadmap rhan_nx.hpc_belief | replace pixel target per E1 Lens finding | code ready, unrun | none (swap vs D) |
| AIS-v2 genuine EIG (D2) | roadmap rhan_nx.ais_v2 | true lookahead gaze | code ready, unrun | candidate head pre-flight |
| Uncertainty-first-class SBR (sbr4) | roadmap rhan_nx.sbr4 | uncertainty gates halting | TRAINING now | sbr3 |
| Distributional belief (μ,Σ), multi-hypothesis workspace | cluster c6; corpus [51–188] | richer belief geometry | NOT STARTED | evidence decomposition |
| Self-monitoring / abstention (c7) | cluster c7; SDT tooling exists offline | selective classification under attack | NOT STARTED | uncertainty, calibration |
| Human-alignment losses | phase4_analysis/.claude.md (#8 Brain-Score/NSD alignment) | representation alignment to IT | idea only | fMRI data pipeline |
| Residual predictive feedback upgrade | phase4_analysis/.claude.md (#4: stem = stem + gate·(feedback − prediction)) | replace gate-only feedback with error form | idea only (trunk already has an error gate form — partially overlaps) | trunk refactor |
| View-agreement error check (c6 #18) | cluster c6 | feature-vs-pixel error agreement as cheapest entry test | NOT STARTED | recon + HPC |
| AutoAttack hardening | corpus [5] + roadmap masking borderline tier | upgrade robustness claims | NOT STARTED | compute |

---

# 16. ImageNet transition constraints (facts vs estimates)

**FACTUAL CURRENT (measured/observed):**
- Hardware: single T4 15.6 GB (Colab) — batch 16 × accum 16 on 96×96 75–84M-param models; RTX 4060 local; occasional Kaggle twin.
- Throughput: ~31.5 min per adv eval cell (PGD-100, 300 imgs, batch 32); ~8.5 h per 16-seed adv leg; 60-epoch STL-10 curriculum ≈ order days with PGD-in-training.
- Checkpoints: ~75–85M-param fp32 → ~300–340 MB each; HF-hosted with rolling+best pairs per stage.
- Optimizer memory: SGD only; no Adam/AMP anywhere in the current path.
- Dataloader: STL-10 fits locally; num_workers ≤4; no distributed training except the just-landed DDP scaffolding (commit fe7a709, untested at ImageNet scale — its own note).
- Attack/eval cost already dominates: per-model 16-seed × {PGD-50, PGD-100} ≈ 12–17 h on the T4.
- Human study: Google-Forms-scale, 100 stimuli, one round; no institutional pipeline.

**ESTIMATED FUTURE (ImageNet-1K, 224², ~1.28M images) — explicitly NOT measured here:**
- 224² at the same token density ⇒ ~784 tokens (vs 144): attention cost ×~30 per layer pair at fixed width; the checkpointed dual-stream trick will not suffice — gradient checkpointing × mixed precision (bf16) + FlashAttention-class kernels become mandatory.
- Adversarial training at ImageNet scale with PGD-in-loop: 10–50× ImageNet compute per epoch vs clean; free/fast-AT literature (corpus [21],[99]) exists precisely for this budget class.
- Eval protocol scaling: 16 seeds × 300 imgs is impossible; seed-averaging must give way to fixed-protocol single-seed + CI methods or subset seeds; PGD-100 × 50K test images ≈ hundreds of GPU-hours per model per ε.
- Checkpoint sizes: 300M–1B+ params → GB-scale artifacts; rolling+best doubling needs real storage budget.
- Storage: ImageNet-1K ~150 GB + perturbation caches; HF dataset hosting limits apply.
- Human study: 1000-class stimuli redesign; SDT pipeline is CIFAR-bound today.
These are engineering estimates from public scaling knowledge, NOT project measurements — treat as UNVERIFIED until a costed plan exists.

---

# 17. Provenance (per reported number)

| Result | Source file / artifact | Checkpoint | Seed set | Script | Notes |
|---|---|---|---|---|---|
| Baseline 54.81/24.23 | roadmap e1_verdict ← HF `sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv` | rhan_stl10_large_pseudolabel_best.pth (sha 37a4eee0…) | 41–56 | eval_rhan.py | 16-seed |
| D 54.96/34.02 | same CSV (donor rows in all ladder sweeps) | rhan_next_ais_hpc_best.pth | 41–56 | eval_rhan.py | 16-seed; 8-seed D numbers (54.25/34.38) are the Stage-3 CSV — distinct, label when citing |
| B 49.4/32.21 | roadmap stage1_verdict ← report/sweep_stage1_ais_v1_halting_only_merged | rhan_next_ais_v1_halting_only_best.pth (sha 19582ff4…) | 41–48 (PGD-50); clean n=5 | eval_rhan.py + eval_sweep_next.py | epoch-final model caveat |
| C 55.2/27.73 | roadmap stage2_verdict | rhan_next_hpc_only_best.pth (sha 5b7dce1e…) | 41–45 | eval_rhan.py --ablation-matrix | 5-seed |
| E1 58.06/33.12 | roadmap e1_verdict | (E1 ckpt on HF) | 41–56 | eval_rhan.py | 16-seed |
| E2 45.06/33.42 | roadmap e2_verdict ← HF sweep_stage4_e2_d_sbr_pgd100 | e2 ckpt | 41–56 | eval_rhan.py | reconstructed 2026-09-11 from CSV (recorder never ran) — flagged in the verdict itself |
| E3 57.25/30.77 | roadmap e3_verdict | e3 ckpt | 41–56 | eval_rhan.py | D+baseline rows = DONOR (not re-evaluated) |
| sbr2 62.42/23.33 | user-supplied per-seed CSV (session paste) + HF sweep_rhan_nx_sbr2_* | rhan_nx_sbr2_best.pth | 41–51 (n=11; 52–56 pending) | eval_rhan.py PGD-100 | n in flight |
| sbr3 59.44/27.69 | HF sweep_rhan_nx_sbr3_pgd100 (96 cells merged) + PGD-50 leg 28.83±2.44 | rhan_nx_sbr3_best.pth | 41–56 | eval_rhan.py; donor comparator cells replaced not re-run | 2026-09-15 |
| sbr0 gate JSON | report/sbr0_gate_verdict.json + roadmap | rhan_nx_sbr0_best.pth | n/a (512-sample gate) | gate script | 2026-09-11 |
| SDT table | phase5_sdt/results/final_report_6model.txt | external models | n/a | SDT pipeline | 2026-05-17 |
| Param counts | this dossier | — | — | instantiation 2026-09-15 | MEASURED |

**DONOR discipline:** comparator rows (baseline, D) inside every ladder sweep are **DONOR / NOT RE-EVALUATED**, byte-verified against their source CSV at load time by `scripts/consistency_assert.py` (post-hoc aggregate verification removed 2026-09-15 as impossible-by-construction; per-seed verification retained + regression-tested).

---

# 18. INTERNAL CONTRADICTIONS (reported, not fixed)

1. **Parameter counts:** v12 docstring ~63.4M vs MEASURED 75,440,469; large-model docstring ~52M vs MEASURED 55,622,347; roadmap/corpus asides "76M"/"81M" match nothing exactly (84.47M = full-pillar RHANNext). Docstrings are stale.
2. **Roadmap runtime staleness:** `docs/rhan_next_roadmap.json` `rhan_nx.stages` still shows sbr1 `gate_failed`/sbr2+ `not_started` while the HF state machine is at sbr4/training. The roadmap itself forbids hand-editing these statuses (single source of truth = `scripts/stage_state_machine.py`); consumers must read runtime state, not the checked-in JSON.
3. **E2 vs Gen-1 report:** `report/rhan_nx_generation1_report.md` (dated 2026-09-08) renders every ladder stage PENDING with only donor rows, while e2/sbr2/sbr3 results exist elsewhere — the report predates its inputs and was not regenerated.
4. **Lens claim drift:** roadmap says AIS-v1 gaze "shows no significant relationship to local perturbation magnitude"; the checked-in JSON shows Pearson r=0.207, p=0.024 (statistically nonzero, practically small). Both statements trace to real artifacts; they disagree in emphasis.
5. **"Crossover real" reproducibility:** stage1's merged verdict (8.5 pp, n.s.) vs stage2 session's baseline@0.094 = 20.4 (CROSSOVER REAL, +12.13) vs stage4's baseline = 24.23 — the baseline's own adv number moved 3.8 pp across sessions under the documented nondeterminism caveat; cross-session donor comparisons must keep the caveat visible (the E3 verdict does; some summary tables don't).
6. **sbr1 gate:** original verdict `gate_failed` recorded beside a 62.49 clean score that *exceeded* D by +7.5 — resolved by amendment, but both the failed verdict and the amendment coexist in the artifact chain (by design, preserved honestly).
7. **PGD-50/PGD-100 ordering at sbr3:** PGD-50 leg (28.83) is *below* PGD-100 (27.69 is 100-step; 28.83 is 50-step → gap −1.1 pp), violating the monotonic intuition the masking bar assumes; within nondeterminism floor, but the report builder's masking tier will need the borderline label, not the GENUINE one.
8. **Dead modules in checkpoints:** motion_encoder/tdv_head params ship in every "RHAN" checkpoint though unused — checkpoints ≠ effective architecture.
9. **Docstring vs registry naming:** `rhan_next_ais_hpc` (checkpoint label) is the D run; "D_ais_plus_hpc" (matrix key); "rhan_nx_sbr*" (ladder). Same lineage, three naming schemes; eval logs bridge them but no single doc does.
10. **STL-10 vs CIFAR-10 class sets:** phase3/5 human stimuli use CIFAR-10 classes (frog, automobile) while RHAN models train on STL-10 (monkey, no frog) — the human-model comparison is cross-dataset; the SDT report's "73% human baseline is pixelation" caveat partially covers this, but class-set mismatch is nowhere stated in phase5 docs.

---

# 19. Final evidence summary

## WHAT WE ACTUALLY KNOW
- D (AIS-v1 halting-only + edge-map HPC) beats the TRADES baseline by ~9.8–11.6 pp adv acc at ε=0.094, clearing the pre-registered 2σ gate, 8–16 seeds, masking-free; clean parity (±0.15 pp).
- Every mechanism is isolated by on/off tests; gradient flow is machine-asserted; donor rows are byte-verified; verdicts are pre-registered and null results are recorded as outcomes.
- Recon-mod: clean-only lever (+3.10, 16/16 seeds), adv-neutral. T=6 foraging: no robustness gain. Legacy N=1 SBR: clean −9.9, adv-neutral.
- sbr0 structural gate passed 4/4; sbr1 over-performed (62.49 clean); sbr2 bought clean (+7.5) and paid adv (−10.7); sbr3's fixed-ε fine-tune recovered ~41% of that adv loss at −3.0 clean; sbr4 (uncertainty-first-class) is training; D2/D3 swap tests are coded but unrun.
- The evaluated stack is honest about: epoch-final vs peak-val artifacts, donor reuse, gate-formula failures, and nondeterminism (~1–1.5 pp).
- Measured parameter counts: 55.6M (baseline) / 75.4M (RHANNext default=v12) / 84.5M (full pillar stack).

## WHAT WE STRONGLY SUSPECT
- The robustness gain comes from the AIS+HPC *combination*, not either alone (D-vs-B +1.38 n.s., C-vs-B −4.8) — but the record cannot separate the interaction cleanly at current n.
- Slot representations have not yet demonstrated object-level specialization (probe accs ~0.50 ≈ 5× random but uniform across slots; everything-slot ablation benign).
- The 2σ criterion under-powers everything below ~8 pp; several "n.s." verdicts (B, sbr3-vs-baseline) are power artifacts, not nulls.
- Statistical-detection limits, not architecture, bound what the STL-10 rig can conclude; the next meaningful lever is n and protocol, not new mechanisms.

## WHAT WE DO NOT KNOW
- Whether sbr4's evidence-uncertainty is a *better halting signal* (the entire point of the ladder's last SBR rung) — training now.
- Whether genuine EIG gaze (D2) or belief-target HPC (D3) beat their v1/pixel counterparts — coded, unrun.
- Any calibrated-uncertainty, OOD, AutoAttack, certified, or human-aligned property of RHAN — never measured.
- Which component of D's gain is curriculum vs architecture (no curriculum-only control at matched n).
- v10/v11-era numbers beyond the recorded null-ablation figure.

## WHAT GENERATION 0 SHOULD DECIDE
1. Freeze the Gen-0 headline as **D vs baseline + the E1/E2/E3 ablation row + the sbr-ladder trade-off curve** — nothing else is 2σ-clean.
2. Close the ladder before adding mechanisms: finish sbr2 seeds 52–56 (matched n=16), run sbr4 eval with uncertainty-distribution logging, then D2 and D3 (they are coded and cheap relative to what they disambiguate).
3. Adopt n=16 + paired-per-seed reporting as the default; retire 2σ verdicts at n<8 from headline use.
4. Fix the docstring parameter counts and the report staleness before any external consumption (they are the two artifacts an outside reviewer will find first).

## QUESTIONS THAT MUST BE ANSWERED BEFORE IMAGENET GENERATION 1
1. Does sbr4's uncertainty separate clean from adversarial inputs (attack detection) — the untested premise of "uncertainty as first-class output"?
2. Do D2/D3 change the D-vs-baseline margin, or is D's effect a one-off of this backbone/curriculum?
3. What is the minimum eval protocol (seeds, samples, attacks incl. AutoAttack) that makes the headline claim survive external review — and what does it cost at ImageNet scale?
4. Which parts of the stack survive 224²/784-token attention economics, and which (checkpointed dual-stream, grid_sample foveal sampling at scale) must be redesigned?
5. Is the pseudo-label pipeline (conf 0.65)replaceable by a validated SSL pretrain at ImageNet scale, and what does that do to the D effect?
6. What is the human-comparison plan for a 1000-class system (the current SDT/human rig is CIFAR-bound and cross-dataset vs STL-10)?

---

*End of dossier. Companion files: `RHAN_NOESIS_EXPERIMENT_REGISTRY.csv`, `RHAN_NOESIS_REFERENCE_SEED.csv`, `RHAN_NOESIS_CLAIM_IMPLEMENTATION_AUDIT.md`, `RHAN_NOESIS_GENERATION0_LESSONS.md`.*

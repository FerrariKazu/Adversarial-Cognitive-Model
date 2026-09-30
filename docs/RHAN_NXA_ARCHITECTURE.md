# RHAN-NXA (Gen-1) Architecture

> Status vocabulary used in this document (and the rest of the current
> documentation): **IMPLEMENTED**, **EXPERIMENTAL CANDIDATE**,
> **DEFERRED**, **REJECTED**, **UNKNOWN**, **NOT YET IMPLEMENTED**.
> Nothing planned is described as implemented. Evidence pointers are given
> for every claim.

RHAN-NXA is a research architecture for **perception as iterative
investigation**: the model maintains an explicit belief state about the
image and refines it across T=4 glimpses —

```text
Predict → Observe → Error → Precision → Update → Attention/Gaze → … → Readout
```

Classification is a **readout** of the belief, not the belief itself.

---

## 1. Where RHAN-NXA sits in the lineage

```text
RHAN lineage (CIFAR-10 study, RHAN v1–v7, STL-10 scale-up)
      ↓  lessons: gradient-flow bugs, "trains ≠ validated", one-mechanism-at-a-time
Gen-0 / RHAN-Next (rhan_core/, frozen)  — AIS-v1 + HPC pillars; curriculum-trained
      ↓  lesson: confounded swaps (SBR legacy in ais_v2/hpc_belief arms) ⇒ isolated effects UNKNOWN
Gen-1 / RHAN-NXA (noesis_vision/)       — CURRENT
```

Authoritative plan: `noesis_vision/RHAN_NXA/MASTER_PLAN.md`.
Chapter-by-chapter documentation: `noesis_vision/RHAN_NXA/docs/`.

## 2. The perception loop and where each step lives

Canonical locations (all under `noesis_vision/` unless noted):

| Step | Component | Location | Status |
|---|---|---|---|
| Observation | 96×96 loader operating point; T=4 glimpses | `evaluation/imagenet100_loader.py`; schema (`core/schema.py`, `num_glimpses=4`) | IMPLEMENTED |
| Foveation | `foveal_sample`, 56×56 crops (56 = 4×14) | `models/foveation.py` | IMPLEMENTED |
| Backbone | CompactViT, D_z=384, DINOv2-small-shaped, ~23.3M params | `models/backbone.py` | IMPLEMENTED |
| Recurrence | **tied** within-glimpse refinement (params independent of iteration count) | `models/recurrent_block.py` | IMPLEMENTED |
| Belief | `B_t = (z_t, S_t, U_t, E_t, A_t)`; **S_t=None canonical** | `beliefs/vector_belief.py`, `beliefs/drift.py` | z/U/E/A IMPLEMENTED; **S_t: DEFERRED** |
| Prediction | next-glimpse latent prediction, `error_target = latent_next_glimpse` (LOCKED) | `predictive_coding/glimpse_predictor.py` | IMPLEMENTED |
| Prediction error | E from prediction vs observation | `predictive_coding/interfaces.py` + integration | IMPLEMENTED |
| Precision / Update | `z_{t+1} = z_t + Pi·UpdateNet(z, E)`; pre-flight \|dW\| criteria | `predictive_coding/precision.py`, `update_net.py` | IMPLEMENTED |
| Uncertainty | Dirichlet evidential head, `U_t = C/Σα` | `uncertainty/evidential_head.py` | IMPLEMENTED (training); **readout/calibration eval: NOT YET IMPLEMENTED** |
| Gaze | AIS-v2 policy, candidate scoring, K=4 (locked 4–8) | `gaze/ais_v2_policy.py`, `gaze/candidate_sampler.py` | IMPLEMENTED (phases 5–6) |
| Readout | class readout from belief (evidential α) | model composition (Agent J integration in `training/train_generation1_foundation.py`) | IMPLEMENTED |
| Training objective | TRADES/PGD curriculum (default) or recorded clean-only | `training/adv_curriculum.py` + trainer | IMPLEMENTED |
| Evaluation | norm-space PGD, seed floor, consistency assertion | `evaluation/clean_and_robust.py` | IMPLEMENTED |

Gaze labeling rule: foundation phases 1–4 use a **PLACEHOLDER fixed
schedule** (never labeled AIS-v2); phases 5–6 use the AIS_V2 policy.
Every artifact carries this label.

## 3. The six-phase foundation ladder

`training/stage_state_machine.py` is the orchestration truth;
`training/train_generation1_foundation.py` is the trainer.

| Phase | Adds | Gaze |
|---|---|---|
| 1 `backbone_only` | substrate + one fixed center fixation + classifier head | PLACEHOLDER |
| 2 `recurrence_only` | + T=4 fixed-schedule glimpse loop + tied refinement | PLACEHOLDER |
| 3 `belief_no_f` | + belief carrier with U_t; IDENTITY update (no learned dynamics) | PLACEHOLDER |
| 4 `belief_with_f` | + predictor/precision/UpdateNet dynamics | PLACEHOLDER |
| 5 `ais_v2_swap` | + AIS-v2 gaze policy (predict→…→GAZE-SELECT) | AIS_V2 |
| 6 `gen1_core` | the integrated system (S_t=None, L_stab diagnostic-only) | AIS_V2 |

Isolation guarantees: per-phase provenance manifests, gradient-reach
pre-flights per head group, best/rolling checkpoint parity, HF-only resume,
and the silent-inheritance guard (a cold-started phase whose best checkpoint
is bitwise-identical to its parent's STOPs the run).

## 4. The training objective (exact, do not drift)

Ported verbatim from Gen-0's canonical trainer
(`phase1_training/train_rhan_next.py`); owned by
`training/adv_curriculum.py`:

```text
60-epoch ramp per phase:  epochs 1–20   ε=0.031  β=2.0  PGD-4
                          epochs 21–40  ε=0.062  β=2.0  PGD-4
                          epochs 41–60  ε=0.094  β=2.5  PGD-4
TRADES loss:  CE(clean) + β · KL(softmax(adv) ‖ softmax(clean)),  w_trades = 0.55
attack:       PGD-KL on normalized pixels, α = ε/4, random start 0.001·randn, ±4.0 clip
```

ε is **normalized-space only** (the protocol's hard rule — no pixel-space
path). Changing any of these constants is a new experiment, not a refactor.

Two arms exist by design and are never merged:
- **Historical control arm** — pure cross-entropy (2026-09-25→26 run;
  `report/GEN1_RESULTS_MASTER.md`; archived on HF under
  `archive/gen1_pure_ce_<stamp>/`). Objective recorded per phase as
  `adv_curriculum.clean_only = true` in manifests.
- **Corrected adversarial arm** — the curriculum above
  (`clean_only = false`), recipe version `gen1-adv-curriculum-v1`.

## 5. Scientific status of the key mechanisms

| Mechanism | Status | Evidence |
|---|---|---|
| S_t (structured belief slots) | **DEFERRED** (plan §1.A/1.D; S=None canonical) | `MASTER_PLAN.md` 1.D |
| Legacy SBR (16×512 slots) | **REJECTED** for Gen-1 core (Gen-0 confound carrier) | `MASTER_PLAN.md` Part 0 |
| Representation-level uncertainty | **PENDING DECISION** (documented tension with "classification is a readout") | `MASTER_PLAN.md` 1.A |
| AIS-v2 isolated effect on clean/robust acc | **UNKNOWN** (Gen-0 arms confounded; r=0.706 smoke-gate preference correlation is the salvageable signal) | `MASTER_PLAN.md` Part 0 |
| L_stab stability regularizer | diagnostic-only in phase 6 (thin in code) | plan §20/21; trainer docstring |
| d′ / calibration / gaze-telemetry evals | **NOT YET IMPLEMENTED** (absent metric families) | `report/GEN1_RESULTS_MASTER.md` |
| Belief-dynamics contribution on ImageNet-100 (pure-CE arm) | measured: recurrence→belief_no_f clean Δ **−9.875 pp** (Holm p<0.0001) — the largest negative effect; adversarial-arm rerun in progress | `GEN1_RESULTS_MASTER.md` |
| IWM (internal world model) pillar | scaffold only (`enable_iwm` stays False) | `docs/ARCHITECTURE.md` (Gen-0); no Gen-1 module |

## 6. What is deliberately NOT here

- Steps 7+ of the Part-2 plan (S_t arm, L_stab arm, full ablation matrix) —
  not in the foundation trainer by design.
- Any pixel-space ε path. Any output-space attack for training.
- The five future RHAN applications (future layer; see `applications/`
  README when introduced) — RHAN-NXA exposes model-level interfaces;
  applications must consume those, not internal training code.

## 7. Verification pointers

- Gradient reachability per component: `tests/test_*gradient_flow*.py`
- Curriculum semantics (ramp slicing, attack, loss, restore-mode):
  `tests/test_adv_curriculum.py` (22), `tests/test_adv_curriculum_integration.py` (7)
- Checkpoint/resume discipline: `tests/test_checkpoint_never_silently_restarts.py`,
  `tests/test_resume_commit_guard.py`
- Phase machine: `tests/test_stage_state_machine.py`,
  `tests/test_stage_state_machine_resume.py`
- Eval protocol: `tests/test_eval_rhan_protocol.py`,
  `tests/test_eval_structural_consistency.py`

Run: `python3 -m pytest tests/ -q` (never bare `pytest` here), and after any
trainer change: `python3 training/train_generation1_foundation.py --smoke`.

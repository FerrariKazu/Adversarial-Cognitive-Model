# RHAN-IM — redesign dossier (branch `redesign/rhan-im`)

Status: **research redesign + working prototype of the mathematics.** Nothing here has been measured on the trained `ff_112` checkpoint, and **no accuracy, robustness or efficiency claim is made.** The branch only *adds* paths (`rhan_im/`, `tests/rhan_im/`, `scripts/rhan_im/`, these documents); no existing source file, checkpoint, baseline, evaluation artifact, pre-registration or history is touched.

## The decision

Seven candidate architectures were worked out under the same ten headings (`RHAN_IM_03_CANDIDATES.md`); one is chosen:

> **RHAN-IM — perception as search with a stopping rule.** The recurrent state is a *Bayesian posterior* over what-is-where (a hypothesis `H_0` plus `H_(b,c)` for class `c` at cell `b`; exact for the specified model, while the optional per-unit cap makes it an explicit capped recursion, `04` §4) together with a per-unit *accumulated-precision* map (where it has not yet looked well enough). Each recurrent step is a **look** that exposes pixels at a resolution or place the earlier looks did not. The sensor's detectability `d′` is **measured, not learned**. The next look is chosen by a closed-form expected-ignorance-reduction rule, the stop is set by **Learn-then-Test** (a finite-sample guarantee on the error rate), and the influence of any one look is bounded (clip, per-unit cap) and, where the geometry allows, certified. There are no recurrent weights: everything that is learned lives in the sensor.

**Thesis.** *Recurrence earns its cost only where information is acquired, not merely recomputed.* This is the reading of the forensic audit that I can defend: the current official model is a feedforward ViT (the refinement block and gist gate never run on the `ff_112` path), glimpses of a 112-px frame add no new pixels, and the Stage-0 NULL for recurrence is therefore uninformative about recurrence-as-perception.

**Central question** (`04` §1): on cluttered scenes where the label-relevant object is small and resolution-limited, does RHAN-IM reach a prescribed error rate in fewer looks than fixed-coverage and feedforward-gist baselines at matched compute, while its stopping rule's error stays below a chosen α with probability ≥ 1 − δ, its confidence is calibrated, and single-look influence is bounded? A "no" on any part is a reportable result.

The ChatGPT frequency-aware recurrence is candidate C4, kept as a cheap *control* (it recomputes rather than acquires). C1 scores 33/35 in my own selection matrix (`03` §8; judgements, not measurements) and has a one-day falsification: gate 1.

## Documents

Numbering inside the documents (01–07) maps to files `RHAN_IM_0N_*.md` in this folder.

| # | File | Content |
|---|---|---|
| 01 | `RHAN_IM_01_FORENSIC_AUDIT.md` | every mechanism classified (executable / conceptual / implemented-never-evaluated / rejected / possibly rejected under confounds); what the recurrent NULL does and does not show |
| 02 | `RHAN_IM_02_LITERATURE_SYNTHESIS.md` | verified sources (arXiv ids only for papers I fetched), closest prior art for each candidate, scoped novelty claim |
| 03 | `RHAN_IM_03_CANDIDATES.md` | C1–C7, selection matrix, decision and switch conditions written before the evidence |
| 04 | `RHAN_IM_04_ARCHITECTURE_SPEC.md` | the specification: generative model, state, policy, stopping, guarantees, block-by-block shapes, parameters, cost, training schedule, falsifiable predictions |
| 05 | `RHAN_IM_05_EXPERIMENTS.md` | arms, statistics, experiments E1–E7, ablations, adaptive-attack protocol |
| 06 | `RHAN_IM_06_MIGRATION_PLAN.md` | fate of each existing asset, phases and gates, risk register, decisions that are yours |
| 07 | `RHAN_IM_07_GATE1_PREREG_DRAFT.md` | the first real experiment, as a draft pre-registration with its decision rule, operating characteristics and freeze checklist |
| – | `RHAN_IM_RESULTS.md` | generated tables from the JSON files below (`scripts/rhan_im/summarize_results.py`) |

Machine-readable results: `RHAN_IM_toy_theory_checks.json`, `RHAN_IM_toy_image_world.json`, `RHAN_IM_model_budget.json`, `RHAN_IM_gate1_selftest_random_trunk.json`, `RHAN_IM_gate1_operating_characteristics.json`, `RHAN_IM_dependence_calibration.json`; `RHAN_IM_forensic_checks_output.md` is the output of the audit script.

## What is verified, what is simulated, what is not measured

| Statement | Status |
|---|---|
| The implemented posterior equals brute-force enumeration over all `1 + N·C` hypotheses (footprints, `N ≠ V`, sum mode, clip, mask, branching); with the per-unit cap it equals the explicit capped recursion, which is order dependent and **not** a Bayes posterior; the "best" combination mode is the exact Bayes weight only for a degradation chain; LTT thresholds (under the assumptions stated in `04` §6), certificates and policy gains are exact where stated | **verified** by tests (`tests/rhan_im`, including two independent brute-force enumerations and a Gaussian check of the "best" claim) |
| Exact Gaussian identities behind the within-look correction (Lemma 1b: exact for an isolated block of equicorrelated tokens with one `d`, conservative otherwise, and both cases are tested) and the shared-difficulty mixture (Lemma 1c); the estimators of `VIF` and of the difficulty spread | **verified** against the multivariate Gaussian / analytic VIF in tests |
| The gate-1 pipeline runs end to end (scenes, head, standardiser, measurements, verdict, freeze/aggregate tools) at the real 224/112 geometry; the verdict and the aggregate fail closed (an unmeasured quantity or any protocol deviation gives INCONCLUSIVE or INVALID, never a quiet PASS or FAIL) | **verified as plumbing** (random-init trunk, synthetic or fake data); the numbers of that run mean nothing about the checkpoint |
| The whole chain (real-architecture trunk, window sensor, Bayes state, Koopman policy, stopping) runs at the real geometry and produces the shapes quoted in `04` §9; a scene's result does not depend on its batch neighbours | **verified as plumbing** (`tests/rhan_im/test_real_trunk_integration.py`, random weights) |
| Calibration, policy gain, Learn-then-Test violation rates, patch certificates, curtailment | **simulated** (toy SDT world; Gabor-on-1/f image world with measured `d′`); mechanism-level evidence that the mathematics is implemented correctly and how it behaves, **not** evidence about deep networks |
| Calibration cost of ignoring token dependence, and the remedies | **simulated** (`RHAN_IM_dependence_calibration.json`); the correction is in the code path, the mixture is not implemented in `belief.py` |
| Operating characteristics of the gate-1 decision rule at the planned sample sizes | **simulated** (`RHAN_IM_gate1_operating_characteristics.json`) |
| Parameter, FLOP and memory counts | **measured** on the real `CompactViTGen2` at random init (weight-independent) |
| `d′`, z-ROC slope, cross-look ρ, `VIF`, spillover of the real frozen trunk | **not measured** (needs the checkpoint and ImageNet-100; the script and the pre-registration draft are ready) |
| Any accuracy, robustness, efficiency or novelty-of-result claim on real images | **none made.** Novelty is scoped in `02` §5 to the combination, not to any component |

## How to reproduce

```
python -m pytest tests/rhan_im -q                                    # 151 tests, about a minute on CPU
python scripts/rhan_im/toy_theory_checks.py                          # toy SDT world
python scripts/rhan_im/toy_image_world.py                            # Gabor-on-1/f image world
python scripts/rhan_im/model_budget.py                               # parameters / FLOPs / memory of the real trunk, random init
python scripts/rhan_im/dependence_calibration_check.py               # token dependence: cost and remedies
python scripts/rhan_im/gate1_operating_characteristics.py            # gate-1 rule at the planned sample sizes (~30 min on 2 CPUs; saved after each condition, --resume continues)
python scripts/rhan_im/gate1_measure_dprime.py --self-test           # pipeline plumbing, no data, random trunk
python scripts/rhan_im/summarize_results.py                          # regenerate RHAN_IM_RESULTS.md
```

The real experiment is `gate1_measure_dprime.py` with the frozen checkpoint, run as `07` §11 prescribes (freeze first, commit the freeze file, then run). No script here reads a credential or touches the network.

## What I need from you

1. **Freeze gate 1 or edit it first** (`07`): thresholds, the class split (first half / second half of the sorted ImageNet-100 class names), `n_test` 6 000 vs 12 000. Later, before gate 2: the image-level four-way split of ImageNet-100 that the stopping-rule claims need (`05` §3).
2. **Run gate 1 on Colab or Kaggle** where the checkpoints and ImageNet-100 already are: three primary runs and two secondary runs (`07` §11). Nothing in it needs a token.
3. **Commit and push the branch** (it is local; pushing needs the repository attached with push access, or I can hand you a git bundle).
4. Whether to rewrite the `ff_112` model card now around the claim "DINOv2-S/14-shaped ViT, TRADES-trained on ImageNet-100 at 112 px" (recommended; `06` §1).
5. Whether to run the optional Stage A replication (closes Gen-2; off the critical path).

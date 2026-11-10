# 01 — Forensic audit of the RHAN code base (what actually exists, what the NULL does and does not show)

Branch `redesign/rhan-im`. Nothing in this audit modifies an existing file, checkpoint, baseline or evaluation artifact.

**Evidence legend.** `[E1]` executed in this branch (reproducible: `python scripts/rhan_im/forensic_checks.py`, log in `RHAN_IM_forensic_checks_output.md`). `[E2]` read in the code, file and line given. `[E3]` taken from the repo's own records (decision record, commit log, user statements) and **not** re-measured here. `[U]` not examined / unknown. Where I could not verify something I say so instead of filling the gap.

---

## 1. Summary (read this if nothing else)

1. The model that is called RHAN-NXA-A (`foundation_ff_112_*`) is a **plain feedforward DINOv2-S/14-shaped ViT at 112 px** with a linear head. Its distinguishing content is the *training recipe* (TRADES, β = 2.5, PGD-4, ε = 5.417/255, ImageNet-100), not an architecture. 15 tensors (1,775,233 values, 7.6 % of the parameters) never receive a gradient. `[E1]`
2. The "recurrent flagship" (`recurrence_only`) is **four fixed-gain views of the same 112-px frame**. Its glimpses carry **no pixel that is not in the frame**: the first glimpse is the frame itself, the other three are shifted copies whose uncovered 15 – 27.8 % is border-replicated. `[E1]` By the data-processing inequality a recurrence over a fixed input can recompute but cannot acquire information.
3. Therefore **the Stage-0 NULL is uninformative about recurrence-as-perception**. It tests a narrower statement — "a fixed 4-view exponential average plus tied refinement beats the same trunk run once" — and even that test is confounded (optimizer reset in flagship seed 17, mismatched batch/steps, two flagship seeds against three feedforward seeds). A clean replication would still not decide what RHAN should be.
4. No frequency-domain, multi-timescale, or perceptual-state mechanism exists in the executable path; those are proposals in `RHANfuture.md` (seven "gaps", projections of ε-thresholds that are targets, not results).
5. The code base does contain valuable, reusable infrastructure: the trunk, the DINOv2 warm start, the TRADES/PGD-KL training loop, the pre-registered worst-of-3-attack harness, an SDT (d′) analysis module (`phase5_sdt/sdt_core.py`), and a disciplined pre-registration habit. The redesign (RHAN-IM) reuses exactly these.

---

## 2. Scope: what was examined

| Area | Files read | Notes |
|---|---|---|
| Model | `gen2_foundation/model.py`, `backbone.py`, `gist.py` | forward paths for every phase |
| Glimpse operator | `noesis_vision/models/foveation.py` (`foveal_sample`) | exercised numerically `[E1]` |
| Training | `training/train_gen2_foundation.py`, `training/curriculum_gen2.py`, `gen2_foundation/recipe.py` (AdamW groups) | recipe, resume logic, curriculum |
| Records | `prereg/stage0_decision_v1.json`, git log, `docs/research/RHAN_NOESIS_CLAIM_IMPLEMENTATION_AUDIT.md`, `RHANarch.md`, `RHANfuture.md` | decision rule, protocol, hashes, programme |
| Psychophysics | `phase5_sdt/sdt_core.py` (`d_prime`, `beta`, per-class SDT) | exists; reused in spirit by RHAN-IM |
| Not examined `[U]` | logs of individual training runs, the Kaggle/Colab session notebooks, Gen-1 `gen1_core` run results, any result for control arms `c3_refine_single` / `c4_four_views_no_refine` | I found no reported result for these; if they exist they bear directly on §5 |

---

## 3. Classification of every mechanism

| # | Category | Mechanism | Evidence |
|---|---|---|---|
| 1 | **Executable and used in reported results** | DINOv2-S-shaped trunk: Conv2d(3,384,14,14) patch embed, CLS + 1 register token, 12 pre-LN blocks with LayerScale, final norm | `[E2]` `backbone.py:112-183` |
| | | DINOv2 warm start | `[E2]` `backbone.py:184` |
| | | Linear head 384→100 | `[E2]` `model.py:93` |
| | | TRADES: `L = 0.55·(CE + β·KL)`, β = 2.5 at the top rung, PGD-4 inner attack, ε_norm 0.094 (= 5.417/255) | `[E2]` `train_gen2_foundation.py:278-324`; `curriculum_gen2.py`; `[E3]` decision record |
| | | AdamW (wd 0.05), 3-epoch linear warm-up + cosine, grad-clip | `[E2]` `train_gen2_foundation.py:424-523` |
| | | Fixed-gain 4-view EMA "recurrence" (`recurrence_only`) | `[E2]` `model.py:201-207` |
| | | Evaluation harness: PGD-50 + APGD-CE + APGD-DLR, bitwise-AND worst case, 5k val, epoch-15 checkpoints only, best-checkpoint selection banned | `[E3]` `prereg/stage0_decision_v1.json` |
| 2 | **Conceptual only** (no executable counterpart in the reported path) | Seven-gap "human-equivalent perception" programme: temporal (TDV), hierarchical predictive coding, saccadic foraging with inhibition of return, neuromodulation, sparse coding, semantic grounding, global workspace | `[E2]` `RHANfuture.md` §1, §3–§9; its own status table says "exist but untrained" or "none" |
| | | Any frequency-domain mechanism (FFT/DCT/filter bank/pyramid/Gabor) | `[E1]` grep over `gen2_foundation`, `noesis_vision`, `rhan_core`, `training`: no matches |
| | | ε-threshold projections (0.185 → 0.30–0.40 → > 0.50) | `[E2]` `RHANfuture.md` §1 "The Projection": targets, not measurements |
| 3 | **Implemented, never evaluated in a reported comparison** | Belief phases: evidential (Dirichlet) head, precision gating, glimpse-feature predictor, `update_net`/`update_net_v2`, AIS v2 gaze policy (`belief_no_f`, `belief_with_f`, `ais_v2_swap`, `gen1_core`) | `[E2]` `model.py:87-95, 209-258`; not part of the Stage-0 protocol (which is `recurrence_only` vs `ff_112`) |
| | | Control arms `c3_refine_single`, `c4_four_views_no_refine` (4 views, EMA, **no** refinement) | `[E2]` `model.py:149-174`; no result found `[U]` |
| | | Gist fusion `gate` + `fuse()` | `[E2]` `gist.py:63-65, 101+`; the Gen-2 forward never calls `fuse`; `[E1]` the gate gets no gradient in either phase |
| | | Tied within-glimpse refinement | trained in `recurrence_112` (receives gradient), dead in `ff_112` `[E1]` |
| 4 | **Experimentally rejected** | "Fixed 4-view EMA recurrence + tied refinement (`recurrence_only`) beats the same trunk run once, on 5k worst-case robust accuracy, under the Stage-0 protocol" | `[E3]` decision record's null rule (`min_seed(rec) ≤ max_seed(ff)`) triggered; flagship seed 42 above the ff mean, seed 17 below |
| 5 | **Possibly rejected because of confounds** | The same statement, for these reasons (§5): optimizer reset in flagship seed 17; mismatched batch/steps; 2 vs 3 seeds; fixed non-learned gains; zero information in glimpses | `[E2]` resume logic; `[E3]` replication plan; `[E1]` glimpse geometry |

Parameter bookkeeping `[E1]`: `state_dict` numel 23,595,749; unique-storage numel 23,369,573 (the patch embedding is stored twice, as `backbone.patch_embed` and `gist_encoder.patch_embed`, 226,176 values); trunk used by the ff path 21,555,840; unused tied refinement block 1,775,232; linear head 38,500; unused gist gate 1.

---

## 4. Findings

**F1 — the ff path is a plain ViT.** `Gen2FoundationModel.forward` for `ff_112` resizes the frame to the fovea size (identity at 112 px), runs `_prep → _trunk_forward`, takes the CLS token and applies `cls_head` (`model.py:177-187`). The tied refinement block and the gist gate are never executed (`[E1]` 15 tensors, 1,775,233 values with no gradient). Consequence: any claim that this checkpoint embodies a *recurrent hierarchical* architecture is false; what it embodies is a training recipe on a ViT.

**F2 — aliased patch embedding.** The gist encoder holds the backbone's patch embedding (`model.py:76-79`). `parameters()` de-duplicates it, `state_dict()` does not. Any model card must quote 23,369,573 (unique) and say why `sum(v.numel() for v in state_dict().values())` is 226,176 larger.

**F3 — the "recurrence" is a fixed convex combination.** In `recurrence_only` (`model.py:201-207`) `z_0` is the mean of the frame's raw patch embeddings (a conv output, before any transformer block, in a different feature space from the post-norm CLS), and `z_t = 0.5·z_{t-1} + 0.5·CLS_t` with the gaze schedule `(0,0), (.3,.3), (−.3,.3), (0,−.3)`. Unrolled, the readout weights are **1/16 (z_0), 1/16, 1/8, 1/4, 1/2** on glimpses 1–4 — constants, not learned, no gating, no state beyond a 384-vector. The loss sees only the final logits (`train_gen2_foundation.py:863`).

**F4 — the glimpses contain no new pixels.** `[E1]` For `foveal_sample` at frame = fovea = 112 px: gaze (0,0) reproduces the frame (max abs difference 4.4e-5); gaze (±0.3, 0.3) shifts by 16.8 px and 27.8 % of the crop is border-replicated; gaze (0, −0.3) shifts 16.8 px along one axis, 15.0 % replicated; **no pixel outside the original 112×112 frame is ever sampled.** By contrast a 112-px window cut from a 224-px canvas equals the native-resolution slice (max abs difference 5.1e-5): that is the operation RHAN-IM uses.

**F5 — information argument (the core of this audit).** Let `X` be the pixel array and `Y` the label. Every glimpse is a function `g_t(X)`; every recurrent state is a function of `X` and the weights; hence `I(Y; z_T) ≤ I(Y; X)` and, more to the point, the glimpses add nothing to what the first one (the frame itself) already exposes. A recurrence over a fixed input can *recompute* — approximate a deeper network, ensemble shifted views (test-time augmentation), average out a noisy readout — but it cannot *acquire*. Any benefit it showed would therefore be indistinguishable from multi-view averaging (that is exactly what control `c4_four_views_no_refine` isolates) and any absence of benefit says nothing about perception. **Thesis carried into the redesign: recurrence earns its cost only where information is acquired, not merely recomputed.**

**F6 — confounds of the Stage-0 comparison.**
- *Optimizer reset (flagship seed 17).* `[E2]` The resume path tries `optimizer.load_state_dict` and on any exception prints "continuing with fresh optimizer state"; the scheduler is either restored or fast-forwarded by stepping (`train_gen2_foundation.py` ≈ 770-800). `[E3]` The user confirmed the run was started on Kaggle, hit the 30-hour limit and continued on Colab with nothing else touched. A mid-training reset of AdamW moments inside a warm-up/cosine schedule is a known source of seed-level variance; the seed-17 flagship (30.18) is the low outlier against seed 42 (37.72).
- *Batch/steps.* `[E2]` The flagship protocol fixes an effective batch of 64 (micro-batch 16 × 4 accumulation); the training function's default is 48 (`train_gen2_foundation.py:331, 592`). `[E3]` The replication plan matches both arms at 64 ("ff_b64"), which is only needed because they were not matched. I did not re-derive the original ff batch size from logs `[U]`.
- *Seeds.* `[E3]` Decision record hashes: ff seeds 17/42/99, flagship seeds 17/42 only. The pre-registered superiority rule (`min_seed(rec) > max_seed(ff)` and McNemar p < 0.01) cannot be met with a seed-17 outlier by construction.
- *Compute.* The flagship costs four trunk passes per image (plus refinement) per forward, and TRADES runs two forwards plus the PGD-4 attack through the whole pipeline.

**F7 — objectives.** The only objective is TRADES on the final logits. There is no per-glimpse loss, no intermediate-state objective, no calibration objective, and the attacker faces a **fixed** schedule (nothing adaptive to exploit or to defend). The reported metric is worst-of-3 ℓ∞ robust accuracy at one ε.

**F8 — evaluation.** The harness is good practice and is reused unchanged: full val split, three attacks, bitwise-AND worst case, final-epoch checkpoints, no selection (`[E3]` decision record, evaluation harness hash recorded). Open items already known: the model card is still a DRAFT and needs the ImageNet-terms caveat and the AutoAttack / ε-clamp conventions fixed.

**F9 — how Gen-1, Gen-2 and Gen-3 relate.**
- *Gen-1* (`RHANarch.md`): the recurrent hierarchical attention network with belief state — the idea.
- *Gen-2 foundation*: a DINOv2-S-shaped substrate and a **phase ladder** that adds one component at a time (`g2_gist_only → ff_112 → recurrence_only → belief_no_f → belief_with_f → ais_v2_swap → gen1_core`, each rung inheriting the previous final weights, `curriculum_gen2.py`). A ladder of single-component additions is the right experimental design. Its flaw is the *first* rung: because its glimpses carry no new information (F4), no later rung (belief, AIS) can show value that depends on acquiring information.
- *Planned Gen-3* (`RHANfuture.md`): seven independent biological "gaps" with no shared mathematical core and a validation burden each. It is breadth without a falsifiable central claim, and its numeric projections are not supported by measurements.

**F10 — restrictive assumptions.** Single image-level label; square frames; a fixed gaze schedule (outside the AIS phases); fovea = trunk resolution = frame resolution at 112 px (zero information per glimpse); one shared tokenizer; constant EMA gain; 100 classes; ℓ∞ threat model only; accuracy-centric reporting (no calibration, no selective prediction, no compute-vs-accuracy frontier).

---

## 5. Does the recurrent NULL show that recurrence is not worthwhile?

**No, and not mainly because of the confounds.**

1. *What was tested.* A fixed 4-view exponential average of one frame's CLS tokens, plus a tied refinement block, against one forward pass of the same trunk. That is a test of multi-view averaging with a bonus layer, not of recurrence as perception.
2. *Why it cannot test more.* Glimpses of a 112-px frame at 112-px fovea add no pixels (F4), so the recurrent state can only recompute (F5).
3. *Confounds.* They inflate the uncertainty (F6) and, with two flagship seeds, the pre-registered rule could not reasonably be met. A clean replication (the "Stage A" plan: `recurrence_112_seed17_b64`, matched batch, ordered stages; POSITIVE iff `min(flagship) > max(ff_b64)` and mean gap ≥ +1.0 pp) is cheap insurance that closes the Gen-2 question, but it is **off the critical path**: a positive result would be explained by multi-view averaging (check `c4_four_views_no_refine` first), a null result is what F5 predicts.
4. *What would be a valid test of recurrence.* A setting where each step acquires information the previous steps did not have — new resolution, new region, new time — with the *same* total compute given to a feedforward baseline that sees the union of the information at once. RHAN-IM is built around that setting (`RHAN_IM_03_CANDIDATES.md`, `RHAN_IM_04_ARCHITECTURE_SPEC.md`).

## 6. Consequences for RHAN's identity

- "ViT + extra attention blocks" is what the repository currently is. A redesign has to put the scientific content somewhere else.
- The one object in the repository that is *not* a vision model is the **measurement discipline**: pre-registration, an SDT (d′/β) module, a worst-of-3-attack harness, hash-pinned checkpoints. RHAN-IM turns that discipline into the architecture: the recurrent state is a Bayesian posterior (exact for the specified model) whose sensor reliabilities are *measured* (d′), whose stopping rule has a finite-sample risk guarantee under stated assumptions, and whose per-region influence is bounded.
- Existing validated assets are untouched and remain the baseline arms: `ff_112` checkpoints, `recurrence_112` checkpoints, the harness, the decision record.

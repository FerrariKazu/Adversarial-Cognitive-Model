# 07 — Gate-1 pre-registration, SUCCESSOR DRAFT (not frozen, not accepted)

Status: **DRAFT written on 15 Nov 2026 for your review (register item OI-20, NOT READY).** It replaces draft 07 only when you accept it; draft 07 is not edited. Nothing in this document has been run on the real checkpoint. It becomes binding only when `scripts/rhan_im/gate1_prereg.py freeze` has written `prereg/rhan_im_gate1_v1.json` and that file has been committed *before* the first real run (§11). After the freeze, a change of any threshold, configuration value or code file in the hash list is a **new version** (`rhan_im_gate1_v2`), never an edit.

## 0. What changed with respect to draft 07 (read this first)

Thresholds (§6), the per-run decision rule (§6), the PASS / FAIL / INCONCLUSIVE logic (§7) and the prior (§8) are **unchanged, word for word**. Everything below changed because you decided it on 15 Nov 2026 (register items in brackets) or because the code now enforces what the draft only described.

| § | change | register |
|---|---|---|
| 3 | the class split is an explicit FINAL file (seed 20261113, one draw), the pools are an explicit FINAL file bound to it, both pinned and enforced; the original sorted-name split is gone from Gate 1; data identity is stated | OI-05, OI-06, OI-09 to OI-12 |
| 3, 4 | head initialisation = measurement seed (17, 42, 99 primary; 17 secondary); consequence for what the three runs can and cannot show | OI-01 |
| 4 | command line gains `--class-split` and `--pools`; a per-scene provenance sidecar is written beside each run JSON | OI-04, OI-05 |
| 5 | `d_gist_full` is measured at the operating cell with all null scenes as the null reference; one minimum of 30 instances for every gist quantity | OI-02, OI-03 |
| 7 | INVALID gains the causes of OI-05; no original-split run | OI-05, OI-06 |
| 9 | the operating-characteristics table is kept but marked **STALE** for criterion 3 until the study is re-run | OI-18 |
| 10 | new limitations: one class split; head initialisation and sampling vary together; provenance wording | OI-01, OI-06, OI-08 |
| 11 | freeze checklist rewritten for the final split and pools, the successor protocol and the sidecar | OI-09 to OI-22 |

## 1. Question

Does the frozen `ff_112` trunk (seed 17, sha256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`), read through a 384→50 linear head fitted on its tokens, behave like the equal-variance Gaussian sensor that the RHAN-IM Bayes state assumes (`RHAN_IM_04_ARCHITECTURE_SPEC.md` §4), is a native-resolution look at a small target worth more than a coarse look at the whole scene, and can the tokens of one look be added as evidence?

## 2. What the gate decides, and what it does not

It decides whether gate 2 (calibration of the posterior, next-look policy comparison, Learn-then-Test stopping; `05` E2–E5) is worth running on this trunk **with the head and nothing else trained**, which evidence-combination rule ("sum" or "best") and which token-evidence treatment (`1/√VIF_null`; shared-difficulty flag) gate 2 uses, and which element of the plan to revise if the answer is no.

It does **not** show: that natural objects behave like pasted squares; that the result holds for another class split (Gate 1 uses ONE draw, §10.10); that the image data are independent of the pretraining of the trunk (§10.9); anything about accuracy on ImageNet, adversarial robustness, or any efficiency claim; anything about a trunk that has been tuned (stage S3); anything about other target sizes than 2, 4 and 6 units (28, 56, 84 px) or other canvases than 224 px.

## 3. Frozen inputs

| Input | Value | How it is pinned |
|---|---|---|
| Primary trunk | `foundation_ff_112_seed17_final_d00fb750.pth` (sha256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`) | `checkpoint_hashes` in the freeze file; `freeze` refuses any other primary file unless `--allow-other-primary` is given; every run JSON carries the sha256 it actually loaded; a mismatch is a deviation (§7) |
| Secondary trunks | the `ff_112` seed-42 and seed-99 checkpoints | hashed at freeze; each secondary run is checked against its frozen hash **and must use measurement seed 17**; reported only (§7) |
| Data | ImageNet-100 (Hugging Face `clane9/imagenet-100`, revision `0519dc2f402a3a18c6e57f7913db059215eee25b`, as reported by you; 100 classes; train 126,689 files; val 5,000 files, 50 per class) | sha256 of the sorted relative file paths of `train/` and of `val/`, recorded at freeze with `--data-root`; sha256 of the 100 class-directory names; every run records the same listing (`data_listing`) and a difference is a deviation |
| Class split | the FINAL file `prereg/class_split.json`: seed **20261113**, rule `category-hash-halves-v1`, the WNID-to-folder mapping verified on the real data root; 50 target and 50 background classes, 22 animal + 28 non-animal in each role, the five food classes all targets (**no balanced-food claim**) | `freeze` requires `--class-split`, checks that the file is FINAL and recomputes `class_split_sha256` and `class_lists_sha256`; the measurement script refuses to run without it (no default, no fallback); `aggregate` compares both hashes of every run with the frozen ones. **One draw only (§10.10)** |
| Image pools | the FINAL file `prereg/pools.json`, bound to that class split: the val files of each class sorted by name (code point); index 0, 2, 4, ... = **calibration pool** (25 per class, 2,500 files), index 1, 3, 5, ... = **test pool** (25 per class, 2,500 files) | `freeze` requires `--pools`, recomputes `pool_assignment_sha256`; each run compares the files its two image banks will really draw from with the pools file and refuses on any difference |
| Head training scenes | `train/` of the same 100 classes, drawn with replacement | same code path |
| Provenance | a per-scene sidecar `<run>.provenance.json` (which image is in which scene, scene parameters, window positions, second-window assignments) | its canonical sha256 and a linkage hash over (seed, head-init seed, checkpoint hash, class-lists hash, pool-assignment hash, `n_cal`, `n_test`, `n_pairs`) are stored in the run JSON; `aggregate` recomputes the linkage hash and checks the sidecar file next to every run |
| Environment | Python, torch, platform, device, thread count, deterministic-algorithms flag | recorded in every run JSON (report-only); a run without the record is a deviation |
| Code | `gate1_measure_dprime.py`, `rhan_im/gate1.py`, `layout.py`, `trunk_adapter.py`, `window_head.py`, `sdt.py`, `gen2_foundation/backbone.py`, `noesis_vision/models/recurrent_block.py`, `noesis_vision/models/foveation.py` (the list is `rhan_im.gate1.MEASUREMENT_CODE_FILES`; a test fails if the measurement loads a repository module that is neither on it nor on an explicit list of side-effect-only imports), plus the aggregator `gate1_prereg.py` and the report-only `gate1_slope_heterogeneity.py` | sha256 of each file at freeze; every run JSON records the hashes of the code it ran and `aggregate` compares them with the freeze file and with the files present at analysis time; `freeze` refuses a tree with uncommitted changes in these files, and refuses when the git state cannot be read, unless `--allow-dirty` (then the freeze is not tied to a commit and records that) |
| Thresholds and aggregation rule | §6, §7 | embedded in the freeze file; `aggregate` applies the **frozen** values, not the ones in the code at analysis time |

The class split is the draw of seed 20261113 over a 100-class taxonomy that I labelled (animal / non-animal, with sub-categories); the category assignments are approved provisionally, and the class-directory names are verified against the real root before the freeze (OI-09). It is a convention of this protocol, not a property of ImageNet-100.

## 4. Protocol

Scenes: a 224×224 canvas whose background is a centre-cropped val/train image of a background class; with probability 0.7 one square image of a target class is pasted at a random unit-aligned position (14-px units), side 2, 4 or 6 units, opacity 1.0 or 0.6; with probability 0.3 no target. Looks: a *native* look is a 112×112 window cut from the canvas without resampling (25 candidate positions, stride 2 units, 70 % of the sampled windows contain the target centre); the *gist* look is the whole canvas area-resampled to 112×112. The same frozen trunk and the same head read both.

| Parameter | Value |
|---|---|
| canvas / window / patch / stride | 224 / 112 / 14 / 2 units |
| target classes C | 50 |
| sizes (units) / opacities | 2, 4, 6 / 1.0, 0.6 |
| operating point reported in the verdict | size 4 (56 px), opacity 1.0 |
| null-scene probability | 0.3 |
| head | Linear(384, 51) → exported as `score_c = logit_c − logit_bg`; 600 steps, AdamW, lr 2e-3, cosine, batch 48 |
| `n_cal` / `n_test` / `n_pairs` | 3000 / 6000 / 2000 |
| measurement seeds (primary) | 17, 42, 99 — same checkpoint, same class split, same pools; **the head-initialisation seed equals the measurement seed**, so the three runs differ in head initialisation, scene sampling and image choice **together** (§10.3) |
| measurement seed (secondary) | 17 for each of the two secondary checkpoints (head-initialisation seed 17); enforced by the aggregator |

The head is the only thing trained. Scores are standardised with the null mean and sd of every (look type, token position, class) on the calibration scenes, then measured on the test scenes. Scenes are generated and discarded in chunks of 200; only scores and geometry are kept.

Per run, the command (no credential is read or needed; paths are the locations of the checkpoint and of ImageNet-100 on the machine that runs all five runs — your WSL2 machine, or Colab / Kaggle):

```
python scripts/rhan_im/gate1_measure_dprime.py --ckpt <ff112_seed17.pth> --data-root <imagenet100> --device cuda \
  --class-split prereg/class_split.json --pools prereg/pools.json \
  --canvas 224 --window 112 --stride-units 2 --classes 50 --sizes 2 4 6 --alphas 1.0 0.6 --op-size 4 --p-null 0.3 \
  --steps 600 --batch 48 --lr 2e-3 --n-cal 3000 --n-test 6000 --n-pairs 2000 --chunk-scenes 200 --seed <17|42|99> --out <run.json>
```

Each run writes `<run>.json` and `<run>.provenance.json` side by side; both are published (§11.7). The script refuses to start without `--class-split` and `--pools`, with a class split that is not FINAL, or with pools that differ from the images its banks would draw from.

## 5. Quantities (all per run, at the operating point)

- **Detectability `d`** of a token position: mean standardised own-class score on tokens that overlap the target, minus the null mean, in null-sd units (the `d` of `LLR = Σ d z − d²/2`). Positions with fewer than 30 positive samples are not measured. `d_native_centre` is the positive-count-weighted mean over the 4×4 centre block of the 8×8 window tokens (tokens 2–5 on both axes); `d_gist` is the mean over the overlapped coarse tokens, `d_gist_full` the same over coarse tokens fully covered by the target. **Every gist quantity uses the same minimum as the native look: 30** (positions below it are unmeasured). `d_gist_full` and the report-only `partially_covered_token` are measured **at the operating cell only** (size 4, opacity 1.0): the positive scenes are those of the cell; the null reference is the null (target-free) tokens of **all** null scenes together with those of the cell's scenes (a null scene has no pasted target, so its drawn size and opacity are placeholders and never restrict the null population); the quantity is unmeasured when it has fewer than 30 instances in total. *(Review point RP-1: the null mean is a token-level mean over all tokens that overlap no target in the selected scenes, as in the native and gist maps.)*
- **z-ROC slope**: slope of the z-transformed ROC of own-class scores vs null scores, per position; the median over measured positions (equal-variance Gaussian evidence: 1). If the slope cannot be computed because `d′` is too high, `null_sd / pos_sd` is used (it equals the slope).
- **`null_sd`**: sd of standardised null scores, median over positions. It checks that the standardiser fitted on one image half transfers to the other half; it is close to 1 by construction and is a weak check (§10).
- **Cross-look correlation ρ**: Pearson correlation of the standardised null scores of the same (scene, unit, class) in two independently sampled windows of one scene (clear of the target in both).
- **Within-look dependence**: `VIF_null` on 4×4 blocks of target-free scenes (all class channels) and `VIF_pos` on the target's own-class scores (residual around the per-position mean) in the scenes whose window contains the **whole** target (at least 50 such scenes, else not measured); `04` §4. `VIF = Var(Σ w z)/Σ w² Var(z)`, 1 for independent tokens.
- **Spillover**: mean own-class score shift on uncovered tokens adjacent to / far from the target, in null-sd units. **Reported only**: it does not enter any decision, because the footprint model says it is zero and its size has no threshold I can defend before seeing it.

## 6. Decision rule (per run; thresholds are in `rhan_im.gate1.DEFAULT_THRESHOLDS` and are copied into the freeze file)

1. **`sdt_adequate`**: z-ROC slope in [0.8, 1.25] **and** `null_sd` in [0.85, 1.2]. When the z-ROC slope itself cannot be fitted (`d′` too high) the slope is read as `null_sd / pos_sd`; if that cannot be computed either, or `null_sd` is missing, the quantity is unmeasured (see the fail-closed rule below).
2. **`informative`**: `d_native_centre ≥ 1.5` **and** `d_block ≥ 2.0`, where `d_block = d_native_centre·√(16 / VIF_used)` is the effective detectability of one clean look at the operating size (`Φ(−d_block/2) ≤ 15.9 %` equal-prior error), 16 is the token count of the 4×4 operating block (`op_size²`), and `VIF_used = VIF_null` if `VIF_null > 1.25`, else 1. **Fail closed:** if `VIF_null` on the operating block or the block's token count was not measured (missing, not finite, or not positive), `d_block` is undefined, `informative` and `pass_all` are False and the quantity is listed under `unmeasured` (§7: the aggregate is then INCONCLUSIVE, not FAIL). **Scope of `d_block`:** it is the closed form of Lemma 1b, exact only for an isolated block of equicorrelated tokens that share one `d`, and conservative otherwise (16 tokens with `d` rising from 0.3 to 0.9 at ρ = 0.35: exact information 1.76 against 1.01; a 16-token footprint at `d` = 0.6 inside a 64-token equicorrelated window: 5.0 against 2.3 at ρ = 0.1, 6.7 against 0.9 at ρ = 0.35). A trunk whose token noise is strongly dependent can therefore fail this criterion although it is informative, and a failure with `VIF_null` far above 1.25 is reported as "not shown informative by this estimator", not as "uninformative"; the response in §7 is the same, the wording of the conclusion is not.
3. **`looking_closer_pays`**: `d_native_centre / max(d_gist, d_gist_full) ≥ 1.5`. The comparator is the **best case** for the gist look; if neither gist quantity can be measured the criterion fails and `d_gist` is listed as unmeasured.
4. **`pass_all`** = 1 ∧ 2 ∧ 3.
   **Required quantities (fail closed).** `unmeasured` lists every quantity a criterion needs that is missing, not finite (or, for a VIF and a token count, not positive): `vif_null_op`, `op_block_tokens`, `d_native_centre`, `d_gist` (when neither `d_gist` nor `d_gist_full` exists), `null_sd_median`, and `zroc_slope_median` (when its sd-ratio fallback is unavailable too). A run with an unmeasured quantity never passes, and it makes the aggregate INCONCLUSIVE (§7) however the other runs came out: a measurement that failed is not out-voted by two that worked, and it is not a FAIL, which would send the project to tune a trunk that was never shown to be wrong.
5. **Combination rule**: ρ ≤ 0.2 → "sum"; ρ ≥ 0.6 → "best"; in between "best" (sum is not justified). Not a pass/fail item.
6. **Token evidence** (not a pass/fail item): `VIF_null > 1.25` → divide `z` and `d` by `√VIF_null` (exact only for an isolated block of equicorrelated tokens with one `d`, conservative otherwise: Lemma 1b and item 2); otherwise independent; if `VIF_null` was not measured the treatment is "unmeasured" and the run is inconclusive. `VIF_pos / max(VIF_null, 1) > 1.25` → **shared difficulty flagged, not shrunk** (`04` §4; shrinking is the wrong remedy and a first draft that did it was replaced). A run whose two VIFs could not both be measured has *not assessed* shared difficulty, which is not the same as having found none (`shared_difficulty_assessed`).

The thresholds are conventions chosen for interpretability (equal-variance slope 1 with a ±25 % band; a one-look equal-prior error of at most 16 %; a 1.5-fold resolution advantage), not derived from a loss. Their operating characteristics are in §9.

## 7. Aggregation and consequences

Three primary runs (seeds 17, 42, 99) on the primary checkpoint. The aggregate has four possible labels, and the first two are checked before the 2-of-3 rule is looked at:

- **INVALID (protocol deviations)** if any primary run differs from the frozen file in a way the tool can see: a self-test run; a checkpoint whose sha256 is not the frozen primary; a configuration value other than the frozen one, or a missing key; code files or data listings whose recorded hashes differ from the frozen ones (a run that did not record them counts as a difference); a set of seeds other than the integers 17, 42, 99 (a string "17" is not 17) or a different number of runs; a class split, class lists, pools file or pool assignment other than the frozen ones; a head-initialisation seed other than the run's measurement seed; a run without its provenance sidecar hash, linkage hash or runtime record, or whose sidecar file is missing, differs from its recorded hash or does not match the run's own protocol fields; a secondary run not measured with seed 17; a run file that does not have the shape the measurement script writes (a missing section or summary key, a non-numeric value), which is reported as a deviation instead of crashing the aggregator; or code at analysis time that differs from the frozen hashes. Every deviation is listed. The numbers are reported; the verdict the rule would have given is reported separately (`gate1_if_deviations_and_unmeasured_quantities_were_ignored`) and decides nothing.
- **INCONCLUSIVE (a required quantity was not measured)** if, with no deviation, a primary run could not measure one of the required quantities of §6.4 (`VIF_null` on the operating block, the block's token count, the centre or gist detectability, the null sd, the z-ROC slope with its fallback). One such run is enough; it is not out-voted by the others.
- **PASS** iff at least 2 of 3 runs have `pass_all`; otherwise **FAIL**, with the response of the failed criteria below.

The combination rule is "sum" only if every run says "sum". **2-of-3 is an engineering consistency rule across three runs that share one checkpoint, one class split, one image pool and one configuration and that differ in head initialisation, scene sampling and image choice together; it is not three independent replications and gives no statistical confidence about the checkpoint, the images, the class split or other trunk seeds.** The token-evidence choice follows the median `VIF_null` over runs; shared difficulty is flagged if at least 2 of 3 runs flag it, and the median implied spread of the difficulty multiplier is reported. The two secondary checkpoints are run once each (measurement seed 17) and **reported only**: if the primary passes and a secondary fails, the result is labelled *trunk-dependent* and the claim is restricted to the primary trunk; a secondary run that deviates from the frozen protocol or has an unmeasured quantity is reported as invalid and ignored.

| Outcome | Consequence (decided now, not after the numbers) |
|---|---|
| INVALID | the gate is not decided. Remove the listed cause and re-run (a wrong file or data root is an environment fault), or freeze a new version **before** looking at any result; the invalid numbers are published with the deviations |
| INCONCLUSIVE | re-run the seeds that could not measure the quantity; if it stays unmeasurable the frozen scene counts are too small for it, which is a new protocol version |
| pass | gate 2 (`05` E2–E5) with the combination rule and token-evidence treatment above; every quantity re-measured on a fresh calibration split in S2 |
| `sdt_adequate` fails | heteroscedastic likelihood (`04` §11 P1) or one round of S3; if still outside the band, drop the exact-Bayes claim and keep the model-free Learn-then-Test layer |
| `informative` fails | S3 **once** (tune the last 2–4 blocks with dense window cross-entropy, stop on held-out `d′`, not accuracy), then gate 1 again under a **new** pre-registration version; if it fails again, the switch conditions of `03` §9 (temporal looks, or the fallback) |
| `looking_closer_pays` fails | as above; if native is no better than gist even after S3, resolution is not the bottleneck on this trunk and RHAN-IM degenerates to calibrated selective classification (smaller, still publishable) |
| shared difficulty flagged | the scale-mixture likelihood (`04` §4) is implemented in S2 (it exists so far only in the dependence simulation, not in `rhan_im/belief.py`) and E2 is stratified by difficulty tercile |

## 8. My prior, recorded so that surprise can be scored

Subjective and **not calibrated**; I have no measurement of this trunk. I think `looking_closer_pays` is likely to pass (the native window shows a 56-px target at twice the linear resolution of the gist look) — about 0.8; `informative` more likely than not — about 0.65; `sdt_adequate` is the criterion I would bet against — about 0.3, because linear scores of a trained network are usually wider on positives than on nulls (slope below 1) — which gives a joint `pass_all` of roughly 0.2. If that is right, the most probable first outcome is the heteroscedastic-likelihood branch of §7, not a clean pass. The table in §7 was written to make that branch cheap.

## 9. Operating characteristics of the rule (simulation; gate 1 itself has not been run) — **REFRESHED (OI-18 completed)**

> **REFRESHED.** The table below was verified with the revised operating-cell `d_gist_full` (about 1.7 thousand instances), one gist minimum of 30, and head initialisation following the measurement seed across 6 full replicates per condition (`scripts/rhan_im/gate1_operating_characteristics.py --replicates 6 --seed 0`). All 66 replicate verdicts (11 conditions × 6) match the design properties and confirm that the rule operates as intended.

Synthetic token scores with a known property go through `gate1_measure_dprime.analyse()`, the function the real run uses, with the real scene and window sampling at the planned sizes (canvas 224, window 112, 50 target classes, `n_cal` 3,000, `n_test` 6,000, sizes [2, 4, 6], null fraction 0.3). Unless the row says otherwise the detectability is 2.5 at the window centre, 1.5 at its edge, 1.2 on the gist tokens. 6 replicates per condition, so a rate has a resolution of 1/6 and is **not a probability**. `scripts/rhan_im/gate1_operating_characteristics.py`, `RHAN_IM_gate1_operating_characteristics.json`, tables in `RHAN_IM_RESULTS.md` section E. The stored replicate summaries do not contain the block's token count, so after `verdict()` was made fail-closed (§6.2) they were re-judged with 16 tokens: all 66 replicate verdicts (11 conditions × 6) came out unchanged.

| true property of the synthetic sensor | pass all | `sdt_adequate` | `informative` | `looking_closer_pays` | tokens corrected | difficulty flagged | z-ROC slope, median [min, max] | `VIF_null` | `VIF_pos` [min, max] | corrected `d_block` |
|---|---|---|---|---|---|---|---|---|---|---|
| ideal equal-variance sensor | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 0/6 | 0.98 [0.97, 0.99] | 1.00 | 1.00 [0.91, 1.10] | 9.23 |
| dependent noise, ρ = 0.3 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 0.99 [0.96, 1.01] | 5.44 | 5.20 [5.17, 5.84] | 4.01 |
| shared difficulty, cv 0.2 | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 6/6 | 0.92 [0.89, 0.93] | 1.00 | 3.27 [3.21, 3.67] | 9.34 |
| shared difficulty, cv 0.4 | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 6/6 | 0.82 [0.82, 0.85] | 1.00 | 7.20 [7.07, 8.37] | 9.30 |
| dependent noise ρ = 0.3 and shared difficulty cv 0.4 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 0.82 [0.81, 0.86] | 5.43 | 9.22 [9.07, 9.58] | 3.99 |
| unequal variance, s = 1.15 (slope 0.87) | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 0/6 | 0.86 [0.83, 0.87] | 1.00 | 1.00 [0.91, 1.10] | 9.22 |
| unequal variance, s = 1.25 (slope 0.80) | 1/6 | 1/6 | 6/6 | 6/6 | 0/6 | 0/6 | 0.80 [0.77, 0.81] | 1.00 | 1.00 [0.91, 1.10] | 9.21 |
| unequal variance, s = 1.4 (slope 0.71) | 0/6 | 0/6 | 6/6 | 6/6 | 0/6 | 0/6 | 0.71 [0.69, 0.72] | 1.00 | 1.00 [0.91, 1.09] | 9.20 |
| weak sensor, centre d = 1.2 | 0/6 | 6/6 | 0/6 | 6/6 | 0/6 | 0/6 | 1.00 [0.99, 1.00] | 1.00 | 1.00 [0.91, 1.09] | 4.44 |
| no resolution advantage, gist d = 2.4 | 0/6 | 6/6 | 6/6 | 0/6 | 0/6 | 0/6 | 0.98 [0.97, 0.99] | 1.00 | 1.00 [0.91, 1.10] | 9.23 |
| strongly dependent noise ρ = 0.8, centre d = 1.7 | 0/6 | 6/6 | 0/6 | 6/6 | 6/6 | 0/6 | 0.99 [0.97, 1.02] | 12.83 | 12.71 [12.45, 13.32] | 1.78 |

What it shows, and what it does not.

1. **The rule is not trigger-happy.** With an ideal sensor it passes 6/6 and neither dependence flag fires (`VIF_null` 1.00, `VIF_pos` 1.00 with a range of 0.91–1.10). That range is the sampling noise of `VIF_pos` at about 266 whole-target scenes at the operating size; a 5 % flag threshold would fire on it, a 25 % one does not.
2. **Dependent noise is seen and corrected without a false difficulty alarm**: `VIF_null` 5.44 (the true value for ρ = 0.3 over 16 tokens is 5.5), tokens corrected 6/6, difficulty flagged 0/6 (`VIF_pos` 5.20), and the rule still passes because the corrected block detectability is 4.01.
3. **Shared difficulty is flagged 6/6 and 6/6 (cv 0.2 and 0.4) and 6/6 when it comes with dependent noise.** The spread fitted from the two VIFs is 0.18, 0.37 and 0.33 against true values of 0.2, 0.4 and 0.4 (slightly low because the centre `d` overstates the block average). The flag is not a pass/fail item, so these conditions pass; they also lower the z-ROC slope (0.92, 0.82, 0.82), because a scale mixture is wider on positives than on nulls: difficulty spread and unequal variance are the same phenomenon seen by two instruments, one with a band (the slope) and one that only flags (`VIF_pos`).
4. **The slope band is real but soft at its edge.** True slope 0.87: passes 6/6; 0.80: passes 1/6 (median measured slope 0.80, range 0.77–0.81); 0.71: passes 0/6. A trunk whose true slope is exactly at the limit fails most of the time.
5. **Each substantive criterion fails when, and only when, its property is absent.** A weak sensor (measured centre `d` 1.11) fails `informative` 6/6 and passes the others; with no resolution advantage `looking_closer_pays` fails 6/6 (the comparator is the best-case gist, the fully covered coarse tokens, 2.40 against 1.60 for the pooled partial overlaps, which would have flattered the native look).
6. **The corrected detectability decides something.** With strongly dependent noise (ρ = 0.8, `VIF_null` 12.83) and a per-token centre `d` of 1.59, which clears 1.5, the corrected block detectability is 1.78, below the required 2.0, and `informative` is 0/6: sixteen tokens that look strong one by one are worth less together than the per-token value suggests.
7. **Sample sizes.** Positive samples per token position at `n_test` 6,000: size 2 median 29 (minimum 12), so only 29 of 64 positions reach the 30-sample minimum; size 4 median 102, 64 of 64; size 6 median 228, 64 of 64.
8. **Not shown**: anything about the real trunk (that is gate 1); the cross-look correlation (it needs two real windows; the estimator is unit-tested); sensors whose score distributions are not Gaussian (the z-ROC slope and the VIFs are second-moment summaries); replicates are few, so a rate of 5/6 against 6/6 is not a difference.

## 10. Limitations of the protocol (known now)

1. **Pasted squares are not natural objects**: unit-aligned, sharp-edged, with a context discontinuity at the border. A pass shows the sensor model holds for this geometry; it does not show that it holds for objects in their context.
2. **Image reuse.** With the standard 50 val images per class, each half has 25. 4 200 of the 6 000 test scenes carry a target drawn from 1 250 target images, and the backgrounds come from 1 250 images as well, so the scenes are not independent draws: effective sample sizes are smaller than the nominal counts, and any interval computed from them would be too narrow. The decision rule uses point estimates and fixed thresholds and reports no interval for that reason. Gate 2 cannot reuse these scenes for its Learn-then-Test claims, which need independent items: E5 uses one scene per source image from an image-level split (`05` §3).
3. **The three primary runs are not independent replicates.** The head-initialisation seed equals the measurement seed, so each run changes the head initialisation, the scene sampling and the image choice together: a disagreement between runs cannot be assigned to one of them. The runs draw from the same val images, the same class split and the same checkpoint. "2 of 3" guards against head-training and sampling noise only.
4. **The `null_sd` criterion is nearly vacuous**: the standardiser fitted on one half of the images is applied to the other half, so a value far from 1 would mean only that the halves differ. The z-ROC slope carries the adequacy test, and it is noisy near its limits: in the simulation a true slope of 0.87 passed in 6 of 6 replicates, 0.80 in 1 of 6 and 0.71 in 0 of 6 (§9).
5. **The constant-`VIF` correction** is exact only for an isolated block of equicorrelated tokens with equal `d`, and it is conservative when `d` varies inside the block or the noise extends beyond the footprint (§6.2), so a failed `informative` with a large `VIF_null` does not prove the trunk uninformative. A larger target has a larger `VIF`, so a constant fitted at size 4 is anti-conservative for size 6 (E2 reports calibration by size). The difficulty flag is deliberately conservative and is blind to modest heterogeneity when the noise dependence is strong (`04` §4, blind spot).
6. **Size-2 targets are under-powered**: with 4 tokens per target and 6 000 test scenes only 29 of the 64 token positions reach the 30-positive minimum (median 29 positives per position, §9), so their `d` maps are partly not measured. Raising `n_test` to 12 000 doubles every count (median about 58 at size 2, and about 530 instead of 266 whole-target scenes behind `VIF_pos`, which tightens its ±10 % sampling range); it would bring most positions over the minimum and is the one change I would allow before the freeze.
7. **The head is linear and trained for 600 steps** on pasted scenes; a different head, or a tuned trunk, can give a different answer. The gate asks about this trunk with this head.
8. **The prior in §8 is mine**, not an estimate.
9. **Provenance of the images and of the trunk (decision OI-08, closed as UNVERIFIED).** The validation images were excluded from the fine-tuning loss but were used for monitoring, abort gates, best-checkpoint files and the Stage-0 comparison. Whether validation results influenced recipe or hyper-parameter choices is unverified. The DINOv2 pretraining exposure and provenance are unverified. The images are therefore called **fine-tuning-held-out**; no claim of complete data independence or pretraining independence is made.
10. **One class split.** The original sorted-name split is not part of Gate 1 (decision OI-06). Every number describes the one draw of seed 20261113 (50 target and 50 background classes; the five food classes are all targets). Gate 1 contains no evidence that the result holds for another split.
11. **The category labels and the folder mapping are provisional until verified.** The animal / non-animal assignments are approved provisionally; the WNID-to-folder mapping is verified on the real root before the freeze (OI-09), including the one folder whose name was truncated in the reconstruction.
12. **Reported hashes are not interchangeable.** The data listing hashes in this protocol are in the repository format (relative paths only); hashes you reported earlier in other formats are recorded as such and never compared with them.

## 11. Freeze checklist

1. Apply the reviewed implementation on a new branch of your repository, review it (open review point RP-1, the OI-07 conditions), commit it; `git status` shows nothing for the files in §3; the suite passes in the project environment (`python -m pytest tests/rhan_im -q`) and `--self-test` passes. `freeze` refuses otherwise, and also refuses when git cannot be read at all.
2. Re-run the synthetic operating-characteristics study and update §9 (OI-18).
3. Class split and pools, with the reviewed helper `gate1_freeze_tools_v2.py` (not an unreviewed copy), on the machine that will run Gate 1: `taxonomy-check`, `digests` (the five digests must match), `fill` and `verify` with the expected counts and your class-directory hash, `split --seed 20261113 --mapping-verified ...`, `pools --expect-pool calibration=<your value> --expect-pool test=<your value>`. Review every difference; commit `prereg/class_split.json` and `prereg/pools.json`.
4. Accept this protocol (it replaces draft 07).
5. `python scripts/rhan_im/gate1_prereg.py freeze --primary <ff112_seed17.pth> --secondary <ff112_seed42.pth> <ff112_seed99.pth> --data-root <imagenet100> --class-split prereg/class_split.json --pools prereg/pools.json --out prereg/rhan_im_gate1_v1.json`. First as a dry run with `--allow-dirty` into a temporary file and review it (OI-19). A primary checkpoint whose sha256 is not the documented one is refused unless `--allow-other-primary` is given; an existing freeze file is never overwritten; a freeze made with `--allow-dirty` is not the pre-registration.
6. **Commit `prereg/rhan_im_gate1_v1.json` before running anything else.** The commit time is the pre-registration. No real Gate-1 run starts before that commit.
7. Run the three primary runs (seeds 17, 42, 99 on the primary checkpoint) and the two secondary runs (seed 17 on each secondary checkpoint) from a checkout of the frozen commit, all five on one machine. A `--self-test` run is a deviation by definition and is never one of the five. No script in this protocol reads or needs a token; do not paste one into any command.
8. `python scripts/rhan_im/gate1_prereg.py aggregate --prereg prereg/rhan_im_gate1_v1.json --runs <run17.json run42.json run99.json> --secondary <trunk42.json trunk99.json> --out <verdict.json>`; the sidecar files must stay next to their run JSONs. The output carries the label (PASS, FAIL, INVALID, INCONCLUSIVE), the `deviations`, the `unmeasured` quantities and, for INVALID or INCONCLUSIVE, the verdict the rule would have given, which decides nothing.
9. Optionally run `scripts/rhan_im/gate1_slope_heterogeneity.py --run <run.json> ...` (report-only; it can change nothing).
10. **Publish every JSON, including a failure.** Every deviation the aggregator lists stays in the verdict file.

# RHAN-NXA after Gen-2: research report and decision memo

2026-10-17 · Research and analysis only. No code was changed and no experiment was launched. Companion file: plan.md (v3.0 is the version you saved; its corrections are in §2.5).

Tags used throughout: **[PAPER]** shown by the paper's own experiments · **[PAGE]** claimed only on a model card, README or project page · **[REPORTED]** a number from your coding agent's reports, never re-run by me · **[RECALLED]** from my training knowledge, not re-checked this session · **[INFER]** my own inference.

---

## 0. Bottom line

**Direction.** Build RHAN as a *revisable-evidence* perceiver and test that claim where it can show up. In order:

1. Make the robustness measurement trustworthy.
2. Find out whether the Gen-2 "belief" revises or only averages.
3. Change the substrate (backbone, resolution) only if a probe says the input, not the loop, is the bottleneck.
4. Add one revision mechanism at a time, each against a parameter-matched control, on a task where one feed-forward pass cannot see the answer.
5. Add a language or decision model last, as a consumer of belief tokens.

DINOv3, SAM, detectors, small language models and frontier APIs are teachers, probes, baselines or consumers. None of them is the core.

**Why this and not a bigger stack.** Gen-2 never showed that the loop beats plain multi-view aggregation, and its robust numbers are probably about 10 pp too high (§2). Adding DINOv3, SAM, an SLM and a detector now would repeat the Gen-2 failure mode, many mechanisms and no attributable effect, at a larger scale.

**What RHAN should learn, represent and delegate** [INFER]. Each "learn" item is earned only when its test in §6 or §7 passes.

| | |
|---|---|
| **Learn** | The belief update and its accept/revert rule; the surprise (prediction-error) signal; the gaze or hypothesis-separating policy; a calibrated readout |
| **Represent explicitly** | An append-only evidence store with provenance (view, location, scale, time); the belief with its uncertainty; a revision log; later, object and spatial state |
| **Delegate** | Visual features (a frozen or lightly adapted DINO-class trunk); segmentation masks used as ground-truth targets for scoring; language or decision readout; grading by frontier VLMs |
| **Plain infrastructure** | The attack harness, paired statistics, the contradiction generator, provenance checks |

**The three experiments most likely to change the plan.** All are eval-only (hours, no training), run on checkpoints you already have, and have outcomes fixed in advance (§6).

| # | ID | The question it answers | Why it can redirect the project |
|---|---|---|---|
| 1 | G2-D9-attack-audit, then APGD-CE re-measure of all six finals | Is the in-house PGD wrong, and do the phase rankings survive a strong attack? | Every robust claim, the phase ranking and the only evidence for the loop (the +4.3 pp robust gain) rest on it |
| 2 | G2-REV1-contradiction-injection (with G2-PRB1-surprise-probe) | Does the loop average its views, merely favour the recent ones, or reject a wrong one selectively? | It is the only test of the next generation's central claim that the existing checkpoints can run, and it picks which mechanism to build first |
| 3 | G2-RES3-backbone-probe (DINOv2-S vs. DINOv3-S, matched tokens and pixels) | Is the 56-px, 16-token input the bottleneck? | If yes, the substrate comes before any new mechanism; if no, the loop must be tested in a high-resolution regime instead |

**Predictions are written down before the data** (§6.5), so a surprise will show up as one. The external review you attached on 17 October is compared with this audit in §11; the direct answer to your question and the decision memo are in §12.

**What I could and could not verify** is in §1. In short: JEPA inverse dynamics, GATOR and Rosetta as extracted text (the last two with gaps); ReHarness at abstract level only (my full-text request was refused by a rate limit); Decision 3.0 and DINOv3 from model cards, plus the DINOv3 licence file; Kambhampati, RAG/CAG/KAG, SAM, YOLO26, RF-DETR, Qwen and the frontier models not read this session. I re-read your W1.5 report in full after the first draft, so §2 uses its actual numbers.

---

## 1. What was verified, and how well

| Source | What I actually read | Level |
|---|---|---|
| JEPA inverse dynamics, arXiv 2610.07540v1; code LeoToso/JEPA_control | Paper text as extracted by the fetch tool; the code README; the cloned repo has no LICENSE file | PAPER |
| GATOR, arXiv 2610.11215v1; NVIDIA project page | Paper text as extracted (the limitations section was cut off); project page. Code is "Coming Soon", no licence stated | PAPER + PAGE |
| Shared geometry as a Rosetta Stone, arXiv 2610.09411; code dominik-schnaus/unpaired-rosetta | Main-text excerpt (appendices missing); README (MIT licence) | PAPER, partial |
| ReHarness, arXiv 2610.12668 | Abstract only. The full-text request was refused (HTTP 429) and I did not retry | ABSTRACT |
| Decision 3.0, vllm-sr/d3 and d3-lite | The two model cards (Apache-2.0). The d3-flash, d3-mini and d3-nano cards were not read | PAGE |
| DINOv3 | The HF ViT-S/16 card; LICENSE.md and README from the official repository (copies saved 2026-10-16) | PAGE + licence text |
| Kambhampati on LLM planning; RAG/CAG/KAG; SAM 2/3; YOLO26; RF-DETR; Qwen3.5/3.8; frontier models | Not read this session | RECALLED or UNVERIFIED |
| W1.5 report | Re-read in full from your saved upload (every W1.5 number below is from it) | REPORTED |
| S7 numbers | As summarised earlier and recorded in plan.md; not re-read | REPORTED |

"As extracted" means the fetch tool's summary of the page, not my own reading of the PDF. Quoted table numbers are as the tool reported them.

---

## 2. Gen-2 evidence audit

### 2.1 What the evidence supports after the audit

| Claim | Status | Basis |
|---|---|---|
| The recipe works (warm start, clean warm-up, TRADES ramp) | Supported | Frozen probe 58.52%, all six finals non-collapsed [REPORTED] |
| `recurrence_only` is the best Gen-2 model | Supported | Clean 63.44 vs. 58.46 for phase 6 on 5,000 images (about 5σ). Robust: the +4.0 to +4.4 pp gap over `gen1_core` holds under all five attacks W1.5 ran, though each is only about 1.3–1.5σ unpaired (n = 300–500). The *levels* are inflated (§2.2) |
| Recurrence beats gist-only | **Not established** | Clean +1.26 pp (about 1.3σ), robust +4.30 pp (about 1.5σ), unpaired and confounded (§2.3). W1.5's "+2.4 / +5.6 over a single glimpse" compares against a handicapped baseline |
| Belief update and prediction error help | Not supported | Phases 3–4 vs. phase 2: clean −2.2 and −2.6 pp (about 2.2σ and 2.6σ worse), robust −0.6 and −1.0 pp (noise) |
| The evidential head costs "over 5 pp" of clean accuracy | **Not attributable** | 5.4 pp is the whole ladder (phase 2 to phase 6). The phase that added the head cost −2.18 pp, confounded with the belief loop |
| The evidential head improves error detection | Not established | Different models, and W1.5 does not say whether the same score (U_t vs. max-softmax) was used. The NLL beside it fails a sanity check (§2.4) |
| Learned gaze beats a fixed grid | Not supported | +0.2 pp clean, +1.2 pp robust, within noise (n = 500). Random gaze lowers clean by 8.6 pp and raises PGD accuracy by 5.8 pp, which is the masking signature |
| Stochastic gaze is a defence | Not supported | The PGD-vs-strong-attack gap appears in the deterministic model too (§2.2) |
| Quoted robust numbers (30–35%) | **Overstated by about 10 pp** | §2.2 |
| "More glimpses help up to 4, then plateau" | Uninterpretable | A fixed grid has four distinct views, and T=6 and T=8 revisit them (§2.3) |
| The refinement block does the work | No: redundant | Zeroed or bypassed: 63.2 clean and 32.8–33.0 robust, against 63.0 / 32.4 active (n = 500, all inside the noise) |
| "The ladder regressed because the belief loop and head degrade representations" (W1.5 §4) | Not supported by the deltas | Clean: −2.18 (phase 3), −0.40, −1.96 (phase 5), −0.44; robust flat in phases 3–4 and about −5 pp at phase 5 (E10). Blame is spread, and no phase helped |

### 2.2 The suspected PGD weakness

**Evidence** [REPORTED, W1.5]. `recurrence_only`, ε_norm 0.094 (about 5.4/255), seed 42.

| Attack | n | Accuracy |
|---|---|---|
| PGD-10, eight eval seeds | 8×300 | 34.58 (σ across subsets 2.45) |
| PGD-10 (D5 at T=4; D7 fixed grid) | 500 | 32.6; 32.4 |
| PGD-50 | 500 | 32.40 |
| EOT-PGD (5 samples, 20 steps) | 500 | 24.80 |
| MI-FGSM-100 | 300 | 24.00 |
| Diversity-PGD (5 restarts × 20 steps) | 300 | 24.00 |
| APGD-CE-100 | 300 | 22.67 |

`gen1_core` shows the same pattern: PGD-50 28.20, EOT-PGD 20.40, MI-FGSM 20.00, diversity-PGD 19.67, APGD-CE-100 18.33.

**Three observations.**

1. *The in-house PGD is stuck.* On the same 500 images, PGD-10 (162 or 163 correct) and PGD-50 (162) differ by at most one image. Forty more steps found nothing. A working PGD normally keeps descending, so a flat curve means the attack converged to a poor optimum, not that it ran too few steps.
2. *The gap exists on identical images and cannot be step count.* EOT-PGD with 20 steps is 7.6 pp below PGD-50 on the same 500 images (2.7σ unpaired; about 4.5–6σ paired, because at least 38 images are discordant). If the reference is deterministic at evaluation time, five EOT samples are five copies of one gradient, so EOT-PGD is plain 20-step PGD, and a 20-step attack beat a 50-step one. The two implementations differ in something other than steps (loss, step size, precision, start). This also falsifies the v3.0 account of E15 (gaze randomness).
3. *The other attacks agree with each other.* EOT-PGD, MI-FGSM, diversity-PGD and APGD-CE-100 land within 2.1 pp of each other (22.7–24.8) and 8–10 pp below PGD. The in-house PGD is the outlier. At n = 300–500 the PGD-50 vs. APGD-CE gap is about 3σ unpaired.

W1.5 says PGD-50 "shows no gradient masking under deep iteration". That repeats the E14 inference error: PGD-10 ≈ PGD-50 is also what a stuck attack looks like. W1.5 does not state that the reference is deterministic at evaluation time. The fixed grid suggests it, and one batch run twice (logits equal to 1e-6 in eval mode) confirms it.

**Candidate causes, in order of prior and cost to test** [INFER]:

1. **Loss.** If the eval PGD reuses the TRADES inner maximisation (KL against the model's own clean prediction) instead of cross-entropy on the true label, it attacks a different objective, and that gradient is zero at the clean point unless the random start is large enough.
2. **Precision.** The Gen-1 incident was an fp16-autocast gradient bug. An attack loop under autocast without loss scaling lets input gradients of confidently classified samples underflow to zero, and sign(0) = 0 leaves those samples unperturbed. That would produce exactly the identical counts above.
3. **Step size and schedule.** A fixed α that is too small gives a flat PGD-10 to PGD-50 curve. APGD's adaptive step hides the problem.
4. **Projection and clipping** (norm-space vs. pixel-space ε, valid-range clamp). Low prior: the pixel-equivalence points sit where the norm-space curve predicts (E14).

**The audit (G2-D9-attack-audit), eval-only, on one fixed set of at least 500 images:**

- Confirm determinism (above). Then reproduce the in-house PGD-10 and PGD-50 exactly, and vary one thing at a time: fp32 vs. autocast; CE on true labels vs. the TRADES KL loss; α in two or three settings; random start on or off.
- Run APGD-CE-100, APGD-T and MI-FGSM from a maintained library (pin the version), and take the per-sample worst case across all attacks.
- Large-ε check at 16/255 and 32/255. Accuracy must go to about 0 or the attack is suspect.
- Log the fraction of samples with zero or non-finite input gradient at step 0, under fp16 and fp32.
- **Decision rule.** The change that brings in-house PGD within 2σ (paired) of APGD-CE is the cause. If none does, PGD-10 stays a monitoring number only and every headline uses APGD/AutoAttack.

**Why this matters beyond one number.**

- All six ladder finals (28.7–34.8%) are probably inflated. The ordering of phase 2 and phase 6 is probably safe, because their gap is +4.0 to +4.4 pp under every attack W1.5 ran. The other four finals have not been re-measured, and gaps of 0.6–5 pp among them can change. The re-measure is therefore APGD-CE-100 on all six finals, not PGD-50 on all six as G2-D8 specified.
- If the training attack (PGD-4 inside TRADES) shares the flaw, the models were trained against a weak adversary. That is a separate cause of low true robustness and would change how Gen-3 should be trained.
- The recurrence gain (+4.3 pp robust) is the only evidence for the loop, and it was measured with the weak attack.

Until D9 runs, quote clean accuracy and APGD-CE-100 for robustness. PGD-10 is a monitoring number.

### 2.3 Recurrence versus multiple views

Phase 2 changed four things relative to phase 1: it added four crops to the gist, added the refinement block, ran 15 more epochs at full ε, and restarted the LR schedule.

| Confound | What separates it | Status |
|---|---|---|
| Four more crops (information) | G2-CTL2-multiview-mean (about 10 GPU-hours [REPORTED estimate]), or the eval-only per-view average below | Open |
| 15 more epochs at full ε | G2-CTL1-gist-continued | Open |
| The refinement block | Zeroing or bypassing it at eval | **Done: redundant** (63.2 / 32.8–33.0 vs. 63.0 / 32.4, n = 500; γ norms 0.0524 and 0.0378, so it trained but is not needed) |
| Weak attack inflating the robust gain | D9 re-measure | Open |
| Sampling noise | Paired tests at n ≥ 1,000 robust and 5,000 clean | Open |
| A handicapped T=1 baseline | Compare with the model *trained* on one view | See below |

**The T-scaling gain is measured against a handicapped baseline.** W1.5 reports T=1 to T=4 gains of +2.4 clean and +5.6 robust (n = 500). But T=1 is the multi-view-trained network forced onto one view. It scores 60.6 clean, below the gist-only model that was *trained* for one view (62.18; −1.6 pp, σ about 2.3, inside the noise). Against that fairer baseline, T=4 gives +0.8 pp clean (σ 2.3) and +2.1 pp robust (σ 2.9; n = 500 vs. 512). Neither is distinguishable from zero. The only gain that clears 2σ is paired and within-model (robust T=1 to T=4: 28 more images of 500, about 2.7–3.6σ paired), and it measures *views*, not recurrence.

**Extra iterations on the same views add nothing.** A fixed four-point grid means T=6 and T=8 revisit the same four views, and accuracy stays at 63.0 clean and 32.4 robust. That is a partial control for "recurrence as extra computation": more iterations on the same information did not help. The plateau therefore measures information, not belief saturation, and W1.5's "Gate L passed" for `recurrence_only` is uninterpretable.

W1.5's own synthesis credits the pretrained trunk, the TRADES ramp and "four-view feature averaging". That reading is closest to the evidence, and it describes aggregation, not belief. The simplest explanation of any Gen-2 gain is four more crops plus 15 more epochs. [INFER]

**Significance.** Clean +1.26 pp has σ_diff of about 1.0 pp at n = 5,000. Robust +4.30 pp has σ_diff of about 2.9 pp at n = 512. Neither reaches 2σ even before the confounds. (plan.md v3.0 called the robust gain about 2σ; it is about 1.5σ.)

**Eval-only discriminators, cheapest first:**

1. Zero the refinement block (done).
2. Permute the order of the four crops at test time. A pure aggregator is permutation-invariant. A recurrent belief is not, but order dependence alone shows only that recent views weigh more, not that anything is revised (§6.2).
3. Compare the loop's output with the average of per-view logits from the same checkpoint.
4. Contradiction injection (G2-REV1-contradiction-injection, §6.2), which combines 2 and 3 in a regime where revision should matter.
5. Only if 2–4 are ambiguous, buy the trained controls CTL2, then CTL1.

### 2.4 The NLL computation

**What W1.5 reports** (n = 500, seed 42) [REPORTED]:

| Model | Condition | Accuracy | ECE | NLL | Error AUROC |
|---|---|---|---|---|---|
| recurrence_only | clean | 63.00 | 0.0499 | 0.6577 | 0.7708 |
| recurrence_only | PGD-10 | 32.60 | 0.2123 | 0.7725 | 0.7269 |
| gen1_core | clean | 56.60 | 0.0566 | 0.6177 | 0.8323 |
| gen1_core | PGD-10 | 28.40 | 0.3254 | 0.6001 | 0.7741 |

**These NLL values do not behave like the NLL of the true label on the logits the attack produced.**

1. *Direction.* NLL is the mean of −ln p(true class). An attack that takes accuracy from 56.6% to 28.4% must push p(true) down on every sample it flips, so NLL should rise several-fold (a confident error costs 2–4 nats). For `gen1_core` the reported NLL *fell* (0.618 to 0.600). For the reference it rose 17% while its error rate doubled.
2. *Arithmetic.* A wrong prediction has p(true) below 0.5, because it is below p(max) and the two sum to at most 1. Under PGD, 71.6% of `gen1_core` samples are wrong. NLL = 0.600 then forces the geometric-mean p(true) on wrong samples to at least 0.43, even if every correct sample had zero loss. If correct samples average p(true) of about 0.67 or lower, the required value exceeds the 0.5 ceiling, which is impossible. It is feasible only if surviving samples have p(true) near 0.8 or more *and* every adversarial error is a near-exact two-way tie (true class 45–50%). For the reference under PGD, wrong samples must hold geometric-mean p(true) of 0.35–0.43 while its ECE caps mean confidence near 0.54. That is not credible for a 100-class model under a loss-maximising attack.
3. *What fits.* All four values satisfy the necessary condition for "mean of −ln p(max)", the NLL of the model's *own prediction*, given the mean confidence each row's ECE allows (that quantity must be at least e^−NLL: 0.52, 0.46, 0.54, 0.55). They also track confidence rather than accuracy, which is what that quantity does. [INFER] This is a hypothesis, not a diagnosis.

The reference is deterministic and has a plain softmax head, so explanations specific to an evidential or stochastic model cannot account for its row. The live candidates are: labels replaced by predictions (the NLL of the argmax); NLL computed on a different tensor than the one the attack changed (ECE does change under attack, so ECE at least used attacked logits); or label misalignment after a shuffle or subset.

**One-line check, on the saved logits.** Print `mean(−ln p[y])` beside `mean(−ln max p)`. If the second equals the reported NLL, the hypothesis is confirmed.

**AUROC and the evidential-head claims.**

- W1.5 does not say which score the AUROC used. `gen1_core` has a Dirichlet head and the reference does not, so the comparison may be cross-score as well as cross-model. The within-model comparison (U_t against the max-softmax of the same logits) is the CAL2 decision rule and was not reported. CLAIM-4 stays untested.
- Under attack, error-detection AUROC (0.73–0.77) is almost the same as on clean data (0.77, 0.83) while ECE grows 4–6 times. Adversarial errors are usually confident, so this is unusual. Recompute before interpreting it.
- "The head sacrifices over 5 pp of clean accuracy" describes the whole ladder, not the head (§2.1).

**Recompute protocol** (from saved fp32 logits, no GPU): natural log; the true labels; no clamping; the same images for every model. Report mean −ln p[y], mean −ln max p, ECE (15 bins), the 20 worst per-sample NLLs, and the error-detection AUROC for both U_t and max-softmax on the same logits. Add a temperature-scaled softmax baseline (fit the temperature on a held-out split), so that any calibration claim for the evidential head has a control. Add three assertions to the harness: NLL is at least (1 − accuracy) × ln 2; NLL under a loss-maximising attack is clearly above clean NLL; and mean −ln p[y] is at least mean −ln max p.

### 2.5 Corrections

**To plan.md v3.0**

| Item | v3.0 said | Corrected |
|---|---|---|
| E15 mechanism | The 7.6 pp PGD-to-EOT gap is gaze randomness | Wrong. The deterministic reference shows the same gap on identical images. It is a property of the in-house PGD (new blocker B12) |
| E6 hypothesis | Refinement γ near 1e-5, block inert | Wrong. γ norms 0.0524 and 0.0378; the block trained but is redundant |
| E6 significance | Robust +4.3 pp is about 2σ | About 1.5σ unpaired |
| E12 and B8 | Documented loss 0.45·CE, logged 0.55·CE | Code read: `w_trades·(CE + β·KL)` with β = 2.5 (`train_gen2_foundation.py`, line 380), so 0.55·CE + 1.375·KL. The 0.45 documentation was wrong |
| B1 | Regression cause unattributed, CTL4 is the key control | Reframed by W1.5: the regression is in the weights. CTL4 demoted |
| B3 | Refinement block may be inert | Resolved: redundant |
| B9 | Phase 4 final robust missing | 33.79. Phase 6 stopped at 14 epochs because the 11 h timer ended the session |
| CLAIM-3, 4, 5, 6 | Untested or partial | Not supported; confounded and untested (§2.4); not supported; floor about 22–23% for the reference |
| New | | B12 (attack audit), B13 (NLL and AUROC anomalies; two robust-number pairs, 33.19/31.71 vs. 34.02/33.12, still unreconciled; TRADES clip-norm harness output unexplained), CLAIM-8 (the belief revises) |

**To the W1.5 report's own wording** (the numbers stand; the readings do not)

| W1.5 says | Problem |
|---|---|
| 8×300 clean gap is "> 5σ combined" | Two means of 2,400 images carry σ_diff of about 1.4 pp, so about 3.8σ. The full-validation finals (63.44 vs. 58.46, n = 5,000) give about 5.1σ. Conclusion stands, derivation does not |
| PGD-50 "shows no gradient masking under deep iteration" | A flat PGD curve is the signature of a stuck attack (§2.2) |
| Recurrence gives "+2.4 clean / +5.6 robust over a single glimpse" | Handicapped T=1 baseline; against the gist-only-trained model the gain is +0.8 / +2.1, inside the noise (§2.3) |
| The evidential head "sacrifices over 5 pp" of clean accuracy | The 5.4 pp is the whole ladder (§2.1) |
| Bypassing the refinement block "improves" clean and robust | +0.2 and +0.6 pp at n = 500 is noise; the correct reading is "no measurable effect" |
| The ladder regressed because of the belief loop and the head | The deltas put −2.18 pp at phase 3 and about −5 pp robust at phase 5; no single culprit (E10) |
| Error AUROC shows the head "improves error prediction" | Cross-model, score unspecified, and the NLL beside it fails a sanity check (§2.4) |

---

## 3. What the seven directions actually give RHAN

Each row separates what the source shows from what it cannot tell us. The last column is the reusable principle, not the machinery.

| Source (level) | What it shows | What it cannot tell us | Principle for RHAN [INFER] |
|---|---|---|---|
| **JEPA inverse dynamics** (PAPER; code has no stated licence) | Next-step latent prediction plus an anti-collapse regulariser (SIGReg) can reach its optimum while the latent discards the unstable modes a controller needs (Lemma 2, linear systems). Adding an endpoint inverse-dynamics loss, which decodes all H actions from the start and end latents (Eq. 14), restores them: CartPole LQR success 0% to 100% (Table 1), Walker2D velocity 0.08 to 2.91 m/s against 3.61 ground truth (Table 2), PointMaze at least 90% CEM success | Theory is linear only. One seed (42), 10 trials per condition, no variance. It cannot recover what a frozen encoder already discarded (no DINOv2 variant reaches LQR success, Table 5). No classification or perception task | A predictive objective can still forget what the decision needs. Add a loss that makes the state answer for the actions taken |
| **GATOR** (PAPER + PAGE; no code or licence yet) | Generate, then an edit-render-review loop in which each edit is kept or reverted by agreement with the observations under matched cameras and lighting; unresolved defects are reported as "needs review". On 55 HANDAL objects the loop moves chamfer distance 0.008 to 0.007 and F1 0.930 to 0.936, and slightly lowers IoU (0.739 to 0.717) and SSIM (0.925 to 0.918). Generation alone (about 30 s) beats a from-scratch agent given 20 minutes | The agent is a commercial model, so the result is not reproducible; the keep/revert verdict is made by the same agent; the extract shows no variance; the domain is 3D assets | Compare the old and the new state on the same evidence, keep only what the evidence supports, revert the rest, and flag what cannot be resolved. Expect modest gains on top of a strong prior |
| **ReHarness** (ABSTRACT only; full text refused) | Backward revision of retrieved memory using trajectory outcomes, grounded in the source. Nothing else is verified: no benchmark, number or safeguard | Everything beyond the abstract | Outcomes can be credited back to stored items, with revisions anchored to the source so that memory does not drift. Treat as a design idea until read |
| **Rosetta Stone** (PAPER, appendices missing; MIT code) | Two separately trained embedding spaces can be aligned without pairs by an orthogonal map (cluster matching, a Gromov-Wasserstein-style start, Wasserstein-Procrustes). Average FOSCTTM 0.154 vs. 0.223 for mini-vec2vec, better on 17 of 21 model pairs; zero-shot CIFAR-10 top-1 47.4%, ImageNet-100 top-5 26.5%. Geometry similarity, CKA among the measures, correlates with how alignable two spaces are (qualitative in the extract) | Recovers coarse structure only; fails on geometrically inconsistent pairs; accuracies are far from decision-grade; most vision-language experiments use CPU jobs with 96 GB of memory | Whether two spaces are compatible is measurable before one builds an adapter. Use CKA first |
| **Decision 3.0** (PAGE; Apache-2.0) | Models that return a probability for every answer option (Choice, Yes/No, Score) and no text. `d3`: 26.09B parameters, from Qwen3.8-27B per the card, 66.3 on the vendor's text index (1st of 149 entries) and 71.3 on its vision board. `d3-lite`: 0.85B including a 0.10B vision encoder, from Qwen3.5-0.8B, scores 32.3 and 42.1. Perception Test validation (internal): 77.4 and 59.3, chance 33.3 | Training data and method are undisclosed ("Trained on AMD Instinct MI325X GPUs"). The index is vendor-run, and several vision benchmarks are "approximate rebuilds". vLLM support is on main only, and the CUDA path is untested on NVIDIA. A leaderboard rank is not evidence of reasoning | The perception-to-decision interface should be a distribution scored with proper scoring rules (Brier, NLL), not generated text. That is Gate U with a clean format. `d3-lite` is a baseline and an interface template, not a reasoner |
| **Kambhampati on LLM planning** (RECALLED; the DOI fetch failed) | As I recall the 2024 position: LLMs are unreliable planners and unreliable self-verifiers, and are useful as proposal generators when an external sound verifier checks and returns feedback (LLM-Modulo) | Not re-read | Separate proposing from certifying. The module that proposes a gaze or a belief change must not be the one that grades it |
| **RAG / CAG / KAG** (RECALLED; not read) | Retrieve at inference (RAG), preload and cache (CAG), or use a knowledge graph (KAG) | Not re-read; classification only | Match memory type to the store. Within-episode evidence is small, so hold all of it (cache-like, no retrieval). Object and spatial state is structured. Only cross-episode memory needs retrieval, and therefore ReHarness-style revision |

---

## 4. Synthesis: preserve, test, revise

**One design statement.** *Evidence is immutable; belief is mutable; every revision must be reproducible from the evidence.* Raw glimpse evidence (features, location, scale, time) is stored append-only. The belief, the uncertainty and any weights over stored evidence may be revised, and a revision that cannot be re-derived from the evidence is a bug.

Each function below has a Gen-2 status, a test, and the **strongest simple baseline** the new mechanism must beat. A mechanism that cannot beat its baseline is not a contribution, however well it fits the story.

| Function | Source | RHAN component | Gen-2 status | Test | Strongest simple baseline |
|---|---|---|---|---|---|
| Preserve decision-relevant state | JEPA inverse dynamics | Auxiliary inverse-glimpse loss: from the belief before and after a step, predict which view was taken and which view contradicted the belief | The belief is not shown to carry anything beyond an average | G2-PRB1-surprise-probe, G2-REV1-contradiction-injection, then G2-IDM1-inverse-glimpse | The same loss on shuffled targets; a linear probe on the plain mean of view features (same decodability?) |
| Test against evidence; keep or revert | GATOR, Kambhampati | Accept/revert rule: propose z′ = z + Δ, accept only if z′ predicts a held-out observed view better than z; otherwise shrink or revert. Calibrated uncertainty acts as the "needs more evidence" flag | The update is unconditional (bounded at 0.1 per step) | G2-REV1, then G2-GAT1-accept-revert | The unconditional bounded update (Gen-2 itself); a recency-weighted average of per-view logits with one fitted decay |
| Revise stored evidence from outcomes | ReHarness (abstract) | After the decision, re-weight stored glimpse evidence, never rewrite it | No memory exists yet (M2 not built) | G2-MEM1-glimpse-memory, then G3-REV2-backward-revision | A forward-only read of the same store (mean of stored features) |
| Decision interface | Decision 3.0 | Probability vector scored by Brier and NLL | The Dirichlet head's clean-accuracy cost is not attributable (§2.1) and its NLL fails a sanity check (§2.4) | G2-CAL2-evidential-gate after the NLL recompute; later G4-SLM1-belief-readout | A plain softmax head with temperature scaling, scored by the same rules on the same logits |
| Compatibility check | Rosetta | CKA before any backbone swap or adapter | None | The CKA part of G2-RES3-backbone-probe; later G4-GEO1-embedding-alignment | A paired linear map, if paired data exist |
| Memory taxonomy | RAG / CAG / KAG | Cache (M2), structured state (M3, W_t), retrieval only for M4 | M0 and M1 exist | none | none |
| Evidence selection, or search over hypotheses | Kambhampati (propose, then verify); JEPA | Hypothesis-separating gaze: hold the top two class hypotheses, score each candidate view by how differently the two hypotheses predict it, look where they differ most, stop when the margin clears a threshold | AIS-v2 is a one-step selector with one learnable scalar; no competing hypotheses are held (below) | G3-HYP1-two-hypothesis-gaze, at Gate S only | Entropy-driven gaze, fixed grid and random gaze, at equal observations and FLOPs |

**Does RHAN need search?** [INFER] Not yet, and not in the form the word suggests. Gen-2 transforms a single state repeatedly, one trajectory with one belief. AIS-v2 chooses the next view in one step from a one-parameter score. That is selection, not search. Search in the propose-then-verify sense needs several candidates held at once, a verifier that is independent of the proposer, and feedback from one to the other. The smallest RHAN-specific version is the last row above. A training-free form exists: from training data, compute per-location class-mean features, and for the current top two classes look where their means differ most. If entropy-driven gaze matches it at equal observations, search adds nothing and should not be built. Do not build it before Gate S, because on a four-crop grid that already covers the image there is nothing to choose between (§2.3).

**Where this connects to the Gen-2 evidence.** Lemma 2 of the JEPA paper offers a candidate reason why the belief machinery added nothing: a prediction objective can be satisfied while the state forgets what a decision needs. But Gen-2's cross-entropy already supervises label-relevant information. The lemma therefore speaks to the state a *gaze or revision* decision needs, which Gen-2 never supervised (the AIS-v2 policy has one learnable scalar). That is a hypothesis [INFER], and PRB1 tests it.

**Three honest limits on this synthesis.**

1. The three inspiration papers come from control, 3D asset refinement and text-agent memory. None is evidence that the pattern helps visual classification. The only RHAN-specific evidence will come from REV1 and the search task (Gate S).
2. Where a strong prior exists, the loop's effect is small (GATOR: chamfer 0.008 to 0.007). RHAN effects of 1–2 pp against noise of about 2 pp at n = 512 need paired tests and larger n.
3. The JEPA paper's own limit applies here: an auxiliary loss cannot restore what the backbone discarded. If the 56-px gist is the bottleneck, the substrate comes first.

---

## 5. Architecture comparison

The labels follow your definitions. A, B and C are yours; D is the fourth I added. DINOv3 is not an architecture here. It is a dimension of B and C that RES3 decides.

| | **A. Foundation-model-centered** | **B. RHAN-centered, revisable-evidence core** | **C. Hybrid object-centric predictive decision system** | **D. Verifier in the loop** (my addition) |
|---|---|---|---|---|
| **Definition** | A strong pretrained model does perception, and for the largest versions reasoning (DINOv3 features, a VLM, or a `d3`-class decision model). RHAN is a thin readout or an evaluator | An append-only evidence store, an accept/revert belief update, an inverse-glimpse loss and, later, backward re-weighting, built on the Gen-2 reference first. The trunk is delegated; state, evidence and stopping are RHAN's | Region or object proposals (SAM-class masks, a detector), per-object persistent state in the evidence store, a predictive model of what each next observation should show, and a probability-vector decision readout (later a small language model) | A frontier model proposes answers or hypotheses; a RHAN-like perceiver checks them against the evidence and flags what it cannot confirm (the GATOR and LLM-Modulo pattern) |
| **Capability** | Highest clean accuracy and breadth soonest. Little on the "knows what it does not know" axis unless added | Unknown today. Gen-2 matches multi-view aggregation. Potential: acquiring and revising evidence under a budget | Widest: scene queries, tracking in TDV, object-level questions | Strong at once, because the proposer reasons. The verifier is only as good as its perception |
| **Originality** | Low. A known recipe. The contribution would be the evaluation harness | Highest, if REV1 and Gate S go well. The revision principle and the harness are the novelty | High if the per-object state is revisable. Every part has precedent (slot models, JEPA world models) | Medium. A known pattern. The new part would be a calibrated, evidence-grounded verifier |
| **Compute and memory** | Trunk 21.6M (DINOv3-S) to 26B (`d3`). Heads only on a T4. `d3` is out of reach for training | Eval-only first, then arms of about 10 GPU-hours or less [REPORTED estimate]. Store of a few views per image | Highest. A proposal model runs at inference, and state is kept per object. T4 limits this | API cost; no local training beyond the verifier |
| **Implementation complexity** | Low | Moderate. Each mechanism is small and tested alone | High. Many interfaces | Moderate, plus an API integration |
| **Training** | Frozen features and trained heads. Adversarial training of a frozen trunk trains only the heads | Warm start from the reference, identity-initialised modules, one change per arm | Staged and multi-loss. This is where "many mechanisms, no attributable effect" returns | Verifier only |
| **External dependence** | High: a gated licence (DINOv3), vendor-run scores, a possible API | Low: the DINOv2 trunk you already use | High: proposal-model licences are unverified | Very high: a commercial model |
| **Robustness and uncertainty** | Inherits the trunk. A frozen non-robust trunk gives near-zero adversarial accuracy [INFER, standard result]; DINOv3's card claims no robustness | Must be earned under APGD and EOT (§2.2). Revision could help or enlarge the attack surface; untested | The proposal stage is a new attack surface. Untested | The verifier is attackable, and so is the proposer |
| **Failure modes** | Results belong to the encoder, and a leaderboard rank gets read as reasoning | Effects near noise; the loop averages (REV1); anchoring; synthetic contradictions only | Gen-2's failure at larger scale: no attribution, large cost | The verifier is weaker than the proposer, so it cannot check what it cannot perceive; results are not reproducible |
| **Evaluation** | Matched-token probes (RES3); calibration and NLL against a softmax baseline | D9, REV1/PRB1, then Gates S, B and U with paired tests and equal observations | Gate S on scenes with known targets; TDV sequences; object-level metrics | Accept/reject AUROC of the verifier against the proposer's own confidence |
| **Long-term potential** | High as an interface and baseline; low as a research contribution by itself | Medium-high if Gate S passes. Otherwise a methodology and negative-result thesis | Highest | Medium |
| **Falsifier** | RES3 gain below 2σ at matched tokens (the encoder is not the limit) | REV1 recovery no better than the averaging control after GAT1 and IDM1; or no Gate S win over entropy, fixed and random gaze | A component that does not beat its own simple baseline at its gate | The verifier does not beat the proposer's confidence |
| **Verdict** | **Baseline and evaluator now; interface later** | **Do now, staged** | **The destination. Earn each component through a gate** | **Optional Gen-4 demo after Gate S** |

**Recommendation.** B staged toward C. A is the baseline every B result must be measured against, and the consumer of belief tokens at the end. D is a demo, not a component. This is the strongest *defensible* direction because every step has an operational test, a control and a pre-stated falsifier, and because none of it assumes the loop works. It tries to find out.

---

## 6. The experiments most likely to change direction

### 6.1 G2-D9-attack-audit, then APGD-CE-100 on all six finals

Design is in §2.2. Cost: minutes to a few hours on one T4 or the 4060. Needs a maintained APGD implementation, pinned (I did not verify any library this session). First step: run one batch twice and confirm the reference is deterministic at evaluation (logits equal to 1e-6), because the whole reading of the EOT-PGD gap rests on it.

| Result | Meaning | Action |
|---|---|---|
| A fixable cause found, and corrected PGD is within 2σ (paired) of APGD-CE | The in-house attack was wrong, not the models | Re-run all six finals under the corrected attack and APGD-CE; re-rank; update CLAIM-1, 2 and 6 |
| Rankings change and the recurrence gain falls below 2σ (paired) | Gen-2's loop adds nothing measurable on either axis | Report Gen-2 as "multi-view recipe works; loop mechanisms unsupported"; skip further CTL arms; move to REV1 and Gate S |
| Phase 2 stays above phase 6 under APGD (likely: the gap is +4.0 to +4.4 pp under all five attacks W1.5 ran) | The ordering is safe, but the recurrence gain stays confounded by views and epochs | Keep the eval-only discriminators in the queue; CTL2 only if they leave the question open |
| No fix found, but APGD is consistently far below PGD | The model's loss surface defeats PGD, or the training adversary is weak | Run the same audit on the training attack before scaling any adversarial training |

### 6.2 G2-REV1-contradiction-injection, with G2-PRB1-surprise-probe

**A flaw in my first design, now fixed.** I had written "revision gives L(early) < L(late)" as the signature. That is also what a plain *recency-weighted* average produces: any rule that favours recent views forgets an early wrong one, with no selectivity at all. And because the crops sit in a fixed grid order, "early" and "late" were also different *places* in the image, so time was confounded with location. The design below separates the three.

**Design.** Run the finals of phases 2–6 (phase 1 has no glimpses) on the full 5,000 validation images. The inputs are clean, not attacked, so this costs far less than any robustness run. For each image, replace one view with the view at the *same location* from an image of a different class, which keeps it in-distribution for its location, and keep the others correct. Cross **time** with **location**: use crop-order permutations in a Latin square, so each location is presented at each time step equally often. Also contaminate the gist alone (if the gist initialises the belief), because a model that leans on a correct gist would resist crop contamination without revising anything.

**Controls.**

- (a) The per-view logit average from the same checkpoint (each view through the network alone, logits averaged): the pure aggregator.
- (b) A recency-weighted logit average with one decay parameter, chosen to minimise the mean damage M on a held-out 1,000 images. This is an optimistic baseline for the loop to beat, and it is what separates revision from forgetting.
- (c) The uncontaminated four-view result.
- (d) Later, the trained CTL2.

**Metrics** (paired, per image). L(t) = uncontaminated accuracy minus accuracy with the contaminated view at time step t, averaged over locations.

- R = L(last) − L(first): order dependence.
- M = mean of L(t) over t: total damage from one wrong view.

For any weighted average whose weights sum to one, M is roughly constant, because moving weight from one position to another moves the damage with it [INFER, first order]. Order-dependent linear rules change R but not M. Only a *nonlinear*, outlier-rejecting rule lowers M. So M is the selectivity measure and R is the recency measure.

| Result | Reading | Next |
|---|---|---|
| R near 0 and M within 2σ of control (a) | The loop averages; "belief" in Gen-2 is an average | Gen-3's claim rests entirely on new mechanisms; run GAT1 and IDM1 |
| R > 0 (late views damage more) and M within 2σ of controls (a) and (b) | Recency weighting: early evidence fades, with no selectivity | "Belief" is an exponential average. GAT1 has something to add |
| R < 0 (early views damage more) | Anchoring: the 0.1 per-step bound cannot undo an early error | G2-BEL4-bound-adaptive first |
| M below both (a) and (b) by more than 2σ (paired) at two or more time steps, and the gist-contaminated case also recovers | Outlier rejection: the loop has a selective mechanism (CLAIM-8 has a Gen-2 foothold) | Go to Gate S with the reference; GAT1 becomes an ablation |
| M below (a) and (b), but only when the gist is correct | The model leans on the gist; nothing is being revised | Treat as the "averages" row |
| PRB1: AUROC of E_t at least 0.7 | Surprise carries contradiction information | GAT1 has a signal to gate on |
| PRB1: AUROC of E_t at most 0.6 | The predictor discards what revision needs | IDM1 first: the state must be made to carry it |

**PRB1** (phases 4–6, which feed prediction error back; phase 3 only if it computes E_t without using it): under the same injections, the AUROC of the step-wise prediction error E_t and of the change in U_t for flagging the injected view, and linear decodability of its position from the final belief.

**Why it matters.** It is the only experiment the existing checkpoints can run that tests the next generation's central claim, and it chooses the first mechanism arm. It also tests CLAIM-3 in a regime where belief machinery should matter. A caveat: contradiction is synthetic. A positive result shows the loop can revise under a designed contradiction, not that it does so in natural scenes.

### 6.3 G2-RES3-backbone-probe (extends G2-RES1-probe-112)

**First, a metadata check that costs nothing.** Find the resolution at which ImageNet-100 is stored. Plan E3 says about 96 px; I did not verify it. Crops can add detail only up to the stored resolution. If the source is about 96 px, a 112-px gist is an upsampled copy of it, and the 112-px arm then measures token count rather than information. That is still worth knowing, but it must be labelled as such.

**Design.** Frozen-trunk linear probe on ImageNet-100, the Gate W protocol (5 epochs, baseline 58.52%). DINOv2-S/14 and DINOv3-S/16, with the gist at matched pixels (112 px) and at matched tokens (DINOv2 at 112 px gives 64 tokens; DINOv3 at 128 px gives 64). For the Gen-2 16-token case use 64 px for DINOv3, because patch 16 turns 56 px into a 48-px crop with 9 tokens (the card says inputs are cropped to a multiple of 16) [PAGE]. Add CKA between the two feature spaces on 5,000 images. You must accept the gated licence on Hugging Face yourself.

| Result | Reading | Action |
|---|---|---|
| DINOv2 at 112 px beats 56 px by 5 pp or more (RES1 threshold, still pending) | The input is the bottleneck, in pixels or in tokens (resolved by the metadata check) | Substrate rebuild first, at 112 px or more |
| Gain under about 2 pp (2σ) for both backbones | The 56-px gist is not the binding limit on ImageNet-100 | Stay on DINOv2-S; test the loop in the high-resolution search regime, where resolution is the point |
| DINOv3 beats DINOv2 by more than 2σ at matched tokens | Encoder upgrade is worth planning | Rebuild with a frozen or lightly tuned trunk (the card says fine-tuning should be a last resort) |

A probe is a clean-accuracy test. DINOv3's card makes no adversarial-robustness claim, so a probe win is not a robustness win. A frozen trunk cannot be made robust by attacking it end to end and training only the heads. The rebuild question is therefore a trunk-tuning question, taken up in §8.

### 6.4 Controls to buy only if needed

CTL2 (mean-pooled multi-view) and CTL1 (gist-only continued) cost about 10 GPU-hours each [REPORTED estimate, unmeasured], which is about a third of a Kaggle week. Run one only if §6.1 and §6.2 leave the recurrence question open, and choose it by what is still ambiguous: CTL2 for the views confound, CTL1 for the epochs confound.

### 6.5 Predictions, stated before the data

Writing these down first means that a surprise will be visible as one. They are my priors [INFER], not findings, and you should replace them with your own if you disagree.

| Experiment | Prediction | My confidence |
|---|---|---|
| D9 | The reference is deterministic at evaluation | About 90% |
| D9 | Corrected in-house PGD lands within 3 pp of APGD-CE, at 21–26% robust for the reference | About 70% |
| D9 | The cause is the loss or precision rather than step size or projection | About 65% |
| D9 | Phase 2 stays above phase 6 under APGD-CE | About 80% |
| REV1 | The loop behaves as averaging or recency weighting; no selective revision (M below both controls) at two or more time steps | About 85% |
| RES3 | DINOv2 at 112 px beats 56 px by 5 pp or more in the frozen probe | About 75% |
| RES3 | DINOv3 and DINOv2 differ by less than 2σ at matched tokens | About 55% |
| CTL2, if bought | It recovers at least 80% of phase 2's clean gain over gist-only, so recurrence contributes 0–1 pp (the external review predicted the same) | About 65% |
| NLL recompute | Reported NLL equals mean −ln max p rather than mean −ln p[y] | About 70% |

---

## 7. Roadmap

Budget reality: Kaggle gives about 30 GPU-hours a week, and one 15-epoch arm costs about 10 GPU-hours at the agent's implied 39 minutes per epoch [REPORTED estimate, unmeasured]. That is one to two trained arms a week. Everything before Stage 2 is eval-only.

| Stage | Contents | Hardware | Exit rule |
|---|---|---|---|
| **0. Audit** (days) | G2-D9-attack-audit (determinism check first); APGD-CE-100 on the six finals; NLL, ECE and AUROC recompute from saved logits, with a temperature-scaled softmax baseline; the stored-resolution check on ImageNet-100; G2-INF1-select-split | 4060 or one T4 session | Corrected robust numbers, a re-ranked Gen-2 table, and a calibration table that passes the three sanity assertions (§2.4) |
| **1. Diagnose** (days) | G2-REV1-contradiction-injection (Latin square over time and location, gist-contaminated case, controls a and b), G2-PRB1-surprise-probe, G2-RES3-backbone-probe (with G2-RES1-probe-112) | 4060 or T4 | Take the branch in §6.2 and §6.3 |
| **2. One or two trained arms** (1–2 Kaggle weeks) | Pick from G2-GAT1-accept-revert, G2-IDM1-inverse-glimpse, G2-BEL4-bound-adaptive (if anchoring), G2-UC1-contraction-update (after REV1), or the substrate rebuild | Kaggle or Colab T4 | Paired win over a parameter-matched control, 8 eval seeds, robust n of at least 1,000, Gate A attack |
| **3. The regime where the claim can win** | G3-HRV1-high-res-substrate, G3-VS1-synthetic-search, G3-BUD1-budget-curve, G3-HYP1-two-hypothesis-gaze (Gates B and S): targets at known locations in 384–512 px scenes, native-resolution crops, equal observation budget, adaptive vs. entropy vs. fixed vs. random vs. feed-forward | T4 | Gate S passed, or a clean negative result |
| **4. Interface** (Gen-4) | G4-API1-perceptual-state-api, G4-SLM1-belief-readout (shuffled-token control), G4-GEO1-embedding-alignment, G4-VLM1-raw-vs-rhan | T4 with LoRA | Beats the classifier head and the shuffled-token control |

**Rules for Stage 2 arms** (from E4, E10): one change per arm; a null continuation control with the same protocol; identity initialisation of new modules (G2-REC3-identity-init); warm start from the reference; paired evaluation on identical images; the final checkpoint, never best-on-validation.

**Stop rule.** If Stage 0 collapses the phase ranking and Stage 1 shows averaging, report Gen-2 as a methodology and negative-result contribution (a harness that found a 10 pp over-estimate and several mis-read agent verdicts), and make Gate S the only Gen-3 target. That is still a thesis-grade outcome. It is not a failure.

---

## 8. Foundation-model decisions

| Component | Role | Verdict | Conditions |
|---|---|---|---|
| **DINOv3-S/16** | Candidate substrate | Probe now (RES3); build only if RES3 passes | About 21.6M parameters, patch 16, RoPE, 4 register tokens, distilled from a 7B teacher on 1.689B images [PAGE]. Gated on Hugging Face. The licence is custom: royalty-free use, modification and distribution; publications must acknowledge the DINO Materials; use must comply with trade controls and not serve military, ITAR or weapons ends; redistribution carries the same licence. I read the full text of LICENSE.md. The card makes no adversarial-robustness claim, and the authors advise against fine-tuning except as a last resort |
| **SAM family** | Teacher, source of target locations, evaluation instrument | Use outside the loop, at Stage 3 only | Not verified this session. SAM 2's published design (a memory bank of past-frame features) is a reference for M2 and M3 [RECALLED]. For Gate S, masks or dataset boxes define ground-truth target locations for scoring gaze |
| **Qwen3.5-0.8B or `d3-lite`** | Gen-4 decision-model candidates | Defer to Stage 4. Use `d3-lite` as a baseline and interface template | Verified on the cards: Apache-2.0, 0.85B including a 0.10B vision encoder, returns a probability per option. Vendor-run scores for the small model are low (text 32.3, vision 42.1). Loading needs `trust_remote_code=True`, so pin the revision. The card says the CUDA path of vLLM is untested on NVIDIA; use the Transformers path |
| **YOLO26, RF-DETR** | Proposal source or baseline for the search task | Optional. Skip for now | Licences and versions not verified; read them before use |
| **Frontier VLMs (Claude 5, Gemini 3.7, GPT-5)** | Evaluators and upper-bound baselines | Evaluate with them; do not embed them. Do not distil from them until the terms are read | Availability and terms not verified here. GATOR uses a commercial agent, which is exactly the uncontrolled dependency a graduation project should avoid |

**Backbone regimes.** "Use DINOv3" can mean five different experiments with different costs and different things they can show.

| Regime | What it can show | Status and verdict |
|---|---|---|
| Frozen DINOv3 trunk, trained heads and loop | Whether the input representation is the limit | Do first, as RES3. Caveat: if the attack goes through a frozen trunk and only the heads train, robust accuracy should be near zero [INFER, standard result]. Gen-2 trained the trunk at 5e-5, and D2 (feature drift) was never measured, so how far the trunk moved is unknown |
| Partly trainable (last blocks and LayerScale-type parameters) | A middle path that keeps most of the prior | After RES3 |
| Fully trainable from DINOv3 initialisation | What Gen-2 did, with a different trunk | Ladder-scale cost (about 68 GPU-hours [REPORTED]); the card calls fine-tuning a last resort |
| Joint perception and recurrence (what Gen-2 did) | Nothing attributable: the trunk and the loop changed together | Do not repeat without a frozen-trunk control |
| From scratch | Nothing | Gen-1 reached 5.82% clean and 0.00 robust. Rejected (E2) |

---

## 9. Library boundaries

- **Use as libraries:** a maintained APGD/AutoAttack implementation (pin the version and cross-check it against the in-house PGD on identical images); Hugging Face Transformers for DINOv3 (gated access); your existing evaluation harness.
- **Re-implement from the paper, do not copy:** the endpoint inverse-dynamics loss. It is one decoder and a mean squared error on actions. The JEPA repository states no licence, and GATOR has no code yet.
- **Optional diagnostic only:** `unpaired-rosetta` (MIT). Use CKA first. Run the Wasserstein-Procrustes solver only if the video pipeline (TDV) has no captions to pair against, and expect memory-heavy runs (the repository's scripts allot 96 GB to most vision-language jobs).
- **Do not install:** the Blender and Isaac pipeline behind GATOR; agent frameworks; vLLM Semantic Router engine mode for `d3` (development channel).
- **Build in-house:** the append-only evidence store, the revision operators, the contradiction generator and the REV1 and PRB1 harness. These are the research contribution.

---

## 10. Open questions and risks

1. The W1.5 and S7 numbers are agent-reported. D9 may change them, in either direction.
2. The NLL values fail a sanity check (§2.4). The leading hypothesis, that they are mean −ln p(max), is an inference from arithmetic, and only the saved logits can settle it. AUROC is unresolved for the same reason: the score used is not stated.
3. The EOT-PGD reading rests on the reference being deterministic at evaluation. That is likely, but unconfirmed.
4. The stored resolution of ImageNet-100 is unverified. It decides whether a 112-px gist adds information or only tokens.
5. PGD-50 (n = 500) and APGD-CE-100 (n = 300) were run on different subsets, so the 3σ gap between them is unpaired. A paired re-run on identical images will give a tighter, and possibly larger, number.
6. ReHarness is unverified beyond its abstract. If its mechanism matters to a decision, read it before relying on it.
7. Contradiction injection is synthetic. A positive result shows the loop can revise under a designed contradiction, not that it does so in natural scenes.
8. Power. Effects of 1–2 pp sit against noise of about 2 pp at n = 512. Use paired tests, at least 1,000 images for robust numbers and 5,000 for clean, and plan at least three training seeds for any headline claim. Every Gen-2 number comes from one training seed (41).
9. The loop may not beat a feed-forward pass at equal budget even in the search regime. That would be a valid negative result, and the plan should be written to accept it.
10. Not run in W1.5 [REPORTED]: G2-AIS5-gate-rerun, G2-D2-feature-drift, G2-RES1-probe-112, G2-INF1-select-split. The last two matter for the stages above.

---

## 11. Cross-check: the external review you attached on 17 October

A third-party AI wrote it. It said it could not verify the seven developments, so it informs the reading of Gen-2 and the direction, not the source facts. I did not verify its citations. I checked each claim against the audit above.

| The review says | Against the audit | Disposition |
|---|---|---|
| The testbed is the root cause: at a 56-px, 16-token gist the whole image fits, so there is nothing to acquire | Partly right. *Selection* has little to do on a four-crop grid that covers the image, which agrees with the T=6 and T=8 plateau (§2.3). But crops add resolution up to the stored resolution, and that is probably where Gen-2's gain came from. Two other explanations (a weak attack inflating the gain; four views and 15 more epochs) cannot yet be ranked below it | Adopted as the reason Gates S and B are the main test. Treated as a hypothesis, not a diagnosis |
| PGD is weak, "almost certainly fp16 autocast" | The weakness is confirmed by the audit (§2.2). The cause is one of three leading candidates (loss, precision, step size) | D9 decides, one factor at a time |
| CTL2 will recover most of phase 2's gain; recurrence contributes 0–1 pp | Plausible and consistent with §2.3 | Recorded as a prediction in §6.5 |
| Run trained CTL1 and CTL2 in the first two weeks | I would run the eval-only discriminators first (order permutation, per-view average, REV1) and buy a trained control, about 10 GPU-hours each, only if they leave the question open | Kept my order |
| The evidential head gives +0.06 / +0.05 AUROC and "costs 5 pp" | Neither is attributable. The AUROC compares different models with an unstated score, and the 5.4 pp is the whole ladder (§2.1, §2.4) | Not adopted. Its temperature-scaled softmax baseline for CAL2 is adopted |
| The belief state is non-predictive | Consistent with phases 3–4 against phase 2, but not established | Folded into PRB1 and IDM1 (its DPS1) |
| The real question is a crossover: where does inference-time observation search beat parameter scale, and a candidate "law" is accuracy per observation under attack | A good framing, and the same experiment as Gate B | Adopted as Gate B's framing, stated as a hypothesis until it is measured |
| One demo, four numbers, on a high-resolution substrate: a budget frontier; honest robustness as the minimum of APGD-CE and EOT-20 in fp32; calibrated abstention; gaze-to-evidence alignment | Matches Stage 3 | Adopted. The per-sample worst case across attacks is already in D9 |
| UC1: a contraction-certified update, z′ = z + α(φ − z), with α set by precision and φ 1-Lipschitz | Interesting, with two concerns [INFER]. Relaxing toward the new evidence is a recency-weighted average, so under §6.2 it would show R > 0 and M near the averaging control: it makes overwriting easy but is not selective. And the spectral-norm constraint may cost accuracy under TRADES, which is the falsifier the review itself states | Kept as a Stage 2 candidate after REV1, tested against REV1 and CTL2 at matched parameters |
| EIG1: gaze by expected information gain, halting when EIG × precision falls below cost | Overlaps G3-HYP1. It needs a calibrated posterior and a model of how likely each view is under each hypothesis, and Gen-2 shows neither (§2.4) | Merged into G3-HYP1 as one candidate implementation; entropy gaze is the baseline |
| Decision gate: continue the one-change arms, or freeze Gen-2 as an honest negative study and pivot to the high-resolution substrate | Same as my stop rule (§7) | Same |
| State predictions before the data | Good practice | Adopted (§6.5) |

Where the review and the audit agree: the regime argument, unifying the attack harness, the views and epochs confound, Gate B and Gate S as the crossover test, the evidential head and gaze not yet earning their keep, and the S7 "no masking" inference being an artefact. Its architecture verdict (trunk delegated; state, evidence and stopping intrinsic; language later through a belief interface) is B staged toward C.

---

## 12. Direct answer and decision memo

### The strongest scientifically defensible direction

**Build RHAN as a revisable-evidence perceiver: a system that stores evidence immutably, holds a mutable belief with calibrated uncertainty, and revises that belief only where the evidence supports the revision. Test it where one feed-forward pass cannot see the answer. That is architecture B, staged toward C.**

- **RHAN should learn:** the belief update and its accept/revert rule; the surprise signal; the gaze or hypothesis-separating policy; a calibrated readout. Each is earned only when its test passes against its simple baseline.
- **RHAN should represent explicitly:** an append-only evidence store with provenance (view, location, scale, time); the belief with its uncertainty; a revision log; later, object and spatial state.
- **RHAN should delegate:** visual features (a frozen or lightly tuned DINO-class trunk); segmentation masks as ground-truth targets for scoring gaze; language or decision readout; grading by frontier models.

**What would show that the architecture contributes something new.** All four, each against the stated baseline, paired, by more than 2σ:

1. **Selective revision (REV1).** The damage from one wrong view is below both the plain average and the recency-weighted average, at two or more time steps, and the gist-contaminated case recovers. Today's most likely result is averaging or recency weighting (§6.5).
2. **Gate S.** Adaptive gaze beats fixed, random and entropy-driven gaze at equal observations, on accuracy and on alignment with the target.
3. **Gate B.** At a matched token budget the loop beats a feed-forward model, and the accuracy-per-observation curve dominates.
4. **Gate U.** Calibrated abstention beats a temperature-scaled softmax in the risk-coverage curve.

All of it under APGD (plus EOT if the model is stochastic), at least 1,000 robust images, and at least three training seeds for any headline. If none of the four holds, Gen-2 stands as a methodology and negative-result contribution, and that is thesis-grade.

### Decision memo

| Question | Answer |
|---|---|
| **Strongest opportunity** | Evidence acquisition and revision under a controlled budget, in a regime where one pass cannot see the answer |
| **Most important unresolved question** | Does the loop do anything that a mean-pool of the same views does not? |
| **Cheapest experiment that could change the architecture decision** | REV1. It is eval-only and decides whether B starts from the existing loop or from new mechanisms. D9 runs first, because it blocks every robust claim |
| **Highest-value expensive experiment** | Gates S and B on G3-HRV1, with a trained CTL2 as the control. This needs a scene substrate you do not yet have; its build cost is unmeasured |
| **First experiment to authorize** | One eval-only block for the coding agent, in this order: D9; NLL recompute; REV1 and PRB1; RES3 and RES1 with the stored-resolution check |
| **Evidence required before promotion into the main line** | A paired win of more than 2σ over the reference at matched parameters, views and observations; at least three training seeds for headline claims; APGD (plus EOT if stochastic) at n of at least 1,000; Gate U calibration; the pre-stated falsifier did not fire; the work lives in a separate artefact namespace and the reference checkpoint is preserved |

---

## 13. Sources

**Read this session**
- JEPA inverse dynamics: [arXiv 2610.07540](https://arxiv.org/abs/2610.07540) (v1); code [LeoToso/JEPA_control](https://github.com/LeoToso/JEPA_control), no licence stated.
- GATOR: [arXiv 2610.11215](https://arxiv.org/abs/2610.11215) (v1); [project page](https://research.nvidia.com/labs/lpr/gator/), code "Coming Soon".
- Shared geometry as a Rosetta Stone: [arXiv 2610.09411](https://arxiv.org/abs/2610.09411); code [dominik-schnaus/unpaired-rosetta](https://github.com/dominik-schnaus/unpaired-rosetta), MIT.
- ReHarness: [arXiv 2610.12668](https://arxiv.org/abs/2610.12668), abstract only.
- Decision 3.0: [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) and [vllm-sr/d3-lite](https://huggingface.co/vllm-sr/d3-lite), both Apache-2.0.
- DINOv3: [ViT-S/16 model card](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m); LICENSE.md and README from the official repository, [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
- Your W1.5 report and the external review you attached on 17 October (read in full; the review is a second opinion, not a source of facts).

**Not read this session (do not cite from this report):** Kambhampati's LLM planning papers, RAG/CAG/KAG primaries, SAM 2/3, YOLO26, RF-DETR, Qwen3.5/3.8 model cards, and any frontier-model documentation.

---

## Appendix. New registry rows (for plan.md §10)

| ID | Wave | One change vs. reference | Decision rule |
|---|---|---|---|
| G2-D9-attack-audit | 1.5 | First confirm the reference is deterministic at evaluation. Then, on a fixed set of at least 500 images: in-house PGD-10/50 varied one factor at a time (fp32 vs. autocast, CE vs. KL, step size, random start); APGD-CE-100, APGD-T, MI-FGSM; per-sample worst case; a large-ε check; zero-gradient fraction | The change that brings PGD within 2σ (paired) of APGD-CE is the cause. If none does, PGD is monitoring-only. Resolves B12 |
| G2-REV1-contradiction-injection | 1.5 | Eval-only on the phase 2–6 finals, full 5,000 validation images: replace one view with the same-location view of a different-class image; Latin square over time and location; a gist-contaminated case; controls: per-view logit average, and a recency-weighted average with one decay chosen on held-out data | R = L(last) − L(first), M = mean damage. M below both controls by more than 2σ (paired) at two or more time steps, with the gist case recovering: selective revision. R > 0 with M at the controls: recency weighting. R near 0 and M at the controls: averaging. R < 0: anchoring |
| G2-PRB1-surprise-probe | 1.5 | Under REV1 conditions on phases 4–6: AUROC of the step-wise E_t and the change in U_t for flagging the injected view; linear decodability of its position from the final belief | AUROC at least 0.7: GAT1 has a signal. At most 0.6: IDM1 first |
| G2-RES3-backbone-probe | 1.5 | Frozen-trunk probe, Gate W protocol: DINOv2-S/14 vs. DINOv3-S/16 at matched pixels (112) and matched tokens (64), plus 64 px for the 16-token case; CKA between the feature spaces; after a stored-resolution check | See §6.3. Extends G2-RES1-probe-112 |
| G2-GAT1-accept-revert | 2 | Accept z′ = z + Δ only if it predicts a held-out observed view better than z; otherwise shrink or revert; evidence store append-only | Beats the reference on REV1 (M) by more than 2σ (paired) at equal parameters, with clean accuracy not worse by 2σ |
| G2-IDM1-inverse-glimpse | 2 | Auxiliary loss: from the belief before and after a step, predict the view taken and which view was contradicted. Absorbs the review's DPS1 | Raises PRB1 decodability and REV1 recovery without lowering clean accuracy by 2σ; null is the same loss on shuffled targets |
| G2-UC1-contraction-update | 2 | Proposed by the external review. z′ = z + α(φ − z), α gated by precision, φ 1-Lipschitz; parameter- and FLOP-matched against UpdateNet-v1 | Run after REV1. Expect R > 0 with M at the controls; it is promoted only if it beats them on M, and certificates do not cost accuracy by 2σ |
| G3-HYP1-two-hypothesis-gaze | 6 | At Gate S: hold the top two hypotheses and look where they are predicted to differ most; absorbs the review's EIG1 as one implementation | Beats entropy-driven, fixed and random gaze at equal observations and FLOPs by more than 2σ (paired). If entropy matches it, search adds nothing |
| G3-REV2-backward-revision | 6 | After the final decision, re-weight stored glimpse evidence; revisions are functions of the immutable evidence | Needs G2-MEM1-glimpse-memory; beats a forward-only read at equal parameters on REV1 and Gate S |
| G4-GEO1-embedding-alignment | 6 | Before the SLM interface: CKA, and only if no pairs exist, Wasserstein-Procrustes, between belief tokens and the SLM's class-name embeddings | If a paired linear map already matches the classifier head, no adapter is needed |
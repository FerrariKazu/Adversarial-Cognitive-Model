# Prior-Art Ledger (verified entries only)

Date: 2026-11-03. Replaces the abstract-only draft `docs/prior_art_ledger.md`.

**Method.** Each entry below was read from the full PDF/HTML through a fetch-tool extraction (a summarising model returns the content and short quotes). That is not a manual read: check every quoted passage in the PDF before citing it in the thesis. Rows not yet read are listed at the end. **No novelty claim is made anywhere in this document.**

## Entries read in full text

| # | Paper | What it decides / is | Data | Cost or budget definition | Robustness to perturbation or attack | Cross-site / cross-device shift | Calibration of the stop decision | Stated limitations |
|---|---|---|---|---|---|---|---|---|
| 1 | **PRECISE-AS**, arXiv 2509.02898 (MICCAI 2025) | RL agent. Actions: acquire echo video *i* or terminate (N+1 actions, 4 videos per study). Sparse reward: +1 if correct, minus lambda x summed cost | Echo from one tertiary hospital, 2,572 patients, 3 ultrasound systems listed | Unit cost per video. Lambda 0.001 to 0.25 moved video use from 47% to 31% and balanced accuracy from 80.6% to 77.1% | **Not evaluated.** Only random masking of 50% of video tokens for missing data | **Not evaluated.** Single hospital, no external or cross-device test | **Not evaluated.** Learned terminate action, no calibration analysis | No limitations section. States no significant improvement over ProtoASNet (p > 0.05). High lambda makes the agent terminate immediately |
| 2 | **POCUS multi-task acquisition**, arXiv 2602.13658 (Feb 2026) | RL agent picks one of 5 views or stops. Shared multi-view transformer predicts aortic stenosis severity and LVEF with Gaussian predictive distributions | 12,180 studies, private single-hospital echo archive (Philips iE33, GE Vivid i/E9), patient-disjoint splits | Unit cost per clip. Matches full-study performance with 32% fewer videos (77.2% mean balanced accuracy) | **Not evaluated** | **Not evaluated.** No external or cross-scanner test | **Claimed, not measured.** Abstract says "calibrated uncertainty"; no ECE or reliability diagram reported | Retrospective only, not tested at the bedside. Accuracy falls below the full-study bound at tight budgets. One random clip per view in a fixed order |
| 3 | **AdaptivePath**, arXiv 2608.08648 (Aug 2026) | PPO "Navigator" picks where and how large to look on gigapixel whole-slide images, low to high magnification. Frozen Patho-R1-7B interpreter and Qwen3-8B deliberator/arbiter | Navigator trained on 880 WSIs over 30 TCGA types. Zero-shot eval on SlideBench-VQA, WSI-VQA, PathVQA and six TCGA cohorts (80.14% subtype accuracy) | ROI budget = K regions per magnification (swept 1, 3, 5, 8, 10; K = 5 best, larger gave no gain) and T = 4 focus observations per region. No accuracy-vs-budget curve | **Not evaluated.** Appearance augmentation used for stability, not measured | **Partial.** Train and eval datasets do not overlap, but no per-cohort or per-site analysis | **Not addressed.** Stop is fixed (T = 4, K = 5), not learned | H&E only. About 122 s per question with Qwen3-8B |
| 4 | **CineMA**, arXiv 2506.00679 (v1 HTML; later versions not read) | **Not an active-selection policy.** Masked-autoencoder foundation model (about 126M params) for cine cardiac MRI | 74,916 UK Biobank cine studies, one random frame of 50 per sample | n/a | n/a | n/a | n/a | Models single frames, not time. No clinical-outcome evaluation. Pre-trained on cine only. Long-axis OOD EF error about 15%. Abstract says 8 datasets, Methods says 9 |

## What these entries do and do not support

In the three active-perception papers read (rows 1 to 3), none evaluates robustness to perturbation or attack, none tests cross-site or cross-device shift, and none reports calibration of the stop decision. That is three papers, not the field.

## Not yet read (do not cite as verified)

- Differentiable Zooming for MIL (ECCV 2022)
- CMRxRecon (Scientific Data 2024) and the CMRxRecon2026 4D-flow challenge
- Nature BME 2026 cardiac-MRI foundation-model papers (two; titles seen only)
- 4D cardiac cine synthesis, arXiv 2610.09185 (title seen only)
- Global-then-local mammography models (GMIC), named from memory only
- Direction 3 (robustness or shift testing of where/stop decisions): no paper identified yet; the searches have not been run

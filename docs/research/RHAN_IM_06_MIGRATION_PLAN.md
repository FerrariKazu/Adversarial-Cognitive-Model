# 06 — Migration plan: from the current RHAN code base to RHAN-IM

Principle: **add, never rewrite.** The redesign lives on branch `redesign/rhan-im` and only *adds* paths (`rhan_im/`, `tests/rhan_im/`, `scripts/rhan_im/` and the documents). No existing source file, checkpoint, baseline, evaluation artifact, pre-registration or git history is modified. The validated checkpoints stay the baseline arms of `RHAN_IM_05_EXPERIMENTS.md`.

## 1. What happens to each existing asset

| Asset | Fate | Why |
|---|---|---|
| `CompactViTGen2` trunk, DINOv2 warm start | **Reuse** through `rhan_im.trunk_adapter.ff_token_fn` (exactly the `ff_112` forward path; the tied refinement block is not run) | the trunk is the sensor; a test proves bit-identical tokens after loading a checkpoint in the training format |
| `ff_112` checkpoints (seeds 17, 42, 99; official = seed 17, sha256 `d00fb750…`) | **Frozen trunk** for gates 1–2; **baseline arm A0** | validated, hash-pinned; only ever read |
| `recurrence_112` checkpoints (seeds 17, 42) | **Baseline arm A2**; untouched | the recompute comparator |
| TRADES loop, curriculum constants, AdamW grouping, warm-up/cosine | **Reuse** in stage S4 (adversarial training on the final posterior) | comparability with every existing result |
| Evaluation harness (PGD-50, APGD-CE, APGD-DLR, worst-case AND, 5k val), pre-registration practice | **Reuse unchanged** for E6; new attacks (adaptive, patch, compute) are *added* beside it | the harness is the part of the repo that is already publication-grade |
| `phase5_sdt/sdt_core.py` (d′, β) | **Align conventions**; RHAN-IM's measured `d′` follows the same definitions | the lab's psychophysics habit becomes the sensor model |
| Tied refinement block, gist gate, belief phases, AIS v2, `c3`/`c4` controls | **Freeze as archival Gen-2 arms**; not deleted, not extended | they cannot show value on glimpses that carry no new pixels (`RHAN_IM_01_FORENSIC_AUDIT.md` F4–F5) |
| Model card for `ff_112` (DRAFT) | **Rewrite the claim, not the checkpoint**: "DINOv2-S/14-shaped ViT, TRADES-trained on ImageNet-100 at 112 px"; quote 23,369,573 unique parameters (state_dict is 226,176 larger because of the aliased patch embedding); keep DRAFT until the ImageNet-terms caveat and AutoAttack/ε-clamp conventions are fixed | the card should describe what the artifact is |
| `RHANfuture.md` seven-gap programme | **Re-scope**, see §2 | one mathematical core instead of seven features |

## 2. How the seven Gen-3 "gaps" map onto RHAN-IM (and which are deferred)

| Gap in `RHANfuture.md` | In RHAN-IM |
|---|---|
| 3 Saccadic foraging, inhibition of return | **Subsumed**: the ignorance map *is* inhibition of return (a cell seen at its best `d′` has zero marginal gain in best mode); the policy is computed, not learned |
| 7 Global workspace / metacognition | **Subsumed**: the posterior and its calibrated risk are the metacognitive signal; LTT makes it actionable |
| 1 Temporal processing | **Same machinery, different looks**: frames as looks; the exact-Bayes state and the stopping rule apply unchanged (a pivot option, `RHAN_IM_03_CANDIDATES.md` §9) |
| 2 Hierarchical predictive coding | **Deferred**; could supply the evidence head's features |
| 4 Neuromodulation, 5 Sparse coding, 6 Semantic grounding | **Deferred**; not needed for the central question |

The ε-threshold projections in `RHANfuture.md` §1 are targets, not results, and are not carried forward.

## 3. Phases, gates, deliverables

| Phase | Entry | Work (new files only) | Exit gate | If the gate fails |
|---|---|---|---|---|
| **M0** (this branch) | — | prototype of the mathematics, trunk adapter, measurement script, budget, tests, documents | tests pass; documents reviewed; branch committed | — |
| **M1 Gate 1** | prereg frozen (`RHAN_IM_07_GATE1_PREREG_DRAFT.md`); checkpoint + ImageNet-100 available to the user's agent | run `scripts/rhan_im/gate1_measure_dprime.py` on the official `ff_112`; 3 head seeds; secondary trunks 42, 99 | verdict per pre-registered rule | S3 (tune last blocks) once; else switch conditions in `RHAN_IM_03_CANDIDATES.md` §9 |
| **M2 Gate 2** | gate 1 passed | scene generator script; S1 head training; LTT calibration; policy comparison A0–A7; E2, E3, E4, E5 | pre-registered E2/E3/E5 criteria | report; policy or risk layer revised; no robustness work yet |
| **M3 Gate 3** | gate 2 passed | look-granular certificate (E7); adaptive attacks (E6); S4 training; E8–E12 | pre-registered E6/E7 criteria | publish the negative robustness result; keep calibration/risk-control claims |
| **M4 write-up** | gates 1–3 reported | honest model card for `ff_112`; paper outline around the central question; natural-scene check E14 | — | — |

No phase estimates GPU-hours (you asked for none); each has a cost expressed in looks, windows and scenes instead.

## 4. Environment notes

- The sandbox this branch was built in has no network access to the weights or data, so **nothing here has touched the real checkpoint**; everything real-trunk is `random-init` or a *script ready to run*.
- The scripts read the checkpoint only; they never write next to it, never read credentials, use no network. (Tokens stay in your `.env` / Kaggle / Colab environment; none is needed.)
- Runs fit the target hardware by construction: gate 1 holds one frozen trunk forward and a 384×(C+1) linear head; stage S4 needs micro-batches of ≤ 24 scenes on an 8 GB card (`04` §9).

## 5. Risk register

| Risk | Likelihood (my judgement) | Mitigation |
|---|---|---|
| `d′` of the frozen trunk is too low at 112 px | medium | S3 dense window CE; else temporal looks / fallback |
| Cross-look correlation ≈ 1 (looks add no independent evidence) | medium-high for a context-poor trunk | "best" mode is designed for it; the gain then comes from new regions and resolution only |
| Pasted-square artefact inflates `d′` | medium | E14 natural-scene check; claims limited to the stimulus |
| Tokens of one window are strongly dependent (`VIF` ≫ 1): the independent-token posterior is overconfident | medium-high for a global-attention trunk | measured at gate 1; `1/√VIF_null` correction; shared difficulty (`VIF_pos / VIF_null`) flagged and handled by the scale mixture in S2; E2 by target size; a look-level evidence unit if `d_block` < 2 |
| Calibration and test scenes share source images | fixed in the gate-1 script | disjoint halves of the val images (tested) |
| Look-granular certificate is weak because a patch touches ≈ 25–50 % of windows | high | report as measured; test masking, smaller windows |
| Adaptive attacks defeat the policy | medium | E6 strategies 2–3; compute-exhaustion margin |
| Scope creep back to seven features | high | this document's §2 |
| Overclaiming from simulations | — | `RHAN_IM_RESULTS.md` and every document label simulated numbers as mechanism-level |

## 6. Decisions that are yours

1. Commit and push the branch (it is currently local and uncommitted; pushing needs the repository to be attached with push access).
2. Freeze the gate-1 pre-registration (thresholds in `07`), or edit them first.
3. Approve the target/background split of the ImageNet-100 classes (first half / second half of the sorted class list).
4. Whether to run the optional Stage A replication (closes Gen-2; off the critical path).
5. Whether to rewrite the model card now (recommended: yes, with the claim in §1).

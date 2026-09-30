# RHAN / NOESIS — CLAIM vs IMPLEMENTATION AUDIT

Generated 2026-09-15 · branch `feature/rhan-next` @ fe7a709 · companion to `RHAN_NOESIS_PROJECT_EVIDENCE_DOSSIER.md` (§13 condensed there; this file is the full register).

**Method:** every row pairs a statement found in README / roadmap / architecture docs / paper draft / code comments with what the code and artifacts actually contain. Statuses: **VERIFIED-MATCH** (claim consistent with implementation), **PARTIAL** (claim overstates; the named subset is real), **METAPHOR** (framing language, no operational content), **STALE/WRONG** (contradicted by evidence), **NOT IMPLEMENTED**, **UNVERIFIED** (cannot be checked from repo).

---

## A. Terminology claims

| # | Claim (source) | Actual implementation | Status | Evidence |
|---|---|---|---|---|
| A1 | "Biologically inspired perceptual intelligence" (NOESIS_FOUNDATION.md title + README) | Structural metaphors: ventral/dorsal channel split, recurrence depth 2, halting gate, π_d precision weighting. No neural-data constraints, no biological learning rules, no Brain-Score evaluation of any RHAN model. Doc itself disclaims: "NOESIS is not a claim about biological fidelity" (line ~486). | METAPHOR (with self-aware disclaimer) | `[CODE: model_rhan_stl10_large.py, rhan_core/*]` `[ART: NOESIS_FOUNDATION.md]` |
| A2 | "Human-like vision" / "human alignment" (RHANfuture.md, paper framing) | Robustness numbers exist; the SDT pipeline compares external CNNs/ViT + humans on CIFAR-10; **no RHAN model has been scored against human data**; human stimuli use CIFAR-10 classes vs RHAN's STL-10 training (class-set mismatch). | UNVERIFIED (as a claim about RHAN itself) | `[ART: phase5_sdt/final_report_6model.txt; phase3_human_study/manifest.csv]` |
| A3 | "Predictive coding" (widespread: pillar 1 name, docs, corpus [33]) | ONE level; tap = foveal crop; target = non-learnable edge map; scalar error → auxiliary loss (w=0.10). No hierarchy (>1 level rejected by config), no generative model across levels, no precision-weighted error propagation, no local/Hebbian learning rule. Trunk's `PredictiveCodingLayerLarge` is a separate feedback-error gate. | PARTIAL — accurate phrasing: "predictive-coding-inspired auxiliary loss" | `[CODE: hpc_level1.py, pillar_config.py, model_rhan_stl10_large.py]` |
| A4 | "Active inference" (v12 header: "Locked-In Active Inference Architecture"; NOESIS §8.2) | π_d-weighted belief update + gradient-based gaze on prediction error are present; there is no free-energy objective, no variational bound, no expected-free-energy action selection. | PARTIAL — say "active-inference-inspired belief update + gaze" | `[CODE: model_rhan_v12.py, rhan_core/model.py]` |
| A5 | "CLIP-grounded" (NOESIS history: v4 live CLIP loss) | CLIP was used as a live loss in v4-era training and documented as harmful (smoothed features); no CLIP term exists in current code or losses. | HISTORICAL, RETIRED | `[ART: NOESIS_FOUNDATION.md ~line 726-728]` `[CODE: no CLIP import in phase1_training/rhan_core]` |
| A6 | "Concept bottleneck" (Finding-9 framing in NOESIS_FOUNDATION ~line 1365) | No CBM heads, no concept supervision, no CBM loss anywhere. The referenced finding is a Pi_D per-class diagnostic (car/truck/horse collapse under auxiliary losses). | NOT IMPLEMENTED (diagnostic only) | `[CODE: no cbm/concept modules]` `[ART: roadmap gate telemetry Pi_D]` |
| A7 | "Self-supervised" (pseudo-labeling descriptions) | Training uses confidence-thresholded pseudo-labels (conf ≥ 0.65 over 100K unlabeled → ~46K mixed) — that is semi-supervised self-training, not an SSL objective (no contrastive/JEPA/masking losses). Corpus [91] FixMatch explicitly marked "Not implemented — could adopt". | PARTIAL — say "pseudo-label semi-supervision" | `[CODE: trainer pipeline]` `[ART: isolation verdict data-mix ruling]` |
| A8 | "Adversarially robust" (all results docs) | PGD-50/PGD-100 white-box, norm-space ε ∈ {0, 0.031, 0.062, 0.094}; masking checks vs PGD-50→100. No AutoAttack / C&W / FAB / square / black-box / transfer evaluation anywhere (corpus flags AutoAttack as the required standard). | PROVISIONAL — robust *to PGD at tested ε*, under the documented nondeterminism caveat | `[CODE: phase2_attacks/pgd.py, eval_rhan.py]` `[ART: all verdicts]` |
| A9 | "World model" (Pillar 4) | `NullWorldModel` passthrough with zero params; `enable_iwm` validates to False by design; `SimulatedGazePolicy` interface reserved, unimplemented. | SCAFFOLD ONLY | `[CODE: rhan_core/world_model/null_world_model.py, pillar_config.validate()]` |
| A10 | "Adaptive computation" (halting descriptions) | Soft continuation weights σ(softness·(u−thr)); the loop always runs T=max_steps (4 or 6); `trajectory['steps']` is constant. Hard per-sample early exit explicitly deferred. | PARTIAL — "soft uncertainty-gated belief accumulation" | `[CODE: rhan_core/model.py _forage, halting.py]` `[ART: deferred_increments]` |
| A11 | "Uncertainty-aware" | Three implemented uncertainty sources (1−π_d → slot-attention entropy → evidence-decomposition uncertainty). No calibration metrics (ECE, reliability diagrams, NLL) anywhere; SDT d′ is an offline analysis, not a model property. | PARTIAL — uncertainty is *consumed*, never *calibrated* | `[CODE]` `[NOT FOUND: calibration metrics]` |
| A12 | "Object-centric representations" (SBR framing) | Slot attention over spatial features implemented (16×512, 3 iters); sbr0 gate shows healthy occupancy/diversity but per-slot probes ≈0.50 across ALL slots and benign ablation — object specialization is not demonstrated by any recorded metric. | IMPLEMENTED; OBJECT-CLAIM UNPROVEN | `[ART: sbr0_gate_verdict.json]` `[MEASURED]` |
| A13 | "Information gain" gaze (AIS naming) | AIS-v1 is relocated Eq.-II gradient ascent + identity-init residual; the roadmap *forbids* calling it information gain (`result_labeling_rule`). Genuine EIG exists only as untrained AIS-v2 code. | METAPHOR for v1 (policy enforces honest labeling); IMPLEMENTED-UNVALIDATED for v2 | `[ART: roadmap stage1 result_labeling_rule]` `[CODE: info_gain_policy_v2.py]` |
| A14 | "Evidence accumulation" (corpus tag, SBR-3 framing) | EvidenceDecomposition heads (shape/texture/spatial) + pooled-evidence belief exist and train; no drift-diffusion/sequential-sampling machinery; "accumulation" = the 4-step belief average. | PARTIAL | `[CODE: evidence_decomposition.py, structured_belief.py]` |
| A15 | "Relational" (SBR-3) | One attention-based inter-slot message-passing round (SlotRelationalLayer, B,K,K relation matrix) per step. | VERIFIED-MATCH (to its own modest description) | `[CODE: beliefs/relational.py wiring in structured_belief.py]` |

## B. Numeric / factual claims

| # | Claim (source) | Reality | Status | Evidence |
|---|---|---|---|---|
| B1 | "Parameters: ~52M" (`model_rhan_stl10_large.py` docstring) | 55,622,347 | STALE/WRONG | `[MEASURED 2026-09-15]` |
| B2 | "~63.4M parameters" (`model_rhan_v12.py` docstring) | 75,440,469 | STALE/WRONG | `[MEASURED 2026-09-15]` |
| B3 | "76M params" (stage-2 starvation note), "81M" (corpus HYDRA aside) | matches nothing measured; full-pillar RHANNext = 84,470,149 | STALE (approximations of different configs) | `[MEASURED]` |
| B4 | "RHANNext default is byte-identical to v12" (rhan_core/model.py contract) | param counts equal (75,440,469); design enforced by tests | VERIFIED-MATCH | `[MEASURED]` + `[ART: stage 0 acceptance]` |
| B5 | Stage-1 headline "+8.5 pp, n.s." | matches verdict JSON incl. 2σ=8.84 | VERIFIED-MATCH | `[ART: stage1_verdict]` |
| B6 | Stage-2 "CROSSOVER REAL" (+7.33) | matches verdict (2σ=5.16, 5 seeds) | VERIFIED-MATCH | `[ART: stage2_verdict]` |
| B7 | Stage-3 "CROSSOVER REAL" (+11.55 8-seed; +9.79 16-seed) | matches both verdict records; two n's coexist and must be labeled | VERIFIED-MATCH (with labeling duty) | `[ART: stage3_verdict, e1_verdict]` |
| B8 | E1 "clean 16/16 wins, adv −0.9 n.s." | matches paired stats | VERIFIED-MATCH | `[ART: e1_verdict]` |
| B9 | E2 "SBR does not improve on D in Gen-0 form" | matches (adv −0.6 n.s., clean −9.9) | VERIFIED-MATCH | `[ART: e2_verdict]` |
| B10 | E3 "T=6 deferred" | matches (adv −3.25 n.s.) | VERIFIED-MATCH | `[ART: e3_verdict]` |
| B11 | sbr0 gate "passed" | all four criteria pass in JSON | VERIFIED-MATCH (probe-uniformity caveat is ours, not the gate's) | `[ART: sbr0_gate_verdict.json]` |
| B12 | sbr1 "62.49 clean, gate passed" | passes only under the 2026-09-11 one-sided amendment; original symmetric gate failed it | VERIFIED-MATCH post-amendment; amendment honestly recorded | `[ART: sbr1 verdict + gate_amendment]` |
| B13 | sbr3 eval table (2026-09-15 log) | 59.44±2.78 / 27.69±3.11 PGD-100; 28.83±2.44 PGD-50; structural assertion passed | VERIFIED-MATCH | `[LOG]` |
| B14 | "With 3 seeds it is noisy" (eval_rhan.py output note) | hardcoded stale text; current protocol is 16 seeds | STALE COSMETIC BUG (misleading if quoted) | `[LOG: eval output note]` |
| B15 | "STL-10: 5,000 train / 8,000 test / 100,000 unlabeled, 96×96, 10 classes" | matches loader + torchvision STL10 | VERIFIED-MATCH | `[CODE: dataset_stl10.py]` |
| B16 | "Human clean baseline 73% is pixelation, not perceptual failure" (SDT report) | self-declared caveat in the report; underlying participant data not in repo | UNVERIFIED (self-report) | `[ART: final_report_6model.txt]` |
| B17 | "AIS-v1 gaze shows no significant relationship to local perturbation magnitude" (roadmap ais_v2.eval) | checked-in JSON: Pearson r=0.207 (p=0.024), Spearman ρ=0.260 (p=0.004) — statistically nonzero, practically small | CONTRADICTED IN EMPHASIS (both statements trace to real artifacts) | `[ART: report/gaze_perturbation_correlation.json; roadmap]` |
| B18 | "eval_rhan.py is frozen" (roadmap frozen_files) | file is in frozen_files list; but note text inside it (B14) is stale — frozen ≠ correct | VERIFIED-MATCH (frozen) with known-cosmetic-defect caveat | `[ART: roadmap]` `[LOG]` |

## C. Paper-draft claims (Paper/, ACD_paper_v1.tex)

| # | Claim | Reality | Status | Evidence |
|---|---|---|---|---|
| C1 | "RHAN… biologically motivated architecture" (paper) | see A1/A3/A4 | METAPHOR+PARTIAL | `[ART: ACD_paper_v1.tex]` |
| C2 | "biological fidelity does not automatically imply adversarial robustness" (paper's CORnet-S finding) | supported by phase5 (CORnet-S ε_thresh 0.009) | VERIFIED-MATCH (external evidence) | `[ART: phase5 report]` |
| C3 | Any claim that RHAN's mechanisms are the cause of the human-machine gap closure | no RHAN-in-SDT or RHAN-human comparison exists | UNVERIFIED | `[NOT FOUND]` |
| C4 | Paper's model table accuracy/robustness figures | draft-stage; must be regenerated from the per-seed CSVs before submission (Gen-1 report is stale per dossier §18.3) | PROVISIONAL | `[ART: report/rhan_nx_generation1_report.md]` |

## D. Special-duty claims (the ones the dossier brief called out)

| Claim | Status | One-line verdict |
|---|---|---|
| "Biological" | METAPHOR | structure-inspired naming; zero biological measurement of the models |
| "Human-like" | UNVERIFIED | no RHAN-human comparison exists |
| "Predictive coding" | PARTIAL | single-level edge-map auxiliary loss |
| "CLIP-grounded" | RETIRED | v4 history only; harmful-then-removed |
| "Concept bottleneck" | NOT IMPLEMENTED | Pi_D diagnostics only |
| "Active inference" | PARTIAL | belief update + gradient gaze; no free-energy machinery |
| "Self-supervised" | PARTIAL | pseudo-labeling, not SSL |
| "Adversarially robust" | PROVISIONAL | PGD-white-box evidence only; AutoAttack pending |
| "World model" | SCAFFOLD | NullWorldModel; enable_iwm locked False |
| "Human alignment" | UNVERIFIED | pipeline exists for external models; RHAN never scored |

## E. Registry of stale documentation that must be fixed before external consumption

1. Parameter counts in both model docstrings (B1, B2).
2. `report/rhan_nx_generation1_report.md` renders all ladder stages PENDING (predates its inputs).
3. eval_rhan.py's "with 3 seeds" note (B14).
4. Roadmap inline `rhan_nx.stages.*` statuses (runtime HF state is authoritative; the file itself says never hand-edit — so consumers must be told to read runtime).
5. Emphasis drift on the gaze-perturbation result (B17) — pick one phrasing and cite the JSON numbers.

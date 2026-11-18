# 17 — References

The **status** column is part of the reference, not decoration: this dossier may not rest on a source
marked *unverified* without saying so. "Verified" here means the identifier was checked against the
canonical record at the time of writing; it does not mean the claim drawn from the paper was replicated.

## Methods the programme relies on

| Ref | Work | Identifier | Status |
|---|---|---|---|
| R1 | Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 | arXiv:2304.07193 | verified |
| R2 | Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020 | arXiv:2006.15055 | verified |
| R3 | Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), CVPR 2023 | arXiv:2301.08243 | verified |
| R4 | Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025 | arXiv:2506.09985 | verified |
| R5 | Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, 2022 | arXiv:2208.02814 | verified |
| R6 | Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks*, NeurIPS 2017 | arXiv:1705.08500 | verified |
| R7 | Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 | arXiv:1706.04599 | verified |
| R8 | Najemnik & Geisler, *Optimal eye movement strategies in visual search*, Nature 2005 | doi:10.1038/nature03390 | verified |
| R9 | Koopman, *The theory of search I. Kinematic bases*, 1956 | doi:10.1287/opre.4.3.324 | verified |
| R10 | Lindley, *On a measure of the information provided by an experiment*, 1956 | doi:10.1214/aoms/1177728069 | verified |

R8 and R9 are the intellectual ancestors of the "look where the expected gain is highest" policy; the
programme's planner is a descendant of that line with a different objective (displacement of the
leader, not reduction of search uncertainty). R10 is the origin of expected information gain, the
baseline the planner must beat.

## Vision memo sources (not independently verified)

The memo that motivated this programme cites the following. They are recorded as supplied; this
dossier does **not** rely on their numbers, and they must be checked before any claim uses them.

| Ref | Claim as supplied | Identifier | Status |
|---|---|---|---|
| M1 | active multi-view recognition jointly selects informative features and viewpoints | ScienceDirect S0921889016303554 | unverified |
| M2 | 2026 review: active visual perception requires selecting actions for their effect on future observations | PubMed 42944409 | unverified |
| M3 | MVTec AD 2: >8,000 images, eight scenarios, lighting changes, reflective/transparent objects, leading methods below 60 % AU-PRO | Springer s11263-026-02743-0 | unverified |
| M4 | I-JEPA's large-scale ImageNet experiments included ViT-Huge trained on 16 A100 GPUs | CVF Open Access (I-JEPA paper) | consistent with R3, not separately checked |

If M3 is confirmed, it is the intended real-data target for the transfer check in `docs/11` §5. If it
is not confirmable, the transfer check uses a different public industrial anomaly dataset and says so.

## This repository's prior work cited by the programme

| Ref | Work | Where |
|---|---|---|
| P1 | RHAN-IM redesign: search-with-stopping prototype, dependence measurement (VIF ≈ 11.6) | `docs/research/RHAN_IM_00_README.md`, `rhan_im/` |
| P2 | RHAN-IM Gate 2 v1.2: risk control at ≈11 % coverage | `report/`, `rhan_im/gate2.py` |
| P3 | RHAN Technical Handbook v3.4: evidence classes, number provenance, figure system | `docs/rhan_nxa/`, `docs/research/RHAN_HB_00_README.md` |
| P4 | Gen-0 shared-gradient competition findings | `FINDINGS.md`, `RHAN-history.md` |

These are internal records, not peer-reviewed evidence. They are cited as design context, not as
external validation.

## How to cite this dossier

> RHAN-Bi v1: Evidence-Integrity Active Perception — research programme and reference implementation.
> Branch `rhan-exp`, 2026.

No DOI exists; the repository commit is the identifier. `results/run_manifest.json` records the commit
for every run.

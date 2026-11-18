# 14 — Literature and novelty

## 1. Purpose

This document fixes what the programme may and may not claim. It is written before the literature
audit is complete, so that the audit has a target: the claim below is the one that must be checked
against prior art, and it is deliberately narrower than "a system that looks again".

## 2. Established areas the programme depends on

| Area | Representative work | What it establishes | What RHAN-Bi takes |
|---|---|---|---|
| self-supervised visual features | DINOv2 (Oquab et al., 2023) | strong frozen features transfer across tasks without full fine-tuning | the optional backbone; feature quality is not the contribution |
| object-centric representation | Slot Attention (Locatello et al., 2020) | slots bind to objects and support compositional scenes | a candidate module, **not** in v1; must be earned by a measured limitation |
| predictive representation learning | I-JEPA (Assran et al., 2023), V-JEPA 2 (2025) | latent prediction is a viable objective; video extends it to planning | context and long-term direction, not a v1 component; large-scale pretraining is out of scope |
| active perception | multi-view active recognition (e.g. 2016–2026 literature), the 2026 review cited by the vision memo | selecting actions for their effect on future observations is established | the baseline class (EIG, uncertainty) and the framing |
| uncertainty and selective prediction | conformal risk control (Angelopoulos et al., 2022), selective classification literature | risk can be controlled under assumptions; coverage is separable from risk | the calibration and abstention machinery; the coverage floor is a direct response to RHAN-IM's Gate 2 |
| evidence and provenance in ML | data provenance, audit trails, argumentation | traceability is a systems property, not a model property | the ledger's design |

The citations above are standard and are recorded with status in `docs/17_REFERENCES.md`. The vision
memo's specific 2026 sources (a PubMed review and an MVTec AD 2 evaluation) are marked
*user-supplied, unverified*; this dossier does not rest on their numbers.

## 3. The candidate contribution

> **Evidence-integrity active perception:** a perception loop in which each hypothesis carries a
> traceable, quality- and dependence-aware evidence account, contradictions remain visible, and the
> next observation is chosen for its expected ability to displace the leading interpretation, with
> belief revisions attributable to the records that caused them.

The claim is about the *combination and its measurement*, not about any single ingredient. Each
ingredient exists in the literature; what is claimed as potentially new is:

1. dependence-discounted evidence accounting as part of the belief state, not a post-hoc correction;
2. a falsification-directed selection objective compared at matched budget against EIG and a strong
   hand-designed protocol;
3. revision accuracy as a first-class metric, measured on episodes where the initial belief was wrong;
4. an explicit, reported evidence-debt representation that turns uncertainty into a named reason to
   act.

Any one of these alone is likely not novel. The novelty audit must find prior work that combines at
least three of them; if it does, the contribution is downgraded to "systematisation and evaluation".

## 4. Closest prior art to check

The audit must search, at minimum:

- active object recognition / next-best-view with hypothesis discrimination;
- Bayesian experimental design and expected information gain in vision;
- sequential diagnosis and fault isolation with evidence ledgers;
- uncertainty-aware active perception and belief-space planning;
- selective prediction and conformal risk control in perception;
- provenance-aware and auditable ML systems;
- falsification/testing-oriented ML (e.g. adversarial testing, counterfactual explanation);
- industrial inspection with multi-illumination (MVTec AD 2 and the illumination-change literature).

For each, record: does it (a) keep per-observation evidence with dependence, (b) select observations
for discrimination rather than information, (c) evaluate revision after contradiction, (d) report
coverage with risk? A row of four "no"s is the space the contribution claims. Anything else narrows
the claim.

## 5. Novelty audit protocol

1. Search scholarly sources first (OpenAlex, CORE, arXiv), then web.
2. For every candidate, read the method section, not the abstract.
3. Fill the four-column table in §4 from the method, with a page/section pointer.
4. Record the strongest single paper that comes closest, and write one paragraph on why the programme
   is or is not different.
5. If the closest paper already does (a)+(b)+(c), stop and rewrite the contribution as replication.
6. Date the audit and commit it; a novelty claim without a dated audit is not permitted in any report.

## 6. What will not be claimed

- that RHAN-Bi "understands" images or "knows" when it is wrong in a human sense;
- that the simulator demonstrates real-world competence;
- that falsification-directed selection is superior in general (only at this budget, on this task, with
  this evidence model);
- that the ledger makes the system trustworthy (it makes it auditable);
- any causal claim about defects from an illumination change (that is Gate E / CEE, not v1).

## 7. Positioning against RHAN-IM

RHAN-IM framed perception as search with a stopping rule and an exact posterior over locations. RHAN-Bi
keeps the sequential framing but changes the object of belief from *where is the target* to *why does
this appearance exist*, and changes the action from *where to look next* to *which intervention would
discriminate*. The dependences measured in RHAN-IM (VIF ≈ 11.6) are the reason dependence is a
first-class term here rather than a note in the limitations.

That is a genuine difference in framing, but it is not yet a demonstrated difference in capability.
Gate D/E exist to turn the framing into a measurement.

## 8. Publication shape (if gates pass)

A methods paper with three figures (loop, ledger, gate ladder), one scenario plate, and four result
panels: single-observation ceiling, risk-coverage, revision accuracy, matched-budget policy comparison.
The negative-result version is the same paper with the policy panel flat and the calibration/revision
panels intact; it is writable now, which is a reason to run the experiment rather than postpone it.

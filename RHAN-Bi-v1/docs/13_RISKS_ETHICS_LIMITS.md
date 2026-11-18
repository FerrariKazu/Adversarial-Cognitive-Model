# 13 — Risks, ethics, limits

## 1. Risk register

| ID | Risk | Likelihood | Impact | Mitigation | Residual |
|---|---|---|---|---|---|
| R1 | simulator is built (unintentionally) so the preferred policy wins | medium | fatal to H1 | shared features/model, strong EIG and protocol baselines, oracle headroom report, ambiguity band in Gate A | a simulator artefact could still favour light changes; real-data check is the only real defence |
| R2 | simulator results do not transfer to real inspection | high | high | explicit transfer check (Gate C/D real data), claim level capped at feasibility | transfer remains unknown in v1 |
| R3 | single-observation ambiguity is weaker or stronger than intended | medium | medium | Gate A acceptance band; scene generator review if outside | the band is a design judgement |
| R4 | dependence constants (ρ) are not supported by the data | medium | medium | empirical residual-correlation report; ablation `no_dep` | constants may over- or under-discount |
| R5 | planner's forward model and stopping share errors | high | medium | reported; separate stopping model specified for Gate D | correlated errors are not corrected in v1 |
| R6 | evidence model is a diagonal Gaussian and miscalibrated | medium | medium | temperature scaling, ECE reporting, calibration residual report | calibration may not hold under shift |
| R7 | compute limits prevent the full confirmatory run | low | medium | demo runs on CPU; full run is minutes; DINOv2 optional and cached | none material |
| R8 | novelty claim is wrong (prior art does this) | medium | high | literature audit protocol (`docs/14`), no novelty claim before it | must be checked before submission, not after |
| R9 | the planner's gain is just the light-change prior, not falsification | medium | high | `fixed_light_first` baseline; per-action decomposition; report where the gain comes from | if the gain is entirely "change the light", that is the finding, reported as such |
| R10 | metric gaming via abstention | low | high | abstention counted as error in primary; coverage floor in Gate C | none |
| R11 | p-hacking across gates and policies | medium | high | pre-registration, Holm, deviation log, demo labelled feasibility | requires discipline, not tooling |
| R12 | evidence ledger gives a false sense of auditability | medium | medium | ledger records provenance, not truth; stated in `docs/03` §9 and in every report | irreducible |

R9 deserves emphasis: on this task the informative action is almost always "change the light". If the
falsification policy's advantage over EIG disappears once the first action is forced to be a light
change, then the contribution is a *task insight*, not a *planner*. The Gate E report includes that
decomposition, and the paper's claim would be rewritten accordingly.

## 2. Ethics

The system is a perception aid for inspection. Three commitments:

1. **Abstention is visible.** The system never presents a low-evidence guess as a confident answer. A
   commit carries its posterior, debt and observation count; an abstention is a first-class outcome
   that a human sees.
2. **No autonomous action on the physical world.** The planner requests observations (light, view,
   zoom) only. It does not move parts, reject products biologically or actuate a production line. Gate
   E is a simulation study; any deployment requires a separate safety review.
3. **Human oversight of the evidence.** Because the ledger is exportable, a human can inspect exactly
   which observations supported a decision and which contradicted it. This is the practical value of
   provenance even when the model is wrong.

Data ethics: v1 uses synthetic images only. The real-data check uses a public industrial dataset under
its licence, with no personal data. If a deployment later uses customer images, the ledger must be
treated as potentially sensitive production data (it contains image hashes and could contain crops),
and retention must be specified before collection.

## 3. What would invalidate the programme

- **H1 fails** at the confirmatory run (CI includes 0). The active-perception contribution is not
  supported; the calibration/auditing contribution may still stand.
- **Gate A fails** (encoder incompetent): nothing downstream is interpretable; fix the encoder first.
- **Gate C fails on coverage** (risk target met only by abstaining on most scenes): the reliability
  machinery is decorative, and H3 cannot be claimed.
- **Real-data transfer fails** while simulator results hold: the laboratory is the limitation, and the
  programme needs a real-data gate before any applied claim.
- **Novelty audit finds the combination already published**: the contribution becomes replication and
  engineering, and must be reported as such.

## 4. Limits that no amount of code fixes

- **A reflection that is pixel-identical to a defect under every available intervention cannot be
  distinguished.** The simulator makes them distinguishable by a light change; the real world may not.
  Identifiability is a property of the scene and the action set, not of the model.
- **A single observation cannot certify its own reliability.** Quality is estimated from the image; an
  image that is uniformly corrupted in a way the estimator does not model will pass.
- **The ledger records what was observed, not what was true.** A systematically biased sensor produces
  a perfectly auditable wrong belief.
- **Dependence is modelled, not measured per pair.** The constants are a stand-in for a covariance the
  simulator could provide but a real camera would not.
- **The stopping rule's value estimate uses the same forward model as the planner.** Model error is
  shared.

## 5. Reporting obligations

Every report generated by this package must include:

1. the config hash, git commit and seed hash;
2. N, exclusions, and the split used;
3. the claim level (demonstrated / feasibility / exploratory);
4. coverage beside every risk number;
5. the limitations from this document, at least in summary;
6. the statement that the ledger records provenance, not truth.

`rhan_bi.evaluation.report` emits these by construction, and `python -m rhan_bi.cli verify` fails if a
report lacks them.

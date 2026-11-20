# RHAN Phase 2 — PCI Dossier Amendment 1 (v1.1)

**Parent document:** *RHAN Research Phase 2 — Visual Perception Foundation Dossier (PCI)*, v1.0. The parent is preserved unchanged. This file is a separate, versioned amendment and does not edit, replace, or reinterpret the parent except for the matters explicitly listed in §0.

**Status:** design amendment for review. **Not** an implementation authorization, training authorization, protocol change, or result revision. No code, checkpoint, experiment, dataset split, threshold, or result is modified by this document. Stage 0A remains independently scoped under §H.

**Date:** 2026-11-20.

**Review basis:**

1. Independent adversarial review of the PCI proposal (DeepSeek report), findings 1–7.
2. The read-only RHAN-Bi-v1 audit, committed as `9ba5149` on `rhan-exp`, especially confirmed defects F1–F18 and unresolved questions Q1–Q12.
3. The PCI review packet supplied by the project owner.

**Repository note:** at the time of writing, no tracked file matching the PCI dossier title was present in this workspace. The parent is therefore identified by title and version rather than by a repository path. This amendment is self-contained so that it does not depend on line-level access to the parent.

---

## 0. What this amendment changes, in one page

| # | Parent PCI position | Amendment 1 position | Consequence |
|---|---|---|---|
| 1 | Per-observation outputs are described as calibrated LLR vectors. | **Rejected as stated.** PCI is redefined as a **discriminative set-based predictor** using calibrated posterior log-odds scores. LLR terminology is retained only inside an explicitly generative branch that derives and validates likelihoods. | No claim of Bayesian evidence combination unless the generative branch is instantiated. |
| 2 | A dependence descriptor makes integration dependence-aware. | **Partially accepted.** The descriptor is redefined as a set of pairwise, testable features. It is not assumed to represent conditional dependence; that claim must be earned by prediction of held-out residual dependence and by calibration gains under constructed dependence. | “Dependence-aware” becomes a hypothesis, not a label. |
| 3 | Invariants were qualitative. | **Accepted in full.** Seven invariants are made mathematical and paired with automated tests. | The integrator cannot be trained until the invariant harness exists. |
| 4 | The integrator is motivated as a better evidence combiner. | **Accepted with a stronger hypothesis.** The integrator must beat named baselines on a pre-registered primary endpoint, and must lose its advantage when constrained to mean pooling if the advantage is genuinely set-level. | “Beats pooling” is necessary but not sufficient. |
| 5 | Insufficient evidence is an output state. | **Accepted.** It is defined operationally as a decision-theoretic abstention under a risk or cost constraint, not as a label-free confidence threshold. | Abstention is evaluated by risk–coverage, not by accuracy alone. |
| 6 | Evaluation relies on accuracy plus OOD. | **Accepted.** Multi-observation integration is evaluated on real multi-view data with object-level separation, plus constructed dependence/contradiction conditions. OOD remains a secondary stress test. | A generic OOD gain cannot establish integration. |
| 7 | RHAN-Bi-v1 offers reusable components. | **Rejected as a merge premise.** RHAN-Bi-v1 is a source of failure cases and mandatory tests, not an implementation to merge. | No code from RHAN-Bi-v1 enters PCI by default. |

**Stage 0A ring-fence.** The DINOv2-S/B visual baseline in Stage 0A does **not** depend on the integrator mathematics in §§C–G. Stage 0A may proceed under separate authorization while this amendment is reviewed. It must not be delayed to resolve the later integrator questions, and it must not be used to retroactively justify the integrator.

---

## A. Point-by-point response to the adversarial findings

### A.1 Finding 1 — Likelihood ratios versus posterior probabilities

**Finding.** Temperature-scaled discriminative posterior outputs cannot automatically be interpreted as likelihood ratios. Combining them as LLRs double-counts priors and misstates the assumptions required for evidence accumulation.

**Disposition: ACCEPTED.**

**Resolution.** PCI v1.1 adopts the **discriminative formulation as the primary definition**. The per-observation head produces a calibrated posterior \(q_i(c)=p_\theta(c\mid x_i,a_i)\). PCI calls

\[
s_i(c)=\log q_i(c)-\log q_i(c_0)
\]

a **posterior log-odds score**, not a likelihood ratio. It is a feature. It is not assumed to be \(\log p(x_i\mid c)/p(x_i\mid c_0)\).

The generative/LLR branch is retained only as an optional research branch with explicit assumptions (§C.3). It is not the default, and no result from the discriminative branch may be reported using LLR language.

**Why not simply keep LLRs with a prior correction?** Because a prior correction repairs the prior term, not the likelihood term. A temperature-scaled softmax still does not identify \(p(x\mid c)\). It identifies a posterior under the training distribution. Calling it an LLR after subtracting a prior is a relabeling, not a derivation. The amendment therefore removes the claim rather than decorating it.

**What is retained.** Temperature scaling remains valid for **calibrating a posterior**. It is not valid for converting a posterior into a likelihood ratio. This distinction is now explicit in the glossary and in the output schema.

---

### A.2 Finding 2 — Conditional dependence

**Finding.** A single dependence descriptor cannot represent arbitrary conditional dependence; it may simply be another learned feature. The proposal must distinguish exact duplicates, near-duplicates, independent observations, and contradictory observations, and must not claim dependence-aware inference without demonstrating it.

**Disposition: PARTIALLY ACCEPTED.**

**Accepted part.** A scalar “dependence score” is not a model of conditional dependence. The amendment replaces it with a **pairwise dependence feature set** and makes the dependence claim conditional on validation.

**Partially rejected part.** The criticism that a learned descriptor can never be useful is too strong. A learned pairwise compatibility function can capture dependence if it is trained and tested against held-out residual dependence. What is rejected is the **unearned claim**, not the mechanism.

**Resolution.** The descriptor is defined in §C.4 as

\[
d_{ij}=\bigl[\mathbb{1}_{\text{exact dup}},\ \mathbb{1}_{\text{near dup}},\ \hat\rho_{ij},\ \psi_\phi(h_i,h_j,a_i,a_j)\bigr],
\]

where \(h_i\) is a frozen or jointly trained observation embedding, \(a_i\) is acquisition metadata, \(\hat\rho_{ij}\) is an estimated residual correlation on calibration data, and \(\psi_\phi\) is a learned pairwise compatibility score. The integrator may consume \(d_{ij}\), but the descriptor is a **feature**, not a certificate of dependence.

**Evidence required before using the phrase “dependence-aware.”**

1. \(\hat\rho_{ij}\) predicts held-out residual correlation with a pre-registered error bound.
2. The integrator with \(d_{ij}\) has lower selective risk than the same integrator with \(d_{ij}\) removed, on sets constructed to have known dependence.
3. The gain is not reproduced by a set-size feature or by mean pooling.
4. The gain survives a duplicate/near-duplicate stress test.

If any of 1–3 fails, the correct description is “set-based integrator with pairwise features,” not “dependence-aware inference.”

---

### A.3 Finding 3 — Mandatory invariants

**Finding.** PCI must incorporate static-scene permutation invariance, duplicate-observation resistance, valid empty-set behavior, single-observation consistency, prior-shift handling, contradiction handling, and observation provenance, each with mathematical expectations and automated tests.

**Disposition: ACCEPTED IN FULL.**

**Resolution.** The seven invariants are formalized in §E with pass criteria and test procedures. The invariant harness is a **prerequisite** for any integrator training run. A training run that has not passed the harness is not a PCI result.

**Additional invariant accepted from the RHAN-Bi-v1 audit.** The harness includes an explicit order-permutation test, because F1 showed that a documented commutativity claim can be false in practice. The harness also includes a duplicate-saturation test derived from F3 and a provenance test derived from F4/F5.

---

### A.4 Finding 4 — Integrator justification

**Finding.** The central hypothesis is underspecified. PCI must define what the integrator learns, its inputs and outputs, its training objective, and its expected advantage over temperature-scaled mean pooling, logistic regression on set-level features, an integrator without dependence descriptors, conformal abstention, and other justified non-neural baselines. It must also show that the integrator is not merely learning to average its inputs.

**Disposition: ACCEPTED.**

**Resolution.** §F restates the hypothesis as a pre-registered superiority claim with a named primary endpoint, a named baseline family, an ablation that removes dependence features, a mean-pooling-constrained ablation, and a test for whether the non-pooling component carries predictive value.

**Key clarification.** The hypothesis is not “a neural integrator can fit the training sets.” It is:

> On held-out multi-observation sets, a set-based integrator \(g_\phi\) achieves lower selective risk at a fixed coverage than the best simple baseline, by a pre-registered margin, after matching training data, encoder, and compute; and the advantage is attributable to set-level/dependence information rather than to extra capacity.

The second clause is what makes the hypothesis falsifiable. The first clause alone can be satisfied by a larger model that learns average-logit pooling.

---

### A.5 Finding 5 — Abstention and insufficient evidence

**Finding.** “Insufficient evidence” must be defined operationally, with a training or derivation procedure, risk control, and a way to distinguish genuine abstention from rejecting difficult examples.

**Disposition: ACCEPTED.**

**Resolution.** §C.6 and §G define abstention as a decision-theoretic action:

\[
\text{abstain if } \mathbb{E}[\ell(\hat y,y)\mid S] > c_{\text{abstain}},
\]

or, when no cost is specified, as a selective-risk constraint at a target coverage. The threshold is set on calibration data by conformal or Learn-then-Test risk control, never on test. The evaluation reports risk–coverage curves, AURC, and the fraction of correct-but-abstained cases. A system that abstains only on intrinsically difficult examples may be valid, but it must still show a risk reduction per unit coverage; otherwise abstention is relabeled as low-confidence rejection.

---

### A.6 Finding 6 — Experimental protocol

**Finding.** Metrics must separately establish classification accuracy, calibration, selective risk, evidence integration, and correct belief revision. The falsification experiment must evaluate actual multi-observation integration, preferably on appropriately separated real multi-view data, not exclusively on a generic OOD test. Sample sizes must follow from a minimum meaningful effect size.

**Disposition: ACCEPTED WITH ONE MODIFICATION.**

**Accepted.** The primary endpoint moves from generic OOD accuracy to **multi-observation selective risk / AURC** on real multi-view sets. OOD remains a secondary stress test. Sample size is derived from a stated minimum effect in §G.5.

**Modification.** Stage 0A remains a single-observation visual baseline and is **not** required to test integration. It is independently scoped and may run first. This is not a loophole: no Stage 0A result may be presented as evidence for the integrator hypothesis.

---

### A.7 Finding 7 — RHAN-Bi-v1 lessons

**Finding.** Incorporate the RHAN-Bi-v1 audit as mandatory architectural safeguards, do not merge its implementation into PCI, and treat it as a source of failure cases and test requirements.

**Disposition: ACCEPTED.**

**Resolution.** §E.4 maps each confirmed audit defect F1–F18 to a PCI safeguard or test. §H states explicitly that no RHAN-Bi-v1 module is merged. The audit is cited as a failure catalogue.

---

## B. Accepted, partially accepted, and rejected criticisms

### B.1 Accepted

| Criticism | Why accepted | Where addressed |
|---|---|---|
| Calibrated posteriors are not LLRs. | Mathematically correct; prior and likelihood are conflated by temperature-scaled softmax. | §C.1–C.3 |
| Temperature scaling does not create likelihoods. | Temperature is a monotone recalibration of posterior confidence. | §C.2 |
| A scalar dependence descriptor cannot represent arbitrary conditional dependence. | Dependence is pairwise/structured and hypothesis-dependent. | §C.4 |
| The integrator hypothesis is underspecified. | No primary endpoint, baseline, or attribution test was stated. | §F |
| “Insufficient evidence” is not identifiable from confidence alone. | It is a decision, not a label. | §C.6 |
| Generic OOD is insufficient to establish integration. | OOD tests distribution shift, not multi-observation combination. | §G |
| Sample sizes must be derived from an effect size. | Arbitrary N cannot support a superiority claim. | §G.5 |
| RHAN-Bi-v1 defects must become safeguards. | F1–F18 are concrete failure modes. | §E.4 |

### B.2 Partially accepted

| Criticism | Accepted part | Rejected or limited part | Reason |
|---|---|---|---|
| A learned dependence descriptor is useless. | An unvalidated descriptor is not evidence of dependence. | A validated pairwise compatibility function can be useful. | Learned pairwise features can predict residual dependence if tested; the claim, not the mechanism, was the problem. |
| PCI must be a generative likelihood model. | LLR language requires a likelihood model. | A discriminative set predictor is a legitimate primary formulation. | The project’s near-term goal is prediction under uncertainty, not generative modelling; forcing a generative model would delay Stage 0A and add unvalidated assumptions. |
| Temperature scaling is invalid. | It cannot convert posteriors to LLRs. | It remains valid for posterior calibration. | The mathematical role of temperature is calibration, not likelihood identification. |
| OOD testing is not useful. | It cannot establish integration. | It remains a secondary stress test for calibration and abstention. | OOD is informative about robustness; it is simply not the primary integration endpoint. |
| The integrator is just average pooling. | The risk is real and must be tested. | It is not proven for every possible integrator. | The mean-pooling-constrained ablation and attribution test decide it empirically. |

### B.3 Rejected

| Criticism | Rejection | Technical justification |
|---|---|---|
| PCI should be abandoned because its evidence language was wrong. | Rejected. | The mathematical error is in the **label and inference claim**, not in the idea of a calibrated set predictor with abstention. Redefining the object preserves the testable part. |
| A neural integrator can never beat pooling. | Rejected as an a priori claim. | Pooling is optimal only under conditional independence and equal reliability; real multi-view observations violate both. Whether a learned integrator captures the violation is empirical. |
| Only a fully generative model can be scientifically valid. | Rejected. | Discriminative models are standard scientific instruments when their claims are stated as conditional predictions and calibrated. The amendment removes Bayesian evidence claims rather than forbidding discriminative prediction. |
| A single OOD number can falsify integration. | Rejected. | OOD performance can change for reasons unrelated to integration (encoder shift, prior shift, calibration drift). The primary endpoint must be a multi-observation comparison. |
| Stage 0A should wait until the integrator mathematics is settled. | Rejected. | Stage 0A is a single-observation visual baseline with independent value and independent failure modes. Delaying it would confound encoder quality with integrator quality. |

---

## C. Revised mathematical formulation

### C.1 Notation

| Symbol | Meaning |
|---|---|
| \(x_i\) | observation image \(i\) |
| \(a_i\) | acquisition metadata: view, light, ROI, timestamp, sensor settings |
| \(\tau_i\) | provenance record: image hash, preprocessing hash, model version |
| \(c\in\mathcal{C}\) | hypothesis/cause; \(c_0\) is the reference cause |
| \(q_i(c)\) | calibrated per-observation posterior |
| \(s_i(c)\) | posterior log-odds score, **not** an LLR |
| \(S=\{o_1,\dots,o_n\}\) | set of observations, \(o_i=(x_i,a_i,\tau_i)\) |
| \(g_\phi\) | set integrator |
| \(\pi\) | training prior |
| \(\pi'\) | deployment prior |

### C.2 Primary formulation: calibrated discriminative set prediction

The per-observation head is

\[
q_i(c)=p_\theta(c\mid x_i,a_i)=\operatorname{softmax}\bigl(f_\theta(x_i,a_i)/T\bigr)_c,
\]

where \(T\) is fitted on a held-out calibration split by minimizing NLL or Brier score. The score passed to the integrator is

\[
s_i(c)=\log q_i(c)-\log q_i(c_0).
\]

**Definition.** \(s_i(c)\) is a **posterior log-odds score**. It is not a likelihood ratio. It is not assumed to satisfy \(s_i(c)=\log p(x_i\mid c)/p(x_i\mid c_0)\).

The integrator is

\[
p_\phi(c\mid S)=\operatorname{softmax}\Bigl(g_\phi\bigl(\{(s_i,u_i,a_i)\}_{i=1}^n,\{d_{ij}\}_{i<j}\bigr)\Bigr)_c,
\]

where \(u_i\) is an uncertainty summary (entropy, max-probability, or a calibrated variance) and \(d_{ij}\) is the pairwise dependence feature of §C.4.

**Training objective.** For set-level labels \(y\):

\[
\mathcal{L}(\phi)=
\frac{1}{N}\sum_{k=1}^{N}\Bigl[
-\log p_\phi(y_k\mid S_k)
+\lambda_{\text{cal}}\,\mathcal{L}_{\text{cal}}(p_\phi(\cdot\mid S_k),y_k)
+\lambda_{\text{dep}}\,\mathcal{L}_{\text{dep}}(\hat\rho_{ij},d_{ij})
\Bigr].
\]

- \(\mathcal{L}_{\text{cal}}\) is Brier or NLL on a calibration fold; ECE may be reported but is not the training objective.
- \(\mathcal{L}_{\text{dep}}\) is optional and is used only if the dependence descriptor is claimed to predict residual dependence. It is a prediction loss on held-out pairs, not a reconstruction of the integrator’s attention.
- No term encourages confidence directly. Accuracy-only training is not sufficient.

**Prior handling.** If the model is trained under prior \(\pi\) and deployed under \(\pi'\), the **single-observation** posterior may be adjusted exactly when the likelihood is unchanged:

\[
\operatorname{logit} q_i'(c)=\operatorname{logit} q_i(c)+\log\frac{\pi'_c}{\pi_c}.
\]

For the set integrator, the prior is either (a) supplied as an input during training and tested under shift, or (b) applied externally to a prior-free set score. Applying the single-observation correction after a nonlinear integrator is not automatically valid and must be tested, not assumed.

### C.3 Optional generative/LLR branch

This branch is **not** the primary PCI definition and is not authorized for the first integrator experiment. It is recorded so that LLR language has a precise meaning if the project later chooses it.

Assume a class-conditional observation model \(p_\theta(x\mid c)\), a prior \(\pi_c\), and observations conditionally independent given \(c\). Then

\[
\ell_i(c)=\log p_\theta(x_i\mid c)-\log p_\theta(x_i\mid c_0),
\]

and the posterior is

\[
p(c\mid S)\propto \pi_c\exp\Bigl(\sum_{i=1}^n \ell_i(c)\Bigr).
\]

This is valid **only if**:

1. \(p_\theta(x\mid c)\) is a normalized likelihood, not a discriminative softmax;
2. the prior is external and explicitly stated;
3. observations are conditionally independent given \(c\), or the joint \(p_\theta(x_1,\dots,x_n\mid c)\) is modelled directly;
4. the likelihood is trained on data whose acquisition process matches deployment;
5. calibration is evaluated on the posterior after prior correction.

If any of 1–3 fails, the sum is not a Bayesian posterior. The correct response is to use the discriminative formulation of §C.2, not to relabel the sum.

**Why this branch is deferred.** Estimating a normalized class-conditional likelihood in DINOv2 feature space is a separate research programme. It would introduce generative assumptions that are harder to validate than the discriminative baseline. The project should not adopt it merely to preserve the phrase “likelihood ratio.”

### C.4 Dependence descriptor, precisely

For each pair \((i,j)\), define

\[
d_{ij}=
\bigl[
\mathbb{1}_{\text{exact dup}}(i,j),
\ \mathbb{1}_{\text{near dup}}(i,j),
\ \hat\rho_{ij},
\ \psi_\phi(h_i,h_j,a_i,a_j)
\bigr].
\]

**Exact duplicate.** \(x_i\) and \(x_j\) have identical pixel content after the declared preprocessing, or identical canonical image hash. Exact duplicates are collapsed to one canonical observation before integration. The integrator must be idempotent under exact duplication.

**Near-duplicate.** A perceptual hash distance below a pre-registered threshold, or embedding cosine similarity above a threshold, **and** compatible acquisition metadata. Near-duplicates are grouped into an equivalence class. The class contributes at most the maximum evidence of its members, not the sum, unless a validated dependence model says otherwise.

**Independent observation.** Different acquisition metadata, different scene region or viewpoint, and residual correlation below a pre-registered threshold estimated on calibration data. Independence is **not** the default; it is a tested label.

**Contradictory observation.** Two observations whose calibrated posteriors have high confidence in different top causes, or whose pairwise residual is an outlier under the calibration residual model. Define the disagreement statistic

\[
\kappa_{ij}=\operatorname{JS}\bigl(q_i\,\|\,q_j\bigr),
\]

and a contradiction flag when \(\kappa_{ij}>\kappa^{\ast}\) and \(\arg\max q_i\ne\arg\max q_j\). Contradiction is not averaged away; it must increase entropy or trigger abstention.

**What the descriptor is not.** It is not a proof of conditional dependence. \(\psi_\phi\) is a learned feature. The phrase “dependence-aware” is permitted only after the validation in §A.2 passes.

**What the integrator may do with it.** The integrator may use \(d_{ij}\) as an attention bias, a graph edge feature, or a gating input. The ablation in §F.4 removes \(d_{ij}\) entirely; if performance is unchanged, the descriptor is not carrying dependence information.

### C.5 Duplicate and near-duplicate saturation

Let \(S_m\) be a set containing \(m\) copies of the same canonical observation. The required behavior is:

\[
p_\phi(c\mid S_{m+1})=p_\phi(c\mid S_m)\quad\text{for exact duplicates},
\]

and for near-duplicates,

\[
D_{\mathrm{KL}}\bigl(p_\phi(c\mid S_{m+1})\,\|\,p_\phi(c\mid S_m)\bigr)\le \epsilon_{\text{dup}}
\]

for all \(m\), with \(\epsilon_{\text{dup}}\) pre-registered. The sequence \(\{p_\phi(c\mid S_m)\}_{m\ge1}\) must be bounded and converge; it must not grow like \(\sum_{k=1}^m 1/k\) as in RHAN-Bi-v1 F3.

### C.6 Abstention as a decision

Let \(\ell(\hat y,y)\) be the task loss and \(c_{\text{abstain}}\) the cost of abstaining. The optimal action is

\[
a^\*(S)=
\begin{cases}
\arg\min_{c}\mathbb{E}[\ell(c,y)\mid S], & \text{if } \min_c\mathbb{E}[\ell(c,y)\mid S]\le c_{\text{abstain}},\\
\text{abstain}, & \text{otherwise}.
\end{cases}
\]

When no cost is available, report the **risk–coverage curve** and choose a threshold \(\tau\) on calibration data such that the selective risk at coverage \(1-\beta\) is at most \(\alpha\) with confidence \(1-\delta\). The threshold must be selected on the same probability scale used at evaluation.

“Insufficient evidence” is therefore an operational state: the expected loss of predicting exceeds the cost of abstaining, or the calibrated selective-risk constraint cannot be met. It is not “the model is not confident,” and it is not a label supplied by the dataset.

---

## D. Revised PCI architecture and component interfaces

### D.1 Components

| Component | Input | Output | Status |
|---|---|---|---|
| Observation encoder | \(x_i\) | embedding \(h_i\) | Frozen DINOv2-S/B in Stage 0A; fine-tuning only in a later authorized stage. |
| Per-observation head | \(h_i,a_i\) | \(q_i(c)\), \(s_i(c)\), \(u_i\) | Trained and calibrated on single observations. |
| Provenance layer | \(x_i,a_i,\tau_i\) | canonical observation id, hashes | Mandatory; rejects mismatched bytes. |
| Duplicate/near-duplicate grouper | \(x_i,h_i,a_i\) | equivalence class, \(d_{ij}\) duplicate flags | Mandatory before integration. |
| Dependence feature builder | \(h_i,h_j,a_i,a_j\), calibration residuals | \(d_{ij}\) | Optional; claim contingent on §A.2. |
| Set integrator | \(\{(s_i,u_i,a_i)\}\), \(\{d_{ij}\}\) | \(p_\phi(c\mid S)\) | Not trained until invariant harness passes. |
| Abstention layer | \(p_\phi(c\mid S)\), calibration threshold | predict/abstain | Conformal or LTT threshold; no test tuning. |
| Evidence record | all of the above | immutable record with hashes | For audit, not for tamper-evidence claims. |

### D.2 Interfaces

**Observation record.**

```text
ObservationRecord:
  observation_id: str          # hash of canonical pixels + preprocessing
  image_sha256: str
  preprocess_sha256: str
  model_version: str
  acquisition: {view, light, roi, timestamp, sensor}
  embedding_ref: str
```

**Per-observation output.**

```text
ObservationPosterior:
  observation_id: str
  class_probabilities: float[K]   # calibrated, sums to 1
  score: float[K]                 # posterior log-odds, reference class 0
  uncertainty: float              # entropy or calibrated variance
  calibration_version: str
```

**Set output.**

```text
SetPrediction:
  class_probabilities: float[K]
  abstain: bool
  abstention_reason: str          # "risk", "cost", "contradiction", "empty"
  evidence_ids: list[str]
  dependence_features_used: bool
  model_version: str
  threshold_version: str
```

**Invariant.** Every `SetPrediction` must be reproducible from its `evidence_ids` and the declared model/threshold versions. If the record cannot be reproduced, the prediction is not a PCI result.

### D.3 What is deliberately absent

- No prefix-dependent weights.
- No `n_eff` in the inference path. If reported at all, it is descriptive and labelled as such.
- No tamper-evidence claim without authentication.
- No threshold selected on dev or test.
- No pair term labelled as a likelihood ratio.
- No RHAN-Bi-v1 module merged into PCI.

---

## E. Mandatory invariants and automated tests

### E.1 Invariants

Let \(S\) be a set of observations, \(\pi(S)\) a permutation, \(g\) the integrator, and \(q_o\) the single-observation posterior.

**I1 — Static-scene permutation invariance.**

\[
g(S)=g(\pi(S))\quad\text{for all permutations }\pi.
\]

Scope: static scene, fixed acquisition set. Active acquisition order is handled by the policy, not by the integrator. If the integrator consumes temporal order intentionally, it must be tested separately and the static-scene claim must not be made.

**I2 — Exact-duplicate resistance.**

For an exact duplicate \(o'\) of \(o\in S\),

\[
g(S\cup\{o'\})=g(S).
\]

**I2b — Near-duplicate saturation.**

For \(S_m\) with \(m\) near-duplicates,

\[
D_{\mathrm{KL}}\bigl(g(S_{m+1})\,\|\,g(S_m)\bigr)\le\epsilon_{\text{dup}},
\]

and \(\lim_{m\to\infty} g(S_m)\) exists.

**I3 — Valid empty set.**

\[
g(\varnothing)=\pi,\qquad \text{abstain}=\text{true}.
\]

No NaN, Inf, or undefined softmax.

**I4 — Single-observation consistency.**

\[
g(\{o\})=q_o
\]

within numerical tolerance, or the difference is explicitly documented and tested.

**I5 — Prior-shift handling.**

For a known prior change \(\pi\to\pi'\), the single-observation output satisfies

\[
\operatorname{logit} q'(c)=\operatorname{logit} q(c)+\log(\pi'_c/\pi_c).
\]

For the set integrator, either the same transformation holds (if the integrator is prior-separable) or the integrator is retrained with the prior as an input. The test must state which case applies.

**I6 — Contradiction handling.**

For two high-confidence observations with different top causes,

\[
H\bigl(g(\{o_i,o_j\})\bigr)\ge \max\bigl(H(q_i),H(q_j)\bigr)-\epsilon_{\text{contra}},
\]

or the confidence in either top cause must not exceed the maximum single-observation confidence. An integrator that becomes more confident under contradiction fails.

**I7 — Observation provenance.**

For every observation, `image_sha256` equals the hash of the stored bytes, and every prediction lists the observation ids it used. Mutating bytes, metadata, or model version without updating the record must be detected.

### E.2 Automated tests

| Test | Setup | Assertion | Pass criterion |
|---|---|---|---|
| T1 permutation | Random permutations of a fixed set, 100 trials | \(g(S)=g(\pi(S))\) | Max absolute logit difference \(<10^{-6}\) (or exact for deterministic implementation). |
| T2 exact duplicate | \(m=1,\dots,50\) copies of one observation | \(g(S_m)=g(S_1)\) | Exact equality; no drift. |
| T3 near duplicate | JPEG/crop/small-shift variants, \(m=1,\dots,20\) | KL bound and convergence | KL \(<\epsilon_{\text{dup}}\); monotone bounded sequence. |
| T4 empty set | \(S=\varnothing\) | Prior and abstain | Output equals \(\pi\); finite; `abstain=true`. |
| T5 single observation | One observation | Equals head output | Max absolute difference \(<10^{-6}\). |
| T6 prior shift | Synthetic labels reweighted by known \(\pi'\) | Analytic correction | Predicted logit shift equals \(\log(\pi'_c/\pi_c)\) within tolerance for the separable case; for the integrator, the retrained/conditioned case is tested. |
| T7 contradiction | Constructed pairs with opposite high-confidence evidence | Entropy/confidence bound | I6 holds; no overconfidence. |
| T8 provenance | Mutate image bytes, metadata, model version | Detection | Record rejected or `verify` fails. |
| T9 duplicate under new metadata | Same image, different step/action | Canonicalization | Grouped as exact duplicate; no extra evidence. |
| T10 schema/numerical | NaN, Inf, extreme scores, missing metadata | Validation | Rejected with a typed error, not a silent NaN posterior. |
| T11 batch invariance | Different batch composition and order | Same per-set output | Max difference \(<10^{-6}\). |
| T12 calibration coverage | Threshold on calibration, evaluate on held-out test | Empirical coverage | Within the pre-registered binomial interval; report the interval, not a point estimate alone. |

### E.3 Test harness requirements

1. Tests run on CPU with deterministic seeds.
2. The harness is independent of the training loop; it can test a saved model or a stub.
3. Every test writes a machine-readable pass/fail record with the model hash, threshold version, and commit.
4. A failed invariant blocks integrator training. There is no “known failure” waiver without a documented amendment.
5. The harness includes a **negative control**: a deliberately broken integrator (e.g., prefix-weighted) must fail T1 and T2. If the harness cannot detect the known RHAN-Bi-v1 failures, the harness is not adequate.

### E.4 RHAN-Bi-v1 audit → PCI safeguards

| Audit defect | PCI safeguard | Test |
|---|---|---|
| F1 order-dependent posterior | Set-symmetric integrator; no prefix weights. | T1 |
| F2 `n_eff` reported but unused | `n_eff` removed from inference; if reported, labelled descriptive. | Schema check; no inference dependence. |
| F3 duplicate harmonic accumulation | Exact-duplicate canonicalization; near-duplicate cap. | T2, T3, T9 |
| F4 ledger not tamper-evident | No tamper-evidence claim; provenance hashes only; optional HMAC if authentication is added. | T8 |
| F5 image bytes not checked | Hash verification at ingestion. | T8 |
| F6 train/deployed model mismatch | One declared deployed model; train/cal/test separation. | Reproduction test. |
| F7 threshold scale mismatch | Threshold selected and applied on the same calibrated scale. | T12 |
| F8 risk omitted from gate | Selective risk at threshold is a pass criterion. | §G acceptance table. |
| F9 dev threshold leakage | Threshold only on calibration; test once. | Preregistration audit. |
| F10 revision metric is recovery rate | Trajectory Brier, justified-revision rate, confidence movement. | §G metrics. |
| F11 incomplete episode artifacts | Save every arm and every episode. | Artifact manifest test. |
| F12 DINOv2 not consumed | Stage 0A integration test; no feature path may be marked unused. | Stage 0A gate. |
| F13 planner ignores action identity | If a planner is added, condition on action identity. | Planner interface review. |
| F14 duplicates under new metadata | Canonicalization by image hash. | T9 |
| F15 no schema validation | Finite scores, typed schema, provenance validation. | T10 |
| F16 docs cite nonexistent artifacts | No artifact may be cited before it exists. | Documentation check. |
| F17 quality formula undocumented | Exact formula in the spec; no hidden factors. | Code/spec diff review. |
| F18 pair term not a likelihood | Pair term labelled feature; not summed as LLR. | Glossary and output schema. |

---

## F. Revised integrator hypothesis and baselines

### F.1 Hypothesis

**H1 (primary).** Let \(\mathcal{B}\) be the baseline family in §F.2. On held-out multi-observation sets drawn from the same acquisition process as training, the integrator \(g_\phi\) achieves a lower **selective risk at 80% coverage** (equivalently, lower AURC) than the best baseline in \(\mathcal{B}\), by at least \(\Delta_{\text{risk}}\), with a 95% confidence interval excluding zero, after matching the encoder, training-set size, and inference compute.

**H2 (attribution).** The advantage in H1 is attributable to set-level/dependence information, not merely to extra capacity. Operational test: the integrator constrained to mean pooling (or with \(d_{ij}\) removed) loses at least \(\rho_{\text{attr}}\) of the advantage, and the non-pooling component has predictive value on held-out dependence regimes.

**H3 (calibration).** The integrator is non-inferior to the best baseline on calibration (Brier and NLL), with a pre-registered non-inferiority margin. A selective-risk gain achieved by becoming overconfident is not a valid gain.

**H4 (abstention).** The abstention layer reduces selective risk relative to confidence-only rejection at the same coverage, or it is reported as not established.

H1–H4 are jointly necessary. A result that satisfies H1 but fails H2 is reported as “extra capacity,” not “dependence-aware integration.”

### F.2 Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Temperature-scaled mean pooling of \(s_i\) | The simplest evidence combiner; the primary baseline. |
| B1 | Logistic regression on set-level features \([\text{mean},\text{max},\text{var},\text{set size},\text{duplicate count}]\) | A non-neural set model. |
| B2 | Integrator without dependence descriptors (\(d_{ij}\) removed) | Tests whether \(d_{ij}\) carries information. |
| B3 | Integrator constrained to mean pooling (or average-logit head) | Tests whether the integrator is learning to average. |
| B4 | Conformal abstention wrapper on B0/B1 | Tests whether gains come from abstention rather than integration. |
| B5 | Random forest or Gaussian process on set aggregates | A non-neural nonlinear baseline. |
| B6 | Oracle with true cause labels (reported separately) | Upper bound, not a competitor. |

**Why these baselines.** B0 is the simplest defensible combiner under conditional independence. B1 and B5 test whether a simple set-level model is sufficient. B2 isolates the dependence descriptor. B3 isolates the integrator’s non-pooling capacity. B4 isolates abstention. B6 bounds the possible gain.

### F.3 What the integrator learns

The integrator learns a mapping from a **set** of calibrated observation scores, uncertainties, acquisition metadata, and pairwise features to a set-level posterior. It does not learn a likelihood. It may learn:

- reliability weighting (which observations are more informative),
- redundancy suppression (which observations duplicate each other),
- contradiction resolution (which observations conflict and how that changes uncertainty),
- set-level context (how many independent views support a cause).

These are hypotheses, not guarantees. The ablation in §F.4 determines which, if any, are realized.

### F.4 Ablations and attribution

1. **Remove \(d_{ij}\).** If performance is unchanged, the dependence descriptor is decorative.
2. **Constrain to mean pooling.** If performance matches the full integrator, the integrator learned to average.
3. **Remove uncertainty inputs \(u_i\).** Tests whether uncertainty is used or ignored.
4. **Remove acquisition metadata \(a_i\).** Tests whether the integrator uses view/light information.
5. **Train on shuffled set labels.** Negative control; should fail.
6. **Train on sets with known dependence structure.** If the integrator cannot exploit known dependence, it is unlikely to exploit real dependence.
7. **Capacity control.** Match parameter count and compute to the strongest baseline; report both matched and unmatched.

### F.5 Decision rule

H1 is supported only if:

\[
\Delta\text{AURC}>0,\quad \text{95% CI excludes }0,\quad \Delta\text{ECE}\le \epsilon_{\text{cal}},
\]

and the attribution test in H2 passes. If H1 passes but H2 fails, the claim is “set model with extra capacity,” not “dependence-aware integrator.” If H1 fails, the integrator hypothesis is falsified for this data and protocol; that is a reportable negative result.

---

## G. Revised experimental design and acceptance criteria

### G.1 Stage 0A — independent DINOv2-S/B visual baseline

Stage 0A is scoped independently and is **not** an integrator experiment. It answers: do frozen DINOv2-S/B features support the single-observation perception task at a useful accuracy and calibration?

| Item | Stage 0A specification |
|---|---|
| Encoder | DINOv2-S and DINOv2-B, frozen. |
| Head | Linear probe or shallow MLP; no set integrator. |
| Data | The Stage 0A dataset defined in the parent dossier; if ImageNet-1k is used, report the split and do not reuse it for integrator evaluation. |
| Splits | Train/cal/test separated by the declared unit (image or object, as appropriate). No test tuning. |
| Metrics | Top-1 and balanced accuracy; ECE (equal-width and equal-frequency); Brier; NLL; per-class intervals. |
| Acceptance | Pre-registered minimum accuracy and maximum ECE; reproducibility from a manifest; encoder hash recorded. |
| What it cannot show | Anything about multi-observation integration, dependence, abstention, or belief revision. |
| Dependency on this amendment | None. Stage 0A may proceed under separate authorization. |

**Guard.** Stage 0A must not be reported as evidence that PCI works. It is an encoder/head baseline. A strong Stage 0A result makes the later integrator experiment possible; it does not make the integrator hypothesis true.

### G.2 Stage 1 — multi-observation integration pilot

**Question.** Does a set integrator outperform the baseline family on actual multi-observation prediction?

**Data.** Real multi-view data with object-level separation. CO3D is the primary candidate because it provides multiple views of the same object and permits object-level splits. If a real inspection dataset is available, it is preferred for the defect task. Unreal is a synthetic stress test only. OpenOOD is a secondary OOD stress test only.

**Separation requirements.**

- Split by object instance, not by frame. No object appears in both train and test.
- Split by scene/sequence where applicable.
- Report the split hash and the number of objects, views, and classes per split.
- If CO3D categories are mapped to PCI causes, publish the mapping and validate it; do not silently relabel.

**Conditions.**

| Condition | Purpose |
|---|---|
| \(n=1,2,4,8\) views | Accuracy vs evidence; integration curve. |
| Permuted view order | Invariance. |
| Exact duplicates | Saturation. |
| Near-duplicates (JPEG, crop, small shift) | Robustness. |
| Contradictory pairs | Revision and abstention. |
| Prior shift | Calibration under changed class balance. |
| OOD scenes | Secondary stress test. |

### G.3 Arms

| Arm | Description |
|---|---|
| A0 | Stage 0A single-view head, no integration. |
| A1 | Temperature-scaled mean pooling (B0). |
| A2 | Logistic regression on set-level features (B1). |
| A3 | Integrator without dependence descriptors (B2). |
| A4 | Full integrator with dependence descriptors (H1). |
| A5 | Full integrator + conformal abstention (H4). |
| A6 | Mean-pooling-constrained integrator (B3). |
| A7 | Oracle (reported separately). |

### G.4 Metrics

| Construct | Metrics | What it establishes |
|---|---|---|
| Classification accuracy | Top-1, balanced accuracy, per-class recall, Wilson intervals. | Whether the prediction is correct. |
| Calibration | ECE (equal-width and equal-frequency), Brier, NLL, reliability diagram, bootstrap intervals. | Whether confidence matches frequency. |
| Selective risk | Risk–coverage curve, AURC, selective risk at 80% coverage, coverage at fixed risk. | Whether abstention is useful. |
| Evidence integration | Accuracy/AURC vs \(n\); slope; redundancy gain; duplicate saturation; permutation invariance. | Whether additional observations help appropriately. |
| Belief revision | Trajectory Brier, justified-revision rate, confidence movement, contradiction recovery, abstention after contradiction. | Whether the system revises rather than averages. |
| Dependence | Held-out residual correlation prediction; VIF; descriptor ablation. | Whether “dependence-aware” is earned. |
| Provenance | Hash verification; reproduction from evidence ids. | Whether the result is auditable. |

**Metric separation.** Accuracy is not calibration. Calibration is not selective risk. Selective risk is not integration. Integration is not revision. No single metric may stand in for another in the acceptance decision.

### G.5 Sample size

The primary endpoint is a paired difference in selective risk/AURC. Derive \(N\) from a minimum meaningful effect, not from convenience.

**Paired accuracy difference (McNemar).** For a two-sided test at significance \(\alpha\), power \(1-\beta\), expected discordant proportion \(\pi_d\), and minimum difference \(\Delta\):

\[
N \approx \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,\pi_d}{\Delta^2}.
\]

With \(\alpha=0.05\), power \(0.80\), \(\pi_d=0.25\), and \(\Delta=0.05\):

\[
N \approx \frac{(1.96+0.84)^2(0.25)}{0.05^2}\approx 314.
\]

Round up to **360 paired evaluation sets per arm** to allow for exclusions. For \(\Delta=0.03\), the same assumptions give \(N\approx 872\), rounded to **1,000**.

**Calibration.** For an ECE difference of 0.02–0.03, a bootstrap simulation at preregistration must determine \(N\); as a working minimum, **2,000 held-out predictions** are required for a stable ECE estimate. If the dataset cannot supply 2,000, the calibration claim is downgraded to “observed” and no superiority claim is made.

**Multi-view.** Each evaluation set is a scene/object with \(n\) views. The unit of pairing is the scene, not the view. At least 360 scenes per arm are required for the primary endpoint; 1,000 scenes are preferred for calibration and subgroup analysis. Three training seeds are required; report per-seed and pooled results.

**Power honesty.** These calculations assume the discordant proportion is known. It is not. The preregistration must state the assumed \(\pi_d\), and the final report must report the realized \(\pi_d\) and the achieved power. If the realized discordance is lower, the study is underpowered and the result is “not established,” not “no effect.”

### G.6 Acceptance criteria

**Stage 0A (separate).**

1. Accuracy at or above the pre-registered floor on the held-out test split.
2. ECE at or below the pre-registered ceiling; Brier/NLL reported.
3. No train/test leakage; split hash and encoder hash recorded.
4. Reproducible from a single command and manifest.

**Stage 1 (integrator).**

1. All invariant tests T1–T12 pass.
2. H1: lower selective risk/AURC than the best baseline, 95% CI excluding zero.
3. H2: attribution test passes; the mean-pooling-constrained integrator loses a pre-registered fraction of the advantage.
4. H3: calibration non-inferiority passes.
5. H4: abstention reduces selective risk versus confidence-only rejection, or is reported as not established.
6. No threshold tuned on test; no result reported without intervals.
7. All episode artifacts saved, including baselines and failed runs.

**Falsification.**

- H1 fails: the integrator does not beat the best baseline. Report as a negative result.
- H1 passes, H2 fails: the gain is capacity, not dependence. Do not use “dependence-aware.”
- H3 fails: the gain is overconfidence. Reject the claim.
- H4 fails: abstention is not useful under the stated cost/coverage. Report the risk–coverage curve without an abstention claim.
- Invariant failure: no integrator result is valid until fixed.

---

## H. Updated implementation and training roadmap

Nothing in this roadmap is authorized by this document. Each stage requires separate approval.

| Stage | Work | Gate | Compute | Deliverable |
|---|---|---|---|---|
| P0 | Documentation freeze: this amendment, glossary, output schema, invariant spec. | Review approval. | None. | This file + review record. |
| P1 | Stage 0A: frozen DINOv2-S/B linear probe. | Stage 0A acceptance in §G.6. | Separate authorization. | Encoder/head baseline report with hashes and intervals. |
| P2 | Invariant harness and baselines B0–B5 on synthetic sets; no integrator training. | T1–T12 pass; negative control fails as expected. | Small. | Harness report; baseline numbers. |
| P3 | Integrator pilot on synthetic multi-view sets with known dependence. | H1/H2 pilot criteria; preregistration frozen. | Moderate. | Pilot report; decision on real-data stage. |
| P4 | Real multi-view integration on CO3D or inspection data, object-level splits. | Full acceptance in §G.6. | Substantial. | Preregistered evaluation report; all artifacts. |
| P5 | Decision: adopt, revise, or reject the integrator. | Independent review. | None. | Decision record. |

**Sequencing rule.** P1 may run in parallel with P0/P2. P3 may not start before P2 passes. P4 may not start before P3 reports. No stage may use a test set more than once.

**Implementation prohibitions.**

- Do not merge RHAN-Bi-v1 implementation into PCI.
- Do not reuse prefix weights, `n_eff` as an inference correction, or the heuristic pair term as a likelihood.
- Do not claim tamper-evidence without authentication.
- Do not tune thresholds on dev/test.
- Do not report a result without the invariant harness and artifact manifest.

---

## I. Remaining unresolved scientific questions

| # | Question | Why it remains open | What would close it |
|---|---|---|---|
| Q1 | Are frozen DINOv2-S/B features sufficient for the PCI cause set? | Stage 0A has not run. | Stage 0A accuracy/calibration with intervals. |
| Q2 | Is the cause set exhaustive and mutually exclusive? | Real scenes can be multi-label or open-set; PCI v1.0 used a closed softmax. | A multi-label/open-set pilot; per-cause confusion analysis. |
| Q3 | Can pairwise dependence be estimated reliably at small \(n\)? | Residual correlation estimates are noisy; \(n=2\)–8 gives few pairs. | Simulation study of \(\hat\rho_{ij}\) bias/variance; held-out prediction. |
| Q4 | Does a learned dependence descriptor capture dependence or dataset shortcuts? | The descriptor can correlate with set size or scene identity. | Ablation with set-size-only and scene-id-only controls. |
| Q5 | Is the integrator’s advantage stable across acquisition processes? | Training and deployment acquisition may differ. | Multi-domain evaluation; acquisition-shift stress test. |
| Q6 | Does calibration transfer from single observations to sets? | Temperature fitted on singles may not calibrate the set posterior. | Set-level calibration on held-out sets; compare to refitted temperature. |
| Q7 | Is prior shift identifiable without target labels? | Prior estimation from unlabeled data is possible only under assumptions. | Sensitivity analysis; declared prior; label-shift experiment. |
| Q8 | Does conformal abstention retain its guarantee under distribution shift? | Exchangeability is required. | Shift-aware conformal or a declared limitation. |
| Q9 | What is the correct cost of abstention? | No cost matrix is specified. | Decision-theoretic analysis with the project owner. |
| Q10 | What counts as a “justified” belief revision? | Revision quality is not defined by accuracy alone. | Pre-registered trajectory metric; human or rule-based justification on a sample. |
| Q11 | Is the real multi-view data separation adequate? | Object-level splits must be verified, not assumed. | Split audit; duplicate/near-duplicate scan across splits. |
| Q12 | Can the integrator be audited? | Neural set models are less transparent than a ledger. | Provenance at the observation level; attribution tests; reproducible artifacts. |
| Q13 | Is the minimum effect size scientifically meaningful? | \(\Delta=0.05\) selective risk is a placeholder. | Project-owner decision before preregistration. |
| Q14 | Does Stage 0A’s encoder choice constrain the integrator? | A frozen encoder may limit the achievable set-level gain. | Compare frozen vs authorized fine-tuned encoder in a later stage. |

---

## Appendix — Glossary changes

| Term | v1.0 usage | v1.1 usage |
|---|---|---|
| Evidence vector | Calibrated LLR vector. | Calibrated posterior log-odds score. |
| LLR | Used for discriminative outputs. | Reserved for the generative branch with a normalized likelihood and explicit prior. |
| Dependence descriptor | Scalar/vector representing dependence. | Pairwise feature set, not a certificate; claim contingent on validation. |
| Insufficient evidence | Output state. | Decision-theoretic abstention under a risk or cost constraint. |
| Belief revision | Final label change. | Trajectory change with a quality metric; not identical to recovery. |
| `n_eff` | Effective sample size correction. | Descriptive only; never used in inference. |
| Tamper-evident | Claimed for the ledger. | Not claimed without authentication. |

---

**End of Amendment 1 (v1.1).** The parent dossier remains the authoritative v1.0 record. This amendment is the authoritative record for the seven findings above and for the mathematical, architectural, and experimental changes it specifies.

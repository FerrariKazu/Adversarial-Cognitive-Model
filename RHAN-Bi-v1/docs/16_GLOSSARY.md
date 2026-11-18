# 16 — Glossary

Terms are used with exactly these meanings throughout the dossier and the code.

| Term | Meaning |
|---|---|
| **action** | a request to the environment: change light, change view, or zoom. Not a physical manipulation. |
| **abstention** | a decision to commit to no cause. Counted as an error in the primary endpoint and reported separately as a rate. |
| **cause** | the world state that produced the appearance: `DEFECT`, `REFLECTION`, `SHADOW` or `NONE`. The belief is over causes. |
| **commit** | the system's final cause label for an episode, with its posterior, debt and observation count. |
| **conflict** | a record whose evidence opposes the current leader (LLR ≤ −log 3 against it, or pair evidence inconsistent with it). |
| **confidence-driven** | a baseline policy that selects the action expected to maximize the posterior of the current leader — it seeks confirmation. |
| **debt** | the structured list of unresolved weaknesses attached to a hypothesis (`OCCLUDED`, `GLARE`, `CONFLICT`, …), capped at 1. |
| **dependence (ρ)** | the modelled redundancy between two observations, from the action pair. Reduces a record's weight; never increases it. |
| **disambiguating observation** | one whose likelihood ratio between the leading and second hypothesis is ≥ log 3 in magnitude. |
| **EIG** | expected information gain: expected mutual information between the next observation and the cause. A baseline policy. |
| **evidence model** | the fitted diagonal Gaussian naive Bayes mapping features to per-cause likelihoods and LLRs. |
| **evidence record** | one immutable, content-addressed observation with its features, quality, LLRs, pair evidence, dependence and provenance. |
| **falsification-directed** | selecting the action expected to displace the current leader if the leader is wrong; the programme's candidate policy. |
| **forward model** | the fitted per-cause, per-action feature distribution the planner uses to predict outcomes. |
| **feature version** | string identifying the feature definitions; records from another version are rejected rather than mis-scored. |
| **ledger** | the append-only store of evidence records plus images (by hash) and the dependence graph. |
| **LLR** | log-likelihood ratio of a cause against `NONE`, for one record. |
| **MAP** | the hypothesis with the largest posterior. Not a claim of truth. |
| **n_eff** | effective sample size after dependence: `n / (1 + (n−1)ρ̄)`. |
| **oracle policy** | a reported-only upper bound that selects actions using ground truth; not deployable. |
| **pair evidence** | `persistence` and `bright_shift`, computed between two observations at the same view under different light. |
| **persistence** | overlap of thresholded dark-anomaly maps across a light change. High for a fixed defect, low for a reflection. |
| **posterior** | the normalised distribution over the four causes, recomputed from the full ledger at each step. |
| **quality** | the image-side index q ∈ [0,1] from blur, contrast, exposure, glare, occlusion and SNR. Predicts error; is not a probability. |
| **reliability** | a fitted probability that a record's evidence is correct, used to scale its weight within [0.25, 1]. |
| **revision** | a change of the MAP cause, recorded with the record id that caused it. |
| **revision accuracy** | among episodes whose initial MAP was wrong, the fraction whose final MAP is correct. |
| **selective risk** | error rate among committed episodes at a given coverage. |
| **coverage** | fraction of episodes with a commit. Always reported beside risk. |
| **VIF** | variance inflation factor of the feature set on the calibration split; a diagnostic, not a correction. |
| **claim level** | demonstrated / feasibility / exploratory — required on every reported number. |
| **scene** | one synthetic plate with a hidden cause and a ground-truth record; the unit of analysis. |
| **episode** | one scene observed by one policy for up to B observations. |
| **simulator version** | `inspection-v1.0`; a change to the renderer or action space bumps it and invalidates comparisons. |

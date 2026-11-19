# 6 — Unresolved mathematical questions

These are questions that cannot be settled by reading the code. They are listed as research questions,
not defects. Each includes why it is open and what evidence would close it.

## Q1. Is the logistic LLR a valid likelihood ratio under class imbalance?

The update uses `ℓ(c) = log P(c|z) - log P(none|z)` from a multinomial logistic trained on
calibration rows (`likelihood.py:52-84, 116-119`). For this to equal `log p(z|c)/p(z|none)` the
calibration class prior must cancel, which requires the calibration set to be balanced and the belief
prior to equal the calibration prior. The demo is balanced by construction
(`simulator/scene.py:122`). No experiment varies the prior. **Open:** does the method survive a skewed
prior, and if not, what prior correction is needed?

## Q2. What is the correct dependence model for a sequence of looks?

The implemented rule is a prefix sum of heuristic ρ values (`dependence.py:44-51`). There is no
derivation that this approximates the variance of a sum under heterogeneous correlation, and no
estimate of ρ from data. **Open:** is there a joint model (e.g. Gaussian precision, or a
state-space model over views/lights) that (a) is order-invariant, (b) saturates for duplicates, and
(c) matches the empirical correlation of feature residuals? The docs claim (c) is checked; it is not
(`docs/03:122`, `dependence.py:67-82` unused).

## Q3. What does `n_eff` measure, and is it calibrated?

`n_eff = n / (1 + (n-1) ρ̄)` is the design effect under equicorrelation (`ledger.py:89-96`). Here ρ̄ is
a constant, not an estimate, and the formula is never used in inference. **Open:** if `n_eff` is
retained, what is its sampling distribution, bias and coverage at n = 2–4? If it is dropped, what
replaces it in reporting?

## Q4. Is the pair term a likelihood, and how does it depend on its parent?

`pair_llr_matrix` centres and clips a Gaussian score over three anomaly causes
(`likelihood.py:124-137`). It is not a likelihood ratio (it cannot say "no anomaly") and its
dependence on the previous record is not represented by the action-level ρ. **Open:** can the pair
term be fitted as a likelihood ratio, and what is its conditional distribution given the parent
record? If not, it should be reported as a feature, not as evidence.

## Q5. What is the right posterior over causes for a real image?

The belief state is a softmax over four causes with a uniform prior. A real encoder produces
high-dimensional features; the cause set {defect, reflection, shadow, none} may not be exhaustive or
mutually exclusive (a scene can be both defective and shadowed). **Open:** is the four-way softmax
the right state representation, or should the state be multi-label / open-set?

## Q6. Does temperature scaling transfer to multi-look episodes?

Gate C fits temperature on single-look calibration posteriors (`stages.py:182-183`) and reports ECE
0.072. The sequential posterior used by D and E is never temperature-scaled. **Open:** what is the
calibration of the *sequential* posterior, and does the single-look temperature transfer? If not, a
multi-look calibration procedure is needed.

## Q7. What is the revision metric, and what would count as a good revision?

The current metric is conditional recovery rate (`stages.py:299-312`). It does not distinguish a
justified revision from a lucky one, and it ignores confidence movement. **Open:** define a revision
quality metric (e.g. change in Brier score, or a proper scoring rule on the trajectory) and decide
what threshold would constitute evidence of genuine revision.

## Q8. Does the planner's objective match the decision problem?

Falsification, EIG and confidence are three different functionals (`planner.py:114-171`); the demo
shows they choose different actions with the same error. **Open:** which objective is appropriate for
defect detection under cost, and is there a principled way to choose? The current forward model
averages over action geometry (§2.9), so the comparison is not even over the real action space.

## Q9. What is the minimum detectable effect, and what N is needed?

At N = 120 the minimum detectable accuracy difference is ≈ 5.7% at 80% power (`audit/03` §3.4).
**Open:** what effect size is practically meaningful for a defect-detection system, and what N
follows? Until that is fixed, "no significant difference" cannot be interpreted.

## Q10. How does the system behave under distribution shift?

No experiment varies the renderer noise, the class prior, the number of actions, or the cost model.
The DINOv2 path is untested. **Open:** what is the robustness envelope, and which components (quality
score, ρ constants, temperature) need to be re-estimated per domain?

## Q11. Is the conformal coverage claim derivable?

Gate C reports coverage 0.875 at a threshold selected on raw probabilities
(`stages.py:190-203`). Conformal prediction requires exchangeability and a threshold on the same
scale as the test score. **Open:** can a coverage guarantee be derived for this sequential,
temperature-mismatched procedure? If not, the coverage number should be relabelled as an observed
rate.

## Q12. Are the four causes identifiable from real images at all?

The simulator encodes the discrimination in the renderer. Whether DINOv2 features (or any real
features) carry the defect/reflection/shadow distinction at the resolution and noise levels of real
inspection images is an empirical question that no experiment in this repository addresses.
**Open:** what is the per-cause accuracy of a simple classifier on real images, and does it exceed
the ambiguity band [0.55, 0.80]?

Each of Q1–Q12 is a research question. None can be closed by code changes alone; each needs an
experiment with a pre-registered protocol and a held-out test set.

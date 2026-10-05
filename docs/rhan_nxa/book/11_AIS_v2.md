# Chapter 11 — Active Information Sampling (AIS-v2) Gaze Policy and Gaze State A_t

> *Level 2–3 reading. Active visual inquiry: discrete candidate generation, analytical entropy reduction scoring, and the center-bias failure gate.*

---

## 1. In One Sentence

AIS-v2 (Active Information Seeking, version 2) is the trained gaze policy that selects each subsequent fixation location by scoring $K$ candidate locations against the expected reduction in Dirichlet uncertainty, using the shared predictor and evidential head as its exclusive scoring mechanism—no auxiliary scoring head is permitted.

---

## 2. Intuition

After each glimpse, the system must decide where to look next. The AIS-v2 policy answers this question by asking: *"For each of $K$ candidate locations, how much would my uncertainty decrease if I looked there?"* The candidate with the highest expected entropy reduction is selected.

This is implemented without introducing duplicate uncertainty or prediction modules: the **same** `ConcreteGlimpseFeaturePredictor` predicts what tokens would be observed at each candidate, and the **same** `EvidentialHead` maps those predicted tokens to a predicted Dirichlet distribution. The entropy of the predicted Dirichlet is the expected uncertainty after looking there; the score is the difference from the current entropy.

---

## 3. AIS-v2 Active Gaze Architecture

![Figure 5. AIS-v2 Active Information Sampling: Perception as Active Investigation. Candidates are sampled across the visual field avoiding past fixation history A_t. The shared predictor projects anticipated patch tokens for each candidate, and Dirichlet entropy reduction ΔH is evaluated. Candidate 3 (diagnostic boundary saccade) achieves the highest information gain and is selected.](figures/ais_v2.svg)

---

## 4. Mathematical Formulation

### Candidate Scoring ($t \ge 1$)

$$r_k = H(U_t) - H\!\left(\text{Dirichlet}\!\left(\hat{e}(B_t, c_k)\right)\right)$$

where $c_k$ is candidate $k$'s location, $\hat{e}(B_t, c_k)$ is the predicted Dirichlet evidence at $c_k$ via the shared predictor, and $H(\cdot)$ is the Dirichlet differential entropy (digamma formulation, `DirichletParams.entropy()`).

### Candidate Scoring ($t = 0$, Fallback)

$$r_k^{(0)} = \text{SaliencyMap}(|\text{observed tokens}|^2)_{@c_k}$$

No predictor call or Dirichlet construction exists in this initial branch. The `reduction` field of `CandidateScores` is `None`—structurally absent, not zero.

### Action Selection

- **Soft Selection (Training)**:
  $$w_k = \text{Gumbel-Softmax}\!\left(r_k \cdot \lambda, \tau\right), \quad a_{t+1} = \sum_{k=1}^K w_k \cdot c_k$$
- **Hard Selection (Evaluation)**:
  $$a_{t+1} = c_{\arg\max_k r_k}$$

---

## 5. The Parameter Reuse Boundary (Inviolable)

`AISv2GazePolicy` has one learned parameter: `logit_scale`, a scalar temperature. No per-candidate scoring head, no second predictor, and no second uncertainty module are permitted. This is enforced by:
- `__init__` type-checking: `predictor` must be `ConcreteGlimpseFeaturePredictor`; `evidential_head` must be `EvidentialHead`.
- Test: `tests/test_ais_v2_gaze_policy.py::test_shares_agent_e_predictor`.

---

## 6. Candidate Generation Protocol

`HeuristicCandidateSampler` generates $K \in \{4, \dots, 8\}$ candidates per glimpse:

| Property | Gen-0 (STL-10) | RHAN-NXA Gen-1 | Scientific Rationale |
|:---|:---|:---|:---|
| Anchor | argmax of pixel reconstruction error | argmax of **per-token prediction error** $\|g_t^{\text{obs}} - \hat{g}_t\|^2$ | Gen-1 operates in latent token space |
| Candidate 0 | Anchor (stay put) | **Unchanged** | Allows temporal persistence |
| Candidates $1 \dots K-1$ | Anchor + Gaussian noise ($\sigma = 0.12$) | **Unchanged** | Local exploration around surprise |
| Coordinate Clamp | $[-0.9, +0.9]$ | **Unchanged** | Prevents boundary clipping |
| Candidate Count $K$ | 4 to 8 (default 4) | **Unchanged** | Bound on evaluation compute |

---

## 7. Gaze State $A_t$ (Coordinate Record)

`noesis_vision.gaze.gaze_state.GazeState` is the canonical $A_t$ carrier:

- **Detached by construction**: Every `record()` call stores `location.detach().clone()`. The autograd graph cannot leak into the history record.
- **Capacity enforcement**: History raises loudly when `len(history) >= T`. Silent truncation was a known failure mode in Gen-0.
- **Guard logic**: `current` property raises `IndexError` at $t = 0$. The initial fixation is an explicit system input; no default is fabricated.

---

## 8. The Center-Bias Failure Condition

A gaze policy that consistently returns coordinates near the image center is a **FAILED** result, not a passing result with an asterisk. This is explicitly enforced by pre-registered verification gates: if AIS-v2 produces center-biased fixations across test sets, the run is rejected as dysfunctional.

---

## 9. Scientific Status and Epistemic Distinctions

- **AIS-v2 mechanism validity**: **REQUIRED** (Targeted empirical evidence: $r \approx 0.706$, $n = 512$).
- **Isolated system-level robustness contribution**: **UNKNOWN** (Gen-0 runs D2 and D3 were confounded by accidental `--enable-sbr` flag; currently under unconfounded DAG re-evaluation).
- **Center-bias result**: **TERMINAL FAILURE CONDITION**.
- **Candidate sampler heuristic**: **LOCKED**.

---

## 10. Related Components and System Cross-References

- Gaze policy implementation: [noesis_vision/gaze/ais_v2_policy.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/ais_v2_policy.py)
- Candidate generator: [noesis_vision/gaze/candidate_sampler.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/candidate_sampler.py)
- Gaze state container: [noesis_vision/gaze/gaze_state.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/gaze_state.py)
- Figure Reference: [ais_v2.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/ais_v2.svg).

"""
AIS-v2 gaze policy — Agent F. Selection over candidates, one predictor.
================================================================================

THE REUSE BOUNDARY (Agent F contract, restated so no reader can miss it):
candidate SCORING goes through Agent E's ConcreteGlimpseFeaturePredictor
and Agent D's EvidentialHead — both INJECTED at construction (not
constructed here, not reimplemented here, not subclassed here). This
module contains no second predictor and no second scoring head; the
identity is asserted by tests/test_ais_v2_gaze_policy.py::

    test_shares_agent_e_predictor.

Scoring, t >= 1 (schema: candidate_scoring = "dirichlet_entropy_reduction"):
for each candidate c_k the shared predictor predicts the token features
the encoder would produce there (predict_features — NOT detached, per its
contract); the predicted features go through the SHARED EvidentialHead to
a predicted Dirichlet; the per-candidate score is the uncertainty-
reduction proxy

    r_k = H(U_t) - H(Dirichlet(predicted evidence at c_k))

with H the canonical DirichletParams.entropy() (digamma form, Agent D).
r_k carries gradient into the predictor AND the evidential head — the
gaze choice trains the shared stack. H(U_t) is common across candidates,
so argmax r == argmin predicted entropy; the reduction form is kept
because it is the quantity Part 1.E names.

Scoring, t = 0 (LOCKED, AIS_T0_SCORING): there is no U_0-conditioned
prediction (E_0 := 0). The t=0 branch is STRUCTURALLY different code:
no predictor call, no Dirichlet construction, no entropy — candidates
are scored by the generation heuristic's own saliency surface
(candidate_sampler.saliency_at). The reduction term is `None` in the
returned CandidateScores — absent, not present-and-zero.

Selection (Part 1.E): soft during training — Gumbel-softmax over the
scores (mode "soft" for fully soft weights, "straight_through" for a
one-hot forward with soft backward); hard argmax at inference, computed
under no_grad with NO Gumbel noise (deterministic). The policy's ONLY
learned state is `logit_scale`, a scalar temperature on the scores:
any per-candidate head would violate the no-second-scoring-head rule;
a monotone per-sample scale does not change which candidate wins at
init and gives the policy's own optimizer group ("gaze_policy") real
parameters and a real gradient path.

Bookkeeping: selected locations are DETACHED when recorded into the
canonical GazeState (gaze_state.py) — A_t is a coordinate record
(Part 1.A); differentiability lives in the selection, here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from noesis_vision.core.schema import AIS_T0_SCORING
from noesis_vision.gaze.candidate_sampler import (
    MAX_ABS_GAZE_DEFAULT,
    NUM_CANDIDATES_RANGE,
    HeuristicCandidateSampler,
    _validate_num_candidates,
    saliency_at,
    token_error_map,
)
from noesis_vision.gaze.gaze_state import GazeState
from noesis_vision.predictive_coding.glimpse_predictor import (
    ConcreteGlimpseFeaturePredictor,
)
from noesis_vision.uncertainty.evidential_head import (
    DirichletParams,
    EvidentialHead,
)

__all__ = ["AISv2GazePolicy", "CandidateScores", "SelectionResult"]


@dataclass
class CandidateScores:
    """Per-candidate scores with the t=0 boundary made inspectable.

    kind == "uncertainty_reduction": `reduction` (B, K) is H(U_t) -
    H(predicted) and carries gradient; `saliency` is None.
    kind == "heuristic_saliency" (t = 0): `saliency` (B, K) is the
    detached heuristic surface; `reduction` IS None — the uncertainty-
    reduction term is structurally absent at t = 0 (LOCKED), never a
    zero tensor.
    """

    scores: torch.Tensor                 # (B, K) — what selection consumes
    kind: str
    reduction: Optional[torch.Tensor]    # (B, K) or None
    saliency: Optional[torch.Tensor]     # (B, K) or None
    t: int


@dataclass
class SelectionResult:
    """What one policy step produces (all coordinates detached)."""

    selected: torch.Tensor               # (B, 2) hard or soft-weighted
    candidates: torch.Tensor             # (B, K, 2) detached
    weights: Optional[torch.Tensor]      # (B, K) soft weights or None (hard)
    chosen_idx: Optional[torch.Tensor]   # (B,) hard argmax or None (soft)
    scores: CandidateScores


class AISv2GazePolicy(nn.Module):
    """K-candidate expected-uncertainty-reduction gaze selection.

    Args:
        predictor: Agent E's ConcreteGlimpseFeaturePredictor — the ONE
            shared predictor (injected; identity-checked in tests).
        evidential_head: Agent D's EvidentialHead — the ONE uncertainty
            representation (injected; maps predicted token features
            (B, N, D_feat) to a predicted Dirichlet, pooling by mean
            inside the head).
        num_candidates: K in the LOCKED range 4-8 (Part 1.E).
        candidate_sigma / max_abs_gaze: the Gen-0 validated sampling
            constants (see candidate_sampler).
        selection_mode: "straight_through" (default) or "soft" — the
            TRAINING-time behavior; inference is always hard argmax.
        tau: Gumbel-softmax temperature.
    """

    def __init__(self, predictor: ConcreteGlimpseFeaturePredictor,
                 evidential_head: EvidentialHead,
                 num_candidates: int = 4,
                 candidate_sigma: float = HeuristicCandidateSampler(
                     num_candidates=4).candidate_sigma,
                 max_abs_gaze: float = MAX_ABS_GAZE_DEFAULT,
                 selection_mode: str = "straight_through",
                 tau: float = 1.0):
        super().__init__()
        _validate_num_candidates(num_candidates)
        if selection_mode not in ("soft", "straight_through"):
            raise ValueError(
                f"selection_mode must be 'soft' or 'straight_through' "
                f"(training-time; inference is always hard); got "
                f"{selection_mode!r}")
        # THE reuse boundary: injected shared modules, stored as-is.
        if not isinstance(predictor, ConcreteGlimpseFeaturePredictor):
            raise TypeError(
                "predictor must be Agent E's ConcreteGlimpseFeaturePredictor "
                "(no second predictor may exist — Agent F contract)")
        if not isinstance(evidential_head, EvidentialHead):
            raise TypeError(
                "evidential_head must be Agent D's EvidentialHead (the ONE "
                "uncertainty representation — no second mechanism)")
        self.predictor = predictor
        self.evidential_head = evidential_head
        self.num_candidates = int(num_candidates)
        self.max_abs_gaze = float(max_abs_gaze)
        self.selection_mode = selection_mode
        self.tau = float(tau)

        # The policy's ONLY learned state: a scalar logit temperature.
        # Initialized at 1.0 — argmax over scores is unchanged at init;
        # a second scoring head is LOCKED OUT (Part 1.E).
        self.logit_scale = nn.Parameter(torch.ones(()))

        self.sampler = HeuristicCandidateSampler(
            num_candidates=self.num_candidates,
            candidate_sigma=candidate_sigma,
            max_abs_gaze=max_abs_gaze)

        # Diagnostics-only (never part of the graph decisions):
        self.last_scores: Optional[CandidateScores] = None
        self.last_selection: Optional[SelectionResult] = None

    # ── scoring ──────────────────────────────────────────────────────────────
    def score_candidates(self, belief, observed_tokens: torch.Tensor,
                         predicted_tokens: Optional[torch.Tensor],
                         candidates: torch.Tensor,
                         current_gaze: Optional[torch.Tensor] = None
                         ) -> CandidateScores:
        """Score (B, K, 2) candidates -> CandidateScores.

        predicted_tokens given (t >= 1): uncertainty-reduction scoring
        through the SHARED predictor + SHARED evidential head, gradient
        intact. predicted_tokens None (t = 0): the LOCKED heuristic-
        saliency fallback — computed under no_grad; the reduction term
        does not exist in this branch. `current_gaze` is required for
        the t=0 branch: the saliency map is expressed in CURRENT-GAZE
        coordinates (the fovea frame the tokens were encoded at), never
        in anchor-relative coordinates.
        """
        K = candidates.shape[1]
        if K != self.num_candidates:
            raise ValueError(
                f"candidates K={K} != policy num_candidates="
                f"{self.num_candidates}")
        if predicted_tokens is None:
            # ── t = 0 (LOCKED): heuristic-saliency only. Structurally no
            # predictor call, no Dirichlet, no entropy in this branch.
            if current_gaze is None:
                raise ValueError(
                    "t=0 scoring needs current_gaze: the saliency map is "
                    "expressed in the current fovea's coordinates")
            err_map = token_error_map(observed_tokens, None,
                                      self.sampler.grid_size)
            sal = saliency_at(err_map, current_gaze, candidates,
                              self.sampler.fovea_size, self.sampler.img_size)
            return CandidateScores(scores=sal.detach(),
                                   kind="heuristic_saliency",
                                   reduction=None, saliency=sal.detach(), t=0)

        # ── t >= 1: predicted uncertainty reduction via the shared stack.
        B = candidates.shape[0]
        reductions = []
        for k in range(K):
            pred_k = self.predictor.predict_features(
                belief, candidates[:, k, :])              # (B, N, D), attached
            dp_k: DirichletParams = self.evidential_head(pred_k)
            reductions.append(dp_k.entropy())             # (B,)
        h_pred = torch.stack(reductions, dim=1)           # (B, K), attached
        h_current = DirichletParams(evidence=belief.evidence).entropy() \
            .unsqueeze(1)                                 # (B, 1), attached
        reduction = (h_current - h_pred) / self.logit_scale.clamp_min(1e-4)
        return CandidateScores(scores=reduction,
                               kind="uncertainty_reduction",
                               reduction=reduction, saliency=None, t=-1)

    # ── selection ────────────────────────────────────────────────────────────
    def select(self, candidates: torch.Tensor,
               scores: CandidateScores,
               training: bool) -> SelectionResult:
        """Soft (training) / hard (inference) selection over candidates."""
        cand = candidates.detach()
        B = cand.shape[0]
        if not training:
            with torch.no_grad():
                idx = scores.scores.argmax(dim=1)         # (B,) hard argmax
                selected = cand.gather(
                    1, idx.reshape(B, 1, 1).expand(B, 1, 2)).squeeze(1)
            return SelectionResult(selected=selected, candidates=cand,
                                   weights=None, chosen_idx=idx, scores=scores)
        # Training: soft path — gradient flows through the weights into
        # the scores (predictor + evidential head) and into logit_scale.
        logits = scores.scores * self.logit_scale
        weights = F.gumbel_softmax(logits, tau=self.tau,
                                   hard=(self.selection_mode
                                         == "straight_through"))
        selected = (weights.unsqueeze(-1) * cand).sum(dim=1)   # (B, 2)
        return SelectionResult(selected=selected, candidates=cand,
                               weights=weights, chosen_idx=None,
                               scores=scores)

    # ── one step ─────────────────────────────────────────────────────────────
    def select_next_location(self, belief, observed_tokens: torch.Tensor,
                             current_gaze: torch.Tensor,
                             predicted_tokens: Optional[torch.Tensor] = None,
                             training: bool = True,
                             generator: Optional[torch.Generator] = None
                             ) -> SelectionResult:
        """Generate, score, and select — one AIS-v2 step.

        Args:
            belief: current belief (z_t, U_t condition the prediction).
            observed_tokens: (B, N, D) encoder output at the CURRENT gaze
                (detached by the caller — the observed path is a target).
            current_gaze: (B, 2) current fixation.
            predicted_tokens: (B, N, D) the shared predictor's prediction
                AT the current gaze — None only at t = 0 (no U_0-conditioned
                prediction exists; LOCKED fallback applies).
            training: soft vs hard selection (the Part 1.E phase rule).
        """
        B = current_gaze.shape[0]
        err_map = token_error_map(observed_tokens, predicted_tokens,
                                  self.sampler.grid_size)
        candidates, _ = self.sampler.forward(err_map, current_gaze.detach(),
                                             generator=generator)
        cs = self.score_candidates(belief, observed_tokens,
                                   predicted_tokens, candidates,
                                   current_gaze=current_gaze)
        result = self.select(candidates, cs, training)
        self.last_scores, self.last_selection = cs, result
        return result

    def record(self, gaze_state: GazeState,
               selection: SelectionResult) -> GazeState:
        """Append the selected location to the canonical GazeState.

        Coordinates are detached (A_t is a record, Part 1.A); the
        GazeState cap (T) raises loudly at capacity rather than silently
        truncating.
        """
        return gaze_state.record(selection.selected.detach())

    # ── Agent A registry integration (own optimizer group) ──────────────────
    def register_optimizer_group(self, registry, lr_multiplier: float = 1.0,
                                 clip_norm: float = 1.0) -> None:
        """Register the policy's own parameters under 'gaze_policy'.

        Same standing rule as every prior agent (the shared predictor and
        evidential head belong to THEIR own groups — never here).
        """
        registry.register("gaze_policy", list(self.parameters()),
                          lr_multiplier=lr_multiplier, clip_norm=clip_norm)

    def forward(self, belief, observed_tokens: torch.Tensor,
                current_gaze: torch.Tensor,
                predicted_tokens: Optional[torch.Tensor] = None,
                training: bool = True,
                generator: Optional[torch.Generator] = None
                ) -> SelectionResult:
        """Module convention — routes through select_next_location."""
        return self.select_next_location(belief, observed_tokens,
                                         current_gaze, predicted_tokens,
                                         training, generator)

    def __repr__(self) -> str:
        return (f"AISv2GazePolicy(K={self.num_candidates}, "
                f"mode={self.selection_mode!r}, tau={self.tau}, "
                f"predictor={type(self.predictor).__name__} (shared, "
                f"injected), head={type(self.evidential_head).__name__} "
                f"(shared, injected)) — t=0: {AIS_T0_SCORING[:48]}...")

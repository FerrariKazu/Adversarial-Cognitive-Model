"""
Predictive-coding interface stubs — Agent 0 output (stub-only).

THE SHARED PREDICTOR INTERFACE — the single most load-bearing artifact
Agent 0 produces. Part 1.B unified the predictor (one module, two
consumers: the realized-glimpse error signal for the belief update, and
the candidate-scoring signal for gaze selection). Part 4 Adjustment 1
makes that unification organizational: Agents E and F are NOT merged,
but this interface defines the predictor's exact call signature ONCE
and both consume it identically. **Neither may implement its own copy.**

Why this exists (Part 0): Generation 0 lost real time to interface
drift — multiple eval scripts disagreeing on conventions, a runner
hardcoding a flag that silently confounded two "isolated" experiments.
One shared interface artifact — not good intentions — is what prevents
predictor drift between E and F.

Engineering constraints baked into this interface (Part 1.B / 1.E):
  - Compute: K candidate evaluations use the CHEAP glimpse-feature
    predictor, NOT K full backbone passes. Enforced as an interface
    constraint, not left ambiguous.
  - Gradients: the predicted path is NEVER detached (the observed
    target is detached at the consumer's call site); E_t itself is
    never detached before the update.
  - t = 0 boundary (LOCKED, MASTER_PLAN Part 1.B): E_0 := 0 and there
    is no U_0-conditioned prediction — candidate scoring falls back to
    heuristic-saliency sampling at t = 0. See AIS_T0_SCORING in
    noesis_vision.core.schema.

Agent E implements these ABCs. Agent F consumes them for candidate
generation/selection/gaze bookkeeping and must not reimplement them.
"""

from __future__ import annotations

from abc import ABC
from typing import Tuple

import torch

from noesis_vision.beliefs.interfaces import BeliefState


class GlimpseFeaturePredictor(ABC):
    """The ONE predictor, two consumers — Parts 1.B/1.E, Adjustment 1.

    Given the current belief and candidate/chosen gaze location(s),
    predict what the ENCODER will produce when it actually looks there
    (Part 1.B's LOCKED DEFAULT — deliberately NOT "predict the next
    global z_t from the current global z_t").

    Stub-only: raises NotImplementedError("Agent E: ...").
    """

    def predict_features(
        self, belief: BeliefState, gaze_location: torch.Tensor
    ) -> torch.Tensor:
        """Predicted token/patch-level features at a fixation.

        Args:
            belief: current belief B_t (its z_t and U_t condition the
                prediction).
            gaze_location: (B, 2) fixation coordinates.

        Returns:
            Predicted local features in the encoder's own
            patch/token-embedding space — shape (B, N, D_feat). N and
            D_feat are set by the substrate (Agent C); Part 1.B
            deliberately leaves the token count to the substrate so
            predicted and observed features are comparable BY
            CONSTRUCTION (same encoder, no ad-hoc projection).

        Gradient contract: the returned tensor is the "predicted" path
        and must NOT be detached. The observed target is detached at
        the consumer's call site (Part 1.A gradient rules).
        """
        raise NotImplementedError(
            "Agent E: implement the shared glimpse-feature predictor "
            "per Part 1.B (Agent F consumes this same interface — do "
            "not reimplement it)"
        )

    def score_candidates(
        self, belief: BeliefState, candidate_locations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score K candidate gaze locations — AIS-v2's scoring consumer.

        Args:
            belief: current belief B_t.
            candidate_locations: (B, K, 2) candidate fixations
                (K within the LOCKED range 4-8, Part 1.E).

        Returns:
            (predicted_features, scores) where predicted_features has
            shape (B, K, N, D_feat) — the same predictor output as
            `predict_features`, one pass per candidate — and scores has
            shape (B, K): per-candidate scalar = predicted uncertainty
            reduction via U_t's Dirichlet entropy. NO separate scoring
            head (one uncertainty representation, Part 1.E).

        Compute constraint (Part 1.E, interface-enforced): K cheap
        predictor passes, NEVER K full backbone passes — candidate
        scoring must stay cheap at ImageNet scale.

        t = 0 (LOCKED, MASTER_PLAN Part 1.B): there is no
        U_0-conditioned prediction to score against — callers MUST use
        the heuristic-saliency fallback (same sampling as candidate
        generation) at t = 0 instead of this method.
        """
        raise NotImplementedError(
            "Agent E: implement candidate scoring through the SAME "
            "predictor as predict_features per Parts 1.B/1.E (Agent F "
            "consumes this — one predictor module, two consumers)"
        )


class UpdateNet(ABC):
    """The belief-update network — Part 1.B's REVISED UPDATE EQUATION.

        z_{t+1} = z_t + Pi_t * UpdateNet(z_t, E_t)

    A small learned MLP or single GRU cell — NOT a raw linear addition.
    Local glimpse-feature error and the global pooled vector are not
    assumed compatible without a learned mapping (the earlier
    `z_t + lambda*Pi*E_t` draft is superseded/REJECTED).

    Implementation note for Agent E: implement this as a
    torch.nn.Module subclassing this ABC — the interface method is
    named `forward` to match Module convention, so the implementation
    slots in without renaming.

    Stub-only: raises NotImplementedError("Agent E: ...").
    """

    def forward(self, z: torch.Tensor, prediction_error: torch.Tensor) -> torch.Tensor:
        """Map (current content, prediction error) into z's update space.

        Args:
            z: current global content z_t, shape (B, D_z).
            prediction_error: E_t from the shared predictor
                (token-feature space per Part 1.B; shape per the
                predictor's output contract). At t = 0 this is the zero
                tensor (LOCKED boundary condition — the update reduces
                to pure observation with no correction term). NEVER
                detached before reaching here.

        Returns:
            The update term, shape (B, D_z). Precision Pi_t scales this
            output at the call site (Part 1.B applies Pi_t OUTSIDE
            UpdateNet).

        Engineering contract (Part 1.B): this component is small, new,
        and independently gradient-isolated — its own optimizer group,
        its own pre-flight |dW| check before any smoke test (the
        standing Gen-0 rule, applied from day one).
        """
        raise NotImplementedError(
            "Agent E: implement UpdateNet per Part 1.B (learned mapping, "
            "own optimizer group + |dW| pre-flight)"
        )

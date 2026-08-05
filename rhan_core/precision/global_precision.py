"""
Global precision modulator — Pillar 2.

Unsupervised per-sample precision (Pi_D) computed from prediction error,
exposed for OTHER components to consume. The modulator itself does not
modulate anything: GazePolicy, the recurrence/halting loop, and the loss
function each query it separately, so every consumer's use of precision can
be isolated and tested independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rhan_core.precision.base import PrecisionModulator


class GlobalPrecisionModulator(PrecisionModulator, nn.Module):
    """
    Per-sample precision from image-space prediction error.

    Precision is UNSUPERVISED — never trained against a saturating binary
    correctness target (the diagnosed v10/v11 failure mode). The update rule
    is the frozen ImageSpacePrecision equation (Eq. III):
        dPi/dt = (error^2 - Pi) / tau, clamped to [0.2, 0.8].

    State-dict design: the model's `image_precision` module is shared BY
    PLAIN REFERENCE (`object.__setattr__`), so no parameter keys are
    duplicated. The modulator owns exactly one parameter — `gain` — which
    scales precision before it reaches consumers (initialized to 1.0 =
    exactly v12 behavior). The gradient-flow test asserts gradients reach
    this gain.

    Consumers (each isolated & testable):
      modulate_step_size(pi_d)         -> v12 step formula, gain-scaled
      modulate_halting_threshold(pi_d) -> recurrence-depth wiring
      modulate_recon_weight(w, pi_d)   -> loss-side recon weight wiring
    Attention-gating and skip-connection gating are explicitly DEFERRED.
    """

    def __init__(self, image_precision_module: nn.Module, tau: float = 0.1,
                 gain: float = 1.0):
        super().__init__()
        # Plain reference — NOT a submodule (no state-dict duplication).
        object.__setattr__(self, '_image_precision', image_precision_module)
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.tau = float(tau)

    # ── PrecisionModulator ABC ───────────────────────────────────────────────
    def compute_precision(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """(B,) precision from a (B,) error magnitude (Eq. III, init Pi=0.5).

        Standalone entry point for the ABC; the model's forward uses
        `precision_from_crops` (the exact v12 path) instead.
        """
        if prediction_error.dim() > 1:
            err = prediction_error.flatten(1).mean(dim=1)
        else:
            err = prediction_error
        prec = torch.clamp(0.5 + 0.1 * ((err ** 2 - 0.5) / self.tau),
                           0.2, 0.8)
        return prec * self.gain

    # ── Exact v12 precision path (used by the model's forward) ───────────────
    def precision_from_crops(self, actual_crop: torch.Tensor,
                             predicted_crop: torch.Tensor,
                             belief: torch.Tensor):
        """(pi_d (B,), error_mag (B,)) — identical to v12's image_precision.

        The returned pi_d is RAW (gain NOT applied) so the belief-update
        math stays byte-identical to v12; gain only enters via the explicit
        consumer modulations below.
        """
        return self._image_precision(actual_crop, predicted_crop, belief)

    # ── Consumer 1: gaze step size ───────────────────────────────────────────
    def modulate_step_size(self, pi_d: torch.Tensor) -> torch.Tensor:
        """(B,) step formula 0.20 + 0.30 * Pi_D * gain (== v12 when gain=1)."""
        return 0.20 + 0.30 * pi_d * self.gain

    # ── Consumer 2: recurrence depth (halting threshold) ─────────────────────
    def modulate_halting_threshold(self, pi_d: torch.Tensor,
                                   base_threshold: float) -> torch.Tensor:
        """(B,) effective halt threshold = base * (0.5 + Pi_D * gain).

        Higher precision -> earlier halting (shallower recurrence).
        """
        return base_threshold * (0.5 + pi_d * self.gain)

    # ── Consumer 3: reconstruction loss weight (trainer side) ────────────────
    def modulate_recon_weight(self, w_recon: float,
                              pi_d: torch.Tensor) -> torch.Tensor:
        """Scalar effective recon weight = w_recon * mean(0.5 + Pi_D * gain)."""
        return w_recon * (0.5 + pi_d * self.gain).mean()

    def __repr__(self) -> str:
        return f"GlobalPrecisionModulator(tau={self.tau}, gain={self.gain.item():.3f})"

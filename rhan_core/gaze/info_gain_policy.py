"""
Information-gain gaze policy — Pillar 2 (Active Information-Seeking).

Selects the next fixation to maximize an EXPECTED REDUCTION IN BELIEF
UNCERTAINTY. Exact mutual information I(z_future; a) is intractable; the
exact proxy used is documented in the class docstring below.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from rhan_core.beliefs.base import BeliefState
from rhan_core.gaze.base import GazePolicy
from rhan_core.gaze.halting import EntropyGatedHalting


class InformationGainGazePolicy(GazePolicy, nn.Module):
    """
    Fixation selection by prediction-error gradient ascent.

    MECHANISTIC IDENTITY — READ FIRST:
      The gaze update implemented here is, at initialization, MECHANICALLY
      IDENTICAL to the v12 Eq. II v12 gaze update (RHANv12._gaze_gradients + step
      rule in model_rhan_v12.py): autograd through grid_sample of the prediction
      error w.r.t. the gaze action, normalized, precision-scaled step
      0.20 + 0.30*Pi_D, clamped to +/-0.9. The lineage is: v10 used the
      FEATURE-space error only, v11 switched to the PIXEL-space error only, and
      v12 (which RHANNext inherits) is the lambda-blend of both — see
      model_rhan_v10.py / model_rhan_v11.py for the single-term predecessors.
      It is the SAME underlying mechanism, relocated into the new GazePolicy
      class structure — a clean architecture outcome, NOT a new computation. The only additive knobs are:
        (a) `step_net`, a small learned re-scale of the step size initialized to
            the identity (scale ~= 1.0, so behavior starts exactly at v12's
            formula), and
        (b) the optional GlobalPrecisionModulator `modulate_step_size` consumer
            (gain=1 -> identical to v12).
      It therefore implements v12's prediction-error gradient ascent, not a
      literal expected-information-gain computation.

    INFORMATION-GAIN INTERPRETATION (framing only, not a different mechanism):
      Under a Gaussian likelihood model of sensory prediction, the expected
      information gain at a fixation is proportional to the gradient of the
      expected *surprise* at that fixation (Friston 2010; Itti & Koch 2001).
      We approximate expected surprise with the lambda-blended prediction error
      the v12 forward pass already computes:

          J(a) = lambda * R(x, a) + (1 - lambda) * ||f_stem(a) - P(s)||

      where R(x,a) is the pixel-space reconstruction error of the foveal crop
      at gaze a (generative/top-down pathway) and ||f(a) - P(s)|| is the
      feature-space prediction error (bottom-up epistemic foraging).
      `select_action` moves the gaze along grad_a J(a), normalized, with the
      v12 precision-scaled step size re-scaled by `step_net` (identity at init).
      This is a CURRENT-error gradient (no one-step-ahead uncertainty
      prediction, no variance-reduction proxy, no world-model rollout) — see
      the roadmap for a genuine forward-looking EIG increment.

    Learnable state:
      step_net: (B, proj_dim + 1) -> (B, 1) sigmoid-gated re-scale of the
      v12 step formula. This is the parameter set the gradient-flow test
      asserts gradients reach.

    Machinery (plain dict, NOT submodules — no state-dict duplication):
      foveal_sample / generative_prior / foveal_stream / prior_predictor:
          the frozen v12 components used to evaluate the surprise gradient;
      modulate_step_size (optional): the GlobalPrecisionModulator's gaze-step
          consumer, so the precision gain re-scales the step exactly like the
          other consumers. When absent, the plain v12 formula is used.

    `EntropyGatedHalting` is owned here as `halter` and exposed as
    `halt_policy` by the model.
    """

    def __init__(self, proj_dim: int = 512, gaze_lambda: float = 0.5,
                 fovea_size: int = 48, base_step: float = 0.20,
                 precision_step_range: float = 0.30,
                 max_abs_gaze: float = 0.9,
                 halt_threshold: float = 0.35,
                 halt_softness: float = 8.0,
                 machinery: Dict = None):
        super().__init__()
        self.proj_dim = proj_dim
        self.gaze_lambda = float(gaze_lambda)
        self.fovea_size = fovea_size
        self.base_step = float(base_step)
        self.precision_step_range = float(precision_step_range)
        self.max_abs_gaze = float(max_abs_gaze)

        # Frozen v12 machinery, plain references (see class docstring).
        self._mach = machinery or {}

        # Learned step re-scale; last layer zero-initialized so the gate
        # starts at sigmoid(0) = 0.5 -> (0.5 + 0.5) = 1.0 == v12's step.
        self.step_net = nn.Sequential(
            nn.Linear(proj_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.step_net[-1].weight)
        nn.init.zeros_(self.step_net[-1].bias)

        # Uncertainty-gated halting (owned here, exposed by the model).
        self.halter = EntropyGatedHalting(threshold=halt_threshold,
                                          softness=halt_softness)

        # Diagnostics-only storage (never part of the graph):
        self.last_recon_map = None    # (B, fovea, fovea)
        self.last_surprise = None     # scalar

    # ── GazePolicy ───────────────────────────────────────────────────────────
    def select_action(self, belief: BeliefState,
                      history: list) -> torch.Tensor:
        """(B, 2) next gaze action via expected-surprise gradient ascent.

        Reads the current step's context from `history[-1]`:
            'action' (B,2), 'image' (B,3,96,96), 'belief_tensor' (B,D),
            'precision' (B,), 'halt' (B,) bool from the current step.
        Halting samples keep their previous gaze (no further foraging).
        """
        ctx = history[-1]
        a = ctx['action']
        x = ctx['image']
        s = ctx['belief_tensor']
        pi_d = ctx['precision']
        halt = ctx.get('halt')

        # ── Expected-surprise gradient (Eq. II v12, one blended objective) ──
        a_grad = a.detach().requires_grad_(True)
        with torch.enable_grad():
            x_fov_g = self._mach['foveal_sample'](
                x, a_grad, fovea_size=self.fovea_size)
            pred_g = self._mach['generative_prior'](s.detach())
            recon_map = (x_fov_g - pred_g).pow(2).mean(dim=1)   # (B,48,48)
            recon_err = recon_map.mean()
            f_g = self._mach['foveal_stream'](x_fov_g)
            prior_pred = self._mach['prior_predictor'](s.detach())
            feat_err = (f_g - prior_pred).norm(dim=-1).mean() / math.sqrt(
                f_g.shape[-1])
            obj = (self.gaze_lambda * recon_err
                   + (1.0 - self.gaze_lambda) * feat_err)
            g_total = torch.autograd.grad(obj, a_grad, create_graph=False)[0]

        self.last_recon_map = recon_map.detach()
        self.last_surprise = obj.detach()

        # ── Normalize + precision-scaled step, re-scaled by step_net ────────
        grad_norm = g_total.norm(dim=-1, keepdim=True) + 1e-8
        normed_grad = g_total / grad_norm

        gate_in = torch.cat([s.detach(), pi_d.detach().unsqueeze(-1)], dim=-1)
        scale = 0.5 + torch.sigmoid(self.step_net(gate_in))     # ~1.0 at init

        # v12 step formula, gain-scaled by the precision modulator when it is
        # wired in (gain=1 -> identical to v12); step_net re-scales on top.
        mod_step = self._mach.get('modulate_step_size', None)
        if mod_step is not None:
            step_size = mod_step(pi_d) * scale.squeeze(-1)      # (B,)
        else:
            step_size = (self.base_step
                         + self.precision_step_range * pi_d) \
                * scale.squeeze(-1)                             # (B,)

        a_new = torch.clamp(a + step_size.unsqueeze(-1) * normed_grad,
                            -self.max_abs_gaze, self.max_abs_gaze)
        if halt is not None:
            # Halted samples stop foraging: keep the previous gaze (detached).
            a_new = torch.where(halt.unsqueeze(-1), a.detach(), a_new)
        return a_new

    def should_halt(self, belief: BeliefState, history: list) -> torch.Tensor:
        """(B,) bool — delegates to the EntropyGatedHalting gate."""
        return self.halter.should_halt(belief, history)

    def __repr__(self) -> str:
        return (f"InformationGainGazePolicy(lambda={self.gaze_lambda}, "
                f"proxy='expected-surprise-gradient', "
                f"halter={self.halter})")

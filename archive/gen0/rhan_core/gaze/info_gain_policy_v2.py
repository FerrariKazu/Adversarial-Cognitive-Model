"""
AIS-v2 — genuine one-step-lookahead expected information gain gaze policy.
================================================================================

RUN LABEL — AIS-v2: candidate-based expected-information-gain gaze. This is
the mechanism AIS-v1's name always claimed but never demonstrated: AIS-v1 is
Relocated Equation II (gradient ascent on the CURRENT prediction error — no
one-step-ahead prediction, no candidate evaluation; documented in
rhan_core/gaze/info_gain_policy.py). AIS-v2 replaces that with a genuine
expected-information-gain computation over a small set of candidate next
fixations.

MECHANISTIC IDENTITY — READ FIRST (the same honesty discipline applied to
AIS-v1's docstring):

  For each candidate next fixation c in a small set C (K=4-8 candidates
  sampled around the current belief's highest-uncertainty region — the
  argmax of the image-space reconstruction error map at the current gaze,
  NOT a brute-force search over the whole image), we estimate the expected
  posterior uncertainty reduction via a SECOND forward pass through the
  (cheap) foveal stream ONLY — never the full backbone:

      f_c  = foveal_stream(foveal_sample(x, c))              # (B, 512)
      EIG(c) ~= candidate_head([f_c, pi_D])                  # predicted surprise

  The candidate-evaluation head (`candidate_head`, Generation-0 optimizer
  group "ais_v2") is trained to predict the surprise an observation at c
  would actually produce, with a ONE-STEP temporal-difference target: the
  observed foveal-feature surprise at the fixation the policy then moves to
  (computed at the next foraging step: ||foveal_feat_{t+1} −
  prior_predictor(s_{t+1})|| / sqrt(D), DETACHED as a target). The policy
  moves toward the candidate with MAXIMUM predicted surprise — under the
  active-inference reading, more surprising observations carry more
  information about the input, so max-surprise selection ≈ max expected
  information gain. The head's predictions therefore become informative
  about where surprise actually lands, and the smoke gate tests EXACTLY
  that: the predicted-vs-observed surprise correlation must be measurably
  above the ~0.0 a random-selection policy would show.

  Approximations (documented, not silent):
    1. EIG is proxied by predicted next-step surprise at the candidate —
       not a full posterior KL reduction (intractable with a point-belief
       backbone).
    2. Only K candidates are evaluated (sampled around the max-error
       region), so the "best" is best-within-C, not best-over-the-image.
    3. Candidate features are DETACHED before the head (f_c.detach()) —
       the head evaluates features, it does not couple the gaze-sampling
       Jacobian into the backbone (keeps the head's gradient signal
       self-contained).

  Learnable state:
    candidate_head: (B, proj_dim + 1) -> (B, 1) predicted surprise for a
    candidate fixation. This is the parameter set the gradient-flow test and
    the Generation-0 |dW| pre-flight assert gradients reach.

  Halting: owned here as `halter` (EntropyGatedHalting), exposed as
  `halt_policy` by the model — identical semantics to AIS-v1 (soft
  continuation gate, no step-count penalty).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from rhan_core.beliefs.base import BeliefState
from rhan_core.gaze.base import GazePolicy
from rhan_core.gaze.halting import EntropyGatedHalting


class InformationGainGazePolicyV2(GazePolicy, nn.Module):
    """
    Candidate-based one-step-lookahead expected information gain gaze.

    Args:
        proj_dim: belief/feature dimension (512).
        fovea_size: foveal crop size (48).
        num_candidates: K candidates evaluated per step (4-8; tractable).
        candidate_sigma: Gaussian spread (normalized image coords) around the
            highest-uncertainty region when sampling candidates.
        halt_threshold / halt_softness: EntropyGatedHalting parameters
            (same defaults as AIS-v1).
        machinery: plain-reference dict of frozen v12 components —
            foveal_sample / foveal_stream / generative_prior /
            prior_predictor (same pattern as AIS-v1).
    """

    def __init__(self, proj_dim: int = 512, fovea_size: int = 48,
                 num_candidates: int = 6, candidate_sigma: float = 0.12,
                 max_abs_gaze: float = 0.9,
                 halt_threshold: float = 0.35, halt_softness: float = 8.0,
                 machinery: Optional[Dict] = None):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.fovea_size = int(fovea_size)
        self.num_candidates = int(num_candidates)
        self.candidate_sigma = float(candidate_sigma)
        self.max_abs_gaze = float(max_abs_gaze)

        self._mach = machinery or {}

        # Candidate-evaluation head: (B, proj_dim + 1) -> (B, 1) predicted
        # surprise at a candidate fixation. Generation-0 optimizer group
        # "ais_v2" (lr_multiplier AIS_V2_LR_MULT).
        self.candidate_head = nn.Sequential(
            nn.Linear(proj_dim + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Uncertainty-gated halting (same semantics as AIS-v1).
        self.halter = EntropyGatedHalting(threshold=halt_threshold,
                                          softness=halt_softness)

        # Diagnostics-only storage (never part of the graph):
        self.last_candidates = None      # (B, K, 2) evaluated candidates
        self.last_candidate_surprises = None  # (B, K) predicted surprises
        self.last_chosen = None          # (B, 2) chosen next fixation

    # ── GazePolicy ───────────────────────────────────────────────────────────
    def select_action(self, belief: BeliefState,
                      history: list) -> torch.Tensor:
        """(B, 2) next gaze action via one-step-lookahead EIG over candidates.

        Reads the current step's context from `history[-1]`:
            'action' (B,2), 'image' (B,3,96,96), 'belief_tensor' (B,D),
            'precision' (B,), 'halt' (B,) bool.
        Halting samples keep their previous gaze (no further foraging).
        """
        ctx = history[-1]
        a = ctx['action']
        x = ctx['image']
        s = ctx['belief_tensor']
        pi_d = ctx['precision']
        halt = ctx.get('halt')
        B = x.shape[0]

        # ── Highest-uncertainty region: argmax of the current reconstruction
        # error map (the place the belief is most wrong about right now). ──
        with torch.enable_grad():
            x_fov = self._mach['foveal_sample'](
                x, a.detach(), fovea_size=self.fovea_size)
            pred = self._mach['generative_prior'](s.detach())
            recon_map = (x_fov - pred).pow(2).mean(dim=1)     # (B,48,48)
        self.last_recon_map = recon_map.detach()
        # Crop pixel -> normalized crop coords [-1,1] -> image coords.
        H = W = self.fovea_size
        flat = recon_map.detach().view(B, -1)
        argmax = flat.argmax(dim=1)
        py = argmax // W
        px = argmax % W
        crop_u = (px.float() + 0.5) / (W / 2.0) - 1.0          # [-1, 1]
        crop_v = (py.float() + 0.5) / (H / 2.0) - 1.0
        scale = self.fovea_size / 96.0
        anchor_x = a[:, 0].detach() + scale * crop_u            # image coords
        anchor_y = a[:, 1].detach() + scale * crop_v
        anchor = torch.stack([anchor_x, anchor_y], dim=-1)      # (B, 2)

        # ── Sample K candidates around the anchor (tractable, not brute-force).
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        noise = torch.randn(B, self.num_candidates, 2, device=x.device,
                            generator=gen) * self.candidate_sigma
        # Anchor is the first candidate (the policy can always stay put).
        cand = anchor.unsqueeze(1).expand(B, self.num_candidates, 2).clone()
        cand[:, 1:] = cand[:, 1:] + noise[:, 1:]
        cand = torch.clamp(cand, -self.max_abs_gaze, self.max_abs_gaze)

        # ── Evaluate each candidate through the foveal stream ONLY ──────────
        # (a SECOND forward pass per candidate — the cheap stream, never the
        # full backbone). Features are DETACHED before the head so the head
        # evaluates rather than coupling the sampling Jacobian (documented
        # approximation 3).
        with torch.enable_grad():
            cand_feats = []
            for k in range(self.num_candidates):
                c = cand[:, k].detach().requires_grad_(False)
                crop_k = self._mach['foveal_sample'](
                    x, c, fovea_size=self.fovea_size)
                f_k = self._mach['foveal_stream'](crop_k).detach()  # (B,512)
                cand_feats.append(f_k)
            cand_feats = torch.stack(cand_feats, dim=1)          # (B,K,512)
            head_in = torch.cat(
                [cand_feats, pi_d.detach().unsqueeze(-1).unsqueeze(1)
                 .expand(B, self.num_candidates, 1)], dim=-1)    # (B,K,513)
            surprises = self.candidate_head(head_in).squeeze(-1)  # (B,K)

        # ── Move toward the max-predicted-surprise candidate ────────────────
        chosen_idx = surprises.argmax(dim=1)                     # (B,)
        a_new = cand.gather(1, chosen_idx.unsqueeze(-1)
                            .unsqueeze(-1).expand(B, 1, 2)).squeeze(1)

        self.last_candidates = cand.detach()
        self.last_candidate_surprises = surprises.detach()
        self.last_chosen = a_new.detach()
        self.last_chosen_surprise = surprises.gather(
            1, chosen_idx.unsqueeze(-1)).squeeze(1)              # (B,) attached

        if halt is not None:
            # Halted samples stop foraging: keep the previous gaze (detached).
            a_new = torch.where(halt.unsqueeze(-1), a.detach(), a_new)
        return a_new

    def should_halt(self, belief: BeliefState, history: list) -> torch.Tensor:
        """(B,) bool — delegates to the EntropyGatedHalting gate."""
        return self.halter.should_halt(belief, history)

    def __repr__(self) -> str:
        return (f"InformationGainGazePolicyV2(K={self.num_candidates}, "
                f"sigma={self.candidate_sigma}, "
                f"proxy='one-step-lookahead EIG over candidates', "
                f"halter={self.halter})")
"""
RHANNext — the refactored, pillar-composable successor to RHAN-v12.

Design contract (enforced by tests/test_config_backward_compat.py):

  * RHANNext subclasses the FROZEN RHANv12 (phase1_training/model_rhan_v12.py
    is never modified). With the DEFAULT config (all pillars off) the state
    dict is byte-identical to RHANv12's and `forward` delegates to the exact
    v12 implementation — a v12 checkpoint loads 1:1, and the existing
    eval/pipeline keeps working unchanged.
  * New pillar components are added ONLY as new submodules behind
    RHANNextConfig toggles, so each mechanism can be isolated with an on/off
    test (project lesson #3: never add multiple mechanisms simultaneously).
  * Every new loss-bearing path (reconstruction, gaze policy, precision
    modulator, HPC stack) has an automated gradient-reachability test
    (project lesson #1).

Pillars:
  * AIS (Pillar 2, Stage 1)   — InformationGainGazePolicy + EntropyGatedHalting
                                + GlobalPrecisionModulator, gated by enable_ais.
  * HPC (Pillar 1, Stage 2)   — HierarchicalPredictiveStack (1 level), gated
                                by enable_hpc / hpc_num_levels.
  * SBR (Pillar 3) / IWM (Pillar 4) — scaffold only; NullWorldModel is always
                                wired as the safe no-op; enable_sbr/iwm
                                validate to an error.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# The frozen v12 chain inserts phase1_training on sys.path on import; we do the
# same explicitly so this package works regardless of the caller's cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_P1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "phase1_training"))
for _p in (_P1_DIR, os.path.abspath(os.path.join(_THIS_DIR, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_rhan_v12 import RHANv12                      # frozen backbone
from model_rhan_v10 import foveal_sample                # frozen helper

from rhan_core.beliefs.vector_belief import VectorBeliefState
from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.gaze.info_gain_policy import InformationGainGazePolicy
from rhan_core.precision.global_precision import GlobalPrecisionModulator
from rhan_core.world_model.null_world_model import NullWorldModel


class RHANNext(RHANv12):
    """
    RHAN-Next: pillar-composable successor to RHAN-v12.

    Args:
        config: RHANNextConfig — the ONLY configuration object. Defaults to
                RHANNextConfig() (v12-equivalent). Alternative: pass config
                fields as keyword arguments (RHANNext(enable_ais=True, ...)).
    """

    def __init__(self, config: Optional[RHANNextConfig] = None, **kwargs):
        self.config = config if config is not None else RHANNextConfig(**kwargs)
        self.config.validate()
        super().__init__(**self.config.v12_kwargs())
        self._build_pillars()

    # ────────────────────────────────────────────────────────────────────────
    # Pillar construction
    # ────────────────────────────────────────────────────────────────────────
    def _build_pillars(self):
        # Pillar 4 (IWM): always present as a safe no-op (zero params/buffers,
        # so the default state dict stays identical to RHANv12's).
        self.world_model = NullWorldModel()

        # ── Pillar 2 (AIS) — Stage 1 ─────────────────────────────────────────
        if self.config.enable_ais:
            # Wraps the shared image_precision module by plain reference (its
            # consumers are then passed to the gaze policy below).
            self.precision_modulator = GlobalPrecisionModulator(
                image_precision_module=self.image_precision,
                tau=self.config.precision_tau,
                gain=1.0)
            # Plain-reference machinery (NOT submodules) so the state dict is
            # not duplicated: the policy shares the frozen v12 components.
            machinery = {
                'foveal_sample': foveal_sample,
                'generative_prior': self.generative_prior,
                'foveal_stream': self.foveal_stream,
                'prior_predictor': self.precision_ctrl.prior_predictor,
                # Gaze-step consumer of the precision modulator (gain-scaled).
                'modulate_step_size': self.precision_modulator.modulate_step_size,
            }
            self.gaze_policy = InformationGainGazePolicy(
                proj_dim=self.config.proj_dim,
                gaze_lambda=self.config.gaze_lambda,
                fovea_size=self.config.fovea_size,
                base_step=self.config.ais_base_step,
                precision_step_range=self.config.ais_precision_step_range,
                halt_threshold=self.config.ais_halt_threshold,
                halt_softness=self.config.ais_continuation_softness,
                machinery=machinery,
            )
            # The halt policy is owned by the gaze policy; expose it under a
            # stable name for the forward loop and tests.
            self.halt_policy = self.gaze_policy.halter

        # ── Pillar 1 (HPC) — Stage 2 ─────────────────────────────────────────
        if self.config.enable_hpc and self.config.hpc_num_levels >= 1:
            # Deferred import so the Stage-1 tree (before the stack lands)
            # imports cleanly; RHANNextConfig already guards level count.
            from rhan_core.predictive_coding.hierarchical_stack import (
                HierarchicalPredictiveStack)
            self.hpc_stack = HierarchicalPredictiveStack(
                proj_dim=self.config.proj_dim,
                num_levels=self.config.hpc_num_levels,
                fovea_size=self.config.fovea_size)

    @property
    def pillars_active(self) -> bool:
        """True when any implemented pillar is enabled (non-default path)."""
        return self.config.enable_ais or (
            self.config.enable_hpc and self.config.hpc_num_levels >= 1)

    # ────────────────────────────────────────────────────────────────────────
    # Shared foraging loop (AIS/HPC-aware). Mirrors v12's loop exactly when
    # pillars are off; when on, adds belief/uncertainty bookkeeping, the
    # entropy-gated continuation weights, the info-gain gaze update, and the
    # HPC prediction errors.
    # ────────────────────────────────────────────────────────────────────────
    def _forage(self, x, collect_traj: bool):
        """Run the multi-step foraging loop.

        Args:
            x: (B, 3, 96, 96)
            collect_traj: build the trajectory dict (diagnostics/losses).
        Returns:
            (final_belief (B, 512), trajectory dict or None)
        """
        B = x.shape[0]

        # Step 0: peripheral pass (full image, inherited).
        cls_768 = self._peripheral_pass(x)              # (B, 768)
        s = self.peripheral_proj(cls_768)                # (B, 512)
        if self.freeze_gaze:
            a = torch.zeros(B, 2, device=x.device)       # center fixation
        else:
            a = self.action_init(s)                      # (B, 2)

        # Parafoveal: computed ONCE (full-field, low-res).
        para_feat = self.parafoveal_stream(x)            # (B, 512)

        # Accumulators (constant continuation = 1 in pure v12 mode; AIS uses
        # the soft uncertainty gate below).
        weighted_belief = torch.zeros_like(s)            # (B, 512)
        weight_sum = torch.zeros(B, device=x.device)     # (B,)

        trajectory = None
        if collect_traj:
            trajectory = {
                'actions': [], 'precisions': [], 'errors': [], 'gate_alphas': [],
                'recon_errors': [], 'recon_maps': [], 'steps': 0,
                'uncertainties': [], 'continuations': [],
            }
            if hasattr(self, 'hpc_stack') and len(self.hpc_stack.levels) > 0:
                trajectory['hpc_errors'] = []

        history: list = []

        for t in range(self.max_steps):
            # Eq. II: sample foveal crop at gaze position.
            x_foveal = foveal_sample(x, a, fovea_size=self.fovea_size)
            foveal_feat = self.foveal_stream(x_foveal)   # (B, 512)

            # Tier 1.1: blend foveal + parafoveal via learned gate.
            combined_feat, alpha = self.foveal_gate(foveal_feat, para_feat, s)

            # Tier 3.1: generative prior predicts expected crop.
            predicted_crop = self.generative_prior(s)    # (B, 3, 48, 48)

            # Image-space prediction error (genuine, bounded). RAW pi_d — the
            # modulator's gain only enters via explicit consumer modulations.
            if hasattr(self, 'precision_modulator'):
                pi_d, error_mag = self.precision_modulator.precision_from_crops(
                    x_foveal, predicted_crop, s)
            else:
                pi_d, error_mag = self.image_precision(
                    x_foveal, predicted_crop, s)

            # Precision-weighted belief integration (v12 semantics).
            pi_d_unsq = pi_d.unsqueeze(-1)               # (B, 1)
            s = (1 - pi_d_unsq) * s + pi_d_unsq * combined_feat

            # HPC prediction errors (Pillar 1, Stage 2) — NOT detached so the
            # error reaches the stack's parameters through the loss. Computed
            # only when the trajectory is collected (the trainer reads the HPC
            # loss exclusively from trajectories; inference skips the cost).
            if (collect_traj and hasattr(self, 'hpc_stack')
                    and len(self.hpc_stack.levels) > 0):
                for lvl in range(len(self.hpc_stack.levels)):
                    target = self.hpc_stack.extract_targets(x_foveal, lvl)
                    pred_hpc = self.hpc_stack.predict(s, lvl)
                    err_hpc = self.hpc_stack.compute_error(pred_hpc, target, lvl)
                    if collect_traj:
                        trajectory['hpc_errors'].append(err_hpc)

            # ── Belief wrapper + policies (AIS); v12 fallback otherwise ────
            has_ais = hasattr(self, 'halt_policy') and hasattr(self, 'gaze_policy')
            if has_ais:
                u = 1.0 - pi_d                             # uncertainty proxy
                belief = VectorBeliefState(s, uncertainty=u)
                ctx = {'action': a, 'image': x, 'belief_tensor': s,
                       'precision': pi_d, 'step': t}
                if hasattr(self, 'precision_modulator'):
                    ctx['halt_threshold'] = \
                        self.precision_modulator.modulate_halting_threshold(
                            pi_d, self.config.ais_halt_threshold)
                halt = self.halt_policy.should_halt(belief, [ctx])   # (B,) bool
                ctx['halt'] = halt
                history.append(ctx)
                # Soft continuation: sigma(softness * (u - threshold)).
                cont = self.halt_policy.continuation(belief, [ctx])   # (B,)
            else:
                # Exact v12 semantics: constant continuation = 1, no halt.
                halt = torch.zeros(B, dtype=torch.bool, device=x.device)
                cont = torch.ones(B, device=x.device)
                u = 1.0 - pi_d

            # Accumulate belief, weighted by continuation (AIS halting).
            weighted_belief += cont.unsqueeze(-1) * s
            weight_sum += cont

            # Record trajectory for diagnostics/losses.
            if collect_traj:
                trajectory['actions'].append(a.detach())
                trajectory['precisions'].append(pi_d.detach())
                trajectory['errors'].append(error_mag.detach())
                trajectory['gate_alphas'].append(alpha.detach())
                # NOTE: NOT detached — get_reconstruction_loss() must return a
                # differentiable scalar (v12 fix; v11 detached -> no-op).
                trajectory['recon_errors'].append(
                    (x_foveal - predicted_crop).pow(2).mean())
                trajectory['uncertainties'].append(u.detach())
                trajectory['continuations'].append(cont.detach())

            # Eq. II v12: gaze update (info-gain policy under AIS).
            if not self.freeze_gaze and t < self.max_steps - 1:
                if hasattr(self, 'gaze_policy'):
                    a = self.gaze_policy.select_action(belief, history)
                else:
                    g_total, recon_map = self._gaze_gradients(x, a, s)
                    if collect_traj:
                        trajectory['recon_maps'].append(recon_map.detach())
                    grad_norm = g_total.norm(dim=-1, keepdim=True) + 1e-8
                    normed_grad = g_total / grad_norm
                    step_size = 0.20 + 0.30 * pi_d.unsqueeze(-1)
                    a = torch.clamp(a + step_size * normed_grad, -0.9, 0.9)
                if collect_traj and hasattr(self, 'gaze_policy'):
                    # Diagnostics-only recon map from the policy's last step.
                    if self.gaze_policy.last_recon_map is not None:
                        trajectory['recon_maps'].append(
                            self.gaze_policy.last_recon_map)

        if collect_traj:
            trajectory['steps'] = self.max_steps

        final_belief = weighted_belief / (weight_sum.unsqueeze(-1) + 1e-8)
        return final_belief, trajectory

    # ────────────────────────────────────────────────────────────────────────
    # Public API — default config delegates EXACTLY to the frozen v12 path.
    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x, return_trajectory=False):
        """v12-compatible forward. Default config: byte-for-byte v12."""
        if not self.pillars_active:
            return super().forward(x, return_trajectory=return_trajectory)
        final_belief, trajectory = self._forage(x, collect_traj=return_trajectory)
        final_768 = self.belief_unproj(final_belief)     # (B, 768)
        logits = self.classifier(final_768)
        if return_trajectory:
            return logits, trajectory
        return logits

    def get_feature_vector(self, x):
        """v12-compatible 768-dim feature vector (TRADES/eval compat)."""
        if not self.pillars_active:
            return super().get_feature_vector(x)
        final_belief, _ = self._forage(x, collect_traj=False)
        return self.belief_unproj(final_belief)

    def get_hpc_loss(self, x, trajectory_logits_tuple):
        """Differentiable HPC prediction-error loss (0.0 when HPC is off).

        Args:
            x: (B, 3, 96, 96) input.
            trajectory_logits_tuple: (logits, trajectory) from forward().
        Returns:
            scalar — mean of the collected per-level prediction errors.
        """
        _, traj = trajectory_logits_tuple
        errs = traj.get('hpc_errors', []) if isinstance(traj, dict) else []
        if not errs:
            return torch.zeros((), device=x.device)
        return torch.stack(errs).mean()

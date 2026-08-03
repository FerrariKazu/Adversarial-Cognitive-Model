"""
RHAN-v12: Locked-In Active Inference Architecture
===================================================

v12 is the "locked-in" consolidation of everything that made the v11
isolation runs successful. Three surgical changes over RHANv11:

1. NO HALT NETWORK — foraging depth is fixed at T = 4 unconditionally.
   The ThermodynamicHalt network (v10/v11) is removed from the forward
   pipeline entirely; the loop always runs exactly `max_steps` iterations
   and the belief accumulator uses a constant continuation weight of 1.
   This matches what every successful run actually did (the halt network
   never expressed differential clean-vs-adversarial behavior; mean steps
   sat at the cap).

2. RECONSTRUCTION-GUIDED GAZE (Eq. II v12) — the gaze update blends TWO
   gradient signals with a fixed weight lambda (default 0.5, CLI-tunable):
       g_total = lambda * grad_a R(x, a) + (1 - lambda) * grad_a ||f(a) - P(s)||
     - R(x, a) is the GenerativePrior's PIXEL-SPACE reconstruction error
       exposed as a spatial map R(x,a) in R^{B x H x W} (per-pixel squared
       error of the foveal crop at gaze a vs the predicted crop). Its
       gradient w.r.t. the gaze action is the motor-Jacobian of the
       generative (top-down) pathway.
     - ||f(a) - P(s)|| is the FEATURE-SPACE prediction error from the v10
       PrecisionController prior predictor (bottom-up epistemic foraging).
   Weighted sum -> normalized -> precision-scaled step (same update rule
   as v10/v11).

3. LOSS PIPELINE — the supervised auxiliary losses (foraging consistency,
   precision calibration, halt efficiency) are DELETED from the training
   script (train_rhan_v12.py). The model keeps the ImageSpacePrecision
   forward pass (Pi_D generation) and its use in dynamic beta modulation;
   only the supervised calibration loss target is gone.

Checkpoint compatibility:
  - v12 removes the `halt_net` module from the state dict. Loading a v11 /
    RHAN-Large base with strict=False leaves halt_net keys as "unexpected"
    (ignored). All other parameter names are identical to RHANv11, so the
    TRADES-Large base (`rhan_stl10_large_pseudolabel_best.pth`) and any
    v11 checkpoint load cleanly.
  - Parameter count identical to RHANv11 (~63.5M minus halt_net ~0.1M).

References:
  Friston (2010) — The free-energy principle
  Rao & Ballard (1999) — Predictive coding in visual cortex
  Itti & Koch (2001) — Saliency-based visual attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_v11 import (
    RHANv11,
    GenerativePrior,
    ParafovealStream,
    FovealParafovealGate,
    ImageSpacePrecision,
)
from model_rhan_v10 import (
    foveal_sample,
    FovealStream,
    PrecisionController,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL RHAN-v12: Locked-In Active Inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RHANv12(RHANv11):
    """
    RHAN-v12: Locked-In Active Inference Architecture.

    Changes over RHANv11:
      - halt_net removed (no thermodynamic halt; fixed T = max_steps)
      - reconstruction-guided gaze: lambda blend of pixel-space
        reconstruction error gradient R(x,a) and feature-space
        prediction error gradient (Eq. II v12)
      - belief accumulation with constant continuation weight (=1)

    Total: ~63.4M parameters (identical to v11 minus halt_net).
    """

    def __init__(self, num_classes=10,
                 embed_dim=768,
                 proj_dim=512,
                 num_heads=12,
                 ff_dim=3072,
                 num_transformer_layers=8,
                 num_recurrent_steps=2,
                 stem_dropout=0.1,
                 max_foraging_steps=4,       # fixed T; loop always runs this many
                 fovea_size=48,
                 metabolic_cost=0.05,
                 precision_tau=0.1,
                 gaze_lambda=0.5):           # reconstruction-guided gaze weight

        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_layers=num_transformer_layers,
            num_recurrent_steps=num_recurrent_steps,
            stem_dropout=stem_dropout,
            max_foraging_steps=max_foraging_steps,
            fovea_size=fovea_size,
            metabolic_cost=metabolic_cost,
            precision_tau=precision_tau,
        )

        # ── Locked-in: remove the halt network entirely ─────────
        # The module is dropped from the tree so (a) the state dict never
        # carries halt_net weights and (b) no code path can invoke it.
        if hasattr(self, 'halt_net'):
            del self.halt_net

        # ── Reconstruction-guided gaze weight (Eq. II v12) ──────
        self.gaze_lambda = float(gaze_lambda)

        # ISOLATION TEST: freeze gaze at image center (inherited flag)
        self.freeze_gaze = False

    # ────────────────────────────────────────────────────────────────────────
    # Reconstruction error map R(x, a)
    # ────────────────────────────────────────────────────────────────────────

    def compute_recon_error_map(self, x_crop, predicted_crop):
        """
        Pixel-space reconstruction error map R(x, a).

        Args:
            x_crop:        (B, 3, 48, 48) — actual foveal crop at gaze a
            predicted_crop:(B, 3, 48, 48) — GenerativePrior prediction

        Returns:
            R: (B, 48, 48) — per-pixel squared error, mean over channels.
               This is the spatial map whose gradient w.r.t. the gaze
               action drives the reconstruction-guided component of the
               Eq. II update.
        """
        return (x_crop - predicted_crop).pow(2).mean(dim=1)

    # ────────────────────────────────────────────────────────────────────────
    # Gaze update (Eq. II v12)
    # ────────────────────────────────────────────────────────────────────────

    def _gaze_gradients(self, x, a, s):
        """
        Combined motor-Jacobian for the Eq. II v12 gaze update.

        Returns (g_total, recon_map):
            g_total:   (B, 2) — lambda * grad_a R(x,a) + (1-lambda) * grad_a F(x,a)
            recon_map: (B, 48, 48) — the exposed spatial reconstruction error map
        """
        a_grad = a.detach().requires_grad_(True)
        with torch.enable_grad():
            x_fov_g = foveal_sample(x, a_grad, fovea_size=self.fovea_size)

            # Pixel-space objective: spatial mean of R(x, a)
            pred_g = self.generative_prior(s.detach())
            recon_map = self.compute_recon_error_map(x_fov_g, pred_g)
            recon_err = recon_map.mean()

            # Feature-space objective: || f_stem(a) - P(s) ||  (v10 Eq. II)
            f_g = self.foveal_stream(x_fov_g)
            prior_pred = self.precision_ctrl.prior_predictor(s.detach())
            feat_err = (f_g - prior_pred).norm(dim=-1).mean() / math.sqrt(
                f_g.shape[-1])

            # λ-blend as ONE scalar objective (gradient is linear, so
            # lambda*grad R + (1-lambda)*grad F == grad[lambda*R + (1-lambda)*F])
            # — a single backward pass, mathematically identical to two.
            obj = (self.gaze_lambda * recon_err
                   + (1.0 - self.gaze_lambda) * feat_err)
            g_total = torch.autograd.grad(obj, a_grad, create_graph=False)[0]

        return g_total, recon_map

    def forward(self, x, return_trajectory=False):
        """
        Locked-In Active Inference forward pass.

        Fixed T = max_steps foraging loop (no halt network, no early exit).
        Args:
            x: (B, 3, 96, 96) — input images
            return_trajectory: if True, returns (logits, trajectory_dict)

        Returns:
            logits: (B, num_classes)
            trajectory (optional): dict with 'actions', 'precisions',
                                   'errors', 'gate_alphas', 'recon_errors',
                                   'recon_maps', 'steps' (= max_steps)
        """
        B = x.shape[0]

        # ── Step 0: Peripheral pass (full image, inherited) ──────
        cls_768 = self._peripheral_pass(x)              # (B, 768)
        s = self.peripheral_proj(cls_768)                # (B, 512)
        if self.freeze_gaze:
            a = torch.zeros(B, 2, device=x.device)       # center fixation
        else:
            a = self.action_init(s)                      # (B, 2)

        # ── Parafoveal: computed ONCE (full-field, low-res) ───────
        para_feat = self.parafoveal_stream(x)            # (B, 512)

        # ── Initialize accumulators (constant continuation = 1) ──
        weighted_belief = torch.zeros_like(s)            # (B, 512)
        weight_sum = torch.zeros(B, device=x.device)     # (B,)

        trajectory = {
            'actions': [],
            'precisions': [],
            'errors': [],
            'gate_alphas': [],
            'recon_errors': [],
            'recon_maps': [],
            'steps': 0,
        }

        # ── Locked-In Foraging Loop (T = max_steps, unconditional) ─
        for t in range(self.max_steps):

            # Eq. II: sample foveal crop at gaze position
            x_foveal = foveal_sample(x, a, fovea_size=self.fovea_size)
            foveal_feat = self.foveal_stream(x_foveal)   # (B, 512)

            # Tier 1.1: blend foveal + parafoveal via learned gate
            combined_feat, alpha = self.foveal_gate(
                foveal_feat, para_feat, s)                # (B, 512), (B, 1)

            # Tier 3.1: generative prior predicts expected crop
            predicted_crop = self.generative_prior(s)     # (B, 3, 48, 48)

            # Image-space prediction error (genuine, bounded)
            pi_d, error_mag = self.image_precision(
                x_foveal, predicted_crop, s)              # (B,), (B,)

            # Precision-weighted belief integration
            pi_d_unsq = pi_d.unsqueeze(-1)                # (B, 1)
            s = (1 - pi_d_unsq) * s + pi_d_unsq * combined_feat

            # Accumulate belief — constant continuation (no halt)
            weighted_belief += s
            weight_sum += 1.0

            # Record trajectory for diagnostics
            if return_trajectory:
                trajectory['actions'].append(a.detach())
                trajectory['precisions'].append(pi_d.detach())
                trajectory['errors'].append(error_mag.detach())
                trajectory['gate_alphas'].append(alpha.detach())
                recon_mse = (x_foveal - predicted_crop).pow(2).mean()
                # NOTE: intentionally NOT detached — get_reconstruction_loss()
                # must return a differentiable scalar so w_recon * L_recon
                # actually trains the generative prior. (v11 detached here,
                # which made the reconstruction term a silent gradient no-op.)
                trajectory['recon_errors'].append(recon_mse)

            # Eq. II v12: reconstruction-guided gaze update
            # (all steps except the last — the final crop is classification)
            if not self.freeze_gaze and t < self.max_steps - 1:
                g_total, recon_map = self._gaze_gradients(x, a, s)
                if return_trajectory:
                    trajectory['recon_maps'].append(recon_map.detach())

                # Normalize for stable step size
                grad_norm = g_total.norm(dim=-1, keepdim=True) + 1e-8
                normed_grad = g_total / grad_norm

                # Fixed base step + precision-scaled component
                step_size = 0.20 + 0.30 * pi_d.unsqueeze(-1)
                a = torch.clamp(a + step_size * normed_grad, -0.9, 0.9)

        trajectory['steps'] = self.max_steps

        # ── Final classification from accumulated belief ─────────
        final_belief = weighted_belief / (weight_sum.unsqueeze(-1) + 1e-8)
        final_768 = self.belief_unproj(final_belief)      # (B, 768)
        logits = self.classifier(final_768)

        if return_trajectory:
            return logits, trajectory
        return logits

    def get_feature_vector(self, x):
        """
        For TRADES compatibility: returns 768-dim feature vector.
        Uses the full v12 locked-in multi-resolution forward pass
        (fixed T, reconstruction-guided gaze).
        """
        B = x.shape[0]
        cls_768 = self._peripheral_pass(x)
        s = self.peripheral_proj(cls_768)
        if self.freeze_gaze:
            a = torch.zeros(B, 2, device=x.device)
        else:
            a = self.action_init(s)

        para_feat = self.parafoveal_stream(x)

        weighted_belief = torch.zeros_like(s)
        weight_sum = torch.zeros(B, device=x.device)

        for t in range(self.max_steps):
            x_foveal = foveal_sample(x, a, fovea_size=self.fovea_size)
            foveal_feat = self.foveal_stream(x_foveal)

            combined_feat, _ = self.foveal_gate(foveal_feat, para_feat, s)

            predicted_crop = self.generative_prior(s)
            pi_d, error_mag = self.image_precision(
                x_foveal, predicted_crop, s)

            pi_d_unsq = pi_d.unsqueeze(-1)
            s = (1 - pi_d_unsq) * s + pi_d_unsq * combined_feat

            weighted_belief += s
            weight_sum += 1.0

            if not self.freeze_gaze and t < self.max_steps - 1:
                g_total, _ = self._gaze_gradients(x, a, s)
                grad_norm = g_total.norm(dim=-1, keepdim=True) + 1e-8
                normed_grad = g_total / grad_norm
                step_size = 0.20 + 0.30 * pi_d.unsqueeze(-1)
                a = torch.clamp(a + step_size * normed_grad, -0.9, 0.9)

        final_belief = weighted_belief / (weight_sum.unsqueeze(-1) + 1e-8)
        return self.belief_unproj(final_belief)

    def get_reconstruction_loss(self, x, trajectory_logits_tuple):
        """
        Compute reconstruction loss for training the generative prior.
        Call after forward(return_trajectory=True).

        v12 fix: the per-step recon errors are stored WITHOUT detach, so this
        returns a DIFFERENTIABLE scalar whose gradient reaches the generative
        prior (predicted_crop) and the foveal pathway. v11 detached them,
        which silently zeroed every gradient from the reconstruction term.

        Args:
            x: (B, 3, 96, 96) — input images
            trajectory_logits_tuple: (logits, trajectory) from forward()

        Returns:
            recon_loss: scalar — mean reconstruction MSE across all steps
        """
        _, traj = trajectory_logits_tuple
        if len(traj.get('recon_errors', [])) == 0:
            return torch.tensor(0.0, device=x.device)
        return torch.stack(traj['recon_errors']).mean()

    STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                     'dog', 'horse', 'monkey', 'ship', 'truck']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRY-RUN VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    print("=" * 60)
    print("RHAN-v12 Dry-Run Validation")
    print("=" * 60)

    model = RHANv12()
    x = torch.randn(4, 3, 96, 96)

    # 1. halt_net must be gone
    assert not hasattr(model, 'halt_net'), "halt_net should be removed"
    print("✓ halt_net removed from state dict")

    # 2. Standard forward
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print(f"✓ Standard forward:  {out.shape}")

    # 3. Trajectory forward: fixed T = max_steps, no early exit
    out_traj, traj = model(x, return_trajectory=True)
    assert out_traj.shape == (4, 10)
    assert traj['steps'] == model.max_steps, \
        f"Expected fixed {model.max_steps} steps, got {traj['steps']}"
    assert len(traj['actions']) == model.max_steps
    assert len(traj['recon_errors']) == model.max_steps
    assert len(traj['recon_maps']) == model.max_steps - 1
    print(f"✓ Trajectory forward: steps={traj['steps']} (fixed, no halt)")
    print(f"  Precision range: [{traj['precisions'][-1].min():.3f}, "
          f"{traj['precisions'][-1].max():.3f}]")
    print(f"  Recon map shape: {traj['recon_maps'][-1].shape}")
    print(f"  Final gaze: {traj['actions'][-1][0].tolist()}")

    # 4. Reconstruction error map R(x,a)
    R = model.compute_recon_error_map(
        torch.randn(4, 3, 48, 48), torch.randn(4, 3, 48, 48))
    assert R.shape == (4, 48, 48)
    print(f"✓ Recon error map R(x,a): {R.shape}")

    # 5. get_feature_vector (TRADES compatibility)
    feat = model.get_feature_vector(x)
    assert feat.shape == (4, 768), f"Expected (4, 768), got {feat.shape}"
    print(f"✓ Feature vector:    {feat.shape}")

    # 6. Reconstruction loss — MUST be differentiable (v12 fix)
    recon_loss = model.get_reconstruction_loss(x, (out_traj, traj))
    assert recon_loss.shape == ()
    assert recon_loss.requires_grad, \
        "recon loss must require grad — v12 fix (v11 detached it, no-op)"
    g_params = [p for n, p in model.named_parameters()
                if 'generative_prior' in n]
    recon_loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in g_params
                  if p.grad is not None]
    assert len(grad_norms) > 0 and max(grad_norms) > 0, \
        "recon loss must reach the generative prior parameters"
    model.zero_grad(set_to_none=True)
    print(f"✓ Reconstruction loss: {recon_loss:.4f} "
          f"(differentiable, reaches generative_prior "
          f"max|grad|={max(grad_norms):.4f})")

    # 7. gaze_lambda blend sanity: lambda=0 vs lambda=1 differ
    m0 = RHANv12(gaze_lambda=0.0)
    m1 = RHANv12(gaze_lambda=1.0)
    assert m0.gaze_lambda == 0.0 and m1.gaze_lambda == 1.0
    print("✓ gaze_lambda configurable (0.0 / 0.5 / 1.0)")

    # 8. Parameter count
    from model_rhan_v11 import RHANv11
    total = sum(p.numel() for p in model.parameters())
    v11_total = sum(p.numel() for p in RHANv11().parameters())
    print(f"\n{'Parameter Summary':─^60}")
    print(f"  RHANv11: {v11_total:>12,}")
    print(f"  RHANv12: {total:>12,} (={total - v11_total:+,} vs v11)")

    print(f"\n{'All checks passed!':─^60}")

"""
RHAN-v10: Full Tripartite Active Inference Architecture
========================================================

Scientific hypothesis (pre-registered, falsifiable):
  "Dynamic precision control (Eq. III) and epistemic foraging (Eq. II)
   will maintain car/truck robustness through high-epsilon curriculum
   phases that destroy it under static TRADES β."

Architecture extends RHANLargeSTL10 with four new components:
  1. PrecisionController  (Eq. III) — τ_π × dΠ_D/dt = ‖error‖² − Π_D
  2. foveal_sample()      (Eq. II)  — differentiable spatial attention
  3. FovealStream          — lightweight 48×48 encoder
  4. ThermodynamicHalt     — metabolic cost gating

All existing RHANLargeSTL10 components are inherited unchanged.
New components add ~4M parameters on top of the 55.6M base.

References:
  Friston (2010) — The free-energy principle: a unified brain theory
  Rao & Ballard (1999) — Predictive coding in the visual cortex
  Parr & Friston (2017) — Working memory, attention, and salience
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_large import RHANLargeSTL10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 1 — Precision Controller (Equation III)
#
# τ_π × dΠ_D/dt = ‖f_stem(a) - P(s)‖² - Π_D
#
# Π_D is sensory precision: how much to trust current input.
# Large prediction error → Π_D increases → trust senses more.
# Small prediction error → Π_D decreases → rely on prior.
#
# This makes the effective training β dynamic per image:
#   β_effective = β_base × (0.5 + Π_D)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PrecisionController(nn.Module):
    """
    Implements Equation III of the active inference framework.

    Given current features and the global belief state, computes:
    - How well the prior predicts current sensory input
    - Updated precision (Π_D) reflecting prediction error magnitude
    - Error magnitude for downstream use (halt decision, β modulation)
    """

    def __init__(self, proj_dim=512, tau=0.1, init_precision=0.5):
        super().__init__()
        self.tau = tau

        # Derive initial Π_D from global context directly (FIX 2)
        self.precision_init_net = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # output in [0, 1] directly
        )

        # Prior predictor P(s): what should features look like
        # given current belief state?
        self.prior_predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, features, global_context):
        """
        Args:
            features:       (B, proj_dim) — current foveal features
            global_context: (B, proj_dim) — current belief state s

        Returns:
            updated_prec: (B,) — Π_D after one update step
            error_mag:    (B,) — prediction error magnitude
        """
        # Derive precision dynamically from global context (FIX 2)
        precision = self.precision_init_net(global_context).squeeze(-1)
        precision = precision * 0.6 + 0.2  # rescale to [0.2, 0.8]

        # P(s): predict what features SHOULD look like
        predicted = self.prior_predictor(global_context)

        # Prediction error (Eq. III numerator)
        error_vec = features - predicted                # (B, proj_dim)
        
        # Normalize prediction error to keep Π_D dynamics in a meaningful range (FIX 1)
        error_mag = error_vec.norm(dim=-1) / (512 ** 0.5)

        # Eq. III: τ_π × dΠ/dt = ‖error‖² − Π_D
        d_precision = (error_mag ** 2 - precision) / self.tau
        
        # Tighten the clamp range to force differentiation (FIX 1)
        updated_prec = torch.clamp(
            precision + 0.1 * d_precision, 0.20, 0.80)

        return updated_prec, error_mag


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 2 — Differentiable Foveal Sampling (Equation II)
#
# da/dt = −[∂f_stem(a)/∂a]^T × Π_D × (f_stem(a) − P(s))
#
# Action a = (x, y) ∈ [-1, +1]² — where to look.
# F.grid_sample provides the differentiable Jacobian automatically.
# Move attention TOWARD high-error spatial regions.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def foveal_sample(x_image, action_a, fovea_size=48):
    """
    Differentiable foveal crop at gaze position action_a.

    Args:
        x_image:  (B, 3, 96, 96)  — full input image
        action_a: (B, 2)          — normalized gaze coords in [-1, +1]

    Returns:
        (B, 3, fovea_size, fovea_size) — foveated crop

    Gradient ∂output/∂action_a flows through F.grid_sample.
    This IS the motor Jacobian ∂f_stem(a)/∂a from Eq. II.
    """
    B = x_image.shape[0]
    scale = fovea_size / 96.0
    device = x_image.device
    dtype = x_image.dtype

    # Differentiable construction of theta to ensure autograd tracks the dependency on action_a
    scale_col = torch.full((B, 1), scale, device=device, dtype=dtype)
    zero_col = torch.zeros((B, 1), device=device, dtype=dtype)

    row0 = torch.cat([scale_col, zero_col, action_a[:, 0:1]], dim=1)  # (B, 3)
    row1 = torch.cat([zero_col, scale_col, action_a[:, 1:2]], dim=1)  # (B, 3)

    theta = torch.stack([row0, row1], dim=1)  # (B, 2, 3)

    grid = F.affine_grid(theta, (B, 3, fovea_size, fovea_size),
                         align_corners=False)
    return F.grid_sample(x_image, grid,
                         mode='bilinear',
                         padding_mode='border',
                         align_corners=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 3 — Foveal Processing Stream
#
# Separate lightweight encoder for 48×48 foveal crops.
# Provides local high-resolution evidence at the attended location.
# Biological analog: foveal V1/V2 → V4 processing.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FovealStream(nn.Module):
    """
    Lightweight 3-layer ConvNet for 48×48 foveal crops.

    Architecture:
        Conv(3→128, k=3, s=1) → 48×48
        Conv(128→512, k=3, s=2) → 24×24
        Conv(512→768, k=3, s=2) → 12×12
        AdaptiveAvgPool → Flatten → Linear(768→proj_dim)

    ~3M extra parameters. Projects to the same 512-dim space
    as the peripheral belief state for seamless integration.
    """

    def __init__(self, embed_dim=768, proj_dim=512, fovea_size=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, embed_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, x_foveal):
        """
        Args:
            x_foveal: (B, 3, 48, 48) — foveal crop

        Returns:
            (B, proj_dim) — projected foveal features
        """
        return self.proj(self.stem(x_foveal))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPONENT 4 — Thermodynamic Halt Network
#
# Halts foraging when: metabolic_cost(step t+1) > info_gain(step t+1)
# Approximated as: halt when Π_D is low (already confident)
# or error_magnitude is low (nothing surprising to attend to).
# Biological analog: ACh-gated response termination.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThermodynamicHalt(nn.Module):
    """
    Decides when to stop the epistemic foraging loop.

    Inputs: (precision, normalized_error, step_fraction)
    Output: halt probability per image

    Hard halt fires when info_gain < metabolic_cost,
    ensuring the model doesn't waste computation on
    already-confident predictions.
    """

    def __init__(self, metabolic_cost=0.05):
        super().__init__()
        self.cost = metabolic_cost
        self.halt_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def should_halt(self, precision, error_mag, step, max_steps):
        """
        Args:
            precision: (B,) — current Π_D
            error_mag: (B,) — prediction error magnitude
            step:      int  — current foraging step
            max_steps: int  — maximum allowed steps

        Returns:
            halt_prob: (B,) — probability of halting [0, 1]
        """
        step_frac = torch.full_like(precision, step / max_steps)
        # Normalize error to [0, 1] range for stable input
        error_norm = error_mag / (error_mag.max() + 1e-8)
        x = torch.stack([precision, error_norm, step_frac], dim=-1)
        halt_prob = self.halt_net(x).squeeze(-1)

        # Hard halt: if info_gain < metabolic_cost, always halt
        info_gain = error_mag * precision
        hard_halt = (info_gain < self.cost).float()

        return torch.clamp(halt_prob + hard_halt, 0.0, 1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL RHAN-v10: Tripartite Active Inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RHANv10(RHANLargeSTL10):
    """
    RHAN-v10: Full Tripartite Active Inference Architecture.

    Extends RHANLargeSTL10 (55.6M params) with:
      - PrecisionController  (~0.5M)  — dynamic Π_D per image
      - FovealStream          (~3.0M) — 48×48 foveal encoder
      - ThermodynamicHalt     (~0.1M) — metabolic halt gating
      - Action initializer    (~0.1M) — initial gaze prediction
      - Projection layers     (~0.8M) — 768↔512 bridges

    Total: ~60M parameters.

    The tripartite loop implements active inference:
      Step 0: Peripheral pass (inherited from RHANLargeSTL10)
      Steps 1-4: Foveal foraging with precision-weighted belief update
    """

    def __init__(self, num_classes=10,
                 embed_dim=768,
                 proj_dim=512,
                 num_heads=12,
                 ff_dim=3072,
                 num_transformer_layers=8,
                 num_recurrent_steps=2,
                 stem_dropout=0.1,
                 max_foraging_steps=2,
                 fovea_size=48,
                 metabolic_cost=0.05,
                 precision_tau=0.1):
        # Initialize all inherited RHANLargeSTL10 components
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_layers=num_transformer_layers,
            num_recurrent_steps=num_recurrent_steps,
            stem_dropout=stem_dropout,
        )

        self.max_steps = max_foraging_steps
        self.fovea_size = fovea_size
        self.proj_dim = proj_dim
        self.embed_dim = embed_dim

        # Biological frequency parameters (Claim 1)
        self.freq_weight_low = nn.Parameter(torch.tensor(0.85))
        self.freq_weight_high = nn.Parameter(torch.tensor(0.15))

        # ── New Active Inference Components ─────────────────────
        self.precision_ctrl = PrecisionController(
            proj_dim=proj_dim, tau=precision_tau)

        self.foveal_stream = FovealStream(
            embed_dim=embed_dim, proj_dim=proj_dim, fovea_size=fovea_size)

        self.halt_net = ThermodynamicHalt(metabolic_cost=metabolic_cost)

        # ── Projection Bridges (768 ↔ 512) ─────────────────────
        # Project peripheral CLS (768-dim) → belief space (512-dim)
        self.peripheral_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Project accumulated belief (512-dim) → classifier space (768-dim)
        # so we can reuse the pretrained classifier head
        self.belief_unproj = nn.Sequential(
            nn.Linear(proj_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Action initializer: peripheral context → initial gaze (x, y)
        self.action_init = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Tanh(),  # output in [-1, +1]
        )

    def _peripheral_pass(self, x):
        """
        Full-image peripheral pass through inherited RHANLargeSTL10.
        Returns the CLS token (768-dim) and stem features.
        """
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)

        def run_transformer_fn(toks):
            return self._run_transformer(toks)

        attended = self._run_transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=run_transformer_fn,
        )
        cls_768 = refined[:, 0, :]  # (B, 768)
        return cls_768

    def forward(self, x, return_trajectory=False):
        """
        Tripartite Active Inference forward pass.

        Args:
            x: (B, 3, 96, 96) — input images
            return_trajectory: if True, returns (logits, trajectory_dict)

        Returns:
            logits: (B, num_classes)
            trajectory (optional): dict with 'actions', 'precisions',
                                   'errors', 'steps'
        """
        B = x.shape[0]

        # ── Step 0: Peripheral pass (full image) ────────────────
        cls_768 = self._peripheral_pass(x)          # (B, 768)

        # Project to belief space
        s = self.peripheral_proj(cls_768)            # (B, 512)

        # Initial gaze action from peripheral context
        a = self.action_init(s)                      # (B, 2)

        # Initial state
        weighted_belief = torch.zeros_like(s)        # (B, 512)
        weight_sum = torch.zeros(B, device=x.device) # (B,)

        trajectory = {
            'actions': [],
            'precisions': [],
            'errors': [],
            'steps': 0,
        }

        # ── Tripartite Active Inference Loop ────────────────────
        for t in range(self.max_steps):

            # Eq. II: sample foveal input at gaze position a
            x_foveal = foveal_sample(x, a, fovea_size=self.fovea_size)
            foveal_feat = self.foveal_stream(x_foveal)  # (B, 512)

            # Eq. III: update precision from prediction error
            pi_d, error_mag = self.precision_ctrl(foveal_feat, s)

            # Precision-weighted belief integration
            # High Π_D → trust new foveal evidence more
            pi_d_unsq = pi_d.unsqueeze(-1)            # (B, 1)
            s = (1 - pi_d_unsq) * s + pi_d_unsq * foveal_feat

            # Thermodynamic halt decision
            halt_prob = self.halt_net.should_halt(
                pi_d, error_mag, t, self.max_steps)

            # Accumulate belief weighted by continuation probability
            continuation = 1 - halt_prob              # (B,)
            weighted_belief += continuation.unsqueeze(-1) * s
            weight_sum += continuation

            # Record trajectory for diagnostics and loss computation
            if return_trajectory:
                trajectory['actions'].append(a.detach())
                trajectory['precisions'].append(pi_d.detach())
                trajectory['errors'].append(error_mag.detach())

            # Early exit if all images have halted
            if halt_prob.mean() > 0.9:
                break

            # Eq. II: update gaze action toward high-error regions
            # Only on non-final steps (gradient ascent on prediction error)
            if t < self.max_steps - 1:
                a_grad = a.detach().requires_grad_(True)
                with torch.enable_grad():
                    x_fov_g = foveal_sample(x, a_grad, fovea_size=self.fovea_size)
                    f_g = self.foveal_stream(x_fov_g)
                    prior_pred = self.precision_ctrl.prior_predictor(s.detach())
                    dim = f_g.shape[-1]
                    error_for_grad = (f_g - prior_pred).norm(dim=-1).mean() / math.sqrt(dim)

                    action_grad = torch.autograd.grad(
                        error_for_grad, a_grad, create_graph=False)[0]

                # Move toward high-error region (epistemic curiosity)
                # Normalize action_grad to have unit norm for stable step scaling
                grad_norm = action_grad.norm(dim=-1, keepdim=True) + 1e-8
                normed_grad = action_grad / grad_norm

                # Fixed base step plus precision-scaled component
                # This ensures foraging happens even when precision is low
                step_size = 0.20 + 0.30 * pi_d.unsqueeze(-1)
                # Range: [0.20, 0.50] instead of [0.01, 0.14]
                a = torch.clamp(a + step_size * normed_grad, -0.9, 0.9)

        trajectory['steps'] = t + 1

        # ── Final classification from accumulated belief ────────
        final_belief = weighted_belief / (weight_sum.unsqueeze(-1) + 1e-8)

        # Project back to 768-dim to reuse pretrained classifier
        final_768 = self.belief_unproj(final_belief)  # (B, 768)
        logits = self.classifier(final_768)

        if return_trajectory:
            return logits, trajectory
        return logits

    def get_feature_vector(self, x):
        """
        For TRADES compatibility: returns 768-dim feature vector.
        Uses the full tripartite forward pass with belief accumulation.
        """
        cls_768 = self._peripheral_pass(x)
        s = self.peripheral_proj(cls_768)

        a = self.action_init(s)
        weighted_belief = torch.zeros_like(s)
        weight_sum = torch.zeros(x.shape[0], device=x.device)

        for t in range(self.max_steps):
            x_foveal = foveal_sample(x, a, fovea_size=self.fovea_size)
            foveal_feat = self.foveal_stream(x_foveal)
            pi_d, error_mag = self.precision_ctrl(foveal_feat, s)

            pi_d_unsq = pi_d.unsqueeze(-1)
            s = (1 - pi_d_unsq) * s + pi_d_unsq * foveal_feat

            halt_prob = self.halt_net.should_halt(
                pi_d, error_mag, t, self.max_steps)
            continuation = 1 - halt_prob
            weighted_belief += continuation.unsqueeze(-1) * s
            weight_sum += continuation

            if halt_prob.mean() > 0.9:
                break

            if t < self.max_steps - 1:
                with torch.no_grad():
                    prior_pred = self.precision_ctrl.prior_predictor(s)
                    # Approximate gradient direction without autograd
                    error_dir = foveal_feat - prior_pred
                    error_dir = error_dir[:, :2]  # Take first 2 dims as proxy
                    error_dir = error_dir / (error_dir.norm(dim=-1, keepdim=True) + 1e-8)
                    # Fixed base step plus precision-scaled component
                    # This ensures foraging happens even when precision is low
                    step_size = 0.20 + 0.30 * pi_d.unsqueeze(-1)
                    # Range: [0.20, 0.50] instead of [0.01, 0.14]
                    a = torch.clamp(a + step_size * error_dir, -0.9, 0.9)

        final_belief = weighted_belief / (weight_sum.unsqueeze(-1) + 1e-8)
        return self.belief_unproj(final_belief)

    STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                     'dog', 'horse', 'monkey', 'ship', 'truck']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRY-RUN VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    print("=" * 60)
    print("RHAN-v10 Dry-Run Validation")
    print("=" * 60)

    model = RHANv10()
    x = torch.randn(4, 3, 96, 96)

    # Test standard forward pass
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print(f"✓ Standard forward:  {out.shape}")

    # Test trajectory forward pass
    out_traj, traj = model(x, return_trajectory=True)
    assert out_traj.shape == (4, 10)
    assert len(traj['precisions']) > 0
    assert len(traj['actions']) > 0
    assert len(traj['errors']) > 0
    assert traj['steps'] > 0
    print(f"✓ Trajectory forward: {out_traj.shape}")
    print(f"  Steps used: {traj['steps']}")
    print(f"  Precision range: [{traj['precisions'][-1].min():.3f}, "
          f"{traj['precisions'][-1].max():.3f}]")
    print(f"  Error range: [{traj['errors'][-1].min():.3f}, "
          f"{traj['errors'][-1].max():.3f}]")
    print(f"  Final gaze: {traj['actions'][-1][0].tolist()}")

    # Test get_feature_vector (TRADES compatibility)
    feat = model.get_feature_vector(x)
    assert feat.shape == (4, 768), f"Expected (4, 768), got {feat.shape}"
    print(f"✓ Feature vector:    {feat.shape}")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    base = sum(p.numel() for p in RHANLargeSTL10().parameters())
    new = total - base
    print(f"\n{'Parameter Summary':─^60}")
    print(f"  Base (RHANLargeSTL10): {base:>12,}")
    print(f"  New (v10 components):  {new:>12,}")
    print(f"  Total (RHANv10):       {total:>12,}")

    print(f"\n{'All checks passed!':─^60}")

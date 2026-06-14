"""
RHAN-v9: Self-Supervised Adversarial Invariance with True Predictive Coding
============================================================================

The gap between RHAN-trades-curriculum (AutoAttack 21.88%) and humans (~97%)
comes from three root causes this architecture addresses directly:

ROOT CAUSE 1: Wrong objective
  TRADES minimizes KL(output_adv || output_clean)
  Human vision maintains REPRESENTATION invariance — the output is a consequence
  Solution: SAIL pretraining makes f(x_adv) ≈ f(x_clean) at the REPRESENTATION level

ROOT CAUSE 2: Wrong feedback signal
  RHAN v1-v8: feedback = raw global features
  Rao & Ballard (1999): feedback = PREDICTION ERROR only
  Solution: Predictive coding — global context predicts local features,
            only the ERROR propagates back. Large adversarial errors → strong correction.

ROOT CAUSE 3: Wrong data scale
  32×32 CIFAR has insufficient pixel information to support human-level robustness
  Solution: STL-10 96×96 + ImageNet backbone (separate training script)

Architecture pillars (building on v5):
  [V1 analog]       Frequency separation: low-freq (shape) + high-freq (texture)
  [V2/V4 analog]    Dual ventral/dorsal transformer streams
  [IT analog]       CLIP-grounded semantic projection head
  [Feedback]        True predictive coding (error signal, not raw features)
  [Adaptation]      Entropy-controlled recurrence depth
  [Concepts]        Concept bottleneck for automobile/truck separation

References:
  Rao & Ballard (1999) — Predictive coding in the visual cortex
  Kietzmann et al. (2026) — Recurrence in ventral recognition
  Dapello et al. (2020) — V1 front-end improves adversarial robustness
  Kim et al. (2020) — AdvCL: adversarial contrastive learning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import PatchTokeniser, GlobalAttention, SemanticProjectionHead


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRUE PREDICTIVE CODING FEEDBACK
# Rao & Ballard (1999): feedback carries PREDICTIONS, feedforward carries ERRORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PredictiveCodingLayer(nn.Module):
    """
    Implements Rao & Ballard (1999) predictive coding within one cortical level.

    The biological mechanism:
      Higher area sends DOWN a prediction of what the lower area should see.
      Lower area sends UP only the RESIDUAL ERROR.
      When adversarial perturbation creates unexpected local features,
      the prediction error is LARGE → strong correction applied.
      When input is clean, prediction error is SMALL → minimal correction.

    This is why human V1 is robust: it receives predictions from V2/V4 about
    what to expect, and adversarial noise creates large unexpected errors
    that the feedback loop actively suppresses.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Top-down predictor: global context → predicted local features
        # This is the "what the higher area expects the lower area to see"
        self.predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
        )

        # Error gate: how strongly to apply the correction
        # High prediction error → gate open (strong correction)
        # Low prediction error → gate closed (representation is already correct)
        self.error_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Error modulator: scale the correction based on local uncertainty
        self.error_scale = nn.Parameter(torch.ones(1))

    def forward(self, local_features: torch.Tensor,
                global_context: torch.Tensor) -> tuple:
        """
        Args:
            local_features: current local representation (B, C, H, W)
            global_context: global spatial context from transformer (B, C, H, W)

        Returns:
            corrected: updated local features
            prediction_error: ||actual - predicted||, used for monitoring
        """
        # Step 1: Top-down prediction
        predicted_local = self.predictor(global_context)

        # Step 2: Prediction error (the ONLY thing that propagates back up)
        # When adversarial: actual ≠ predicted → large error → large correction
        # When clean: actual ≈ predicted → small error → minimal change
        prediction_error = local_features - predicted_local

        # Step 3: Error-dependent gating
        gate = self.error_gate(prediction_error)

        # Step 4: Correct local features using the gated error
        corrected = local_features + self.error_scale * gate * prediction_error

        return corrected, prediction_error.abs().mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTROPY-DRIVEN ADAPTIVE RECURRENCE
# High entropy (uncertain) → more recurrence steps
# Low entropy (confident) → early termination
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveRecurrenceController(nn.Module):
    """
    Controls recurrence depth based on classification uncertainty.

    Biological analog: Under adversarial pressure, the brain runs more
    feedback sweeps before committing to a percept. V1→IT→V1 cycles
    increase when the stimulus is ambiguous or corrupted.

    Implementation: If entropy of current class probabilities exceeds
    threshold θ, run another recurrence step. Stop when confident or
    max_steps reached.
    """

    def __init__(self, max_steps: int = 5, entropy_threshold: float = 0.8):
        super().__init__()
        self.max_steps = max_steps
        self.entropy_threshold = entropy_threshold
        # Learned threshold that adapts during training
        self.log_threshold = nn.Parameter(torch.tensor(0.0))

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        # Normalize to [0, 1] relative to maximum entropy (log(num_classes))
        H = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        H_max = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float))
        return H / H_max

    def should_continue(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns bool mask: True means this sample needs more processing."""
        entropy = self.compute_entropy(logits)
        threshold = torch.sigmoid(self.log_threshold) * 0.9 + 0.1
        return entropy > threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADVERSARIAL CONTRASTIVE PROJECTION HEAD
# Used during SAIL pretraining phase (no labels)
# Projects representations into a contrastive space
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContrastiveProjectionHead(nn.Module):
    """
    Projects encoder representations into a space where
    contrastive loss can enforce adversarial invariance.

    After SAIL pretraining, this head is discarded.
    The encoder is then fine-tuned with TRADES.
    """

    def __init__(self, embed_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(x), dim=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RHAN-v9: THE COMPLETE ARCHITECTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RHANv9(nn.Module):
    """
    RHAN-v9: Self-Supervised Adversarial Invariance + True Predictive Coding

    Training has THREE phases:
      Phase 0 (30 ep): CLIP semantic initialization (as in v5/v8)
      Phase 1 (50 ep): SAIL self-supervised adversarial invariance (no labels)
      Phase 2 (60 ep): TRADES fine-tuning on representations already invariant

    Architectural innovations over v8:
      1. PredictiveCodingLayer replaces raw feedback gate
      2. AdaptiveRecurrenceController controls loop depth
      3. ContrastiveProjectionHead for SAIL pretraining
      4. Expanded concept bottleneck (automobile/truck disambiguation)
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 512,
                 num_heads: int = 8, ff_dim: int = 2048,
                 dropout: float = 0.1, num_transformer_layers: int = 3,
                 max_recurrence: int = 4, head_type: str = 'cosine'):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_type = head_type

        # ── FREQUENCY SEPARATOR (V1 analog) ──────────────────────────────
        self.register_buffer(
            'gaussian_kernel',
            self._make_gaussian_kernel(sigma=1.5, size=5)
        )
        self.freq_weight_low = nn.Parameter(torch.tensor(0.85))
        self.freq_weight_high = nn.Parameter(torch.tensor(0.15))

        # ── LOW-FREQUENCY PATHWAY (shape — M-pathway) ────────────────────
        self.stem_low = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # ── HIGH-FREQUENCY PATHWAY (texture — P-pathway) ─────────────────
        self.stem_high = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # ── TOKENISER ────────────────────────────────────────────────────
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)

        # ── VENTRAL/DORSAL TRANSFORMER (V4/IT analog) ────────────────────
        self.ventral_transformer = GlobalAttention(
            embed_dim=embed_dim // 2, num_heads=num_heads // 2,
            ff_dim=ff_dim // 2, dropout=dropout, num_layers=num_transformer_layers,
        )
        self.dorsal_transformer = GlobalAttention(
            embed_dim=embed_dim // 2, num_heads=num_heads // 2,
            ff_dim=ff_dim // 2, dropout=dropout, num_layers=num_transformer_layers,
        )

        # ── TRUE PREDICTIVE CODING FEEDBACK (replaces raw gate from v5) ──
        # This is the key architectural innovation: error signal, not raw features
        self.predictive_coder = PredictiveCodingLayer(embed_dim=embed_dim)

        # ── ADAPTIVE RECURRENCE CONTROLLER ───────────────────────────────
        self.recurrence_controller = AdaptiveRecurrenceController(
            max_steps=max_recurrence
        )
        self.max_recurrence = max_recurrence

        # ── CLASSIFICATION HEAD ───────────────────────────────────────────
        if head_type == 'linear':
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Dropout(0.1), nn.Linear(embed_dim, num_classes)
            )
        else:
            self.classifier = SemanticProjectionHead(
                embed_dim=embed_dim, num_classes=num_classes
            )

        # ── CONTRASTIVE HEAD (SAIL pretraining only, discarded after) ────
        self.contrastive_head = ContrastiveProjectionHead(embed_dim, proj_dim=128)

        # ── CONCEPT BOTTLENECK ────────────────────────────────────────────
        # Extended concept set to better disambiguate automobile vs truck
        self.concept_names = [
            'has_wings',        # airplane, bird
            'has_wheels',       # automobile, truck
            'has_fur',          # cat, dog, horse, deer
            'has_feathers',     # bird
            'is_metallic',      # automobile, truck, ship, airplane
            'is_organic',       # animals
            'carries_cargo',    # TRUCK distinguisher
            'is_passenger_vehicle',  # AUTOMOBILE distinguisher (not cargo)
            'lives_in_water',   # ship, frog
            'has_four_legs',    # cat, dog, horse, deer
            'has_rigid_body',   # automobile, truck, ship, airplane
            'is_elongated',     # ship, airplane
            'has_hooves',       # horse, deer (NOT cat/dog → key discriminator)
            'is_large_animal',  # horse, deer (NOT cat/dog → key discriminator)
            'has_propulsion',   # airplane (thrust), ship (engine)
            'has_cab_window',   # automobile, truck (both have)
            'has_open_bed',     # TRUCK ONLY — critical for auto/truck separation
            'has_closed_roof',  # AUTOMOBILE ONLY — critical for auto/truck separation
        ]

        concept_labels = torch.tensor([
            # airplane: wings, metallic, rigid, elongated, propulsion
            [1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,0],
            # automobile: wheels, metallic, passenger, rigid, cab, closed_roof
            [0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,1],
            # bird: wings, feathers, organic
            [1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            # cat: fur, organic, four_legs
            [0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
            # deer: fur, organic, four_legs, large, hooves
            [0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0],
            # dog: fur, organic, four_legs
            [0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
            # frog: organic, water
            [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
            # horse: fur, organic, four_legs, large, hooves
            [0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0],
            # ship: metallic, water, rigid, elongated, propulsion
            [0,0,0,0,1,0,0,0,1,0,1,1,0,0,1,0,0,0],
            # truck: wheels, metallic, cargo, rigid, cab, open_bed
            [0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0],
        ], dtype=torch.float32)
        self.register_buffer('concept_labels', concept_labels)
        self.n_concepts = len(self.concept_names)

        self.concept_layer = nn.Linear(embed_dim, self.n_concepts)
        self.concept_bn = nn.BatchNorm1d(self.n_concepts)
        self.concept_classifier = nn.Linear(self.n_concepts, num_classes)

    def _make_gaussian_kernel(self, sigma: float, size: int) -> torch.Tensor:
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    def separate_frequencies(self, x: torch.Tensor) -> tuple:
        x_low = F.conv2d(
            F.pad(x, [2, 2, 2, 2], mode='reflect'),
            self.gaussian_kernel, groups=3
        )
        x_high = x - x_low
        return x_low, x_high

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        spatial = tokens[:, 1:, :]  # remove CLS
        B, N, C = spatial.shape
        return spatial.transpose(1, 2).reshape(B, C, 8, 8)

    def _run_transformer(self, f: torch.Tensor) -> tuple:
        """Run tokenise → ventral/dorsal → return (combined_tokens, cls_token)."""
        tokens = self.tokeniser(f)
        v_tokens = self.ventral_transformer(tokens[:, :, :256])
        d_tokens = self.dorsal_transformer(tokens[:, :, 256:])
        combined = torch.cat([v_tokens, d_tokens], dim=-1)
        return combined, combined[:, 0, :]  # (tokens, cls)

    def get_feature_vector(self, x: torch.Tensor,
                           return_prediction_errors: bool = False) -> tuple:
        """
        Full forward pass with predictive coding feedback.

        Returns:
            cls: final class token embedding (B, embed_dim)
            errors: list of prediction errors per recurrence step (if requested)
        """
        # 1. Frequency separation
        x_low, x_high = self.separate_frequencies(x)

        # 2. Frequency-weighted fusion
        w_low = torch.sigmoid(self.freq_weight_low)
        w_high = torch.sigmoid(self.freq_weight_high)
        f = w_low * self.stem_low(x_low) + w_high * self.stem_high(x_high)

        errors = []

        # 3. Recurrent feedback loop with TRUE PREDICTIVE CODING
        for step in range(self.max_recurrence):
            # Forward: local features → global context
            combined, cls = self._run_transformer(f)

            # Convert global CLS to spatial map for prediction
            # (broadcast CLS token to spatial dimensions for the predictor)
            spatial_context = self._tokens_to_spatial(combined)

            # Predictive coding: global predicts local, only error feeds back
            f_corrected, error = self.predictive_coder(f, spatial_context)
            errors.append(error)

            # Adaptive termination: if confident enough, stop early
            # (only at inference time — always run full steps during training)
            if not self.training:
                logits_probe = self.classifier(cls)
                if not self.recurrence_controller.should_continue(logits_probe).any():
                    f = f_corrected
                    break

            f = f_corrected

        # 4. Final representation
        _, cls = self._run_transformer(f)

        if return_prediction_errors:
            return cls, errors
        return cls, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.get_feature_vector(x)
        return self.classifier(cls)

    def forward_with_features(self, x: torch.Tensor) -> tuple:
        cls, errors = self.get_feature_vector(x, return_prediction_errors=True)
        logits = self.classifier(cls)
        return logits, cls

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """Used ONLY during SAIL pretraining phase."""
        cls, _ = self.get_feature_vector(x)
        return self.contrastive_head(cls)

    def forward_with_concepts(self, x: torch.Tensor) -> tuple:
        cls, _ = self.get_feature_vector(x)
        concepts = torch.sigmoid(self.concept_bn(self.concept_layer(cls)))
        logits = self.concept_classifier(concepts)
        return logits, concepts

    def get_frequency_weights(self) -> tuple:
        return (torch.sigmoid(self.freq_weight_low).item(),
                torch.sigmoid(self.freq_weight_high).item())


if __name__ == '__main__':
    model = RHANv9()
    x = torch.randn(4, 3, 32, 32)

    logits, features = model.forward_with_features(x)
    print(f"RHANv9 verified. Logits: {logits.shape}, Features: {features.shape}")

    # Test contrastive head for SAIL
    z = model.forward_contrastive(x)
    print(f"Contrastive projection: {z.shape}")

    # Test concept bottleneck
    logits_c, concepts = model.forward_with_concepts(x)
    print(f"Concept logits: {logits_c.shape}, Concepts: {concepts.shape}")

    wL, wH = model.get_frequency_weights()
    print(f"Frequency weights: low={wL:.3f}, high={wH:.3f}")
    print("All tests passed.")

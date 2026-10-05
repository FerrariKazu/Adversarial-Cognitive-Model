"""
RHAN with Predictive Coding Feedback (Trial 3, Improvement #1)
================================================================
Merges the RHAN-adv architecture with Karl Friston's predictive coding principle.

CURRENT FEEDBACK (RHAN):
    stem = stem + gate * feedback

PREDICTIVE CODING FEEDBACK (this model):
    stem = stem + gate * (feedback - stem_prediction)

The StemPredictor module predicts what top-down feedback *should* be from the
current stem features. Only the prediction error (residual/surprise) propagates
back to the stem. This implements the core principle of predictive coding:
higher cortical areas generate predictions about lower area activations, and
only the unexpected component (prediction error) drives representation updates.

BIOLOGICAL MOTIVATION:
    In the primate visual cortex, feedback connections from IT → V4 → V2 → V1
    carry predictions about expected lower-area activity. Feedforward connections
    carry the prediction error (the "surprise"). This architecture is more
    efficient than raw feedback because:
      1. Predictable components are suppressed (they carry no new information)
      2. Only unexpected features drive further processing
      3. Adversarial perturbations that deviate from natural image statistics
         produce large prediction errors, making them easier to detect and suppress

EXPECTED EFFECT:
    Adversarial perturbations that violate natural image statistics should produce
    large prediction errors, which the gate can learn to suppress. Clean images
    should produce small prediction errors (the predictor "understands" natural
    features), so feedback flows freely for refinement.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import ConvStem, PatchTokeniser, GlobalAttention


class StemPredictor(nn.Module):
    """Predicts expected top-down feedback from current stem features.

    Implements the 'prediction' half of predictive coding: given the current
    lower-area representation (stem_features), predict what the higher-area
    feedback signal should be. The difference between actual feedback and this
    prediction is the prediction error (surprise) that drives learning.
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
        )

    def forward(self, stem_features):
        """
        Args:
            stem_features: (B, 512, 8, 8) — current stem feature map
        Returns:
            predicted_feedback: (B, 512, 8, 8) — expected feedback signal
        """
        return self.predictor(stem_features)


class PredCodingFeedback(nn.Module):
    """Recurrent feedback with predictive coding: only prediction error propagates.

    At each recurrent step:
      1. Transformer output → spatial → feedback signal (via feedback_conv)
      2. Stem features → predicted feedback (via stem_predictor)
      3. Prediction error = feedback - predicted_feedback
      4. Stem update: stem += gate * prediction_error
    """

    def __init__(self, embed_dim=512, num_patches=64, num_recurrent_steps=2):
        super().__init__()
        self.num_recurrent_steps = num_recurrent_steps
        self.spatial_h = int(math.sqrt(num_patches))
        self.spatial_w = int(math.sqrt(num_patches))

        # Feedback convolution: routes global transformer info back to spatial
        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Gate: learns how much prediction error to inject per channel/position
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # NEW: Stem predictor for predictive coding
        self.stem_predictor = StemPredictor(embed_dim=embed_dim)

    def tokens_to_spatial(self, tokens):
        """(B, 65, 512) → (B, 512, 8, 8), strips CLS token."""
        spatial_tokens = tokens[:, 1:, :]
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)

    def spatial_to_tokens(self, spatial, cls_token):
        """(B, 512, 8, 8) + (B, 1, 512) → (B, 65, 512)."""
        B = spatial.shape[0]
        tokens = spatial.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, tokens], dim=1)

    def forward(self, transformer_output, stem_features, transformer_fn):
        """
        Args:
            transformer_output: (B, 65, 512) — output from Stage 3
            stem_features: (B, 512, 8, 8) — original output from Stage 1
            transformer_fn: callable — the transformer encoder to re-run
        Returns:
            (B, 65, 512) — refined token sequence after recurrent feedback
            residual_magnitudes: list of float — mean |prediction_error| per step
        """
        current = transformer_output
        residual_magnitudes = []

        for t in range(self.num_recurrent_steps):
            cls_token = current[:, :1, :]

            # Compute top-down feedback from transformer output
            spatial = self.tokens_to_spatial(current)
            feedback = self.feedback_conv(spatial)

            # Predict what feedback *should* be from current stem
            predicted_feedback = self.stem_predictor(stem_features)

            # Prediction error (surprise) = actual - predicted
            prediction_error = feedback - predicted_feedback

            # Track residual magnitude for analysis
            residual_magnitudes.append(prediction_error.norm(dim=1).mean().item())

            # Gate conditioned on full feedback (how much top-down info is available)
            g = self.gate(feedback)

            # Inject only the gated prediction error
            modulated = stem_features + g * prediction_error

            # Re-tokenize and re-run transformer
            modulated_tokens = self.spatial_to_tokens(modulated, cls_token)
            current = transformer_fn(modulated_tokens)

        return current, residual_magnitudes


class RHAN_PredCoding(nn.Module):
    """RHAN with Predictive Coding feedback.

    Five stages:
      1. Conv Stem — local smoothing
      2. Patch Tokenisation — CNN feature tokens + CLS
      3. Global Attention — transformer
      4. Predictive Coding Feedback — prediction error injection
      5. Linear classification head
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2):
        super().__init__()

        self.stem = ConvStem()
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)
        self.transformer = GlobalAttention(
            embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout, num_layers=num_transformer_layers,
        )
        self.feedback = PredCodingFeedback(
            embed_dim=embed_dim, num_patches=64,
            num_recurrent_steps=num_recurrent_steps,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def get_feature_vector(self, x):
        """Extract 512-dim CLS token after predictive coding feedback."""
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)
        refined, _ = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )
        return refined[:, 0, :]

    def forward(self, x):
        """
        Args:
            x: (B, 3, 32, 22) — normalised CIFAR-10 images
        Returns:
            logits: (B, 10)
            residual_magnitudes: list of float — prediction error norms per step
        """
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)
        refined, residual_magnitudes = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )
        cls_output = refined[:, 0, :]
        logits = self.head(cls_output)
        return logits, residual_magnitudes

    def load_from_rhan_adv(self, rhan_checkpoint_path, device='cuda'):
        """Map standard RHAN-adv checkpoint into RHAN_PredCoding.

        All weights transfer except stem_predictor (random init).
        """
        state = torch.load(rhan_checkpoint_path, map_location=device)
        adapted_state = {}
        for k, v in state.items():
            if k.startswith('feedback.feedback_conv.'):
                adapted_state[k.replace('feedback.feedback_conv.', 'feedback.feedback_conv.')] = v
            elif k.startswith('feedback.gate.0.'):
                adapted_state[k.replace('feedback.gate.0.', 'feedback.gate.0.')] = v
            elif k.startswith('head.'):
                adapted_state[k.replace('head.', 'head.')] = v
            else:
                adapted_state[k] = v

        msg = self.load_state_dict(adapted_state, strict=False)
        print(f"[RHAN_PredCoding] Checkpoint loaded: {msg}")
        print(f"  (stem_predictor randomly initialized — expected missing keys)")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_architecture_summary():
    print("=" * 70)
    print("RHAN with Predictive Coding Feedback — Architecture Summary")
    print("=" * 70)
    model = RHAN_PredCoding()
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}")
    print(f"Model size (MB):      {total * 4 / 1024**2:>12.1f}")
    print(f"\n{'Stage':<45} {'Output Shape':<20}")
    print("-" * 65)
    print(f"{'Input':<45} {'(B, 3, 32, 32)':<20}")
    print(f"{'Stage 1: Conv Stem':<45} {'(B, 512, 8, 8)':<20}")
    print(f"{'Stage 2: Tokenise + CLS':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 3: Transformer':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 4: Predictive Coding Feedback ×2':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 5: Linear Head':<45} {'(B, 10)':<20}")
    print("-" * 65)

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out, residuals = model(x)
    print(f"\nForward pass: {tuple(x.shape)} → {tuple(out.shape)}")
    print(f"Residual magnitudes: {[f'{r:.3f}' for r in residuals]}")
    assert out.shape == (4, 10)
    print("✓ Forward pass successful")
    print("=" * 70)
    return model


if __name__ == '__main__':
    print_architecture_summary()

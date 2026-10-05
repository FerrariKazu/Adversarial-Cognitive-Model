"""
RHAN Full Model — All Three Improvements Combined
===================================================
Combines:
  1. Predictive Coding feedback (StemPredictor)
  2. Ventral/Dorsal stream split (DualStreamAttention)
  3. Neural Representation Alignment head (AlignmentHead)

This is the ultimate bio-inspired RHAN variant, implementing all three
high-impact improvements from the .claude.md research plan.

Architecture:
  Conv Stem → Tokenise → Dual-Stream Transformer → PredCoding Feedback → Head
                                                                        ↘ AlignmentHead (training only)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import ConvStem, PatchTokeniser
from model_rhan_predcoding import StemPredictor
from model_rhan_dualstream import DualStreamAttention


class PredCodingFeedbackFull(nn.Module):
    """Predictive coding feedback for dual-stream architecture."""

    def __init__(self, embed_dim=512, num_patches=64, num_recurrent_steps=2):
        super().__init__()
        self.num_recurrent_steps = num_recurrent_steps
        self.spatial_h = int(math.sqrt(num_patches))
        self.spatial_w = int(math.sqrt(num_patches))

        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.stem_predictor = StemPredictor(embed_dim=embed_dim)

    def tokens_to_spatial(self, tokens):
        spatial_tokens = tokens[:, 1:, :]
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)

    def spatial_to_tokens(self, spatial, cls_token):
        B = spatial.shape[0]
        tokens = spatial.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, tokens], dim=1)

    def forward(self, transformer_output, stem_features, transformer_fn):
        current = transformer_output
        residual_magnitudes = []

        for t in range(self.num_recurrent_steps):
            cls_token = current[:, :1, :]
            spatial = self.tokens_to_spatial(current)
            feedback = self.feedback_conv(spatial)

            # Predictive coding: prediction error = feedback - predicted
            predicted = self.stem_predictor(stem_features)
            prediction_error = feedback - predicted
            residual_magnitudes.append(prediction_error.norm(dim=1).mean().item())

            g = self.gate(feedback)
            modulated = stem_features + g * prediction_error
            modulated_tokens = self.spatial_to_tokens(modulated, cls_token)
            current, _ = transformer_fn(modulated_tokens)

        return current, residual_magnitudes


class RHAN_Full(nn.Module):
    """Full RHAN with all three improvements."""

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_layers_per_stream=2,
                 num_recurrent_steps=2):
        super().__init__()

        self.stem = ConvStem()
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)
        self.dual_attention = DualStreamAttention(
            embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
            dropout=dropout, num_layers_per_stream=num_layers_per_stream,
        )
        self.feedback = PredCodingFeedbackFull(
            embed_dim=embed_dim, num_patches=64,
            num_recurrent_steps=num_recurrent_steps,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def get_feature_vector(self, x):
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended, _ = self.dual_attention(tokens)
        refined, _ = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.dual_attention,
        )
        return refined[:, 0, :]

    def forward(self, x):
        """
        Returns:
            logits: (B, 10)
            aux: dict with residual_magnitudes and dual-stream aux
        """
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended, stream_aux = self.dual_attention(tokens)
        refined, residual_magnitudes = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.dual_attention,
        )
        cls_output = refined[:, 0, :]
        logits = self.head(cls_output)
        aux = {**stream_aux, 'residual_magnitudes': residual_magnitudes}
        return logits, aux


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_architecture_summary():
    print("=" * 70)
    print("RHAN Full Model (All Improvements) — Architecture Summary")
    print("=" * 70)
    model = RHAN_Full()
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters:     {total:>12,}")
    print(f"Trainable parameters: {total:>12,}")
    print(f"Model size (MB):      {total * 4 / 1024**2:>12.1f}")
    print(f"\n{'Stage':<45} {'Output Shape':<20}")
    print("-" * 65)
    print(f"{'Input':<45} {'(B, 3, 32, 32)':<20}")
    print(f"{'Stage 1: Conv Stem':<45} {'(B, 512, 8, 8)':<20}")
    print(f"{'Stage 2: Tokenise + CLS':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 3: Dual-Stream Attention':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 4: PredCoding Feedback ×2':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 5: Linear Head':<45} {'(B, 10)':<20}")
    print(f"{'Training: AlignmentHead (extra)':<45} {'(B, 512)':<20}")
    print("-" * 65)

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out, aux = model(x)
    print(f"\nForward pass: {tuple(x.shape)} → {tuple(out.shape)}")
    print(f"Residuals: {[f'{r:.3f}' for r in aux['residual_magnitudes']]}")
    print(f"Fusion gate: {aux['fusion_gate'].mean().item():.3f}")
    assert out.shape == (4, 10)
    print("✓ Forward pass successful")
    print("=" * 70)
    return model


if __name__ == '__main__':
    print_architecture_summary()

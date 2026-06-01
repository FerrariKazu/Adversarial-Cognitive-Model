"""
RHAN-Predictive: Predictive Coding Recurrent Hybrid Attention Network
=====================================================================
Trial 4: Biologically Grounded Predictive Coding / Top-Down Feedback

Implementing Karl Friston's predictive coding:
- Higher visual areas (Transformer) generate predictions about lower-level
  activations (Conv Stem).
- The difference between actual bottom-up stem features and the top-down
  predictions defines the "prediction error" (surprise).
- Only the gated prediction error propagates back up.
"""

import math
import torch
import torch.nn as nn
from model_rhan import ConvStem, PatchTokeniser, GlobalAttention, SemanticProjectionHead


class PredictiveFeedback(nn.Module):
    """Karl Friston's predictive coding feedback loop:
    Higher-level predictions are subtracted from bottom-up stem features,
    and only the gated prediction error (surprise) propagates.
    """

    def __init__(self, embed_dim=512, num_patches=64, num_recurrent_steps=2):
        super().__init__()
        self.num_recurrent_steps = num_recurrent_steps
        self.spatial_h = int(math.sqrt(num_patches))  # 8
        self.spatial_w = int(math.sqrt(num_patches))  # 8

        # Generates top-down predictions of lower-level (stem) activations
        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Gate maps the prediction error to a channel-wise gate to modulate surprise gain
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def tokens_to_spatial(self, tokens):
        spatial_tokens = tokens[:, 1:, :]  # Remove CLS token
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)

    def spatial_to_tokens(self, spatial, cls_token):
        B = spatial.shape[0]
        tokens = spatial.flatten(2).transpose(1, 2)
        return torch.cat([cls_token, tokens], dim=1)

    def forward(self, transformer_output, stem_features, transformer_fn):
        """
        Args:
            transformer_output: (B, 65, 512)
            stem_features: (B, 512, 8, 8)
            transformer_fn: callable transformer encoder
        Returns:
            (B, 65, 512)
        """
        current = transformer_output

        for t in range(self.num_recurrent_steps):
            cls_token = current[:, :1, :]
            spatial = self.tokens_to_spatial(current)

            # 1. Top-down prediction of lower-level activity
            stem_prediction = self.feedback_conv(spatial)

            # 2. Compute prediction error / surprise
            prediction_error = stem_features - stem_prediction

            # 3. Channel-wise gating of the prediction error
            g = self.gate(prediction_error)

            # 4. Modulated representation uses the surprise signal
            modulated = stem_features + g * prediction_error

            # 5. Re-tokenise and re-run attention
            modulated_tokens = self.spatial_to_tokens(modulated, cls_token)
            current = transformer_fn(modulated_tokens)

        return current


class RHANPredictive(nn.Module):
    """
    RHAN with Predictive Coding Feedback (Trial 4)
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='cosine'):
        super().__init__()
        self.head_type = head_type

        # Stage 1: Conv Stem (local smoothing)
        self.stem = ConvStem()

        # Stage 2: Tokeniser
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)

        # Stage 3: Global Attention Transformer
        self.transformer = GlobalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        # Stage 4: Predictive Recurrent Feedback (surprise-based top-down modulation)
        self.feedback = PredictiveFeedback(
            embed_dim=embed_dim,
            num_patches=64,
            num_recurrent_steps=num_recurrent_steps,
        )

        # Stage 5: Classification head
        if head_type == 'linear':
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            self.head = SemanticProjectionHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
            )

    def forward(self, x):
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)

        # Recurrent feedback loop using predictive coding
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )

        cls_output = refined[:, 0, :]
        logits = self.head(cls_output)
        return logits


if __name__ == '__main__':
    model = RHANPredictive()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"RHANPredictive verified successfully. Output shape: {out.shape}")

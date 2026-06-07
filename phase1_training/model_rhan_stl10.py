"""
RHAN for STL-10: 96x96 input adaptation of the proven v5 architecture.

Key changes from CIFAR-10 model (model_rhan.py):
  1. Stem: 4 conv layers instead of 3
     96×96 → 96×96 → 48×48 → 24×24 → 12×12
     Output: (B, 512, 12, 12) = 144 spatial tokens (vs 64 at 8×8)
  2. Gaussian kernel sigma: 3.0 (vs 1.5) for appropriate freq cutoff at 96px
  3. Positional embeddings: (1, 145, 512) for 144 patches + CLS (vs 65)
  4. Tokeniser: num_patches=144 (vs 64)

Everything else identical to model_rhan.py (v5).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_rhan import PatchTokeniser, GlobalAttention, SemanticProjectionHead


# =============================================================================
# STL-10 CONVOLUTIONAL STEM — 4 layers for 96×96 → 12×12
# =============================================================================
# Layer 1: 96×96 → 96×96 (stride=1, preserves resolution)
# Layer 2: 96×96 → 48×48 (stride=2, first downsample)
# Layer 3: 48×48 → 24×24 (stride=2, second downsample)
# Layer 4: 24×24 → 12×12 (stride=2, third downsample)  ← NEW for STL-10
# Output: (B, 512, 12, 12) = 144 spatial positions
# =============================================================================

class ConvStemSTL10(nn.Module):
    """Four-layer convolutional stem for 96×96 STL-10 images."""

    def __init__(self):
        super().__init__()
        # Layer 1: 96×96 → 96×96
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Layer 2: 96×96 → 48×48
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Layer 3: 48×48 → 24×24
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Layer 4: 24×24 → 12×12  ← NEW for STL-10
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Residual shortcut: 3→512, stride=8 total (1×2×2×2)
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=1, stride=8, bias=False),
            nn.BatchNorm2d(512),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 96, 96) — raw STL-10 images (normalised)
        Returns:
            (B, 512, 12, 12) — locally smoothed feature map
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out + identity


# =============================================================================
# RECURRENT FEEDBACK — updated for 12×12 spatial (144 tokens)
# =============================================================================

class RecurrentFeedbackSTL10(nn.Module):
    """Recurrent top-down feedback loop for 12×12 spatial features."""

    def __init__(self, embed_dim=512, num_recurrent_steps=2):
        super().__init__()
        self.num_recurrent_steps = num_recurrent_steps
        self.spatial_h = 12
        self.spatial_w = 12

        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

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
        for t in range(self.num_recurrent_steps):
            cls_token = current[:, :1, :]
            spatial = self.tokens_to_spatial(current)
            feedback = self.feedback_conv(spatial)
            g = self.gate(feedback)
            modulated = stem_features + g * feedback
            modulated_tokens = self.spatial_to_tokens(modulated, cls_token)
            current = transformer_fn(modulated_tokens)
        return current


# =============================================================================
# FULL RHAN-STL10 MODEL
# =============================================================================

class RHANSTL10(nn.Module):
    """
    RHAN for STL-10: 96×96 input, 144 spatial tokens, sigma=3.0 gaussian.
    Based on proven v5 architecture.
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='cosine'):
        super().__init__()
        self.head_type = head_type

        # Stage 1: Conv stem — 96×96 → 12×12
        self.stem = ConvStemSTL10()

        # Stage 2: Tokenise 12×12 → 144 tokens + CLS = 145
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=144)

        # Stage 3: Global attention transformer
        self.transformer = GlobalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        # Stage 4: Recurrent feedback for 12×12
        self.feedback = RecurrentFeedbackSTL10(
            embed_dim=embed_dim,
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

    def get_feature_vector(self, x):
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )
        return refined[:, 0, :]

    def forward(self, x):
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )
        cls_output = refined[:, 0, :]
        logits = self.head(cls_output)
        return logits


if __name__ == '__main__':
    model = RHANSTL10()
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"
    print("STL-10 RHAN forward pass: OK")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

"""
RHAN-Split: Ventral/Dorsal Stream Split Recurrent Hybrid Attention Network
==========================================================================
Trial 3: Ventral/Dorsal Stream Split

Splits the visual attention pathways into parallel Ventral ("What") and Dorsal
("Where") visual streams to process object identity and spatial features
separately, before fusing them and feeding them to the recurrent feedback loop.
"""

import math
import torch
import torch.nn as nn
from model_rhan import ConvStem, PatchTokeniser, GlobalAttention, RecurrentFeedback, SemanticProjectionHead


class VentralDorsalAttention(nn.Module):
    """
    Splits the 512-dimensional embedding dimension into parallel
    256-dimensional Ventral ("what") and Dorsal ("where") attention pathways.
    """

    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, num_layers=3):
        super().__init__()
        self.split_dim = embed_dim // 2  # 256

        # Ventral Stream: dedicated to identity, color, texture
        self.ventral_transformer = GlobalAttention(
            embed_dim=self.split_dim,
            num_heads=num_heads // 2,  # 4 heads
            ff_dim=ff_dim // 2,        # 1024 ff_dim
            dropout=dropout,
            num_layers=num_layers
        )

        # Dorsal Stream: dedicated to spatial relationship, layout, motion-like geometry
        self.dorsal_transformer = GlobalAttention(
            embed_dim=self.split_dim,
            num_heads=num_heads // 2,  # 4 heads
            ff_dim=ff_dim // 2,        # 1024 ff_dim
            dropout=dropout,
            num_layers=num_layers
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, 65, 512)
        Returns:
            (B, 65, 512)
        """
        # Split tokens: (B, 65, 512) -> (B, 65, 256) and (B, 65, 256)
        ventral_tokens = tokens[:, :, :self.split_dim]
        dorsal_tokens = tokens[:, :, self.split_dim:]

        # Run through parallel pathways
        ventral_out = self.ventral_transformer(ventral_tokens)
        dorsal_out = self.dorsal_transformer(dorsal_tokens)

        # Fuse (concatenate) along channel dimension back to (B, 65, 512)
        fused = torch.cat([ventral_out, dorsal_out], dim=-1)
        return fused


class RHANSplit(nn.Module):
    """
    RHAN model implementing a Ventral/Dorsal attention split (Trial 3).
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

        # Stage 3: Split Ventral/Dorsal Attention Pathways
        self.transformer = VentralDorsalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        # Stage 4: Recurrent Feedback
        self.feedback = RecurrentFeedback(
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

    def get_feature_vector(self, x):
        """
        Extract the 512-dimensional CLS token feature vector after recurrent feedback.
        Used for alignment loss and consistency loss in RHAN-v2 training.
        """
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )
        return refined[:, 0, :]  # (B, 512) — CLS token

    def forward(self, x):
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended = self.transformer(tokens)

        # Recurrent feedback loop runs the split transformer recurrently
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )

        cls_output = refined[:, 0, :]
        logits = self.head(cls_output)
        return logits


if __name__ == '__main__':
    model = RHANSplit()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"RHANSplit verified successfully. Output shape: {out.shape}")

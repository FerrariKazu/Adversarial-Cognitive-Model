"""
RHAN-Aligned: Neural Representation Alignment Recurrent Hybrid Attention Network
================================================================================
Trial 8: Neural Representation Alignment

This model extends the standard RHAN to provide direct access to the 512-dimensional
projected CLS token features, which are normalized and aligned against primate IT
visual representations (proxied by a pre-trained CORnet-S model).
"""

import torch
import torch.nn as nn
from model_rhan import RHAN


class RHANAligned(RHAN):
    """
    RHAN model with explicit features retrieval for representation alignment.
    Inherits from the base RHAN model to maintain structural consistency.
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='cosine'):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_transformer_layers=num_transformer_layers,
            num_recurrent_steps=num_recurrent_steps,
            head_type=head_type
        )

    def forward_with_features(self, x):
        """
        Executes forward pass and returns both the logits and the normalized 512-dim features.
        
        Args:
            x: (B, 3, 32, 32)
        Returns:
            logits: (B, 10)
            normalized_features: (B, 512)
        """
        # 1. Conv Stem
        stem_features = self.stem(x)
        
        # 2. Tokenise
        tokens = self.tokeniser(stem_features)
        
        # 3. Transformer Attention
        attended = self.transformer(tokens)
        
        # 4. Recurrent Feedback
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )
        
        # Extract CLS token
        cls_output = refined[:, 0, :]  # (B, 512)
        
        # 5. Project and normalize features
        if self.head_type == 'cosine':
            features = self.head.projection(cls_output)
            normalized_features = nn.functional.normalize(features, dim=-1)
            
            # Normalise class prototypes to unit sphere
            prototypes = nn.functional.normalize(self.head.class_prototypes, dim=-1)
            
            # Cosine similarity × temperature -> logits
            temperature = self.head.logit_scale.exp().clamp(max=100.0)
            logits = temperature * (normalized_features @ prototypes.T)
        else:
            normalized_features = nn.functional.normalize(cls_output, dim=-1)
            logits = self.head(cls_output)
            
        return logits, normalized_features


if __name__ == '__main__':
    model = RHANAligned()
    x = torch.randn(2, 3, 32, 32)
    logits, feats = model.forward_with_features(x)
    print(f"RHANAligned verified successfully.")
    print(f"  Logits shape:   {logits.shape}")
    print(f"  Features shape: {feats.shape}")

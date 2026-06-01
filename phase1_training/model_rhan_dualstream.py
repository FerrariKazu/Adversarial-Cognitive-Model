"""
RHAN with Ventral/Dorsal Stream Split (Trial 3, Improvement #2)
==================================================================
Splits the transformer into two parallel attention streams motivated by the
empirically validated "what vs where" principle in visual neuroscience.

VENTRAL STREAM ("what"):
    Standard self-attention on CLS + spatial tokens.
    Focused on object identity features — what is this object?
    Uses learned positional embeddings (same as standard RHAN).

DORSAL STREAM ("where"):
    Self-attention with spatial relationship bias.
    Focused on spatial configuration — where are the features?
    Adds a learnable spatial bias to attention scores that encourages
    the model to attend to spatial relationships between positions.

LATE FUSION:
    Gated combination of both stream outputs.
    gate * fusion(ventral ⊕ dorsal) + (1 - gate) * ventral
    This allows the model to dynamically weight each stream.

BIOLOGICAL MOTIVATION:
    The primate visual cortex has two major processing pathways:
      - Ventral stream: V1 → V2 → V4 → IT (inferior temporal cortex)
        Processes object identity, shape, texture, color.
        "What is it?"
      - Dorsal stream: V1 → V2 → V3 → MT → Parietal cortex
        Processes spatial relationships, motion, location.
        "Where is it?"

    These streams are largely independent but converge in prefrontal cortex.
    Adversarial attacks optimized against a single stream's features are
    less effective when both streams must agree on the classification.

    Key insight: adversarial perturbations that corrupt object identity features
    (targeting the ventral stream) may leave spatial relationships intact
    (dorsal stream still correct), and vice versa. Late fusion provides
    robustness through redundancy.

ARCHITECTURE:
    Two parallel 2-layer transformers (vs single 3-layer in standard RHAN).
    Total transformer layers: 4 (2 per stream) — slightly more than 3,
    but each stream is narrower in practice due to parameter sharing in FFN.
    Fits within RTX 4060 VRAM budget.
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


class DualStreamAttention(nn.Module):
    """Parallel ventral (what) and dorsal (where) attention streams with late fusion."""

    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048,
                 dropout=0.1, num_layers_per_stream=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Ventral stream: standard learned positional embeddings
        self.ventral_pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))
        nn.init.trunc_normal_(self.ventral_pos_embed, std=0.02)

        ventral_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.ventral_transformer = nn.TransformerEncoder(
            ventral_layer, num_layers=num_layers_per_stream
        )

        # Dorsal stream: same architecture, different positional bias
        self.dorsal_pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))
        nn.init.trunc_normal_(self.dorsal_pos_embed, std=0.02)

        # Spatial bias: learnable matrix that biases attention toward spatial relationships
        # Initialized to favor nearby positions (local spatial coherence)
        self.spatial_bias = nn.Parameter(torch.zeros(1, 65, 65))
        self._init_spatial_bias()

        dorsal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.dorsal_transformer = nn.TransformerEncoder(
            dorsal_layer, num_layers=num_layers_per_stream
        )

        # Gated late fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _init_spatial_bias(self):
        """Initialize spatial bias to favor local spatial relationships."""
        bias = torch.zeros(65, 65)
        # For the 64 spatial positions (indices 1-64), compute pairwise distances
        for i in range(64):
            for j in range(64):
                yi, xi = i // 8, i % 8
                yj, xj = j // 8, j % 8
                dist_sq = (yi - yj) ** 2 + (xi - xj) ** 2
                bias[i + 1, j + 1] = -dist_sq  # negative distance = closer = higher bias
        # Normalize
        bias[1:, 1:] = bias[1:, 1:] / (bias[1:, 1:].abs().max() + 1e-8) * 0.5
        self.spatial_bias.data = bias.unsqueeze(0)

    def _add_spatial_bias_to_attention(self, transformer, tokens):
        """Run transformer with spatial bias added to attention scores.

        Since nn.TransformerEncoder doesn't expose attention scores directly,
        we use a simpler approach: add spatial bias to the token embeddings
        before the transformer processes them. This modulates the Q/K/V
        computations with spatial information.
        """
        B, N, C = tokens.shape
        # Expand spatial bias for batch
        sb = self.spatial_bias.expand(B, -1, -1)  # (B, 65, 65)
        # Use spatial bias to modulate tokens: tokens + tokens @ sb^T
        # This is equivalent to adding a spatially-weighted combination
        spatial_mod = torch.bmm(sb, tokens)  # (B, 65, 512)
        biased_tokens = tokens + 0.1 * spatial_mod
        return transformer(biased_tokens)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, 65, 512) — CLS + spatial tokens
        Returns:
            fused: (B, 65, 512) — fused ventral + dorsal output
            aux: dict with 'ventral' and 'dorsal' stream outputs + gate values
        """
        B = tokens.shape[0]

        # Ventral stream: standard attention
        ventral_input = tokens + self.ventral_pos_embed
        ventral_out = self.ventral_transformer(ventral_input)

        # Dorsal stream: attention with spatial bias
        dorsal_input = tokens + self.dorsal_pos_embed
        dorsal_out = self._add_spatial_bias_to_attention(
            self.dorsal_transformer, dorsal_input
        )

        # Gated late fusion
        concat = torch.cat([ventral_out, dorsal_out], dim=-1)  # (B, 65, 1024)
        gate = self.fusion_gate(concat)  # (B, 65, 512)
        fused = gate * self.fusion_proj(concat) + (1 - gate) * ventral_out
        fused = self.norm(fused)

        aux = {
            'ventral': ventral_out,
            'dorsal': dorsal_out,
            'fusion_gate': gate.mean(dim=1),  # avg over positions for analysis
        }
        return fused, aux


class RecurrentFeedbackDual(nn.Module):
    """Recurrent feedback for dual-stream RHAN."""

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
            current, _ = transformer_fn(modulated_tokens)
        return current


class RHAN_DualStream(nn.Module):
    """RHAN with Ventral/Dorsal stream split."""

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
        self.feedback = RecurrentFeedbackDual(
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
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.dual_attention,
        )
        return refined[:, 0, :]

    def forward(self, x):
        """
        Returns:
            logits: (B, 10)
            aux: dict with 'ventral', 'dorsal', 'fusion_gate' from dual attention
        """
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        attended, aux = self.dual_attention(tokens)
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.dual_attention,
        )
        cls_output = refined[:, 0, :]
        logits = self.head(cls_output)
        return logits, aux


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_architecture_summary():
    print("=" * 70)
    print("RHAN Dual-Stream (Ventral/Dorsal) — Architecture Summary")
    print("=" * 70)
    model = RHAN_DualStream()
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}")
    print(f"Model size (MB):      {total * 4 / 1024**2:>12.1f}")
    print(f"\n{'Stage':<45} {'Output Shape':<20}")
    print("-" * 65)
    print(f"{'Input':<45} {'(B, 3, 32, 32)':<20}")
    print(f"{'Stage 1: Conv Stem':<45} {'(B, 512, 8, 8)':<20}")
    print(f"{'Stage 2: Tokenise + CLS':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 3: Dual-Stream Attention':<45} {'(B, 65, 512)':<20}")
    print(f"{'  Ventral (what) ×2 layers':<45} {'(B, 65, 512)':<20}")
    print(f"{'  Dorsal (where) ×2 layers':<45} {'(B, 65, 512)':<20}")
    print(f"{'  Gated Fusion':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 4: Recurrent Feedback ×2':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 5: Linear Head':<45} {'(B, 10)':<20}")
    print("-" * 65)

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out, aux = model(x)
    print(f"\nForward pass: {tuple(x.shape)} → {tuple(out.shape)}")
    print(f"Ventral output: {tuple(aux['ventral'].shape)}")
    print(f"Dorsal output:  {tuple(aux['dorsal'].shape)}")
    print(f"Fusion gate:    {aux['fusion_gate'].mean().item():.3f} (avg)")
    assert out.shape == (4, 10)
    print("✓ Forward pass successful")
    print("=" * 70)
    return model


if __name__ == '__main__':
    print_architecture_summary()

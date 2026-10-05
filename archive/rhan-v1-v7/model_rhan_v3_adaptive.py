"""
RHAN-v3 Adaptive Recurrence
===========================
Trial 2 + Trial 3 Combination

Combines the Ventral/Dorsal stream split attention of RHAN-v3 with Graves'
Adaptive Computation Time (ACT) halting logic. The model dynamically adjusts
its recurrence step count (1 to 6 steps) based on input difficulty.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import ConvStem, PatchTokeniser, GlobalAttention, SemanticProjectionHead


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


class AdaptiveRHANSplit(nn.Module):
    """
    RHAN model combining Ventral/Dorsal stream split attention (v3)
    with Adaptive Computation Time (ACT) feedback loop.
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 max_steps=6, epsilon_halt=0.01, head_type='cosine'):
        super().__init__()
        self.max_steps = max_steps
        self.epsilon_halt = epsilon_halt
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

        # Stage 4: Recurrent Feedback convolution & gate
        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

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

        # Halting network: takes current spatial feature map, outputs halt probability
        self.halting_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pool spatial features
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def tokenise(self, stem_features):
        return self.tokeniser(stem_features)

    def tokens_to_spatial(self, tokens):
        spatial_tokens = tokens[:, 1:, :]  # Remove CLS token
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, 8, 8)

    def load_from_rhan_split(self, checkpoint_path, device='cuda'):
        """
        Maps standard RHANSplit checkpoint weights into AdaptiveRHANSplit.
        """
        state = torch.load(checkpoint_path, map_location=device)

        adapted_state = {}
        for k, v in state.items():
            if k.startswith('feedback.feedback_conv.'):
                adapted_state[k.replace('feedback.feedback_conv.', 'feedback_conv.')] = v
            elif k.startswith('feedback.gate.0.'):
                adapted_state[k.replace('feedback.gate.0.', 'gate_conv.')] = v
            else:
                adapted_state[k] = v

        # Load the mapped weights (halting_network will be initialized randomly)
        msg = self.load_state_dict(adapted_state, strict=False)
        print(f"[AdaptiveRHANSplit] Mapped checkpoint weights loaded with message: {msg}")

    def forward(self, x):
        # Stage 1: Conv stem
        stem_features = self.stem(x)  # (B, 512, 8, 8)

        # Stage 2: Tokenise
        tokens = self.tokenise(stem_features)  # (B, 65, 512)

        # Stage 3: Adaptive recurrent feedback
        B = x.size(0)
        cumulative_halt = torch.zeros(B, device=x.device)
        remainder = torch.ones(B, device=x.device)
        weighted_output = torch.zeros_like(tokens)
        steps_used = torch.zeros(B, device=x.device)

        for t in range(self.max_steps):
            # Transformer forward pass
            transformer_out = self.transformer(tokens)

            # Compute halting probability for this step
            spatial = self.tokens_to_spatial(transformer_out)
            h_t = self.halting_network(spatial).squeeze(-1)  # (B,)

            # Adaptive halting logic (Graves 2016)
            still_running = (cumulative_halt < 1 - self.epsilon_halt).float()

            # New cumulative halt
            new_cumulative = cumulative_halt + h_t * still_running

            # Weight for this step's output
            exceeds = (new_cumulative > 1 - self.epsilon_halt).float()
            weight = (exceeds * remainder + (1 - exceeds) * h_t) * still_running

            # Accumulate weighted output
            weighted_output += weight.view(B, 1, 1) * transformer_out

            # Update state
            remainder -= weight * still_running
            cumulative_halt = new_cumulative
            steps_used += still_running

            # Recurrent feedback for next step
            spatial = self.tokens_to_spatial(transformer_out)
            feedback = self.feedback_conv(spatial)
            gate = torch.sigmoid(self.gate_conv(spatial))
            stem_features = stem_features + gate * feedback
            tokens = self.tokenise(stem_features)

            # Early exit if all samples in batch have halted
            if still_running.sum() == 0:
                break

        # Classify from weighted accumulated output
        cls_token = weighted_output[:, 0, :]  # (B, 512)
        logits = self.head(cls_token)

        return logits, steps_used, cumulative_halt


if __name__ == '__main__':
    model = AdaptiveRHANSplit()
    x = torch.randn(2, 3, 32, 32)
    logits, steps, halt = model(x)
    print(f"AdaptiveRHANSplit verified. Logits: {logits.shape}, Steps: {steps}, Halt: {halt}")

"""
RHAN-v4: Multi-Scale Feedback + CLIP Semantic Grounding + Contrastive Adversarial Training
===========================================================================================
Proposed by: Mina Antonius (FerrariKazu), 2026

RHAN-v4 extends RHAN-v3 (Ventral/Dorsal Split) with THREE improvements:

  1. TRUE MULTI-SCALE FEEDBACK: Three parallel gated feedback paths
     operating at fine (32×32), medium (16×16), and coarse (8×8) scales.
     This replaces the single-scale feedback in v3.

  2. CLIP SEMANTIC PROJECTION HEAD: A learnable projection from the
     RHAN 512-dim CLS feature space into CLIP's 512-dim text embedding
     space. Used ONLY during training for semantic alignment loss.
     Zero inference overhead.

  3. InfoNCE CONTRASTIVE ADVERSARIAL CONSISTENCY: Replaces MSE
     consistency loss with proper InfoNCE contrastive loss, treating
     (clean_i, adv_i) as positive pairs and all cross-sample pairs
     as negatives.

Target: εthresh > 0.120 (closing toward human >0.300)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_rhan import PatchTokeniser, GlobalAttention, SemanticProjectionHead


# =============================================================================
# STAGE 1 — MULTI-SCALE CONVOLUTIONAL STEM
# =============================================================================
# Unlike the original ConvStem which only exposes the final (B, 512, 8, 8)
# output, MultiScaleConvStem exposes ALL THREE intermediate feature maps:
#   f1: (B, 64, 32, 32)  — fine scale (edge/texture)
#   f2: (B, 256, 16, 16) — medium scale (parts/patches)
#   f3: (B, 512, 8, 8)   — coarse scale (objects/scenes)
#
# This enables the multi-scale feedback loop to modulate features at
# every level of the spatial hierarchy, analogous to how the primate
# visual cortex has feedback connections at V1, V2, V4, and IT.
# =============================================================================

class MultiScaleConvStem(nn.Module):
    """Three-layer convolutional stem exposing intermediate feature maps."""

    def __init__(self):
        super().__init__()
        # Layer 1: 32×32 → 32×32 (stride=1, preserves spatial resolution)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Layer 2: 32×32 → 16×16 (stride=2, first spatial downsample)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Layer 3: 16×16 → 8×8 (stride=2, second spatial downsample)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Residual shortcut for the final output
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(512),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 32, 32)
        Returns:
            f1: (B, 64, 32, 32)   — fine scale
            f2: (B, 256, 16, 16)  — medium scale
            f3: (B, 512, 8, 8)    — coarse scale (includes residual shortcut)
        """
        identity = self.shortcut(x)
        f1 = self.layer1(x)           # (B, 64, 32, 32)
        f2 = self.layer2(f1)          # (B, 256, 16, 16)
        f3 = self.layer3(f2) + identity  # (B, 512, 8, 8)
        return f1, f2, f3

    def rerun_from_f1(self, f1, f2_orig, f3_orig):
        """
        Re-run layer2 and layer3 on feedback-modulated f1,
        with residual connections to the original f2 and f3.

        Args:
            f1: (B, 64, 32, 32)   — feedback-modulated fine features
            f2_orig: (B, 256, 16, 16)  — original medium features
            f3_orig: (B, 512, 8, 8)    — original coarse features
        Returns:
            f3_new: (B, 512, 8, 8) — updated coarse features
        """
        f2_new = self.layer2(f1) + f2_orig     # (B, 256, 16, 16)
        f3_new = self.layer3(f2_new) + f3_orig  # (B, 512, 8, 8)
        return f3_new


# =============================================================================
# STAGE 3 — VENTRAL/DORSAL SPLIT ATTENTION (from model_rhan_split.py)
# =============================================================================

class VentralDorsalAttention(nn.Module):
    """
    Splits the 512-dimensional embedding into parallel
    256-dim Ventral ("what") and Dorsal ("where") attention pathways.
    """

    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, num_layers=3):
        super().__init__()
        self.split_dim = embed_dim // 2  # 256

        self.ventral_transformer = GlobalAttention(
            embed_dim=self.split_dim,
            num_heads=num_heads // 2,
            ff_dim=ff_dim // 2,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.dorsal_transformer = GlobalAttention(
            embed_dim=self.split_dim,
            num_heads=num_heads // 2,
            ff_dim=ff_dim // 2,
            dropout=dropout,
            num_layers=num_layers,
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, 65, 512)
        Returns:
            (B, 65, 512)
        """
        ventral_tokens = tokens[:, :, :self.split_dim]
        dorsal_tokens = tokens[:, :, self.split_dim:]

        ventral_out = self.ventral_transformer(ventral_tokens)
        dorsal_out = self.dorsal_transformer(dorsal_tokens)

        return torch.cat([ventral_out, dorsal_out], dim=-1)


# =============================================================================
# STAGE 4 — MULTI-SCALE RECURRENT FEEDBACK
# =============================================================================
# Instead of a single feedback path (CLS → 8×8 features), v4 implements
# THREE parallel feedback paths that modulate features at every spatial
# scale in the convolutional hierarchy:
#
#   Coarse (8×8):  CLS token → f3   (global "what" → deep features)
#   Medium (16×16): Ventral mean → f2 (identity → intermediate features)
#   Fine (32×32):   Dorsal mean → f1  (spatial → early features)
#
# Each path has its own gated residual connection to prevent catastrophic
# overwriting. After modulation, the stem layers are re-run bottom-up
# to propagate the feedback corrections through the hierarchy.
# =============================================================================

class MultiScaleFeedback(nn.Module):
    """Three-path multi-scale recurrent feedback loop."""

    def __init__(self, embed_dim=512, num_patches=64, num_recurrent_steps=2):
        super().__init__()
        self.num_recurrent_steps = num_recurrent_steps
        self.spatial_h = int(math.sqrt(num_patches))  # 8
        self.spatial_w = int(math.sqrt(num_patches))   # 8
        split_dim = embed_dim // 2  # 256

        # ── Coarse feedback: CLS token (512) → f3 (512, 8, 8) ──
        self.coarse_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.coarse_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # ── Medium feedback: ventral mean (256) → f2 (256, 16, 16) ──
        self.medium_linear = nn.Linear(split_dim, split_dim)
        self.medium_proj = nn.Conv2d(split_dim, split_dim, kernel_size=1, bias=False)
        self.medium_gate = nn.Sequential(
            nn.Conv2d(split_dim, split_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # ── Fine feedback: dorsal mean (256) → f1 (64, 32, 32) ──
        self.fine_linear = nn.Linear(split_dim, 64)
        self.fine_proj = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.fine_gate = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def tokens_to_spatial(self, tokens):
        """(B, 65, 512) → (B, 512, 8, 8), discarding CLS."""
        spatial = tokens[:, 1:, :]  # (B, 64, 512)
        B, N, C = spatial.shape
        return spatial.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)

    def spatial_to_tokens(self, spatial, cls_token):
        """(B, 512, 8, 8) + (B, 1, 512) → (B, 65, 512)"""
        tokens = spatial.flatten(2).transpose(1, 2)  # (B, 64, 512)
        return torch.cat([cls_token, tokens], dim=1)

    def forward(self, transformer_output, f1, f2, f3, stem, tokeniser, transformer_fn):
        """
        Multi-scale recurrent feedback loop.

        Args:
            transformer_output: (B, 65, 512) — output from split attention
            f1: (B, 64, 32, 32)   — fine-scale stem features
            f2: (B, 256, 16, 16)  — medium-scale stem features
            f3: (B, 512, 8, 8)    — coarse-scale stem features
            stem: MultiScaleConvStem — for re-running layers
            tokeniser: PatchTokeniser — for re-tokenising
            transformer_fn: VentralDorsalAttention — for re-running attention
        Returns:
            (B, 65, 512) — refined token sequence
        """
        current = transformer_output
        split_dim = 256  # embed_dim // 2

        for t in range(self.num_recurrent_steps):
            cls_token = current[:, :1, :]   # (B, 1, 512)
            B = cls_token.shape[0]

            # ── Extract stream-specific signals from transformer output ──
            patch_tokens = current[:, 1:, :]  # (B, 64, 512)

            # Ventral = first 256 dims of patch tokens
            ventral_mean = patch_tokens[:, :, :split_dim].mean(dim=1)  # (B, 256)
            # Dorsal = last 256 dims of patch tokens
            dorsal_mean = patch_tokens[:, :, split_dim:].mean(dim=1)   # (B, 256)

            # ── Coarse feedback: CLS → f3 ──
            cls_vec = cls_token.squeeze(1)  # (B, 512)
            cls_spatial = cls_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)  # (B, 512, 8, 8)
            coarse_fb = self.coarse_proj(cls_spatial)
            gate_c = self.coarse_gate(coarse_fb)
            f3_mod = f3 + gate_c * coarse_fb

            # ── Medium feedback: ventral mean → f2 ──
            ventral_proj = self.medium_linear(ventral_mean)  # (B, 256)
            ventral_spatial = ventral_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)
            medium_fb = self.medium_proj(ventral_spatial)
            gate_m = self.medium_gate(medium_fb)
            f2_mod = f2 + gate_m * medium_fb

            # ── Fine feedback: dorsal mean → f1 ──
            dorsal_proj = self.fine_linear(dorsal_mean)  # (B, 64)
            dorsal_spatial = dorsal_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)
            fine_fb = self.fine_proj(dorsal_spatial)
            gate_f = self.fine_gate(fine_fb)
            f1_mod = f1 + gate_f * fine_fb

            # ── Re-run stem layers bottom-up with residual ──
            f3_new = stem.rerun_from_f1(f1_mod, f2_mod, f3_mod)

            # ── Re-tokenise and re-run attention ──
            new_tokens = tokeniser(f3_new)
            current = transformer_fn(new_tokens)

        return current


# =============================================================================
# FULL ARCHITECTURE — RHAN-v4
# =============================================================================

class RHANv4(nn.Module):
    """
    RHAN-v4: Multi-Scale Feedback + CLIP Semantic Grounding
    + Contrastive Adversarial Training

    Architecture:
      Input (B, 3, 32, 32)
        → MultiScaleConvStem → f1, f2, f3
        → PatchTokeniser(f3)  → (B, 65, 512)
        → VentralDorsalAttention ×3 layers
        → MultiScaleFeedback ×2 steps (modulates f1, f2, f3)
        → CLS token → SemanticProjectionHead → Logits (B, 10)
                    → clip_projection → CLIP space (training only)
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='cosine'):
        super().__init__()
        self.head_type = head_type

        # Stage 1: Multi-scale convolutional stem
        self.stem = MultiScaleConvStem()

        # Stage 2: Patch tokeniser (operates on f3: 512, 8, 8)
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)

        # Stage 3: Ventral/Dorsal split attention
        self.transformer = VentralDorsalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        # Stage 4: Multi-scale recurrent feedback
        self.feedback = MultiScaleFeedback(
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

        # CLIP semantic projection head (training-only, zero inference cost)
        self.clip_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def get_feature_vector(self, x):
        """
        Extract the 512-dim CLS token feature vector after multi-scale
        recurrent feedback. Used for alignment, CLIP, and contrastive losses.
        """
        f1, f2, f3 = self.stem(x)
        tokens = self.tokeniser(f3)
        attended = self.transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            f1=f1, f2=f2, f3=f3,
            stem=self.stem,
            tokeniser=self.tokeniser,
            transformer_fn=self.transformer,
        )
        return refined[:, 0, :]  # (B, 512) — CLS token

    def forward_with_features(self, x):
        """
        Forward pass returning both logits and CLS feature vector.
        Used during training for the multi-component loss.

        Returns:
            logits: (B, 10)
            features: (B, 512)
        """
        f1, f2, f3 = self.stem(x)
        tokens = self.tokeniser(f3)
        attended = self.transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            f1=f1, f2=f2, f3=f3,
            stem=self.stem,
            tokeniser=self.tokeniser,
            transformer_fn=self.transformer,
        )
        cls_output = refined[:, 0, :]  # (B, 512)
        logits = self.head(cls_output)  # (B, 10)
        return logits, cls_output

    def forward(self, x):
        """
        Standard forward pass returning only logits.
        No CLIP projection — zero inference overhead.
        """
        logits, _ = self.forward_with_features(x)
        return logits


if __name__ == '__main__':
    model = RHANv4()
    x = torch.randn(2, 3, 32, 32)
    logits, features = model.forward_with_features(x)
    print(f"RHANv4 verified. Logits: {logits.shape}, Features: {features.shape}")

    # Verify clip projection
    clip_proj = model.clip_projection(features)
    print(f"CLIP projection: {clip_proj.shape}")

    # Verify get_feature_vector
    fv = model.get_feature_vector(x)
    print(f"Feature vector: {fv.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

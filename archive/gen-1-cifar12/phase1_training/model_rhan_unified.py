"""
RHAN-UNIFIED: Proven v5 architecture adapted for STL-10 96×96.

Combines all proven components from RHAN-v5 plus biological improvements:
  1. Wide SE ConvStem — wider channels + Squeeze-Excitation channel attention
     (biological analog: V4 channel gain control for diagnostic feature selection)
  2. Patch tokenisation + positional encoding (144 tokens + CLS)
  3. Global attention transformer
  4. Recurrent top-down feedback loop
  5. Cosine similarity classification head (concept anchoring)

STL-10 specific adaptations:
  - 4-layer wide stem: 96×96 → 12×12 (144 spatial tokens)
  - SE blocks after each conv layer for channel-wise attention
  - Stochastic depth dropout in stem
  - STL-10 normalization constants
  - ~22M parameters (wider stem for feature richness on 5K samples)

NO VAE. NO generative prior. Just the proven architecture + biological improvements.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_rhan import PatchTokeniser, GlobalAttention, SemanticProjectionHead


# =============================================================================
# IMPROVEMENT 1 — SE Block (Squeeze-and-Excitation)
# Biological analog: V4 channel gain control
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.
    
    Biological analog: V4 neurons selectively attend to diagnostic
    features while suppressing noise channels. SE blocks learn to
    downweight channels carrying adversarial high-frequency
    perturbations because those channels are globally uninformative
    (they correlate with local patches, not the global scene context
    that SE uses for channel weighting).
    
    For adversarial robustness: SE blocks learn to downweight
    channels carrying adversarial high-frequency perturbations
    because those channels are globally uninformative (they
    correlate with local patches, not the global scene context
    that SE uses for channel weighting).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)
        return x * w


# =============================================================================
# IMPROVEMENT 1 — Wide SE ConvStem
# Replaces ConvStemUnified with wider channels + SE blocks + stochastic depth
# =============================================================================

class WideSEConvStem(nn.Module):
    """
    Wide stem with Squeeze-and-Excitation channel attention.
    
    Width increases feature diversity for small datasets (5K STL-10).
    SE blocks implement biological channel gain control — the same
    mechanism V4 uses to selectively attend to diagnostic features
    while suppressing noise channels.
    
    For adversarial robustness: SE blocks learn to downweight
    channels carrying adversarial high-frequency perturbations
    because those channels are globally uninformative (they
    correlate with local patches, not the global scene context
    that SE uses for channel weighting).
    
    Architecture:
        Conv(3→128, k=3, s=1) → SE → 96×96
        Conv(128→512, k=3, s=2) → SE → 48×48
        Conv(512→1024, k=3, s=2) → SE + StochasticDrop → 24×24
        Conv(1024→512, k=3, s=2) → SE → 12×12
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # Wider channels for STL-10 feature richness
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),   # 96→96, 128ch
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),                                # channel attention
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, 3, 2, 1, bias=False),  # 96→48, 512ch
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1, bias=False), # 48→24, 1024ch
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            SEBlock(1024),
        )
        # Compress back to 512 for transformer compatibility
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 2, 1, bias=False), # 24→12, 512ch
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
        )
        # Stochastic depth dropout (applied to conv3 output)
        self.stochastic_drop = nn.Dropout2d(p=dropout_rate)
        
        # Updated shortcut: 3→512, stride=8
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 512, 1, 8, bias=False),
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
        out = self.stochastic_drop(out)  # stochastic depth here
        out = self.conv4(out)
        return out + identity


# =============================================================================
# RECURRENT FEEDBACK — updated for 12×12 spatial (144 tokens)
# =============================================================================

class RecurrentFeedbackUnified(nn.Module):
    """Recurrent top-down feedback loop for 12x12 spatial features."""
    
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
# FULL RHAN-UNIFIED MODEL
# =============================================================================

class RHANUnified(nn.Module):
    """
    RHAN-UNIFIED for STL-10: 96×96 input, 144 spatial tokens.
    
    Improvements over base v5:
      - WideSEConvStem: wider channels (128→512→1024→512) + SE blocks
      - Stochastic depth dropout in stem
      - ~22M parameters for feature richness on small datasets
    """
    
    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='cosine',
                 stem_dropout=0.1):
        super().__init__()
        self.head_type = head_type
        
        # Stage 1: Wide SE Conv Stem — 96×96 → 12×12
        self.stem = WideSEConvStem(dropout_rate=stem_dropout)
        
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
        self.feedback = RecurrentFeedbackUnified(
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
    model = RHANUnified()
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    assert out.shape == (4, 10), "Expected (4,10), got {}".format(out.shape)
    print("RHAN-UNIFIED (WideSE) forward pass: OK")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters: {:,} total, {:,} trainable".format(total, trainable))
    # Expected: ~22M parameters

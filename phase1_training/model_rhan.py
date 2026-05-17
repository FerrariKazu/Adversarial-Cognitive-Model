"""
RHAN: Recurrent Hybrid Attention Network
=========================================
Proposed by: Mina Antonius (FerrariKazu), 2026
Motivated by: Adversarial Cognition Divergence empirical findings

THEORETICAL MOTIVATION:
Our psychophysical study (n=18 participants, 5 neural architectures,
6 epsilon levels, SDT analysis) identified two distinct robustness
failure regimes:

Regime 1 (ε < 0.03): ViT fails, ResNet succeeds
  → Local convolutional smoothing protects against imperceptible noise

Regime 2 (ε ≥ 0.05): ResNet fails, ViT partially succeeds
  → Global attention enables integration of uncorrupted patches

No single existing architecture handles both regimes.
Additionally, no tested architecture approaches human d' > 0.30
(all AI models cross d'=1.0 before ε=0.03).

We hypothesise that human robustness arises from:
  1. Local feature stability (convolutional prior)
  2. Global structural integration (attention)
  3. Top-down feedback modulation (recurrence)
  4. Semantic concept anchoring (contrastive representation)

RHAN implements all four principles in a unified lightweight model.

PREDICTED RESULTS:
  - RHAN d' threshold: between ViT (0.026) and Human (>0.300)
  - RHAN should outperform all tested models at ε=0.01 (Regime 1)
    AND at ε=0.10+ (Regime 2) simultaneously
  - If confirmed: first empirical validation of multi-principle
    adversarial robustness theory from psychophysical data

CITATION:
  Antonius et al. (2026). Adversarial Cognition Divergence: Mapping
  Perceptual Robustness Across Five Neural Architectures and Human
  Vision Using Signal Detection Theory. Sadat Academy, Giza, Egypt.
"""

import math
import torch
import torch.nn as nn


# =============================================================================
# STAGE 1 — CONVOLUTIONAL STEM
# =============================================================================
# WHY LOCAL CONVOLUTIONS PROVIDE REGIME 1 ROBUSTNESS:
#
# Convolutional kernels (3×3 here) compute weighted averages over local
# spatial neighbourhoods. This is mathematically equivalent to a spatial
# low-pass filter: high-frequency perturbations — precisely what PGD
# injects at small ε — are attenuated because the kernel averages them
# out across the receptive field.
#
# At ε < 0.03 (Regime 1), adversarial perturbations are imperceptible
# to humans and operate in the high-frequency domain. A convolutional
# stem with BatchNorm smooths these perturbations before they can
# propagate to the classification head. This is why ResNet-18 maintains
# 80%+ accuracy at ε=0.01 while ViT (which tokenises raw pixels with
# no spatial smoothing) drops to ~55%.
#
# The three-layer stem progressively expands channels (3→64→128→256)
# while halving spatial resolution twice (32→16→8), building a compact
# 8×8 feature map that retains local structure but has already smoothed
# away sub-pixel adversarial noise.
# =============================================================================

class ConvStem(nn.Module):
    """Three-layer convolutional stem with residual-style connections."""

    def __init__(self):
        super().__init__()
        # Layer 1: 32×32 → 32×32 (stride=1, preserves spatial resolution)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Layer 2: 32×32 → 16×16 (stride=2, first spatial downsample)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Layer 3: 16×16 → 8×8 (stride=2, second spatial downsample)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Residual projection shortcut: maps input directly to output dims
        # so the stem can learn a residual delta on top of a downsampled identity
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(512),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 32, 32) — raw CIFAR-10 images (normalised)
        Returns:
            (B, 512, 8, 8) — locally smoothed feature map
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + identity


# =============================================================================
# STAGE 2 — PATCH TOKENISATION & POSITIONAL ENCODING
# =============================================================================
# Instead of tokenising raw pixels (ViT-style, vulnerable at low ε), we
# tokenise the CNN feature map. Each spatial position in the 8×8 grid
# becomes a 256-dimensional token. This means:
#   - Tokens inherit the local smoothing from the conv stem
#   - Adversarial noise has already been attenuated before tokenisation
#   - We still get the full benefits of global attention across tokens
#
# A learnable CLS token is prepended (position 0) to aggregate global
# information for classification, following the ViT/BERT convention.
# Learnable positional embeddings (65 positions: 64 spatial + 1 CLS)
# encode spatial layout without hardcoded assumptions.
# =============================================================================

class PatchTokeniser(nn.Module):
    """Converts CNN feature maps into a sequence of tokens with positional encoding."""

    def __init__(self, embed_dim=512, num_patches=64):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Learnable CLS token — aggregates global information for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for 64 spatial tokens + 1 CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Initialise with truncated normal (standard ViT practice)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, feature_map):
        """
        Args:
            feature_map: (B, 512, 8, 8) from conv stem
        Returns:
            (B, 65, 512) — 64 spatial tokens + 1 CLS token, with positional encoding
        """
        B = feature_map.shape[0]

        # Reshape: (B, 256, 8, 8) → (B, 256, 64) → (B, 64, 256)
        tokens = feature_map.flatten(2).transpose(1, 2)

        # Prepend CLS token: (B, 64, 256) → (B, 65, 256)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        return tokens


# =============================================================================
# STAGE 3 — GLOBAL ATTENTION TRANSFORMER
# =============================================================================
# WHY GLOBAL ATTENTION OVER FEATURE TOKENS (NOT RAW PIXELS) IS SAFER:
#
# ViT's raw patch tokenisation creates tokens from contiguous pixel blocks.
# At low ε, adversarial perturbations corrupt individual pixels within a
# patch, and since the patch embedding is a simple linear projection,
# the corruption propagates directly into the token representation.
#
# By applying attention to CNN feature tokens instead:
#   1. Each token has already been spatially smoothed by the conv stem
#   2. The 256-dim embedding is learned (not a raw projection)
#   3. BatchNorm has normalised feature magnitudes, limiting adversarial
#      amplification through the attention softmax
#
# At high ε (Regime 2), global self-attention allows uncorrupted spatial
# positions to "vote" on the correct classification. Even if 30% of the
# 8×8 grid is corrupted, the remaining 70% can dominate through attention
# weighting — this is the mechanism that gives ViT its high-ε advantage.
#
# Two transformer layers are sufficient because:
#   - We only have 64 spatial tokens (not 196+ as in standard ViT)
#   - The conv stem already extracted rich local features
#   - Depth beyond 2 layers shows diminishing returns on CIFAR-10
# =============================================================================

class GlobalAttention(nn.Module):
    """Two-layer pre-norm Transformer encoder for global patch attention."""

    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm: more stable training, better gradients
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, 65, 512) — tokenised features with CLS
        Returns:
            (B, 65, 512) — globally attended features
        """
        return self.norm(self.encoder(tokens))


# =============================================================================
# STAGE 4 — RECURRENT FEEDBACK LOOP (The Key Innovation)
# =============================================================================
# BIOLOGICAL MOTIVATION:
#
# The primate visual cortex is not purely feedforward. After initial
# bottom-up processing (V1→V2→V4→IT, ~100ms), extensive top-down
# feedback connections (IT→V4→V2→V1) modulate earlier representations
# based on higher-level context. This feedback serves to:
#
#   1. RECONSTRUCT SIGNAL FROM NOISE: Higher-level "concept" activations
#      (e.g., "this is a cat") send feedback that reinforces features
#      consistent with that concept and suppresses inconsistent noise.
#      This is directly analogous to adversarial denoising.
#
#   2. RESOLVE AMBIGUITY: When local features are corrupted (as in
#      adversarial attack), global context from IT cortex disambiguates
#      the percept. This is why humans maintain >70% accuracy even at
#      ε=0.30 — their recurrent loops "clean up" the representation.
#
#   3. SHARPEN BOUNDARIES: Feedback sharpens spatial representations,
#      making object boundaries more robust to perturbation.
#
# WHY 2 RECURRENT STEPS IS SUFFICIENT:
#   - CORnet-S uses 2 recurrent steps in V2 and V4 areas
#   - Empirically, 2 steps capture >90% of the feedback benefit
#   - More steps risk gradient issues and add compute cost
#   - Our 8×8 spatial map is small enough that 2 rounds of global
#     attention + feedback can reach every spatial position
#
# WHY THIS IMPROVES ROBUSTNESS AT ALL EPSILON LEVELS:
#   - Low ε: Feedback suppresses residual noise that survived the stem
#   - Mid ε: Global context from transformer helps reconstruct corrupted
#     local features through the feedback convolution
#   - High ε: Multiple passes allow the network to iteratively refine
#     its representation, similar to how humans "look again" at
#     ambiguous stimuli. Each pass integrates more global context.
# =============================================================================

class RecurrentFeedback(nn.Module):
    """Recurrent top-down feedback loop: transformer outputs modulate stem features."""

    def __init__(self, embed_dim=512, num_patches=64, num_recurrent_steps=2):
        super().__init__()
        self.num_recurrent_steps = num_recurrent_steps
        self.spatial_h = int(math.sqrt(num_patches))  # 8
        self.spatial_w = int(math.sqrt(num_patches))  # 8

        # Feedback convolution: 1×1 conv that routes global information
        # back to spatial features. Lightweight but sufficient — it learns
        # a channel-wise reweighting conditioned on global context.
        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Gate mechanism: learns how much feedback to inject per channel.
        # Prevents catastrophic overwriting of stem features.
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def tokens_to_spatial(self, tokens):
        """Convert spatial tokens (excluding CLS) back to feature map format.

        Args:
            tokens: (B, 65, 512) — includes CLS at position 0
        Returns:
            (B, 512, 8, 8) — spatial feature map
        """
        spatial_tokens = tokens[:, 1:, :]  # Remove CLS token
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, self.spatial_h, self.spatial_w)

    def spatial_to_tokens(self, spatial, cls_token):
        """Convert spatial feature map back to tokens with CLS prepended.

        Args:
            spatial: (B, 512, 8, 8)
            cls_token: (B, 1, 512)
        Returns:
            (B, 65, 512)
        """
        B = spatial.shape[0]
        tokens = spatial.flatten(2).transpose(1, 2)  # (B, 64, 256)
        return torch.cat([cls_token, tokens], dim=1)

    def forward(self, transformer_output, stem_features, transformer_fn):
        """
        Args:
            transformer_output: (B, 65, 512) — output from Stage 3
            stem_features: (B, 512, 8, 8) — original output from Stage 1
            transformer_fn: callable — the transformer encoder to re-run
        Returns:
            (B, 65, 512) — refined token sequence after recurrent feedback
        """
        current = transformer_output

        for t in range(self.num_recurrent_steps):
            # Extract CLS token to preserve it across iterations
            cls_token = current[:, :1, :]  # (B, 1, 256)

            # Reshape transformer output back to spatial: (B, 256, 8, 8)
            spatial = self.tokens_to_spatial(current)

            # Feedback: compute modulation signal from global representation
            feedback = self.feedback_conv(spatial)

            # Gated residual injection: g ∈ [0,1] controls feedback strength
            g = self.gate(feedback)
            modulated = stem_features + g * feedback

            # Re-tokenise the modulated features with the preserved CLS token
            modulated_tokens = self.spatial_to_tokens(modulated, cls_token)

            # Re-run transformer attention on the feedback-modulated tokens
            current = transformer_fn(modulated_tokens)

        return current


# =============================================================================
# STAGE 5 — SEMANTIC PROJECTION HEAD
# =============================================================================
# WHY COSINE SIMILARITY CLASSIFICATION IS MORE ROBUST THAN LINEAR:
#
# A standard Linear(256, 10) classifier learns a weight matrix W and bias b,
# computing logits = Wx + b. This is vulnerable because:
#   1. The magnitude of x matters — adversarial perturbations can inflate
#      activations to push logits past decision boundaries
#   2. The classifier can exploit texture shortcuts (high-frequency features
#      that correlate with class but don't capture semantic meaning)
#
# Cosine similarity classification normalises both the feature vector and
# the class prototypes to the unit sphere before computing similarity:
#   logits = temperature × cos(feature, prototype)
#
# WHY NORMALISING TO UNIT SPHERE PREVENTS TEXTURE-BASED SHORTCUTS:
#   - All feature vectors have identical L2 norm (= 1.0), so the classifier
#     cannot exploit magnitude differences
#   - Classification depends purely on directional alignment in feature space
#   - Adversarial perturbations that change magnitude without changing
#     direction have zero effect on the output
#   - This forces the network to learn semantically meaningful directions
#     rather than texture-correlated magnitudes
#
# CONNECTION TO CLIP'S CONTRASTIVE OBJECTIVE:
#   - CLIP learns aligned image-text embeddings on the unit hypersphere
#   - The learnable temperature (initialised to log(100) ≈ 4.6) controls
#     the sharpness of the softmax distribution, identical to CLIP's design
#   - Our class prototypes play the role of CLIP's text embeddings — they
#     act as "concept anchors" in the shared embedding space
#   - This contrastive-style head encourages semantic clustering rather
#     than surface-level texture discrimination
# =============================================================================

class SemanticProjectionHead(nn.Module):
    """CLIP-inspired cosine similarity classification with learnable prototypes."""

    def __init__(self, embed_dim=512, num_classes=10):
        super().__init__()
        # Project CLS token to the semantic embedding space
        self.projection = nn.Linear(embed_dim, embed_dim)

        # Learnable class prototypes — each is a 256-dim "concept anchor"
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.trunc_normal_(self.class_prototypes, std=0.02)

        # Learnable temperature — controls softmax sharpness
        # Initialised to log(100) ≈ 4.6, same as CLIP
        self.logit_scale = nn.Parameter(torch.tensor(math.log(100.0)))

    def forward(self, cls_token):
        """
        Args:
            cls_token: (B, 512) — CLS token from transformer output
        Returns:
            (B, 10) — cosine similarity logits scaled by temperature
        """
        # Project and normalise features to unit sphere
        features = self.projection(cls_token)
        features = nn.functional.normalize(features, dim=-1)

        # Normalise class prototypes to unit sphere
        prototypes = nn.functional.normalize(self.class_prototypes, dim=-1)

        # Cosine similarity × temperature → logits
        temperature = self.logit_scale.exp().clamp(max=100.0)
        logits = temperature * (features @ prototypes.T)

        return logits


# =============================================================================
# FULL ARCHITECTURE — RHAN
# =============================================================================
# Input (B, 3, 32, 32)
#   → Conv Stem (B, 512, 8, 8)          [local smoothing — Regime 1]
#   → Tokenise + CLS + PosEmbed         [B, 65, 512]
#   → Transformer ×4                    [global attention — Regime 2]
#   → Recurrent Feedback ×2             [top-down modulation]
#   → CLS token → Semantic Head         [concept anchoring]
#   → Logits (B, 10)
# =============================================================================

class RHAN(nn.Module):
    """
    Recurrent Hybrid Attention Network (RHAN)
    Combines convolutional local smoothing, global self-attention,
    recurrent top-down feedback, and semantic concept anchoring.
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 num_recurrent_steps=2, head_type='cosine'):
        super().__init__()
        self.head_type = head_type

        # Stage 1: Convolutional stem — local smoothing (Regime 1 defence)
        self.stem = ConvStem()

        # Stage 2: Patch tokenisation + positional encoding
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)

        # Stage 3: Global attention transformer (Regime 2 defence)
        self.transformer = GlobalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        # Stage 4: Recurrent feedback loop (biological top-down modulation)
        self.feedback = RecurrentFeedback(
            embed_dim=embed_dim,
            num_patches=64,
            num_recurrent_steps=num_recurrent_steps,
        )

        # Stage 5: Classification head
        if head_type == 'linear':
            # Standard linear head — gives honest gradients for PGD
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            # Cosine similarity head (original RHAN design)
            self.head = SemanticProjectionHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
            )

    def forward(self, x):
        """
        Full forward pass through all five stages.

        Args:
            x: (B, 3, 32, 32) — normalised CIFAR-10 images
        Returns:
            (B, 10) — classification logits
        """
        # Stage 1: Local convolutional smoothing
        stem_features = self.stem(x)  # (B, 512, 8, 8)

        # Stage 2: Tokenise CNN features
        tokens = self.tokeniser(stem_features)  # (B, 65, 512)

        # Stage 3: Global self-attention
        attended = self.transformer(tokens)  # (B, 65, 512)

        # Stage 4: Recurrent feedback — re-runs transformer on modulated features
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=self.transformer,
        )  # (B, 65, 512)

        # Stage 5: Classification via CLS token
        cls_output = refined[:, 0, :]  # (B, 512) — CLS token
        logits = self.head(cls_output)  # (B, 10)

        return logits


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_architecture_summary():
    """Print a visual summary of RHAN with tensor shapes at each stage."""
    print("=" * 70)
    print("RHAN: Recurrent Hybrid Attention Network — Architecture Summary")
    print("=" * 70)

    model = RHAN()
    total, trainable = count_parameters(model)
    print(f"\nTotal parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}")
    print(f"Model size (MB):      {total * 4 / 1024**2:>12.1f}")

    print(f"\n{'Stage':<45} {'Output Shape':<20}")
    print("-" * 65)
    print(f"{'Input':<45} {'(B, 3, 32, 32)':<20}")
    print(f"{'Stage 1: Conv Stem (local smoothing)':<45} {'(B, 512, 8, 8)':<20}")
    print(f"{'Stage 2: Tokenise + CLS + PosEmbed':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 3: Transformer ×4 (global attention)':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 4: Recurrent Feedback ×2 (top-down)':<45} {'(B, 65, 512)':<20}")
    print(f"{'Stage 5: CLS → Semantic Head':<45} {'(B, 10)':<20}")
    print("-" * 65)

    # Verify forward pass
    print("\nForward pass verification...")
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print("  ✓ Forward pass successful — no shape errors")
    print("=" * 70)

    return model


if __name__ == '__main__':
    print_architecture_summary()

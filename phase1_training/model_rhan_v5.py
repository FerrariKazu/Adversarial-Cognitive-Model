"""
RHAN-v5: Frequency-Separated Biologically-Grounded Model
=========================================================

Core new principle: Human V1 explicitly separates spatial frequency channels. 
Low-frequency channels encode shape/structure and project to V4/IT. 
High-frequency channels encode texture/edges.

Adversarial attacks operate predominantly in high-frequency space. 
By routing classification through low-frequency features primarily,
RHAN-v5 is inherently less vulnerable to adversarial perturbations.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import PatchTokeniser, GlobalAttention, SemanticProjectionHead


class RHANv5(nn.Module):
    """
    RHAN-v5: Frequency-Separated Biologically-Grounded Model
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 head_type='cosine'):
        super().__init__()
        self.head_type = head_type

        # ── FREQUENCY SEPARATOR ──────────────────────────────────────
        # Learnable Gaussian blur for low-frequency extraction
        self.register_buffer('gaussian_kernel', self._make_gaussian_kernel(sigma=1.5, size=5))

        # Learnable frequency weighting parameters
        self.freq_weight_low = nn.Parameter(torch.tensor(0.85))
        self.freq_weight_high = nn.Parameter(torch.tensor(0.15))

        # ── LOW-FREQUENCY PATHWAY (primary — shape) ──────────────────
        self.stem_low = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── HIGH-FREQUENCY PATHWAY (secondary — texture/detail) ──────
        self.stem_high = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Stage 2: Tokeniser (operates on the fused 8x8 maps)
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)

        # Stage 3: Split Ventral/Dorsal Transformers (256-dim each)
        self.ventral_transformer = GlobalAttention(
            embed_dim=embed_dim // 2,
            num_heads=num_heads // 2,
            ff_dim=ff_dim // 2,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        self.dorsal_transformer = GlobalAttention(
            embed_dim=embed_dim // 2,
            num_heads=num_heads // 2,
            ff_dim=ff_dim // 2,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )

        # Stage 4: Recurrent Feedback convolution & gate
        self.feedback_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

        # Stage 5: Classification head
        if head_type == 'linear':
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            self.classifier = SemanticProjectionHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
            )

        # --- Concept Bottleneck Model Additions ---
        self.concepts = [
            'has_wings',        # airplane, bird
            'has_wheels',       # automobile, truck, bicycle
            'has_fur',          # cat, dog, horse, deer
            'has_feathers',     # bird
            'is_metallic',      # automobile, truck, ship, airplane
            'is_organic',       # animals
            'carries_cargo',    # truck (distinguishes from automobile)
            'is_small_vehicle', # automobile (distinguishes from truck)
            'lives_in_water',   # ship, frog
            'has_four_legs',    # cat, dog, horse, deer
            'has_rigid_body',   # automobile, truck, ship, airplane
            'is_elongated',     # ship, airplane
            'has_spotted_coat', # cat, dog
            'is_large_animal',  # horse, deer (not cat/dog)
            'has_hooves',       # horse, deer (not dog/cat)
        ]
        
        concept_labels = torch.tensor([
            # airplane: wings, metallic, rigid, elongated
            [1,0,0,0,1,0,0,0,0,0,1,1,0,0,0],
            # automobile: wheels, metallic, small_vehicle, rigid
            [0,1,0,0,1,0,0,1,0,0,1,0,0,0,0],
            # bird: wings, feathers, organic
            [1,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
            # cat: fur, organic, four_legs, spotted
            [0,0,1,0,0,1,0,0,0,1,0,0,1,0,0],
            # deer: fur, organic, four_legs, large, hooves
            [0,0,1,0,0,1,0,0,0,1,0,0,0,1,1],
            # dog: fur, organic, four_legs, spotted
            [0,0,1,0,0,1,0,0,0,1,0,0,1,0,0],
            # frog: organic, water
            [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
            # horse: fur, organic, four_legs, large, hooves
            [0,0,1,0,0,1,0,0,0,1,0,0,0,1,1],
            # ship: metallic, water, rigid, elongated
            [0,0,0,0,1,0,0,0,1,0,1,1,0,0,0],
            # truck: wheels, metallic, cargo, rigid
            [0,1,0,0,1,0,1,0,0,0,1,0,0,0,0],
        ], dtype=torch.float32)
        self.register_buffer('concept_labels', concept_labels)
        
        self.concept_layer = nn.Linear(512, 15)  # features -> concepts
        self.concept_bn = nn.BatchNorm1d(15)
        self.concept_classifier = nn.Linear(15, 10)

    def _make_gaussian_kernel(self, sigma, size):
        """Creates a 2D Gaussian kernel for frequency separation."""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)  # 2D Gaussian
        return kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    def separate_frequencies(self, x):
        """
        Separates image into low and high frequency components.
        """
        # Apply Gaussian blur depthwise
        x_low = F.conv2d(
            F.pad(x, [2, 2, 2, 2], mode='reflect'),
            self.gaussian_kernel,
            groups=3
        )
        x_high = x - x_low
        return x_low, x_high

    def tokenise(self, spatial_features):
        return self.tokeniser(spatial_features)

    def tokens_to_spatial(self, tokens):
        spatial_tokens = tokens[:, 1:, :]  # Remove CLS token
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, 8, 8)

    def transformer_final(self, tokens):
        v_tokens = self.ventral_transformer(tokens[:, :, :256])
        d_tokens = self.dorsal_transformer(tokens[:, :, 256:])
        return torch.cat([v_tokens, d_tokens], dim=-1)

    def get_feature_vector(self, x):
        """
        Extract features after frequency separation, fusion, and recurrent feedback.
        """
        # 1. Frequency separation
        x_low, x_high = self.separate_frequencies(x)

        # 2. Process each frequency band through its stem
        f_low = self.stem_low(x_low)
        f_high = self.stem_high(x_high)

        # 3. Weighted frequency fusion
        w_low = torch.sigmoid(self.freq_weight_low)
        w_high = torch.sigmoid(self.freq_weight_high)
        f = w_low * f_low + w_high * f_high

        # 4. Recurrent feedback (2 steps)
        for _ in range(2):
            tokens = self.tokenise(f)

            # Ventral/dorsal split transformers
            v_tokens = self.ventral_transformer(tokens[:, :, :256])
            d_tokens = self.dorsal_transformer(tokens[:, :, 256:])
            combined = torch.cat([v_tokens, d_tokens], dim=-1)

            # Feedback gate
            spatial = self.tokens_to_spatial(combined)
            feedback = self.feedback_conv(spatial)
            gate = torch.sigmoid(self.gate_conv(spatial))
            f = f + gate * feedback

        # 5. Final transformer attention run
        tokens = self.tokenise(f)
        combined_final = self.transformer_final(tokens)
        cls = combined_final[:, 0, :]
        return cls

    def forward_with_features(self, x):
        features = self.get_feature_vector(x)
        logits = self.classifier(features)
        return logits, features

    def forward(self, x):
        logits, _ = self.forward_with_features(x)
        return logits

    def forward_with_concepts(self, x):
        # Get RHAN features (all existing processing unchanged)
        features = self.get_feature_vector(x)  # (B, 512)
        
        # Concept bottleneck
        concepts = torch.sigmoid(self.concept_bn(
                       self.concept_layer(features)))  # (B, 15)
        
        # Classify from concepts
        logits = self.concept_classifier(concepts)  # (B, 10)
        
        return logits, concepts


if __name__ == '__main__':
    model = RHANv5()
    x = torch.randn(2, 3, 32, 32)
    logits, features = model.forward_with_features(x)
    print(f"RHANv5 verified. Logits: {logits.shape}, Features: {features.shape}")
    x_low, x_high = model.separate_frequencies(x)
    print(f"Freq separation: low {x_low.shape}, high {x_high.shape}")
    print(f"Weights: low={torch.sigmoid(model.freq_weight_low).item():.3f}, high={torch.sigmoid(model.freq_weight_high).item():.3f}")

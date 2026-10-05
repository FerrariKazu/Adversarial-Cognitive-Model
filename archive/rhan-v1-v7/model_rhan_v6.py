"""
RHAN-v6: Dynamic Adaptive Frequency Gating with Predictive Coding Feedback
========================================================================

New mechanisms over v5:
1. Dynamic frequency gates (input-dependent, from pre-trained noise backbone)
2. Predictive coding feedback (error signal, not raw feature addition)
3. Top-down high-frequency suppression (recurrent decay of high-freq features)
4. ACT adaptive pondering (dynamic steps per image with ponder cost)
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


class RHANv6(nn.Module):
    """
    RHAN-v6: Dynamic Adaptive Frequency Gating with
    Predictive Coding Feedback and Adaptive Computation Time
    """

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 head_type='cosine', max_ponder_steps=6, epsilon_halt=0.05):
        super().__init__()
        self.head_type = head_type
        self.max_ponder_steps = max_ponder_steps
        self.epsilon_halt = epsilon_halt

        # ── FREQUENCY SEPARATOR ──────────────────────────────────────
        self.register_buffer('gaussian_kernel', self._make_gaussian_kernel(sigma=1.5, size=5))

        # ── DYNAMIC GATE NETWORK (NEW — replaces static weights) ─────
        # Takes the high-frequency residual as input (which contains concentrated noise)
        # Outputs: [alpha_low, alpha_high] in (0,1) per image
        self.noise_estimator_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # -> 1x1
            nn.Flatten(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [alpha_low, alpha_high]
            nn.Sigmoid()       # both bounded in (0,1)
        )

        # ── FREQUENCY STEMS (same as v5) ─────────────────────────────
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

        # ── PREDICTIVE CODING FEEDBACK (NEW) ─────────────────────────
        # Decodes the global context (CLS token) back to predicted spatial features at layer 3
        self.prediction_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.error_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # ── HIGH-FREQUENCY SUPPRESSOR (NEW) ──────────────────────────
        # Computes top-down suppression scalar per step
        self.hf_suppressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # ── ACT HALTING NETWORK (NEW) ────────────────────────────────
        self.halting_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Stage 5: Classification head (linear fallback or semantic cosine projection)
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

    def _make_gaussian_kernel(self, sigma, size):
        """Creates a 2D Gaussian kernel for frequency separation."""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    def separate_frequencies(self, x):
        """Separates image into low and high frequency components."""
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

    def load_noise_estimator_weights(self, path, device='cpu'):
        """Loads pretrained noise estimator backbone weights."""
        if not os.path.exists(path):
            print(f"WARNING: Pretrained noise estimator not found at {path}")
            return
        state = torch.load(path, map_location=device)
        backbone_dict = {}
        for k, v in state.items():
            if k.startswith('net.'):
                parts = k.split('.')
                idx = int(parts[1])
                if idx < 5:  # indices 0-4 are the backbone layers (Conv2d, ReLU, Conv2d, ReLU, AdaptiveAvgPool)
                    new_key = f"{idx}." + ".".join(parts[2:])
                    backbone_dict[new_key] = v
        self.noise_estimator_backbone.load_state_dict(backbone_dict, strict=False)
        print(f"Successfully loaded noise estimator backbone from {path}")

    def forward(self, x):
        B = x.size(0)

        # Step 1: Frequency separation
        x_low, x_high = self.separate_frequencies(x)

        # Step 2: Dynamic input-dependent gating from high-frequency residual
        noise_feats = self.noise_estimator_backbone(x_high)
        gates = self.gate_head(noise_feats)  # (B, 2)
        alpha_low = gates[:, 0].view(B, 1, 1, 1)
        alpha_high = gates[:, 1].view(B, 1, 1, 1)

        # Step 3: Process frequencies through stems
        f_low = self.stem_low(x_low)
        f_high = self.stem_high(x_high)

        # Step 4: Dynamically weighted fusion
        f = alpha_low * f_low + alpha_high * f_high

        # Step 5: Adaptive recurrent loop with predictive coding & top-down suppression
        cumulative_halt = torch.zeros(B, device=x.device)
        remainder = torch.ones(B, device=x.device)
        weighted_output = torch.zeros(B, 512, device=x.device)
        hf_suppress = torch.ones(B, 1, 1, 1, device=x.device)

        steps_used = torch.zeros(B, device=x.device)

        for t in range(self.max_ponder_steps):
            still_running = (cumulative_halt < 1 - self.epsilon_halt).float()
            if still_running.sum() == 0:
                break

            steps_used += still_running

            # Tokenise and split Ventral/Dorsal
            tokens = self.tokenise(f)
            v_out = self.ventral_transformer(tokens[:, :, :256])
            d_out = self.dorsal_transformer(tokens[:, :, 256:])
            combined = torch.cat([v_out, d_out], dim=-1)

            # Get CLS token global context
            cls_token = combined[:, 0, :]  # (B, 512)

            # ── PREDICTIVE CODING FEEDBACK ──
            # Predict spatial representation f from global context
            predicted_f = self.prediction_decoder(cls_token)
            predicted_f = predicted_f.view(B, 512, 1, 1).expand_as(f)

            # Prediction error (residual)
            prediction_error = f - predicted_f

            # Gate the surprise (residual error) instead of raw feature addition
            err_gate = self.error_gate(self.tokens_to_spatial(combined))

            # Update f with surprise-gated error
            f = f + err_gate * prediction_error

            # ── HIGH-FREQUENCY TOP-DOWN SUPPRESSION ──
            # Context-driven active high-frequency suppression
            hf_suppress_t = self.hf_suppressor(cls_token)
            hf_suppress_t = hf_suppress_t.view(B, 1, 1, 1)
            hf_suppress = hf_suppress * (1.0 - 0.3 * hf_suppress_t)

            # Re-fuse stems with suppressed high frequency
            f_high_suppressed = hf_suppress * f_high
            f = alpha_low * f_low + alpha_high * f_high_suppressed

            # ── ACT HALTING ──
            h_t = self.halting_network(f).squeeze(-1)
            new_cumulative = cumulative_halt + h_t * still_running
            exceeds = (new_cumulative > 1 - self.epsilon_halt).float()
            weight = (exceeds * remainder + (1.0 - exceeds) * h_t) * still_running

            # Accumulate weighted global context representations
            weighted_output += weight.view(B, 1) * cls_token
            remainder -= weight * still_running
            cumulative_halt = new_cumulative

        logits = self.classifier(weighted_output)
        return logits, steps_used.mean()

    def forward_with_features(self, x):
        """
        Runs the full forward pass returning logits, weighted features, and mean steps used.
        """
        B = x.size(0)
        x_low, x_high = self.separate_frequencies(x)

        noise_feats = self.noise_estimator_backbone(x_high)
        gates = self.gate_head(noise_feats)
        alpha_low = gates[:, 0].view(B, 1, 1, 1)
        alpha_high = gates[:, 1].view(B, 1, 1, 1)

        f_low = self.stem_low(x_low)
        f_high = self.stem_high(x_high)
        f = alpha_low * f_low + alpha_high * f_high

        cumulative_halt = torch.zeros(B, device=x.device)
        remainder = torch.ones(B, device=x.device)
        weighted_output = torch.zeros(B, 512, device=x.device)
        hf_suppress = torch.ones(B, 1, 1, 1, device=x.device)
        steps_used = torch.zeros(B, device=x.device)

        for t in range(self.max_ponder_steps):
            still_running = (cumulative_halt < 1 - self.epsilon_halt).float()
            if still_running.sum() == 0:
                break

            steps_used += still_running
            tokens = self.tokenise(f)
            v_out = self.ventral_transformer(tokens[:, :, :256])
            d_out = self.dorsal_transformer(tokens[:, :, 256:])
            combined = torch.cat([v_out, d_out], dim=-1)

            cls_token = combined[:, 0, :]
            predicted_f = self.prediction_decoder(cls_token)
            predicted_f = predicted_f.view(B, 512, 1, 1).expand_as(f)
            prediction_error = f - predicted_f
            err_gate = self.error_gate(self.tokens_to_spatial(combined))
            f = f + err_gate * prediction_error

            hf_suppress_t = self.hf_suppressor(cls_token)
            hf_suppress_t = hf_suppress_t.view(B, 1, 1, 1)
            hf_suppress = hf_suppress * (1.0 - 0.3 * hf_suppress_t)

            f_high_suppressed = hf_suppress * f_high
            f = alpha_low * f_low + alpha_high * f_high_suppressed

            h_t = self.halting_network(f).squeeze(-1)
            new_cumulative = cumulative_halt + h_t * still_running
            exceeds = (new_cumulative > 1 - self.epsilon_halt).float()
            weight = (exceeds * remainder + (1.0 - exceeds) * h_t) * still_running

            weighted_output += weight.view(B, 1) * cls_token
            remainder -= weight * still_running
            cumulative_halt = new_cumulative

        logits = self.classifier(weighted_output)
        return logits, weighted_output, steps_used.mean()


if __name__ == '__main__':
    model = RHANv6()
    x = torch.randn(2, 3, 32, 32)
    logits, steps = model(x)
    print(f"RHANv6 verified. Logits: {logits.shape}, Avg Steps: {steps.item():.2f}")
    logits, feats, steps = model.forward_with_features(x)
    print(f"forward_with_features matches. Feats: {feats.shape}")

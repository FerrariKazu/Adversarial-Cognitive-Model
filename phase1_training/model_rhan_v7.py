import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import PatchTokeniser, GlobalAttention, SemanticProjectionHead

class RHANv7(nn.Module):
    """
    RHAN-v7: Generative World-Model RHAN
    
    The generative prior adds a biological world-model:
    The brain doesn't just classify — it maintains an internal
    generative model of what the visual world should look like.
    Adversarial attacks must fool the classifier AND keep
    features in a region that decodes to a plausible image.
    
    VAE decoder creates a second constraint on adversarial paths:
    cross the decision boundary AND remain reconstructable.
    Together these constraints require a larger perturbation.
    """
    
    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 head_type='cosine', latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_type = head_type
        
        # ── EXISTING RHAN-v5 COMPONENTS ───────────────
        self.register_buffer('gaussian_kernel', self._make_gaussian_kernel(sigma=1.5, size=5))

        self.freq_weight_low = nn.Parameter(torch.tensor(0.85))
        self.freq_weight_high = nn.Parameter(torch.tensor(0.15))

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

        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)

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

        self.feedback_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

        # ── VAE ENCODER HEAD ────────────────────────────────
        self.vae_mu      = nn.Linear(embed_dim, latent_dim)
        self.vae_log_var = nn.Linear(embed_dim, latent_dim)
        
        # ── DECODER ─────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, 4, 1, 0),   # → 4×4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # → 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1),   # → 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  3,   4, 2, 1),   # → 32×32
            nn.Tanh()
        )
        
        # ── GENERATIVE CLASSIFIER ─────
        self.generative_classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, num_classes)
        )
        
        # ── PERCEPTUAL CRITIC (frozen) ─────────────────────
        # Frozen copy of stem_low used as feature extractor for
        # perceptual reconstruction loss. Never trains.
        self.perceptual_critic = copy.deepcopy(self.stem_low)
        for p in self.perceptual_critic.parameters():
            p.requires_grad = False

    def _make_gaussian_kernel(self, sigma, size):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    def separate_frequencies(self, x):
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
        spatial_tokens = tokens[:, 1:, :]
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, 8, 8)

    def transformer_final(self, tokens):
        v_tokens = self.ventral_transformer(tokens[:, :, :256])
        d_tokens = self.dorsal_transformer(tokens[:, :, 256:])
        return torch.cat([v_tokens, d_tokens], dim=-1)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def perceptual_reconstruction_loss(self, x_original, x_reconstructed):
        """
        Measures reconstruction quality in FEATURE space, not pixel space.
        
        Instead of: MSE(decoder_output_pixels, original_pixels)
        We compute: MSE(stem(decoder_output), stem(original))
        
        Compatible with adversarial training because the critic
        extracts semantic features — if the decoder produces an image
        with correct semantic content, loss is low regardless of
        pixel-level differences.
        """
        x_low_orig, _ = self.separate_frequencies(x_original)
        x_low_recon, _ = self.separate_frequencies(x_reconstructed)
        
        feats_orig  = self.perceptual_critic(x_low_orig)   # (B,512,8,8)
        feats_recon = self.perceptual_critic(x_low_recon)  # (B,512,8,8)
        
        # Normalize before MSE to focus on feature patterns not magnitude
        feats_orig  = F.normalize(feats_orig.flatten(1), dim=1)
        feats_recon = F.normalize(feats_recon.flatten(1), dim=1)
        
        return F.mse_loss(feats_recon, feats_orig)

    def get_feature_vector(self, x):
        x_low, x_high = self.separate_frequencies(x)

        f_low = self.stem_low(x_low)
        f_high = self.stem_high(x_high)

        w_low = torch.sigmoid(self.freq_weight_low)
        w_high = torch.sigmoid(self.freq_weight_high)
        f = w_low * f_low + w_high * f_high

        for _ in range(2):
            tokens = self.tokenise(f)

            v_tokens = self.ventral_transformer(tokens[:, :, :256])
            d_tokens = self.dorsal_transformer(tokens[:, :, 256:])
            combined = torch.cat([v_tokens, d_tokens], dim=-1)

            spatial = self.tokens_to_spatial(combined)
            feedback = self.feedback_conv(spatial)
            gate = torch.sigmoid(self.gate_conv(spatial))
            f = f + gate * feedback

        tokens = self.tokenise(f)
        combined_final = self.transformer_final(tokens)
        cls = combined_final[:, 0, :]
        return cls

    def forward(self, x):
        features = self.get_feature_vector(x)
        
        mu      = self.vae_mu(features)       # (B, latent_dim)
        log_var = self.vae_log_var(features)  # (B, latent_dim)
        z = self.reparameterize(mu, log_var)  # (B, latent_dim)
        
        x_recon = self.decoder(z)             # (B, 3, 32, 32)
        logits  = self.generative_classifier(mu)  # (B, 10)
        
        return logits, x_recon, mu, log_var

    def forward_with_features(self, x):
        logits, x_recon, mu, log_var = self.forward(x)
        return logits, mu

    def forward_full(self, x):
        features = self.get_feature_vector(x)
        mu      = self.vae_mu(features)
        log_var = self.vae_log_var(features)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        logits  = self.generative_classifier(mu)
        return logits, x_recon, mu, log_var, features

if __name__ == '__main__':
    model = RHANv7()
    x = torch.randn(2, 3, 32, 32)
    logits, x_recon, mu, log_var = model(x)
    print(f"RHANv7 verified.")
    print(f"Logits: {logits.shape}")
    print(f"x_recon: {x_recon.shape}")
    print(f"mu: {mu.shape}")
    print(f"log_var: {log_var.shape}")

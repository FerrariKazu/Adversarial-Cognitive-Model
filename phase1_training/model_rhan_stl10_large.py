import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlockLarge(nn.Module):
    """Squeeze-and-Excitation channel attention block scaled for Large stem."""
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

class WideSEConvStemLarge(nn.Module):
    """
    Wide stem with Squeeze-and-Excitation channel attention, scaled for Large embed_dim (768).
    Architecture:
        Conv(3→128, k=3, s=1) → SE → 96×96
        Conv(128→512, k=3, s=2) → SE → 48×48
        Conv(512→1024, k=3, s=2) → SE + StochasticDrop → 24×24
        Conv(1024→768, k=3, s=2) → SE → 12×12
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlockLarge(128),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlockLarge(512),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            SEBlockLarge(1024),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 768, 3, 2, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            SEBlockLarge(768),
        )
        self.stochastic_drop = nn.Dropout2d(p=dropout_rate)
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 768, 1, 8, bias=False),
            nn.BatchNorm2d(768),
        )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.stochastic_drop(out)
        out = self.conv4(out)
        return out + identity

class PatchTokeniserLarge(nn.Module):
    """Converts CNN feature maps to tokens with positional encoding for embed_dim=768."""
    def __init__(self, embed_dim=768, num_patches=144):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        self.token_proj = nn.Sequential(
            nn.Conv2d(768, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, feature_map):
        B = feature_map.shape[0]
        tokens_2d = self.token_proj(feature_map)
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens + self.pos_embed

class PredictiveCodingLayerLarge(nn.Module):
    """Predictive coding feedback layer scaled for Large dimensions (768)."""
    def __init__(self, channels=768):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.error_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )
        self.error_scale = nn.Parameter(torch.ones(1))

    def forward(self, local_f, global_spatial):
        predicted = self.predictor(global_spatial)
        error = local_f - predicted
        gate = self.error_gate(error)
        corrected = local_f + self.error_scale * gate * error
        return corrected, error.abs().mean()

class RecurrentFeedbackLarge(nn.Module):
    """Recurrent top-down feedback loop for Large (12x12) features."""
    def __init__(self, embed_dim=768, num_recurrent_steps=2):
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
        self.predictive_coder = PredictiveCodingLayerLarge(channels=embed_dim)
    
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
        f = stem_features
        for t in range(self.num_recurrent_steps):
            cls_token = current[:, :1, :]
            spatial = self.tokens_to_spatial(current)
            feedback = self.feedback_conv(spatial)
            g = self.gate(feedback)
            f_modulated = f + g * feedback
            
            # Predictive coding step
            f, _ = self.predictive_coder(f_modulated, spatial)
            
            modulated_tokens = self.spatial_to_tokens(f, cls_token)
            current = transformer_fn(modulated_tokens)
        return current

class MotionEncoderLarge(nn.Module):
    """Encodes motion between frame t and t+1 scaled for Large model (768->512)."""
    def __init__(self, embed_dim=768, out_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x_t, x_t1):
        x_cat = torch.cat([x_t, x_t1], dim=1)
        return self.encoder(x_cat)

class TDVProjectionHeadLarge(nn.Module):
    """Projects Large feature representations to TDV space (768->512)."""
    def __init__(self, embed_dim=768, proj_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z):
        return self.projector(z)

class RHANLargeSTL10(nn.Module):
    """
    RHAN-Large: ViT-B scale architecture for STL-10 at 96×96.
    Parameters: ~52M. Designed for A100.
    """
    def __init__(self, num_classes=10,
                 embed_dim=768,
                 num_heads=12,
                 ff_dim=3072,
                 num_transformer_layers=8,
                 num_recurrent_steps=2,
                 stem_dropout=0.1):
        super().__init__()
        
        self.stem = WideSEConvStemLarge(dropout_rate=stem_dropout)
        self.tokeniser = PatchTokeniserLarge(embed_dim=embed_dim, num_patches=144)
        
        # ── VENTRAL/DORSAL SPLIT TRANSFORMER (ViT-B Scale) ──────────────────
        # embed_dim // 2 = 384, num_heads // 2 = 6, ff_dim // 2 = 1536
        ventral_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2,
            nhead=num_heads // 2,
            dim_feedforward=ff_dim // 2,
            dropout=0.1,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.ventral = nn.TransformerEncoder(
            ventral_layer, num_layers=num_transformer_layers)
            
        dorsal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2,
            nhead=num_heads // 2,
            dim_feedforward=ff_dim // 2,
            dropout=0.1,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.dorsal = nn.TransformerEncoder(
            dorsal_layer, num_layers=num_transformer_layers)
        
        self.feedback = RecurrentFeedbackLarge(
            embed_dim=embed_dim,
            num_recurrent_steps=num_recurrent_steps
        )
        
        self.motion_encoder = MotionEncoderLarge(embed_dim=embed_dim, out_dim=512)
        self.tdv_head = TDVProjectionHeadLarge(embed_dim=embed_dim, proj_dim=512)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        
    def _run_transformer(self, tokens):
        B = tokens.shape[0]
        # Split tokens channel-wise (768 -> 384 + 384)
        v_tokens = self.ventral(tokens[:, :, :384])
        d_tokens = self.dorsal(tokens[:, :, 384:])
        combined = torch.cat([v_tokens, d_tokens], dim=-1)
        return combined

    def get_feature_vector(self, x):
        stem_features = self.stem(x)
        tokens = self.tokeniser(stem_features)
        
        # Helper to pass _run_transformer to recurrent feedback loop
        def run_transformer_fn(toks):
            return self._run_transformer(toks)
            
        attended = self._run_transformer(tokens)
        refined = self.feedback(
            transformer_output=attended,
            stem_features=stem_features,
            transformer_fn=run_transformer_fn
        )
        return refined[:, 0, :]

    def forward(self, x):
        features = self.get_feature_vector(x)
        return self.classifier(features)

    STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                     'dog', 'horse', 'monkey', 'ship', 'truck']

if __name__ == '__main__':
    # Dry-run validation
    model = RHANLargeSTL10()
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Forward pass shape: {out.shape}")
    print(f"Parameters: {params:,}")
    assert out.shape == (4, 10), f"Expected shape (4, 10), got {out.shape}"

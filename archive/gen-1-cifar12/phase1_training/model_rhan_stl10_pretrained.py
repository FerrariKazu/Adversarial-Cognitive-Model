import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

class PredictiveCodingLayerSTL(nn.Module):
    """Predictive coding feedback adapted for STL-10 feature dimensions."""
    def __init__(self, channels=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.error_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid(),
        )
        self.error_scale = nn.Parameter(torch.ones(1))

    def forward(self, local_f, global_spatial):
        predicted = self.predictor(global_spatial)
        error = local_f - predicted
        gate = self.error_gate(error)
        corrected = local_f + self.error_scale * gate * error
        return corrected, error.abs().mean()

class MotionEncoder(nn.Module):
    """
    Encodes the 'motion' between frame t and frame t+1.
    Input: concatenated [x_t, x_t1] → 6-channel input
    """
    def __init__(self, embed_dim=512, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, 2, 1),    # 96→48, 6ch input (concat frames)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, 2, 1),  # 48→24
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1), # 24→12
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),       # 12→1
            nn.Flatten(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x_t, x_t1):
        # Concatenate frames along channel dimension
        x_cat = torch.cat([x_t, x_t1], dim=1)  # (B, 6, 96, 96)
        return self.encoder(x_cat)  # (B, out_dim)

class TDVProjectionHead(nn.Module):
    """
    Projects encoder output to TDV prediction space.
    The TDV equation: z_t + m_t = z_{t+1}
    """
    def __init__(self, embed_dim=512, proj_dim=256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z):
        return self.projector(z)

class RHANUnifiedSTL10(nn.Module):
    """
    RHAN-Unified for STL-10 96×96.
    Uses ImageNet pretrained ResNet-50 layers 1-2 (conv1 through layer2) as stem.
    Includes MotionEncoder and TDVProjectionHead for TDV training.
    """
    STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                     'dog', 'horse', 'monkey', 'ship', 'truck']

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3):
        super().__init__()
        self.embed_dim = embed_dim

        # ── IMAGENET PRETRAINED STEM ──────────────────────────────────────
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            resnet.conv1,       # 96 → 48×48
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,     # 48 → 24×24
            resnet.layer1,      # 24×24, 256ch
            resnet.layer2,      # 12×12, 512ch
        )

        # ── FREQUENCY ANALYSIS LAYER ──────────────────────────────────────
        self.freq_weights = nn.Parameter(torch.ones(512))

        # ── TOKENISER ────────────────────────────────────────────────────
        self.token_proj = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, embed_dim), nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 145, embed_dim) * 0.02)

        # ── VENTRAL/DORSAL TRANSFORMER ───────────────────────────────────
        ventral_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2, nhead=num_heads // 2,
            dim_feedforward=ff_dim // 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.ventral = nn.TransformerEncoder(ventral_layer, num_layers=num_transformer_layers)

        dorsal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2, nhead=num_heads // 2,
            dim_feedforward=ff_dim // 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.dorsal = nn.TransformerEncoder(dorsal_layer, num_layers=num_transformer_layers)

        # ── PREDICTIVE CODING FEEDBACK ───────────────────────────────────
        self.predictive_coder = PredictiveCodingLayerSTL(channels=512)

        # ── CLASSIFICATION HEAD ───────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.orthogonal_(self.prototypes)
        self.log_scale = nn.Parameter(torch.tensor(10.0).log())

        # ── TDV COMPONENTS ────────────────────────────────────────────────
        self.motion_encoder = MotionEncoder(embed_dim=embed_dim, out_dim=256)
        self.tdv_head = TDVProjectionHead(embed_dim=embed_dim, proj_dim=256)

    def freeze_stem(self, freeze: bool):
        for p in self.stem.parameters():
            p.requires_grad = not freeze
        print(f"Stem {'frozen' if freeze else 'unfrozen'}")

    def _run_transformer(self, spatial_features):
        B = spatial_features.shape[0]
        tokens_2d = self.token_proj(spatial_features)
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        v_tokens = self.ventral(tokens[:, :, :256])
        d_tokens = self.dorsal(tokens[:, :, 256:])
        combined = torch.cat([v_tokens, d_tokens], dim=-1)

        cls_out = combined[:, 0, :]
        spatial_out = combined[:, 1:, :]
        spatial_map = spatial_out.transpose(1, 2).reshape(B, 512, 12, 12)

        return cls_out, spatial_map

    def get_features(self, x):
        f = self.stem(x)
        freq_w = torch.sigmoid(self.freq_weights).view(1, 512, 1, 1)
        f = f * freq_w

        for step in range(3):
            cls, spatial_map = self._run_transformer(f)
            f, _ = self.predictive_coder(f, spatial_map)

        return cls

    def get_feature_vector(self, x):
        return self.get_features(x)

    def classify(self, features):
        features = self.norm(features)
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        scale = self.log_scale.exp().clamp(1, 100)
        return scale * (features @ prototypes.T)

    def forward(self, x):
        return self.classify(self.get_features(x))

RHANSTL10Pretrained = RHANUnifiedSTL10

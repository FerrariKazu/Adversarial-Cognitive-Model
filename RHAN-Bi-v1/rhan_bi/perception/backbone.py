"""Feature backbones.

``HandcraftedBackbone`` is the default and the one every reported number uses in v1.
``DinoV2Backbone`` is optional and requires torch + a local ViT-S/14 checkpoint; it attaches CLS and
patch tokens to the handcrafted anomaly maps so the evidence model remains comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from rhan_bi.config import FeatureConfig
from rhan_bi.perception.features import z_extract
from rhan_bi.types import Encoding


class Backbone(Protocol):
    name: str

    def encode(self, image: np.ndarray) -> Encoding: ...


class HandcraftedBackbone:
    name = "handcrafted"

    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()

    def encode(self, image: np.ndarray) -> Encoding:
        feats, maps = z_extract(image, self.cfg)
        return Encoding(features=feats, maps=maps, meta={"size": list(image.shape)}, backbone=self.name)


class DinoV2Backbone:
    """Optional frozen DINOv2-S feature extractor.

    The checkpoint is expected at ``checkpoints/dinov2_vits14_pretrain.pth`` relative to the
    repository root, or at ``ckpt_path``. Only inference is supported; the model is frozen and the
    handcrafted maps are still computed, so downstream evidence is comparable across backbones.
    """

    name = "dinov2_vits14"

    def __init__(self, ckpt_path: str | Path | None = None, cfg: FeatureConfig | None = None,
                 image_size: int = 224):
        self.cfg = cfg or FeatureConfig()
        self.image_size = int(image_size)
        root = Path(__file__).resolve().parents[3]
        self.ckpt_path = Path(ckpt_path) if ckpt_path else root / "checkpoints" / "dinov2_vits14_pretrain.pth"
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"DINOv2 checkpoint not found at {self.ckpt_path}; use HandcraftedBackbone or pass ckpt_path"
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("DINOv2Backbone requires torch") from exc
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
        state = torch.load(self.ckpt_path, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"checkpoint does not match dinov2_vits14 ({len(missing)} missing keys)")
        model.eval()
        self._model = model
        return model

    def encode(self, image: np.ndarray) -> Encoding:  # pragma: no cover - optional path
        import torch
        import torch.nn.functional as F

        model = self._load()
        feats_h, maps = z_extract(image, self.cfg)
        x = torch.from_numpy(image.astype(np.float32))[None, None]
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x.repeat(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            tokens = model.forward_features(x)["x_norm_patchtokens"]
            cls = model(x)
        return Encoding(
            features=feats_h,
            maps=maps,
            meta={"cls": cls[0].cpu().numpy().tolist(), "patch_tokens": tokens[0].cpu().numpy().tolist()},
            backbone=self.name,
        )

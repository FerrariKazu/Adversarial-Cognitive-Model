"""Stage 0A configuration — frozen DINOv2-S/14 and DINOv2-B/14 linear probes.

Protocol ID: rhan-stage0a-dinov2/1

This module is intentionally separate from the RHAN-NXA/IM/Bi trainers. Stage
0A is a single-observation visual baseline: frozen backbones, independently
trained linear heads, no integrator, no CEE, no backbone fine-tuning.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

PROTOCOL_ID = "rhan-stage0a-dinov2/1"
PROTOCOL_VERSION = "1.0.0"

#: The two backbones required by the assignment.  timm's DINOv2 weights are
#: the official LVD-142M checkpoints; the reference linear numbers are from
#: the DINOv2 model card and are used only as an external sanity reference.
BACKBONES: Dict[str, Dict[str, Any]] = {
    "vit-s14": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "embed_dim": 384,
        "params_m": 22.06,
        "reference_linear_top1": 81.1,
        "official_weights": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    },
    "vit-b14": {
        "timm_name": "vit_base_patch14_dinov2.lvd142m",
        "embed_dim": 768,
        "params_m": 86.58,
        "reference_linear_top1": 84.5,
        "official_weights": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    },
}

SEEDS: Tuple[int, ...] = (17, 42, 99)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_TRAIN_N = 1_281_167
IMAGENET_VAL_N = 50_000
IMAGENET_CLASSES = 1_000

HF_BEST_REPO = "FerrariKazu/rhan-stage0a-dinov2-checkpoints"
HF_ROLLING_REPO = "FerrariKazu/rhan-stage0a-dinov2-checkpoints-rolling"
HF_FEATURE_REPOS: Dict[str, str] = {
    "vit-s14": "FerrariKazu/rhan-stage0a-dinov2-features-vits14",
    "vit-b14": "FerrariKazu/rhan-stage0a-dinov2-features-vitb14",
}


def hf_feature_repo(backbone: str) -> str:
    """Return the dedicated Hugging Face feature cache repository for a backbone."""
    if backbone not in HF_FEATURE_REPOS:
        raise ValueError(f"unknown backbone {backbone!r}; expected {sorted(HF_FEATURE_REPOS)}")
    return HF_FEATURE_REPOS[backbone]


#: Runtime/deployment fields.  They are deliberately excluded from the
#: canonical experiment hash: Kaggle, Colab, and local paths must produce the
#: same experiment identity.
RUNTIME_FIELDS = {
    "data_root",
    "cache_root",
    "output_root",
    "hf_best_repo",
    "hf_rolling_repo",
    "num_workers",
    "require_dual_gpu",
    "allow_single_gpu",
    "hf_token",
}

#: Fixed recipe.  DINOv2's published linear-probe protocol is SGD for 12,500
#: iterations.  We use a fixed, pre-registered recipe rather than tuning on
#: the official validation split.
DEFAULTS: Dict[str, Any] = {
    "img_size": 224,
    "num_classes": IMAGENET_CLASSES,
    "feature_batch_size": 64,
    "eval_batch_size": 512,
    "head_batch_size": 1024,
    "max_steps": 12_500,
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "warmup_steps": 500,
    "eval_every_steps": 500,
    "checkpoint_every_steps": 500,
    "holdout_per_class": 50,
    "amp_dtype": "float16",
    "num_workers": 4,
    "target_top1": 0.80,
    "seed_floor_top1": 0.79,
}


@dataclass(frozen=True)
class Stage0AConfig:
    """Resolved experiment configuration.

    The config is frozen so a caller cannot mutate the experiment identity
    after the checkpoint/run manifest has been written.
    """

    backbone: str
    seed: int
    data_root: str
    cache_root: str
    output_root: str
    img_size: int = DEFAULTS["img_size"]
    num_classes: int = DEFAULTS["num_classes"]
    feature_batch_size: int = DEFAULTS["feature_batch_size"]
    eval_batch_size: int = DEFAULTS["eval_batch_size"]
    head_batch_size: int = DEFAULTS["head_batch_size"]
    max_steps: int = DEFAULTS["max_steps"]
    lr: float = DEFAULTS["lr"]
    momentum: float = DEFAULTS["momentum"]
    weight_decay: float = DEFAULTS["weight_decay"]
    warmup_steps: int = DEFAULTS["warmup_steps"]
    eval_every_steps: int = DEFAULTS["eval_every_steps"]
    checkpoint_every_steps: int = DEFAULTS["checkpoint_every_steps"]
    holdout_per_class: int = DEFAULTS["holdout_per_class"]
    amp_dtype: str = DEFAULTS["amp_dtype"]
    num_workers: int = DEFAULTS["num_workers"]
    target_top1: float = DEFAULTS["target_top1"]
    seed_floor_top1: float = DEFAULTS["seed_floor_top1"]
    hf_best_repo: str = HF_BEST_REPO
    hf_rolling_repo: str = HF_ROLLING_REPO
    require_dual_gpu: bool = True
    allow_single_gpu: bool = False

    def __post_init__(self) -> None:
        if self.backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {self.backbone!r}; expected {sorted(BACKBONES)}")
        if self.img_size % 14 != 0:
            raise ValueError(f"img_size {self.img_size} is not divisible by 14")
        if self.seed not in SEEDS:
            raise ValueError(f"Stage 0A seeds are locked to {SEEDS}; got {self.seed}")
        if self.num_classes != IMAGENET_CLASSES:
            raise ValueError(
                f"Stage 0A requires ImageNet-1k ({IMAGENET_CLASSES} classes); got {self.num_classes}. "
                "Use the smoke/test helpers for reduced-class fixtures."
            )
        if self.holdout_per_class < 1:
            raise ValueError("holdout_per_class must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.checkpoint_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be positive")

    @property
    def backbone_info(self) -> Dict[str, Any]:
        return dict(BACKBONES[self.backbone])

    @property
    def embed_dim(self) -> int:
        return int(BACKBONES[self.backbone]["embed_dim"])

    @property
    def run_id(self) -> str:
        return f"stage0a/{self.backbone}/seed{self.seed}"

    @property
    def experiment_slug(self) -> str:
        return f"stage0a_{self.backbone}_seed{self.seed}_{self.img_size}"

    @property
    def hf_feature_repo(self) -> str:
        return hf_feature_repo(self.backbone)

    def to_dict(self, *, include_runtime: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        d["protocol_id"] = PROTOCOL_ID
        d["protocol_version"] = PROTOCOL_VERSION
        d["backbone_timm_name"] = self.backbone_info["timm_name"]
        d["embed_dim"] = self.embed_dim
        d["seeds"] = list(SEEDS)
        if not include_runtime:
            for k in RUNTIME_FIELDS:
                d.pop(k, None)
        return d

    def canonical_hash(self) -> str:
        payload = json.dumps(self.to_dict(include_runtime=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path: str) -> None:
        from stage0a.utils import atomic_write_json

        atomic_write_json(path, self.to_dict(include_runtime=True))


def configs_for_grid(data_root: str, cache_root: str, output_root: str, **overrides: Any) -> list[Stage0AConfig]:
    """Return the six pre-registered runs (2 backbones x 3 seeds)."""
    out: list[Stage0AConfig] = []
    for backbone in BACKBONES:
        for seed in SEEDS:
            base = dict(
                backbone=backbone,
                seed=seed,
                data_root=data_root,
                cache_root=cache_root,
                output_root=output_root,
            )
            base.update(overrides)
            out.append(Stage0AConfig(**base))
    return out


def feature_cache_name(backbone: str, split: str, img_size: int) -> str:
    return f"{backbone}_{split}_{img_size}px"


def estimate_feature_bytes(backbone: str, n_images: int, dtype_bytes: int = 2) -> int:
    return int(n_images) * int(BACKBONES[backbone]["embed_dim"]) * int(dtype_bytes)

"""Model registry v2 — lazy-loading registry with profiling.

Torchvision baselines (ImageNet) plus all RHAN checkpoints discovered in the
project's checkpoints/ directories (STL-10). Loading is lazy and cached;
profiling reports parameters, latency, memory and an FLOPs estimate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T

from cognitive_vision_lab.config import (
    CHECKPOINTS_DIR,
    CHECKPOINTS_TIER2_DIR,
    CIFAR10_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    STL10_CLASSES,
    STL10_MEAN,
    STL10_STD,
)

# ── Torchvision baselines (ImageNet) ─────────────────────────────────────────
TORCHVISION_MODELS: dict[str, dict] = {
    "ResNet-18": {
        "builder": ("torchvision.models", "resnet18"),
        "family": "CNN",
        "dataset": "ImageNet",
        "paper": "He et al., Deep Residual Learning (CVPR 2016)",
        "params_m": 11.7,
    },
    "EfficientNet-B0": {
        "builder": ("torchvision.models", "efficientnet_b0"),
        "family": "CNN",
        "dataset": "ImageNet",
        "paper": "Tan & Le, EfficientNet (ICML 2019)",
        "params_m": 5.3,
    },
    "ViT-B-16": {
        "builder": ("torchvision.models", "vit_b_16"),
        "family": "Transformer",
        "dataset": "ImageNet",
        "paper": "Dosovitskiy et al., AN IMAGE IS WORTH 16x16 WORDS (ICLR 2021)",
        "params_m": 86.6,
    },
    "Swin-T": {
        "builder": ("torchvision.models", "swin_t"),
        "family": "Transformer",
        "dataset": "ImageNet",
        "paper": "Liu et al., Swin Transformer (ICCV 2021)",
        "params_m": 28.3,
    },
}

# ── RHAN architectures (module name, class name) ──────────────────────────────
RHAN_ARCHES: dict[str, tuple[str, str]] = {
    "rhan_stl10": ("model_rhan_stl10", "RHANSTL10"),
    "rhan_stl10_large": ("model_rhan_stl10_large", "RHANLargeSTL10"),
    "rhan_stl10_pretrained": ("model_rhan_stl10_pretrained", "RHANUnifiedSTL10"),
    "rhan_v10": ("model_rhan_v10", "RHANv10"),
    "rhan_v11": ("model_rhan_v11", "RHANv11"),
    "rhan": ("model_rhan", "RHAN"),
    "rhan_v5": ("model_rhan_v5", "RHANv5"),
}

# Curated display checkpoints (label -> (filename, arch_key, family))
RHAN_MODELS: dict[str, dict] = {
    "RHAN-Large (Pseudolabel rolling)": {
        # Canonical file for the Finding-16 static-large (ep45) anchor; the
        # sweep pipeline resolves ep45 to this checkpoint.
        "checkpoint": "rhan_stl10_large_pseudolabel_rolling.pth",
        "arch": "rhan_stl10_large",
        "family": "Hybrid Recurrent",
    },
    "RHAN-Large (Pseudolabel best)": {
        "checkpoint": "rhan_stl10_large_pseudolabel_best.pth",
        "arch": "rhan_stl10_large",
        "family": "Hybrid Recurrent",
    },
    "RHAN-v10 (Final)": {
        # train_rhan_v10.py saves the final v10 checkpoint under this name.
        "checkpoint": "rhan_stl10_v10_best.pth",
        "arch": "rhan_v10",
        "family": "Active Inference",
    },
    "RHAN-v11 (Best)": {
        "checkpoint": "rhan_stl10_v11_best.pth",
        "arch": "rhan_v11",
        "family": "Active Inference",
    },
    "RHAN-v11 (Isolation Run A)": {
        "checkpoint": "rhan_v11_isolation_norecon_best.pth",
        "arch": "rhan_v11",
        "family": "Hybrid Recurrent",
    },
    "RHAN-v11 (Isolation Run B)": {
        "checkpoint": "rhan_v11_isolation_fixedgaze_best.pth",
        "arch": "rhan_v11",
        "family": "Hybrid Recurrent (frozen gaze)",
        # Trained with --freeze-gaze; inference must freeze the foveal gaze to
        # center (0,0) or predictions won't match the evaluated behavior.
        "freeze_gaze": True,
    },
}

CKPT_SUFFIXES = [
    "_best", "_final", "_rolling", "_checkpoint", "_ep10", "_ep45", "_start",
    "_phase_a_final", "_phase_b_final", "_phase_c_final",
    "_trades", "_trades_actual", "_trades_clean_consistency", "_trades_rolling",
    "_labeled", "_labeled_rolling", "_pretrained", "_pretrained_rolling",
]


def _stem(filename: str) -> str:
    name = filename.replace(".pth", "").replace(":Zone.Identifier", "")
    for suffix in CKPT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


# Prefix priority — longest/most-specific first (rhan_stl10_v11 is a RHANv11
# checkpoint despite the stl10 prefix). Maps filename prefixes to RHAN_ARCHES keys.
_ARCH_PRIORITY: list[tuple[str, str]] = [
    ("rhan_stl10_v11", "rhan_v11"),
    ("rhan_stl10_v10", "rhan_v10"),
    ("rhan_stl10_large", "rhan_stl10_large"),
    ("rhan_stl10_pretrained", "rhan_stl10_pretrained"),
    ("rhan_stl10", "rhan_stl10"),
    ("rhan_v11", "rhan_v11"),
    ("rhan_v10", "rhan_v10"),
    ("rhan_v5", "rhan_v5"),
    ("rhan", "rhan"),
]
_MODULE_TO_KEY = {v[0]: k for k, v in RHAN_ARCHES.items()}


def _resolve_arch(filename: str) -> Optional[tuple[str, str]]:
    stem = _stem(filename)
    if stem in RHAN_ARCHES:
        return RHAN_ARCHES[stem]
    for prefix, arch_key in _ARCH_PRIORITY:
        if stem.startswith(prefix):
            return RHAN_ARCHES[arch_key]
    return None


def _arch_key_for(filename: str) -> str:
    arch = _resolve_arch(filename)
    if arch is None:
        return "rhan"
    return _MODULE_TO_KEY.get(arch[0], "rhan")


@dataclass
class ModelHandle:
    model: nn.Module
    transform: T.Compose
    is_stl10: bool
    name: str
    family: str
    dataset: str
    checkpoint: Optional[Path] = None
    profile: dict = field(default_factory=dict)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return torch.softmax(out.float(), dim=1)


# ── Transform builders ────────────────────────────────────────────────────────
def _transform(stl10: bool) -> T.Compose:
    if stl10:
        return T.Compose([
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize(STL10_MEAN, STL10_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Discovery ─────────────────────────────────────────────────────────────────
def _checkpoint_dirs() -> list[Path]:
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_TIER2_DIR.exists():
        dirs.append(CHECKPOINTS_TIER2_DIR)
    return dirs


def discover_checkpoints() -> dict[str, dict]:
    """filename -> {path, arch, arch_key, family} for loadable RHAN checkpoints."""
    found: dict[str, dict] = {}
    for cdir in _checkpoint_dirs():
        if not cdir.exists():
            continue
        for fpath in cdir.iterdir():
            if not fpath.name.endswith(".pth") or ":Zone.Identifier" in fpath.name:
                continue
            arch = _resolve_arch(fpath.name)
            if arch is None:
                continue
            found[fpath.name] = {
                "path": fpath,
                "arch": arch,
                "family": "RHAN" if not fpath.name.startswith("rhan_stl10_large") else "Hybrid Recurrent",
            }
    return found


def list_models() -> list[dict]:
    """All selectable models with availability flags (Streamlit-friendly)."""
    rows: list[dict] = []
    for mid, info in TORCHVISION_MODELS.items():
        rows.append({
            "id": mid,
            "name": mid,
            "family": info["family"],
            "dataset": info["dataset"],
            "stl10": False,
            "available": True,
            "checkpoint": None,
            "source": "torchvision",
        })
    for label, info in RHAN_MODELS.items():
        ckpt = CHECKPOINTS_DIR / info["checkpoint"]
        rows.append({
            "id": f"rhan:{label}",
            "name": label,
            "family": info["family"],
            "dataset": "STL-10",
            "stl10": True,
            "available": ckpt.exists(),
            "checkpoint": info["checkpoint"],
            "source": "checkpoint",
        })
    for fname, info in discover_checkpoints().items():
        if fname in {e["checkpoint"] for e in RHAN_MODELS.values()}:
            continue
        rows.append({
            "id": f"ckpt:{fname}",
            "name": _stem(fname).replace("_", " ").title().replace("Stl", "STL").replace("Rhan", "RHAN"),
            "family": info["family"],
            "dataset": "STL-10",
            "stl10": True,
            "available": True,
            "checkpoint": fname,
            "source": "checkpoint",
        })
    return rows


# ── Loading ───────────────────────────────────────────────────────────────────
_CACHE: dict[str, ModelHandle] = {}


def _instantiate_rhan(arch_key: str) -> nn.Module:
    import importlib
    import sys

    sys.path.insert(0, str(CHECKPOINTS_DIR.parent / "phase1_training"))
    mod_name, cls_name = RHAN_ARCHES[arch_key]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if arch_key == "rhan_v11":
        return cls(max_foraging_steps=4, fovea_size=48, metabolic_cost=0.05)
    return cls()


def _load_state(model: nn.Module, path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in state:
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    return len(state) - len(missing)


def load_model(model_id: str, device: Optional[str] = None, force: bool = False) -> ModelHandle:
    """Load a model by id. Ids: torchvision name, 'rhan:<label>', or 'ckpt:<file>'."""
    if model_id in _CACHE and not force:
        return _CACHE[model_id]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    handle: Optional[ModelHandle] = None
    if model_id in TORCHVISION_MODELS:
        info = TORCHVISION_MODELS[model_id]
        mod_name, cls_name = info["builder"]
        import importlib

        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        model = None
        try:
            model = cls(weights="DEFAULT")
        except Exception:
            try:
                model = cls(weights=None)
            except Exception:
                model = cls()
        model.eval()
        handle = ModelHandle(
            model=model.to(device), transform=_transform(False),
            is_stl10=False, name=model_id, family=info["family"],
            dataset="ImageNet",
        )
    else:
        filename: Optional[str] = None
        label = model_id
        if model_id.startswith("rhan:"):
            label = model_id[5:]
            entry = RHAN_MODELS[label]
            filename = entry["checkpoint"]
        elif model_id.startswith("ckpt:"):
            filename = model_id[5:]
        if filename is None:
            raise ValueError(f"Unknown model_id: {model_id}")
        arch_key = None
        if model_id.startswith("rhan:") and label in RHAN_MODELS:
            arch_key = RHAN_MODELS[label]["arch"]
        else:
            arch_key = _arch_key_for(filename)
        path = None
        for d in _checkpoint_dirs():
            cand = d / filename
            if cand.exists():
                path = cand
                break
        if path is None:
            raise FileNotFoundError(f"Checkpoint {filename} not found in {CHECKPOINTS_DIR}")
        freeze_gaze = False
        if model_id.startswith("rhan:") and label in RHAN_MODELS:
            freeze_gaze = bool(RHAN_MODELS[label].get("freeze_gaze", False))
        model = _instantiate_rhan(arch_key)
        if freeze_gaze and hasattr(model, "freeze_gaze"):
            model.freeze_gaze = True
        n_loaded = _load_state(model, path)
        model.eval()
        family = "RHAN"
        if "rhan_stl10_large" in arch_key:
            family = "Hybrid Recurrent"
        handle = ModelHandle(
            model=model.to(device), transform=_transform(True),
            is_stl10=True, name=label, family=family, dataset="STL-10",
            checkpoint=path,
        )
    _CACHE[model_id] = handle
    return handle


def clear_cache() -> None:
    _CACHE.clear()


# ── Profiling ─────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def model_size_mb(model: nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / 1e6


def estimate_flops(model: nn.Module, input_shape: tuple[int, ...]) -> float:
    """Lightweight conv/linear FLOPs counter (MACs * 2), no external deps."""
    hooks, flops = [], {"value": 0.0}

    def hook_fn(module, inp, out):
        x = inp[0]
        if isinstance(module, nn.Conv2d):
            n, c, h, w = x.shape
            kh, kw = module.kernel_size
            flops["value"] += (
                2.0 * n * module.out_channels * (h // module.stride[0]) * (w // module.stride[1])
                * c * kh * kw
            )
        elif isinstance(module, nn.Linear):
            flops["value"] += 2.0 * x.shape[0] * x.shape[-1] * module.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape, device=next(model.parameters()).device))
    except Exception:
        pass
    for h in hooks:
        h.remove()
    return flops["value"] / 1e9  # GFLOPs


def profile_model(handle: ModelHandle, input_shape: tuple[int, ...] = (3, 96, 96),
                  trials: int = 5) -> dict:
    """Parameters, size, GFLOPs, latency (ms) — measured lazily."""
    model = handle.model
    dev = next(model.parameters()).device
    params_m = count_parameters(model)
    size_mb = model_size_mb(model)
    flops_g = estimate_flops(model, input_shape)
    x = torch.randn(1, *input_shape, device=dev)
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(trials):
            _ = model(x)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        latency_ms = (time.perf_counter() - t0) / trials * 1000.0
    handle.profile = {
        "params_m": round(params_m, 2),
        "size_mb": round(size_mb, 1),
        "gflops": round(flops_g, 2),
        "latency_ms": round(latency_ms, 2),
        "device": str(dev),
    }
    return handle.profile


def class_names(handle: ModelHandle) -> list[str]:
    if handle.is_stl10:
        return STL10_CLASSES
    if handle.dataset == "CIFAR-10":
        return CIFAR10_CLASSES
    # ImageNet
    try:
        import json
        import urllib.request

        url = ("https://raw.githubusercontent.com/raghakot/keras-vis/master/"
               "resources/imagenet_class_index.json")
        with urllib.request.urlopen(url, timeout=5) as f:
            data = json.load(f)
        return [data[str(k)][1] for k in range(1000)]
    except Exception:
        return [f"class_{i}" for i in range(1000)]

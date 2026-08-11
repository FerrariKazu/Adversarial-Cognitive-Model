"""Image I/O: normalization helpers, sample loaders, procedural fallbacks."""
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from cognitive_vision_lab.config import (
    CIFAR10_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    STL10_CLASSES,
    STL10_MEAN,
    STL10_STD,
)


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def denormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    return x * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)


def tensor_to_pil(x: torch.Tensor, mean, std) -> Image.Image:
    """Convert a (1,3,H,W) normalized tensor to a PIL image in [0,1]."""
    img = denormalize(x.detach().cpu().float(), mean, std)
    img = img.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def pil_to_tensor(img: Image.Image, mean, std, size: int) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return normalize(t.unsqueeze(0), mean, std)


def procedural_sample(class_name: Optional[str] = None, size: int = 96) -> Image.Image:
    """Generate a clean procedural demo image (colored shape on gradient).

    Used when no real dataset is cached so every page remains functional offline.
    """
    rng = np.random.default_rng(abs(hash(class_name or "demo")) % 2**31)
    img = Image.new("RGB", (size, size))
    px = img.load()
    base = np.array([rng.integers(60, 200), rng.integers(60, 200), rng.integers(60, 200)])
    for y in range(size):
        for x in range(size):
            shade = 1.0 - 0.4 * (x / size)
            px[x, y] = tuple(int(c * shade) for c in base)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 5
    shape = (class_name or "demo").lower()
    color = (230, 180, 40)
    if "car" in shape or "truck" in shape or "automobile" in shape:
        draw.rounded_rectangle([cx - r, cy - r // 2, cx + r, cy + r // 2], radius=8, fill=color)
    elif "bird" in shape or "plane" in shape or "airplane" in shape:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif "cat" in shape or "dog" in shape or "deer" in shape or "horse" in shape or "monkey" in shape:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=6)
        draw.line([cx - r, cy, cx + r, cy], fill=color, width=4)
        draw.line([cx, cy - r, cx, cy + r], fill=color, width=4)
    return img


def stl10_demo_sample(label_idx: int = 3) -> Image.Image:
    """Demo STL-10 image; deterministic per class."""
    name = STL10_CLASSES[label_idx % len(STL10_CLASSES)]
    return procedural_sample(name, STL10_CLASSES and 96)


def load_stl10_sample(index: int = 0) -> Optional[Tuple[Image.Image, int]]:
    """Load a real STL-10 test image if the dataset is cached; else None."""
    try:
        from datasets import load_dataset

        ds = load_dataset("mteb/stl10", split="test", streaming=True).shuffle(seed=42)
        item = list(ds.take(index + 1))[-1]
        return item["image"].convert("RGB"), item["label"]
    except Exception:
        return None


def class_names_for(dataset: str) -> list[str]:
    ds = (dataset or "stl10").lower()
    if "cifar10" in ds or "cifar-10" in ds:
        return CIFAR10_CLASSES
    if "stl" in ds:
        return STL10_CLASSES
    return IMAGENET_MEAN and ["class_" + str(i) for i in range(1000)]  # ImageNet fallback

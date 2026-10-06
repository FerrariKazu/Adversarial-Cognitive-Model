"""
ImageNet-C corruption-evaluation infrastructure — Agent I (data infra ONLY).
================================================================================

The corruption suite is Hendrycks & Dietterich's published benchmark
("Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations", arXiv:1903.12261 / ICLR 2019): 15 corruptions x 5
severities. These names and the 1-5 severity scale are the benchmark's own
protocol constants — recorded, not invented. Part 2 step 11's Tier-1
evaluation includes ImageNet-C.
"""
from __future__ import annotations

import os

#: The benchmark's 15 corruptions (arXiv:1903.12261, Table 1 order).
IMAGENET_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)

#: The benchmark's severity scale.
IMAGENET_C_SEVERITIES = (1, 2, 3, 4, 5)

#: Standard ImageNet normalization (same stats as the clean set).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def corruption_dir_name(corruption: str, severity: int) -> str:
    """ImageNet-C's on-disk layout: <root>/<corruption>/<severity>/."""
    if corruption not in IMAGENET_C_CORRUPTIONS:
        raise ValueError(
            f"unknown corruption {corruption!r} — the benchmark's 15 are "
            f"{IMAGENET_C_CORRUPTIONS}")
    if severity not in IMAGENET_C_SEVERITIES:
        raise ValueError(
            f"severity must be one of {IMAGENET_C_SEVERITIES} (the "
            f"benchmark's scale); got {severity}")
    return os.path.join(corruption, str(severity))


def validate_imagenet_c_root(root: str, require_all: bool = False) -> int:
    """Structural validation of an ImageNet-C root.

    Checks which <corruption>/<severity> directories exist. With
    require_all=True, raises unless the full 15x5 grid is present
    (a partial grid silently reported as full coverage would
    misrepresent robustness); with the default, returns the count of
    present cells and the caller reports coverage explicitly.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"ImageNet-C root not found: {root}")
    present = 0
    missing = []
    for c in IMAGENET_C_CORRUPTIONS:
        for s in IMAGENET_C_SEVERITIES:
            if os.path.isdir(os.path.join(root, corruption_dir_name(c, s))):
                present += 1
            else:
                missing.append(corruption_dir_name(c, s))
    if require_all and missing:
        raise ValueError(
            f"ImageNet-C root {root} is missing {len(missing)} of 75 "
            f"corruption/severity cells (e.g. {missing[:3]}) — full-grid "
            f"coverage was required; report partial coverage explicitly "
            f"instead")
    return present


def make_imagenet_c_loader(root: str, corruption: str, severity: int,
                           batch_size: int, num_workers: int = 4):
    """Loader for one corruption/severity cell (validated on demand)."""
    cell = os.path.join(root, corruption_dir_name(corruption, severity))
    if not os.path.isdir(cell):
        raise FileNotFoundError(
            f"ImageNet-C cell missing: {cell} — validate the root first "
            f"(validate_imagenet_c_root) and report coverage explicitly")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(cell, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)

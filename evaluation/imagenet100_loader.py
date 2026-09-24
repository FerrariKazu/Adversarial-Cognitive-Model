"""
ImageNet-100 data infrastructure — Agent I (data infra ONLY).
================================================================================

No training, no attack logic — loaders and validation. The harness's first
REAL exercise waits for Agent J's checkpoint (Agent I contract note).

ImageNet-100 = the 100-class subset the plan's staged runs train on
(Part 2). Validation is structural: a valid root has EXACTLY 100 class
subdirectories (the subset's defining property) — a root with 99 or 101
classes is a data error that must surface at load time, never mid-run.
"""
from __future__ import annotations

import os

#: Dataset identity constants (the benchmark's own published facts).
IMAGENET100_NUM_CLASSES = 100
IMAGENET100_IMG_SIZE = 96          # the plan's compact-substrate operating point

#: Standard ImageNet normalization (the benchmark's published stats).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_imagenet100_root(root: str, split: str = "val") -> int:
    """Structural validation of an ImageNet-100 root directory.

    Returns the class count (must be exactly 100). Raises (loudly) when
    the root is missing, not a directory, or holds the wrong class count —
    a silently-wrong dataset is a run invalidating its own result.
    """
    split_dir = os.path.join(root, split) if os.path.isdir(
        os.path.join(root, split)) else root
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"ImageNet-100 root not found: {split_dir} (root={root!r})")
    classes = sorted(d for d in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, d)))
    if len(classes) != IMAGENET100_NUM_CLASSES:
        raise ValueError(
            f"ImageNet-100 root {split_dir} holds {len(classes)} class "
            f"directories, expected exactly {IMAGENET100_NUM_CLASSES} — "
            f"refusing to load a silently-wrong subset")
    return len(classes)


def make_imagenet100_loaders(root: str, batch_size: int, num_workers: int = 4,
                             img_size: int = IMAGENET100_IMG_SIZE):
    """Train/val loaders for a validated ImageNet-100 root.

    Requires torchvision (imported lazily so the validation and constants
    above are usable in data-less environments, e.g. harness self-tests).
    """
    validate_imagenet100_root(root, split="train")
    validate_imagenet100_root(root, split="val")
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train = datasets.ImageFolder(os.path.join(root, "train"), transform=tf_train)
    val = datasets.ImageFolder(os.path.join(root, "val"), transform=tf_val)
    if len(train.classes) != IMAGENET100_NUM_CLASSES or \
            len(val.classes) != IMAGENET100_NUM_CLASSES:
        raise ValueError("ImageFolder class count mismatch after load")
    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "num_classes": IMAGENET100_NUM_CLASSES,
    }


def make_synthetic_loaders(batch_size: int = 8, batches: int = 2,
                           img_size: int = IMAGENET100_IMG_SIZE,
                           num_classes: int = IMAGENET100_NUM_CLASSES,
                           seed: int = 0):
    """SYNTHETIC random-tensor loaders — for harness self-tests ONLY.

    Marked loudly: numbers produced against these are NOT results (the
    eval_rhan convention — dev sanity is an explicit, documented path,
    never reported as evidence).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed)

    def _ds(n):
        x = torch.rand(n, 3, img_size, img_size, generator=g)
        y = torch.randint(0, num_classes, (n,), generator=g)
        return TensorDataset(x, y)

    def _dl(n):
        return DataLoader(_ds(n * batch_size), batch_size=batch_size)

    return {"train": _dl(batches), "val": _dl(batches),
            "num_classes": num_classes, "synthetic": True}

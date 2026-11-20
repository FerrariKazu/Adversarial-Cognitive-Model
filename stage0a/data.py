"""ImageNet-1k discovery, deterministic splits, fingerprints, and loaders.

The official validation split is a one-shot test set for Stage 0A.  We carve a
per-class holdout out of *train* for best-checkpoint selection; the official
validation images are never used to choose an epoch or a threshold.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from stage0a.config import IMAGENET_MEAN, IMAGENET_STD
from stage0a.utils import atomic_write_json, atomic_write_text, sha256_text

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _sorted_classes(root: Path) -> List[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing directory: {root}")
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


@dataclass
class ImageNetIndex:
    root: str
    split: str
    paths: List[str]
    labels: np.ndarray
    class_names: List[str]
    fingerprint: str
    source_kind: str

    def __len__(self) -> int:
        return len(self.paths)

    def absolute(self, i: int) -> Path:
        return Path(self.root) / self.paths[i]

    def summary(self) -> Dict[str, Any]:
        counts = np.bincount(self.labels, minlength=len(self.class_names)).astype(int).tolist()
        return {
            "root": self.root,
            "split": self.split,
            "count": int(len(self.labels)),
            "num_classes": len(self.class_names),
            "fingerprint": self.fingerprint,
            "source_kind": self.source_kind,
            "min_class_count": int(min(counts)) if counts else 0,
            "max_class_count": int(max(counts)) if counts else 0,
            "class_names": self.class_names,
        }


def _discover_candidates(root: Path) -> List[Tuple[Path, Path]]:
    """Return (train_dir, val_dir) candidates in priority order."""
    candidates = [
        (root / "train", root / "val"),
        (root / "imagenet1k" / "train", root / "imagenet1k" / "val"),
        (root / "ILSVRC" / "Data" / "CLS-LOC" / "train", root / "ILSVRC" / "Data" / "CLS-LOC" / "val"),
        (root / "Data" / "CLS-LOC" / "train", root / "Data" / "CLS-LOC" / "val"),
        (root / "imagenet-1k" / "train", root / "imagenet-1k" / "val"),
    ]
    return candidates


def discover_imagenet1k(root: str | Path) -> Tuple[Path, Path]:
    root = Path(root).expanduser().resolve()
    for train, val in _discover_candidates(root):
        if train.is_dir() and val.is_dir():
            return train, val
    raise FileNotFoundError(
        "Could not find ImageNet-1k train/val. Expected one of: "
        f"{[str(a) + ' + ' + str(b) for a, b in _discover_candidates(root)]}. "
        "Attach the official ImageNet-1k Kaggle dataset and set STAGE0A_DATA_ROOT."
    )


def _csv_val_labels(val_dir: Path, class_names: Sequence[str]) -> Tuple[List[Path], np.ndarray, str]:
    """Read flat validation images + official LOC_val_solution.csv."""
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    csv_candidates = [
        val_dir / "LOC_val_solution.csv",
        val_dir.parent / "LOC_val_solution.csv",
        val_dir.parent.parent / "LOC_val_solution.csv",
        val_dir / "val_labels.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Validation at {val_dir} is not class-foldered and no LOC_val_solution.csv "
            "or val_labels.csv was found. Official ImageNet-1k evaluation needs labels."
        )

    rows: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = {c.strip() for c in (reader.fieldnames or [])}
        for rec in reader:
            if "ImageId" in fields and "PredictionString" in fields:
                image_id = rec["ImageId"].strip()
                label = rec["PredictionString"].strip().split()[0]
            elif "image" in fields and "label" in fields:
                image_id = rec["image"].strip()
                label = rec["label"].strip()
            else:
                raise ValueError(f"Unsupported validation CSV columns: {sorted(fields)}")
            rows[image_id] = label

    image_paths = [p for p in sorted(val_dir.iterdir()) if p.is_file() and _is_image(p)]
    paths: List[str] = []
    labels: List[int] = []
    for p in image_paths:
        image_id = p.stem
        label = rows.get(image_id)
        if label is None:
            raise KeyError(f"Validation image {p.name} has no label in {csv_path}")
        if label not in name_to_idx:
            raise KeyError(f"Validation label {label!r} is not in the train class list")
        paths.append(str(p.relative_to(val_dir.parent)))
        labels.append(name_to_idx[label])
    if len(paths) != len(rows):
        raise ValueError(f"Validation CSV has {len(rows)} rows but found {len(paths)} images in {val_dir}")
    return paths, np.asarray(labels, dtype=np.int64), "flat_csv"


def build_index(root: str | Path, split: str, *, expected_val_count: Optional[int] = 50_000) -> ImageNetIndex:
    """Build a deterministic index and content-independent fingerprint.

    The fingerprint hashes relative path, file size, and label for every image.
    It is a *split provenance* fingerprint, not a claim that every pixel byte was
    hashed.  Image bytes are hashed separately by the feature cache manifest.
    """
    train_dir, val_dir = discover_imagenet1k(root)
    class_names = _sorted_classes(train_dir)
    if len(class_names) != 1000:
        raise ValueError(
            f"ImageNet-1k requires 1000 train classes; found {len(class_names)} at {train_dir}. "
            "This pipeline refuses to silently substitute ImageNet-100."
        )

    if split == "train":
        split_dir = train_dir
        paths = []
        labels = []
        for ci, cname in enumerate(class_names):
            for p in sorted((train_dir / cname).iterdir()):
                if p.is_file() and _is_image(p):
                    paths.append(str(p.relative_to(train_dir)))
                    labels.append(ci)
        labels_arr = np.asarray(labels, dtype=np.int64)
        source_kind = "class_folders"
        base = train_dir
    elif split == "val":
        split_dir = val_dir
        val_classes = _sorted_classes(val_dir)
        if len(val_classes) >= 2:
            val_map = {c: i for i, c in enumerate(val_classes)}
            if set(val_classes) != set(class_names):
                # Common case: val contains only a subset in a smoke fixture.
                # For the real protocol, require exact class parity.
                if len(val_classes) != len(class_names):
                    raise ValueError(
                        f"Validation classes ({len(val_classes)}) do not match train classes "
                        f"({len(class_names)}); official evaluation requires 1000 classes."
                    )
            paths = []
            labels = []
            for cname in val_classes:
                for p in sorted((val_dir / cname).iterdir()):
                    if p.is_file() and _is_image(p):
                        paths.append(str(p.relative_to(val_dir)))
                        labels.append(val_map[cname])
            labels_arr = np.asarray(labels, dtype=np.int64)
            source_kind = "class_folders"
            base = val_dir
        else:
            paths, labels_arr, source_kind = _csv_val_labels(val_dir, class_names)
            base = val_dir.parent
        if expected_val_count is not None and len(paths) != expected_val_count:
            raise ValueError(
                f"Official ImageNet-1k validation must have {expected_val_count} images; "
                f"found {len(paths)} in {val_dir}."
            )
    else:
        raise ValueError(f"unknown split {split!r}")

    h = sha256_text("\n".join(class_names))
    for rel, y in sorted(zip(paths, labels_arr.tolist())):
        size = (base / rel).stat().st_size
        h = sha256_text(h + f"\0{rel}\0{size}\0{y}\n")
    return ImageNetIndex(
        root=str(base),
        split=split,
        paths=paths,
        labels=labels_arr,
        class_names=class_names,
        fingerprint=h,
        source_kind=source_kind,
    )


def save_index(index: ImageNetIndex, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "paths.txt").write_text("\n".join(index.paths) + "\n", encoding="utf-8")
    np.save(directory / "labels.npy", index.labels)
    atomic_write_json(
        directory / "index.json",
        {
            "root": index.root,
            "split": index.split,
            "fingerprint": index.fingerprint,
            "source_kind": index.source_kind,
            "class_names": index.class_names,
            "count": len(index.paths),
        },
    )
    return directory


def load_index(directory: str | Path) -> ImageNetIndex:
    directory = Path(directory)
    meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    paths = (directory / "paths.txt").read_text(encoding="utf-8").splitlines()
    labels = np.load(directory / "labels.npy")
    if len(paths) != len(labels):
        raise ValueError("index path/label length mismatch")
    return ImageNetIndex(
        root=meta["root"],
        split=meta["split"],
        paths=paths,
        labels=labels.astype(np.int64, copy=False),
        class_names=meta["class_names"],
        fingerprint=meta["fingerprint"],
        source_kind=meta["source_kind"],
    )


def stratified_holdout(labels: np.ndarray, per_class: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, holdout_indices), deterministic per seed."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_chunks: List[np.ndarray] = []
    holdout_chunks: List[np.ndarray] = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) <= per_class:
            raise ValueError(f"class {c} has only {len(idx)} images; cannot hold out {per_class}")
        perm = rng.permutation(idx)
        holdout_chunks.append(perm[:per_class])
        train_chunks.append(perm[per_class:])
    train_idx = np.sort(np.concatenate(train_chunks)).astype(np.int64)
    holdout_idx = np.sort(np.concatenate(holdout_chunks)).astype(np.int64)
    return train_idx, holdout_idx


def split_fingerprint(dataset_fingerprint: str, train_idx: np.ndarray, holdout_idx: np.ndarray, seed: int) -> str:
    h = sha256_text(dataset_fingerprint + f"|seed={seed}|n={len(train_idx)}/{len(holdout_idx)}")
    h = sha256_text(h + "|train=" + ",".join(map(str, train_idx.tolist())))
    h = sha256_text(h + "|holdout=" + ",".join(map(str, holdout_idx.tolist())))
    return h


def make_transform(img_size: int) -> Any:
    import torchvision.transforms as T

    resize = int(round(img_size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class IndexedImageDataset(Dataset):
    def __init__(self, index: ImageNetIndex, transform: Any, indices: Optional[np.ndarray] = None):
        self.index = index
        self.transform = transform
        self.indices = np.arange(len(index)) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int) -> Tuple[torch.Tensor, int, int]:
        gi = int(self.indices[j])
        p = self.index.absolute(gi)
        with Image.open(p) as im:
            im = im.convert("RGB")
        return self.transform(im), int(self.index.labels[gi]), gi


def class_counts(labels: np.ndarray, num_classes: int) -> List[int]:
    return np.bincount(labels, minlength=num_classes).astype(int).tolist()


def split_summary(index: ImageNetIndex, train_idx: np.ndarray, holdout_idx: np.ndarray, seed: int) -> Dict[str, Any]:
    return {
        "dataset_fingerprint": index.fingerprint,
        "split_fingerprint": split_fingerprint(index.fingerprint, train_idx, holdout_idx, seed),
        "seed": seed,
        "train_count": int(len(train_idx)),
        "holdout_count": int(len(holdout_idx)),
        "holdout_per_class": int(len(holdout_idx) // len(index.class_names)),
        "official_val_count": 50_000,
        "official_val_used_for_selection": False,
        "class_counts": class_counts(index.labels, len(index.class_names)),
    }


def audit_dataset(root: str | Path) -> Dict[str, Any]:
    """Non-destructive dataset availability/capacity audit."""
    root = Path(root).expanduser().resolve()
    out: Dict[str, Any] = {"requested_root": str(root)}
    try:
        train, val = discover_imagenet1k(root)
        out["train_dir"] = str(train)
        out["val_dir"] = str(val)
        train_classes = _sorted_classes(train)
        val_classes = _sorted_classes(val)
        out["train_class_dirs"] = len(train_classes)
        out["val_class_dirs"] = len(val_classes)
        out["train_images"] = sum(1 for c in train_classes for p in (train / c).iterdir() if p.is_file() and _is_image(p))
        if val_classes:
            out["val_images"] = sum(1 for c in val_classes for p in (val / c).iterdir() if p.is_file() and _is_image(p))
            out["val_layout"] = "class_folders"
        else:
            out["val_images"] = sum(1 for p in val.iterdir() if p.is_file() and _is_image(p))
            out["val_layout"] = "flat"
        out["official_val_ready"] = out.get("val_images") == 50_000 and (
            len(val_classes) == 1000 or out.get("val_layout") == "flat"
        )
        out["status"] = "ok" if len(train_classes) == 1000 and out.get("official_val_ready") else "invalid"
    except Exception as e:
        out["status"] = "missing_or_invalid"
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def audit_storage(paths: Iterable[str | Path]) -> Dict[str, Any]:
    from stage0a.utils import disk_usage, human_bytes

    out = {}
    for p in paths:
        p = Path(p)
        exists = p.exists()
        rec = {"path": str(p), "exists": exists}
        if exists:
            usage = disk_usage(p if p.is_dir() else p.parent)
            rec.update(usage)
            rec["free_human"] = human_bytes(usage["free"])
        out[str(p)] = rec
    return out

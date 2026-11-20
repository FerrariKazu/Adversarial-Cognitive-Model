"""Frozen-backbone feature extraction and pooled-feature cache.

Only the pooled CLS feature is persisted.  Full spatial patch-token maps are
deliberately not cached: they are unnecessary for a linear probe and would make
ImageNet-1k storage explode.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage0a.config import Stage0AConfig
from stage0a.data import ImageNetIndex, IndexedImageDataset, load_index, make_transform
from stage0a.utils import (
    atomic_write_json,
    human_bytes,
    json_load,
    sha256_file,
    state_dict_sha256,
    utc_now,
)


# ── backbone ────────────────────────────────────────────────────────────────
def load_backbone(cfg: Stage0AConfig, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """Load the official frozen DINOv2 weights through the project's timm dep."""
    import timm

    info = cfg.backbone_info
    model = timm.create_model(
        info["timm_name"],
        pretrained=True,
        num_classes=0,
        img_size=cfg.img_size,
        dynamic_img_size=True,
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    weights_sha = state_dict_sha256(model.state_dict())
    return model, weights_sha


# ── GPU monitor ─────────────────────────────────────────────────────────────
class GPUMonitor:
    """Best-effort nvidia-smi sampler for one physical GPU."""

    def __init__(self, physical_index: int, interval_s: float = 0.5):
        self.physical_index = int(physical_index)
        self.interval_s = interval_s
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                p = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for line in p.stdout.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 3 and int(parts[0]) == self.physical_index:
                        self.samples.append((float(parts[1]), float(parts[2])))
                        break
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "GPUMonitor":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        utils = [s[0] for s in self.samples]
        mem = [s[1] for s in self.samples]
        return {
            "samples": len(self.samples),
            "util_mean_pct": float(np.mean(utils)) if utils else None,
            "util_max_pct": float(np.max(utils)) if utils else None,
            "memory_used_max_mib": float(np.max(mem)) if mem else None,
        }


# ── cache paths / metadata ──────────────────────────────────────────────────
@dataclass
class FeatureCache:
    directory: Path
    features: Path
    labels: Path
    meta: Dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.meta["count"])

    @property
    def dim(self) -> int:
        return int(self.meta["dim"])

    def store(self) -> np.memmap:
        return np.load(self.features, mmap_mode="r")


def cache_name(cfg: Stage0AConfig, split: str, dataset_fingerprint: str) -> str:
    return f"{cfg.backbone}_{split}_{cfg.img_size}px_{dataset_fingerprint[:12]}"


def cache_dir(cfg: Stage0AConfig, split: str, dataset_fingerprint: str) -> Path:
    return Path(cfg.cache_root) / "features" / cache_name(cfg, split, dataset_fingerprint)


def expected_meta(cfg: Stage0AConfig, index: ImageNetIndex, split: str) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "backbone": cfg.backbone,
        "backbone_timm_name": cfg.backbone_info["timm_name"],
        "split": split,
        "img_size": cfg.img_size,
        "dim": cfg.embed_dim,
        "count": len(index),
        "dtype": "float16",
        "representation": "pooled_cls_token",
        "dataset_fingerprint": index.fingerprint,
        "class_names_sha256": __import__("hashlib").sha256(
            "\n".join(index.class_names).encode()
        ).hexdigest(),
        "class_names": list(index.class_names),
        "transform": "Resize(round(img_size*256/224), bicubic) + CenterCrop(img_size) + Normalize(ImageNet)",
    }


def prepare_cache(cfg: Stage0AConfig, index: ImageNetIndex, split: str, n_shards: int) -> FeatureCache:
    directory = cache_dir(cfg, split, index.fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    features = directory / "features.npy"
    labels_path = directory / "labels.npy"
    meta_path = directory / "meta.json"
    expected = expected_meta(cfg, index, split)

    if meta_path.exists():
        meta = json_load(meta_path)
        if any(meta.get(k) != v for k, v in expected.items()):
            raise RuntimeError(
                f"feature cache metadata mismatch at {directory}; refusing to reuse a cache "
                "built from a different dataset/backbone/resolution"
            )
    else:
        meta = expected
        atomic_write_json(meta_path, meta)

    if not labels_path.exists():
        np.save(labels_path, index.labels.astype(np.int64))
    elif not np.array_equal(np.load(labels_path, mmap_mode="r"), index.labels):
        raise RuntimeError("feature cache labels do not match the current dataset index")

    if not features.exists():
        arr = np.lib.format.open_memmap(
            features, mode="w+", dtype=np.float16, shape=(len(index), cfg.embed_dim)
        )
        arr.flush()
        del arr
    else:
        arr = np.load(features, mmap_mode="r")
        if arr.shape != (len(index), cfg.embed_dim):
            raise RuntimeError(f"feature cache shape {arr.shape} != {(len(index), cfg.embed_dim)}")
        del arr

    # Remove a stale completion marker before any new extraction.  A partial
    # cache is never treated as complete.
    for marker in directory.glob("complete.json"):
        marker.unlink()
    atomic_write_json(
        directory / "reservation.json",
        {
            "created_at_utc": utc_now(),
            "n_shards": int(n_shards),
            "dataset_fingerprint": index.fingerprint,
            "status": "reserved",
        },
    )
    return FeatureCache(directory=directory, features=features, labels=labels_path, meta=meta)


def shard_bounds(n: int, rank: int, n_shards: int) -> Tuple[int, int]:
    per = int(math.ceil(n / n_shards))
    start = rank * per
    end = min(n, start + per)
    return start, end


def extract_shard(
    cfg: Stage0AConfig,
    *,
    index_dir: str | Path,
    split: str,
    dataset_fingerprint: str,
    rank: int,
    n_shards: int,
    physical_gpu: int,
) -> Dict[str, Any]:
    """Extract one contiguous shard. Intended to run in its own subprocess."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)  # one physical GPU per process via CUDA_VISIBLE_DEVICES
        torch.backends.cudnn.benchmark = True
    index = load_index(index_dir)
    directory = cache_dir(cfg, split, dataset_fingerprint)
    features_path = directory / "features.npy"
    labels_path = directory / "labels.npy"
    meta = json_load(directory / "meta.json")
    start, end = shard_bounds(len(index), rank, n_shards)
    count = end - start
    if count <= 0:
        return {"rank": rank, "count": 0, "skipped": True}

    ds = IndexedImageDataset(index, make_transform(cfg.img_size), np.arange(start, end))
    loader = DataLoader(
        ds,
        batch_size=cfg.feature_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model, weights_sha = load_backbone(cfg, device)
    out = np.load(features_path, mmap_mode="r+")
    labels_out = np.load(labels_path, mmap_mode="r+")
    monitor = GPUMonitor(physical_gpu).start()
    t0 = time.time()
    n_done = 0
    try:
        with torch.inference_mode():
            for x, y, gi in loader:
                x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    z = model(x)
                z = z.float().cpu().numpy().astype(np.float16)
                rows = gi.numpy()
                out[rows] = z
                labels_out[rows] = y.numpy()
                n_done += len(rows)
        out.flush()
        labels_out.flush()
        del out, labels_out
    finally:
        monitor_stats = monitor.stop()

    elapsed = time.time() - t0
    result = {
        "rank": int(rank),
        "physical_gpu": int(physical_gpu),
        "start": int(start),
        "end": int(end),
        "count": int(n_done),
        "elapsed_s": round(elapsed, 3),
        "images_per_s": round(n_done / max(elapsed, 1e-9), 3),
        "feature_batch_size_per_gpu": int(cfg.feature_batch_size),
        "feature_global_batch_size": int(cfg.feature_batch_size * n_shards),
        "gpu": monitor_stats,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else 0.0,
        "backbone_weights_sha256": weights_sha,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(directory / f"shard_{rank:02d}.json", result)
    return result


def finalize_cache(cfg: Stage0AConfig, index: ImageNetIndex, split: str, n_shards: int) -> Dict[str, Any]:
    directory = cache_dir(cfg, split, index.fingerprint)
    shards = []
    for rank in range(n_shards):
        p = directory / f"shard_{rank:02d}.json"
        if not p.exists():
            raise RuntimeError(f"feature extraction shard {rank} did not produce a result at {p}")
        shards.append(json_load(p))
    total = sum(int(s.get("count", 0)) for s in shards)
    if total != len(index):
        raise RuntimeError(f"feature cache incomplete: {total} images extracted, expected {len(index)}")
    features = directory / "features.npy"
    labels = directory / "labels.npy"
    meta = json_load(directory / "meta.json")
    meta.update(
        {
            "complete": True,
            "completed_at_utc": utc_now(),
            "n_shards": n_shards,
            "backbone_weights_sha256": shards[0].get("backbone_weights_sha256") if shards else None,
            "features_sha256": sha256_file(features),
            "labels_sha256": sha256_file(labels),
            "features_bytes": features.stat().st_size,
            "labels_bytes": labels.stat().st_size,
            "shards": shards,
        }
    )
    atomic_write_json(directory / "complete.json", meta)
    # Keep meta.json aligned for readers that only inspect it.
    atomic_write_json(directory / "meta.json", meta)
    return meta


def load_complete_cache(cfg: Stage0AConfig, std: str, dataset_fingerprint: str) -> FeatureCache:
    directory = cache_dir(cfg, std, dataset_fingerprint)
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing feature cache metadata at {meta_path}")
    meta = json_load(meta_path)
    if not meta.get("complete"):
        raise RuntimeError(f"feature cache at {directory} is incomplete")
    features = directory / "features.npy"
    labels = directory / "labels.npy"
    if not features.exists() or not labels.exists():
        raise FileNotFoundError(f"feature cache files missing under {directory}")
    for key, path in (("features_sha256", features), ("labels_sha256", labels)):
        if sha256_file(path) != meta[key]:
            raise RuntimeError(f"feature cache {path.name} failed SHA-256 verification")
    return FeatureCache(directory=directory, features=features, labels=labels, meta=meta)


def cache_summary(cache: FeatureCache) -> Dict[str, Any]:
    return {
        "directory": str(cache.directory),
        "count": cache.n,
        "dim": cache.dim,
        "features_bytes": cache.features.stat().st_size,
        "features_human": human_bytes(cache.features.stat().st_size),
        "labels_bytes": cache.labels.stat().st_size,
        "features_sha256": cache.meta.get("features_sha256"),
        "labels_sha256": cache.meta.get("labels_sha256"),
        "dtype": cache.meta.get("dtype"),
        "representation": cache.meta.get("representation"),
    }


def upload_cache(cache: FeatureCache, hf_store: Any, run_id: str = "stage0a/cache") -> None:
    """Push the pooled-feature cache so a new session can skip extraction."""
    for filename in ("meta.json", "complete.json", "labels.npy", "features.npy"):
        local = cache.directory / filename
        remote = f"{run_id}/{cache.directory.name}/{filename}"
        hf_store.upload_verified(local, hf_store.best_repo, remote)


def restore_cache(cfg: Stage0AConfig, split: str, dataset_fingerprint: str, hf_store: Any) -> Optional[FeatureCache]:
    """Download a complete cache if present; return None only for verified absence."""
    name = cache_name(cfg, split, dataset_fingerprint)
    remote_prefix = f"stage0a/cache/{name}"
    complete_remote = f"{remote_prefix}/complete.json"
    status = hf_store.status(hf_store.best_repo, complete_remote)
    if status == "absent":
        return None
    directory = cache_dir(cfg, split, dataset_fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("meta.json", "complete.json", "labels.npy", "features.npy"):
        local = directory / filename
        hf_store.download_verified(hf_store.best_repo, f"{remote_prefix}/{filename}", local)
    meta = json_load(directory / "complete.json")
    if meta.get("dataset_fingerprint") != dataset_fingerprint:
        raise RuntimeError("downloaded feature cache has a different dataset fingerprint")
    return load_complete_cache(cfg, split, dataset_fingerprint)

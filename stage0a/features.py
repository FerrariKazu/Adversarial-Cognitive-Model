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
    atomic_write_text,
    human_bytes,
    json_load,
    resolve_hf_token,
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


def restore_shard(
    cfg: Stage0AConfig,
    *,
    split: str,
    dataset_fingerprint: str,
    rank: int,
    n_shards: int,
    total_count: int,
    hf_store: Any = None,
) -> Optional[Dict[str, Any]]:
    """Check if a shard is available locally or on Hugging Face and verified.

    If found and verified:
      - Ensures shard files (features, labels, json, and .sha256 sidecars) are valid.
      - Populates the pre-allocated memmaps in cache_dir at [start:end].
      - Returns the shard metadata dict.
    If absent or unverified, returns None.
    """
    directory = cache_dir(cfg, split, dataset_fingerprint)
    start, end = shard_bounds(total_count, rank, n_shards)
    count = end - start
    if count <= 0:
        return {"rank": rank, "count": 0, "skipped": True}

    shard_json_path = directory / f"shard_{rank:02d}.json"
    shard_feats_path = directory / f"shard_{rank:02d}_features.npy"
    shard_labels_path = directory / f"shard_{rank:02d}_labels.npy"
    features_path = directory / "features.npy"
    labels_path = directory / "labels.npy"

    def _verify_local() -> Optional[Dict[str, Any]]:
        if not (shard_json_path.exists() and shard_feats_path.exists() and shard_labels_path.exists()):
            return None
        info = json_load(shard_json_path)
        if int(info.get("start", -1)) != start or int(info.get("end", -1)) != end or int(info.get("count", -1)) != count:
            return None
        # Check SHA-256 sidecars
        feats_sidecar = shard_feats_path.with_suffix(shard_feats_path.suffix + ".sha256")
        labels_sidecar = shard_labels_path.with_suffix(shard_labels_path.suffix + ".sha256")
        if not feats_sidecar.exists() or not labels_sidecar.exists():
            return None
        feats_sha = feats_sidecar.read_text(encoding="utf-8").strip().split()[0]
        labels_sha = labels_sidecar.read_text(encoding="utf-8").strip().split()[0]
        if sha256_file(shard_feats_path) != feats_sha or sha256_file(shard_labels_path) != labels_sha:
            return None
        if info.get("features_sha256") and info["features_sha256"] != feats_sha:
            return None
        if info.get("labels_sha256") and info["labels_sha256"] != labels_sha:
            return None
        return info

    info = _verify_local()
    if info is None and getattr(hf_store, "enabled", False):
        repo = hf_store.feature_repo(cfg.backbone) if hasattr(hf_store, "feature_repo") else getattr(hf_store, "best_repo", "")
        prefix = f"stage0a/cache/{cache_name(cfg, split, dataset_fingerprint)}"
        remote_json = f"{prefix}/shard_{rank:02d}.json"
        remote_feats = f"{prefix}/shard_{rank:02d}_features.npy"
        remote_labels = f"{prefix}/shard_{rank:02d}_labels.npy"
        try:
            if (
                hf_store.status(repo, remote_json) == "present"
                and hf_store.status(repo, remote_feats) == "present"
                and hf_store.status(repo, remote_labels) == "present"
            ):
                directory.mkdir(parents=True, exist_ok=True)
                hf_store.download_verified(repo, remote_json, shard_json_path)
                hf_store.download_verified(repo, remote_feats, shard_feats_path)
                hf_store.download_verified(repo, remote_labels, shard_labels_path)
                info = _verify_local()
        except Exception:
            info = None

    if info is not None:
        if features_path.exists() and labels_path.exists():
            feats_data = np.load(shard_feats_path)
            labels_data = np.load(shard_labels_path)
            out = np.load(features_path, mmap_mode="r+")
            labels_out = np.load(labels_path, mmap_mode="r+")
            out[start:end] = feats_data
            labels_out[start:end] = labels_data
            out.flush()
            labels_out.flush()
            del out, labels_out
        res = dict(info)
        res["restored"] = True
        return res

    return None


def extract_shard(
    cfg: Stage0AConfig,
    *,
    index_dir: str | Path,
    split: str,
    dataset_fingerprint: str,
    rank: int,
    n_shards: int,
    physical_gpu: int,
    hf_store: Any = None,
) -> Dict[str, Any]:
    """Extract one contiguous shard. Intended to run in its own subprocess."""
    index = load_index(index_dir)
    directory = cache_dir(cfg, split, dataset_fingerprint)
    start, end = shard_bounds(len(index), rank, n_shards)
    count = end - start
    if count <= 0:
        return {"rank": rank, "count": 0, "skipped": True}

    if hf_store is None:
        tok = resolve_hf_token()
        if tok:
            try:
                from stage0a.hf_sync import HFStore

                hf_store = HFStore(token=tok)
            except Exception:
                hf_store = None

    restored = restore_shard(
        cfg,
        split=split,
        dataset_fingerprint=dataset_fingerprint,
        rank=rank,
        n_shards=n_shards,
        total_count=len(index),
        hf_store=hf_store,
    )
    if restored is not None:
        return restored

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)  # one physical GPU per process via CUDA_VISIBLE_DEVICES
        torch.backends.cudnn.benchmark = True

    features_path = directory / "features.npy"
    labels_path = directory / "labels.npy"

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
    finally:
        monitor_stats = monitor.stop()

    # Save standalone shard arrays and SHA-256 sidecars
    shard_feats = np.array(out[start:end])
    shard_labels = np.array(labels_out[start:end])
    del out, labels_out

    shard_feats_path = directory / f"shard_{rank:02d}_features.npy"
    shard_labels_path = directory / f"shard_{rank:02d}_labels.npy"
    np.save(shard_feats_path, shard_feats)
    np.save(shard_labels_path, shard_labels)

    feats_sha = sha256_file(shard_feats_path)
    labels_sha = sha256_file(shard_labels_path)
    atomic_write_text(str(shard_feats_path) + ".sha256", feats_sha + "\n")
    atomic_write_text(str(shard_labels_path) + ".sha256", labels_sha + "\n")

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
        "features_sha256": feats_sha,
        "labels_sha256": labels_sha,
        "features_bytes": shard_feats_path.stat().st_size,
        "labels_bytes": shard_labels_path.stat().st_size,
        "completed_at_utc": utc_now(),
    }
    shard_json_path = directory / f"shard_{rank:02d}.json"
    atomic_write_json(shard_json_path, result)
    atomic_write_text(str(shard_json_path) + ".sha256", sha256_file(shard_json_path) + "\n")

    if getattr(hf_store, "enabled", False):
        repo = hf_store.feature_repo(cfg.backbone) if hasattr(hf_store, "feature_repo") else getattr(hf_store, "best_repo", "")
        prefix = f"stage0a/cache/{cache_name(cfg, split, dataset_fingerprint)}"
        hf_store.upload_verified(shard_feats_path, repo, f"{prefix}/shard_{rank:02d}_features.npy")
        hf_store.upload_verified(shard_labels_path, repo, f"{prefix}/shard_{rank:02d}_labels.npy")
        hf_store.upload_verified(shard_json_path, repo, f"{prefix}/shard_{rank:02d}.json")

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
    meta_json_path = directory / "meta.json"
    complete_json_path = directory / "complete.json"
    atomic_write_json(complete_json_path, meta)
    # Keep meta.json aligned for readers that only inspect it.
    atomic_write_json(meta_json_path, meta)
    atomic_write_text(str(features) + ".sha256", meta["features_sha256"] + "\n")
    atomic_write_text(str(labels) + ".sha256", meta["labels_sha256"] + "\n")
    atomic_write_text(str(complete_json_path) + ".sha256", sha256_file(complete_json_path) + "\n")
    atomic_write_text(str(meta_json_path) + ".sha256", sha256_file(meta_json_path) + "\n")
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
    backbone = cache.meta.get("backbone", "")
    repo = hf_store.feature_repo(backbone) if hasattr(hf_store, "feature_repo") else getattr(hf_store, "best_repo", "")
    for filename in ("meta.json", "complete.json", "labels.npy", "features.npy"):
        local = cache.directory / filename
        if local.exists():
            remote = f"{run_id}/{cache.directory.name}/{filename}"
            hf_store.upload_verified(local, repo, remote)
    for shard_file in sorted(cache.directory.glob("shard_*.json")):
        rank_str = shard_file.stem
        for sf in (shard_file, cache.directory / f"{rank_str}_features.npy", cache.directory / f"{rank_str}_labels.npy"):
            if sf.exists():
                remote = f"{run_id}/{cache.directory.name}/{sf.name}"
                hf_store.upload_verified(sf, repo, remote)


def restore_cache(cfg: Stage0AConfig, split: str, dataset_fingerprint: str, hf_store: Any) -> Optional[FeatureCache]:
    """Download a complete cache if present; return None only for verified absence."""
    name = cache_name(cfg, split, dataset_fingerprint)
    remote_prefix = f"stage0a/cache/{name}"
    complete_remote = f"{remote_prefix}/complete.json"
    repo = hf_store.feature_repo(cfg.backbone) if hasattr(hf_store, "feature_repo") else getattr(hf_store, "best_repo", "")
    status = hf_store.status(repo, complete_remote)
    if status == "absent":
        return None
    directory = cache_dir(cfg, split, dataset_fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("meta.json", "complete.json", "labels.npy", "features.npy"):
        local = directory / filename
        hf_store.download_verified(repo, f"{remote_prefix}/{filename}", local)
    meta = json_load(directory / "complete.json")
    if meta.get("dataset_fingerprint") != dataset_fingerprint:
        raise RuntimeError("downloaded feature cache has a different dataset fingerprint")
    return load_complete_cache(cfg, split, dataset_fingerprint)

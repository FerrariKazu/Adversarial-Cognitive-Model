#!/usr/bin/env python3
"""Measure Stage 0B backbone inference cost on calibration images only.

No logits, temperatures, labels, or evaluation images are used.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "audit" / "stage0b_final_freeze_v1"
DATA_GATE = ROOT / "audit" / "stage0b_data_gate_v1"
DATASET = DATA_GATE / "work" / "extracted" / "imagenetv2-matched-frequency-format-val"
BACKBONE = FREEZE / "work" / "backbone_cache" / "vit_base_patch14_dinov2.lvd142m" / "model.safetensors"
OUT = FREEZE / "work" / "compute_probe.json"

sys.path.insert(0, str(ROOT))
from stage0a.data import make_transform  # noqa: E402


def main() -> int:
    manifest = json.loads((DATA_GATE / "PARTITION_MANIFEST.json").read_text(encoding="utf-8"))
    rows = sorted(
        (row for row in manifest["assignments"] if row["split"] == "calibration"),
        key=lambda row: (row["class_index"], row["within_class_rank"]),
    )[:64]
    transform = make_transform(224)
    tensors = []
    for row in rows:
        path = DATASET / row["relative_path"]
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    del rows

    import timm

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda:0")
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        pretrained_cfg_overlay={"file": str(BACKBONE)},
        num_classes=0,
        img_size=224,
        dynamic_img_size=True,
    )
    model.eval().to(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    batch_size = 8
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    elapsed = 0.0
    images_done = 0
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            batch = batch.contiguous(memory_format=torch.channels_last)
            if start == 0:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(batch)
                torch.cuda.synchronize()
                continue
            begin = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = model(batch)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - begin
            images_done += features.shape[0]
            if not torch.isfinite(features).all():
                raise SystemExit("non-finite image features")
            del features
    peak = int(torch.cuda.max_memory_allocated())
    result = {
        "schema": "rhan-stage0b-compute-probe/v1",
        "status": "PASS",
        "scope": "calibration images only; backbone inference only; zero logits, temperatures, labels, or evaluation images",
        "device": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "images_timed": images_done,
        "timed_seconds": round(elapsed, 6),
        "images_per_second": round(images_done / max(elapsed, 1e-9), 6),
        "seconds_per_batch": round(elapsed / max(images_done / batch_size, 1e-9), 6),
        "peak_cuda_allocated_bytes": peak,
        "peak_cuda_allocated_gib": round(peak / 1024**3, 6),
        "extrapolated_10000_images_seconds": round(10000 / max(images_done / max(elapsed, 1e-9), 1e-9), 3),
        "feature_dtype": "float16",
        "feature_dim": 768,
        "shared_backbone": True,
        "three_heads_reuse_same_features": True,
        "feature_cache_bytes_for_10000x768_float16": 10_000 * 768 * 2,
        "logit_cache_bytes_for_10000x1000_float32_per_seed": 10_000 * 1000 * 4,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "images_per_second", "peak_cuda_allocated_gib", "extrapolated_10000_images_seconds")}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Hardware/storage audit and the mandatory pre-training smoke benchmark."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from stage0a.config import BACKBONES, Stage0AConfig
from stage0a.data import load_index, make_transform, IndexedImageDataset
from stage0a.features import GPUMonitor, load_backbone
from stage0a.utils import atomic_write_json, human_bytes, read_env_int


def _nvidia_smi() -> List[Dict[str, Any]]:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = []
        for line in p.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 6:
                out.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": float(parts[2]),
                        "memory_used_mib": float(parts[3]),
                        "utilization_pct": float(parts[4]),
                        "driver": parts[5],
                    }
                )
        return out
    except Exception:
        return []


def hardware_inventory() -> Dict[str, Any]:
    import psutil

    cuda = torch.cuda.is_available()
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "index": i,
                "name": p.name,
                "total_memory_gb": round(p.total_memory / 1e9, 3),
                "major": p.major,
                "minor": p.minor,
                "multi_processor_count": p.multi_processor_count,
            }
        )
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "nvidia_smi": _nvidia_smi(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 3),
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 3),
        "is_kaggle": Path("/kaggle/working").exists(),
    }


def benchmark_worker(args: Any) -> Dict[str, Any]:
    """One GPU, both backbones. Runs inside a CUDA_VISIBLE_DEVICES shard."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physical = int(os.environ.get("STAGE0A_PHYSICAL_GPU", args.gpu_index))
    index = load_index(args.index_dir)
    n = min(args.n_images, len(index))
    ds = IndexedImageDataset(index, make_transform(args.img_size), np.arange(n))
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    results: Dict[str, Any] = {
        "gpu_index": args.gpu_index,
        "physical_gpu": physical,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "batch_size_per_gpu": args.batch_size,
        "num_workers": args.num_workers,
        "img_size": args.img_size,
        "requested_images": n,
        "backbones": {},
    }
    for backbone in args.backbones:
        model, weights_sha = load_backbone(
            Stage0AConfig(
                backbone=backbone,
                seed=17,
                data_root=args.data_root,
                cache_root=args.cache_root,
                output_root=args.output_root,
                img_size=args.img_size,
                num_classes=1000,
                feature_batch_size=args.batch_size,
                num_workers=args.num_workers,
                require_dual_gpu=False,
                allow_single_gpu=True,
            ),
            device,
        )
        # Warm the kernels before timing; otherwise the first model measured
        # absorbs cudnn/timm load latency and the comparison is meaningless.
        with torch.inference_mode():
            warm = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
            if device.type == "cuda":
                warm = warm.contiguous(memory_format=torch.channels_last)
            for _ in range(2):
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    model(warm)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        monitor = GPUMonitor(physical).start()
        data_wait = 0.0
        forward_time = 0.0
        seen = 0
        batches = 0
        t0 = time.time()
        try:
            with torch.inference_mode():
                for x, y, gi in loader:
                    t_data = time.time()
                    x = x.to(device, non_blocking=True)
                    if device.type == "cuda":
                        x = x.contiguous(memory_format=torch.channels_last)
                    data_wait += time.time() - t_data
                    t_fwd = time.time()
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                    ):
                        z = model(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    forward_time += time.time() - t_fwd
                    seen += int(x.shape[0])
                    batches += 1
                    if batches >= args.max_batches:
                        break
        finally:
            gpu_stats = monitor.stop()
        elapsed = time.time() - t0
        results["backbones"][backbone] = {
            "timm_name": BACKBONES[backbone]["timm_name"],
            "weights_sha256": weights_sha,
            "images": seen,
            "batches": batches,
            "elapsed_s": round(elapsed, 3),
            "images_per_s": round(seen / max(elapsed, 1e-9), 3),
            "forward_images_per_s": round(seen / max(forward_time, 1e-9), 3),
            "dataloader_wait_s": round(data_wait, 3),
            "gpu": gpu_stats,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else 0.0,
            "feature_bytes_per_image_fp16": int(BACKBONES[backbone]["embed_dim"]) * 2,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Feature-cache write throughput (the durable artifact path).
    dim = 768
    arr = np.zeros((max(n, 1), dim), dtype=np.float16)
    tmp = Path(args.output_root) / "smoke_feature_write.npy"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    np.save(tmp, arr)
    os.sync()
    write_s = time.time() - t0
    results["feature_cache_write"] = {
        "bytes": tmp.stat().st_size,
        "seconds": round(write_s, 4),
        "mib_per_s": round((tmp.stat().st_size / (1024 * 1024)) / max(write_s, 1e-9), 3),
    }
    tmp.unlink(missing_ok=True)

    # Small head-training step throughput on synthetic pooled features.
    from stage0a.training import LinearHead

    head = LinearHead(dim, 1000).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=0.001, momentum=0.9)
    xb = torch.randn(1024, dim, device=device)
    yb = torch.randint(0, 1000, (1024,), device=device)
    steps = 20
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = torch.nn.functional.cross_entropy(head(xb), yb)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    head_s = time.time() - t0
    results["head_training"] = {
        "steps": steps,
        "batch_size": 1024,
        "steps_per_s": round(steps / max(head_s, 1e-9), 3),
        "samples_per_s": round(steps * 1024 / max(head_s, 1e-9), 3),
    }
    return results


def run_smoke(
    *,
    index_dir: str,
    data_root: str,
    cache_root: str,
    output_root: str,
    img_size: int = 224,
    batch_size: int = 64,
    n_images: int = 512,
    max_batches: int = 4,
    num_workers: int = 4,
    require_dual_gpu: bool = True,
) -> Dict[str, Any]:
    n_gpu = torch.cuda.device_count()
    if require_dual_gpu and n_gpu < 2:
        raise RuntimeError(
            f"dual-T4 smoke test requires 2 visible GPUs; this process sees {n_gpu}. "
            "Run this cell on Kaggle T4 x2. Local single-GPU results are not a substitute."
        )
    out_dir = Path(output_root) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu in range(min(n_gpu, 2)):
        out = out_dir / f"gpu{gpu}.json"
        cmd = [
            sys.executable,
            "-m",
            "stage0a.benchmark",
            "--worker",
            "--gpu-index",
            str(gpu),
            "--index-dir",
            str(index_dir),
            "--data-root",
            data_root,
            "--cache-root",
            cache_root,
            "--output-root",
            output_root,
            "--img-size",
            str(img_size),
            "--batch-size",
            str(batch_size),
            "--n-images",
            str(n_images),
            "--max-batches",
            str(max_batches),
            "--num-workers",
            str(num_workers),
            "--out",
            str(out),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["STAGE0A_PHYSICAL_GPU"] = str(gpu)
        procs.append((gpu, subprocess.Popen(cmd, env=env), out))
    for gpu, p, out in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"smoke worker on GPU {gpu} failed with rc={rc}")
    per_gpu = [json.loads(out.read_text()) for _, _, out in procs]
    participated = (
        len(per_gpu) == 2
        and all(g.get("backbones") for g in per_gpu)
        and all(
            all(int(b.get("images", 0)) > 0 for b in g["backbones"].values())
            for g in per_gpu
        )
    )
    report = {
        "requested_dual_gpu": require_dual_gpu,
        "visible_gpu_count": n_gpu,
        "workers": per_gpu,
        "both_gpus_participated": bool(participated),
        "per_gpu_batch_size": batch_size,
        "global_batch_size": batch_size * len(per_gpu),
        "note": "Each worker is an independent single-GPU process; DataParallel/DDP are intentionally not used on T4.",
    }
    atomic_write_json(out_dir / "smoke_report.json", report)
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-images", type=int, default=512)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--backbones", nargs="+", default=["vit-s14", "vit-b14"])
    args = p.parse_args()
    if not args.worker:
        raise SystemExit("invoke with --worker; use stage0a.cli smoke for orchestration")
    result = benchmark_worker(args)
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

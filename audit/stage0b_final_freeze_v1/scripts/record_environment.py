#!/usr/bin/env python3
"""Record the Stage 0B runtime fingerprint without credentials or secrets."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import scipy
import torch
import torchvision
import timm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit" / "stage0b_final_freeze_v1" / "work" / "environment.json"


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> int:
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            memory[key] = int(value.strip().split()[0]) * 1024
    usage = shutil.disk_usage(ROOT)
    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    try:
        driver = command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[0]
    except Exception:
        driver = None
    git_commit = command("git", "rev-parse", "HEAD")
    payload = {
        "schema": "rhan-stage0b-runtime-environment/v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "is_wsl": "microsoft" in platform.release().lower(),
        },
        "cpu": {
            "model": platform.processor() or "x86_64",
            "logical_cores": os.cpu_count(),
        },
        "memory": {
            "total_bytes": memory.get("MemTotal"),
            "available_bytes_at_capture": memory.get("MemAvailable"),
        },
        "gpu": {**gpu, "nvidia_driver": driver},
        "filesystem": {
            "project_root": str(ROOT),
            "total_bytes": usage.total,
            "free_bytes_at_capture": usage.free,
        },
        "software": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            "huggingface_hub": __import__("huggingface_hub").__version__,
        },
        "git_commit": git_commit,
        "credentials": {
            "hf_token_present": bool(os.environ.get("HF_TOKEN")),
            "hugging_face_login": False,
            "kaggle_mount_present": Path("/kaggle").exists(),
            "note": "Presence flags only; no credential value is recorded.",
        },
        "scope_note": "Stage 0B freeze-preparation runtime. This record does not authorize execution.",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"fingerprint_sha256={payload['fingerprint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

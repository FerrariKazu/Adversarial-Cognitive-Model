"""Hardware introspection: device selection, GPU stats, memory."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Optional

from cognitive_vision_lab.config import CUDA_AVAILABLE, DEFAULT_DEVICE


@dataclass(frozen=True)
class HardwareInfo:
    device: str
    cuda_available: bool
    gpu_name: str = "n/a"
    gpu_vram_gb: float = 0.0
    gpu_util_pct: float = 0.0
    gpu_mem_used_gb: float = 0.0
    cpu_count: int = 0
    python_version: str = ""
    torch_version: str = "n/a"

    @property
    def summary(self) -> str:
        if self.cuda_available:
            return (
                f"{self.gpu_name} · {self.gpu_vram_gb:.1f} GB VRAM · "
                f"{self.gpu_mem_used_gb:.1f} GB in use ({self.gpu_util_pct:.0f}% util)"
            )
        return "CPU only"


def get_hardware() -> HardwareInfo:
    """Gather hardware info. Streamlit-safe (callable from any thread)."""
    gpu_name, vram, util, mem_used = "n/a", 0.0, 0.0, 0.0
    torch_version = "n/a"
    if CUDA_AVAILABLE:
        try:
            import torch

            torch_version = torch.__version__
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1e9
            mem_used = torch.cuda.memory_allocated(0) / 1e9
            util = _nvidia_util()
        except Exception:
            pass
    else:
        try:
            import torch

            torch_version = torch.__version__
        except Exception:
            pass
    return HardwareInfo(
        device=DEFAULT_DEVICE,
        cuda_available=CUDA_AVAILABLE,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        gpu_util_pct=util,
        gpu_mem_used_gb=mem_used,
        cpu_count=os_cpu_count(),
        python_version=platform.python_version(),
        torch_version=torch_version,
    )


def os_cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 0
    except Exception:
        return 0


def _nvidia_util() -> float:
    """Query nvidia-smi util% once (cheap). Returns 0.0 on failure."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def recommend_batch_size(vram_gb: float, model_params_m: float) -> int:
    """Conservative batch-size heuristic given VRAM and model size."""
    if vram_gb <= 0:
        return 8
    per_sample_gb = max(model_params_m / 1000.0 * 2.0, 0.05)
    return max(1, min(64, int(vram_gb * 0.6 / max(per_sample_gb, 1e-3))))

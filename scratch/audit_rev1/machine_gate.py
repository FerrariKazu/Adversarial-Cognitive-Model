"""
Stage-0 benchmark prep (separate runtime, separate worktree).

PURPOSE
-------
Confirm what this machine actually is (T4 vs RTX 4060) before any profiling
numbers are trusted. Report the backend reality: torch version, CUDA, SDPA
mode, cudnn benchmark state, FP32/FP16/BF16 max/min, and whether a
Dollar-T4 fleet is reachable. This file is the gate before running the
per-op CUDA-time audit.

CONSTRAINTS (inherited from the user request)
-------
Do NOT change the recipe. Do NOT prune PGD steps. Do NOT touch the in-flight
Stage-0 run. This file is separate, on a separate branch/notebook, and only
used for measurement.

OUTPUT
-------
JSON-like text written to stdout, plus the SDPA backend and a machine
identity line. The calling harness can redirect this to a report file.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT = []

def line(s: str = "") -> None:
    OUT.append(s)
    print(s)

def machine_id() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
    }

def cudnn_state() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "backend": torch.backends.cudnn.version(),
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "enabled": torch.backends.cudnn.enabled,
    }

def sdpa_modes() -> dict:
    if not torch.cuda.is_available():
        return {"cpu_only_ranked": "unknown"}
    xs = [torch.randn(2, 6, 16, 64, device="cuda") for _ in range(3)]
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        y = F.scaled_dot_product_attention(*xs)
    # torch 2.1+ exposes the sdpa backend choice via torch.backends.cuda.sdp_kernel
    mode_now = torch.backends.cuda.sdp_kernel
    return {
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "warm_shape": [2, 6, 16, 64],
    }

def fp_range(dtype: torch.dtype) -> dict:
    info = torch.finfo(dtype)
    return {"dtype": str(dtype), "min": float(info.min), "max": float(info.max), "eps": float(info.eps)}

def torch_compile_dispatch() -> dict:
    try:
        import torch._dynamo
        return {"dynamo_version": torch._dynamo.__version__ if hasattr(torch._dynamo, "__version__") else "present"}
    except Exception as e:
        return {"dynamo": "unavailable", "note": str(e)[:200]}

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    line("=== Machine identity (separate runtime) ===")
    line(json.dumps(machine_id(), indent=2))
    line("")
    line("=== cuDNN state (after benchmark=True) ===")
    line(json.dumps(cudnn_state(), indent=2))
    line("")
    line("=== SDPA mode sanity (T=4 glimpses, B=2, heads=6, tokens=16, hd=64) ===")
    line(json.dumps(sdpa_modes(), indent=2))
    line("")
    line("=== FP range sanity ===")
    for d in [torch.float32, torch.float16, torch.bfloat16]:
        line(json.dumps(fp_range(d), indent=2))
    line("")
    line("=== Torch compile dispatch ===")
    line(json.dumps(torch_compile_dispatch(), indent=2))
    line("")
    line("=== Data path sanity ===")
    line("REPO_ROOT =", REPO_ROOT)
    line("exists checkpoints/dinov2_vits14_pretrain.pth ->", os.path.exists(os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth")))

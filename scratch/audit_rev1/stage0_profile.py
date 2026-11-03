"""
Stage-0 profiling probe (audit rev-1).

SETUP
------
- Separate runtime from the in-flight trainer.
- Warmed up before timing.
- Fixed seed.
- Batch 48, 112 px, recurrence_only architecture, top-rung TRADES-PGD4.
- torch.profiler with CUDA activity + kernel traces.
- SDPA backend recorded.
- Profile saved to report/audit_rev1/ so the numbers can be inspected later.

NOTE
----
This machine is an RTX 4060, not the T4 the recipe is written for. The probe
is honest about that; the per-op CUDA-time structure is still useful, but raw
s/step numbers here are NOT the recipe's T4 numbers.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel  # noqa: E402
from training.train_gen2_foundation import (  # noqa: E402
    compute_trades_loss,
    run_pgd_kl_attack,
)

OUT: List[str] = []


def line(s: str = "") -> None:
    OUT.append(s)
    print(s)


REPORT_DIR = os.path.join(REPO_ROOT, "report", "audit_rev1")
os.makedirs(REPORT_DIR, exist_ok=True)


def build_model_and_batch(batch_size: int = 48, seed: int = 7) -> Dict[str, Any]:
    torch.manual_seed(seed)
    m = (
        Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112)
        .eval()
        .cuda()
    )
    m.backbone.load_dino_warm_start(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"),
        strict_band=0.90,
    )
    x = torch.randn(batch_size, 3, 112, 112, device="cuda")
    y = torch.randint(0, 100, (batch_size,), device="cuda")
    return {
        "model": m,
        "x": x,
        "y": y,
        "batch_size": batch_size,
        "eps": 0.094,
        "steps": 4,
    }


def sdpa_backend_record() -> Dict[str, Any]:
    return {
        "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_version": getattr(torch.version, "cuda", None),
        "torch_version": str(torch.__version__),
    }


def _device_time_total(evt: Any) -> Optional[float]:
    """CUDA/device total time in us, across torch versions."""
    for attr in ("device_time_total", "cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None:
            return float(val)
    return None


def _self_device_time_total(evt: Any) -> Optional[float]:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None:
            return float(val)
    return None


def collect_key_averages(prof) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for evt in prof.key_averages():
        name = getattr(evt, "key", None)
        if not name:
            continue
        cuda_us = _device_time_total(evt)
        self_us = _self_device_time_total(evt)
        rows.append(
            {
                "event": str(name),
                "device_time_ms": round((cuda_us or 0.0) / 1000.0, 3),
                "self_device_time_ms": round((self_us or 0.0) / 1000.0, 3),
                "cpu_time_ms": round(float(getattr(evt, "cpu_time_total", 0.0)) / 1000.0, 3),
                "count": int(getattr(evt, "count", 0)),
            }
        )
    rows.sort(key=lambda r: -r["device_time_ms"])
    return rows


def is_cuda_event(evt: Any) -> bool:
    dev = getattr(evt, "device_type", None)
    if dev is None:
        return False
    return "CUDA" in str(dev).upper()


def cuda_kernel_intervals(prof) -> List[tuple]:
    """Raw CUDA kernel intervals (start_us, end_us, name) for gap analysis."""
    intervals: List[tuple] = []
    for evt in prof.events():
        if not is_cuda_event(evt):
            continue
        tr = getattr(evt, "time_range", None)
        if tr is None:
            continue
        intervals.append((float(tr.start), float(tr.end), str(getattr(evt, "name", ""))))
    intervals.sort(key=lambda t: t[0])
    return intervals


def launch_gap_summary(intervals: List[tuple]) -> Dict[str, Any]:
    """Non-overlap gaps between consecutive kernels: the launch-stall budget."""
    if len(intervals) < 2:
        return {"n_kernels": len(intervals), "n_gaps": 0}
    gaps: List[float] = []
    spans: List[float] = []
    prev_end = intervals[0][1]
    for start, end, _name in intervals[1:]:
        gap = start - prev_end
        if gap > 0:
            gaps.append(gap)
            prev_end = end
        else:
            prev_end = max(prev_end, end)
        spans.append(end - start)
    gaps.sort()
    total_span = intervals[-1][1] - intervals[0][0]
    covered = sum(spans)
    return {
        "n_kernels": len(intervals),
        "n_gaps": len(gaps),
        "gap_total_ms": round(sum(gaps) / 1000.0, 3),
        "gap_median_us": round(gaps[len(gaps) // 2], 2) if gaps else 0.0,
        "gap_p90_us": round(gaps[int(0.9 * (len(gaps) - 1))], 2) if gaps else 0.0,
        "gap_max_us": round(gaps[-1], 2) if gaps else 0.0,
        "cuda_busy_ms": round(covered / 1000.0, 3),
        "wall_span_ms": round(total_span / 1000.0, 3),
        "cuda_busy_fraction": round(covered / total_span, 4) if total_span > 0 else None,
    }


def run_trained_step_under_profiler(
    cfg: Dict[str, Any], n_warmup: int = 3, n_profile: int = 1
) -> Dict[str, Any]:
    m = cfg["model"]
    x = cfg["x"]
    y = cfg["y"]

    # Warmup outside the profiler: cuDNN/SDPA autotune and allocator growth
    # must already be paid for.
    for _ in range(n_warmup):
        x_adv = run_pgd_kl_attack(m, x.clone().detach(), eps=cfg["eps"], steps=cfg["steps"])
        with torch.amp.autocast("cuda", enabled=True):
            loss, _, _ = compute_trades_loss(
                m, x.clone().detach(), y.clone().detach(), x_adv, beta=2.5, w_trades=0.55
            )
        loss.backward()
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    prof = profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
        with_flops=False,
    )
    m.zero_grad(set_to_none=True)
    with prof:
        for _ in range(n_profile):
            with record_function("stage0_training_step"):
                x_adv = run_pgd_kl_attack(
                    m, x.clone().detach(), eps=cfg["eps"], steps=cfg["steps"]
                )
                with torch.amp.autocast("cuda", enabled=True):
                    loss, ce_val, kl_val = compute_trades_loss(
                        m,
                        x.clone().detach(),
                        y.clone().detach(),
                        x_adv,
                        beta=2.5,
                        w_trades=0.55,
                    )
                loss.backward()
            torch.cuda.synchronize()

    rows = collect_key_averages(prof)
    intervals = cuda_kernel_intervals(prof)
    gaps = launch_gap_summary(intervals)

    # Device-busy total is the sum of the top-level kernel spans, which is the
    # honest denominator for "how much of the step is GPU work".
    total_device_ms = round(sum(r["device_time_ms"] for r in rows if r["count"]), 3)
    step_wall_ms = None
    # Prefer measured step wall time over profiler-derived spans.
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    x_adv = run_pgd_kl_attack(m, x.clone().detach(), eps=cfg["eps"], steps=cfg["steps"])
    with torch.amp.autocast("cuda", enabled=True):
        loss, _, _ = compute_trades_loss(
            m, x.clone().detach(), y.clone().detach(), x_adv, beta=2.5, w_trades=0.55
        )
    loss.backward()
    m.zero_grad(set_to_none=True)
    t1.record()
    torch.cuda.synchronize()
    step_wall_ms = round(t0.elapsed_time(t1), 3)

    return {
        "profile_events": rows,
        "total_device_ms": total_device_ms,
        "step_wall_ms": step_wall_ms,
        "kernel_count": gaps.get("n_kernels", len(intervals)),
        "launch_gaps": gaps,
        "sdpa_backend": sdpa_backend_record(),
        "batch_size": cfg["batch_size"],
        "img_size": 112,
        "n_profile_steps": n_profile,
        "s_per_step_measured": round(step_wall_ms / 1000.0, 4),
        "img_per_sec_measured": round(cfg["batch_size"] / (step_wall_ms / 1000.0), 2),
        "ce": float(ce_val) if ce_val is not None else None,
        "kl": float(kl_val) if kl_val is not None else None,
    }


def emission() -> None:
    torch.backends.cudnn.benchmark = True
    line("=== Stage-0 profiler probe (separate runtime) ===")
    line("device: " + torch.cuda.get_device_name(0))
    line("torch: " + str(torch.__version__) + " cuda: " + str(torch.version.cuda))

    cfg = build_model_and_batch(batch_size=48, seed=7)
    line(
        "batch_size: "
        + str(cfg["batch_size"])
        + " eps: "
        + str(cfg["eps"])
        + " steps: "
        + str(cfg["steps"])
    )

    res = run_trained_step_under_profiler(cfg, n_warmup=3, n_profile=1)

    line("")
    line("=== SDPA + cuDNN backend state during profile ===")
    line(json.dumps(res["sdpa_backend"], indent=2))

    line("")
    line("=== Timing result ===")
    line("total_device_ms: " + str(res["total_device_ms"]))
    line("measured_step_wall_ms: " + str(res["step_wall_ms"]))
    line("s_per_step: " + str(res["s_per_step_measured"]))
    line("img_per_sec: " + str(res["img_per_sec_measured"]))
    line("kernel_count: " + str(res["kernel_count"]))

    line("")
    line("=== Launch gaps (non-overlapping CUDA kernel gaps) ===")
    line(json.dumps(res["launch_gaps"], indent=2))

    line("")
    line("=== Top device-time events ===")
    line(f"{'event':56s} {'dev_ms':>9s} {'self_ms':>9s} {'cpu_ms':>9s} {'count':>7s}")
    for r in res["profile_events"][:50]:
        line(
            f"{r['event'][:56]:56s} {r['device_time_ms']:9.3f} "
            f"{r['self_device_time_ms']:9.3f} {r['cpu_time_ms']:9.3f} {r['count']:7d}"
        )

    line("")
    line("NOTE: single profile on an RTX 4060. Structural probe, NOT the T4 s/step.")

    profile_path = os.path.join(REPORT_DIR, "stage0_profile.json")
    payload = {
        "machine": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "cuda_version": getattr(torch.version, "cuda", None),
        },
        "config": {
            "batch_size": cfg["batch_size"],
            "eps": cfg["eps"],
            "steps": cfg["steps"],
            "img_size": 112,
            "n_warmup": 3,
            "n_profile": 1,
            "seed": 7,
        },
        "result": res,
    }
    with open(profile_path, "w") as f:
        json.dump(payload, f, indent=2)

    text_path = os.path.join(REPORT_DIR, "stage0_profile.txt")
    with open(text_path, "w") as f:
        f.write("\n".join(OUT) + "\n")

    line("")
    line("Wrote: " + profile_path)
    line("Wrote: " + text_path)


if __name__ == "__main__":
    emission()

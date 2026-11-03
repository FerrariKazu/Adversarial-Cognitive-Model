"""
Stage-0b: per-forward cost decomposition.

Explains why a TRADES-PGD-4 step at 112 px costs what it costs: how many trunk
passes one `Gen2FoundationModel.forward` performs, and how one clean forward
compares against a plain 64-token feedforward trunk pass at the same resolution.

Writes report/audit_rev1/stage0b_forward_decomposition.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.backbone import CompactViTGen2  # noqa: E402
from gen2_foundation.model import Gen2FoundationModel  # noqa: E402

REPORT_DIR = os.path.join(REPO_ROOT, "report", "audit_rev1")
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH = 48
WARMUP = 3
TIMED = 7


def count_trunk_passes(model: Gen2FoundationModel, x: torch.Tensor) -> Dict[str, int]:
    """Wrap the backbone trunk and count calls in one model forward."""
    counts = {"trunk_forward": 0, "patch_embed": 0, "refine": 0}
    bb = model.backbone
    orig_trunk = bb._trunk_forward
    orig_embed = bb.patch_embed.forward

    def trunk(*a, **k):
        counts["trunk_forward"] += 1
        return orig_trunk(*a, **k)

    def embed(*a, **k):
        counts["patch_embed"] += 1
        return orig_embed(*a, **k)

    bb._trunk_forward = trunk
    bb.patch_embed.forward = embed
    try:
        with torch.no_grad():
            model(x)
    finally:
        bb._trunk_forward = orig_trunk
        bb.patch_embed.forward = orig_embed
    return counts


def time_forward(fn, n_warmup: int = WARMUP, n_timed: int = TIMED) -> Dict[str, Any]:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts: List[float] = []
    for _ in range(n_timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return {"median_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3)}


def main() -> None:
    torch.backends.cudnn.benchmark = True
    print("=== Stage-0b: forward decomposition ===")
    print("device:", torch.cuda.get_device_name(0))

    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(BATCH, 3, 112, 112, device="cuda", generator=g)

    out: Dict[str, Any] = {
        "machine": {"device": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__)},
        "config": {"batch": BATCH, "img_size": 112, "warmup": WARMUP, "timed": TIMED},
    }

    torch.manual_seed(7)
    rec = Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112).eval().cuda()
    rec.backbone.load_dino_warm_start(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"), strict_band=0.90
    )
    counts = count_trunk_passes(rec, x)
    print("recurrence_only trunk calls per forward:", counts)
    out["recurrence_only"] = {
        "trunk_forward_calls_per_model_forward": counts["trunk_forward"],
        "patch_embed_calls_per_model_forward": counts["patch_embed"],
        "num_glimpses": rec.num_glimpses,
        "within_glimpse_iters": rec.within_glimpse_iters,
        "fovea_size": rec.fovea_size,
        "clean_forward_ms": time_forward(lambda: rec(x))["median_ms"],
    }
    print("recurrence_only clean forward ms:", out["recurrence_only"]["clean_forward_ms"])

    # Plain 64-token trunk pass at 112 px: the ff_112 substrate.
    trunk_model = CompactViTGen2(img_size=112, num_refine_iters=2).eval().cuda()
    trunk_model.load_dino_warm_start(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"), strict_band=0.90
    )

    def ff_pass():
        with torch.no_grad():
            trunk_model._trunk_forward(trunk_model._prep(x))

    t_ff = time_forward(ff_pass)
    out["ff_112_trunk_pass"] = {"clean_forward_ms": t_ff["median_ms"]}
    print("ff_112 single trunk pass ms:", t_ff["median_ms"])

    r = out["recurrence_only"]
    out["ratios"] = {
        "recurrence_forward_over_ff_trunk_pass": round(
            r["clean_forward_ms"] / t_ff["median_ms"], 2
        ),
        "implied_trunk_passes_per_recurrence_forward": counts["trunk_forward"],
    }

    # What a TRADES-PGD-4 step costs in trunk passes.
    out["trades_pgd4_step"] = {
        "model_forwards": {
            "attack_clean_fp32_nograd": 1,
            "attack_adv_fp32_grad": 4,
            "trades_clean_autocast_grad": 1,
            "trades_adv_autocast_grad": 1,
            "total": 7,
        },
        "trunk_passes_per_step": 7 * counts["trunk_forward"],
        "trunk_passes_per_step_vs_ff_trades_pgd4": round(
            (7 * counts["trunk_forward"]) / 6.0, 2
        ),
        "note": (
            "ff_112 TRADES-PGD4 uses the same 6-forward structure on a single-pass "
            "trunk; hence the measured 0.36-0.45 s/batch there vs the recurrent step here."
        ),
    }

    p = os.path.join(REPORT_DIR, "stage0b_forward_decomposition.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", p)
    print(json.dumps(out["ratios"], indent=2))
    print(json.dumps(out["trades_pgd4_step"], indent=2))


if __name__ == "__main__":
    main()

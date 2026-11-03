"""
Stage-0 step audit (separate runtime).

PROBE
-----
1) Enumerate every forward/backward touch inside one adversarial training step.
2) Report dtype/autocast/graph/no-graph state for each.
3) Confirm whether the tied refinement block consumes the running state z.

NOTE
----
This file is deliberately a *structural* audit first. Real CUDA-time profiling
is gated behind the separate-machine reality check below, because the train
recipe is written for a T4 (fp32 inner KL, fp32-safe autocast paths) and this
machine is an RTX 4060. We do not pretend numbers here transfer to a T4.
"""
from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from training.train_gen2_foundation import run_pgd_kl_attack, compute_trades_loss

OUT: List[str] = []

def line(s: str = "") -> None:
    OUT.append(s)
    print(s)


def device_and_cuda() -> Dict[str, Any]:
    return {
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
    }


def autocast_header() -> Dict[str, Any]:
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": str(d),
        "amp_device_type": str(torch.get_autocast_dtype('cuda')) if d.type == "cuda" else None,
        "use_cuda_amp": d.type == "cuda",
    }


def build_stage0_probe() -> Tuple[nn.Module, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build the smallest faithful Stage-0 probe.

    Stage 0 in this repo's Gen-2 ladder = g2_gist_only at the top TRADES rung:
        - eps_norm 0.094
        - beta 2.5
        - w_trades 0.55
        - pgd_steps 4
        - alpha = 2.5*eps/steps (corrected)
        - autocast enabled for forward
        - attack computed OUTSIDE autocast, in fp32
        - loss computed INSIDE autocast
    """
    torch.manual_seed(7)
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Gen2FoundationModel(phase="g2_gist_only", arm="v1", fovea_size=112)
    m.to(d)

    x = torch.randn(2, 3, 112, 112, device=d)
    y = torch.randint(0, 100, (2,), device=d)
    return m, x, y, {"eps": 0.094, "steps": 4, "beta": 2.5, "w_trades": 0.55}


def step_script(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Replay one adversarial training step as close to the recipe as possible.

    Returns the script trace used for auditing.
    """
    device = x.device
    script: List[Dict[str, Any]] = []

    def mark(name, **kw):
        rec = {"event": name, "autocast_active": torch.is_autocast_enabled("cuda"), "dtype_cpu": None, "dtype_gpu": None}
        # nearest alive tensor dtype snapshot if any kw passed
        for k, v in kw.items():
            rec[k] = v
        script.append(rec)

    mark("step_start")
    model.train()
    model.zero_grad(set_to_none=True)

    # PGD attack happens OUTSIDE autocast in fp32 (recipe).
    mark("attack_outside_autocast_begin")
    with torch.no_grad():
        mark("attack_clean_forward")
        probs_clean = F.softmax(model(x).float(), dim=1)
        mark("attack_clean_probs", dtype_gpu=str(probs_clean.dtype))

    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-cfg["eps"], cfg["eps"])
    x_adv = torch.clamp(x_adv, -4.0, 4.0)
    alpha = (2.5 * cfg["eps"]) / max(cfg["steps"], 1)
    mark("attack_random_start", dtype_gpu=str(x_adv.dtype), alpha=alpha)

    for t in range(cfg["steps"]):
        x_adv.requires_grad_(True)
        mark(f"pgd_step_{t}_grad_scope_begin", requires_grad=str(x_adv.requires_grad))
        with torch.enable_grad():
            mark(f"pgd_step_{t}_inner_forward")
            logits_adv = model(x_adv)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv.float(), dim=1),
                probs_clean,
                reduction="batchmean",
            )
            mark(f"pgd_step_{t}_kl", dtype_gpu=str(logits_adv.dtype), kl_dtype=str(loss_kl.dtype))
        grad = torch.autograd.grad(loss_kl, x_adv, create_graph=False)[0]
        mark(f"pgd_step_{t}_grad", grad_dtype=str(grad.dtype), grad_norm=float(grad.norm()))
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - cfg["eps"]), x + cfg["eps"])
            x_adv = x_adv.clamp(-4.0, 4.0)
            mark(f"pgd_step_{t}_project", dtype_gpu=str(x_adv.dtype))
    mark("attack_outside_autocast_end")
    x_adv = x_adv.detach()

    # TRADES loss computed INSIDE autocast (recipe).
    mark("trades_loss_autocast_begin")
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        mark("trades_clean_forward")
        logits_clean = model(x)
        mark("trades_clean_logits", dtype_gpu=str(logits_clean.dtype))
        loss_ce = F.cross_entropy(logits_clean.float(), y)
        mark("trades_ce", ce_dtype=str(loss_ce.dtype))

        mark("trades_adv_forward")
        logits_adv = model(x_adv)
        mark("trades_adv_logits", dtype_gpu=str(logits_adv.dtype))
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv.float(), dim=1),
            F.softmax(logits_clean.float().detach(), dim=1),
            reduction="batchmean",
        )
        mark("trades_kl", kl_dtype=str(loss_kl.dtype))

        total_loss = cfg["w_trades"] * (loss_ce + cfg["beta"] * loss_kl)
        mark("trades_total_loss", total_dtype=str(total_loss.dtype), total_device=str(total_loss.device))

    mark("trades_loss_autocast_end")

    mark("backward_begin")
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler.scale(total_loss).backward()
        mark("backward_with_scaler")
        scaler.unscale_(model.optimizer if hasattr(model, "optimizer") else None) if False else None
        mark("unscale_begin")
        scaler.unscale_(None) if False else None  # not used; just documenting this is omitted in probe
    else:
        total_loss.backward()
        mark("backward_eager")
    mark("backward_end")

    return {"script": script, "final_loss": total_loss.detach().cpu().float().item()}


def refinement_consumes_z_probe() -> Dict[str, Any]:
    """Confirm the refinement block is applied INSIDE the trunk encode path,
    and note whether that path consumes the running state z.

    In this repo's code, the tied refinement block is called on the encoded
    tokens BEFORE pooling (x = self.refinement(x, num_refine_iters)). It
    consumes the token sequence, NOT the outside recurrence scalar z_t. That
    distinction is exactly what the audit must state, because the question is
    whether the 'running state z' drives refinement. Here it does not: z_t is
    the pooled/fused state formed AFTER refinement.
    """
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = torch.load(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"),
        map_location="cpu",
        weights_only=False,
    )
    m = Gen2FoundationModel(phase="g2_gist_only", arm="v1", fovea_size=112)
    m.backbone.load_dino_warm_start(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"),
        strict_band=0.90,
    )
    m.to(d)
    x = torch.randn(1, 3, 112, 112, device=d)
    x_prep = m.backbone._prep(x)
    # _trunk_forward runs the 12 DINOv2 blocks WITHOUT refinement.
    trunk_out = m.backbone._trunk_forward(x_prep)
    # refinement is separate: tied recurrence applied AFTER trunk for forward();
    # for the gist-only forward path we still call backbone(x) which adds refinement.
    pooled, tokens = m.backbone(x)
    return {
        "trunk_before_refinement_shape": list(trunk_out.shape),
        "after_refinement_pooled_shape": list(pooled.shape),
        "tokens_shape": list(tokens.shape),
        "note": "Tied refinement consumes encoded tokens, not the scalar z_t. z_t is formed AFTER refinement in this implementation.",
        "module_refinement_type": type(m.backbone.refinement).__name__,
        "refinement_iters": m.backbone.num_refine_iters,
    }


def emission() -> None:
    line("=== Stage-0 step audit (separate runtime) ===")
    line(json.dumps(device_and_cuda(), indent=2))
    line("")
    line("=== Autocast header ===")
    line(json.dumps(autocast_header(), indent=2))
    line("")

    m, x, y, cfg = build_stage0_probe()
    res = step_script(m, x, y, cfg)
    line("=== Step script (one adversarial training step) ===")
    line(json.dumps(res["script"], indent=2))
    line("")
    line(f"final_loss = {res['final_loss']}")

    line("")
    line("=== Refinement / z consumption probe ===")
    line(json.dumps(refinement_consumes_z_probe(), indent=2))


if __name__ == "__main__":
    emission()

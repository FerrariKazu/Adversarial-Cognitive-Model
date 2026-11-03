"""
Recipe-change track: autocast-inside-inner-KL.

GOAL
----
Quantify what changes if the inner PGD-KL attack is run *inside* autocast
(fp16 matmuls) instead of the recipe's fp32 inner attack. This is a
"recipe change" probe, reported separately, and is NOT recommended without a
harness that compares training-trajectory behavior against the fp32 attack.

MEASURES
--------
- realised L-inf vs the recipe eps 0.094 (already normalised pixel space)
- inner-gradient magnitude and underflow/near-zero fraction
- attack-loss trajectory per step (KL vs clean)
- PGD-monitor-style CE under fp32 final eval

CONSTRAINTS
-----------
Do not touch the in-flight Stage 0 run.
Report only; this file does not produce a recommendation.
"""
from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Tuple

import torch
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

def step_label(t: int) -> str:
    return f"pgd_step_{t}"


def inner_kl_attack_with_mode(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    eps: float,
    steps: int,
    autocast_kl: bool,
    seed: int,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Run the inner PGD-KL attack once, optionally inside autocast.

    Returns (x_adv, step_log).
    """
    g = torch.Generator().manual_seed(seed + (1 if autocast_kl else 0))
    model.eval()
    with torch.no_grad():
        if autocast_kl:
            with torch.amp.autocast("cuda", enabled=True):
                probs_clean = F.softmax(model(x).float(), dim=1)
        else:
            probs_clean = F.softmax(model(x).float(), dim=1)

    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, -4.0, 4.0)
    alpha = (2.5 * eps) / max(steps, 1)

    step_log: List[Dict[str, float]] = []
    for t in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            if autocast_kl:
                with torch.amp.autocast("cuda", enabled=True):
                    logits_adv = model(x_adv)
                    loss_kl = F.kl_div(
                        F.log_softmax(logits_adv.float(), dim=1),
                        probs_clean,
                        reduction="batchmean",
                    )
            else:
                logits_adv = model(x_adv)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv.float(), dim=1),
                    probs_clean,
                    reduction="batchmean",
                )
        grad = torch.autograd.grad(loss_kl, x_adv, retain_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = x_adv.clamp(-4.0, 4.0)

        step_log.append({
            "step": t,
            "kl": float(loss_kl.detach().cpu()),
            "grad_norm": float(grad.norm()),
            "grad_mean_abs": float(grad.abs().mean()),
            "grad_frac_lt_1e4": float((grad.abs() < 1e-4).float().mean()),
            "grad_frac_lt_1e7": float((grad.abs() < 1e-7).float().mean()),
        })
    return x_adv.detach(), step_log


def emission() -> None:
    torch.backends.cudnn.benchmark = True
    line("=== Recipe-change track: inner-KL autocast probe ===")
    line("device: " + str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(21)
    B = 48
    eps = 0.094
    steps = 4

    m = Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112).eval().cuda()
    try:
        rep = m.backbone.load_dino_warm_start(
            os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"),
            strict_band=0.90,
        )
        line("warm_start accepted: " + str(rep["accepted"]) + " matched_fraction: " + str(round(rep["matched_fraction"], 3)))
    except Exception as e:
        line("warm_start note: " + str(e))

    # Use batch 32, not 48, so the fp32-vs-amp comparison fits on this 8.59GB machine.
    B = 32
    x = torch.randn(B, 3, 112, 112, device="cuda")
    y = torch.randint(0, 100, (B,), device="cuda")

    x_adv_fp32, log_fp32 = inner_kl_attack_with_mode(
        m, x, eps=eps, steps=steps, autocast_kl=False, seed=21,
    )
    # detach the probe model from the fp32 attack's autograd graph so the
    # amp run does not keep fp32 attack intermediates alive on the GPU.
    torch.cuda.empty_cache()
    m_amp = Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112).eval().cuda()
    m_amp.backbone.load_dino_warm_start(
        os.path.join(REPO_ROOT, "checkpoints", "dinov2_vits14_pretrain.pth"),
        strict_band=0.90,
    )
    x_adv_amp, log_amp = inner_kl_attack_with_mode(
        m_amp, x.clone().detach(), eps=eps, steps=steps, autocast_kl=True, seed=21,
    )

    line("")
    line("=== Realised L-inf (normalised pixel space; recipe radius 0.094) ===")
    line("fp32 L-inf: " + str(float((x_adv_fp32 - x).abs().max())))
    line("amp  L-inf: " + str(float((x_adv_amp - x).abs().max())))
    line("max abs diff between fp32 and amp x_adv: " + str(float((x_adv_fp32 - x_adv_amp).abs().max())))

    line("")
    line("=== Inner-gradient magnitude and underflow fractions ===")
    for which, log in (("fp32", log_fp32), ("amp", log_amp)):
        line("-- " + which + " --")
        for row in log:
            line(
                "  step " + str(row['step']) + ": kl=" + f"{row['kl']:.4f}" + "  "
                + "grad_norm=" + f"{row['grad_norm']:.4g}" + "  grad_mean_abs=" + f"{row['grad_mean_abs']:.4g}" + "  "
                + "frac_lt_1e-4=" + f"{row['grad_frac_lt_1e4']:.3f}" + "  frac_lt_1e-7=" + f"{row['grad_frac_lt_1e7']:.3f}"
            )

    line("")
    line("=== TRADES loss with recipe x_adv vs amp x_adv (autocast forward, recipe loss) ===")
    with torch.amp.autocast("cuda", enabled=True):
        loss_fp32, ce_fp32, kl_fp32 = compute_trades_loss(
            m, x, y, x_adv_fp32, beta=2.5, w_trades=0.55,
        )
        loss_amp, ce_amp, kl_amp = compute_trades_loss(
            m, x, y, x_adv_amp, beta=2.5, w_trades=0.55,
        )
    line("fp32 x_adv  -> loss/ce/kl: " + str(round(float(loss_fp32), 4)) + " / " + str(round(float(ce_fp32), 4)) + " / " + str(round(float(kl_fp32), 4)))
    line("amp  x_adv  -> loss/ce/kl: " + str(round(float(loss_amp), 4)) + " / " + str(round(float(ce_amp), 4)) + " / " + str(round(float(kl_amp), 4)))

    line("")
    line("=== PGD-monitor-style CE eval (fp32, recipe eval function) ===")
    with torch.amp.autocast("cuda", enabled=True):
        rob_fp32 = float((m(x_adv_fp32).argmax(dim=-1) == y).float().mean())
        rob_amp  = float((m(x_adv_amp).argmax(dim=-1) == y).float().mean())
    line("robust-accuracy-under-x_adv (batch mean): fp32 x_adv: " + str(round(float(rob_fp32), 3)) + " amp x_adv: " + str(round(float(rob_amp), 3)))

    line("")
    line("NOTE: this probe compares two attacks on one fixed batch only.")
    line("A recommendation would require a matched training-harness comparison,")
    line("which is out of scope for this audit and is therefore NOT made here.")


if __name__ == "__main__":
    emission()

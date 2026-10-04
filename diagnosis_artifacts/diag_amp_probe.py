#!/usr/bin/env python3
"""
diag_amp_probe.py — §15 execution-trace probe: why do §8 (AMP recipe path)
and §9 (fp32 decomposition) disagree by orders of magnitude?
Cases on the SAME checkpoint + SAME train batch (48):
  A) fp32, CE only                      -> backbone/classifier grad norms
  B) autocast + GradScaler, CE only     -> same, after unscale_
  C) fp32, full TRADES (attack fp32)    -> same
  D) autocast + GradScaler, full TRADES -> same (the production path)
  E) D + inspect scaler: was the step skipped? measure actual |dW| vs lr*|g|
This is a MEASUREMENT probe, not a tuning step.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT  # noqa: E402

from training.adv_curriculum import pgd_kl_attack, trades_loss  # noqa: E402
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, build_model)

PURE_CE_CKPT = os.path.join(REPO_ROOT, "checkpoints",
                            "foundation_belief_with_f_best.pth")
device = torch.device("cuda")


def gnorms(model, names=("backbone", "classifier")):
    out = {}
    groups = model.group_params()
    for n in names:
        ps = groups[n]
        gs = [p.grad for p in ps if p.grad is not None]
        out[n] = round(sum(float(g.norm() ** 2) for g in gs) ** 0.5, 5) if gs else None
    return out


def main() -> int:
    cfg = FoundationConfig()
    model = build_model(cfg, "belief_with_f").to(device)
    ck = torch.load(PURE_CE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.train()

    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=2)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    W, BETA, EPS = 0.55, 2.0, 0.031

    results = {}

    def case(tag, use_amp, use_trades):
        model.zero_grad(set_to_none=True)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        pre = {n: sum(float(p.norm() ** 2) for p in g)
               for n, g in model.group_params().items() if n in ("backbone", "classifier")}
        with torch.autocast("cuda", enabled=use_amp):
            if use_trades:
                x_adv = pgd_kl_attack(model, x, eps=EPS, steps=4)
                loss, _ = trades_loss(model, x, y, x_adv, beta=BETA)
                loss = W * loss
            else:
                loss = F.cross_entropy(model(x).float(), y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.0))  # placeholder
        else:
            loss.backward()
        results[tag] = {"loss": round(loss.item(), 4), **gnorms(model)}
        print(f"  {tag}: loss={results[tag]['loss']} grads={gnorms(model)}")
        model.zero_grad(set_to_none=True)

    case("A_fp32_CE", False, False)
    case("B_amp_CE", True, False)
    case("C_fp32_TRADES", False, True)
    case("D_amp_TRADES", True, True)

    # -- E: full production step mechanics: is the step skipped? --------------
    from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
    model.zero_grad(set_to_none=True)
    reg = OptimizerGroupRegistry()
    groups = model.group_params()
    reg.register_backbone(groups["backbone"])
    for n in ("classifier", "evidential_head", "predictor", "update_net", "precision"):
        if n in groups:
            reg.register(n, groups[n])
    opt = reg.build_optimizer(0.003, 0.9, 1e-4)
    scaler = torch.amp.GradScaler("cuda")
    pre_bb = sum(float(p.norm() ** 2) for p in groups["backbone"]) ** 0.5
    with torch.autocast("cuda"):
        x_adv = pgd_kl_attack(model, x, eps=EPS, steps=4)
        loss, _ = trades_loss(model, x, y, x_adv, beta=BETA)
        loss = W * loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    post_unscale = gnorms(model)
    scaler.step(opt)
    scaler.update()
    post_bb = sum(float(p.norm() ** 2) for p in groups["backbone"]) ** 0.5
    predicted_dw = 0.003 * post_unscale["backbone"]  # lr * |g| (first step, mom=0)
    print(f"  E: scale={scaler.get_scale()} grads_after_unscale={post_unscale}")
    print(f"     |dW|_backbone measured={pre_bb - post_bb:+.6f}  "
          f"predicted(lr*|g|)={predicted_dw:.6f}")
    results["E_step_mechanics"] = {
        "scaler_scale": scaler.get_scale(),
        "grads_after_unscale": post_unscale,
        "backbone_dW_measured": round(pre_bb - post_bb, 6),
        "backbone_dW_predicted_lr_times_g": round(predicted_dw, 6)}
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "out", "15_amp_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

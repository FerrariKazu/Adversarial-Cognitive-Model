#!/usr/bin/env python3
"""
diag_amp_bisect.py — pin the exact mechanism killing the classifier gradient.
================================================================================
Established (diag_amp_probe): raw ckpt + batch —
  fp32 CE:  backbone 31.65, classifier 2.76     (healthy)
  AMP  CE:  backbone 31.65, classifier 2.76     (healthy)
  fp32 TRADES: backbone 35.03, classifier 2.23  (healthy)
  AMP  TRADES: backbone ~2.0,  classifier NONE  (BROKEN — production path)

Bisect arms (all on the same ckpt + batch):
  F  AMP TRADES, x_adv = fixed small perturbation (attack REMOVED)
  G  AMP forward, loss computed OUTSIDE autocast (attack kept, fp32 loss)
  H  AMP TRADES full production order (registry + clip + step): inf/nan
     counts per group, post-clip norms, step-skip detection, dW
  I  like H but WITHOUT the clip call (isolates clip's role)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, out_path  # noqa: E402

from training.adv_curriculum import pgd_kl_attack, trades_loss  # noqa: E402
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, build_model)

PURE_CE_CKPT = os.path.join(REPO_ROOT, "checkpoints",
                            "foundation_belief_with_f_best.pth")
device = torch.device("cuda")


def fresh_model():
    cfg = FoundationConfig()
    m = build_model(cfg, "belief_with_f").to(device)
    ck = torch.load(PURE_CE_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.train()
    return m


def group_report(model, tag):
    groups = model.group_params()
    rep = {}
    for n, ps in groups.items():
        gs = [p.grad for p in ps if p.grad is not None]
        norm = sum(float(g.norm() ** 2) for g in gs) ** 0.5 if gs else 0.0
        n_inf = sum(int(torch.isinf(g).any().item()) for g in gs)
        n_nan = sum(int(torch.isnan(g).any().item()) for g in gs)
        rep[n] = {"grad_norm": round(norm, 6), "tensors_with_grad": len(gs),
                  "inf_tensors": n_inf, "nan_tensors": n_nan}
    print(f"  [{tag}] {rep}")
    return rep


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=2)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    W, BETA, EPS = 0.55, 2.0, 0.031
    results = {}

    # -- F: AMP TRADES without the attack -------------------------------------
    m = fresh_model()
    with torch.autocast("cuda"):
        logits_c = m(x)
        logits_a = m(x + 0.031)                       # fixed perturbation
        ce = F.cross_entropy(logits_c.float(), y)
        kl = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                      F.softmax(logits_c.float().detach(), dim=1),
                      reduction="batchmean")
        loss = W * (ce + BETA * kl)
    loss.backward()
    results["F_amp_trades_fixed_perturb"] = group_report(m, "F")
    m.zero_grad(set_to_none=True)

    # -- G: AMP forward, fp32 loss outside autocast (attack kept) -------------
    m = fresh_model()
    x_adv = pgd_kl_attack(m, x, eps=EPS, steps=4)     # attack outside autocast
    with torch.autocast("cuda"):
        logits_c = m(x)
        logits_a = m(x_adv)
    ce = F.cross_entropy(logits_c.float(), y)
    kl = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                  F.softmax(logits_c.float().detach(), dim=1),
                  reduction="batchmean")
    loss = W * (ce + BETA * kl)
    loss.backward()
    results["G_amp_forward_fp32_loss"] = group_report(m, "G")
    m.zero_grad(set_to_none=True)

    # -- H/I: full production order, with and without clip ---------------------
    from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
    for tag, do_clip in (("H_full_production_order", True),
                         ("I_no_clip", False)):
        m = fresh_model()
        reg = OptimizerGroupRegistry()
        groups = m.group_params()
        reg.register_backbone(groups["backbone"])
        for n in ("classifier", "evidential_head", "predictor", "update_net",
                  "precision"):
            if n in groups:
                reg.register(n, groups[n])
        opt = reg.build_optimizer(0.003, 0.9, 1e-4)
        scaler = torch.amp.GradScaler("cuda")
        pre = {n: sum(float(p.norm() ** 2) for p in groups[n]) ** 0.5
               for n in ("backbone", "classifier")}
        with torch.autocast("cuda"):
            x_adv = pgd_kl_attack(m, x, eps=EPS, steps=4)
            loss, _ = trades_loss(m, x, y, x_adv, beta=BETA)
            loss = W * loss
        print(f"  [{tag}] loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        results[f"{tag}_after_unscale"] = group_report(m, tag + "/unscaled")
        if do_clip:
            reg.clip_grad_per_group()
            results[f"{tag}_after_clip"] = group_report(m, tag + "/clipped")
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < 65536
        post = {n: sum(float(p.norm() ** 2) for p in groups[n]) ** 0.5
                for n in ("backbone", "classifier")}
        dW = {n: round(pre[n] - post[n], 6) for n in pre}
        results[f"{tag}_step"] = {"scale_after": scaler.get_scale(),
                                  "step_skipped": skipped, "dW": dW}
        print(f"  [{tag}] step_skipped={skipped} scale_after="
              f"{scaler.get_scale()} dW={dW}")

    import json
    with open(out_path("15_amp_bisect.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("bisect saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

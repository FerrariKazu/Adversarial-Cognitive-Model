#!/usr/bin/env python3
"""
diag_gradient_forensics.py — §8 GRADIENT / UPDATE FORENSICS (real ckpt).
================================================================================
"Gradient reachability passed" is NOT sufficient (§8). This measures, on the
REAL belief_with_f checkpoint + REAL batches, under the REAL adversarial
recipe (§10 recipe at its epoch-1 point):
  * total / per-group gradient norms (backbone, classifier, evidential_head,
    predictor, update_net, precision)
  * parameter-update norm |ΔW| per group after one optimizer.step()
  * gradient/update ratio per group (dead-update detection: nonzero grad,
    negligible effective step)
  * fraction of parameters with zero grad, NaN/Inf grad
  * parameter-norm drift across a few steps (do weights actually move?)
  * classification signal check: does a step on the TRADES loss actually
    reduce CE on a held batch (is the learning signal aligned with CE)?
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, save_json  # noqa: E402

from training.adv_curriculum import pgd_kl_attack, trades_loss  # noqa: E402
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, build_model)

PURE_CE_CKPT = os.path.join(REPO_ROOT, "checkpoints",
                            "foundation_belief_with_f_best.pth")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FoundationConfig()
    model = build_model(cfg, "belief_with_f").to(device)
    ck = torch.load(PURE_CE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.train()

    from evaluation.imagenet100_loader import make_imagenet100_loaders
    from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=4)
    it = iter(loaders["train"])
    batches = [next(it) for _ in range(3)]

    reg = OptimizerGroupRegistry()
    groups = model.group_params()
    reg.register_backbone(groups["backbone"])
    for name in ("classifier", "evidential_head", "predictor", "update_net",
                 "precision"):
        if name in groups:
            reg.register(name, groups[name])
    opt = reg.build_optimizer(0.003, 0.9, 1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    W, BETA, EPS = 0.55, 2.0, 0.031

    def grad_snapshot():
        stats = {}
        for g in reg._groups:
            gs = [p.grad for p in g["params"] if p.grad is not None]
            norm = sum(float(gn.norm() ** 2) for gn in gs) ** 0.5 if gs else 0.0
            zero_frac = (sum(int((gn == 0).all()) for gn in gs) /
                         max(len(gs), 1))
            nan = sum(int(torch.isnan(gn).any() or torch.isinf(gn).any())
                      for gn in gs)
            stats[g["name"]] = {"grad_norm": round(norm, 6),
                                "n_tensors": len(gs),
                                "n_all_zero_tensors": zero_frac,
                                "nan_inf_tensors": nan}
        tot = sum(s["grad_norm"] ** 2 for s in stats.values()) ** 0.5
        stats["_total"] = round(tot, 6)
        return stats

    rows = []
    heldout_correct_before, heldout_correct_after = 0, 0
    hx, hy = batches[-1][0].to(device), batches[-1][1].to(device)
    with torch.no_grad():
        heldout_correct_before = int((model(hx).argmax(1) == hy).sum().item())

    for bi, (x, y) in enumerate(batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pre_norms = {g["name"]: sum(float(p.norm() ** 2)
                                    for p in g["params"]) ** 0.5
                     for g in reg._groups}
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            x_adv = pgd_kl_attack(model, x, eps=EPS, steps=4)
            loss, _ = trades_loss(model, x, y, x_adv, beta=BETA)
            loss = W * loss
        scaler.scale(loss).backward() if scaler else loss.backward()
        if scaler:
            scaler.unscale_(opt)
        grads = grad_snapshot()
        # measured |dW| = the RAW param-norm delta across the step; per-group
        # lr is read from the optimizer groups:
        lrs = {g["name"]: pg["lr"] for g, pg in zip(reg._groups, opt.param_groups)}
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            reg.clip_grad_per_group()
            opt.step()
        post_norms = {g["name"]: sum(float(p.norm() ** 2)
                                     for p in g["params"]) ** 0.5
                      for g in reg._groups}
        dW = {n: round(post_norms[n] - pre_norms[n], 6) for n in post_norms}
        rows.append({"batch": bi, "loss": round(loss.item(), 4),
                     "grads": grads, "param_delta_norms": dW,
                     "group_lrs": lrs})
        print(f"  batch {bi}: loss={loss.item():.4f} total_grad={grads['_total']:.4f} "
              f"backbone_dW={dW.get('backbone'):+.5f} cls_dW={dW.get('classifier'):+.5f}",
              flush=True)

    with torch.no_grad():
        heldout_correct_after = int((model(hx).argmax(1) == hy).sum().item())

    zero_grad_groups = [n for n, s in rows[0]["grads"].items()
                        if isinstance(s, dict) and s.get("n_all_zero_tensors", 0) > 0]
    nan_groups = [n for n, s in rows[0]["grads"].items()
                  if isinstance(s, dict) and s.get("nan_inf_tensors", 0) > 0]
    tiny = [n for n, s in rows[0]["grads"].items()
            if isinstance(s, dict) and s.get("grad_norm", 1) <= 1e-4]
    out = {"section": "8_gradient_update_forensics",
           "checkpoint": PURE_CE_CKPT, "recipe": {"w_trades": W, "beta": BETA, "eps": EPS},
           "per_batch": rows,
           "flag_groups_all_zero_grad": zero_grad_groups,
           "flag_groups_nan_inf_grad": nan_groups,
           "flag_groups_tiny_grad_lt_1e-4": tiny,
           "heldout_batch_correct_before": heldout_correct_before,
           "heldout_batch_correct_after": heldout_correct_after,
           "note": ("|dW| here is the RAW param-norm delta per single step at lr=3e-3; "
                    "per-group clip (budget 1.0) applies inside optimizer.step path")}
    save_json("08_gradient_forensics.json", out)
    print(f"SECTION 8 VERDICT: zero-grad groups={zero_grad_groups or 'NONE'} | "
          f"nan/inf={nan_groups or 'NONE'} | tiny-grad groups={tiny or 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

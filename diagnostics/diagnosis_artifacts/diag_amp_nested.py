#!/usr/bin/env python3
"""
diag_amp_nested.py — last mechanism cell.
  Q1: attack run with nested autocast(enabled=False) INSIDE the outer region
      (attack effectively fp32), training loss in the OUTER region.
      healthy  -> the fp16 attack graph/casts are implicated
      broken   -> merely executing the attack's no_grad/grad forwards inside
                  the region is enough
  Q2: dose-response: production placement with attack steps=1 vs 4.
"""
from __future__ import annotations

import os
import sys
import json

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


def report(model, tag, results):
    groups = model.group_params()
    bb, cls = groups["backbone"], groups["classifier"]
    n_bb = len([p for p in bb if p.grad is not None])
    n_cls = len([p for p in cls if p.grad is not None])
    gnorm = sum(float(p.grad.norm() ** 2) for p in bb if p.grad is not None) ** 0.5
    results[tag] = {"bb": f"{n_bb}/{len(bb)}", "cls": f"{n_cls}/{len(cls)}",
                    "bb_norm": round(gnorm, 4)}
    print(f"  [{tag}] bb={n_bb}/{len(bb)} cls={n_cls}/{len(cls)} "
          f"bb_norm={gnorm:.3f} {'BROKEN' if n_cls == 0 else 'ok'}")


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=2)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    results = {}

    # Q1: fp32 attack nested inside the outer autocast region
    m = fresh_model()
    with torch.autocast("cuda"):
        with torch.autocast("cuda", enabled=False):
            x_adv = pgd_kl_attack(m, x, eps=0.031, steps=4)
        loss, _ = trades_loss(m, x, y, x_adv, beta=2.0)
        loss = 0.55 * loss
    loss.backward()
    report(m, "Q1_attack_fp32_nested_loss_in_region", results)
    m.zero_grad(set_to_none=True)

    # Q2: dose-response on attack steps (production placement)
    for steps in (1, 4):
        m = fresh_model()
        with torch.autocast("cuda"):
            x_adv = pgd_kl_attack(m, x, eps=0.031, steps=steps)
            loss, _ = trades_loss(m, x, y, x_adv, beta=2.0)
            loss = 0.55 * loss
        loss.backward()
        report(m, f"Q2_steps{steps}", results)
        m.zero_grad(set_to_none=True)

    with open(out_path("15_amp_nested.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

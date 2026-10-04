#!/usr/bin/env python3
"""
diag_amp_autograd_grad.py — pin the trigger WITHOUT the attack at all.
  P1: inside ONE autocast region: forward -> torch.autograd.grad(x) ->
      forward -> CE -> backward.  (predicted: BROKEN if autograd.grad
      inside the region is the trigger)
  P2: identical minus the autograd.grad call (predicted: healthy)
  P3: attack inside autocast REGION 1; loss+backward in autocast REGION 2
      (predicted: healthy if region exit clears the poison)
"""
from __future__ import annotations

import os
import sys
import json

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, out_path  # noqa: E402

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
    bb = groups["backbone"]
    cls = groups["classifier"]
    n_bb = len([p for p in bb if p.grad is not None])
    n_cls = len([p for p in cls if p.grad is not None])
    results[tag] = {"backbone_tensors_with_grad": f"{n_bb}/{len(bb)}",
                    "classifier_tensors_with_grad": f"{n_cls}/{len(cls)}"}
    print(f"  [{tag}] bb={n_bb}/{len(bb)} cls={n_cls}/{len(cls)} "
          f"{'BROKEN' if n_cls == 0 else 'ok'}")


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=2)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    results = {}

    # P1: no attack — just an autograd.grad call inside the region
    m = fresh_model()
    with torch.autocast("cuda"):
        f1 = m(x)
        xa = x.clone().detach().requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(f1.float(), y), xa,
                                allow_unused=True)[0]
        f2 = m(x)
        ce = F.cross_entropy(f2.float(), y)
    ce.backward()
    report(m, "P1_autograd_grad_in_region", results)
    m.zero_grad(set_to_none=True)

    # P2: control — same but without the autograd.grad call
    m = fresh_model()
    with torch.autocast("cuda"):
        f1 = m(x)
        f2 = m(x)
        ce = F.cross_entropy(f2.float(), y)
    ce.backward()
    report(m, "P2_no_autograd_grad", results)
    m.zero_grad(set_to_none=True)

    # P3: attack in region 1, training loss in a SEPARATE region 2
    from training.adv_curriculum import pgd_kl_attack
    m = fresh_model()
    with torch.autocast("cuda"):
        x_adv = pgd_kl_attack(m, x, eps=0.031, steps=4)
    with torch.autocast("cuda"):
        loss, _ = __import__("training.adv_curriculum", fromlist=["trades_loss"]) \
            .trades_loss(m, x, y, x_adv, beta=2.0)
        loss = 0.55 * loss
    loss.backward()
    report(m, "P3_separate_regions", results)

    with open(out_path("15_amp_autograd_grad.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

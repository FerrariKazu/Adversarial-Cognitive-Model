#!/usr/bin/env python3
"""
diag_amp_flip.py — does the attack's eval()->train() flip INSIDE an autocast
region break the following in-autocast training backward?
  M: attack WITHOUT the eval/train flip (inline copy), inside autocast,
     then CE inside autocast -> healthy gradients would indict the flip.
  N: control = the flip restored (reproduces breakage).
"""
from __future__ import annotations

import os
import sys

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


def attack_no_flip(model, x, eps=0.031, steps=4):
    """pgd_kl_attack WITHOUT the eval()/train() flip — model mode untouched."""
    with torch.no_grad():
        probs_c = F.softmax(model(x).float(), dim=1)
    x_adv = torch.clamp(x.clone().detach() + 0.001 * torch.randn_like(x),
                        -4.0, 4.0)
    alpha = eps / max(steps, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_a = model(x_adv)
            loss = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                            probs_c, reduction="batchmean")
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = x_adv.clamp(-4.0, 4.0)
    return x_adv.detach()


def attack_with_flip(model, x, eps=0.031, steps=4):
    was_training = model.training
    model.eval()
    try:
        return attack_no_flip(model, x, eps, steps)
    finally:
        if was_training:
            model.train()


def report(model, tag, results):
    groups = model.group_params()
    rep = {n: (len([p for p in ps if p.grad is not None]), len(ps))
           for n, ps in groups.items()}
    bb = groups["backbone"]
    norm = sum(float(p.grad.norm() ** 2) for p in bb if p.grad is not None) ** 0.5
    cls = groups["classifier"]
    cnorm = sum(float(p.grad.norm() ** 2) for p in cls if p.grad is not None) ** 0.5
    results[tag] = {"backbone_tensors_with_grad": f"{rep['backbone'][0]}/{rep['backbone'][1]}",
                    "backbone_grad_norm": round(norm, 4),
                    "classifier_tensors_with_grad": f"{rep['classifier'][0]}/{rep['classifier'][1]}",
                    "classifier_grad_norm": round(cnorm, 6)}
    print(f"  [{tag}] {results[tag]}")


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=2)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    results = {}

    m = fresh_model()
    with torch.autocast("cuda"):
        x_adv = attack_no_flip(m, x)               # no mode flip
        ce = F.cross_entropy(m(x).float(), y)
    ce.backward()
    report(m, "M_no_flip", results)
    m.zero_grad(set_to_none=True)

    m = fresh_model()
    with torch.autocast("cuda"):
        x_adv = attack_with_flip(m, x)             # WITH the eval/train flip
        ce = F.cross_entropy(m(x).float(), y)
    ce.backward()
    report(m, "N_with_flip", results)
    m.zero_grad(set_to_none=True)

    # O: flip, but training forward OUTSIDE autocast (fp32) after an
    # in-autocast flipped attack -> does the breakage follow the region?
    m = fresh_model()
    with torch.autocast("cuda"):
        x_adv = attack_with_flip(m, x)
    ce = F.cross_entropy(m(x).float(), y)
    ce.backward()
    report(m, "O_flip_in_amp_loss_fp32", results)

    import json
    with open(out_path("15_amp_flip.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

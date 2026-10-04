#!/usr/bin/env python3
"""
diag_amp_cells.py — the missing cells + the cut-location map.
================================================================================
Known: F(no attack, loss-in-amp)=OK  G(attack-out, loss-out)=OK
       H(attack-in,  loss-in)=BROKEN (classifier 0 grad tensors, backbone 57/163)
Cells here:
  J  attack-IN autocast, loss-OUT (attack inside, loss fp32 outside)
  K  attack-OUT autocast, loss-IN  (attack fp32 outside, loss inside)
Whichever of J/K breaks names the culprit stage; the tensor-name dump of the
57 grad-bearing backbone tensors in the broken configuration locates the cut.
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


def report(model, tag, results):
    groups = model.group_params()
    rep = {}
    for n, ps in groups.items():
        gs = [p.grad for p in ps if p.grad is not None]
        rep[n] = {"grad_norm": round(sum(float(g.norm() ** 2) for g in gs) ** 0.5, 6),
                  "n": len(gs)} if gs else {"grad_norm": None, "n": 0}
    results[tag] = rep
    print(f"  [{tag}] " + " ".join(
        f"{k}={v['grad_norm'] if v['grad_norm'] is not None else 'NONE'}({v['n']})"
        for k, v in rep.items()))


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=48, num_workers=2)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    W, BETA, EPS = 0.55, 2.0, 0.031
    results = {}

    # -- J: attack INSIDE autocast, loss OUTSIDE (fp32) ------------------------
    m = fresh_model()
    with torch.autocast("cuda"):
        x_adv = pgd_kl_attack(m, x, eps=EPS, steps=4)
    logits_c = m(x)
    logits_a = m(x_adv)
    ce = F.cross_entropy(logits_c.float(), y)
    kl = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                  F.softmax(logits_c.float().detach(), dim=1), reduction="batchmean")
    loss = W * (ce + BETA * kl)
    print(f"  [J] loss={loss.item():.4f} logits_c.requires_grad="
          f"{logits_c.requires_grad} logits_a.requires_grad={logits_a.requires_grad}")
    loss.backward()
    report(m, "J_attackIN_lossOUT", results)
    m.zero_grad(set_to_none=True)

    # -- K: attack OUTSIDE autocast, loss INSIDE -------------------------------
    m = fresh_model()
    x_adv = pgd_kl_attack(m, x, eps=EPS, steps=4)
    with torch.autocast("cuda"):
        loss, _ = trades_loss(m, x, y, x_adv, beta=BETA)
        loss = W * loss
    print(f"  [K] loss={loss.item():.4f} dtype={loss.dtype}")
    loss.backward()
    report(m, "K_attackOUT_lossIN", results)
    m.zero_grad(set_to_none=True)

    # -- L: reproduce broken H and dump WHICH backbone tensors got grads -------
    m = fresh_model()
    with torch.autocast("cuda"):
        x_adv = pgd_kl_attack(m, x, eps=EPS, steps=4)
        loss, _ = trades_loss(m, x, y, x_adv, beta=BETA)
        loss = W * loss
    loss.backward()
    bb = m.group_params()["backbone"]
    got = [i for i, p in enumerate(bb) if p.grad is not None]
    names_got = [f"bb[{i}]" for i in got]
    # map indices to names
    name_list = [n for n, p in m.named_parameters() if n.startswith("backbone.")]
    got_names = [name_list[i] for i in got if i < len(name_list)]
    print(f"  [L] broken-config backbone tensors with grad: {len(got)}/{len(bb)}")
    print(f"      first: {got_names[:3]}")
    print(f"      last:  {got_names[-3:]}")
    # where is the cut? find the deepest block index that got grads
    import re
    block_ids = sorted({int(re.match(r"backbone\.blocks\.(\d+)\.", n).group(1))
                        for n in got_names
                        if re.match(r"backbone\.blocks\.(\d+)\.", n)})
    results["L_cut_map"] = {
        "n_with_grad": len(got), "n_total": len(bb),
        "blocks_with_grad": block_ids,
        "first_last": [got_names[:2], got_names[-2:]]}
    # does logits_c carry grad to cls_head at all? probe the clean branch alone
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda"):
        x_adv2 = pgd_kl_attack(m, x, eps=EPS, steps=4)
        lc = m(x)
        la = m(x_adv2)
    ce2 = F.cross_entropy(lc.float(), y)
    print(f"  [L2] CE-only-in-amp-after-attack: lc.req={lc.requires_grad} "
          f"lc.grad_fn={type(lc.grad_fn).__name__}")
    ce2.backward()
    report(m, "L2_attackIN_CEonly", results)
    print("  [L2 note] if classifier grad appears here but not in H, the "
          "KL branch's interaction with the attack-built x_adv is implicated")

    import json
    with open(out_path("15_amp_cells.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

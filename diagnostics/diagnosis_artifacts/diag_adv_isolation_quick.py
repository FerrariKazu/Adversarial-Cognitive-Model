#!/usr/bin/env python3
"""
diag_adv_isolation_quick.py — lightweight §10 A/B (prototype of
diag_adv_isolation.py, tuned for the 4060's 8GB):
  A: clean CE, B: TRADES 0.55·(CE + beta*KL), PGD-4 eps 0.031
  3 epochs, batch 64, 8-seed eval skipped (top-1 on the fixed val batch),
  all with cache_enabled=False so gradients are healthy.
Pre-registered comparison: A's val curve vs B's at the SAME budget;
success criterion (pre-registered): B does NOT beat A's val accuracy
within 3 epochs under a recipe that must not collapse learning.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, save_json  # noqa: E402

from diag_clean_control import PlainHead  # noqa: E402
from training.adv_curriculum import pgd_kl_attack, trades_loss  # noqa: E402


def main() -> int:
    epochs = int(os.environ.get("DIAG_ADV_EPOCHS", "3"))
    bs = int(os.environ.get("DIAG_ADV_BS", "64"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(41)

    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=bs,
                                       num_workers=4)
    from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry

    def build():
        torch.manual_seed(41)
        m = PlainHead().to(device)
        reg = OptimizerGroupRegistry()
        reg.register_backbone(list(m.backbone.parameters()))
        reg.register("classifier", list(m.cls_head.parameters()))
        return m, reg

    def evaluate(m):
        m.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                c += (m(x.to(device)).argmax(1).cpu() == y).sum().item()
                t += y.numel()
        m.train()
        return c / max(t, 1)

    def run(arm: str):
        m, reg = build()
        opt = reg.build_optimizer(0.003, 0.9, 1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        traj, adv_dists = [], []
        for ep in range(1, epochs + 1):
            m.train()
            tot, nb = 0.0, 0
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=scaler is not None,
                                    cache_enabled=False):
                    if arm == "A_clean":
                        loss = F.cross_entropy(m(x), y)
                    else:
                        x_adv = pgd_kl_attack(m, x, eps=0.031, steps=4)
                        loss, _ = trades_loss(m, x, y, x_adv, beta=2.0)
                        loss = 0.55 * loss
                scaler.scale(loss).backward() if scaler else loss.backward()
                if scaler:
                    scaler.unscale_(opt)
                    reg.clip_grad_per_group()
                    scaler.step(opt)
                    scaler.update()
                else:
                    reg.clip_grad_per_group()
                    opt.step()
                tot += float(loss.item())
                nb += 1
                if ep == 1 and len(adv_dists) < 3:
                    adv_dists.append(float((x_adv - x).abs().mean().item())
                                     if arm == "B_adv" else None)
            sched.step()
            va = evaluate(m)
            traj.append({"epoch": ep, "loss": round(tot / nb, 4),
                         "val_acc": round(va, 4)})
            print(f"  [{arm}] epoch {ep}/{epochs} loss={traj[-1]['loss']:.4f} "
                  f"val_acc={va:.4f}", flush=True)
        return traj, adv_dists

    trajA, _ = run("A_clean")
    torch.manual_seed(41)
    trajB, dists = run("B_adv")

    out = {"section": "10_adv_isolation_probe_quick",
           "epochs": epochs, "batch_size": bs,
           "curriculum": {"eps": 0.031, "beta": 2.0, "pgd": 4, "w_trades": 0.55},
           "A_clean_trajectory": trajA, "B_adv_trajectory": trajB,
           "B_adv_visible_perturbation": {"mean_abs_xadv_minus_x_batches1_3":
                                          [round(d, 5) for d in dists]},
           "epoch1_delta": {"loss_A_minus_B": round(trajA[0]["loss"] - trajB[0]["loss"], 4),
                            "acc_A_minus_B": round(trajA[0]["val_acc"] - trajB[0]["val_acc"], 4)},
           "pre_registered_success_criterion": ("B_adv does NOT exceed A_clean's val "
                                                 "accuracy within the budget = recipe "
                                                 "collapses learning (the production-ladder "
                                                 "outcome)"),
           "verdict": ("B_Adverse_beats_A_Clean -> recipe ok" if
                       trajB[-1]["val_acc"] > trajA[-1]["val_acc"]
                       else ("B_Adv_Defers -> recipe is the dominant suspect; "
                             "matches pure-CE-era evidence") if
                       trajA[-1]["val_acc"] > 0.10 and trajA[-1]["val_acc"] >= trajB[-1]["val_acc"]
                       else "INCONCLUSIVE")}
    save_json("10_adv_isolation_probe_quick.json", out)
    print(f"SECTION 10 QUICK VERDICT: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

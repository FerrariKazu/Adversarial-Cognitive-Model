#!/usr/bin/env python3
"""
diag_clean_control.py — §3 FIRST-PRINCIPLE CLEAN-CE CONTROL (matched recipe).
================================================================================
Compact backbone + ordinary classification head + CE. NO recurrence, NO
BeliefState, NO prediction error, NO AIS, NO gaze, NO aux, NO adversarial.

MATCHED to the production stack (§3's non-negotiable list):
  dataset  data/imagenet100 @ pinned fingerprint, evaluation/imagenet100_loader
  split    ImageFolder train/ val/ (the production split)
  xform    RandomResizedCrop(96)+flip / Resize-256-then-CenterCrop(96) + norm
  optim    OptimizerGroupRegistry -> SGD(momentum=0.9, wd=1e-4)
  sched    CosineAnnealingLR(T_max=epochs), stepped per epoch
  eval     evaluate_val semantics: full val pass, argmax top-1
  seed     41

The backbone is the REAL production trunk: CompactViT(img_size=96) (d_z=384,
12 blocks) — the §3 list says "compact backbone + ordinary classification
head" on the same infrastructure; the trunk is shared with the substrate so
any trunk-level defect shows up here too.

EPOCHS is an env knob (default 12): the pure-CE production log shows the
healthy epoch-1 signature (loss ~4.18 -> val ~0.108) and 0.21 by epoch 10 —
12 epochs is enough to distinguish 'learns normally' from 'broken' with no
ambiguity. This is a CONTROL, not a leaderboard entry.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, IMG_SIZE, REPO_ROOT, save_json  # noqa: E402


class PlainHead(nn.Module):
    """The §3 control model: real CompactViT trunk + ONE linear head.

    Input adaptation (documented): the production trunk's native operating
    point is 56 px (fovea = 4x14 patches); the loader emits 96 px frames.
    The control downsamples the full frame 96 -> 56 (bilinear) so the SAME
    trunk module consumes whole images. Information content differs from
    the foveal path by design — the §3 question is 'can the matched stack
    learn', not 'is information content matched' (the pure-CE backbone_only
    run is the information-matched reference).
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        from noesis_vision.models.backbone import CompactViT
        self.backbone = CompactViT(img_size=56)          # trunk native size
        self.cls_head = nn.Linear(self.backbone.d_z, num_classes)

    def forward(self, x):                       # (B,3,96,96) -> (B,C)
        if x.shape[-1] != 56:
            x = F.interpolate(x, size=(56, 56), mode="bilinear",
                              align_corners=False)
        tokens = self.backbone._prep(x)
        feats = self.backbone._trunk_forward(tokens)[:, 0]
        return self.cls_head(feats)


def main() -> int:
    epochs = int(os.environ.get("DIAG_EPOCHS", "12"))
    bs = int(os.environ.get("DIAG_BS", "96"))
    workers = int(os.environ.get("DIAG_WORKERS", "6"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(41)

    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=bs,
                                       num_workers=workers)

    from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
    model = PlainHead().to(device)
    reg = OptimizerGroupRegistry()
    reg.register_backbone(list(model.backbone.parameters()))
    reg.register("classifier", list(model.cls_head.parameters()))
    opt = reg.build_optimizer(0.003, 0.9, 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    rows = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler is not None):
                loss = F.cross_entropy(model(x), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                reg.clip_grad_per_group()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                reg.clip_grad_per_group()
                opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        # evaluate_val semantics (full val, argmax)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loaders["val"]:
                p = model(x.to(device)).argmax(1).cpu()
                correct += (p == y).sum().item()
                total += y.numel()
        acc = correct / max(total, 1)
        rows.append({"epoch": ep, "loss": round(tot / nb, 4), "val_acc": round(acc, 4)})
        print(f"  [clean-control] epoch {ep}/{epochs} "
              f"loss={tot / nb:.4f} val_acc={acc:.4f}", flush=True)

    out = {"section": "3_clean_ce_control", "epochs": epochs,
           "batch_size": bs, "device": str(device),
           "wall_minutes": round((time.time() - t0) / 60, 1),
           "trajectory": rows,
           "epoch1_signature_vs_pure_ce_run": {
               "pure_ce_run_epoch1": {"loss": 4.1757, "val_acc": 0.1080},
               "control_epoch1": rows[0] if rows else None},
           "verdict_inputs": {
               "loss_decreased": len(rows) > 1 and rows[-1]["loss"] < rows[0]["loss"] - 0.3,
               "val_acc_epoch3_above_0p12": rows[2]["val_acc"] > 0.12 if len(rows) >= 3 else False,
               "final_val_acc": rows[-1]["val_acc"] if rows else None}}
    save_json("03_clean_ce_control.json", out)
    ok = out["verdict_inputs"]["loss_decreased"] and \
        out["verdict_inputs"]["val_acc_epoch3_above_0p12"]
    print(f"SECTION 3 VERDICT: {'PASS — infrastructure learns normally' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

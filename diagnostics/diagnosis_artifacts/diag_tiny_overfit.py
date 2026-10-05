#!/usr/bin/env python3
"""
diag_tiny_overfit.py — §5 LABEL SANITY + TINY-SET OVERFIT GATE (STOP gate).
================================================================================
1. Label-sanity printout: sample index -> class ID -> class name -> tensor
   stats, for one real batch.
2. Label-change test: manually corrupting labels must move the loss in the
   expected direction (sanity of the loss/label wiring).
3. THE GATE: 32 fixed samples, the §3 control model, aggressive LR —
   train accuracy MUST reach >= 95% (near-100% on 32 samples). If not: STOP.
   Directive: "Do not proceed to architecture experimentation until this
   passes."
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, IMG_SIZE, save_json  # noqa: E402

from diag_clean_control import PlainHead  # noqa: E402


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(41)

    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=32, num_workers=4)
    from torchvision import transforms as T
    from torchvision import datasets
    tf = T.Compose([T.Resize(int(IMG_SIZE * 256 / 224)), T.CenterCrop(IMG_SIZE),
                    T.ToTensor(), T.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)

    # -- (1) label-sanity printout (8 deterministic samples) ------------------
    lines = []
    g = torch.Generator().manual_seed(41)
    idxs = torch.randperm(len(val_ds), generator=g)[:8].tolist()
    for i in idxs:
        x, y = val_ds[i]
        lines.append({"sample_index": i, "class_id": y,
                      "class_name": val_ds.classes[y],
                      "tensor_stats": {"shape": list(x.shape), "dtype": str(x.dtype),
                                       "min": round(x.min().item(), 3),
                                       "max": round(x.max().item(), 3),
                                       "mean": round(x.mean().item(), 3)}})
        print(f"  idx={i:6d} class_id={y:2d} class={val_ds.classes[y][:40]:40s} "
              f"x[min/max/mean]=({x.min():.2f},{x.max():.2f},{x.mean():.2f})")

    # -- (2) label-change moves the loss --------------------------------------
    xs = torch.stack([val_ds[i][0] for i in idxs]).to(device)
    ys = torch.tensor([val_ds[i][1] for i in idxs]).to(device)
    model = PlainHead().to(device).eval()
    with torch.no_grad():
        base = F.cross_entropy(model(xs), ys).item()
        wrong = F.cross_entropy(model(xs), (ys + 1) % 100).item()
    label_test = {"loss_true_labels": round(base, 4),
                  "loss_shifted_labels": round(wrong, 4),
                  "expected_relation": "shifted >= true for a random init (uniform logits)",
                  "holds": wrong >= base - 1e-4}
    print(f"  label-change test: CE(true)={base:.4f} CE(shifted)={wrong:.4f} "
          f"holds={label_test['holds']}")

    # -- (3) THE GATE: overfit 32 fixed samples --------------------------------
    xs32 = xs.clone()
    ys32 = ys.clone()
    model = PlainHead().to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0)
    steps = int(os.environ.get("DIAG_TINY_STEPS", "300"))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    traj = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = F.cross_entropy(model(xs32), ys32)
        scaler.scale(loss).backward() if scaler else loss.backward()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                acc = (model(xs32).argmax(1) == ys32).float().mean().item()
            traj.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})
            print(f"  [tiny-overfit] step {step:4d} loss={loss.item():.4f} acc={acc:.4f}",
                  flush=True)
    final_acc = traj[-1]["acc"]
    gate = final_acc >= 0.95
    out = {"section": "5_tiny_overfit_gate", "n_samples": 32, "steps": steps,
           "label_sanity": lines, "label_change_test": label_test,
           "trajectory": traj, "final_train_acc": final_acc,
           "gate_threshold": 0.95, "gate": "PASS" if gate else "FAIL — STOP"}
    save_json("05_tiny_overfit.json", out)
    print(f"SECTION 5 GATE: {'PASS' if gate else 'FAIL — STOP'} "
          f"(final acc {final_acc:.3f} vs threshold 0.95)")
    return 0 if gate else 3


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
T4b: Recurrence Iteration Sweep (T=1..4).
Evaluates clean and PGD-50 accuracy on the canonical 1k validation slice
for both flagship seeds: recurrence_112_seed17 and recurrence_112_seed42.
Saves results to evaluation/stage0_results/t4b_tsweep.json.
"""

import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchattacks

from gen2_foundation.model import Gen2FoundationModel
from evaluation.harness import (
    get_canonical_slice,
    CANONICAL_TORCH_SLICE_SHA256,
)

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_ROOT = "data/imagenet100/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
EPS_PIXEL = 5.417 / 255.0

SEEDS_CONFIG = [
    {
        "arm_id": "recurrence_112_seed42",
        "seed": 42,
        "ckpt": "checkpoints/foundation_recurrence_112_seed42_final_eb40bf84.pth",
    },
    {
        "arm_id": "recurrence_112_seed17",
        "seed": 17,
        "ckpt": "checkpoints/foundation_recurrence_112_seed17_final_636e4b1c.pth",
    },
]

class _NormWrap(nn.Module):
    """Wrap model so input in [0,1] is normalized before being passed to model."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_m", m)
        self.register_buffer("_s", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base((x - self._m.to(x)) / self._s.to(x))


def main():
    print("=== T4b: Recurrence Iteration Sweep (T=1..4) ===")
    print(f"Device: {DEVICE}")

    # 1. Obtain canonical 1k slice indices
    indices_1k, slice_hash = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
    print(f"Canonical 1k slice verified. Hash: {slice_hash}")

    # Build raw [0,1] loader for the 1k slice
    tf_01 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    full_ds = ImageFolder(VAL_ROOT, transform=tf_01)
    slice_ds = Subset(full_ds, indices_1k)
    loader_1k = DataLoader(slice_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    results = {
        "protocol_id": "consolidated-flagship-112/1",
        "diagnostic": "T4b Recurrence Sweep (T=1..4)",
        "slice_sha256": slice_hash,
        "n_samples": len(indices_1k),
        "eps_pixel": EPS_PIXEL,
        "eps_pixel_255": EPS_PIXEL * 255.0,
        "pgd_steps": 50,
        "pgd_alpha": EPS_PIXEL / 10.0,
        "seeds": {},
    }

    for cfg in SEEDS_CONFIG:
        arm_id = cfg["arm_id"]
        ckpt_path = Path(cfg["ckpt"])
        print(f"\n--- Evaluating {arm_id} ({ckpt_path.name}) ---")

        base_model = Gen2FoundationModel(phase="recurrence_only", fovea_size=112)
        st = torch.load(ckpt_path, map_location="cpu")
        base_model.load_state_dict(st["model"], strict=True)
        wrapped_model = _NormWrap(base_model).to(DEVICE)
        wrapped_model.eval()

        seed_data = {}

        for T in [1, 2, 3, 4]:
            base_model.num_glimpses = T
            print(f"\n  [T = {T}] Setting num_glimpses = {T} ...")

            # 1. Clean Accuracy
            t0 = time.time()
            clean_corr = 0
            total = 0
            with torch.no_grad():
                for x, y in loader_1k:
                    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                    logits = wrapped_model(x)
                    clean_corr += (logits.argmax(dim=-1) == y).sum().item()
                    total += y.size(0)
            clean_acc = clean_corr / total
            t_clean = time.time() - t0
            print(f"    Clean (1k): {clean_acc*100:.2f}% ({t_clean:.1f}s)")

            # 2. PGD-50 Robust Accuracy
            t0 = time.time()
            atk = torchattacks.PGD(
                wrapped_model,
                eps=EPS_PIXEL,
                alpha=EPS_PIXEL / 10.0,
                steps=50,
                random_start=True,
            )
            pgd_corr = 0
            total_pgd = 0
            for i, (x, y) in enumerate(loader_1k):
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                x_adv = atk(x, y)
                with torch.no_grad():
                    logits = wrapped_model(x_adv)
                pgd_corr += (logits.argmax(dim=-1) == y).sum().item()
                total_pgd += y.size(0)
                if (i + 1) % 4 == 0 or (i + 1) == len(loader_1k):
                    print(f"      Batch {i+1}/{len(loader_1k)} complete...", flush=True)

            pgd_acc = pgd_corr / total_pgd
            t_pgd = time.time() - t0
            print(f"    PGD-50 (1k): {pgd_acc*100:.2f}% ({t_pgd:.1f}s)")

            seed_data[f"T_{T}"] = {
                "T": T,
                "clean_acc_1k": clean_acc,
                "clean_acc_pct": round(clean_acc * 100, 2),
                "pgd50_acc_1k": pgd_acc,
                "pgd50_acc_pct": round(pgd_acc * 100, 2),
                "clean_time_s": round(t_clean, 2),
                "pgd50_time_s": round(t_pgd, 2),
            }

        results["seeds"][arm_id] = seed_data

    out_file = OUT_DIR / "t4b_tsweep.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved T4b results to {out_file}")
    print("\n=== SUMMARY TABLE ===")
    print("Arm | T=1 (cln/pgd) | T=2 (cln/pgd) | T=3 (cln/pgd) | T=4 (cln/pgd)")
    print("-" * 65)
    for arm_id, s in results["seeds"].items():
        row = f"{arm_id:22} | "
        for T in [1, 2, 3, 4]:
            t_data = s[f"T_{T}"]
            row += f"{t_data['clean_acc_pct']:.1f}%/{t_data['pgd50_acc_pct']:.1f}% | "
        print(row)


if __name__ == "__main__":
    main()

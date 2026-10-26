#!/usr/bin/env python3
"""G2-D7: Gaze Policy Swap Evaluation
======================================
Evaluates gen1_core (which has AIS-v2 gaze) under three gaze regimes
on the identical fixed 500-image test slice:

  (A) AIS_V2    — learned attention-informed gaze (model as trained)
  (B) FIXED_4   — fixed 4-point schedule (same as recurrence_only training)
  (C) RANDOM    — random uniform gaze at each step

For gen1_core, we can swap the gaze policy by monkey-patching use_ais_v2 flag.
For FIXED and RANDOM modes, we override the gaze selection in forward.

Protocol:
  - 500 images, seed=42
  - Clean + PGD-10 @ eps=0.094
  - Also run all three on recurrence_only_final for a baseline comparison

Registration: G2-D7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE, foveal_sample
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from training.train_gen2_foundation import run_pgd_ce_eval

SEED = 42
SAMPLE_SIZE = 500
BATCH_SIZE = 32
EPS = 0.094
PGD_STEPS = 10

FIXED_SCHEDULE = [
    (0.0, 0.0),
    (0.3, 0.3),
    (-0.3, 0.3),
    (0.0, -0.3),
]


class GazePatchedModel(nn.Module):
    """Wraps Gen2FoundationModel with a gaze mode override for evaluation.

    gaze_mode: 'ais_v2' | 'fixed' | 'random'
    """

    def __init__(self, base_model: Gen2FoundationModel, gaze_mode: str = "ais_v2"):
        super().__init__()
        self.base = base_model
        self.gaze_mode = gaze_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gaze_mode == "ais_v2" and self.base.use_ais_v2:
            # Use normal forward — AIS-v2 active
            return self.base(x)

        # Patched forward: replicate the forward but with fixed/random gaze
        base = self.base
        B = x.shape[0]
        device = x.device

        # Gist init
        gist_tokens = base.gist_encoder(x)
        z_t = gist_tokens.mean(dim=1)

        if self.gaze_mode == "fixed":
            def get_gaze(t, B, device):
                if t < len(FIXED_SCHEDULE):
                    gx, gy = FIXED_SCHEDULE[t]
                    return torch.tensor([[gx, gy]], device=device).expand(B, -1)
                return torch.zeros(B, 2, device=device)
        elif self.gaze_mode == "random":
            def get_gaze(t, B, device):
                return torch.empty(B, 2, device=device).uniform_(-0.8, 0.8)
        else:
            raise ValueError(f"Unknown gaze_mode: {self.gaze_mode}")

        if not base.carry_belief:
            # recurrence_only logic
            for t in range(base.num_glimpses):
                current_gaze = get_gaze(t, B, device)
                crop = foveal_sample(x, current_gaze, fovea_size=base.fovea_size)
                pooled_g, _ = base.backbone(crop)
                z_t = 0.5 * z_t + 0.5 * pooled_g
            return base.cls_head(z_t)

        # Belief phases with patched gaze
        dirichlet = base.evidential_head(z_t)
        current_gaze = torch.zeros(B, 2, device=device)
        prev_tokens = None
        pred_feat = None

        for t in range(base.num_glimpses):
            current_gaze = get_gaze(t, B, device)

            crop = foveal_sample(x, current_gaze, fovea_size=base.fovea_size)
            pooled_g, tokens_g = base.backbone(crop)
            prev_tokens = tokens_g

            if base.belief_dynamics:
                prec = base.precision(dirichlet).unsqueeze(-1)
                belief_obj = SimpleNamespace(
                    z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
                )
                pred_feat = base.predictor.predict_features(belief_obj, current_gaze)
                if base.arm == "v2" and base.update_net_v2 is not None:
                    err_map = (pred_feat - tokens_g).norm(dim=-1).view(B, 4, 4)
                    delta_z = base.update_net_v2(z_t, err_map, observed=pooled_g, U=dirichlet.uncertainty)
                else:
                    err = pred_feat - tokens_g
                    delta_z = base.update_net(z_t, err)
                z_t = z_t + prec * delta_z
            else:
                z_t = 0.5 * z_t + 0.5 * pooled_g

            dirichlet = base.evidential_head(z_t)

        feats = torch.cat([z_t, dirichlet.evidence], dim=-1)
        return base.cls_head(feats) + base.ev_readout(dirichlet.evidence)


def get_loader(data_root: str, seed: int, sample_size: int, batch_size: int) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_ds), generator=g)[:sample_size].tolist()
    return DataLoader(Subset(val_ds, indices), batch_size=batch_size, shuffle=False, num_workers=2)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             eps: float = 0.0, pgd_steps: int = 0) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if eps > 0.0 and pgd_steps > 0:
            x = run_pgd_ce_eval(model, x, y, eps=eps, steps=pgd_steps)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return float(correct) / float(max(total, 1))


def main():
    parser = argparse.ArgumentParser(description="G2-D7 Gaze Swap Evaluation")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--output-dir", type=str, default="report/d7_gaze_swap")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-D7] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    loader = get_loader(args.data_root, SEED, SAMPLE_SIZE, BATCH_SIZE)

    all_results = {}
    checkpoints = [
        {"name": "recurrence_only_final", "path": "checkpoints/foundation_recurrence_only_final.pth",
         "phase": "recurrence_only", "arm": "v1", "gaze_modes": ["fixed", "random"]},
        {"name": "gen1_core_best_clean", "path": "checkpoints/foundation_gen1_core_best_clean.pth",
         "phase": "gen1_core", "arm": "v1", "gaze_modes": ["ais_v2", "fixed", "random"]},
    ]

    for ckpt_info in checkpoints:
        name = ckpt_info["name"]
        ckpt_path = ckpt_info["path"]

        if not os.path.exists(ckpt_path):
            print(f"[G2-D7] BLOCKER: {ckpt_path} not found — skipping {name}")
            all_results[name] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        ckpt_size = os.path.getsize(ckpt_path)
        print(f"\n{'='*70}")
        print(f"[G2-D7] Evaluating: {name} ({ckpt_size:,} bytes)")

        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved

        ckpt_results = {"phase": ckpt_info["phase"], "gaze_modes": {}}

        for gaze_mode in ckpt_info["gaze_modes"]:
            print(f"\n  --- Gaze mode: {gaze_mode.upper()} ---")
            base_model = Gen2FoundationModel(phase=ckpt_info["phase"], arm=ckpt_info["arm"]).to(device)
            base_model.load_state_dict(sd, strict=False)
            base_model.eval()

            model = GazePatchedModel(base_model, gaze_mode=gaze_mode).to(device)
            model.eval()

            t0 = time.time()
            clean = evaluate(model, loader, device, eps=0.0, pgd_steps=0)
            t_clean = time.time() - t0
            print(f"    Clean: {clean*100:.2f}% ({t_clean:.1f}s)")

            t0 = time.time()
            robust = evaluate(model, loader, device, eps=EPS, pgd_steps=PGD_STEPS)
            t_robust = time.time() - t0
            print(f"    PGD-{PGD_STEPS}@{EPS}: {robust*100:.2f}% ({t_robust:.1f}s)")

            ckpt_results["gaze_modes"][gaze_mode] = {
                "clean": clean,
                "robust": robust,
                "runtime_clean_s": t_clean,
                "runtime_robust_s": t_robust,
            }

            del model, base_model
            torch.cuda.empty_cache()

        all_results[name] = ckpt_results

        out_path = os.path.join(args.output_dir, f"g2_d7_{name}.json")
        with open(out_path, "w") as f:
            json.dump(ckpt_results, f, indent=2)
        print(f"\n  ✓ Saved: {out_path}")

    # Summary
    print("\n" + "=" * 80)
    print("G2-D7 GAZE SWAP SUMMARY")
    print("=" * 80)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: {res['error']}")
            continue
        print(f"\n{name}:")
        print(f"  {'Mode':<12} | {'Clean':>8} | {'PGD-10':>8}")
        print(f"  {'-'*35}")
        for mode, m in res["gaze_modes"].items():
            print(f"  {mode:<12} | {m['clean']*100:>7.2f}% | {m['robust']*100:>7.2f}%")

    combined_path = os.path.join(args.output_dir, "g2_d7_combined.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Combined: {combined_path}")


if __name__ == "__main__":
    main()

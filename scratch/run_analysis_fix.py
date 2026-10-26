#!/usr/bin/env python3
"""
Diagnostic & Analysis Fix:
1. Per-sample evaluation & McNemar + Paired Bootstrap across:
   - recurrence_only vs C1 (g2_gist_only_cont)
   - recurrence_only vs ff_112
   - C1 vs ff_112
2. Shift-averaging TTA control on C1 (testing whether recurrence is 4-view translation TTA).
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from scipy.stats import chi2
import torchattacks

sys.path.insert(0, ".")
from gen2_foundation.model import Gen2FoundationModel, foveal_sample
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

DATA_ROOT = "data/imagenet100"
EVAL_SEED = 42
EVAL_N = 1000
EVAL_BATCH = 32
EPS_PIXEL = 5.417 / 255.0


class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.base_model((x - self.mean) / self.std)


class C1_TTA_Model(nn.Module):
    """
    Evaluates C1 (g2_gist_only_cont) with 4-shift translation averaging
    matching recurrence_only's fixed gaze schedule:
      g0 = (0, 0)
      g1 = (0.3, 0.3)
      g2 = (-0.3, 0.3)
      g3 = (0, -0.3)
    """
    def __init__(self, c1_base_model, mode="recurrence_weighted"):
        super().__init__()
        self.backbone = c1_base_model.backbone
        self.cls_head = c1_base_model.cls_head
        self.gist_encoder = c1_base_model.gist_encoder
        self.fovea_size = 56
        self.mode = mode

    def forward(self, x):
        B = x.shape[0]
        dev = x.device
        gaze_schedule = [
            torch.zeros(B, 2, device=dev),
            torch.tensor([[0.3, 0.3]], device=dev).expand(B, -1),
            torch.tensor([[-0.3, 0.3]], device=dev).expand(B, -1),
            torch.tensor([[0.0, -0.3]], device=dev).expand(B, -1),
        ]

        if self.mode == "recurrence_weighted":
            gist_tokens = self.gist_encoder(x)
            z_t = gist_tokens.mean(dim=1)
            for g in gaze_schedule:
                crop = foveal_sample(x, g, fovea_size=self.fovea_size)
                pooled_g, _ = self.backbone(crop)
                z_t = 0.5 * z_t + 0.5 * pooled_g
            return self.cls_head(z_t)
        else:
            # Uniform 4-crop logit averaging
            logits_list = []
            for g in gaze_schedule:
                crop = foveal_sample(x, g, fovea_size=self.fovea_size)
                x_prep = self.backbone._prep(crop)
                x_trunk = self.backbone._trunk_forward(x_prep)
                pooled = x_trunk[:, 0]
                logits_list.append(self.cls_head(pooled))
            return torch.stack(logits_list, dim=0).mean(dim=0)


def build_raw_loader(img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
    g = torch.Generator().manual_seed(EVAL_SEED)
    indices = torch.randperm(len(val_ds), generator=g)[:EVAL_N].tolist()
    sub_ds = Subset(val_ds, indices)
    return DataLoader(sub_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)


def run_pgd_pixel(wrapped_model, x, y, eps=EPS_PIXEL, steps=50):
    alpha = eps / 4.0
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = wrapped_model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()
    return x_adv


def mcnemar_test(corr_a, corr_b):
    # a correct, b wrong
    b = int(np.sum((corr_a == 1) & (corr_b == 0)))
    # a wrong, b correct
    c = int(np.sum((corr_a == 0) & (corr_b == 1)))
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "stat": 0.0, "p_value": 1.0}
    stat = ((abs(b - c) - 1.0) ** 2) / n_discordant
    p_val = float(chi2.sf(stat, df=1))
    return {"b": b, "c": c, "stat": float(stat), "p_value": p_val}


def paired_bootstrap(corr_a, corr_b, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = corr_a.astype(float) - corr_b.astype(float)
    observed_diff = float(np.mean(diffs))
    indices = rng.randint(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = np.mean(diffs[indices], axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    se = float(np.std(boot_means))
    return {
        "observed_diff_pp": observed_diff * 100.0,
        "ci95_low_pp": ci_low * 100.0,
        "ci95_high_pp": ci_high * 100.0,
        "bootstrap_se_pp": se * 100.0,
    }


def evaluate_model_battery(wrapped_model, loader):
    apgd_ce = torchattacks.APGD(wrapped_model, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    results = {}
    for atk in ["clean", "pgd50", "apgd_ce"]:
        preds_list, labels_list = [], []
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if atk == "clean":
                x_adv = x
            elif atk == "pgd50":
                x_adv = run_pgd_pixel(wrapped_model, x, y, eps=EPS_PIXEL, steps=50)
            elif atk == "apgd_ce":
                x_adv = apgd_ce(x, y)
            with torch.no_grad():
                preds = wrapped_model(x_adv).argmax(-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(y.cpu().numpy().tolist())
        corr = (np.array(preds_list) == np.array(labels_list)).astype(np.int8)
        acc = float(np.mean(corr) * 100.0)
        print(f"    {atk:<10} Acc: {acc:.2f}% ({time.time()-t0:.1f}s)")
        results[atk] = corr

    worst = np.bitwise_and.reduce([results["pgd50"], results["apgd_ce"]])
    print(f"    worst_case Acc: {float(np.mean(worst)*100.0):.2f}%")
    results["worst_case"] = worst
    return results


def main():
    print("=" * 80)
    print("STEP 1: PER-SAMPLE EVALUATION & PAIRED COMPARISON")
    print("=" * 80)

    # 1. Models to load
    models_config = {
        "ff_112": {
            "phase": "ff_112",
            "img_size": 112,
            "ckpt": "checkpoints/foundation_ff_112_final.pth",
        },
        "recurrence_only": {
            "phase": "recurrence_only",
            "img_size": 56,
            "ckpt": "checkpoints/foundation_recurrence_only_final.pth",
        },
        "c1": {
            "phase": "g2_gist_only",
            "img_size": 56,
            "ckpt": "checkpoints/foundation_g2_gist_only_cont_final.pth",
        },
    }

    per_sample_outcomes = {}

    for name, cfg in models_config.items():
        print(f"\n--- Evaluating {name} ({cfg['img_size']}px) ---")
        base = Gen2FoundationModel(phase=cfg["phase"], arm="v1", fovea_size=cfg["img_size"]).to(device)
        sd = torch.load(cfg["ckpt"], map_location=device)
        base.load_state_dict(sd.get("model", sd), strict=False)
        base.eval()
        wrapped = NormalizedModel(base).to(device)
        wrapped.eval()
        loader = build_raw_loader(cfg["img_size"])
        per_sample_outcomes[name] = evaluate_model_battery(wrapped, loader)

    # 2. Evaluate C1 + TTA (4-shift translation averaging)
    print("\n--- Evaluating C1 + 4-shift TTA (translation averaging) (56px) ---")
    base_c1 = Gen2FoundationModel(phase="g2_gist_only", arm="v1", fovea_size=56).to(device)
    sd_c1 = torch.load("checkpoints/foundation_g2_gist_only_cont_final.pth", map_location=device)
    base_c1.load_state_dict(sd_c1.get("model", sd_c1), strict=False)
    base_c1.eval()

    c1_tta_base = C1_TTA_Model(base_c1, mode="recurrence_weighted").to(device)
    c1_tta_wrapped = NormalizedModel(c1_tta_base).to(device)
    c1_tta_wrapped.eval()
    loader_56 = build_raw_loader(56)
    per_sample_outcomes["c1_tta"] = evaluate_model_battery(c1_tta_wrapped, loader_56)

    # 3. Statistical Comparisons
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL ANALYSIS (McNemar & Paired Bootstrap)")
    print("=" * 80)

    pairs = [
        ("recurrence_only", "c1", "Recurrence vs Matched Feedforward C1 (56px)"),
        ("recurrence_only", "ff_112", "Recurrence (56px) vs Feedforward 112px"),
        ("c1", "ff_112", "Feedforward 56px (C1) vs Feedforward 112px"),
        ("recurrence_only", "c1_tta", "Recurrence vs C1 + 4-Shift TTA (Is recurrence just TTA?)"),
        ("c1_tta", "c1", "C1 + 4-Shift TTA vs C1 Baseline (Effect of pure shift-averaging)"),
    ]

    summary_stats = {}

    for name_a, name_b, desc in pairs:
        print(f"\n>>> {desc} ({name_a} vs {name_b})")
        print(f"  {'Metric':<12} {'Acc A':>8} {'Acc B':>8} {'Diff (A-B)':>11} {'95% CI':>18} {'McNemar p':>12}")
        print("  " + "-" * 75)
        pair_key = f"{name_a}_vs_{name_b}"
        summary_stats[pair_key] = {}

        for metric in ["clean", "pgd50", "apgd_ce", "worst_case"]:
            ca = per_sample_outcomes[name_a][metric]
            cb = per_sample_outcomes[name_b][metric]
            acc_a = float(np.mean(ca) * 100.0)
            acc_b = float(np.mean(cb) * 100.0)

            boot = paired_bootstrap(ca, cb)
            mcn = mcnemar_test(ca, cb)

            ci_str = f"[{boot['ci95_low_pp']:+.2f}, {boot['ci95_high_pp']:+.2f}]"
            p_str = f"{mcn['p_value']:.4e}" if mcn['p_value'] < 1e-3 else f"{mcn['p_value']:.4f}"

            print(f"  {metric:<12} {acc_a:>7.2f}% {acc_b:>7.2f}% {boot['observed_diff_pp']:>+10.2f}% {ci_str:>18} {p_str:>12}")
            summary_stats[pair_key][metric] = {
                "acc_a": acc_a,
                "acc_b": acc_b,
                "diff_pp": boot["observed_diff_pp"],
                "ci95": [boot["ci95_low_pp"], boot["ci95_high_pp"]],
                "bootstrap_se_pp": boot["bootstrap_se_pp"],
                "mcnemar_b": mcn["b"],
                "mcnemar_c": mcn["c"],
                "mcnemar_stat": mcn["stat"],
                "mcnemar_p": mcn["p_value"],
            }

    # Save artifact
    out_file = "report/audit_rev1/paired_analysis_and_tta.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\nSaved analysis results to {out_file}")


if __name__ == "__main__":
    main()

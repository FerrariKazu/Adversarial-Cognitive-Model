#!/usr/bin/env python3
"""
T4a Crop Check:
1. Extract 4 glimpse crops for 8 validation images.
2. Save crops as PNG images in evaluation/stage0_results/t4a_crops/.
3. Test whether the four crops differ in pixel space.
4. Pass through backbone to extract pooled_g and compute cross-glimpse cosine similarities.
5. Save results to evaluation/stage0_results/t4a_crop_check.json.
"""

import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

from noesis_vision.models.foveation import foveal_sample
from gen2_foundation.model import Gen2FoundationModel

OUT_DIR = Path("evaluation/stage0_results")
CROPS_DIR = OUT_DIR / "t4a_crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

VAL_ROOT = Path("data/imagenet100/val")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def main():
    print("=== T4a: Glimpse Crop Diagnostic ===")

    # 1. Load model with seed42 weights
    ckpt_path = Path("checkpoints/foundation_recurrence_112_seed42_final_eb40bf84.pth")
    assert ckpt_path.exists(), f"Missing checkpoint {ckpt_path}"
    
    model = Gen2FoundationModel(phase="recurrence_only", fovea_size=112)
    st = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(st["model"], strict=True)
    model.to(DEVICE)
    model.eval()

    # 2. Raw [0,1] transform for saving PNGs + normalized transform for model backbone
    tf_01 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    
    tf_norm = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds_01 = ImageFolder(VAL_ROOT, transform=tf_01)
    ds_norm = ImageFolder(VAL_ROOT, transform=tf_norm)

    # Pick 8 distinct images across classes
    indices = [0, 50, 100, 150, 200, 250, 300, 350]

    gaze_schedule = [
        torch.zeros(1, 2, device=DEVICE),
        torch.tensor([[0.3, 0.3]], device=DEVICE),
        torch.tensor([[-0.3, 0.3]], device=DEVICE),
        torch.tensor([[0.0, -0.3]], device=DEVICE),
    ]

    all_image_diagnostics = []

    for img_idx, ds_idx in enumerate(indices):
        x_raw, y = ds_01[ds_idx]
        x_norm, _ = ds_norm[ds_idx]
        cls_name = ds_01.classes[y]

        x_raw_b = x_raw.unsqueeze(0).to(DEVICE)
        x_norm_b = x_norm.unsqueeze(0).to(DEVICE)

        crops_raw = []
        crops_norm = []
        pooled_gs = []

        # Extract 4 crops
        for t in range(4):
            gaze = gaze_schedule[t]
            c_raw = foveal_sample(x_raw_b, gaze, fovea_size=112)
            c_norm = foveal_sample(x_norm_b, gaze, fovea_size=112)

            crops_raw.append(c_raw)
            crops_norm.append(c_norm)

            # Save PNG
            c_np = (c_raw[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_png = Image.fromarray(c_np)
            png_path = CROPS_DIR / f"img_{img_idx}_glimpse_{t}.png"
            img_png.save(png_path)

            # Forward through backbone to get pooled_g
            with torch.no_grad():
                p_g, _ = model.backbone(c_norm)
                pooled_gs.append(p_g[0])

        # Compute pairwise pixel differences
        pixel_diffs = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                pixel_diffs[i, j] = float((crops_raw[i] - crops_raw[j]).abs().mean().item())

        # Compute pairwise cosine similarity of pooled_g
        cos_sims = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                sim = F.cosine_similarity(pooled_gs[i].unsqueeze(0), pooled_gs[j].unsqueeze(0)).item()
                cos_sims[i, j] = float(sim)

        differ_from_g0 = [bool(not torch.allclose(crops_raw[t], crops_raw[0], atol=1e-4)) for t in range(4)]

        img_report = {
            "img_index": img_idx,
            "dataset_index": ds_idx,
            "class_name": cls_name,
            "crops_differ_from_glimpse_0": differ_from_g0,
            "mean_pixel_diff_vs_g0": [float(pixel_diffs[0, t]) for t in range(4)],
            "cosine_similarity_matrix": cos_sims.tolist(),
            "mean_off_diagonal_cos_sim": float((cos_sims.sum() - 4.0) / 12.0),
        }
        all_image_diagnostics.append(img_report)

        print(f"Image {img_idx} ({cls_name}):")
        print(f"  Crops differ from Glimpse 0: {differ_from_g0}")
        print(f"  Mean pixel diff vs G0: {[round(x, 4) for x in img_report['mean_pixel_diff_vs_g0']]}")
        print(f"  Mean off-diagonal cosine sim: {img_report['mean_off_diagonal_cos_sim']:.4f}")

    # Summary
    all_differ = all(all(d["crops_differ_from_glimpse_0"][1:]) for d in all_image_diagnostics)
    overall_mean_cos_sim = float(np.mean([d["mean_off_diagonal_cos_sim"] for d in all_image_diagnostics]))
    overall_mean_pixel_diff = float(np.mean([np.mean(d["mean_pixel_diff_vs_g0"][1:]) for d in all_image_diagnostics]))

    summary = {
        "protocol_id": "consolidated-flagship-112/1",
        "diagnostic": "T4a Crop Check",
        "n_images": 8,
        "n_glimpses": 4,
        "fovea_size": 112,
        "image_size": 112,
        "gaze_schedule": [[0.0, 0.0], [0.3, 0.3], [-0.3, 0.3], [0.0, -0.3]],
        "all_glimpses_differ_in_pixels": bool(all_differ),
        "overall_mean_pixel_diff_vs_center": overall_mean_pixel_diff,
        "overall_mean_cross_glimpse_cos_sim": overall_mean_cos_sim,
        "interpretation": (
            "Because fovea_size == image_size (112px), non-zero gaze coordinates shift the sampling grid, "
            "sampling with border padding / edge clamping. The four views are NOT identical copies in pixel space, "
            "but because the shifts sample the same central field with shifted margins, feature cosine similarities are high."
        ),
        "per_image": all_image_diagnostics,
    }

    out_file = OUT_DIR / "t4a_crop_check.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nT4a Complete. Saved {len(indices)*4} PNG crops to {CROPS_DIR}/")
    print(f"Saved diagnostic JSON to {out_file}")
    print(f"All glimpses differ: {all_differ}")
    print(f"Overall cross-glimpse cosine similarity: {overall_mean_cos_sim:.4f}")

if __name__ == "__main__":
    main()

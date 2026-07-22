#!/usr/bin/env python3
"""
Sprint 2 Phase B: CLIP Quality Gate & Diversity Filtering
=========================================================
1. Quality Gate: Filters raw synthetic images keeping CLIP cosine sim > 0.25
2. Diversity Check: Subsamples 1,000 images per class and checks pairwise similarity.
   Flags classes with mean pairwise similarity > 0.75 as too homogeneous.

Usage:
  python3 data_generation/filter_synthetic_clip.py \
    --input-dir ./data/synthetic_stl10_raw \
    --output-dir ./data/synthetic_stl10_filtered
"""

import os
import sys
import json
import tarfile
import io
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]

CLASS_PROMPTS = {
    "airplane": "a photo of an airplane",
    "bird": "a photo of a bird",
    "car": "a photo of a car",
    "cat": "a photo of a cat",
    "deer": "a photo of a deer",
    "dog": "a photo of a dog",
    "horse": "a photo of a horse",
    "monkey": "a photo of a monkey",
    "ship": "a photo of a ship",
    "truck": "a photo of a truck",
}


def load_clip(device):
    print("--> Loading OpenAI CLIP (ViT-B/32)...", flush=True)
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("    ✓ CLIP model ready.", flush=True)
    return model, preprocess


def compute_diversity_score(embeddings):
    """
    Computes mean upper triangle pairwise cosine similarity for embeddings (N, D).
    """
    N = embeddings.size(0)
    if N < 2:
        return 0.0
    
    # Normalize embeddings
    normed = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(normed, normed.t()) # (N, N)
    
    # Extract upper triangle indices (excluding diagonal)
    triu_indices = torch.triu_indices(N, N, offset=1)
    pairwise_sims = sim_matrix[triu_indices[0], triu_indices[1]]
    
    return pairwise_sims.mean().item()


def process_class_shards(cls_name, raw_shards, input_dir, output_dir, clip_model, clip_preprocess,
                         sim_threshold, device, shard_size=1000):
    print(f"\n[-->] Filtering Class '{cls_name}'...", flush=True)
    
    import clip
    text_prompt = CLASS_PROMPTS[cls_name]
    text_tokens = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, p=2, dim=1)

    total_generated = 0
    passed_images = [] # list of (img_bytes, meta_dict, clip_score)
    all_image_embeddings = []

    for shard_file in raw_shards:
        shard_path = os.path.join(input_dir, shard_file)
        if not os.path.exists(shard_path):
            continue

        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            png_members = [m for m in members if m.name.endswith('.png')]
            
            for png_m in png_members:
                total_generated += 1
                key = png_m.name[:-4]
                meta_name = f"{key}.json"
                
                img_file = tar.extractfile(png_m)
                img_bytes = img_file.read()
                
                try:
                    meta_file = tar.extractfile(meta_name)
                    meta_dict = json.loads(meta_file.read().decode('utf-8'))
                except Exception:
                    meta_dict = {"key": key, "class_name": cls_name}

                # Preprocess image for CLIP
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                clip_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    img_emb = clip_model.encode_image(clip_tensor)
                    img_emb_norm = F.normalize(img_emb, p=2, dim=1)
                    sim_score = (img_emb_norm * text_emb).sum().item()

                if sim_score >= sim_threshold:
                    passed_images.append((img_bytes, meta_dict, sim_score))
                    all_image_embeddings.append(img_emb_norm.cpu())

    passed_count = len(passed_images)
    pass_rate = (passed_count / total_generated * 100.0) if total_generated > 0 else 0.0
    print(f"  Quality Gate: {passed_count}/{total_generated} passed (Similarity >= {sim_threshold:.2f}, Pass Rate: {pass_rate:.1f}%)", flush=True)

    # Diversity Check (subsample up to 1000 filtered images)
    diversity_score = 0.0
    homogeneity_warning = False

    if all_image_embeddings:
        stacked_embs = torch.cat(all_image_embeddings, dim=0)
        n_sample = min(1000, stacked_embs.size(0))
        indices = torch.randperm(stacked_embs.size(0))[:n_sample]
        sample_embs = stacked_embs[indices]
        
        diversity_score = compute_diversity_score(sample_embs)
        print(f"  Diversity Metric (Mean Pairwise Sim): {diversity_score:.4f}", flush=True)
        
        if diversity_score > 0.75:
            homogeneity_warning = True
            print(f"\n  [! WARNING !] Class '{cls_name}' is TOO HOMOGENEOUS! (Mean sim {diversity_score:.4f} > 0.75)", flush=True)
            print(f"  [ACTION REQUIRED]: Review prompts or increase temperature/seed variation for '{cls_name}'.\n", flush=True)

    # Save filtered images into clean tar shards
    out_shard_idx = 0
    out_shard_count = 0
    out_tar = None

    for i, (img_bytes, meta_dict, sim_score) in enumerate(passed_images):
        if out_shard_count == 0:
            out_shard_name = f"stl10_synth_{cls_name}_filtered_shard_{out_shard_idx:03d}.tar"
            out_shard_path = os.path.join(output_dir, out_shard_name)
            out_tar = tarfile.open(out_shard_path, "w")

        key = meta_dict.get("key", f"{cls_name}_{i:06d}")
        meta_dict["clip_sim_score"] = sim_score
        
        # Image entry
        info_img = tarfile.TarInfo(name=f"{key}.png")
        info_img.size = len(img_bytes)
        info_img.mtime = int(os.path.getmtime(os.path.join(input_dir, raw_shards[0]))) if raw_shards else 0
        out_tar.addfile(info_img, io.BytesIO(img_bytes))

        # Meta entry
        meta_bytes = json.dumps(meta_dict).encode('utf-8')
        info_meta = tarfile.TarInfo(name=f"{key}.json")
        info_meta.size = len(meta_bytes)
        info_meta.mtime = info_img.mtime
        out_tar.addfile(info_meta, io.BytesIO(meta_bytes))

        out_shard_count += 1
        if out_shard_count >= shard_size or i == len(passed_images) - 1:
            out_tar.close()
            out_shard_idx += 1
            out_shard_count = 0

    return {
        "class_name": cls_name,
        "total_generated": total_generated,
        "passed_quality": passed_count,
        "pass_rate_pct": round(pass_rate, 2),
        "diversity_mean_sim": round(diversity_score, 4),
        "homogeneity_flag": homogeneity_warning,
        "final_usable_count": passed_count
    }


def main():
    parser = argparse.ArgumentParser(description="CLIP Quality & Diversity Filter for Synthetic Data")
    parser.add_argument('--input-dir', type=str, default='./data/synthetic_stl10_raw',
                        help='Directory containing raw generated .tar shards')
    parser.add_argument('--output-dir', type=str, default='./data/synthetic_stl10_filtered',
                        help='Directory to save filtered .tar shards')
    parser.add_argument('--sim-threshold', type=float, default=0.25,
                        help='Minimum CLIP text-image cosine similarity (default: 0.25)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for CLIP evaluation (default: cuda)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    clip_model, clip_preprocess = load_clip(args.device)

    # Group raw shards by class name
    all_files = os.listdir(args.input_dir) if os.path.exists(args.input_dir) else []
    shards_by_class = {c: [] for c in STL10_CLASSES}
    for f in all_files:
        if f.endswith('.tar'):
            for c in STL10_CLASSES:
                if f"_{c}_" in f:
                    shards_by_class[c].append(f)
                    break

    summary_report = {}
    flags_triggered = []

    print(f"\n============================================================", flush=True)
    print(f"  Sprint 2 Phase B: Quality & Diversity Filtering", flush=True)
    print(f"  Input: {args.input_dir}  --> Output: {args.output_dir}", flush=True)
    print(f"  Similarity Threshold: {args.sim_threshold}", flush=True)
    print(f"============================================================", flush=True)

    for cls_name in STL10_CLASSES:
        raw_shards = sorted(shards_by_class[cls_name])
        if not raw_shards:
            print(f"\n[!] No raw shards found for class '{cls_name}'. Skipping.", flush=True)
            continue
            
        stats = process_class_shards(
            cls_name, raw_shards, args.input_dir, args.output_dir,
            clip_model, clip_preprocess, args.sim_threshold, args.device
        )
        summary_report[cls_name] = stats
        if stats["homogeneity_flag"]:
            flags_triggered.append(cls_name)

    # Save summary report
    report_path = os.path.join(args.output_dir, "clip_diversity_report.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n============================================================", flush=True)
    print("  Filtering Summary Report", flush=True)
    print("============================================================", flush=True)
    print(f"  {'Class':<12} | {'Raw':>6} | {'Passed':>6} | {'Pass %':>7} | {'Diversity':>9} | {'Flagged':>7}")
    print("  " + "-"*60)
    for c, stats in summary_report.items():
        flag_str = "⚠️ YES" if stats["homogeneity_flag"] else "OK"
        print(f"  {c:<12} | {stats['total_generated']:>6} | {stats['passed_quality']:>6} | "
              f"{stats['pass_rate_pct']:>6.1f}% | {stats['diversity_mean_sim']:>9.4f} | {flag_str:>7}")

    print(f"\nReport saved to: {report_path}", flush=True)
    if flags_triggered:
        print(f"\n⚠️  [ACTION REQUIRED]: {len(flags_triggered)} class(es) flagged as too homogeneous: {flags_triggered}")
        print("    Do not start training until reviewing or regenerating prompts for these classes!")
    else:
        print("\n  ✓ All classes passed quality and diversity checks cleanly.")

if __name__ == '__main__':
    main()

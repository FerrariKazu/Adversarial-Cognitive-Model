#!/usr/bin/env python3
"""
Sprint 2 Phase A: Synthetic Data Generation using SDXL Turbo
============================================================
Generates 100,000 synthetic STL-10 images (10,000 per class) using SDXL Turbo.
Saves images in WebDataset .tar shards (1,000 images/shard) with resume support.

Usage:
  # Generate all classes sequentially:
  python3 data_generation/generate_synthetic_stl10.py

  # Parallel execution across 2 GPUs (Kaggle T4x2):
  python3 data_generation/generate_synthetic_stl10.py --gpu-split even --device cuda:0
  python3 data_generation/generate_synthetic_stl10.py --gpu-split odd  --device cuda:1

  # Specific class index:
  python3 data_generation/generate_synthetic_stl10.py --class-index 0
"""

import os
import sys
import json
import time
import tarfile
import io
import argparse
import random
import numpy as np
from PIL import Image
import torch

STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]

# 5 prompts per class (50 prompts total)
PROMPTS = {
    "airplane": [
        "aircraft flying in blue sky, photorealistic, side view, 96x96",
        "commercial airplane landing on runway, clear day, photorealistic, 96x96",
        "fighter jet in flight, photorealistic, dramatic angle, 96x96",
        "small propeller plane over mountains, photorealistic, 96x96",
        "airplane on tarmac at airport, photorealistic, ground level, 96x96",
    ],
    "bird": [
        "bird perched on branch, photorealistic, clear background, 96x96",
        "colorful songbird in tree canopy, crisp focus, photorealistic, 96x96",
        "eagle soaring through overcast sky, majestic, photorealistic, 96x96",
        "small robin on wooden fence post, natural lighting, photorealistic, 96x96",
        "exotic bird in tropical rainforest, vibrant colors, photorealistic, 96x96",
    ],
    "car": [
        "automobile parked on street, front three-quarter view, photorealistic, 96x96",
        "modern sedan driving on asphalt road, motion blur, photorealistic, 96x96",
        "classic sports car in bright sunlight, side profile, photorealistic, 96x96",
        "compact car in urban parking spot, sharp details, photorealistic, 96x96",
        "luxury automobile on highway, dramatic reflection, photorealistic, 96x96",
    ],
    "cat": [
        "domestic cat, photorealistic, clear background, 96x96",
        "tabby cat resting on wooden floor, close-up portrait, photorealistic, 96x96",
        "fluffy cat looking directly at camera, soft natural light, photorealistic, 96x96",
        "ginger cat sitting outdoors in green grass, sharp focus, photorealistic, 96x96",
        "black cat with bright eyes on neutral background, photorealistic, 96x96",
    ],
    "deer": [
        "deer in forest, photorealistic, side view, 96x96",
        "antlered stag standing in misty morning meadow, photorealistic, 96x96",
        "young fawn among autumn trees, warm lighting, photorealistic, 96x96",
        "wild deer grazing in open field, clear profile, photorealistic, 96x96",
        "deer looking back through woodland foliage, sharp details, photorealistic, 96x96",
    ],
    "dog": [
        "dog outdoors, photorealistic, clear background, 96x96",
        "golden retriever running in sunny park, dynamic shot, photorealistic, 96x96",
        "german shepherd standing attentively, head portrait, photorealistic, 96x96",
        "small terrier sitting on grassy lawn, natural light, photorealistic, 96x96",
        "loyal dog looking up at camera outdoors, crisp focus, photorealistic, 96x96",
    ],
    "horse": [
        "horse in field, photorealistic, side view, 96x96",
        "brown horse galloping across open pasture, dramatic, photorealistic, 96x96",
        "thoroughbred horse standing by wooden farm fence, photorealistic, 96x96",
        "black horse portrait in golden hour sunlight, sharp detail, photorealistic, 96x96",
        "wild mustang running through grassy plains, photorealistic, 96x96",
    ],
    "monkey": [
        "monkey in tree, photorealistic, clear background, 96x96",
        "macaque sitting on mossy branch, rainforest backdrop, photorealistic, 96x96",
        "curious primate looking toward camera, detailed fur, photorealistic, 96x96",
        "small monkey swinging between jungle vines, natural environment, photorealistic, 96x96",
        "capuchin monkey on tree trunk, sharp eyes, photorealistic, 96x96",
    ],
    "ship": [
        "cargo ship at sea, photorealistic, side view, 96x96",
        "massive container vessel navigating ocean waves, photorealistic, 96x96",
        "large ship sailing during calm sunset, wide perspective, photorealistic, 96x96",
        "freighter ship at deep water harbor, industrial detail, photorealistic, 96x96",
        "maritime vessel moving through coastal waters, photorealistic, 96x96",
    ],
    "truck": [
        "truck on highway, photorealistic, side view, 96x96",
        "semi-trailer truck driving on open road, clear daylight, photorealistic, 96x96",
        "heavy commercial truck parked at rest stop, front angle, photorealistic, 96x96",
        "pickup truck on rural dirt path, rugged detail, photorealistic, 96x96",
        "freight transport truck under blue sky, sharp focus, photorealistic, 96x96",
    ]
}


def load_manifest(manifest_path):
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {"generated_counts": {cls_name: 0 for cls_name in STL10_CLASSES}, "shards": {}}


def save_manifest(manifest, manifest_path):
    os.makedirs(os.path.dirname(manifest_path) or '.', exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SDXL Turbo Synthetic Data Generator for STL-10")
    parser.add_argument('--output-dir', type=str, default='./data/synthetic_stl10_raw',
                        help='Directory to output WebDataset tar shards')
    parser.add_argument('--target-per-class', type=int, default=10000,
                        help='Target image count per class (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Generation batch size (default: 8)')
    parser.add_argument('--shard-size', type=int, default=1000,
                        help='Images per WebDataset .tar shard (default: 1000)')
    parser.add_argument('--gpu-split', type=str, choices=['all', 'even', 'odd'], default='all',
                        help='Filter class indices for multi-GPU split (all, even, odd)')
    parser.add_argument('--class-index', type=int, default=None,
                        help='Generate for a single class index (0-9)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run generation on (default: cuda)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    manifest = load_manifest(manifest_path)

    # Determine target classes
    if args.class_index is not None:
        target_indices = [args.class_index]
    elif args.gpu_split == 'even':
        target_indices = [i for i in range(10) if i % 2 == 0]
    elif args.gpu_split == 'odd':
        target_indices = [i for i in range(10) if i % 2 != 0]
    else:
        target_indices = list(range(10))

    print(f"============================================================", flush=True)
    print(f"  SDXL Turbo Synthetic Generator (Target: {args.target_per_class}/class)", flush=True)
    print(f"  Target Classes: {[STL10_CLASSES[i] for i in target_indices]}", flush=True)
    print(f"  Output Directory: {args.output_dir}", flush=True)
    print(f"  Device: {args.device}", flush=True)
    print(f"============================================================", flush=True)

    # Load Diffusers pipeline
    from diffusers import AutoPipelineForText2Image
    print("--> Loading stabilityai/sdxl-turbo pipeline (FP16)...", flush=True)
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if 'cuda' in args.device else torch.float32,
        variant="fp16" if 'cuda' in args.device else None
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print("    ✓ Pipeline ready.", flush=True)

    for class_idx in target_indices:
        cls_name = STL10_CLASSES[class_idx]
        current_count = manifest["generated_counts"].get(cls_name, 0)
        
        if current_count >= args.target_per_class:
            print(f"\n[+] Class '{cls_name}' already complete ({current_count}/{args.target_per_class}). Skipping.", flush=True)
            continue

        print(f"\n[-->] Generating class '{cls_name}' ({current_count}/{args.target_per_class})...", flush=True)
        prompts_for_class = PROMPTS[cls_name]
        
        shard_idx = current_count // args.shard_size
        shard_samples = current_count % args.shard_size
        
        shard_filename = f"stl10_synth_{cls_name}_shard_{shard_idx:03d}.tar"
        shard_path = os.path.join(args.output_dir, shard_filename)

        # Open or resume shard
        tar_mode = "a" if os.path.exists(shard_path) else "w"
        tar = tarfile.open(shard_path, tar_mode)

        start_time = time.time()
        
        while current_count < args.target_per_class:
            bs = min(args.batch_size, args.target_per_class - current_count)
            # Pick random prompts from the 5 class variations
            batch_prompts = [random.choice(prompts_for_class) for _ in range(bs)]
            
            with torch.no_grad():
                outputs = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=1,
                    guidance_scale=0.0
                ).images

            for i, img in enumerate(outputs):
                # Resize to 96x96 using high-quality Lanczos resampling
                img_96 = img.resize((96, 96), Image.Resampling.LANCZOS)
                
                # Convert image to bytes
                buffer = io.BytesIO()
                img_96.save(buffer, format='PNG')
                img_bytes = buffer.getvalue()
                
                # Format index key
                key = f"{cls_name}_{current_count:06d}"
                
                # Image file inside tar
                tarinfo_img = tarfile.TarInfo(name=f"{key}.png")
                tarinfo_img.size = len(img_bytes)
                tarinfo_img.mtime = int(time.time())
                tar.addfile(tarinfo_img, io.BytesIO(img_bytes))

                # JSON metadata file inside tar
                meta = {
                    "key": key,
                    "label": class_idx,
                    "class_name": cls_name,
                    "prompt": batch_prompts[i]
                }
                meta_bytes = json.dumps(meta).encode('utf-8')
                tarinfo_meta = tarfile.TarInfo(name=f"{key}.json")
                tarinfo_meta.size = len(meta_bytes)
                tarinfo_meta.mtime = int(time.time())
                tar.addfile(tarinfo_meta, io.BytesIO(meta_bytes))

                current_count += 1
                shard_samples += 1

                # If shard is full, close and start a new shard
                if shard_samples >= args.shard_size or current_count >= args.target_per_class:
                    tar.close()
                    manifest["generated_counts"][cls_name] = current_count
                    manifest["shards"][shard_filename] = shard_samples
                    save_manifest(manifest, manifest_path)
                    
                    if current_count < args.target_per_class:
                        shard_idx += 1
                        shard_samples = 0
                        shard_filename = f"stl10_synth_{cls_name}_shard_{shard_idx:03d}.tar"
                        shard_path = os.path.join(args.output_dir, shard_filename)
                        tar = tarfile.open(shard_path, "w")

            elapsed = time.time() - start_time
            rate = current_count / max(elapsed, 1.0) * 3600
            print(f"  Progress '{cls_name}': {current_count:>5}/{args.target_per_class}  |  "
                  f"Shard {shard_idx:03d} ({shard_samples}/{args.shard_size})  |  "
                  f"Est Speed: {rate:.0f} imgs/hr", flush=True)

        if not tar.closed:
            tar.close()

        manifest["generated_counts"][cls_name] = current_count
        save_manifest(manifest, manifest_path)
        print(f"  ✓ Class '{cls_name}' complete with {current_count} images.", flush=True)

    print("\n============================================================", flush=True)
    print("  Generation Session Finished Successfully!", flush=True)
    print(f"  Manifest saved to: {manifest_path}", flush=True)
    print("============================================================", flush=True)

if __name__ == '__main__':
    main()

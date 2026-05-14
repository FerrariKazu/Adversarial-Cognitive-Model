"""
Stimulus Export Pipeline
========================

PURPOSE:
    Convert the raw .npy adversarial arrays (normalized, 32×32, float32) into
    human-viewable PNG images (denormalized, upscaled to 128×128, uint8).

    These exported PNGs are the actual stimuli that human participants will see
    in the psychophysics experiment. Every design choice in this script exists
    to prevent experimental bias.

WHY STIMULI DESIGN CHOICES MATTER FOR A VALID EXPERIMENT:
    In psychophysics, the stimulus is the ONLY controlled variable between
    conditions. If stimuli differ in ANY way beyond the intended manipulation
    (adversarial perturbation level), we introduce confounds:

    1. UPSCALING METHOD — We use NEAREST-neighbor (not bilinear/bicubic).
       Bilinear smoothing would REDUCE the visibility of adversarial perturbations,
       artificially making attacked images look cleaner to humans. NEAREST
       preserves the exact pixel structure the CNN sees.

    2. FIXED RANDOM SEED — We select the same 50 images per class across all
       epsilon levels. This means a participant sees the SAME deer at ε=0.00 and
       ε=0.30, allowing within-image comparison. Without this, differences in
       accuracy could be due to some images being inherently harder.

    3. CONSISTENT SIZING — All images are 128×128 regardless of attack type.
       If FGSM images were 64×64 and PGD images were 128×128, participants
       might unconsciously use image size as a cue.

WHY 50 IMAGES PER CLASS (STATISTICAL POWER BASICS):
    Statistical power is the probability of detecting a real effect if it exists.
    With 10 classes × 50 images = 500 images per epsilon level:
    - For a paired t-test comparing human vs CNN accuracy at each epsilon,
      n=500 gives >95% power to detect a 5% accuracy difference (Cohen's d≈0.3).
    - For per-class analysis, n=50 gives ~80% power for medium effect sizes.
    - Fewer than 30 per class risks underpowered per-class comparisons.
    - More than 100 per class causes participant fatigue (>2hr session), which
      degrades response quality and introduces its own confound.

    50 per class is the standard in vision science for this kind of
    classification study (e.g., Geirhos et al., 2018; Dodge & Karam, 2017).
"""

import os
import sys
import csv
import random
import numpy as np
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.dataset import CLASSES

# =============================================================================
# CIFAR-10 Normalization Constants
# =============================================================================
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)

# Configuration
IMAGES_PER_CLASS = 50       # 50 images × 10 classes = 500 per epsilon
UPSCALE_SIZE = 128          # 32 → 128 (4× magnification)
SEED = 42                   # Fixed seed for reproducibility
ADV_DIR = '../phase2_attacks/adv_images'
OUTPUT_DIR = 'stimuli'


def denormalize(img_array):
    """
    Reverse the CIFAR-10 normalization to recover [0, 1] pixel values.

    1. WHAT: Computes  pixel = (normalized × std) + mean
    2. WHY: The .npy arrays store images in the normalized space the CNN sees:
             normalized = (pixel - mean) / std
       Humans need to see the actual RGB image, not the zero-centered version.
    3. OBSERVE: Output values will be in [0, 1], then scaled to [0, 255] uint8.

    Parameters
    ----------
    img_array : np.ndarray — Shape (3, 32, 32), float32, normalized.

    Returns
    -------
    np.ndarray — Shape (32, 32, 3), uint8, range [0, 255].
    """
    # Reverse normalization: pixel = normalized * std + mean
    img = img_array * CIFAR_STD + CIFAR_MEAN

    # Clamp to valid range (should already be valid after Phase 2 fix,
    # but defensive programming for data integrity)
    img = np.clip(img, 0.0, 1.0)

    # Convert from CHW (PyTorch format) to HWC (image format) and to uint8
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # (3, 32, 32) → (32, 32, 3)

    return img


def upscale_nearest(img_array, target_size):
    """
    Upscale using NEAREST-neighbor interpolation.

    1. WHAT: Magnifies each pixel into a target_size/32 × target_size/32 block.
    2. WHY: NEAREST preserves the exact pixel grid the CNN processed.
       Bilinear/bicubic would blur adversarial perturbations, reducing their
       visibility and biasing human performance upward (making the experiment
       conclude humans are better at ignoring perturbations than they really are).
    3. OBSERVE: At 4× magnification (32→128), each original pixel becomes a
       4×4 block. Adversarial "speckle" patterns are clearly visible.
    """
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.resize((target_size, target_size), Image.NEAREST)
    return pil_img


def select_indices_per_class(labels, images_per_class, seed):
    """
    Select a fixed set of image indices for each class.

    1. WHAT: Groups all test images by class, then randomly samples
       `images_per_class` indices from each group using a fixed seed.
    2. WHY: Using the SAME indices across all attack types and epsilons means
       every condition shows the exact same set of source images. This is a
       within-subjects design — the only variable that changes is the perturbation.
    3. OBSERVE: Returns a dict mapping class_idx → list of image indices.
    """
    rng = random.Random(seed)
    class_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    selected = {}
    for class_idx in range(10):
        pool = class_indices[class_idx]
        selected[class_idx] = sorted(rng.sample(pool, min(images_per_class, len(pool))))

    return selected


def export_stimuli():
    """Main export pipeline."""
    # Load labels
    labels = np.load(os.path.join(ADV_DIR, 'labels.npy'))
    print(f"Loaded labels: {labels.shape[0]} images, {len(np.unique(labels))} classes")

    # Select fixed image indices
    selected = select_indices_per_class(labels, IMAGES_PER_CLASS, SEED)
    total_per_class = {CLASSES[k]: len(v) for k, v in selected.items()}
    print(f"Selected images per class: {total_per_class}")

    # Discover available .npy files
    npy_files = sorted([f for f in os.listdir(ADV_DIR) if f.endswith('_images.npy')])
    print(f"Found {len(npy_files)} attack files to export")

    # Prepare manifest CSV
    manifest_rows = []
    total_exported = 0
    total_bytes = 0

    for npy_file in npy_files:
        # -----------------------------------------------------------------
        # Parse attack name and epsilon from filename
        # -----------------------------------------------------------------
        # Filenames: "fgsm_eps0.05_images.npy", "pgd_eps0.10_images.npy", "cw_images.npy"
        basename = npy_file.replace('_images.npy', '')
        if '_eps' in basename:
            parts = basename.split('_eps')
            attack_name = parts[0]
            epsilon_str = parts[1]
        else:
            attack_name = basename
            epsilon_str = 'auto'

        print(f"\n  Exporting {npy_file}...")
        data = np.load(os.path.join(ADV_DIR, npy_file))

        for class_idx, indices in selected.items():
            class_name = CLASSES[class_idx]
            out_dir = os.path.join(OUTPUT_DIR, attack_name, f"eps{epsilon_str}", class_name)
            os.makedirs(out_dir, exist_ok=True)

            for img_idx in indices:
                # Denormalize and upscale
                raw_img = denormalize(data[img_idx])
                pil_img = upscale_nearest(raw_img, UPSCALE_SIZE)

                # Save as PNG
                filename = f"{class_name}_{img_idx:05d}.png"
                filepath = os.path.join(out_dir, filename)
                pil_img.save(filepath, 'PNG')

                file_size = os.path.getsize(filepath)
                total_bytes += file_size
                total_exported += 1

                # Record in manifest
                manifest_rows.append({
                    'filepath': filepath,
                    'attack': attack_name,
                    'epsilon': epsilon_str,
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'image_idx': img_idx
                })

        print(f"    → {attack_name} eps={epsilon_str}: "
              f"{len(selected) * IMAGES_PER_CLASS} images exported")

    # -----------------------------------------------------------------
    # Write manifest CSV
    # -----------------------------------------------------------------
    # 1. WHAT: A lookup table mapping every exported PNG to its metadata.
    # 2. WHY: The manifest lets Phase 4 analysis scripts load the exact
    #    image shown to a participant by joining on filepath or image_idx.
    #    Without it, we'd have to reverse-engineer metadata from filenames.
    # 3. OBSERVE: CSV with one row per exported image.
    # -----------------------------------------------------------------
    manifest_path = 'stimuli_manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filepath', 'attack', 'epsilon', 'class_idx', 'class_name', 'image_idx'
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images exported : {total_exported}")
    print(f"  Total file size       : {total_bytes / (1024*1024):.1f} MB")
    print(f"  Image dimensions      : {UPSCALE_SIZE}×{UPSCALE_SIZE} PNG")
    print(f"  Images per class      : {IMAGES_PER_CLASS}")
    print(f"  Manifest              : {manifest_path} ({len(manifest_rows)} rows)")
    print(f"\n  Output structure:")
    print(f"    {OUTPUT_DIR}/")
    print(f"    ├── fgsm/eps0.00/airplane/airplane_00042.png")
    print(f"    ├── fgsm/eps0.05/cat/cat_03291.png")
    print(f"    ├── pgd/eps0.10/dog/dog_07845.png")
    print(f"    ├── cw/epsauto/truck/truck_09102.png")
    print(f"    └── ...")


if __name__ == '__main__':
    export_stimuli()

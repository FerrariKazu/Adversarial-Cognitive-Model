"""
Adversarial Dataset Generator
==============================

PURPOSE:
    Generate adversarial versions of the ENTIRE CIFAR-10 test set (10,000 images)
    for every attack type and every epsilon level. These saved .npy arrays will be
    used in Phase 3 (Human Study) as stimuli and in Phase 4 (Analysis) for
    statistical comparisons.

FILE NAMING CONVENTION:
    phase2_attacks/adv_images/
    ├── fgsm_eps0.00_images.npy     ← Clean images (ε=0, no perturbation)
    ├── fgsm_eps0.01_images.npy     ← FGSM at ε=0.01
    ├── fgsm_eps0.05_images.npy
    ├── ...
    ├── pgd_eps0.01_images.npy
    ├── pgd_eps0.05_images.npy
    ├── ...
    ├── cw_images.npy               ← C&W (no epsilon — it finds minimum L2)
    └── labels.npy                  ← True labels (shared by all)

WHY .npy FORMAT:
    NumPy's binary format is fast to save/load and preserves exact float32 values.
    The entire test set (10000 × 3 × 32 × 32 × float32) is about 117MB per file.
    Using .npy avoids lossy JPEG compression, which would corrupt the adversarial
    perturbations and invalidate the experiment.

COGNITIVE SCIENCE NOTE:
    These saved arrays are the "stimulus set" for our human psychophysics
    experiment. In experimental psychology, stimulus preparation is a separate
    step from data collection — you pre-generate all stimuli, verify them, and
    then present them to participants. This script is that preparation step.
"""

import sys
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase1_training.model import CIFARResNet
from phase1_training.dataset import get_dataloaders
from fgsm import fgsm_attack
from pgd import pgd_attack
from cw import cw_attack


def generate_for_attack(model, testloader, attack_fn, device, attack_name,
                        save_dir, total_samples, **attack_kwargs):
    """
    Run an attack on the entire test set and save as .npy file.

    1. WHAT: Iterates over all test batches, applies the attack, collects results.
    2. WHY: We need the complete attacked dataset saved to disk so that:
       - Phase 3 can load specific images to show human participants
       - Phase 4 can compute statistics over the full distribution
       - Results are reproducible without re-running the attack
    3. OBSERVE: Progress bar shows estimated time remaining.
    """
    # -------------------------------------------------------------------------
    # Normalized-space pixel clamp bounds
    # -------------------------------------------------------------------------
    # WHY THIS CLAMP IS NECESSARY:
    #   Adversarial attacks operate in the normalized space that the model sees
    #   (zero-mean, unit-variance per channel). However, attacks like FGSM and
    #   PGD blindly add ε·sign(∇) without checking whether the result maps back
    #   to a valid [0, 1] pixel value. For example, at ε=0.30, a normalized
    #   pixel at 2.51 (already near the channel max) could be pushed to 2.81,
    #   which corresponds to a raw pixel value > 1.0 — physically impossible.
    #
    #   For research-grade experiments this matters for two reasons:
    #   1. HUMAN STIMULI EXPORT (Phase 3): When we un-normalize these images to
    #      display to human participants, out-of-range values produce clipping
    #      artifacts (pure white/black pixels) that are visible to humans but
    #      were never "intended" by the attack. This confounds the psychophysics.
    #   2. MODEL EVALUATION CONSISTENCY: The model was trained on images within
    #      this normalized range. Feeding it out-of-distribution values means
    #      we're measuring the model's response to inputs it was never designed
    #      for, which is scientifically misleading.
    #
    # The bounds below are derived from CIFAR-10's channel statistics:
    #   min_val = (0 - mean) / std    (raw pixel 0 → normalized min)
    #   max_val = (1 - mean) / std    (raw pixel 1 → normalized max)
    # -------------------------------------------------------------------------
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([ 2.5141,  2.6078,  2.7537]).view(1, 3, 1, 1).to(device)

    all_adv_images = []

    pbar = tqdm(testloader, desc=f"  {attack_name}", leave=True)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        adv_images, _ = attack_fn(model, images, labels, device=device, **attack_kwargs)

        # Clamp to valid normalized pixel range before saving
        adv_images = torch.max(torch.min(adv_images, cifar_max), cifar_min)

        all_adv_images.append(adv_images.cpu().numpy())

    # Concatenate all batches into a single array
    all_adv_images = np.concatenate(all_adv_images, axis=0)

    # Save to disk
    save_path = os.path.join(save_dir, f"{attack_name}_images.npy")
    np.save(save_path, all_adv_images)

    return save_path, all_adv_images.shape


def verify_file(filepath):
    """
    Load a saved .npy file and print its properties for verification.

    1. WHAT: Sanity check that the file was saved correctly.
    2. WHY: Corrupted or mis-shaped stimulus files would silently break
       Phase 3 and Phase 4. Better to catch errors immediately.
    3. OBSERVE: Should print shape (10000, 3, 32, 32) and value range.
    """
    data = np.load(filepath)
    print(f"    ✓ {os.path.basename(filepath)}: shape={data.shape}, "
          f"dtype={data.dtype}, range=[{data.min():.4f}, {data.max():.4f}]")
    return data


def main():
    # Load config
    with open('../config/attack_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model = CIFARResNet().to(device)
    checkpoint_path = os.path.join('..', config['checkpoint_path'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Use smaller batch size to fit adversarial computation in 8GB VRAM
    _, testloader = get_dataloaders(batch_size=64, num_workers=4)
    total_samples = 10000  # CIFAR-10 test set size

    # Create output directory
    save_dir = 'adv_images'
    os.makedirs(save_dir, exist_ok=True)

    epsilons = config['epsilons']
    pgd_steps = config['pgd_steps']
    pgd_alpha = config['pgd_alpha']

    saved_files = []

    # =========================================================================
    # Save true labels (shared across all attacks)
    # =========================================================================
    print("\nSaving true labels...")
    all_labels = []
    for _, labels in testloader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    labels_path = os.path.join(save_dir, 'labels.npy')
    np.save(labels_path, all_labels)
    print(f"    ✓ labels.npy: shape={all_labels.shape}, "
          f"classes={np.unique(all_labels).tolist()}")

    # =========================================================================
    # FGSM — All epsilon levels
    # =========================================================================
    print(f"\n{'='*60}")
    print("FGSM Attack Generation")
    print(f"{'='*60}")
    for eps in epsilons:
        name = f"fgsm_eps{eps:.2f}"
        path, shape = generate_for_attack(
            model, testloader, fgsm_attack, device,
            attack_name=name, save_dir=save_dir, total_samples=total_samples,
            epsilon=eps
        )
        saved_files.append(path)

    # =========================================================================
    # PGD — All epsilon levels
    # =========================================================================
    print(f"\n{'='*60}")
    print("PGD Attack Generation")
    print(f"{'='*60}")
    for eps in epsilons:
        name = f"pgd_eps{eps:.2f}"
        path, shape = generate_for_attack(
            model, testloader, pgd_attack, device,
            attack_name=name, save_dir=save_dir, total_samples=total_samples,
            epsilon=eps, alpha=pgd_alpha, steps=pgd_steps
        )
        saved_files.append(path)

    # =========================================================================
    # C&W L2 — Single run (no epsilon parameter)
    # =========================================================================
    # NOTE: C&W is significantly slower than FGSM/PGD because it solves an
    # optimization problem for each image. With steps=100, expect ~10-20 min
    # on an RTX 4060 for the full test set.
    # =========================================================================
    print(f"\n{'='*60}")
    print("C&W L2 Attack Generation (this will be slow)")
    print(f"{'='*60}")
    path, shape = generate_for_attack(
        model, testloader, cw_attack, device,
        attack_name="cw", save_dir=save_dir, total_samples=total_samples
    )
    saved_files.append(path)

    # =========================================================================
    # Verification Pass
    # =========================================================================
    print(f"\n{'='*60}")
    print("Verification — Loading and checking all saved files")
    print(f"{'='*60}")
    verify_file(labels_path)
    for fp in saved_files:
        verify_file(fp)

    print(f"\n✓ All files saved to: {os.path.abspath(save_dir)}/")
    print(f"✓ Total files: {len(saved_files) + 1} (including labels.npy)")
    print("Ready for Phase 3 (Human Study) and Phase 4 (Analysis).")


if __name__ == '__main__':
    main()

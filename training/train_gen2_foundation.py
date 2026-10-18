"""Gen-2 Foundation Trainer — training/train_gen2_foundation.py
================================================================================

PURPOSE
-------
This file is the single entry point for training every Gen-2 model checkpoint.
It implements the full S0-S7 ladder described in the project plan and is the
only source of truth for the training recipe.  All cloud notebooks (colab_j1,
colab_j2, …) delegate to ``run_gen2_phase()`` or invoke this file via the CLI.

ARCHITECTURE OVERVIEW
---------------------
Every phase trains a ``Gen2FoundationModel`` from ``gen2_foundation/model.py``.
The model is a DINOv2-ViT-S backbone (12 blocks, 384 dims, 6 heads) with phase-
specific heads attached on top.  Phases are trained sequentially; each inherits
its predecessor's final-epoch weights.

  Phase 1  g2_gist_only     — 30 epochs.  Trunk + linear classifier only.
                               10 clean epochs then a 3-rung TRADES ramp
                               (eps 0.031 → 0.062 → 0.094 in norm space).
  Phase 2  recurrence_only  — 15 epochs.  T=4 glimpse fixed-EMA recurrence.
  Phase 3  belief_no_f      — 15 epochs.  Dirichlet evidential uncertainty head.
  Phase 4  belief_with_f    — 15 epochs.  UpdateNet belief dynamics + precision.
  Phase 5  ais_v2_swap      — 15 epochs.  AIS-v2 active gaze policy.
  Phase 6  gen1_core        — 15 epochs.  Full end-to-end joint fine-tune.

  Control / new baselines (J2):
  ff_112                    — 15 epochs.  Feedforward ViT-S at 112px / 64 tokens.
                               This is the Gen-3 reference baseline.
  ff_56                     — 15 epochs.  Same recipe, 56px (Gen-2 resolution).
                               Separates resolution gain from attack correction.
  ff_112_seed<N>            — ff_112 trained with a different random seed.

ATTACK RECIPE (CORRECTED)
-------------------------
All adversarial training uses TRADES with a KL-divergence inner problem.
The inner PGD loop uses the corrected step size::

    alpha = 2.5 * eps / steps

and a uniform random start in [-eps, eps], matching the Madry / AutoAttack
convention.  The original Gen-2 ladder used alpha = eps / steps (too small);
this has been corrected for ff_112 and all future checkpoints.

EVALUATION DURING TRAINING
---------------------------
The in-loop robust monitor runs PGD-10 on a fixed 512-image slice.  This is a
WEAK monitor (small n, old-convention step size at older phases).  Do NOT quote
the training log's robust column in any paper table; use the dedicated
``eval_gen2_foundation.py`` script for that.

DURABILITY
----------
- Rolling checkpoint saved after every epoch so cloud sessions can resume.
- Best-clean and best-robust checkpoints saved separately.
- Append-only JSONL log at ``report/gen2_epoch_log.jsonl``.
- 11-hour safety cutoff: saves a clean checkpoint and exits before cloud
  session is terminated mid-write.
- HuggingFace sync is injected via the ``epoch_callback`` hook (see Colab notebooks).

CLI USAGE
---------
  # Train a single phase:
  python3 training/train_gen2_foundation.py --phase ff_112 --img-size 112

  # Train the full original ladder:
  python3 training/train_gen2_foundation.py --phase all

  # Train ff_56 control and rename checkpoints accordingly:
  python3 training/train_gen2_foundation.py \
      --phase g2_gist_only --phase-name ff_56 --img-size 56 --n-epochs 15

  # Second ff_112 seed (for variance estimation):
  python3 training/train_gen2_foundation.py \
      --phase ff_112 --phase-name ff_112_seed42 --seed 42

DEPENDENCIES
------------
  gen2_foundation/model.py        — Gen2FoundationModel
  gen2_foundation/recipe.py       — AdamW group builder
  noesis_vision/models/foveation  — DEFAULT_FOVEA_SIZE constant
  training/curriculum_gen2.py     — per-epoch eps / beta schedule
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.backbone import CompactViTGen2
from gen2_foundation.model import Gen2FoundationModel
from gen2_foundation.recipe import build_adamw, AdamWGroups
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE
from training.curriculum_gen2 import (
    curriculum_for_phase_epoch,
    norm_to_pixel_eps,
    print_curriculum_header,
    MEAN_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Gen-2 Dedicated HF Repos (Partitioned from Gen-0 and Gen-1)
HF_BEST_REPO = "FerrariKazu/rhan-nxa-g2-checkpoints"
HF_ROLLING_REPO = "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"

# Ordered sequence of the six original Gen-2 phases.  Each phase inherits the
# FINAL-epoch weights of the phase immediately before it in this list.
PHASE_LADDER = [
    "g2_gist_only",
    "recurrence_only",
    "belief_no_f",
    "belief_with_f",
    "ais_v2_swap",
    "gen1_core",
]

# Default epoch budget per phase.  Can be overridden at runtime via the
# ``n_epochs`` argument to ``run_gen2_phase()`` or ``--n-epochs`` on the CLI.
PHASE_TOTAL_EPOCHS = {
    "g2_gist_only": 30,
    "recurrence_only": 15,
    "belief_no_f": 15,
    "belief_with_f": 15,
    "ais_v2_swap": 15,
    "gen1_core": 15,
    # J2 baselines
    "ff_112": 15,   # 112-px feedforward reference baseline
    "ff_56":  15,   # 56-px feedforward control (resolution ablation)
}

GATE_W_REFERENCE_ACC = 0.5852  # Gate W passed score


# ── PGD-KL Attack (evaluated strictly outside autocast in fp32) ──────────────
def run_pgd_kl_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    steps: int,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Inner PGD loop for the TRADES training objective.

    Builds x_adv by maximising KL(f(x_adv) || f(x_clean)), which is the
    TRADES inner problem.  Crucially, this runs OUTSIDE torch.cuda.amp.autocast
    so that all gradient computations are in fp32 — autocast is re-enabled only
    for the forward pass inside the outer TRADES loss call.

    Step-size convention (corrected):
        alpha = 2.5 * eps / steps  (Madry / AutoAttack standard)
    Random start:
        x_adv = x + U(-eps, eps)   (uniform, not Gaussian)

    Args:
        model:  The model being adversarially trained.  Temporarily set to
                eval() to suppress dropout/BN train-mode noise during attack.
        x:      Clean input batch, normalised to ImageNet stats, in [-4, 4].
        eps:    Perturbation radius in normalised pixel space (e.g. 0.094).
        steps:  Number of PGD steps.  For TRADES inner loop, typically 4.
        alpha:  Step size.  If None or <= 0, defaults to 2.5*eps/steps.

    Returns:
        x_adv: Adversarial batch, same shape as x, clamped to [x-eps, x+eps]
               and to [-4.0, 4.0].
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probs_clean = F.softmax(model(x).float(), dim=1)

        # Random start uniformly in [-eps, eps] matching standard TRADES/Madry
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        if alpha is None or alpha <= 0.0:
            alpha = (2.5 * eps) / max(steps, 1)

        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits_adv = model(x_adv)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv.float(), dim=1),
                    probs_clean,
                    reduction="batchmean",
                )
            grad = torch.autograd.grad(loss_kl, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = x_adv.clamp(-4.0, 4.0)
        return x_adv.detach()
    finally:
        if was_training:
            model.train()


# ── PGD Cross-Entropy Attack for Robust Evaluation ──────────────────────────
def run_pgd_ce_eval(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
) -> torch.Tensor:
    """PGD attack using cross-entropy loss, used for robust accuracy measurement.

    This is a SEPARATE function from the training attack (run_pgd_kl_attack).
    The training attack maximises KL divergence from the clean prediction;
    this attack maximises cross-entropy against the ground-truth label — a
    stronger evaluation objective.

    Step-size convention:  alpha = 2.5 * eps / steps  (corrected standard).
    Random start:          x_adv = x + U(-eps, eps).

    Used by:
      - In-loop robust monitor (512-image subset, steps=10) during training.
        NOTE: The in-loop monitor is a WEAK estimate.  Do NOT quote its numbers
        in paper tables.  Use eval_gen2_foundation.py for final evaluation.
      - eval_gen2_foundation.py::evaluate_batch() for the final PGD-50 row.

    Args:
        model:  Model to attack.  Set to eval() during the attack.
        x:      Clean input batch.
        y:      Ground-truth class labels.
        eps:    Perturbation radius in normalised pixel space.
        steps:  Number of PGD iterations.

    Returns:
        x_adv: Adversarial batch clamped to [x-eps, x+eps] ∩ [-4, 4].
    """
    was_training = model.training
    model.eval()
    try:
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        alpha = (2.5 * eps) / max(steps, 1) if steps > 0 else 0.0

        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = x_adv.clamp(-4.0, 4.0)
        return x_adv.detach()
    finally:
        if was_training:
            model.train()


# ── TRADES Loss ──────────────────────────────────────────────────────────────
def compute_trades_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    x_adv: torch.Tensor,
    beta: float,
    w_trades: float,
) -> Tuple[torch.Tensor, float, float]:
    """Compute the TRADES adversarial training loss.

    TRADES loss::

        L = w_trades * (CE(f(x_clean), y)  +  beta * KL(f(x_adv) || f(x_clean)))

    The weight ``w_trades`` scales the whole expression (default 0.55), which means
    the effective coefficients at beta=2.5, w_trades=0.55 are::

        0.55 * CE  +  1.375 * KL

    This differs from the doc formula "0.45*CE + 0.55*beta*KL" in plan.md —
    the code is the ground truth (verified via training logs in revision1.md §3.2).

    Args:
        model:    The model being trained.  Called twice: once on x_clean,
                  once on x_adv.  Runs inside torch.cuda.amp.autocast.
        x_clean:  Original (non-adversarial) input batch.
        y:        Ground-truth class labels.
        x_adv:    Adversarial batch produced by run_pgd_kl_attack.
        beta:     TRADES regularisation strength (2.5 at the top rung).
        w_trades: Overall loss weight (0.55 throughout the Gen-2 ladder).

    Returns:
        (total_loss, ce_scalar, kl_scalar):
          total_loss — differentiable tensor for backward().
          ce_scalar  — detached CE value for logging.
          kl_scalar  — detached KL value for logging.
    """
    logits_clean = model(x_clean)
    loss_ce = F.cross_entropy(logits_clean.float(), y)

    logits_adv = model(x_adv)
    loss_kl = F.kl_div(
        F.log_softmax(logits_adv.float(), dim=1),
        F.softmax(logits_clean.float().detach(), dim=1),
        reduction="batchmean",
    )
    total_loss = w_trades * (loss_ce + beta * loss_kl)
    return total_loss, float(loss_ce.item()), float(loss_kl.item())


# ── Dataset Loaders ──────────────────────────────────────────────────────────
def get_gen2_loaders(
    data_root: str,
    batch_size: int = 48,
    num_workers: int = 4,
    smoke: bool = False,
    img_size: int = DEFAULT_FOVEA_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build the four standard DataLoaders for Gen-2 training.

    Returns four loaders:
      train_loader  — full training set with random augmentation.
      val_loader    — full validation set (5,000 images), no augmentation.
      rob_loader    — fixed 512-image subset (seed 41) for in-loop robust eval.
      drift_loader  — fixed 256-image subset (seed 42) for feature-drift monitor.

    Augmentation policy (training only)::

        RandomResizedCrop(img_size, scale=(0.35, 1.0))
        RandomHorizontalFlip
        Normalize(ImageNet mean/std)

    All val/eval paths use deterministic Resize + Normalize only.

    IMPORTANT — img_size:
        The default is 56 (DEFAULT_FOVEA_SIZE), matching the original Gen-2 ladder.
        Pass img_size=112 for ff_112 and ff_112_seed* baselines.
        Pass img_size=56  for ff_56 (same as default; explicit for clarity).

    Smoke mode:
        When smoke=True, training is restricted to a random 2,000-image subset
        (seed 41) so a single epoch completes in ~20s.  Use this to time one
        epoch and project GPU-hour cost before committing a full run.

    Args:
        data_root:   Path to the root directory containing ``train/`` and ``val/``.
        batch_size:  Batch size for train and val loaders.
        num_workers: DataLoader worker processes.
        smoke:       If True, restricts training set to 2,000 images.
        img_size:    Resize all images to (img_size, img_size).

    Returns:
        (train_loader, val_loader, rob_loader, drift_loader)
    """
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tf_train)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf_val)

    if smoke:
        # 2,000 image subset for projection / smoke test
        g_smoke = torch.Generator().manual_seed(41)
        sub_indices = torch.randperm(len(train_ds), generator=g_smoke)[:2000].tolist()
        train_ds = Subset(train_ds, sub_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Fixed 512-image subset for robust eval
    g_rob = torch.Generator().manual_seed(41)
    rob_indices = torch.randperm(len(val_ds), generator=g_rob)[:512].tolist()
    rob_loader = DataLoader(
        Subset(val_ds, rob_indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Fixed 256-image subset for feature-drift monitoring
    g_drift = torch.Generator().manual_seed(42)
    drift_indices = torch.randperm(len(val_ds), generator=g_drift)[:256].tolist()
    drift_loader = DataLoader(
        Subset(val_ds, drift_indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, rob_loader, drift_loader


# ── Optimizer ────────────────────────────────────────────────────────────────
def build_gen2_optimizer(
    model: Gen2FoundationModel,
    warmup_epochs: int = 3,
    total_epochs: int = 30,
    trunk_lr: float = 5e-5,
    new_lr: float = 5e-4,
    weight_decay: float = 0.05,
    clip_norm: float = 1.0,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Build the AdamW multi-group optimizer and cosine LR scheduler.

    Two LR groups:
      backbone  — trunk parameters, lr=5e-5 (slow; preserves DINOv2 features).
      all other — heads, gist, recurrence, etc., lr=5e-4.

    Within each group, weight decay is applied to 2-D+ tensors (weight matrices)
    and disabled for 1-D tensors (bias, LayerNorm, LayerScale gamma, tokens).
    This follows the ViT/DINOv2 convention.

    LR schedule:
      3-epoch linear warmup followed by a cosine decay to 0 over the remaining
      epochs.  The schedule resets at the start of each phase (no carry-over).

    Gradient clipping:
      Applied per parameter group with clip_norm=1.0.  See clip_per_group().

    Returns:
        (optimizer, scheduler)  — pass both to run_gen2_phase.
    """
    groups = model.group_parameters()
    param_groups = []

    for name, params in groups.items():
        base = trunk_lr if name == "backbone" else new_lr
        decay_params = []
        no_decay_params = []

        for p in params:
            if not p.requires_grad:
                continue
            if p.ndim <= 1:
                no_decay_params.append(p)  # bias, norm, gammas, tokens
            else:
                decay_params.append(p)

        if decay_params:
            param_groups.append({
                "params": decay_params,
                "lr": base,
                "weight_decay": weight_decay,
                "name": f"{name}_decay",
                "clip_norm": clip_norm,
            })
        if no_decay_params:
            param_groups.append({
                "params": no_decay_params,
                "lr": base,
                "weight_decay": 0.0,
                "name": f"{name}_no_decay",
                "clip_norm": clip_norm,
            })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=new_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ── Gradient Clipping ────────────────────────────────────────────────────────
def clip_per_group(optimizer: torch.optim.Optimizer) -> Tuple[Dict[str, float], bool]:
    """Clip gradients per parameter group and record pre-clip gradient norms.

    Each group's clip budget comes from its ``clip_norm`` field (default 1.0).
    Pre-clip norms are recorded for the health-flag check (CLIP-SATURATED) and
    printed in the epoch summary.

    Returns:
        pre_norms    — dict mapping group name -> pre-clip gradient norm.
        is_saturated — True if ALL groups were at >= 95% of their clip budget.
                       Triggers the CLIP-SATURATED health flag.
    """
    pre_norms = {}
    is_saturated = True
    for g in optimizer.param_groups:
        params = [p for p in g["params"] if p.grad is not None]
        if not params:
            continue
        pre = torch.nn.utils.clip_grad_norm_(params, max_norm=1e4)
        budget = float(g.get("clip_norm", 1.0))
        torch.nn.utils.clip_grad_norm_(params, max_norm=budget)
        pre_val = float(pre.item())
        pre_norms[g["name"]] = pre_val
        if pre_val < budget * 0.95:
            is_saturated = False
    return pre_norms, is_saturated


# ── Feature Drift Monitor ────────────────────────────────────────────────────
def compute_feature_drift(
    model: Gen2FoundationModel,
    initial_trunk_sd: Dict[str, torch.Tensor],
    drift_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Measure how far the trunk has drifted from its initialisation.

    Reports the mean relative weight change across all trunk parameters::

        drift = mean over layers of  |W_t - W_0|_F / |W_0|_F

    This is computed every 5 epochs on the fixed 256-image drift_loader.
    A large drift (>10%) relative to the DINOv2 warm-start typically indicates
    the trunk LR is too high or the loss has collapsed.

    Args:
        model:            The model being trained (eval() is called internally).
        initial_trunk_sd: Snapshot of backbone.state_dict() taken at epoch 0.
        drift_loader:     Fixed 256-image validation subset for measurement.
        device:           CUDA or CPU device.

    Returns:
        dict with key ``mean_rel_weight_change`` (float, fraction not percent).
    """
    model.eval()
    all_current_feats = []
    with torch.no_grad():
        for x, _ in drift_loader:
            x = x.to(device)
            x_prep = model.backbone._prep(x)
            feats = model.backbone._trunk_forward(x_prep)[:, 0]
            all_current_feats.append(feats.cpu())
    curr_cat = torch.cat(all_current_feats, dim=0)

    # Relative weight change |W - W0| / |W0|
    current_sd = model.backbone.state_dict()
    rel_weight_diffs = []
    for k, v0 in initial_trunk_sd.items():
        if k in current_sd and v0.is_floating_point():
            vt = current_sd[k].cpu()
            norm0 = v0.norm().item()
            if norm0 > 1e-6:
                diff = (vt - v0).norm().item() / norm0
                rel_weight_diffs.append(diff)
    mean_rel_weight = sum(rel_weight_diffs) / max(len(rel_weight_diffs), 1)

    return {
        "mean_rel_weight_change": mean_rel_weight,
    }


# ── Full Phase Training Loop ──────────────────────────────────────────────────
def run_gen2_phase(
    phase: str,
    arm: str,
    data_root: str,
    ckpt_dir: str,
    report_dir: str,
    dino_path: str,
    batch_size: int = 48,
    num_workers: int = 4,
    smoke: bool = False,
    center_crop_mode: bool = False,
    epoch_callback: Optional[Callable[[str, int], None]] = None,
    phase_override_name: Optional[str] = None,
    img_size: Optional[int] = None,
    n_epochs: Optional[int] = None,
    training_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one phase of the Gen-2 (or control/ablation) ladder.

    This is the canonical training entry point.  Colab notebooks call this
    function directly; the CLI in ``main()`` also delegates to it.

    Phase lifecycle
    ---------------
    1. Build model for the requested phase (model architecture is phase-specific).
    2. Load weights:
       - Phases ``g2_gist_only``, ``ff_112``, ``ff_56``: load DINOv2 warm-start.
       - All other phases: inherit FINAL-epoch weights from the previous phase.
    3. Build AdamW optimizer + cosine scheduler (resets each phase).
    4. Check for a rolling checkpoint and resume if found and compatible.
    5. Train for ``total_epochs`` epochs with TRADES-PGD adversarial training.
    6. Save rolling / best-clean / best-robust / final checkpoints.
    7. Write every epoch to an append-only JSONL log.
    8. Call ``epoch_callback(phase, epoch)`` after each epoch (used for HF sync).

    Checkpoint naming
    -----------------
    By default, checkpoints are named ``foundation_{phase}_*.pth``.  Pass
    ``phase_override_name`` to rename them to ``foundation_{override}_*.pth``
    without changing the model architecture.  This is used for:
      - ``ff_56``  (trains as ``g2_gist_only`` but checkpoints named ``ff_56``)
      - ``ff_112_seed42`` / ``ff_112_seed99``  (additional seeds)

    Args:
        phase:               Architecture phase key, must be in PHASE_LADDER
                             or a recognised extension (``ff_112``, ``ff_56``).
        arm:                 Model arm variant (``v1`` or ``v2``, relevant for
                             ``belief_with_f`` only).
        data_root:           Path to ImageNet-100 root (contains ``train/``
                             and ``val/`` subdirectories).
        ckpt_dir:            Directory where checkpoints are written.
        report_dir:          Directory where the JSONL epoch log is written.
        dino_path:           Path to the DINOv2-small pretrained weights
                             (``dinov2_vits14_pretrain.pth``, 88 MB).
        batch_size:          Training and val batch size (default 48).
        num_workers:         DataLoader worker processes (default 4).
        smoke:               If True, restricts training to 2,000 images for
                             one epoch.  Use to project GPU hours.
        center_crop_mode:    Deprecated flag; kept for backward compatibility.
        epoch_callback:      Callable invoked as ``fn(phase, epoch)`` after each
                             epoch.  Use for HuggingFace checkpoint sync.
        phase_override_name: If set, checkpoints and log entries use this name
                             instead of ``phase``.  Does NOT change model arch.
        img_size:            Override the input image size.  Defaults to 112 for
                             ``ff_112`` / ``feedforward_112`` phases, 56 for all
                             others.  Pass explicitly for ``ff_56``.
        n_epochs:            Override the total epoch count from
                             ``PHASE_TOTAL_EPOCHS``.  Useful for shorter ablation
                             runs or for training beyond the default budget.
        training_seed:       If set, calls ``torch.manual_seed(seed)`` before
                             building data loaders and model.  Use for multi-seed
                             variance estimation (ff_112_seed42, ff_112_seed99, …).

    Returns:
        dict with keys:
          ``status``       — ``'complete'`` or ``'checkpointed_11h'``.
          ``phase``        — the effective phase name (may be override).
          ``best_clean``   — best clean val accuracy observed during training.
          ``best_robust``  — best in-loop robust accuracy observed.
    """
    # Apply optional training seed before anything else so data order and
    # weight init are reproducible for multi-seed variance experiments.
    if training_seed is not None:
        torch.manual_seed(training_seed)

    # The "effective" phase name governs checkpoint filenames and log entries.
    # It may differ from the architecture ``phase`` when phase_override_name is set
    # (e.g., training g2_gist_only but naming outputs "ff_56").
    eff_name = phase_override_name if phase_override_name else phase

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n[Gen-2] Launching: arch_phase='{phase}'  eff_name='{eff_name}'  arm={arm}  device={device}")
    if phase_override_name:
        print(f"  NOTE: checkpoints will be named 'foundation_{eff_name}_*.pth'")
    print_curriculum_header()

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    jsonl_path = os.path.join(report_dir, "gen2_epoch_log.jsonl")

    # Resolve epoch budget and image size
    total_epochs = n_epochs if n_epochs is not None else PHASE_TOTAL_EPOCHS.get(phase, 15)
    if img_size is not None:
        fovea_size = img_size
    else:
        fovea_size = 112 if phase in ("ff_112", "feedforward_112") else DEFAULT_FOVEA_SIZE

    train_loader, val_loader, rob_loader, drift_loader = get_gen2_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers, smoke=smoke, img_size=fovea_size
    )

    model = Gen2FoundationModel(
        phase=phase,
        arm=arm,
        fovea_size=fovea_size,
        center_crop_mode=center_crop_mode,
    ).to(device)

    # Weights initialization / inheritance
    if phase in ("g2_gist_only", "ff_112", "feedforward_112"):
        print(f"Loading official DINOv2-small warm-start from {dino_path}...")
        rep = model.backbone.load_dino_warm_start(dino_path)
        print(f"Warm-start load: accepted={rep['accepted']}, "
              f"matched={rep['matched_trunk_keys']}/{rep['trunk_keys']} "
              f"({rep['matched_fraction']*100:.1f}%)")
    else:
        # S4: Later phases inherit previous phase's FINAL-epoch weights (not best.pth)
        prev_idx = PHASE_LADDER.index(phase) - 1
        prev_phase = PHASE_LADDER[prev_idx]
        prev_final_path = os.path.join(ckpt_dir, f"foundation_{prev_phase}_final.pth")
        if os.path.exists(prev_final_path):
            print(f"Inheriting FINAL weights from previous phase: {prev_final_path}")
            st = torch.load(prev_final_path, map_location="cpu")
            saved_sd = st["model"]
            model_sd = model.state_dict()
            filtered_sd = {}
            for k, v in saved_sd.items():
                if k in model_sd:
                    if v.shape == model_sd[k].shape:
                        filtered_sd[k] = v
                    else:
                        print(f"  Note: skipping direct copy of '{k}' due to phase shape transition ({v.shape} -> {model_sd[k].shape})")
                        if k == "cls_head.weight" and v.shape[0] == model.cls_head.weight.shape[0]:
                            with torch.no_grad():
                                model.cls_head.weight[:, :v.shape[1]].copy_(v)
                            print(f"  Partially copied cls_head.weight[:, :{v.shape[1]}] from {prev_phase}.")
            model.load_state_dict(filtered_sd, strict=False)
        else:
            print(f"WARNING: Parent checkpoint {prev_final_path} not found. Cold start.")

    initial_trunk_sd = {k: v.cpu().clone() for k, v in model.backbone.state_dict().items()}

    # Optimizer & Scheduler
    optimizer, scheduler = build_gen2_optimizer(
        model, warmup_epochs=3, total_epochs=total_epochs, clip_norm=1.0
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_clean_acc = 0.0
    best_rob_acc = 0.0
    stall_count = 0
    start_time = time.time()

    # Resume check — with stale-checkpoint detection
    # All checkpoint paths use eff_name so phase_override_name is honoured.
    rolling_path = os.path.join(ckpt_dir, f"foundation_{eff_name}_rolling.pth")
    if os.path.exists(rolling_path):
        print(f"Resuming from rolling checkpoint: {rolling_path}")
        saved = torch.load(rolling_path, map_location="cpu")
        saved_keys = set(saved["model"].keys())
        model_keys = set(model.state_dict().keys())
        missing = model_keys - saved_keys
        unexpected = saved_keys - model_keys
        if missing or unexpected:
            print(f"WARNING: Stale rolling checkpoint detected!")
            if missing:
                print(f"  Missing keys ({len(missing)}): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
            print(f"  Discarding stale checkpoint. Phase will restart from inherited or warm-start weights.")
            os.rename(rolling_path, rolling_path + ".stale")
        else:
            start_epoch = saved.get("epoch", 0) + 1
            model.load_state_dict(saved["model"])
            if "optimizer" in saved:
                optimizer.load_state_dict(saved["optimizer"])
            if "scheduler" in saved:
                scheduler.load_state_dict(saved["scheduler"])
            best_clean_acc = saved.get("best_clean_acc", 0.0)
            best_rob_acc = saved.get("best_rob_acc", 0.0)
            print(f"Resumed at epoch {start_epoch}. Verified rolling state.")
            if start_epoch > total_epochs:
                print(f"✓ Phase '{phase}' rolling checkpoint already completed all {total_epochs} epochs. Marking complete.")
                final_path = os.path.join(ckpt_dir, f"foundation_{phase}_final.pth")
                if not os.path.exists(final_path):
                    torch.save(saved, final_path)
                    print(f"  Saved final checkpoint from rolling state to: {final_path}")
                return {
                    "status": "complete", "phase": phase,
                    "best_clean": best_clean_acc, "best_robust": best_rob_acc,
                }

    ckpt_state: Optional[Dict[str, Any]] = None
    recent_ep_durations: list = []  # rolling window for ETA smoothing
    for epoch in range(start_epoch, total_epochs + 1):
        ep_t0 = time.time()
        elapsed_hours = (time.time() - start_time) / 3600.0

        # Safe 11h cutoff
        if elapsed_hours >= 11.0:
            print(f"\n[Session Safety] 11.0h limit reached ({elapsed_hours:.2f}h). Saving clean checkpoint and halting.")
            torch.save({
                "phase": phase, "epoch": epoch - 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_clean_acc": best_clean_acc, "best_rob_acc": best_rob_acc,
            }, rolling_path)
            if epoch_callback is not None:
                try:
                    epoch_callback(phase, epoch - 1)
                except Exception as e:
                    print(f"  Warning: epoch_callback failed at 11h halt: {e}", flush=True)
            return {"status": "checkpointed_11h", "last_epoch": epoch - 1}

        curric = curriculum_for_phase_epoch(phase, epoch)
        print(curric.banner(), flush=True)

        model.train()
        tot_loss = 0.0
        tot_ce = 0.0
        tot_kl = 0.0
        n_correct_clean = 0
        n_correct_adv = 0
        n_samples = 0
        clip_sat_steps = 0
        step_count = 0

        total_batches = len(train_loader)
        log_interval = 10 if smoke else 25

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if curric.adversarial and curric.w_trades > 0:
                # Attack computed outside autocast in fp32
                x_adv = run_pgd_kl_attack(
                    model, x, eps=curric.eps_norm, steps=curric.pgd_steps
                )
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss, ce_val, kl_val = compute_trades_loss(
                        model, x, y, x_adv, beta=curric.beta, w_trades=curric.w_trades
                    )
            else:
                x_adv = x
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    ce_val = float(loss.item())
                    kl_val = 0.0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            pre_norms, is_sat = clip_per_group(optimizer)
            if is_sat:
                clip_sat_steps += 1
            step_count += 1

            scaler.step(optimizer)
            scaler.update()

            tot_loss += loss.item() * y.size(0)
            tot_ce += ce_val * y.size(0)
            tot_kl += kl_val * y.size(0)
            n_samples += y.size(0)

            if step == 1 or step % log_interval == 0 or step == total_batches:
                step_elapsed = max(time.time() - ep_t0, 1e-4)
                img_per_sec = (step * y.size(0)) / step_elapsed
                rem_steps = total_batches - step
                step_eta_min = (rem_steps * (step_elapsed / step)) / 60.0
                cur_loss = tot_loss / max(n_samples, 1)
                cur_ce = tot_ce / max(n_samples, 1)
                cur_kl = tot_kl / max(n_samples, 1)
                cur_lr = optimizer.param_groups[0]["lr"]
                max_norm = max(pre_norms.values()) if pre_norms else 0.0
                pct = (step / total_batches) * 100.0
                print(
                    f"  [{phase}] Ep {epoch}/{total_epochs} [{step:>4}/{total_batches} | {pct:>5.1f}%] "
                    f"loss: {cur_loss:.4f} (ce: {cur_ce:.4f}, kl: {cur_kl:.4f}) | "
                    f"lr: {cur_lr:.2e} | gnorm: {max_norm:.2f} | "
                    f"{img_per_sec:.1f} img/s | eta: {step_eta_min:.1f}m",
                    flush=True,
                )

        scheduler.step()

        # Validation Clean (Full 5,000 images)
        print(f"  [{phase}] Running clean validation (5,000 images)...", flush=True)
        model.eval()
        v_clean_correct = 0
        v_clean_total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_logits = model(vx)
                v_clean_correct += (v_logits.argmax(dim=-1) == vy).sum().item()
                v_clean_total += vy.size(0)
        val_clean_acc = v_clean_correct / max(v_clean_total, 1)

        # Validation Robust (Fixed 512-image subset with PGD-10 @ current eps)
        rob_eps = curric.eps_norm if curric.adversarial else 0.031
        print(f"  [{phase}] Running robust validation (512 images, PGD-10 @ eps={rob_eps:.3f})...", flush=True)
        v_rob_correct = 0
        v_rob_total = 0
        for rx, ry in rob_loader:
            rx, ry = rx.to(device, non_blocking=True), ry.to(device, non_blocking=True)
            rx_adv = run_pgd_ce_eval(model, rx, ry, eps=rob_eps, steps=10)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    r_logits = model(rx_adv)
            v_rob_correct += (r_logits.argmax(dim=-1) == ry).sum().item()
            v_rob_total += ry.size(0)
        val_rob_acc = v_rob_correct / max(v_rob_total, 1)

        # Health flags checking
        health_flags = []
        if epoch >= 5 and val_clean_acc < 0.02:
            health_flags.append("CHANCE-LEVEL")
        if step_count > 0 and (clip_sat_steps / step_count) >= 0.90:
            health_flags.append("CLIP-SATURATED")
        if curric.adversarial and (tot_kl / max(n_samples, 1)) < 1e-6:
            health_flags.append("TRADES-DEAD")
        if val_rob_acc > val_clean_acc + 0.01:
            health_flags.append("ROBUST>CLEAN")
        if val_clean_acc <= best_clean_acc:
            stall_count += 1
            if stall_count >= 8:
                health_flags.append("STALL")
        else:
            stall_count = 0

        # Feature drift monitor (every 5 epochs)
        drift_info = {}
        if epoch % 5 == 0:
            drift_info = compute_feature_drift(model, initial_trunk_sd, drift_loader, device)

        ep_duration = time.time() - ep_t0
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        rem_epochs = total_epochs - epoch

        # Rolling-average ETA (last 3 epochs to smooth slow first-epoch cache warming)
        recent_ep_durations.append(ep_duration)
        if len(recent_ep_durations) > 3:
            recent_ep_durations.pop(0)
        avg_ep_s = sum(recent_ep_durations) / len(recent_ep_durations)
        phase_eta_min = (rem_epochs * avg_ep_s) / 60.0

        # Session budget (how many epochs fit before the 11h safety halt)
        session_elapsed_h = (time.time() - start_time) / 3600.0
        session_remain_s = max(0.0, 11.0 * 3600.0 - session_elapsed_h * 3600.0)
        epochs_fit_in_session = int(session_remain_s / avg_ep_s) if avg_ep_s > 0 else rem_epochs
        session_warn = f" ⚠ WILL NEED RESUME (only ~{epochs_fit_in_session} more fit this session)" if epochs_fit_in_session < rem_epochs else ""

        # Labeled epoch block
        print("-" * 70)
        print(
            f"[{phase}] Epoch {epoch}/{total_epochs} ({ep_duration:.1f}s, VRAM: {vram_mb:.0f}MB) "
            f"| session: {session_elapsed_h:.2f}h/11h | phase ETA: {phase_eta_min:.1f}m{session_warn}"
        )
        print(f"  val_acc[CLEAN]:  {val_clean_acc*100:.2f}% (best: {max(best_clean_acc, val_clean_acc)*100:.2f}%)")
        print(f"  val_acc[ROBUST]: {val_rob_acc*100:.2f}% (eps={rob_eps:.3f}, best: {max(best_rob_acc, val_rob_acc)*100:.2f}%)")
        print(f"  loss: total={tot_loss/max(n_samples,1):.4f} ce={tot_ce/max(n_samples,1):.4f} kl={tot_kl/max(n_samples,1):.4f}")
        lr_strs = [f"{g['name']}: {g['lr']:.2e}" for g in optimizer.param_groups[:3]]
        print(f"  lr: {', '.join(lr_strs)}")
        norm_strs = [f"{k}: {v:.2f}" for k, v in list(pre_norms.items())[:4]]
        print(f"  gnorm: {', '.join(norm_strs)}")
        print(f"  health flags: {health_flags or ['OK']}")
        if drift_info:
            print(f"  feature drift: mean |dW|/|W0| = {drift_info.get('mean_rel_weight_change', 0.0)*100:.2f}%")
        print("-" * 70, flush=True)

        # Checkpoints saving
        ckpt_state = {
            "phase": phase, "arm": arm, "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_clean_acc": val_clean_acc,
            "val_rob_acc": val_rob_acc,
            "best_clean_acc": max(best_clean_acc, val_clean_acc),
            "best_rob_acc": max(best_rob_acc, val_rob_acc),
        }
        torch.save(ckpt_state, rolling_path)

        if val_clean_acc > best_clean_acc:
            best_clean_acc = val_clean_acc
            torch.save(ckpt_state, os.path.join(ckpt_dir, f"foundation_{eff_name}_best_clean.pth"))

        if val_rob_acc > best_rob_acc:
            best_rob_acc = val_rob_acc
            torch.save(ckpt_state, os.path.join(ckpt_dir, f"foundation_{eff_name}_best_robust.pth"))

        # Durable append-only JSONL record
        log_record = {
            "phase": phase, "arm": arm, "epoch": epoch, "total_epochs": total_epochs,
            "val_acc_clean": val_clean_acc, "val_acc_robust": val_rob_acc,
            "loss_total": tot_loss / max(n_samples, 1),
            "loss_ce": tot_ce / max(n_samples, 1),
            "loss_kl": tot_kl / max(n_samples, 1),
            "health_flags": health_flags,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(jsonl_path, "a", encoding="utf-8") as f_log:
            f_log.write(json.dumps(log_record) + "\n")

        # Invoke per-epoch persistence callback (e.g. HuggingFace sync)
        if epoch_callback is not None:
            try:
                epoch_callback(phase, epoch)
            except Exception as e:
                print(f"  Warning: epoch_callback failed at epoch {epoch}: {e}", flush=True)

        if smoke:
            print(f"\n[Smoke Benchmark] 1 smoke epoch completed in {ep_duration:.1f}s.")
            break

        # ── Gate C Check (epoch 10 of g2_gist_only) ─────────────────────────
        if curric.gate_trigger == "GATE_C":
            if val_clean_acc < GATE_W_REFERENCE_ACC:
                print(f"\nSTOP: GATE C FAILED — clean val ({val_clean_acc*100:.2f}%) did not exceed "
                      f"Gate W probe accuracy ({GATE_W_REFERENCE_ACC*100:.2f}%).")
                sys.exit(1)
            else:
                print(f"\n✓ GATE C PASSED: clean val ({val_clean_acc*100:.2f}%) >= Gate W probe ({GATE_W_REFERENCE_ACC*100:.2f}%)")

        # ── Gate R Check (epoch 17 of g2_gist_only, end of rung 1) ──────────
        if curric.gate_trigger == "GATE_R":
            if val_clean_acc < 0.313:
                print(f"\nPAUSE: GATE R TRIGGERED — clean val ({val_clean_acc*100:.2f}%) < 31.3%. "
                      f"Stopping for user review per specification.")
                sys.exit(1)
            else:
                print(f"\n✓ GATE R PASSED: clean val ({val_clean_acc*100:.2f}%) >= 31.3%")

    # Final checkpoint at completion
    final_path = os.path.join(ckpt_dir, f"foundation_{eff_name}_final.pth")
    if ckpt_state is not None:
        torch.save(ckpt_state, final_path)
    print(f"\n✓ Phase '{eff_name}' COMPLETE. Final checkpoint: {final_path}")
    print(f"  best_clean: {best_clean_acc*100:.2f}%  best_robust: {best_rob_acc*100:.2f}%")
    return {
        "status": "complete", "phase": eff_name,
        "best_clean": best_clean_acc, "best_robust": best_rob_acc,
    }


def main():
    """CLI entry point.  See module docstring for usage examples."""
    parser = argparse.ArgumentParser(
        description="Gen-2 Foundation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full original Gen-2 ladder:
  python3 training/train_gen2_foundation.py --phase all

  # 112-px feedforward baseline (the Gen-3 reference):
  python3 training/train_gen2_foundation.py --phase ff_112

  # 56-px feedforward control (resolution ablation):
  python3 training/train_gen2_foundation.py \\
      --phase g2_gist_only --phase-name ff_56 --img-size 56 --n-epochs 15

  # Second ff_112 seed for variance estimation:
  python3 training/train_gen2_foundation.py \\
      --phase ff_112 --phase-name ff_112_seed42 --seed 42

  # One-epoch smoke test to project GPU hours:
  python3 training/train_gen2_foundation.py --phase ff_112 --smoke
""",
    )
    # ── Architecture ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--phase", type=str, default="g2_gist_only",
        choices=PHASE_LADDER + ["ff_112", "ff_56", "all"],
        help="Architecture phase to train.  Use 'all' to run the full Gen-2 ladder.",
    )
    parser.add_argument(
        "--arm", type=str, default="v1", choices=["v1", "v2"],
        help="Model arm variant.  Only matters for belief_with_f (default: v1).",
    )
    parser.add_argument(
        "--phase-name", type=str, default=None, dest="phase_name",
        help=("Override the name used for checkpoint files and log entries.  "
              "Use when training g2_gist_only but wanting outputs named 'ff_56', "
              "or when training additional seeds of ff_112."),
    )
    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-root", type=str, default="data/imagenet100",
        help="Path to ImageNet-100 root directory (must contain train/ and val/).",
    )
    parser.add_argument(
        "--img-size", type=int, default=None, dest="img_size",
        help=("Input image resolution in pixels (square).  Defaults to 112 for ff_112, "
              "56 for all other phases.  Set explicitly when using --phase-name."),
    )
    # ── Infrastructure ────────────────────────────────────────────────────────
    parser.add_argument(
        "--ckpt-dir", type=str, default="checkpoints",
        help="Directory where checkpoints are written.",
    )
    parser.add_argument(
        "--report-dir", type=str, default="report",
        help="Directory where the epoch JSONL log is written.",
    )
    parser.add_argument(
        "--dino-path", type=str,
        default="checkpoints/dinov2_vits14_pretrain.pth",
        help="Path to the DINOv2-small pretrained weights (~88 MB).",
    )
    # ── Training hyperparameters ───────────────────────────────────────────────
    parser.add_argument(
        "--batch-size", type=int, default=48,
        help="Train and val batch size.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=None, dest="n_epochs",
        help=("Override the total epoch count from PHASE_TOTAL_EPOCHS.  "
              "Useful for shorter ablation runs."),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=("Random seed for multi-seed variance experiments.  When set, calls "
              "torch.manual_seed(seed) before any data loading or model init."),
    )
    # ── Flags ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--smoke", action="store_true",
        help=("Smoke run: restrict training to 2,000 images and run one epoch. "
              "Use to time a batch and project full GPU-hour cost."),
    )
    parser.add_argument(
        "--center-crop", action="store_true",
        help="Center-crop mode (deprecated; kept for backward compatibility).",
    )
    args = parser.parse_args()

    phase_kwargs = dict(
        arm=args.arm,
        data_root=args.data_root,
        ckpt_dir=args.ckpt_dir,
        report_dir=args.report_dir,
        dino_path=args.dino_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoke=args.smoke,
        center_crop_mode=args.center_crop,
        phase_override_name=args.phase_name,
        img_size=args.img_size,
        n_epochs=args.n_epochs,
        training_seed=args.seed,
    )

    if args.phase == "all":
        # Full Gen-2 ladder: phases run in sequence, each inheriting the previous
        # phase's final-epoch weights.  Stops early if a phase fails or is
        # interrupted (returns status != 'complete').
        for p in PHASE_LADDER:
            res = run_gen2_phase(phase=p, **phase_kwargs)
            if res.get("status") != "complete":
                print(f"\nLadder halted after phase '{p}' (status={res.get('status')}).")
                break
    else:
        run_gen2_phase(phase=args.phase, **phase_kwargs)


if __name__ == "__main__":
    main()

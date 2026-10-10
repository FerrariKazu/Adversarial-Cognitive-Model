"""Gen-2 Foundation Trainer — training/train_gen2_foundation.py
================================================================================

Comprehensive Gen-2 Foundation ladder runner implementing S0-S7:
- Targets: Kaggle T4 & Colab T4. Single GPU.
- fp16 autocast + GradScaler, with PGD-KL attack evaluated OUTSIDE autocast.
- Phase g2_gist_only: 30 epochs (10 clean + 20-epoch TRADES ramp).
- Downstream phases: recurrence_only, belief_no_f, belief_with_f (arm v1 & v2),
  ais_v2_swap, gen1_core. 15 epochs each, inheriting previous phase's FINAL-epoch weights.
- AdamW multi-group optimizer with per-group clip norm and warmup/cosine.
- Epoch health flags (CHANCE-LEVEL, CLIP-SATURATED, TRADES-DEAD, STALL, ROBUST>CLEAN).
- Feature drift monitoring every 5 epochs.
- Clean exit at 11h elapsed to prevent cloud session cut corruption.
- Durable JSONL append-only log with resume verification.
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
from typing import Any, Dict, List, Optional, Tuple

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

PHASE_LADDER = [
    "g2_gist_only",
    "recurrence_only",
    "belief_no_f",
    "belief_with_f",
    "ais_v2_swap",
    "gen1_core",
]

PHASE_TOTAL_EPOCHS = {
    "g2_gist_only": 30,
    "recurrence_only": 15,
    "belief_no_f": 15,
    "belief_with_f": 15,
    "ais_v2_swap": 15,
    "gen1_core": 15,
}

GATE_W_REFERENCE_ACC = 0.5852  # Gate W passed score


# ── PGD-KL Attack (Gen-0 mechanics, evaluated strictly outside autocast) ─────
def run_pgd_kl_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    steps: int,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Builds x_adv outside autocast in fp32 with KL(x_adv || clean)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probs_clean = F.softmax(model(x).float(), dim=1)

        x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        if alpha is None or alpha <= 0.0:
            alpha = eps / max(steps, 1)

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


# ── PGD Cross-Entropy Attack for Robust Evaluation ─────────────────────────
def run_pgd_ce_eval(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        alpha = eps / max(steps, 1) if steps > 0 else 0.0

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


# ── TRADES Loss ─────────────────────────────────────────────────────────────
def compute_trades_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    x_adv: torch.Tensor,
    beta: float,
    w_trades: float,
) -> Tuple[torch.Tensor, float, float]:
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


# ── Light Augmentation Dataset Loaders ─────────────────────────────────────
def get_gen2_loaders(
    data_root: str,
    batch_size: int = 48,
    num_workers: int = 4,
    smoke: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(56, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((56, 56)),
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


# ── Optimizer Construction with AdamW Multi-Group Registry ──────────────────
def build_gen2_optimizer(
    model: Gen2FoundationModel,
    warmup_epochs: int = 3,
    total_epochs: int = 30,
    trunk_lr: float = 5e-5,
    new_lr: float = 5e-4,
    weight_decay: float = 0.05,
    clip_norm: float = 1.0,
) -> Tuple[torch.optim.Optimizer, Any]:
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


# ── Per-Group Gradient Clipping with Pre/Post Norm Recording ────────────────
def clip_per_group(optimizer: torch.optim.Optimizer) -> Tuple[Dict[str, float], bool]:
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


# ── Feature Drift Monitor ───────────────────────────────────────────────────
def compute_feature_drift(
    model: Gen2FoundationModel,
    initial_trunk_sd: Dict[str, torch.Tensor],
    drift_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
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


# ── Full Phase Training Loop ────────────────────────────────────────────────
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
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n[Gen-2 Foundation] Launching Phase: '{phase}' (arm={arm}, device={device})")
    print_curriculum_header()

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    jsonl_path = os.path.join(report_dir, "gen2_epoch_log.jsonl")

    total_epochs = PHASE_TOTAL_EPOCHS.get(phase, 15)
    train_loader, val_loader, rob_loader, drift_loader = get_gen2_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers, smoke=smoke
    )

    model = Gen2FoundationModel(
        phase=phase,
        arm=arm,
        fovea_size=DEFAULT_FOVEA_SIZE,
        center_crop_mode=center_crop_mode,
    ).to(device)

    # Weights initialization / inheritance
    if phase == "g2_gist_only":
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
    rolling_path = os.path.join(ckpt_dir, f"foundation_{phase}_rolling.pth")
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

        for x, y in train_loader:
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

        scheduler.step()

        # Validation Clean (Full 5,000 images)
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
        v_rob_correct = 0
        v_rob_total = 0
        rob_eps = curric.eps_norm if curric.adversarial else 0.031
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
        eta_min = (rem_epochs * ep_duration) / 60.0

        # Labeled epoch block
        print("-" * 70)
        print(f"[{phase}] Epoch {epoch}/{total_epochs} ({ep_duration:.1f}s, VRAM: {vram_mb:.0f}MB, ETA: {eta_min:.1f}m)")
        print(f"  val_acc[CLEAN]:  {val_clean_acc*100:.2f}% (best: {max(best_clean_acc, val_clean_acc)*100:.2f}%)")
        print(f"  val_acc[ROBUST]: {val_rob_acc*100:.2f}% (eps={rob_eps:.3f}, best: {max(best_rob_acc, val_rob_acc)*100:.2f}%)")
        print(f"  loss: total={tot_loss/max(n_samples,1):.4f} ce={tot_ce/max(n_samples,1):.4f} kl={tot_kl/max(n_samples,1):.4f}")
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
            torch.save(ckpt_state, os.path.join(ckpt_dir, f"foundation_{phase}_best_clean.pth"))

        if val_rob_acc > best_rob_acc:
            best_rob_acc = val_rob_acc
            torch.save(ckpt_state, os.path.join(ckpt_dir, f"foundation_{phase}_best_robust.pth"))

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
    final_path = os.path.join(ckpt_dir, f"foundation_{phase}_final.pth")
    torch.save(ckpt_state, final_path)
    print(f"\n✓ Phase '{phase}' COMPLETE. Final checkpoint saved to: {final_path}")
    print(f"  best_clean: {best_clean_acc*100:.2f}%, best_robust: {best_rob_acc*100:.2f}%")
    return {
        "status": "complete", "phase": phase,
        "best_clean": best_clean_acc, "best_robust": best_rob_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Gen-2 Foundation Training Pipeline")
    parser.add_argument("--phase", type=str, default="g2_gist_only", choices=PHASE_LADDER + ["all"])
    parser.add_argument("--arm", type=str, default="v1", choices=["v1", "v2"], help="Arm for belief_with_f")
    parser.add_argument("--data-root", type=str, default="data/imagenet100")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--report-dir", type=str, default="report")
    parser.add_argument("--dino-path", type=str, default="checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Smoke run on 2k subset to measure and project GPU hours")
    parser.add_argument("--center-crop", action="store_true", help="Center-crop mode flag (disabled by default)")
    args = parser.parse_args()

    if args.phase == "all":
        for p in PHASE_LADDER:
            res = run_gen2_phase(
                phase=p, arm=args.arm, data_root=args.data_root, ckpt_dir=args.ckpt_dir,
                report_dir=args.report_dir, dino_path=args.dino_path, batch_size=args.batch_size,
                num_workers=args.num_workers, smoke=args.smoke, center_crop_mode=args.center_crop
            )
            if res.get("status") != "complete":
                break
    else:
        run_gen2_phase(
            phase=args.phase, arm=args.arm, data_root=args.data_root, ckpt_dir=args.ckpt_dir,
            report_dir=args.report_dir, dino_path=args.dino_path, batch_size=args.batch_size,
            num_workers=args.num_workers, smoke=args.smoke, center_crop_mode=args.center_crop
        )


if __name__ == "__main__":
    main()

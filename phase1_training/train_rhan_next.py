#!/usr/bin/env python3
"""
train_rhan_next.py — RHAN-Next training entrypoint (strict superset of
train_rhan_v12.py).
=====================================================================

The trainer is a strict superset of train_rhan_v12.py, NOT a divergent
codepath: identical data pipeline (STL-10 real + pseudo + synthetic mixes),
identical curriculum (3 phases), identical warmup freeze schedule, identical
HF rolling-checkpoint resume gates, identical diagnostics. The differences,
all gated behind RHANNextConfig toggles (OFF by default = exactly v12):

  * --enable-ais (Pillar 2, Stage 1):
        - loss gains a precision-modulated reconstruction weight
          w_recon * (0.5 + Pi_D * gain) — the GlobalPrecisionModulator's
          reconstruction-loss consumer;
        - the gaze update and halting go through InformationGainGazePolicy /
          EntropyGatedHalting (no step-count penalty — see
          tests/test_gradient_flow.py::test_no_step_count_penalty_in_loss_path).
  * --enable-hpc / --hpc-num-levels (Pillar 1, Stage 2):
        - loss gains w_hpc * L_hpc (mean hierarchical prediction error).

Loss (pillars on):
    L = w_trades * L_trades + w_recon_eff * L_recon + w_hpc * L_hpc
where w_recon_eff = w_recon when AIS is off.

Checkpoints:
    *_best.pth    -> {'model': state_dict, 'config': RHANNextConfig dict, 'arch': 'rhan_next'}
    *_rolling.pth -> v12's resume dict + 'config'
The embedded config lets phase2_attacks/eval_rhan.py reconstruct the exact
pillar config without any external bookkeeping.

Frozen files: model_rhan_v12.py / eval conventions are never touched.
"""

import os
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import gc
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

# ── Reuse the frozen v12 pipeline (data, pseudo-labeling, HF, diagnostics) ──
from train_rhan_v12 import (
    load_dotenv_fallback,
    set_seed,
    STL10RawUnlabeledDataset,
    CombinedSTL10Dataset,
    BalancedBatchSampler,
    generate_pseudo_labels,
    find_optimal_dataloader_config,
    ensure_checkpoint_exists,
    sync_to_hf,
    wait_for_hf_sync,
    EpochDiagnostics,
    get_stl10_dataloaders as _unused,  # imported below via tdv
)
from train_rhan_stl10_tdv import get_stl10_dataloaders

load_dotenv_fallback()

# Components frozen during the warmup phase (v12 list + new pillar modules).
_WARMUP_FROZEN_FRAGMENTS = [
    'foveal_stream', 'precision_ctrl', 'action_init', 'parafoveal_stream',
    'foveal_gate', 'generative_prior', 'image_precision',
    'gaze_policy', 'precision_modulator', 'hpc_stack',
]


def set_new_component_training(model, trainable):
    """Freeze/unfreeze active-inference + pillar components (warmup schedule)."""
    for name, param in model.named_parameters():
        if any(x in name for x in _WARMUP_FROZEN_FRAGMENTS):
            param.requires_grad = trainable
        else:
            param.requires_grad = True


def dynamic_trades_loss_next(model, imgs, labels, weights, x_adv,
                             beta_base, w_recon, w_hpc):
    """
    The RHAN-Next loss (superset of v12's two-term loss).

        L = w_trades * L_trades + w_recon_eff * L_recon + w_hpc * L_hpc

    with w_recon_eff = w_recon * mean(0.5 + Pi_D * gain) when the precision
    modulator exists (AIS), else w_recon. L_hpc = 0 when HPC is off.
    No step-count penalty term exists anywhere in this function.
    """
    logits_c, traj_c = model(imgs, return_trajectory=True)
    logits_a, traj_a = model(x_adv, return_trajectory=True)

    # Per-image dynamic beta from precision (Pi_D forward pass retained).
    if len(traj_c['precisions']) > 0:
        final_precision_c = traj_c['precisions'][-1]        # (B,)
    else:
        final_precision_c = torch.full((imgs.shape[0],), 0.5, device=imgs.device)

    beta_dynamic = beta_base * (0.5 + final_precision_c)    # [beta/2, 1.5*beta]

    # TRADES robustness term (identical to v12).
    ce = nn.CrossEntropyLoss(reduction='none')
    l_ce = ce(logits_c, labels)
    l_kl = F.kl_div(
        F.log_softmax(logits_a.float(), dim=1),
        F.softmax(logits_c.float().detach(), dim=1),
        reduction='none').sum(dim=1)
    l_trades = ((l_ce + beta_dynamic * l_kl) * weights.to(l_ce.device)).mean()

    # Reconstruction loss for the generative prior (v12 fix: differentiable).
    l_recon = 0.5 * (
        model.get_reconstruction_loss(imgs, (logits_c, traj_c))
        + model.get_reconstruction_loss(x_adv, (logits_a, traj_a)))

    # HPC prediction-error loss (0.0 when HPC is off).
    l_hpc = 0.5 * (
        model.get_hpc_loss(imgs, (logits_c, traj_c))
        + model.get_hpc_loss(x_adv, (logits_a, traj_a)))

    # Precision-modulated recon weight (AIS consumer, gain-scaled).
    modulator = getattr(model, 'precision_modulator', None)
    if modulator is not None:
        w_recon_eff = modulator.modulate_recon_weight(w_recon, final_precision_c)
    else:
        w_recon_eff = torch.tensor(w_recon, device=imgs.device)

    return (l_trades, traj_c, traj_a, beta_dynamic.detach(),
            l_recon, l_hpc, w_recon_eff)


# ────────────────────────────────────────────────────────────────────────────
# Curriculum + data prep (identical to v12)
# ────────────────────────────────────────────────────────────────────────────

CURRICULUM = [
    (1,  20, 0.031, 2.0, 4,  0.003),
    (21, 40, 0.062, 2.0, 4,  0.002),
    (41, 60, 0.094, 2.5, 4,  0.001),
]


def build_config(args) -> RHANNextConfig:
    cfg = RHANNextConfig(
        enable_ais=args.enable_ais,
        enable_hpc=args.enable_hpc,
        hpc_num_levels=args.hpc_num_levels,
        max_foraging_steps=args.max_foraging_steps,
        fovea_size=args.fovea_size,
        metabolic_cost=args.metabolic_cost,
        gaze_lambda=args.gaze_lambda,
        ais_halt_threshold=args.ais_halt_threshold,
        ais_continuation_softness=args.ais_continuation_softness,
        hpc_error_weight=args.w_hpc,
    )
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='RHAN-Next training (superset of train_rhan_v12.py)')
    # ── v12 flags (unchanged) ────────────────────────────────────────────────
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--unlabeled-batch-size', type=int, default=256)
    parser.add_argument('--accum-steps', type=int, default=32)
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--labeling-ckpt', type=str, default='')
    parser.add_argument('--target-ckpt', type=str, default='')
    parser.add_argument('--fixed-samples-per-epoch', type=int, default=0)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force-restart', action='store_true')
    parser.add_argument('--max-foraging-steps', type=int, default=4)
    parser.add_argument('--fovea-size', type=int, default=48)
    parser.add_argument('--metabolic-cost', type=float, default=0.05)
    parser.add_argument('--w-trades', type=float, default=0.55)
    parser.add_argument('--w-recon', type=float, default=0.10)
    parser.add_argument('--gaze-lambda', type=float, default=0.5)
    parser.add_argument('--synthetic-data', type=str, default='')
    parser.add_argument('--ckpt-name', type=str, default='rhan_stl10_next')
    parser.add_argument('--no-pseudo', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--freeze-gaze', action='store_true')
    parser.add_argument('--force-single-gpu', action='store_true')
    # ── RHAN-Next pillar flags ───────────────────────────────────────────────
    parser.add_argument('--enable-ais', action='store_true',
                        help='Pillar 2: info-gain gaze + entropy-gated halting '
                             '+ precision-modulated recon weight (Stage 1)')
    parser.add_argument('--enable-hpc', action='store_true',
                        help='Pillar 1: hierarchical predictive coding (Stage 2)')
    parser.add_argument('--hpc-num-levels', type=int, default=1,
                        help='HPC levels (1 implemented; never jump levels)')
    parser.add_argument('--w-hpc', type=float, default=0.05,
                        help='HPC prediction-error loss weight')
    parser.add_argument('--ais-halt-threshold', type=float, default=0.35,
                        help='EntropyGatedHalting: halt when uncertainty < this')
    parser.add_argument('--ais-continuation-softness', type=float, default=8.0)
    args, _ = parser.parse_known_args()

    # ── Environment / device ────────────────────────────────────────────────
    is_ddp = "WORLD_SIZE" in os.environ and "RANK" in os.environ
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    cfg = build_config(args)

    if rank == 0:
        print(f"{'═'*60}")
        print(f"  RHAN-Next Training (superset of v12)")
        print(f"  Device: {device} | DDP: {is_ddp} (world_size={world_size})")
        print(f"  Config: {cfg}")
        if cfg.enable_ais:
            print(f"    AIS: halt_threshold={cfg.ais_halt_threshold}, "
                  f"softness={cfg.ais_continuation_softness}")
        if cfg.enable_hpc:
            print(f"    HPC: levels={cfg.hpc_num_levels}, "
                  f"w_hpc={cfg.hpc_error_weight}, "
                  f"targets={cfg.hpc_num_levels and 'edge_map'}")
        print(f"  Loss weights: trades={args.w_trades}, recon={args.w_recon}"
              + (f", hpc={args.w_hpc}" if cfg.enable_hpc else ""))
        print(f"  Max epochs: {args.max_epochs}")
        print(f"{'═'*60}", flush=True)

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── 1. Pseudo-labels (unless --no-pseudo) — identical to v12 ────────────
    pseudo_indices = pseudo_lbls = None
    if not args.no_pseudo:
        unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
        if rank == 0:
            from model_rhan_stl10_pretrained import RHANUnifiedSTL10
            labeling_model = RHANUnifiedSTL10().to(device, memory_format=torch.channels_last)
            best_labeling_ckpt = args.labeling_ckpt or os.path.join(
                ckpt_dir, 'rhan_stl10_pseudolabel_best.pth')
            best_labeling_ckpt = ensure_checkpoint_exists(best_labeling_ckpt)
            if os.path.exists(best_labeling_ckpt):
                from checkpoint_utils import compat_load
                labeling_model.load_state_dict(
                    compat_load(best_labeling_ckpt, map_location=device))
            else:
                print("Error: labeling checkpoint not found!", flush=True)
                sys.exit(1)
            num_workers = min(4, os.cpu_count() or 2)
            unlabeled_loader = torch.utils.data.DataLoader(
                unlabeled_dataset, batch_size=args.unlabeled_batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True)
            pseudo_indices, pseudo_lbls, _ = generate_pseudo_labels(
                labeling_model, unlabeled_loader, device, args.confidence_threshold)
            del labeling_model
            torch.cuda.empty_cache()
            gc.collect()
        if is_ddp:
            import torch.distributed as dist
            dist.barrier()
        if len(pseudo_indices) == 0:
            if rank == 0:
                print("Error: No pseudo-labels generated. Exiting.", flush=True)
            sys.exit(1)
    else:
        if rank == 0:
            print("--no-pseudo active: real (+ synthetic only).", flush=True)

    # ── 2/3. Data prep — identical to v12 ───────────────────────────────────
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))])
    trainset_raw = torchvision.datasets.STL10(args.data_root, split='train', download=True)
    num_real = len(trainset_raw)
    real_imgs = torch.zeros(num_real, 3, 96, 96, dtype=torch.float32)
    for i in range(num_real):
        real_imgs[i] = norm_transform(trainset_raw[i][0])
    real_labels = torch.tensor([trainset_raw[i][1] for i in range(num_real)])

    synth_imgs = synth_labels = None
    if args.synthetic_data and os.path.exists(args.synthetic_data):
        print(f"Loading synthetic data from {args.synthetic_data}...", flush=True)
        synth_dict = torch.load(args.synthetic_data, map_location='cpu')
        synth_imgs = synth_dict['imgs'].to(torch.uint8).contiguous()
        synth_labels = synth_dict['labels']
        print(f"  Loaded {synth_imgs.size(0)} synthetic images", flush=True)

    train_transform = T.Compose([
        T.RandomCrop(96, padding=12), T.RandomHorizontalFlip()])
    unlabeled_dataset = None
    if not args.no_pseudo:
        unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
    combined_dataset = CombinedSTL10Dataset(
        real_imgs, real_labels, unlabeled_dataset, pseudo_indices, pseudo_lbls,
        synthetic_imgs=synth_imgs, synthetic_labels=synth_labels,
        transform=train_transform)
    del unlabeled_dataset
    gc.collect()

    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(combined_dataset)))
    if is_ddp:
        import random
        random.Random(args.seed + rank).shuffle(real_indices)
        random.Random(args.seed + rank).shuffle(pseudo_indices_list)
        real_indices = real_indices[rank::world_size]
        pseudo_indices_list = pseudo_indices_list[rank::world_size]

    sampler = BalancedBatchSampler(
        real_indices, pseudo_indices_list,
        batch_size=args.batch_size // world_size if is_ddp else args.batch_size)
    optimal_config = find_optimal_dataloader_config(combined_dataset, sampler, is_ddp, rank)
    loader_kwargs = {"pin_memory": True}
    if optimal_config["num_workers"] > 0:
        loader_kwargs.update(num_workers=optimal_config["num_workers"],
                             persistent_workers=True,
                             prefetch_factor=3)
    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, **loader_kwargs)

    _, testloader, stl_min, stl_max = get_stl10_dataloaders(
        args.data_root, batch_size=64)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    # ── 4. Model — RHANNext ─────────────────────────────────────────────────
    model = RHANNext(config=cfg).to(device, memory_format=torch.channels_last)

    # ── 5. Base checkpoint (strict=False — new pillar modules initialize) ───
    best_target_ckpt = args.target_ckpt or os.path.join(
        ckpt_dir, 'rhan_stl10_large_pseudolabel_best.pth')
    best_target_ckpt = ensure_checkpoint_exists(best_target_ckpt)
    if os.path.exists(best_target_ckpt):
        from checkpoint_utils import compat_load
        ckpt = compat_load(best_target_ckpt, map_location=device)
        for k in ('model_state_dict', 'model', 'state_dict'):
            if isinstance(ckpt, dict) and k in ckpt:
                ckpt = ckpt[k]
                break
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if rank == 0:
            print(f"Loaded base checkpoint: {best_target_ckpt}", flush=True)
            print(f"  Missing (new pillar modules): {len(missing)}", flush=True)
            print(f"  Unexpected keys: {len(unexpected)}", flush=True)
    elif rank == 0:
        print(f"Warning: base checkpoint not found — random init.", flush=True)

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"RHANNext instantiated: {total:,} params ({cfg})", flush=True)

    if args.compile and rank == 0:
        print("Compiling model with torch.compile()...", flush=True)
    if args.compile:
        model = torch.compile(model, mode="default")

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False)
    elif torch.cuda.device_count() > 1 and not args.force_single_gpu:
        if rank == 0:
            print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)", flush=True)
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, 'module') else model
    if args.freeze_gaze:
        raw_model.freeze_gaze = True
        if rank == 0:
            print("  ISOLATION TEST: gaze frozen to center (0,0)", flush=True)

    # ── 6. Curriculum / resume / optimizer — identical to v12 ───────────────
    scaler = GradScaler('cuda')
    best_acc = 0.0
    start_epoch = 1
    checkpoint_data = None
    optimizer = None
    scheduler = None
    current_phase_start = None

    best_path = os.path.join(ckpt_dir, f'{args.ckpt_name}_best.pth')
    rolling_path = os.path.join(ckpt_dir, f'{args.ckpt_name}_rolling.pth')

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
        except Exception:
            pass
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass

    # Mandatory HF resume gate (same semantics as v12 — never silently
    # restart; once a rolling checkpoint exists on HF we restore or abort).
    local_epoch = -1
    if not args.force_restart:
        if os.path.exists(rolling_path):
            try:
                from checkpoint_utils import compat_load
                local_epoch = compat_load(rolling_path, map_location='cpu').get('epoch', -1)
            except Exception:
                local_epoch = -1

        hf_rolling_exists = hf_listing_ok = False
        if rank == 0:
            try:
                from huggingface_hub import HfApi
                rolling_filename = f"{args.ckpt_name}_rolling.pth"
                hf_files = HfApi(token=hf_token).list_repo_files(
                    repo_id='FerrariKazu/rhan-checkpoints-rolling',
                    repo_type='dataset')
                hf_rolling_exists = rolling_filename in hf_files
                hf_listing_ok = True
            except Exception as e:
                print(f"Hugging Face repo listing failed: {e}", flush=True)

        if rank == 0 and hf_rolling_exists:
            print("Hugging Face has a rolling checkpoint — resume is MANDATORY.", flush=True)
            last_err = None
            for attempt in range(1, 4):
                try:
                    from huggingface_hub import hf_hub_download
                    from checkpoint_utils import compat_load
                    temp = hf_hub_download(
                        repo_id='FerrariKazu/rhan-checkpoints-rolling',
                        filename=rolling_filename, repo_type='dataset',
                        token=hf_token)
                    remote_epoch = compat_load(temp, map_location='cpu').get('epoch', -1)
                    if remote_epoch >= local_epoch:
                        os.makedirs(os.path.dirname(rolling_path), exist_ok=True)
                        shutil.copy(temp, rolling_path)
                        local_epoch = remote_epoch
                        print(f"  Synchronized HF checkpoint (Epoch {remote_epoch})", flush=True)
                    break
                except Exception as e:
                    last_err = e
                    print(f"  HF resume attempt {attempt}/3 failed: {e}", flush=True)
                    if attempt < 3:
                        time.sleep(15 * attempt)
            if not os.path.exists(rolling_path):
                print(f"\n[FATAL] {rolling_filename} exists on HF but could not be "
                      f"restored ({last_err}). Aborting instead of a silent restart.",
                      flush=True)
                sys.exit(1)
        elif rank == 0 and not hf_listing_ok and not os.path.exists(rolling_path):
            print(f"\n[FATAL] Could not verify HF rolling checkpoint — no local copy "
                  f"exists. Aborting rather than silently restarting.", flush=True)
            sys.exit(1)

        if os.path.exists(rolling_path):
            from checkpoint_utils import compat_load
            checkpoint_data = compat_load(rolling_path, map_location=device)
            raw_model.load_state_dict(checkpoint_data['model'])
            best_acc = checkpoint_data.get('best_acc', 0.0)
            start_epoch = checkpoint_data['epoch'] + 1
            if rank == 0:
                print(f"Resuming from Epoch {start_epoch} "
                      f"(best val {best_acc:.2f}%)", flush=True)
    elif rank == 0:
        print("--force-restart: starting from Epoch 1.", flush=True)

    # ── 7. Training loop ────────────────────────────────────────────────────
    WARMUP_EPOCHS = 5
    diagnostics = EpochDiagnostics()

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        diagnostics.reset()

        for p_start, p_end, eps, beta, steps, lr in CURRICULUM:
            if p_start <= epoch <= p_end:
                phase_params = (eps, beta, steps)
                phase_lr = lr
                if current_phase_start != p_start:
                    current_phase_start = p_start
                    optimizer = optim.SGD(model.parameters(), lr=phase_lr,
                                          momentum=0.9, weight_decay=1e-4,
                                          foreach=True)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=p_end - p_start + 1,
                        eta_min=phase_lr * 0.1)
                    if (epoch == start_epoch and checkpoint_data is not None
                            and 'optimizer' in checkpoint_data):
                        optimizer.load_state_dict(checkpoint_data['optimizer'])
                        scheduler.load_state_dict(checkpoint_data['scheduler'])
                        if rank == 0:
                            print("Restored optimizer/scheduler state.", flush=True)
                    if rank == 0:
                        print(f"\n--- Epoch {epoch}: phase {p_start}-{p_end} "
                              f"(lr={phase_lr}) ---", flush=True)
                break
        eps, beta, steps = phase_params

        if epoch <= WARMUP_EPOCHS:
            if rank == 0:
                print("Warmup: freezing active-inference + pillar components, "
                      "training generative prior.", flush=True)
            set_new_component_training(raw_model, False)
            for name, param in raw_model.named_parameters():
                if 'generative_prior' in name:
                    param.requires_grad = True
        else:
            if rank == 0:
                print("Main Phase: training all components", flush=True)
            set_new_component_training(raw_model, True)

        model.train()
        total_loss = n_total = correct = 0
        total_batch_size = args.batch_size * world_size if is_ddp else args.batch_size
        num_batches = (min(len(trainloader), 600) if args.fixed_samples_per_epoch <= 0
                       else max(1, args.fixed_samples_per_epoch // total_batch_size))
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (imgs, lbls, weights) in enumerate(trainloader):
            if batch_idx >= num_batches:
                break
            is_accum = ((batch_idx + 1) % args.accum_steps != 0
                        and (batch_idx + 1) < num_batches)
            if is_ddp and is_accum:
                sync_ctx = model.no_sync()
            else:
                from contextlib import nullcontext
                sync_ctx = nullcontext()
            with sync_ctx:
                imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)

                if epoch <= WARMUP_EPOCHS:
                    with autocast('cuda'):
                        logits, traj_c = model(imgs, return_trajectory=True)
                        l_trades = nn.CrossEntropyLoss()(logits, lbls)
                        l_recon = raw_model.get_reconstruction_loss(imgs, (logits, traj_c))
                        l_hpc = raw_model.get_hpc_loss(imgs, (logits, traj_c))
                        loss = (l_trades + args.w_recon * l_recon
                                + args.w_hpc * l_hpc) / args.accum_steps
                        beta_dyn = (beta * (0.5 + traj_c['precisions'][-1])
                                    if len(traj_c['precisions']) > 0
                                    else torch.full((imgs.shape[0],), beta, device=device))
                    scaler.scale(loss).backward()
                    diagnostics.update(beta_dyn, traj_c, lbls)
                else:
                    # ── PGD adversarial examples (identical to v12) ─────────
                    raw_model.eval()
                    with torch.no_grad():
                        with autocast('cuda'):
                            probs_c = F.softmax(raw_model(imgs).float(), dim=1)
                    x_adv = torch.clamp(
                        imgs.clone().detach() + 0.001 * torch.randn_like(imgs),
                        stl_min, stl_max)
                    for _ in range(steps):
                        x_adv.requires_grad_(True)
                        with torch.enable_grad():
                            with autocast('cuda'):
                                logits_a_pgd = raw_model(x_adv)
                                loss_adv = F.kl_div(
                                    F.log_softmax(logits_a_pgd.float(), dim=1),
                                    probs_c, reduction='batchmean')
                        grad = torch.autograd.grad(loss_adv, x_adv)[0]
                        x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                        x_adv = torch.clamp(
                            imgs + torch.clamp(x_adv - imgs, -eps, eps),
                            stl_min, stl_max).detach()
                    model.train()

                    # ── RHAN-Next loss (superset of v12) ────────────────────
                    with autocast('cuda'):
                        (l_trades, traj_c, traj_a, beta_dyn, l_recon, l_hpc,
                         w_recon_eff) = dynamic_trades_loss_next(
                            raw_model, imgs, lbls, weights, x_adv, beta,
                            args.w_recon, args.w_hpc)
                        loss = (args.w_trades * l_trades
                                + w_recon_eff * l_recon
                                + args.w_hpc * l_hpc) / args.accum_steps
                    scaler.scale(loss).backward()
                    diagnostics.update(beta_dyn, traj_c, lbls)

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            B = imgs.size(0)
            total_loss += l_trades.item() * B
            with torch.no_grad():
                with autocast('cuda'):
                    logits_c_acc = model(imgs)
            correct += logits_c_acc.argmax(1).eq(lbls).sum().item()
            n_total += B

            if rank == 0 and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches} | "
                      f"Loss: {l_trades.item():.4f} | β_dyn: {beta_dyn.mean():.3f} "
                      f"| Steps: {traj_c['steps']}", flush=True)
            if args.dry_run and rank == 0:
                print("Dry-run: 1 training step OK.", flush=True)
                break

        if num_batches % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ── Validation (clean test) ─────────────────────────────────────────
        val_acc = 0.0
        if rank == 0 and not args.dry_run:
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for v_imgs, v_lbls in testloader:
                    v_imgs, v_lbls = v_imgs.to(device), v_lbls.to(device)
                    with autocast('cuda'):
                        logits = model(v_imgs)
                    val_correct += logits.argmax(1).eq(v_lbls).sum().item()
                    val_total += v_lbls.size(0)
            val_acc = 100.0 * val_correct / val_total

        if is_ddp:
            import torch.distributed as dist
            va = torch.tensor([val_acc], device=device)
            dist.broadcast(va, src=0)
            val_acc = va.item()

        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            marker = ' ★'
            if rank == 0:
                torch.save({'model': raw_model.state_dict(),
                            'config': cfg.to_dict(),
                            'arch': 'rhan_next'}, best_path)
                sync_to_hf(best_path)

        if rank == 0:
            t_epoch = time.time() - t0
            total_images = n_total * world_size if is_ddp else n_total
            ips = total_images / t_epoch if t_epoch > 0 else 0
            eph = 3600.0 / t_epoch if t_epoch > 0 else 0
            print(f"Epoch {epoch:03d}/{args.max_epochs:03d} (ε={eps:.3f}) | "
                  f"Loss:{total_loss/max(n_total,1):.3f} | "
                  f"TrAcc:{100.*correct/max(n_total,1):.1f}% TeAcc:{val_acc:.1f}% | "
                  f"Throughput:{ips:.2f} img/sec ({eph:.2f} epochs/hour) | "
                  f"{t_epoch:.0f}s{marker}", flush=True)
            diagnostics.report(epoch, eps)

            torch.save({'epoch': epoch,
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc': best_acc,
                        'config': cfg.to_dict(),
                        'arch': 'rhan_next'}, rolling_path)
            sync_to_hf(rolling_path)
            gc.collect()
            torch.cuda.empty_cache()

        if args.dry_run:
            break
        if is_ddp:
            import torch.distributed as dist
            dist.barrier()

    if rank == 0:
        print("\nFinalizing Hugging Face sync...", flush=True)
        sync_to_hf(best_path)
        wait_for_hf_sync()
        print(f"{'═'*60}")
        print(f"  Training complete. Best: {best_acc:.2f}% -> {best_path}")
        print(f"  Config: {cfg}")
        print(f"{'═'*60}")


if __name__ == '__main__':
    main()

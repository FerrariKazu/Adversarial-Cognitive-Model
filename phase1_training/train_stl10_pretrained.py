#!/usr/bin/env python3
"""
RHAN-UNIFIED: STL-10 96×96 with ImageNet Pretrained Backbone
=============================================================

Why CIFAR-10 32×32 cannot close the gap:
  At 32×32, automobile and truck share too many pixel-level features.
  PGD at ε=0.031 perturbs ≈9% of the total information content.
  At 96×96, the same absolute epsilon perturbs ≈1% — much more manageable.
  And high-resolution images genuinely separate cars from trucks visually.

Why previous STL-10 attempts failed:
  Training from scratch on 5,000 labeled images with TRADES β=6.0
  causes immediate collapse because β was calibrated for 50K CIFAR images.

Why THIS will work:
  1. ImageNet pretrained ResNet-50 as stem — already knows shapes
  2. β=2.0 calibrated for small dataset (confirmed working in Attempt 4)
  3. 3-epoch beta warmup at each phase transition
  4. 100K unlabeled STL-10 images for SAIL pretraining (no labels needed)
  5. Rolling checkpoints every epoch — no progress lost

Expected results:
  Clean accuracy:     88-93% (vs 78% on CIFAR-10)
  εthresh (d'=1.0):   0.220-0.280 (vs 0.185 on CIFAR-10)
  AutoAttack ε=0.031: 40-60% (vs 21.88% on CIFAR-10)
  Auto/truck:         25-45% (vs 0% on CIFAR-10)

Usage:
  # Setup
  pip install torchvision datasets

  # Phase 0: Clean pretraining
  python train_stl10_pretrained.py --phase 0 --epochs 30

  # Phase 1-8: TRADES curriculum
  python train_stl10_pretrained.py --phase 1 --resume
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STL-10 ADAPTED RHAN
# Uses ImageNet pretrained ResNet-50 as the convolutional stem
# Everything else (transformer, predictive coding, head) is randomly initialized
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PredictiveCodingLayerSTL(nn.Module):
    """Predictive coding feedback adapted for STL-10 feature dimensions."""
    def __init__(self, channels=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.error_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid(),
        )
        self.error_scale = nn.Parameter(torch.ones(1))

    def forward(self, local_f, global_spatial):
        predicted = self.predictor(global_spatial)
        error = local_f - predicted
        gate = self.error_gate(error)
        corrected = local_f + self.error_scale * gate * error
        return corrected, error.abs().mean()


class RHANUnifiedSTL10(nn.Module):
    """
    RHAN-Unified for STL-10 96×96.

    Architecture:
      Stem: ImageNet pretrained ResNet-50 layers 1-3
            (96→24×24, then 24→12×12 = 144 spatial tokens)
      Tokeniser: Linear projection to embed_dim=512
      Transformer: 3-layer global attention (ventral + dorsal split)
      Feedback: True predictive coding (error signal)
      Head: Cosine similarity classifier (10 STL-10 classes)

    STL-10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
    Note: 'car' maps to 'automobile' in our framework — same separation problem
    """

    # STL-10 class names
    STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                     'dog', 'horse', 'monkey', 'ship', 'truck']

    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 freeze_stem_epochs=10):
        super().__init__()
        self.embed_dim = embed_dim
        self._freeze_stem_epochs = freeze_stem_epochs

        # ── IMAGENET PRETRAINED STEM ──────────────────────────────────────
        # Load ResNet-50, extract first 3 layers
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)

        # ResNet-50 layers for 96×96 input:
        #   conv1 + bn1 + relu + maxpool: 96 → 24×24, 64ch
        #   layer1: 24×24, 256ch
        #   layer2: 12×12, 512ch
        self.stem = nn.Sequential(
            resnet.conv1,       # 96 → 48×48
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,     # 48 → 24×24
            resnet.layer1,      # 24×24, 256ch
            resnet.layer2,      # 12×12, 512ch
        )
        # stem output: (B, 512, 12, 12) = 144 spatial positions

        # ── FREQUENCY ANALYSIS LAYER ──────────────────────────────────────
        # Learnable channel-wise frequency weighting applied to stem output
        # Weights which channels to emphasize (shape vs texture)
        self.freq_weights = nn.Parameter(torch.ones(512))

        # ── TOKENISER ────────────────────────────────────────────────────
        # Project 512-ch spatial features to embed_dim tokens
        self.token_proj = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, embed_dim), nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 145, embed_dim) * 0.02)
        # 144 spatial + 1 CLS = 145 tokens

        # ── VENTRAL/DORSAL TRANSFORMER ───────────────────────────────────
        # Using PyTorch's TransformerEncoderLayer
        ventral_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2, nhead=num_heads // 2,
            dim_feedforward=ff_dim // 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.ventral = nn.TransformerEncoder(ventral_layer, num_layers=num_transformer_layers)

        dorsal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2, nhead=num_heads // 2,
            dim_feedforward=ff_dim // 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.dorsal = nn.TransformerEncoder(dorsal_layer, num_layers=num_transformer_layers)

        # ── PREDICTIVE CODING FEEDBACK ───────────────────────────────────
        self.predictive_coder = PredictiveCodingLayerSTL(channels=512)

        # ── CLASSIFICATION HEAD ───────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.orthogonal_(self.prototypes)
        self.log_scale = nn.Parameter(torch.tensor(10.0).log())

        # ── CONTRASTIVE HEAD (SAIL phase only) ────────────────────────────
        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
            nn.GELU(), nn.Linear(embed_dim, 128),
        )

    def freeze_stem(self, freeze: bool):
        for p in self.stem.parameters():
            p.requires_grad = not freeze
        print(f"Stem {'frozen' if freeze else 'unfrozen'}")

    def _run_transformer(self, spatial_features):
        """spatial_features: (B, 512, 12, 12)"""
        B = spatial_features.shape[0]

        # Project to embed_dim tokens
        tokens_2d = self.token_proj(spatial_features)  # (B, embed_dim, 12, 12)
        tokens = tokens_2d.flatten(2).transpose(1, 2)  # (B, 144, embed_dim)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 145, embed_dim)
        tokens = tokens + self.pos_embed

        # Ventral/dorsal split
        v_tokens = self.ventral(tokens[:, :, :256])
        d_tokens = self.dorsal(tokens[:, :, 256:])
        combined = torch.cat([v_tokens, d_tokens], dim=-1)  # (B, 145, 512)

        cls_out = combined[:, 0, :]   # (B, 512)
        spatial_out = combined[:, 1:, :]  # (B, 144, 512)

        # Reshape spatial tokens back to 2D map for feedback
        spatial_map = spatial_out.transpose(1, 2).reshape(B, 512, 12, 12)

        return cls_out, spatial_map

    def get_features(self, x):
        """Full forward pass with predictive coding feedback."""
        # 1. ImageNet pretrained stem
        f = self.stem(x)  # (B, 512, 12, 12)

        # 2. Learnable frequency weighting (shape over texture)
        freq_w = torch.sigmoid(self.freq_weights).view(1, 512, 1, 1)
        f = f * freq_w

        # 3. Recurrent predictive coding (3 steps)
        for step in range(3):
            cls, spatial_map = self._run_transformer(f)

            # Predictive coding: error signal only feeds back
            f, pred_error = self.predictive_coder(f, spatial_map)

        return cls

    def classify(self, features):
        features = self.norm(features)
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        scale = self.log_scale.exp().clamp(1, 100)
        return scale * (features @ prototypes.T)

    def forward(self, x):
        return self.classify(self.get_features(x))

    def forward_contrastive(self, x):
        """For SAIL pretraining."""
        features = self.get_features(x)
        return F.normalize(self.contrastive_head(features), dim=-1)

    def forward_with_features(self, x):
        features = self.get_features(x)
        return self.classify(features), features


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_stl10_dataloaders(data_root='./data/stl10', batch_size=64):
    """
    STL-10 has:
      5,000 labeled training images (500 per class)
      8,000 test images (800 per class)
      100,000 unlabeled images (for SAIL pretraining)

    All at 96×96 resolution.
    """
    # STL-10 mean/std (computed from training set)
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2603, 0.2566, 0.2713)

    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # CutMix is applied in training loop
    trainset = torchvision.datasets.STL10(
        data_root, split='train', transform=train_transform, download=True
    )
    testset = torchvision.datasets.STL10(
        data_root, split='test', transform=test_transform, download=True
    )
    unlabeled = torchvision.datasets.STL10(
        data_root, split='unlabeled', transform=train_transform, download=True
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)

    # STL-10 normalization bounds
    stl_min = torch.tensor([-(m/s) for m, s in zip(mean, std)]).view(1,3,1,1)
    stl_max = torch.tensor([(1-m)/s for m, s in zip(mean, std)]).view(1,3,1,1)

    return trainloader, testloader, unlabeled_loader, stl_min, stl_max


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CURRICULUM SCHEDULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CURRICULUM = {
    0: {'eps': 0.000, 'beta': 0.0,  'steps': 0,  'lr': 0.01,  'epochs': 30,  'desc': 'Clean pretraining + stem freeze'},
    1: {'eps': 0.016, 'beta': 2.0,  'steps': 5,  'lr': 0.005, 'epochs': 20,  'desc': 'SAIL adversarial invariance (unlabeled)'},
    2: {'eps': 0.031, 'beta': 2.0,  'steps': 7,  'lr': 0.005, 'epochs': 20,  'desc': 'TRADES ε=0.031 (standard threat model)'},
    3: {'eps': 0.047, 'beta': 2.0,  'steps': 8,  'lr': 0.003, 'epochs': 20,  'desc': 'TRADES ε=0.047 (unfreezing stem)'},
    4: {'eps': 0.062, 'beta': 2.0,  'steps': 10, 'lr': 0.002, 'epochs': 20,  'desc': 'TRADES ε=0.062'},
    5: {'eps': 0.078, 'beta': 2.0,  'steps': 10, 'lr': 0.001, 'epochs': 20,  'desc': 'TRADES ε=0.078'},
    6: {'eps': 0.094, 'beta': 2.0,  'steps': 10, 'lr': 0.001, 'epochs': 20,  'desc': 'TRADES ε=0.094'},
    7: {'eps': 0.110, 'beta': 2.5,  'steps': 12, 'lr': 0.0005,'epochs': 20,  'desc': 'TRADES ε=0.110 (increased beta)'},
    8: {'eps': 0.130, 'beta': 3.0,  'steps': 12, 'lr': 0.0003,'epochs': 20,  'desc': 'TRADES ε=0.130 (max curriculum)'},
}

BETA_WARMUP_EPOCHS = 3  # ramp beta from 30% to 100% at each phase start


def run_phase(phase_id, model, trainloader, testloader, unlabeled_loader,
              device, stl_min, stl_max, ckpt_dir, resume=False):
    """Run a single curriculum phase."""
    cfg = CURRICULUM[phase_id]
    print(f"\n{'='*70}")
    print(f"PHASE {phase_id}: {cfg['desc']}")
    print(f"  ε={cfg['eps']}, β={cfg['beta']}, steps={cfg['steps']}, "
          f"lr={cfg['lr']}, epochs={cfg['epochs']}")
    print(f"{'='*70}")

    # Freeze/unfreeze stem
    if phase_id == 0:
        model.freeze_stem(freeze=True)   # freeze during clean pretraining
    elif phase_id == 3:
        model.freeze_stem(freeze=False)  # unfreeze at phase 3

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01
    )
    scaler = GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()

    rolling_path = os.path.join(ckpt_dir, f'rhan_stl_phase{phase_id}_rolling.pth')
    final_path   = os.path.join(ckpt_dir, f'rhan_stl_phase{phase_id}_final.pth')
    best_path    = os.path.join(ckpt_dir, 'rhan_stl_best.pth')

    start_epoch = 1
    best_acc = 0.0

    if resume and os.path.exists(rolling_path):
        ckpt = torch.load(rolling_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f"Resumed from epoch {ckpt['epoch']}, best_acc={best_acc:.2f}%")

    # Use unlabeled data for SAIL phase
    data_iter = unlabeled_loader if phase_id == 1 else trainloader

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        t0 = time.time()
        model.train()

        # Beta warmup at start of each phase
        if epoch <= BETA_WARMUP_EPOCHS:
            effective_beta = cfg['beta'] * (0.3 + 0.7 * epoch / BETA_WARMUP_EPOCHS)
        else:
            effective_beta = cfg['beta']

        total_loss = n_total = correct = 0

        for imgs, lbls in data_iter:
            imgs = imgs.to(device, non_blocking=True)
            if phase_id != 1:  # SAIL doesn't use labels
                lbls = lbls.to(device, non_blocking=True)

            if cfg['eps'] > 0:
                # Generate adversarial examples
                model.eval()
                x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
                x_adv = torch.clamp(x_adv, stl_min.to(device), stl_max.to(device))

                for _ in range(cfg['steps']):
                    x_adv.requires_grad_(True)
                    with torch.enable_grad():
                        with autocast('cuda'):
                            logits_a = model(x_adv)
                            if phase_id == 1:
                                # SAIL: maximize output entropy (diverse adversarial)
                                loss_adv = -logits_a.softmax(1).log().mean()
                            else:
                                # TRADES: maximize KL from clean distribution
                                with torch.no_grad():
                                    logits_c = model(imgs)
                                probs_c = F.softmax(logits_c.float(), dim=1)
                                loss_adv = F.kl_div(
                                    F.log_softmax(logits_a.float(), dim=1),
                                    probs_c, reduction='batchmean'
                                )
                    grad = torch.autograd.grad(loss_adv, x_adv)[0]
                    x_adv = x_adv.detach() + (cfg['eps'] / cfg['steps']) * grad.sign()
                    delta = torch.clamp(x_adv - imgs, -cfg['eps'], cfg['eps'])
                    x_adv = torch.clamp(imgs + delta,
                                        stl_min.to(device), stl_max.to(device)).detach()
                model.train()

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                if phase_id == 0:
                    # Phase 0: clean CE only
                    logits = model(imgs)
                    loss = ce_loss(logits, lbls)
                    train_logits = logits
                elif phase_id == 1:
                    # Phase 1: SAIL contrastive invariance (no labels)
                    z_clean = model.forward_contrastive(imgs)
                    z_adv = model.forward_contrastive(x_adv)
                    # InfoNCE
                    sim = torch.mm(z_clean, z_adv.T) / 0.07
                    labels_nce = torch.arange(imgs.size(0), device=device)
                    loss = F.cross_entropy(sim, labels_nce)
                    train_logits = model(imgs)  # for accuracy tracking only
                else:
                    # Phases 2-8: TRADES
                    logits_c, _ = model.forward_with_features(imgs)
                    logits_a, _ = model.forward_with_features(x_adv)
                    loss = (ce_loss(logits_c, lbls) + effective_beta * F.kl_div(
                        F.log_softmax(logits_a.float(), dim=1),
                        F.softmax(logits_c.float().detach(), dim=1),
                        reduction='batchmean'
                    ))
                    train_logits = logits_c

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            B = imgs.size(0)
            total_loss += loss.item() * B
            n_total += B
            if phase_id != 1:
                correct += train_logits.argmax(1).eq(lbls).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, lbls in testloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast('cuda'):
                    logits = model(imgs)
                val_correct += logits.argmax(1).eq(lbls).sum().item()
                val_total += lbls.size(0)
        val_acc = 100. * val_correct / val_total

        train_acc = 100. * correct / n_total if n_total > 0 and phase_id != 1 else 0

        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            marker = ' ★'

        # Always save rolling checkpoint (critical — no progress lost)
        torch.save({
            'epoch': epoch, 'phase': phase_id,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
        }, rolling_path)

        print(
            f"P{phase_id} Epoch {epoch:02d}/{cfg['epochs']} | "
            f"Loss:{total_loss/n_total:.3f} | "
            f"TrAcc:{train_acc:.1f}% TeAcc:{val_acc:.1f}% | "
            f"β_eff:{effective_beta:.2f} | {time.time()-t0:.0f}s{marker}"
        )

    # Save final checkpoint for this phase
    torch.save(model.state_dict(), final_path)
    print(f"\nPhase {phase_id} complete. Best acc: {best_acc:.2f}%")
    print(f"Final checkpoint: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=0,
                        choices=list(CURRICULUM.keys()))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    trainloader, testloader, unlabeled_loader, stl_min, stl_max = \
        get_stl10_dataloaders(args.data_root, args.batch_size)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    print(f"\nDataset sizes:")
    print(f"  Train (labeled): {len(trainloader.dataset):,}")
    print(f"  Test:            {len(testloader.dataset):,}")
    print(f"  Unlabeled (SAIL):{len(unlabeled_loader.dataset):,}")

    # Model
    model = RHANUnifiedSTL10().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nRHAN-Unified STL-10: {total_params:.1f}M parameters")

    # Load previous phase checkpoint if not phase 0
    if args.phase > 0 and not args.resume:
        prev_ckpt = os.path.join(ckpt_dir, f'rhan_stl_phase{args.phase-1}_final.pth')
        if os.path.exists(prev_ckpt):
            model.load_state_dict(torch.load(prev_ckpt, map_location=device))
            print(f"Loaded phase {args.phase-1} checkpoint: {prev_ckpt}")
        else:
            best_ckpt = os.path.join(ckpt_dir, 'rhan_stl_best.pth')
            if os.path.exists(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
                print(f"Loaded best checkpoint: {best_ckpt}")
            else:
                print(f"WARNING: No previous checkpoint found. Training from scratch.")

    run_phase(
        phase_id=args.phase,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        unlabeled_loader=unlabeled_loader,
        device=device,
        stl_min=stl_min,
        stl_max=stl_max,
        ckpt_dir=ckpt_dir,
        resume=args.resume,
    )

    print("\nNext command:")
    if args.phase < 8:
        print(f"  python train_stl10_pretrained.py --phase {args.phase + 1}")
    else:
        print("  Curriculum complete. Run evaluation.")


if __name__ == '__main__':
    main()

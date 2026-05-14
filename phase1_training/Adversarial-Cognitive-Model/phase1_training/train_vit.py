"""
ViT-Small Training Script for CIFAR-10
=======================================

HOW THIS DIFFERS FROM train.py (ResNet):

    ┌──────────────────┬──────────────────────┬──────────────────────────────┐
    │ Component        │ ResNet (train.py)     │ ViT (train_vit.py)          │
    ├──────────────────┼──────────────────────┼──────────────────────────────┤
    │ Optimizer        │ SGD + Momentum        │ AdamW                       │
    │ Learning Rate    │ 0.01                  │ 0.0001 (100× lower)         │
    │ Weight Decay     │ 5e-4                  │ 0.05 (100× higher)          │
    │ LR Warmup        │ None                  │ 5 epochs (1e-6 → 1e-4)      │
    │ Scheduler        │ CosineAnnealing       │ CosineAnnealing (after warm)│
    │ Batch Size       │ 64                    │ 64 (same, but at 224×224)   │
    │ Input Size       │ 32×32                 │ 224×224                     │
    │ Parameters       │ ~11M                  │ ~22M                        │
    │ VRAM Usage       │ ~2GB                  │ ~5-6GB                      │
    └──────────────────┴──────────────────────┴──────────────────────────────┘

WHY AdamW INSTEAD OF SGD:
    1. ADAPTIVE LEARNING RATES: Adam maintains a per-parameter learning rate
       that adapts based on the running mean and variance of gradients. This
       is critical for transformers because different parts of the model
       (patch embeddings, attention weights, FFN layers) have vastly different
       gradient scales.

    2. WEIGHT DECAY DECOUPLING: In SGD with L2 regularization, weight decay
       is coupled with the gradient update (added to the gradient before
       scaling by LR). In AdamW, weight decay is applied SEPARATELY — directly
       shrinking the weights by a fixed fraction each step. This decoupling
       prevents the adaptive LR from counteracting regularization, which is
       essential for transformers with large parameter counts.

    3. EMPIRICAL EVIDENCE: The original ViT paper (Dosovitskiy 2021) and
       DeiT (Touvron 2021) both use AdamW. SGD consistently underperforms
       on transformers because it lacks per-parameter adaptation.

WHY WARMUP IS CRITICAL:
    At initialization, ViT's 12 attention heads produce nearly uniform
    attention distributions (every patch attends equally to every other patch).
    The model has not yet learned which patches are important.

    If we start training at the full learning rate (1e-4):
    - Large gradient updates hit the attention weights
    - Heads "lock in" to arbitrary patterns (e.g., always attending to the
      top-left patch)
    - These bad early patterns are hard to unlearn later

    Warmup starts at lr=1e-6 (100× lower) and linearly increases to 1e-4
    over 5 epochs. This gives the attention heads time to form weak but
    meaningful patterns before the optimizer pushes hard.

    Analogy: you don't sprint as soon as you wake up. You warm up first.

EXPECTED TRAINING BEHAVIOR:
    Epoch  1-5:  Loss drops slowly (warmup phase, LR still low)
    Epoch  5-10: Loss drops rapidly (full LR kicks in, model finds features)
    Epoch 10-30: Steady improvement, test accuracy climbs to 85-90%
    Epoch 30-50: Plateau / small gains, cosine LR decay helps fine-tune
    Final:       88-92% test accuracy (target range)

    If accuracy is stuck below 80% after 20 epochs, something is wrong —
    most likely the pretrained weights failed to download (check internet).
"""

import os
import time
import yaml
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_vit import CIFARViT
from dataset_vit import get_dataloaders_vit


# =============================================================================
# Reproducibility (same as train.py)
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'train_config_vit.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    # =========================================================================
    # Device and Model Setup
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = CIFARViT().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ViT-Small (patch16, 224×224)")
    print(f"Parameters: {param_count:,} total, {trainable_count:,} trainable")

    trainloader, testloader = get_dataloaders_vit(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'),
    )

    # =========================================================================
    # Loss Function
    # =========================================================================
    # CrossEntropyLoss — same as ResNet. Standard for multi-class classification.
    # Internally applies LogSoftmax + NLLLoss, which is numerically more stable
    # than doing Softmax + log + NLLLoss separately.
    # =========================================================================
    criterion = nn.CrossEntropyLoss()

    # =========================================================================
    # Optimizer: AdamW
    # =========================================================================
    # WHY NOT SGD?
    #   SGD applies the same learning rate (scaled by momentum) to every
    #   parameter. In a transformer:
    #     - Patch embedding weights have large, stable gradients → need moderate LR
    #     - Attention query/key weights have small, noisy gradients → need higher LR
    #     - FFN weights have medium gradients → need medium LR
    #   AdamW adapts the LR per-parameter using running gradient statistics,
    #   so each part of the model trains at its optimal rate.
    #
    # WHY WEIGHT DECAY = 0.05?
    #   ViT-Small has 22M parameters fine-tuned on 50k CIFAR-10 images.
    #   That's a 440:1 parameter-to-sample ratio — extreme overfit risk.
    #   Strong weight decay (0.05) prevents the model from memorizing
    #   training examples by penalizing large weights.
    #   In AdamW, this penalty is applied AFTER the gradient step (decoupled),
    #   so it doesn't interfere with Adam's adaptive learning rates.
    # =========================================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # =========================================================================
    # Scheduler: CosineAnnealingLR (starts AFTER warmup)
    # =========================================================================
    # After warmup completes at epoch `warmup_epochs`, the cosine scheduler
    # takes over for the remaining `epochs - warmup_epochs` epochs.
    # T_max is set to the post-warmup duration so the cosine curve completes
    # exactly when training ends (LR reaches near-zero at the final epoch).
    # =========================================================================
    warmup_epochs = config['warmup_epochs']
    total_epochs = config['epochs']
    target_lr = config['lr']
    warmup_start_lr = 1e-6  # 100× lower than target

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )

    # =========================================================================
    # TensorBoard
    # =========================================================================
    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), '..', 'runs', 'vit_cifar10'))
    os.makedirs(os.path.join(os.path.dirname(__file__), 'checkpoints'), exist_ok=True)

    best_acc = 0.0

    print(f"\nTraining ViT-Small on CIFAR-10")
    print(f"  Epochs: {total_epochs} (warmup: {warmup_epochs})")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Target LR: {target_lr} (warmup from {warmup_start_lr})")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: Linear Warmup → Cosine Annealing")
    print()

    for epoch in range(total_epochs):
        start_time = time.time()

        # =====================================================================
        # Learning Rate Warmup
        # =====================================================================
        # During the first `warmup_epochs` epochs, linearly interpolate the LR
        # from warmup_start_lr to target_lr.
        #
        # WHY LINEAR (not cosine or exponential)?
        #   Linear warmup is the simplest and most widely used in the transformer
        #   literature. The DeiT paper (Touvron 2021) uses exactly this approach.
        #   The choice between linear/cosine/exponential warmup has minimal impact;
        #   what matters is that the LR starts very low.
        #
        # IMPLEMENTATION:
        #   We manually set param_group['lr'] instead of using a separate
        #   warmup scheduler. This avoids scheduler conflicts and is more
        #   transparent about what's happening at each epoch.
        # =====================================================================
        if epoch < warmup_epochs:
            # Linear interpolation: epoch 0 → warmup_start_lr, epoch warmup_epochs-1 → target_lr
            warmup_lr = warmup_start_lr + (target_lr - warmup_start_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            # After warmup, let cosine scheduler control the LR
            if epoch == warmup_epochs:
                # Reset scheduler base LR to target_lr at warmup end
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs - warmup_epochs,
                )
            current_lr = optimizer.param_groups[0]['lr']

        # =====================================================================
        # Training Loop (same structure as train.py)
        # =====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # =================================================================
            # Gradient Clipping
            # =================================================================
            # 1. WHAT: Caps the total gradient norm to 1.0.
            # 2. WHY: Transformers are prone to gradient explosions, especially
            #         during early training when attention patterns are unstable.
            #         A single bad batch can produce enormous gradients that
            #         destabilize the entire model. Clipping prevents this.
            # 3. OBSERVE: Without clipping, you may see sudden loss spikes
            #         (NaN or inf) around epochs 5-10.
            # =================================================================
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(trainloader.dataset)
        train_acc = 100. * train_correct / train_total

        # =====================================================================
        # Evaluation Loop (same structure as train.py)
        # =====================================================================
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total

        # =====================================================================
        # TensorBoard Logging
        # =====================================================================
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # =====================================================================
        # Checkpointing (same logic as train.py)
        # =====================================================================
        # Save the model with the BEST test accuracy, not the final epoch.
        # This is the checkpoint that Phase 2 (attacks) will load.
        # =====================================================================
        ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'vit_small_best.pth')
        if test_acc > best_acc:
            torch.save(model.state_dict(), ckpt_path)
            best_acc = test_acc
            saved_marker = " ★ saved"
        else:
            saved_marker = ""

        # Step cosine scheduler (only after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        elapsed = time.time() - start_time
        phase = "WARMUP" if epoch < warmup_epochs else "TRAIN "
        print(f"[{phase}] Epoch {epoch+1:02d}/{total_epochs} | "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | "
              f"Test: {test_acc:.2f}% | LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s{saved_marker}")

    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    print(f"  Checkpoint saved to: checkpoints/vit_small_best.pth")
    print(f"  TensorBoard logs: runs/vit_cifar10/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

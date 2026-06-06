#!/usr/bin/env python3
"""
RHAN-CBM-TRADES: Hardening Concept Bottleneck Model with TRADES
===============================================================
Loads: checkpoints/rhan_cbm_best.pth  (frozen backbone + trained concept layers)
Trains ONLY the concept head for 20 epochs using TRADES loss.

Loss:
  l_trades  = CrossEntropy(f(x_clean), y) + beta * KL(f(x_clean) || f(x_adv))
  l_concept = BCELoss(pred_concepts, CONCEPT_LABELS[labels])
  l_consist = MSELoss(adv_concepts, clean_concepts.detach())

  total_loss = w_task * l_trades + w_concept * l_concept + w_consist * l_consist

After training:
  • PGD-100 at epsilons [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]  → εthresh
  • AutoAttack standard on 1000 images at ε=0.031 (8/255)
  • Per-class robust accuracy breakdown
  • Per-concept activation print for automobile vs. truck
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.join(script_dir, '..')
sys.path.insert(0, script_dir)
sys.path.insert(0, repo_root)

from model_rhan_v5 import RHANv5
from dataset import get_dataloaders

# AutoAttack (optional — skipped if not installed)
try:
    from autoattack import AutoAttack
    HAS_AA = True
except ImportError:
    HAS_AA = False
    print("autoattack not found – AutoAttack evaluation will be skipped.")

# PGD from phase2_attacks
sys.path.insert(0, os.path.join(repo_root, 'phase2_attacks'))
from pgd import pgd_attack


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── PGD wrapper compatible with concept forward ────────────────────────────────
class CBMWrapper(nn.Module):
    """Wraps model.forward_with_concepts so AutoAttack/PGD see plain logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model.forward_with_concepts(x)
        return logits


# ── TRADES Adversarial Example Generator ───────────────────────────────────────
def generate_trades_adv(model, wrapper, x_natural, step_size, epsilon, perturb_steps, device, clip_min, clip_max):
    """
    Generate adversarial examples for TRADES using KL divergence.
    """
    x_natural = x_natural.detach()
    
    # Freeze model & wrapper BN modules
    model.eval()
    wrapper.eval()
    
    # Initialize with random noise inside L_inf ball
    x_adv = x_natural.clone().detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    
    with torch.no_grad():
        logits_clean = wrapper(x_natural)
        probs_clean = F.softmax(logits_clean, dim=1).detach()
        
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = wrapper(x_adv)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                probs_clean,
                reduction='batchmean'
            )
        # Compute gradient w.r.t input
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        # Gradient step
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        # Projection back onto L_inf epsilon ball
        delta = torch.clamp(x_adv - x_natural, min=-epsilon, max=epsilon)
        x_adv = (x_natural + delta).detach()
        # Clip to valid range
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
        
    # Restore model training mode structure
    model.train()
    model.concept_bn.train()
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.eval()
    model.concept_bn.train()
    
    return x_adv


# ── SDT helpers ────────────────────────────────────────────────────────────────
def run_pgd_eval(model, loader, epsilons, device, max_samples=500):
    """PGD-100 accuracy at each epsilon. Returns dict {eps: acc_%}."""
    wrapper = CBMWrapper(model)
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1,3,1,1).to(device)

    results = {}
    for eps in epsilons:
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for imgs, lbls in loader:
            if total >= max_samples:
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            if eps > 0:
                adv, _ = pgd_attack(wrapper, imgs, lbls, epsilon=eps,
                                    alpha=alpha, steps=100, device=device,
                                    clip_min=cifar_min, clip_max=cifar_max,
                                    random_start=True)
            else:
                adv = imgs
            with torch.no_grad():
                logits = wrapper(adv)
                correct += logits.argmax(1).eq(lbls).sum().item()
                total   += lbls.size(0)
        results[eps] = 100.0 * correct / max(total, 1)
        print(f"  PGD-100 ε={eps:.2f}: {results[eps]:.2f}%  ({total} samples)",
              flush=True)
    return results


def compute_eps_thresh(pgd_results, epsilons):
    """Interpolate ε where d' first crosses 1.0."""
    dprimes = []
    for eps in epsilons:
        acc  = pgd_results[eps] / 100.0
        hr   = np.clip(acc, 1e-5, 1-1e-5)
        far  = np.clip((1-acc)/9, 1e-5, 1-1e-5)
        dprimes.append(float(stats.norm.ppf(hr) - stats.norm.ppf(far)))
    eps_thresh = None
    for i in range(len(dprimes)-1):
        d1, d2 = dprimes[i], dprimes[i+1]
        e1, e2 = epsilons[i], epsilons[i+1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and dprimes and dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    return dprimes, eps_thresh


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("RHAN-CBM-TRADES hardening")
    parser.add_argument('--checkpoint', default='checkpoints/rhan_cbm_best.pth',
                        help='Starting CBM checkpoint path')
    parser.add_argument('--epochs',   type=int,   default=20)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--batch',    type=int,   default=256)
    parser.add_argument('--w_task',   type=float, default=0.5)
    parser.add_argument('--w_concept',type=float, default=0.3)
    parser.add_argument('--w_consist',type=float, default=0.2)
    parser.add_argument('--beta',     type=float, default=6.0,
                        help='TRADES beta regularization strength')
    parser.add_argument('--pgd_eps',  type=float, default=0.062,
                        help='TRADES L_inf epsilon budget')
    parser.add_argument('--pgd_steps',type=int,   default=10,
                        help='TRADES perturbation steps')
    parser.add_argument('--aa_eps',   type=float, default=0.031,
                        help='Epsilon for AutoAttack evaluation (8/255≈0.031)')
    parser.add_argument('--aa_n',     type=int,   default=1000)
    parser.add_argument('--skip_aa',  action='store_true')
    args = parser.parse_args()

    set_seed(42)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)",
              flush=True)

    ckpt_dir = os.path.join(repo_root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(repo_root, args.checkpoint)
    save_path       = os.path.join(ckpt_dir, 'rhan_cbm_trades_best.pth')

    # ── Load model ──────────────────────────────────────────────────────────────
    print(f"\nLoading starting CBM checkpoint from {checkpoint_path} …", flush=True)
    model = RHANv5(head_type='cosine')
    ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=True)
    print("  Checkpoint loaded successfully (strict match).", flush=True)
    model = model.to(device)

    # ── Freeze backbone; train only CBM layers ──────────────────────────────────
    for name, param in model.named_parameters():
        if name.startswith(('concept_layer', 'concept_bn', 'concept_classifier')):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"  Trainable CBM parameters: {n_params:,}", flush=True)

    # ── Data ────────────────────────────────────────────────────────────────────
    print("\nLoading dataset …", flush=True)
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch, num_workers=4, model_name='resnet')

    # ── Optimizer + scheduler ───────────────────────────────────────────────────
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # TRADES attack configuration
    adv_eps   = args.pgd_eps
    adv_alpha = adv_eps / 4
    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([ 2.6400,  2.6210,  2.7615]).view(1,3,1,1).to(device)

    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    # ── Training loop ───────────────────────────────────────────────────────────
    best_test_acc = 0.0
    print(f"\n{'='*70}")
    print(f"RHAN-CBM-TRADES FINE-TUNING  |  {args.epochs} epochs  |  LR={args.lr}")
    print(f"TRADES config: beta={args.beta}  eps={args.pgd_eps}  steps={args.pgd_steps}")
    print(f"Loss weights: task={args.w_task}  concept={args.w_concept}  "
          f"consist={args.w_consist}")
    print(f"{'='*70}\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep BN layers in backbone frozen in eval mode
        for mod in model.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mod.eval()
        model.concept_bn.train()

        t0 = time.time()
        run_loss = run_trades = run_concept = run_consist = 0.0
        train_correct = train_total = 0

        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            # ── Adversarial generation (TRADES KL attack) ──────────────────────
            wrapper = CBMWrapper(model)
            x_adv = generate_trades_adv(
                model=model, wrapper=wrapper, x_natural=imgs,
                step_size=adv_alpha, epsilon=adv_eps, perturb_steps=args.pgd_steps,
                device=device, clip_min=cifar_min, clip_max=cifar_max
            )

            # ── Clean & Adversarial forwards ───────────────────────────────────
            logits_clean, concepts_clean = model.forward_with_concepts(imgs)
            logits_adv, concepts_adv = model.forward_with_concepts(x_adv)

            # ── Compute TRADES classification loss ──────────────────────────────
            loss_natural = F.cross_entropy(logits_clean, labels)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean, dim=1),
                reduction='batchmean'
            )
            l_trades = loss_natural + args.beta * loss_robust

            # ── Compute Concept supervision loss (BCE) ─────────────────────────
            gt_concepts = model.concept_labels[labels]   # (B, 15)
            l_concept   = bce_loss(concepts_clean, gt_concepts)

            # ── Compute Concept consistency loss (MSE) ─────────────────────────
            l_consist   = mse_loss(concepts_adv, concepts_clean.detach())

            # ── Total loss ─────────────────────────────────────────────────────
            loss = (args.w_task    * l_trades +
                    args.w_concept * l_concept +
                    args.w_consist * l_consist)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            run_loss    += loss.item()    * B
            run_trades  += l_trades.item()  * B
            run_concept += l_concept.item() * B
            run_consist += l_consist.item() * B

            preds = logits_clean.argmax(1)
            train_correct += preds.eq(labels).sum().item()
            train_total   += B

        scheduler.step()

        # ── Test accuracy ──────────────────────────────────────────────────────
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model.forward_with_concepts(imgs)
                test_correct += logits.argmax(1).eq(labels).sum().item()
                test_total   += labels.size(0)

        train_acc = 100. * train_correct / train_total
        test_acc  = 100. * test_correct  / test_total
        n = train_total
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Loss:{run_loss/n:.4f} "
              f"(trades:{run_trades/n:.4f} "
              f"concept:{run_concept/n:.4f} "
              f"consist:{run_consist/n:.4f}) | "
              f"TrainAcc:{train_acc:.1f}% TestAcc:{test_acc:.1f}% | "
              f"LR:{scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.0f}s", flush=True)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best test acc {best_test_acc:.2f}% → saved to {save_path}",
                  flush=True)

    # ── Post-training evaluation ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE  |  Best test acc: {best_test_acc:.2f}%")
    print(f"{'='*70}\n", flush=True)

    # Reload best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    model = model.to(device).eval()

    # ── PGD-100 + SDT ε_thresh ─────────────────────────────────────────────────
    print("="*70)
    print("PGD-100 EVALUATION + SDT THRESHOLD")
    print("="*70, flush=True)
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    pgd_res  = run_pgd_eval(model, testloader, epsilons, device, max_samples=500)
    dprimes, eps_thresh = compute_eps_thresh(pgd_res, epsilons)

    print(f"\n  ε_thresh (d'=1.0): {eps_thresh:.4f}" if eps_thresh is not None
          else "\n  ε_thresh (d'=1.0): >0.30")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f} → PGD-100 Acc: {pgd_res[eps]:.2f}%  d'={dprimes[i]:.4f}")

    # ── AutoAttack ────────────────────────────────────────────────────────────
    if not args.skip_aa and HAS_AA:
        print(f"\n{'='*70}")
        print(f"AUTOATTACK (standard, ε={args.aa_eps:.4f}, n={args.aa_n})")
        print(f"{'='*70}", flush=True)

        wrapper = CBMWrapper(model)
        imgs_list, lbls_list = [], []
        for imgs, lbls in testloader:
            imgs_list.append(imgs); lbls_list.append(lbls)
            if sum(x.size(0) for x in imgs_list) >= args.aa_n:
                break
        x_test = torch.cat(imgs_list)[:args.aa_n].to(device)
        y_test = torch.cat(lbls_list)[:args.aa_n].to(device)

        # Clean accuracy on subset
        with torch.no_grad():
            clean_preds = wrapper(x_test).argmax(1)
            subset_clean_acc = 100. * clean_preds.eq(y_test).float().mean().item()
        print(f"Subset clean acc: {subset_clean_acc:.2f}%", flush=True)

        adversary = AutoAttack(wrapper, norm='Linf', eps=args.aa_eps,
                               version='standard', device=device, verbose=True)
        t0 = time.time()
        try:
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=64)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=32)
            else:
                raise
        print(f"AutoAttack finished in {(time.time()-t0)/60:.1f} min", flush=True)

        with torch.no_grad():
            rob_preds = wrapper(x_adv).argmax(1)
            rob_correct = rob_preds.eq(y_test).sum().item()
            overall_aa = 100. * rob_correct / args.aa_n

        print(f"\nOverall AutoAttack robust accuracy: {overall_aa:.2f}%")

        # Per-class breakdown
        class_rob = {i: [0, 0] for i in range(10)}   # [correct, total]
        class_cln = {i: [0, 0] for i in range(10)}
        for i in range(args.aa_n):
            lbl = y_test[i].item()
            class_rob[lbl][1] += 1
            class_cln[lbl][1] += 1
            if clean_preds[i] == lbl:
                class_cln[lbl][0] += 1
            if rob_preds[i] == lbl:
                class_rob[lbl][0] += 1

        print(f"\n{'Class':>12} | {'Clean':>8} | {'Robust':>8} | {'Count':>6}")
        print("-" * 45)
        for i, name in enumerate(class_names):
            n_c = class_cln[i][1]
            c_a = 100. * class_cln[i][0] / max(n_c, 1)
            r_a = 100. * class_rob[i][0] / max(n_c, 1)
            print(f"{name:>12} | {c_a:>7.1f}% | {r_a:>7.1f}% | {n_c:>6}")

        # Decision
        print(f"\n{'='*70}")
        if overall_aa >= 29.0:
            print(f"✅ Target met: AutoAttack {overall_aa:.2f}% >= 29.0% threshold")
        else:
            print(f"→ AutoAttack {overall_aa:.2f}% (< 29.0%). Target not fully met.")
        print(f"{'='*70}")

    # ── Interpretability: concept activations automobile vs truck ──────────────
    print(f"\n{'='*70}")
    print("CONCEPT INTERPRETABILITY: automobile vs. truck")
    print(f"{'='*70}", flush=True)

    # Collect ~50 images of each class
    auto_imgs, truck_imgs = [], []
    for imgs, lbls in testloader:
        for img, lbl in zip(imgs, lbls):
            lbl = lbl.item()
            if lbl == 1 and len(auto_imgs) < 50:
                auto_imgs.append(img)
            if lbl == 9 and len(truck_imgs) < 50:
                truck_imgs.append(img)
        if len(auto_imgs) >= 50 and len(truck_imgs) >= 50:
            break

    model.eval()
    with torch.no_grad():
        _, auto_concepts  = model.forward_with_concepts(
            torch.stack(auto_imgs).to(device))
        _, truck_concepts = model.forward_with_concepts(
            torch.stack(truck_imgs).to(device))
    auto_mean  = auto_concepts.mean(0).cpu().numpy()
    truck_mean = truck_concepts.mean(0).cpu().numpy()

    concept_names = model.concepts
    print(f"\n{'Concept':>18} | {'Automobile':>12} | {'Truck':>10} | {'Δ (auto-truck)':>14}")
    print("-" * 65)
    for i, name in enumerate(concept_names):
        diff = auto_mean[i] - truck_mean[i]
        marker = "  ◄ KEY" if abs(diff) > 0.10 else ""
        print(f"{name:>18} | {auto_mean[i]:>11.3f} | {truck_mean[i]:>9.3f} | "
              f"{diff:>+13.3f}{marker}")

    print(f"\nKey separation:")
    for i, name in enumerate(concept_names):
        if name in ('is_small_vehicle', 'carries_cargo', 'has_wheels', 'is_metallic'):
            print(f"  {name:>18}: auto={auto_mean[i]:.3f}  truck={truck_mean[i]:.3f}")

    print(f"\n{'='*70}")
    print(f"RHAN-CBM-TRADES COMPLETE  |  Best clean: {best_test_acc:.2f}%")
    print(f"Checkpoint saved: {save_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

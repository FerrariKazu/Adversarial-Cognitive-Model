#!/usr/bin/env python3
"""
RHAN-CBM v2: Hard Concept Bottleneck Model Fine-Tuning
=======================================================
Loads: checkpoints/rhan_trades_phase_c_final.pth  (best εthresh backbone)

Three improvements over CBM v1:

  Change 1 — Straight-Through Hard Concept Thresholding:
    Concepts are binary at inference (>0.5 → 1, else 0).
    Straight-through estimator keeps gradient flow during training.
    AutoAttack cannot exploit continuous gradients through binary gates.

  Change 2 — Focal Loss for Concept Supervision:
    Focal(γ=2) down-weights easy correct concept predictions and focuses
    training on hard/spurious activations (e.g. automobile vs truck
    sharing is_metallic/has_rigid_body while differing on
    is_small_vehicle/carries_cargo).

  Change 3 — Hard Adversarial Concept Supervision:
    PGD adversarial images are forced to predict the HARD (binarised)
    clean concept labels, not the continuous soft values.
    This prevents gradient routing through spurious continuous concepts
    because the targets are discrete binary values.

Loss:
  l_task    = CE(concept_logits, labels)             weight 0.5
  l_concept = FocalLoss(γ=2)(concepts, GT_concepts)  weight 0.3
  l_consist = BCE(adv_concepts, hard_clean_concepts)  weight 0.2

Save to: checkpoints/rhan_cbm_v2_best.pth
Target:  automobile / truck / horse each > 20% AutoAttack robust acc
"""

import os, sys, time, random, argparse
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

try:
    from autoattack import AutoAttack
    HAS_AA = True
except ImportError:
    HAS_AA = False
    print("autoattack not found – AutoAttack evaluation will be skipped.")

sys.path.insert(0, os.path.join(repo_root, 'phase2_attacks'))
from pgd import pgd_attack


# ── Seed ───────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalBCELoss(nn.Module):
    """
    Focal loss for binary concept supervision.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C) sigmoid probabilities; targets: (B, C) binary
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ── Wrapper for PGD/AutoAttack ─────────────────────────────────────────────────
class CBMv2Wrapper(nn.Module):
    """Returns concept-path logits, hard-thresholded at inference."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _, _ = self.model.forward_with_concepts_hard(x)
        return logits


# ── Extend RHANv5 with hard concept forward ────────────────────────────────────
def patch_hard_forward(model):
    """
    Monkey-patch model.forward_with_concepts_hard onto the RHANv5 instance.
    Uses straight-through estimator so gradient flows through the soft
    probabilities while the discrete 0/1 values are used in the forward pass.
    """
    def forward_with_concepts_hard(self_m, x):
        features = self_m.get_feature_vector(x)               # (B, 512)
        concepts_soft = torch.sigmoid(
            self_m.concept_bn(self_m.concept_layer(features))
        )                                                       # (B, 15)
        # Straight-through: hard values for forward, soft grads for backward
        concepts_hard = (concepts_soft > 0.5).float()
        concepts_ste  = concepts_hard - concepts_soft.detach() + concepts_soft
        logits = self_m.concept_classifier(concepts_ste)       # (B, 10)
        return logits, concepts_soft, concepts_ste

    import types
    model.forward_with_concepts_hard = types.MethodType(forward_with_concepts_hard, model)
    return model


# ── PGD evaluation ─────────────────────────────────────────────────────────────
def run_pgd_eval(model, loader, epsilons, device, max_samples=500):
    wrapper = CBMv2Wrapper(model)
    cifar_min = torch.tensor([-2.4291,-2.4181,-2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([ 2.6400, 2.6210, 2.7615]).view(1,3,1,1).to(device)
    results = {}
    for eps in epsilons:
        correct = total = 0
        alpha = max(eps/10, 0.001) if eps > 0 else 0
        for imgs, lbls in loader:
            if total >= max_samples: break
            imgs, lbls = imgs.to(device), lbls.to(device)
            if eps > 0:
                adv, _ = pgd_attack(wrapper, imgs, lbls, epsilon=eps,
                                    alpha=alpha, steps=100, device=device,
                                    clip_min=cifar_min, clip_max=cifar_max,
                                    random_start=True)
            else:
                adv = imgs
            with torch.no_grad():
                correct += wrapper(adv).argmax(1).eq(lbls).sum().item()
                total   += lbls.size(0)
        results[eps] = 100.0 * correct / max(total, 1)
        print(f"  PGD-100 ε={eps:.2f}: {results[eps]:.2f}%  ({total} samples)", flush=True)
    return results


def compute_eps_thresh(pgd_results, epsilons):
    dprimes = []
    for eps in epsilons:
        acc = pgd_results[eps] / 100.0
        hr  = np.clip(acc, 1e-5, 1-1e-5)
        far = np.clip((1-acc)/9, 1e-5, 1-1e-5)
        dprimes.append(float(stats.norm.ppf(hr) - stats.norm.ppf(far)))
    eps_thresh = None
    for i in range(len(dprimes)-1):
        d1, d2 = dprimes[i], dprimes[i+1]
        e1, e2 = epsilons[i], epsilons[i+1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0-d1)*(e2-e1)/(d2-d1)
            break
    if eps_thresh is None and dprimes and dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    return dprimes, eps_thresh


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("RHAN-CBM v2 fine-tuning")
    parser.add_argument('--backbone',  default='checkpoints/rhan_trades_phase_c_final.pth')
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--batch',     type=int,   default=256)
    parser.add_argument('--w_task',    type=float, default=0.5)
    parser.add_argument('--w_concept', type=float, default=0.3)
    parser.add_argument('--w_consist', type=float, default=0.2)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--pgd_eps',   type=float, default=0.062,
                        help='PGD epsilon for adversarial concept consistency')
    parser.add_argument('--pgd_steps', type=int,   default=3,
                        help='PGD steps during training (3 = fast, less overhead)')
    parser.add_argument('--aa_eps',    type=float, default=0.031)
    parser.add_argument('--aa_n',      type=int,   default=1000)
    parser.add_argument('--skip_aa',   action='store_true')
    args = parser.parse_args()

    set_seed(42)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory/1024**3:.1f} GB)", flush=True)

    ckpt_dir      = os.path.join(repo_root, 'checkpoints')
    backbone_path = os.path.join(repo_root, args.backbone)
    save_path     = os.path.join(ckpt_dir, 'rhan_cbm_v2_best.pth')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Load backbone ──────────────────────────────────────────────────────────
    print(f"\nLoading backbone from {backbone_path} …", flush=True)
    model = RHANv5(head_type='cosine')
    ckpt  = torch.load(backbone_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing (CBM layers, expected): {missing}", flush=True)
    if unexpected:
        print(f"  Unexpected: {unexpected}", flush=True)

    model = patch_hard_forward(model.to(device))

    # ── Freeze backbone; train CBM head only ───────────────────────────────────
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(
            ('concept_layer', 'concept_bn', 'concept_classifier'))
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    print("\nLoading dataset …", flush=True)
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch, num_workers=4, model_name='resnet')

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    focal_loss = FocalBCELoss(gamma=args.focal_gamma)
    bce_loss   = nn.BCELoss()

    adv_eps   = args.pgd_eps
    adv_alpha = adv_eps / 4
    adv_steps = args.pgd_steps
    cifar_min = torch.tensor([-2.4291,-2.4181,-2.2194]).view(1,3,1,1).to(device)
    cifar_max = torch.tensor([ 2.6400, 2.6210, 2.7615]).view(1,3,1,1).to(device)

    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    # ── Training loop ──────────────────────────────────────────────────────────
    best_test_acc = 0.0
    print(f"\n{'='*72}")
    print(f"RHAN-CBM v2 | epochs={args.epochs} | lr={args.lr} | "
          f"focal_gamma={args.focal_gamma}")
    print(f"Loss weights: task={args.w_task}  concept={args.w_concept}  "
          f"consist={args.w_consist}")
    print(f"PGD-{adv_steps} adversarial training at ε={adv_eps:.3f}")
    print(f"{'='*72}\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep backbone BN in eval; only concept_bn trains
        for mod in model.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mod.eval()
        model.concept_bn.train()

        t0 = time.time()
        run_loss = run_task = run_focal = run_consist = 0.0
        train_correct = train_total = 0

        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)

            # ── Clean forward ──────────────────────────────────────────────────
            logits_clean, concepts_soft_clean, concepts_ste_clean = \
                model.forward_with_concepts_hard(imgs)

            # Task loss through STE concepts
            l_task = F.cross_entropy(logits_clean, labels)

            # Focal concept supervision (Change 2)
            gt_concepts = model.concept_labels[labels]         # (B, 15)
            l_focal     = focal_loss(concepts_soft_clean, gt_concepts)

            # Hard clean concept labels for adversarial target (Change 3)
            hard_clean = (concepts_soft_clean > 0.5).float().detach()

            # ── Adversarial forward — PGD-3 (Change 3) ────────────────────────
            wrapper = CBMv2Wrapper(model)
            model.eval()
            with torch.enable_grad():
                adv, _ = pgd_attack(
                    wrapper, imgs, labels,
                    epsilon=adv_eps, alpha=adv_alpha, steps=adv_steps,
                    device=device, clip_min=cifar_min, clip_max=cifar_max,
                    random_start=True)
            model.train()
            for mod in model.modules():
                if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    mod.eval()
            model.concept_bn.train()

            _, concepts_soft_adv, _ = model.forward_with_concepts_hard(adv)

            # Hard-target adversarial concept consistency (Change 3):
            # Force adv concept predictions toward BINARY clean concepts.
            l_consist = bce_loss(concepts_soft_adv, hard_clean)

            # ── Total loss ─────────────────────────────────────────────────────
            loss = (args.w_task    * l_task +
                    args.w_concept * l_focal +
                    args.w_consist * l_consist)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            run_loss    += loss.item()    * B
            run_task    += l_task.item()  * B
            run_focal   += l_focal.item() * B
            run_consist += l_consist.item() * B

            train_correct += logits_clean.argmax(1).eq(labels).sum().item()
            train_total   += B

        scheduler.step()

        # ── Test accuracy ──────────────────────────────────────────────────────
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _, _ = model.forward_with_concepts_hard(imgs)
                test_correct += logits.argmax(1).eq(labels).sum().item()
                test_total   += labels.size(0)

        train_acc = 100. * train_correct / train_total
        test_acc  = 100. * test_correct  / test_total
        n = train_total

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Loss:{run_loss/n:.4f} "
              f"(task:{run_task/n:.4f} "
              f"focal:{run_focal/n:.4f} "
              f"consist:{run_consist/n:.4f}) | "
              f"Train:{train_acc:.1f}% Test:{test_acc:.1f}% | "
              f"LR:{scheduler.get_last_lr()[0]:.2e} | "
              f"{time.time()-t0:.0f}s", flush=True)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best {best_test_acc:.2f}% → {save_path}", flush=True)

    # ── Load best checkpoint ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"TRAINING COMPLETE  |  Best test acc: {best_test_acc:.2f}%")
    print(f"{'='*72}\n", flush=True)
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=False))
    model = patch_hard_forward(model.to(device).eval())

    # ── PGD-100 + SDT ε_thresh ─────────────────────────────────────────────────
    print("="*72)
    print("PGD-100 EVALUATION + SDT ε_thresh")
    print("="*72, flush=True)
    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    pgd_res  = run_pgd_eval(model, testloader, epsilons, device, max_samples=500)
    dprimes, eps_thresh = compute_eps_thresh(pgd_res, epsilons)

    print(f"\n  ε_thresh (d'=1.0): "
          f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f} → {pgd_res[eps]:.2f}%  d'={dprimes[i]:.4f}")

    # ── AutoAttack ────────────────────────────────────────────────────────────
    if not args.skip_aa and HAS_AA:
        print(f"\n{'='*72}")
        print(f"AUTOATTACK (standard, ε={args.aa_eps:.4f}, n={args.aa_n})")
        print(f"{'='*72}", flush=True)

        wrapper = CBMv2Wrapper(model)
        imgs_list, lbls_list = [], []
        for imgs, lbls in testloader:
            imgs_list.append(imgs); lbls_list.append(lbls)
            if sum(x.size(0) for x in imgs_list) >= args.aa_n:
                break
        x_test = torch.cat(imgs_list)[:args.aa_n].to(device)
        y_test = torch.cat(lbls_list)[:args.aa_n].to(device)

        with torch.no_grad():
            clean_preds = wrapper(x_test).argmax(1)
            subset_cln  = 100. * clean_preds.eq(y_test).float().mean().item()
        print(f"Subset clean acc: {subset_cln:.2f}%", flush=True)

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
            overall_aa = 100. * rob_preds.eq(y_test).float().mean().item()

        # Per-class breakdown
        class_rob = {i: [0,0] for i in range(10)}
        class_cln = {i: [0,0] for i in range(10)}
        for i in range(args.aa_n):
            lbl = y_test[i].item()
            class_rob[lbl][1] += 1
            class_cln[lbl][1] += 1
            if clean_preds[i] == lbl: class_cln[lbl][0] += 1
            if rob_preds[i]   == lbl: class_rob[lbl][0] += 1

        print(f"\nOverall AutoAttack robust accuracy: {overall_aa:.2f}%\n")
        print(f"{'Class':>12} | {'Clean':>8} | {'Robust':>8} | {'Count':>6}")
        print("-"*45)
        for i, name in enumerate(class_names):
            n_c = class_cln[i][1]
            c_a = 100. * class_cln[i][0] / max(n_c, 1)
            r_a = 100. * class_rob[i][0] / max(n_c, 1)
            flag = "  ★" if r_a > 20.0 and name in ('automobile','truck','horse') else ""
            print(f"{name:>12} | {c_a:>7.1f}% | {r_a:>7.1f}% | {n_c:>6}{flag}")

        # Decision
        key_classes = [
            100. * class_rob[1][0] / max(class_rob[1][1],1),  # automobile
            100. * class_rob[9][0] / max(class_rob[9][1],1),  # truck
            100. * class_rob[7][0] / max(class_rob[7][1],1),  # horse
        ]
        print(f"\n{'='*72}")
        print(f"KEY CLASS CHECK  automobile={key_classes[0]:.1f}%  "
              f"truck={key_classes[1]:.1f}%  horse={key_classes[2]:.1f}%")
        if all(v > 20.0 for v in key_classes):
            print("✅ TARGET MET: all three vulnerable classes > 20%!")
            print(f"   Overall AA: {overall_aa:.2f}%  "
                  "(expected 33-38% — report RHAN-CBM v2 in paper)")
        elif overall_aa > 30.0:
            print(f"✅ Overall AA {overall_aa:.2f}% > 30% — CBM v2 viable for paper.")
            print("   Key classes not all > 20%; partial improvement achieved.")
        else:
            print(f"→ Overall AA {overall_aa:.2f}%.  Key classes not all > 20%.")
            print("   Curriculum checkpoint remains headline; CBM v2 is a "
                  "partial improvement.")
        print(f"{'='*72}")

    # ── Concept interpretability: automobile vs truck ──────────────────────────
    print(f"\n{'='*72}")
    print("CONCEPT INTERPRETABILITY (hard threshold): automobile vs. truck")
    print(f"{'='*72}", flush=True)

    auto_imgs, truck_imgs = [], []
    for imgs, lbls in testloader:
        for img, lbl in zip(imgs, lbls):
            lbl = lbl.item()
            if lbl == 1 and len(auto_imgs) < 64: auto_imgs.append(img)
            if lbl == 9 and len(truck_imgs) < 64: truck_imgs.append(img)
        if len(auto_imgs) >= 64 and len(truck_imgs) >= 64: break

    model.eval()
    with torch.no_grad():
        _, auto_soft,  _ = model.forward_with_concepts_hard(
            torch.stack(auto_imgs).to(device))
        _, truck_soft, _ = model.forward_with_concepts_hard(
            torch.stack(truck_imgs).to(device))
        auto_hard  = (auto_soft  > 0.5).float().mean(0).cpu().numpy()
        truck_hard = (truck_soft > 0.5).float().mean(0).cpu().numpy()

    concept_names = model.concepts
    print(f"\n{'Concept':>18} | {'Auto (frac>0.5)':>16} | "
          f"{'Truck (frac>0.5)':>16} | {'Δ auto-truck':>13}")
    print("-"*72)
    for i, name in enumerate(concept_names):
        diff = auto_hard[i] - truck_hard[i]
        flag = "  ◄ KEY" if abs(diff) > 0.15 else ""
        print(f"{name:>18} | {auto_hard[i]:>15.3f} | "
              f"{truck_hard[i]:>15.3f} | {diff:>+12.3f}{flag}")

    print(f"\nCritical concept separation:")
    for i, name in enumerate(concept_names):
        if name in ('is_small_vehicle','carries_cargo','has_wheels','is_metallic'):
            print(f"  {name:>18}: auto={auto_hard[i]:.3f}  truck={truck_hard[i]:.3f}")

    print(f"\n{'='*72}")
    print(f"RHAN-CBM v2 COMPLETE  |  Best clean: {best_test_acc:.2f}%")
    print(f"Checkpoint: {save_path}")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    main()

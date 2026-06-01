import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase1_training.model_rhan_v5 import RHANv5
from phase1_training.dataset import get_dataloaders
from phase2_attacks.pgd import pgd_attack

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, '..', 'checkpoints')
    output_ckpt = os.path.join(ckpt_dir, 'rhan_v5_best.pth')
    
    if not os.path.exists(output_ckpt):
        print(f"ERROR: Checkpoint not found at {output_ckpt}")
        return

    # Load dataloaders
    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False)

    cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(device)
    cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(device)

    print("Loading best checkpoint for evaluation...")
    eval_model = RHANv5(head_type='cosine').to(device)
    eval_model.load_state_dict(torch.load(output_ckpt, map_location=device))
    eval_model.eval()
    for p in eval_model.parameters():
        p.requires_grad = False

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    wrapper = W(eval_model)

    epsilons = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
    max_samples = 500

    # ── Step 1: Gradient masking check ──
    print(f"\n{'='*70}\nGradient Masking Check\n{'='*70}")
    for check_eps in [0.05, 0.10]:
        rn_correct = rn_total = 0
        with torch.no_grad():
            for images, lbls in testloader:
                if rn_total >= max_samples: break
                images, lbls = images.to(device), lbls.to(device)
                noise = torch.empty_like(images).uniform_(-check_eps, check_eps)
                noisy = torch.max(torch.min(images + noise, cifar_max), cifar_min)
                _, preds = eval_model(noisy).max(1)
                rn_correct += preds.eq(lbls).sum().item(); rn_total += lbls.size(0)
        rn_acc = 100.0 * rn_correct / max(rn_total, 1)
        print(f"  Random noise ε={check_eps:.2f}: {rn_acc:.2f}%")

    # PGD-20 at ε=0.05
    p20_correct = p20_total = 0
    for images, lbls in testloader:
        if p20_total >= max_samples: break
        images, lbls = images.to(device), lbls.to(device)
        adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=0.05, alpha=0.005,
            steps=20, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
        with torch.no_grad():
            _, preds = eval_model(adv_images).max(1)
            p20_correct += preds.eq(lbls).sum().item(); p20_total += lbls.size(0)
    p20_acc_05 = 100.0 * p20_correct / max(p20_total, 1)
    print(f"  PGD-20 ε=0.05: {p20_acc_05:.2f}%")

    # ── Step 2: Full PGD-100 evaluation ──
    print(f"\n{'='*70}\nRHAN-v5 PGD-100 Evaluation\n{'='*70}")
    v5_accs = []
    for eps in epsilons:
        t0 = time.time()
        print(f"Evaluating ε={eps:.2f}...", end=' ', flush=True)
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, lbls in testloader:
            if total >= max_samples: break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv_images = images
            with torch.no_grad():
                _, preds = eval_model(adv_images).max(1)
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        acc = 100.0 * correct / max(total, 1); v5_accs.append(acc)
        print(f"Acc:{acc:.2f}% | {time.time()-t0:.1f}s")

    # PGD-20 vs PGD-100 gap
    p100_05 = v5_accs[2]
    pgd_gap = p20_acc_05 - p100_05
    print(f"\nPGD-20 vs PGD-100 gap at ε=0.05: {pgd_gap:.2f}%")
    masking_detected = pgd_gap >= 8.0
    if masking_detected:
        print("  ⚠ Potential gradient masking!")
    else:
        print(f"  ✓ No gradient masking (gap {pgd_gap:.2f}%)")

    # ── Step 3: SDT d-prime & εthresh ──
    import scipy.stats as stats
    v5_dprimes = []
    for acc_pct in v5_accs:
        acc = acc_pct / 100.0
        hr = np.clip(acc, 1e-5, 1 - 1e-5)
        far = np.clip((1 - acc) / 9, 1e-5, 1 - 1e-5)
        dp = stats.norm.ppf(hr) - stats.norm.ppf(far)
        v5_dprimes.append(float(dp))

    eps_thresh = None
    for i in range(len(v5_dprimes) - 1):
        d1, d2 = v5_dprimes[i], v5_dprimes[i + 1]
        e1, e2 = epsilons[i], epsilons[i + 1]
        if d1 >= 1.0 >= d2:
            eps_thresh = e1 + (1.0 - d1) * (e2 - e1) / (d2 - d1)
            break
    if eps_thresh is None and len(v5_dprimes) > 0 and v5_dprimes[0] < 1.0:
        eps_thresh = epsilons[0]
    thresh_str = f"{eps_thresh:.4f}" if eps_thresh is not None else ">0.30"

    # ── Step 4: Frequency weight analysis ──
    w_lo_final = torch.sigmoid(eval_model.freq_weight_low).item()
    w_hi_final = torch.sigmoid(eval_model.freq_weight_high).item()

    print(f"\n{'='*70}\nFrequency Weight Analysis\n{'='*70}")
    print(f"  freq_weight_low (sigmoid):  {w_lo_final:.4f}")
    print(f"  freq_weight_high (sigmoid): {w_hi_final:.4f}")
    if w_lo_final > w_hi_final:
        print(f"  ✓ Model learned shape-dominant processing (low > high)")
        print(f"    Biological hypothesis CONFIRMED: M-pathway dominance")
    else:
        print(f"  ⚠ Model did not learn low-frequency dominance")

    # ── Step 5: Ablation — low-freq only inference ──
    print(f"\n{'='*70}\nAblation: Low-Frequency Only Inference\n{'='*70}")
    saved_high = eval_model.freq_weight_high.data.clone()
    eval_model.freq_weight_high.data.fill_(-100.0)  # sigmoid(-100) ≈ 0

    lo_only_accs = []
    for eps in epsilons:
        correct = total = 0
        alpha = max(eps / 10, 0.001) if eps > 0 else 0
        for images, lbls in testloader:
            if total >= max_samples: break
            images, lbls = images.to(device), lbls.to(device)
            if eps > 0:
                adv_images, _ = pgd_attack(wrapper, images, lbls, epsilon=eps, alpha=alpha,
                    steps=100, device=device, clip_min=cifar_min, clip_max=cifar_max, random_start=True)
            else:
                adv_images = images
            with torch.no_grad():
                _, preds = eval_model(adv_images).max(1)
                correct += preds.eq(lbls).sum().item(); total += lbls.size(0)
        lo_acc = 100.0 * correct / max(total, 1); lo_only_accs.append(lo_acc)
    eval_model.freq_weight_high.data = saved_high

    print(f"{'ε':<8} | {'Full Model':>10} | {'Low-Only':>10} | {'Delta':>8}")
    print("-" * 45)
    for i, eps in enumerate(epsilons):
        delta = lo_only_accs[i] - v5_accs[i]
        sign = '+' if delta >= 0 else ''
        print(f"{eps:<8.2f} | {v5_accs[i]:>9.2f}% | {lo_only_accs[i]:>9.2f}% | {sign}{delta:>7.2f}%")

    # ── Step 6: Final comparison table ──
    rhan_v3 = {0.00: 91.41, 0.01: 85.35, 0.05: 60.74, 0.10: 26.17, 0.20: 1.17, 0.30: 0.00}
    rhan_adv = {0.00: 83.79, 0.01: 77.93, 0.05: 51.95, 0.10: 17.77, 0.20: 0.59, 0.30: 0.00}
    resnet = {0.00: 95.82, 0.01: 75.57, 0.05: 2.84, 0.10: 0.21, 0.20: 0.02, 0.30: 0.00}
    vit = {0.00: 97.80, 0.01: 55.18, 0.05: 8.80, 0.10: 2.78, 0.20: 1.12, 0.30: 0.58}
    human = {0.00: 73.33, 0.01: 'N/A', 0.05: 69.17, 0.10: 59.17, 0.20: 62.22, 0.30: 58.61}

    print(f"\n{'='*85}\nRHAN-v5 FINAL VERDICT\n{'='*85}")
    print(f"{'ε':<8} | {'Human':>8} | {'RHAN-v5':>8} | {'RHAN-v3':>8} | {'RHAN-adv':>8} | {'ResNet':>8} | {'ViT':>8}")
    print("-" * 85)
    for i, eps in enumerate(epsilons):
        h = human[eps]
        h_str = f"{h:.2f}%" if isinstance(h, float) else h
        print(f"{eps:<8.2f} | {h_str:>8} | {v5_accs[i]:>7.2f}% | {rhan_v3[eps]:>7.2f}% | {rhan_adv[eps]:>7.2f}% | {resnet[eps]:>7.2f}% | {vit[eps]:>7.2f}%")
    print("=" * 85)

    print(f"\n--- SDT d-prime ---")
    for i, eps in enumerate(epsilons):
        print(f"  ε={eps:.2f}: d'={v5_dprimes[i]:.4f}")
    print(f"\nε_thresh (d'=1.0): {thresh_str}")

    print(f"\n{'='*70}")
    print("ROBUSTNESS RANKING (SDT ε_thresh)")
    print(f"{'='*70}")
    print(f"  {'System':<20} | {'ε_thresh':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Human':<20} | {'> 0.3000':>10}")
    print(f"  {'RHAN-v5':<20} | {thresh_str:>10}")
    print(f"  {'RHAN-v3':<20} | {'0.0900':>10}")
    print(f"  {'RHAN-adv':<20} | {'0.0764':>10}")
    print(f"  {'ResNet-18':<20} | {'0.0295':>10}")
    print(f"  {'ViT-Small':<20} | {'0.0264':>10}")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    if eps_thresh is not None and eps_thresh > 0.200:
        print(f"🏆 TARGET ACHIEVED: ε_thresh = {thresh_str} > 0.200")
    elif eps_thresh is not None and eps_thresh > 0.120:
        print(f"✅ Strong improvement: ε_thresh = {thresh_str} > 0.120")
    else:
        print(f"⚠  ε_thresh = {thresh_str} — did not reach 0.120 target")
    print(f"Frequency weights: low={w_lo_final:.3f}, high={w_hi_final:.3f}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

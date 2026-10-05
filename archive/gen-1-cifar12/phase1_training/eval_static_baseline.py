#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from phase1_training.model_rhan_stl10_large import RHANLargeSTL10
from phase1_training.dataset_stl10 import get_stl10_loaders, STL10_MEAN, STL10_STD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CorrectWrapper(nn.Module):
    def __init__(self, m, mean, std, device):
        super().__init__()
        self.m = m
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        out = self.m(x_norm)
        return out[0] if isinstance(out, tuple) else out

def run_pgd_eval(model, x, y, eps, steps=20):
    if eps == 0:
        return x.clone().detach()
    stl_min = torch.tensor([-1.7161, -1.7140, -1.4987], device=device).view(1,3,1,1)
    stl_max = torch.tensor([2.1256, 2.1832, 2.1872], device=device).view(1,3,1,1)
    
    model.eval()
    with torch.no_grad():
        logits_c = model(x)
        if isinstance(logits_c, tuple):
            logits_c = logits_c[0]
    probs_c = F.softmax(logits_c.float(), dim=1)
    
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, stl_min, stl_max)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_a = model(x_adv)
            if isinstance(logits_a, tuple):
                logits_a = logits_a[0]
            loss = F.kl_div(
                F.log_softmax(logits_a.float(), dim=1),
                probs_c, reduction='batchmean'
            )
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + (eps / steps) * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, stl_min, stl_max).detach()
    return x_adv

def download_checkpoint_from_hf(ckpt_path):
    filename = os.path.basename(ckpt_path)
    print(f"Checkpoint not found locally at {ckpt_path}. Attempting to download {filename} from Hugging Face...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        for repo in ['FerrariKazu/rhan-checkpoints-rolling', 'FerrariKazu/rhan-checkpoints']:
            try:
                print(f"Checking {repo}...", flush=True)
                downloaded_cache_path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    repo_type='dataset',
                    token=hf_token
                )
                import shutil
                shutil.copy2(downloaded_cache_path, ckpt_path)
                print(f"Successfully downloaded to: {ckpt_path}", flush=True)
                return True
            except Exception as e:
                print(f"Failed to download from {repo}: {e}", flush=True)
        
        return False
    except Exception as e:
        print(f"Hugging Face download failed: {e}", flush=True)
        return False

def main():
    print(f"\n{'='*80}")
    print(" EVALUATING STATIC TRADES LARGE BASELINE MODEL")
    print(f"{'='*80}", flush=True)

    _, test_loader = get_stl10_loaders(batch_size=32, data_root='./data/stl10')
    
    # Check for checkpoint in both potential locations
    ckpt_path = 'checkpoints_tier2/rhan_stl10_large_pseudolabel_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/rhan_stl10_large_pseudolabel_best.pth'
    
    if not os.path.exists(ckpt_path):
        # Default to checkpoints/ folder for download target
        ckpt_path = 'checkpoints/rhan_stl10_large_pseudolabel_best.pth'
        success = download_checkpoint_from_hf(ckpt_path)
        if not success:
            # Try checkpoints_tier2/ as fallback
            ckpt_path = 'checkpoints_tier2/rhan_stl10_large_pseudolabel_best.pth'
            success = download_checkpoint_from_hf(ckpt_path)
            
        if not os.path.exists(ckpt_path):
            print(f"Error: Static baseline checkpoint not found locally and download failed.")
            sys.exit(1)
        
    model = RHANLargeSTL10().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if k.startswith('_orig_mod.'):
            name = k[len('_orig_mod.'):]
        if name.startswith('module.'):
            name = name[len('module.'):]
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Loaded static TRADES Large baseline from: {ckpt_path}", flush=True)

    # 1. Evaluate clean accuracy and PGD-20 sweeps (eps = 0.031, 0.062, 0.094) on 500 samples
    n_samples = 500
    all_imgs = []
    all_lbls = []
    count = 0
    for imgs, lbls in test_loader:
        take = min(n_samples - count, imgs.size(0))
        if take <= 0:
            break
        all_imgs.append(imgs[:take])
        all_lbls.append(lbls[:take])
        count += take

    x_test = torch.cat(all_imgs, dim=0).to(device)
    y_test = torch.cat(all_lbls, dim=0).to(device)
    N = x_test.size(0)

    # Evaluate clean
    correct_clean = 0
    batch_size = 32
    for i in range(0, N, batch_size):
        x_b = x_test[i:i+batch_size]
        y_b = y_test[i:i+batch_size]
        with torch.no_grad():
            out = model(x_b)
            if isinstance(out, tuple):
                out = out[0]
            correct_clean += out.argmax(dim=1).eq(y_b).sum().item()
    print(f"Clean Accuracy (n={N}): {100.0 * correct_clean / N:.2f}%", flush=True)

    # Evaluate PGD sweeps
    for eps in [0.031, 0.062, 0.094]:
        correct = 0
        for i in range(0, N, batch_size):
            x_b = x_test[i:i+batch_size]
            y_b = y_test[i:i+batch_size]
            x_adv = run_pgd_eval(model, x_b, y_b, eps=eps, steps=20)
            with torch.no_grad():
                out = model(x_adv)
                if isinstance(out, tuple):
                    out = out[0]
                correct += out.argmax(dim=1).eq(y_b).sum().item()
        acc = 100.0 * correct / N
        print(f"PGD-20 Robustness (eps={eps:.3f}, n={N}): {acc:.2f}%", flush=True)

    # 2. Evaluate Corrected AutoAttack (n=200)
    n_samples_sq = 200
    x_test_norm = x_test[:n_samples_sq]
    y_test_sq = y_test[:n_samples_sq]
    
    mean_tensor = torch.tensor(STL10_MEAN, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(STL10_STD, device=device).view(1, 3, 1, 1)
    x_test_unnorm = (x_test_norm * std_tensor + mean_tensor).clamp(0.0, 1.0)

    wrapper = CorrectWrapper(model, STL10_MEAN, STL10_STD, device)
    from autoattack import AutoAttack

    print("\nRunning Corrected AutoAttack...", flush=True)
    adversary_aa = AutoAttack(wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=False)
    x_adv_aa = adversary_aa.run_standard_evaluation(x_test_unnorm, y_test_sq, bs=50)
    with torch.no_grad():
        logits_aa = wrapper(x_adv_aa)
        corrects_aa = logits_aa.argmax(dim=1).eq(y_test_sq).cpu()
    print(f"Corrected AutoAttack Accuracy (n={n_samples_sq}): {100.0 * corrects_aa.float().mean().item():.2f}%", flush=True)

    # 3. Evaluate Square Attack alone (n=200)
    print("\nRunning Square Attack alone...", flush=True)
    adversary_sq = AutoAttack(wrapper, norm='Linf', eps=0.031, version='standard', device=device, verbose=False)
    adversary_sq.attacks_to_run = ['square']
    x_adv_sq = adversary_sq.run_standard_evaluation(x_test_unnorm, y_test_sq, bs=50)
    with torch.no_grad():
        logits_sq = wrapper(x_adv_sq)
        corrects_sq = logits_sq.argmax(dim=1).eq(y_test_sq).cpu()
    print(f"Square Attack Robust Accuracy (n={n_samples_sq}): {100.0 * corrects_sq.float().mean().item():.2f}%", flush=True)

if __name__ == '__main__':
    main()

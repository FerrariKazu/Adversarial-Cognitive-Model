#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phase1_training'))
from model_rhan_v10 import RHANv10
from dataset_stl10 import get_stl10_loaders, STL10_CLASSES, STL10_MIN, STL10_MAX

def load_dotenv_fallback():
    """Manual fallback to load HF_TOKEN from .env file in project root."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for base_dir in [script_dir, os.path.join(script_dir, '..')]:
        env_path = os.path.join(base_dir, '.env')
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            v = v.strip().strip('"').strip("'")
                            os.environ[k.strip()] = v
            except Exception:
                pass

load_dotenv_fallback()

def download_checkpoint_from_hf(ckpt_path):
    filename = os.path.basename(ckpt_path)
    print(f"Checkpoint not found locally at {ckpt_path}. Attempting to download {filename} from Hugging Face...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        # Try both rhan-checkpoints-rolling and rhan-checkpoints
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
    parser = argparse.ArgumentParser(description='Quick PGD-20 Per-Class Check')
    parser.add_argument('--ckpt', type=str, default='checkpoints/rhan_stl10_v10_rolling.pth',
                        help='Path to the model checkpoint file')
    parser.add_argument('--eps', type=float, default=0.031,
                        help='Perturbation budget epsilon')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of PGD steps')
    parser.add_argument('--n', type=int, default=500,
                        help='Number of samples to evaluate on')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    if not os.path.exists(args.ckpt):
        success = download_checkpoint_from_hf(args.ckpt)
        if not success:
            print(f"Could not load checkpoint: {args.ckpt}. Exiting.", flush=True)
            sys.exit(1)

    print(f"Loading checkpoint: {args.ckpt}", flush=True)

    # Initialize model
    model = RHANv10().to(device)

    # Load state dict
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            print(f"Found model_state_dict from Epoch {ckpt.get('epoch', 'unknown')}", flush=True)
        elif 'model' in ckpt:
            state_dict = ckpt['model']
            print(f"Found model key in checkpoint", flush=True)
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"Found state_dict key in checkpoint", flush=True)
        else:
            state_dict = ckpt
            print(f"Treating checkpoint dict as raw state_dict", flush=True)
    else:
        state_dict = ckpt
        print(f"Treating checkpoint as raw state_dict", flush=True)

    # Clean up keys if needed (e.g. if saved via DistributedDataParallel or compile)
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

    # Load data
    print("Loading STL-10 test loader...", flush=True)
    _, test_loader = get_stl10_loaders(batch_size=args.batch_size)

    # Collect exactly n samples
    all_imgs = []
    all_lbls = []
    count = 0
    for imgs, lbls in test_loader:
        take = min(args.n - count, imgs.size(0))
        if take <= 0:
            break
        all_imgs.append(imgs[:take])
        all_lbls.append(lbls[:take])
        count += take

    x_test = torch.cat(all_imgs, dim=0).to(device)
    y_test = torch.cat(all_lbls, dim=0).to(device)
    N = x_test.size(0)
    print(f"Evaluating on {N} samples", flush=True)

    # PGD attack limits
    clip_min = torch.tensor(STL10_MIN, device=device).view(1, 3, 1, 1)
    clip_max = torch.tensor(STL10_MAX, device=device).view(1, 3, 1, 1)

    # Setup class tracking arrays
    class_totals = torch.zeros(10)
    class_clean_correct = torch.zeros(10)
    class_robust_correct = torch.zeros(10)

    # Run clean evaluation first
    print("\nRunning clean evaluation...", flush=True)
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            x_b = x_test[i:i+args.batch_size]
            y_b = y_test[i:i+args.batch_size]
            
            logits = model(x_b)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            
            correct = preds.eq(y_b).cpu()
            for label, corr in zip(y_b.cpu(), correct):
                class_totals[label] += 1
                if corr:
                    class_clean_correct[label] += 1

    # Run robust evaluation
    print(f"Running PGD-{args.steps} evaluation (eps={args.eps})...", flush=True)
    
    alpha = args.eps / 4
    for i in range(0, N, args.batch_size):
        x_b = x_test[i:i+args.batch_size]
        y_b = y_test[i:i+args.batch_size]
        
        # PGD attack generation
        delta = torch.empty_like(x_b).uniform_(-args.eps, args.eps)
        x_adv = (x_b + delta).clamp(clip_min, clip_max).detach()
        
        for step in range(args.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits, y_b)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            # Projection step
            delta = torch.clamp(x_adv - x_b, -args.eps, args.eps)
            x_adv = torch.clamp(x_b + delta, clip_min, clip_max).detach()
        
        # Evaluate robust predictions
        with torch.no_grad():
            logits_adv = model(x_adv)
            if isinstance(logits_adv, tuple):
                logits_adv = logits_adv[0]
            preds_adv = logits_adv.argmax(dim=1)
            
            correct_adv = preds_adv.eq(y_b).cpu()
            for label, corr in zip(y_b.cpu(), correct_adv):
                if corr:
                    class_robust_correct[label] += 1

    # Print summary per class
    print("\n" + "="*80)
    print(f" PGD-20 PER-CLASS ACCURACY REPORT (n={N}, eps={args.eps})")
    print("="*80)
    print(f"{'Class':<15} | {'Clean Acc':<12} | {'Robust Acc (PGD-20)':<22} | {'Count':<8}")
    print("-"*80)
    
    clean_total_acc = 0.0
    robust_total_acc = 0.0
    
    for c_idx, class_name in enumerate(STL10_CLASSES):
        count = class_totals[c_idx].item()
        if count == 0:
            continue
        clean_acc = 100.0 * class_clean_correct[c_idx].item() / count
        robust_acc = 100.0 * class_robust_correct[c_idx].item() / count
        
        # Mark car/truck classes
        marker = " ◄" if class_name in ['car', 'truck'] else ""
        print(f"{class_name:<15} | {clean_acc:>10.2f}% | {robust_acc:>20.2f}%{marker:<3} | {int(count):<8}")
        
        clean_total_acc += class_clean_correct[c_idx].item()
        robust_total_acc += class_robust_correct[c_idx].item()
        
    print("-"*80)
    overall_clean = 100.0 * clean_total_acc / N
    overall_robust = 100.0 * robust_total_acc / N
    print(f"{'OVERALL':<15} | {overall_clean:>10.2f}% | {overall_robust:>20.2f}%    | {N:<8}")
    print("="*80)

if __name__ == '__main__':
    main()

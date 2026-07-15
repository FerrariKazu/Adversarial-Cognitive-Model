#!/usr/bin/env python3
"""
Experiment: RHAN-v10 Active Inference Training on STL-10
=========================================================

Pre-registered hypothesis:
  "Dynamic precision control (Eq. III) and epistemic foraging (Eq. II)
   will maintain car/truck robustness through high-epsilon curriculum
   phases that destroy it under static TRADES β."

Key innovations over train_rhan_large_pseudolabel.py:
  1. Dynamic β from Π_D (per-image precision-weighted TRADES)
  2. Foraging consistency loss (clean vs adversarial gaze trajectories)
  3. Precision calibration loss (Π_D should be high when wrong)
  4. Halt efficiency loss (penalize unnecessary computation)
  5. Per-epoch diagnostic telemetry (β_dynamic, Π_D per class, steps used)

60-epoch curriculum starting from the Large pseudolabel checkpoint:
  Ep  1-20: ε=0.031, β_base=2.0, steps=7   (re-establish)
  Ep 21-40: ε=0.062, β_base=2.0, steps=10  (reproduce)
  Ep 41-60: ε=0.094, β_base=2.5, steps=10  (THE KEY TEST)

Saves to: checkpoints/rhan_stl10_v10_best.pth
"""

import os
import sys
import time
import random
import argparse
import subprocess
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_pretrained import RHANUnifiedSTL10
from model_rhan_v10 import RHANv10
from train_rhan_stl10_tdv import get_stl10_dataloaders


def load_dotenv_fallback():
    """Manual fallback to load HF_TOKEN from .env file in project root."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Look in the folder containing this script, and then one level up
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & BALANCED LOADING
# (Identical to train_rhan_large_pseudolabel.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class STL10RawUnlabeledDataset(Dataset):
    def __init__(self, data_root='./data/stl10'):
        self.stl10 = torchvision.datasets.STL10(
            data_root, split='unlabeled', download=True
        )
        self.mean = (0.4467, 0.4398, 0.4066)
        self.std  = (0.2603, 0.2566, 0.2713)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])

    def __len__(self):
        return len(self.stl10)

    def __getitem__(self, idx):
        img, _ = self.stl10[idx]
        return self.transform(img), idx


class CombinedSTL10Dataset(Dataset):
    """
    Combines 5K real labeled + pseudo-labeled unlabeled images.
    Real labels get weight 1.0, pseudo-labels get weight 0.5.
    """
    def __init__(self, real_imgs, real_labels,
                 unlabeled_dataset, pseudo_indices, pseudo_labels, transform=None):
        self.real_imgs = real_imgs.cpu()
        self.real_labels = real_labels.cpu()
        self.pseudo_indices = pseudo_indices.cpu()
        self.pseudo_labels = pseudo_labels.cpu()
        self.transform = transform

        self.n_real = len(real_imgs)
        self.n_pseudo = len(pseudo_indices)

        # Cache normalized pseudo-labeled images in RAM
        print(f"Caching {self.n_pseudo} pseudo-labeled images in RAM...", flush=True)
        self.cached_pseudo_imgs = torch.zeros(self.n_pseudo, 3, 96, 96, dtype=torch.float16)
        temp_loader = DataLoader(
            torch.utils.data.Subset(unlabeled_dataset, pseudo_indices.tolist()),
            batch_size=256, shuffle=False, num_workers=0,
            pin_memory=False
        )
        idx_start = 0
        for batch_imgs, _ in temp_loader:
            num_imgs = batch_imgs.size(0)
            self.cached_pseudo_imgs[idx_start : idx_start + num_imgs] = batch_imgs.to(torch.float16)
            idx_start += num_imgs

        print(f"Combined dataset: {self.n_real} real + "
              f"{self.n_pseudo} pseudo = {self.n_real+self.n_pseudo} total. Caching complete.", flush=True)

    def __len__(self):
        return self.n_real + self.n_pseudo

    def __getitem__(self, idx):
        if idx < self.n_real:
            img = self.real_imgs[idx]
            label = self.real_labels[idx]
            weight = 1.0
        else:
            img = self.cached_pseudo_imgs[idx - self.n_real].to(torch.float32)
            label = self.pseudo_labels[idx - self.n_real]
            weight = 0.5

        if self.transform:
            img = self.transform(img)
        return img, label, torch.tensor(weight, dtype=torch.float32)


class BalancedBatchSampler(Sampler):
    """
    Ensures each batch contains an exact balance of real and pseudo-labeled samples.
    """
    def __init__(self, real_indices, pseudo_indices, batch_size=64):
        self.real_indices = real_indices
        self.pseudo_indices = pseudo_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.num_batches = max(len(real_indices), len(pseudo_indices)) // self.half_batch

    def __iter__(self):
        for _ in range(self.num_batches):
            real_batch = np.random.choice(self.real_indices, self.half_batch, replace=True)
            pseudo_batch = np.random.choice(self.pseudo_indices, self.half_batch, replace=True)
            batch = np.concatenate([real_batch, pseudo_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PSEUDO-LABEL GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_pseudo_labels(model, unlabeled_loader, device, confidence_threshold=0.65):
    """
    Generate pseudo-labels from 100K unlabeled STL-10 images.
    """
    model.eval()
    pseudo_indices = []
    pseudo_labels = []
    confidence_scores = []

    print("Generating pseudo-labels from 100K unlabeled images...")
    with torch.no_grad():
        for batch_idx, (imgs, idx) in enumerate(unlabeled_loader):
            imgs = imgs.to(device, memory_format=torch.channels_last)

            with autocast('cuda'):
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1)
            conf, pred = probs.max(1)

            mask = conf >= confidence_threshold
            if mask.sum() > 0:
                pseudo_indices.append(idx[mask.cpu()])
                pseudo_labels.append(pred[mask].cpu())
                confidence_scores.append(conf[mask].cpu())

            if batch_idx % 50 == 0:
                kept = sum(len(x) for x in pseudo_labels)
                print(f"  Batch {batch_idx}/195 | Kept: {kept} images")

    if len(pseudo_indices) == 0:
        print("WARNING: No high confidence pseudo labels generated!")
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0)

    pseudo_indices = torch.cat(pseudo_indices, dim=0)
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    confidence_scores = torch.cat(confidence_scores, dim=0)

    classes = ['airplane','bird','car','cat','deer',
               'dog','horse','monkey','ship','truck']
    print("\nPseudo-label class distribution and mean confidence:")
    for c in range(10):
        c_mask = (pseudo_labels == c)
        n = c_mask.sum().item()
        if n > 0:
            mean_conf = confidence_scores[c_mask].mean().item()
            print(f"  {classes[c]:<12}: {n:<6} images (mean confidence: {mean_conf:.4f})")
        else:
            print(f"  {classes[c]:<12}: 0      images")

    print(f"\nTotal pseudo-labeled: {len(pseudo_indices)} / 100000 ({100*len(pseudo_indices)/100000:.1f}%)")

    return pseudo_indices, pseudo_labels, confidence_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DYNAMIC TRADES LOSS (THE CORE INNOVATION)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def dynamic_trades_loss(model, imgs, labels, weights,
                        x_adv, beta_base):
    """
    TRADES loss with precision-weighted β.

    Images where the model is uncertain (high Π_D, high prediction error)
    receive stronger adversarial training pressure.
    Images where the model is confident (low Π_D) receive
    lighter pressure, preserving clean accuracy.

    This breaks the static clean-robust tradeoff.

    Returns:
        loss:       scalar — total weighted loss
        traj_c:     dict  — clean trajectory for diagnostics
        traj_a:     dict  — adversarial trajectory for diagnostics
        beta_dyn:   (B,)  — per-image dynamic β values
    """
    # Forward pass on clean (with trajectory to get Π_D)
    logits_c, traj_c = model(imgs, return_trajectory=True)
    logits_a, traj_a = model(x_adv, return_trajectory=True)

    # Per-image dynamic β from precision
    # High uncertainty → higher β → stronger robustness training
    if len(traj_c['precisions']) > 0:
        final_precision_c = traj_c['precisions'][-1]  # (B,)
    else:
        final_precision_c = torch.full((imgs.shape[0],), 0.5, device=imgs.device)

    beta_dynamic = beta_base * (0.5 + final_precision_c)  # range: [β/2, β*1.5]

    # CE loss on clean
    ce = nn.CrossEntropyLoss(reduction='none')
    l_ce = ce(logits_c, labels)

    # KL divergence (TRADES robustness term)
    l_kl = F.kl_div(
        F.log_softmax(logits_a.float(), dim=1),
        F.softmax(logits_c.float().detach(), dim=1),
        reduction='none'
    ).sum(dim=1)

    # Dynamic weighting: uncertain images get more robustness training
    l_total = (l_ce + beta_dynamic * l_kl) * weights.to(l_ce.device)

    return l_total.mean(), traj_c, traj_a, beta_dynamic.detach()


def compute_auxiliary_losses(model, traj_c, traj_a, logits_c, labels):
    """
    Compute the three auxiliary loss terms.

    Returns:
        l_foraging:      scalar — foraging trajectory consistency
        l_precision_cal: scalar — precision calibration
        l_halt:          scalar — halt efficiency
    """
    # 1. Foraging consistency: adversarial foraging should visit
    #    the SAME regions as clean foraging
    if len(traj_a['actions']) > 0 and len(traj_c['actions']) > 0:
        # Pad to same length if different steps used
        min_steps = min(len(traj_a['actions']), len(traj_c['actions']))
        actions_a = torch.stack(traj_a['actions'][:min_steps])
        actions_c = torch.stack(traj_c['actions'][:min_steps])
        l_foraging = F.mse_loss(actions_a, actions_c.detach())
    else:
        l_foraging = torch.tensor(0.0, device=logits_c.device)

    # 2. Precision calibration: Π_D should be high when wrong
    if len(traj_c['precisions']) > 0:
        with torch.no_grad():
            correct = logits_c.argmax(1).eq(labels).float()
            target_precision = 1 - correct  # uncertain when wrong
        l_precision_cal = F.mse_loss(
            traj_c['precisions'][-1], target_precision)
    else:
        l_precision_cal = torch.tensor(0.0, device=logits_c.device)

    # 3. Halt efficiency: penalize unnecessary computation
    mean_steps = torch.tensor(
        float(traj_c['steps']), device=logits_c.device)
    max_steps = model.max_steps if hasattr(model, 'max_steps') else 4
    l_halt = mean_steps / max_steps

    return l_foraging, l_precision_cal, l_halt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIAGNOSTIC TELEMETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EpochDiagnostics:
    """
    Collects per-epoch diagnostic telemetry for the active inference components.
    """
    CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
               'dog', 'horse', 'monkey', 'ship', 'truck']

    def __init__(self):
        self.reset()

    def reset(self):
        self.beta_dynamics = []
        self.precisions_per_class = {c: [] for c in range(10)}
        self.steps_used = []
        self.foraging_entropy = []
        self.errors_per_class = {c: [] for c in range(10)}

    def update(self, beta_dyn, traj_c, labels):
        """Record diagnostics from one batch."""
        self.beta_dynamics.append(beta_dyn.cpu())

        if len(traj_c['precisions']) > 0:
            prec = traj_c['precisions'][-1].cpu()
            for c in range(10):
                mask = (labels.cpu() == c)
                if mask.any():
                    self.precisions_per_class[c].append(prec[mask])

        if len(traj_c['errors']) > 0:
            err = traj_c['errors'][-1].cpu()
            for c in range(10):
                mask = (labels.cpu() == c)
                if mask.any():
                    self.errors_per_class[c].append(err[mask])

        self.steps_used.append(traj_c['steps'])

        # Gaze trajectory entropy: measure diversity of foraging actions
        if len(traj_c['actions']) > 1:
            actions = torch.stack(traj_c['actions'])  # (T, B, 2)
            diffs = (actions[1:] - actions[:-1]).norm(dim=-1)  # (T-1, B)
            entropy = diffs.mean().item()
            self.foraging_entropy.append(entropy)

    def report(self, epoch, eps):
        """Print epoch diagnostic summary."""
        beta_all = torch.cat(self.beta_dynamics)
        mean_steps = np.mean(self.steps_used)

        print(f"\n{'─'*60}")
        print(f"  RHAN-v10 Diagnostics — Epoch {epoch} (ε={eps:.3f})")
        print(f"{'─'*60}")
        print(f"  β_dynamic: mean={beta_all.mean():.4f} "
              f"std={beta_all.std():.4f} "
              f"min={beta_all.min():.4f} max={beta_all.max():.4f}")
        print(f"  Steps used: mean={mean_steps:.2f}")

        if self.foraging_entropy:
            print(f"  Foraging entropy: {np.mean(self.foraging_entropy):.4f}")

        print(f"  Π_D per class (mean):")
        for c in range(10):
            if self.precisions_per_class[c]:
                prec = torch.cat(self.precisions_per_class[c])
                err_list = self.errors_per_class[c]
                err_str = ""
                if err_list:
                    err = torch.cat(err_list)
                    err_str = f" | error={err.mean():.4f}"
                marker = " ◄" if self.CLASSES[c] in ('car', 'truck') else ""
                print(f"    {self.CLASSES[c]:<12}: Π_D={prec.mean():.4f}{err_str}{marker}")
        print(f"{'─'*60}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HUGGING FACE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_checkpoint_exists(ckpt_path):
    if os.path.exists(ckpt_path):
        return ckpt_path
    print(f"Checkpoint not found locally at {ckpt_path}. Attempting to download from Hugging Face...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
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
        filename = os.path.basename(ckpt_path)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        print(f"Downloading {filename} from FerrariKazu/rhan-checkpoints...", flush=True)
        downloaded_cache_path = hf_hub_download(
            repo_id='FerrariKazu/rhan-checkpoints',
            filename=filename,
            repo_type='dataset',
            token=hf_token
        )
        shutil.copy2(downloaded_cache_path, ckpt_path)
        print(f"Successfully downloaded to: {ckpt_path}", flush=True)
        return ckpt_path
    except Exception as e:
        err_str = str(e)
        if "404" in err_str:
            print(f"Checkpoint not found on Hugging Face (404). Starting from scratch.", flush=True)
            return ckpt_path
        else:
            print(f"\n[FATAL ERROR]: Hugging Face download failed: {e}", flush=True)
            print("To prevent accidentally overwriting your training progress, the script is aborting.", flush=True)
            sys.exit(1)


def sync_to_hf(file_path):
    if not os.path.exists(file_path):
        return
    import threading

    # Create copy synchronously to prevent write-during-upload race conditions
    sync_path = file_path + ".sync"
    try:
        shutil.copy2(file_path, sync_path)
    except Exception as e:
        print(f"Failed to create sync copy for upload: {e}", flush=True)
        return

    def _async_sync(temp_path, original_filename):
        try:
            from huggingface_hub import HfApi, create_repo
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
            if hf_token:
                api = HfApi(token=hf_token)
                username = api.whoami()['name']
                repo_id = f"{username}/rhan-checkpoints"

                try:
                    create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=hf_token)
                except Exception:
                    pass

                print(f"Syncing {original_filename} to Hugging Face ({repo_id}) asynchronously...", flush=True)
                api.upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=original_filename,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token
                )
                print(f"Async sync complete for {original_filename}.", flush=True)
            else:
                print("WARNING: HF_TOKEN not found. Skipping Hugging Face sync.", flush=True)
        except Exception as e:
            print(f"Async Hugging Face sync failed: {e}", flush=True)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    threading.Thread(target=_async_sync, args=(sync_path, os.path.basename(file_path)), daemon=True).start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_optimal_dataloader_config(dataset, sampler, is_ddp=False, rank=0):
    configs = [
        {"num_workers": 0, "persistent_workers": False, "prefetch_factor": None},
        {"num_workers": 2, "persistent_workers": True, "prefetch_factor": 3},
        {"num_workers": 4, "persistent_workers": True, "prefetch_factor": 4},
    ]

    cpu_count = os.cpu_count() or 2
    max_workers = max(1, cpu_count // (int(os.environ.get("WORLD_SIZE", 1))))
    configs = [c for c in configs if c["num_workers"] <= max_workers]

    best_config = configs[0]
    min_time = float('inf')

    if rank == 0:
        print("\nAuto-tuning DataLoader configuration...")

    for config in configs:
        kwargs = {"pin_memory": True}
        if config["num_workers"] > 0:
            kwargs["num_workers"] = config["num_workers"]
            kwargs["persistent_workers"] = config["persistent_workers"]
            kwargs["prefetch_factor"] = config["prefetch_factor"]

        loader = DataLoader(dataset, batch_sampler=sampler, **kwargs)

        t_total = 0.0
        try:
            iterator = iter(loader)
            for _ in range(2):
                next(iterator)
            t0 = time.time()
            for _ in range(5):
                next(iterator)
            t_total = time.time() - t0
            if rank == 0:
                print(f"  num_workers={config['num_workers']}, persistent={config['persistent_workers']}, "
                      f"prefetch={config['prefetch_factor']} -> time: {t_total:.4f}s")

            if t_total < min_time:
                min_time = t_total
                best_config = config
        except Exception as e:
            if rank == 0:
                print(f"  Config num_workers={config['num_workers']} failed: {e}")

    if rank == 0:
        print(f"Optimal DataLoader Config: num_workers={best_config['num_workers']}, "
              f"persistent_workers={best_config['persistent_workers']}, "
              f"prefetch_factor={best_config['prefetch_factor']}\n")
    return best_config


def main():
    parser = argparse.ArgumentParser(description='RHAN-v10 Active Inference Training')
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8 for T4, use 16+ for A100)')
    parser.add_argument('--unlabeled-batch-size', type=int, default=256,
                        help='Batch size for pseudo-label generation')
    parser.add_argument('--accum-steps', type=int, default=32,
                        help='Gradient accumulation steps (default: 32 for effective batch size 256)')
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--labeling-ckpt', type=str, default='')
    parser.add_argument('--target-ckpt', type=str, default='')
    parser.add_argument('--fixed-samples-per-epoch', type=int, default=0,
                        help='If > 0, normalizes epoch length to this many images')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile()')
    parser.add_argument('--dry-run', action='store_true', help='Runs a single batch step and exits')
    # v10-specific args
    parser.add_argument('--max-foraging-steps', type=int, default=4,
                        help='Maximum foraging steps in tripartite loop')
    parser.add_argument('--fovea-size', type=int, default=48,
                        help='Foveal crop size in pixels')
    parser.add_argument('--metabolic-cost', type=float, default=0.05,
                        help='Metabolic cost threshold for thermodynamic halt')
    # Loss weights
    parser.add_argument('--w-trades', type=float, default=0.55)
    parser.add_argument('--w-foraging', type=float, default=0.20)
    parser.add_argument('--w-precision', type=float, default=0.15)
    parser.add_argument('--w-halt', type=float, default=0.10)
    args, _ = parser.parse_known_args()

    # DDP Initialization
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
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed + rank)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if rank == 0:
        print(f"{'═'*60}")
        print(f"  RHAN-v10: Tripartite Active Inference Training")
        print(f"  Device: {device} | DDP: {is_ddp} (world_size={world_size})")
        print(f"  Max foraging steps: {args.max_foraging_steps}")
        print(f"  Fovea size: {args.fovea_size}")
        print(f"  Metabolic cost: {args.metabolic_cost}")
        print(f"  Loss weights: trades={args.w_trades}, foraging={args.w_foraging}, "
              f"precision={args.w_precision}, halt={args.w_halt}")
        print(f"{'═'*60}", flush=True)

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── 1. Generate pseudo-labels ────────────────────────────────
    pseudo_indices = None
    pseudo_lbls = None

    if rank == 0:
        labeling_model = RHANUnifiedSTL10().to(device, memory_format=torch.channels_last)
        best_labeling_ckpt = args.labeling_ckpt if args.labeling_ckpt else os.path.join(ckpt_dir, 'rhan_stl10_pseudolabel_best.pth')
        best_labeling_ckpt = ensure_checkpoint_exists(best_labeling_ckpt)
        if os.path.exists(best_labeling_ckpt):
            labeling_model.load_state_dict(torch.load(best_labeling_ckpt, map_location=device))
            print(f"Loaded labeling model checkpoint: {best_labeling_ckpt}", flush=True)
        else:
            print(f"Error: Labeling checkpoint {best_labeling_ckpt} not found!", flush=True)
            sys.exit(1)

        num_cpus = os.cpu_count() or 2
        num_workers = min(4, num_cpus)
        unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)
        pseudo_indices, pseudo_lbls, _ = generate_pseudo_labels(labeling_model, unlabeled_loader, device, args.confidence_threshold)

        del labeling_model
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        if is_ddp:
            torch.save({'indices': pseudo_indices, 'labels': pseudo_lbls}, os.path.join(ckpt_dir, 'temp_pseudo_labels.pth'))

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    if rank != 0:
        temp_data = torch.load(os.path.join(ckpt_dir, 'temp_pseudo_labels.pth'), map_location='cpu')
        pseudo_indices = temp_data['indices']
        pseudo_lbls = temp_data['labels']

    if len(pseudo_indices) == 0:
        if rank == 0:
            print("Error: No pseudo-labels generated. Exiting.", flush=True)
        sys.exit(1)

    # ── 2. Load raw real labeled training data ───────────────────
    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
    ])
    trainset_raw = torchvision.datasets.STL10(args.data_root, split='train', download=True)
    num_real = len(trainset_raw)
    real_imgs = torch.zeros(num_real, 3, 96, 96, dtype=torch.float32)
    for i in range(num_real):
        real_imgs[i] = norm_transform(trainset_raw[i][0])
    real_labels = torch.tensor([trainset_raw[i][1] for i in range(len(trainset_raw))])
    unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)

    # ── 3. Combined dataset & balanced loader ────────────────────
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
    ])
    combined_dataset = CombinedSTL10Dataset(real_imgs, real_labels, unlabeled_dataset, pseudo_indices, pseudo_lbls, transform=train_transform)

    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(real_imgs) + len(pseudo_indices)))

    if is_ddp:
        random.Random(args.seed + rank).shuffle(real_indices)
        random.Random(args.seed + rank).shuffle(pseudo_indices_list)
        real_indices = real_indices[rank::world_size]
        pseudo_indices_list = pseudo_indices_list[rank::world_size]

    sampler_batch_size = args.batch_size // world_size if is_ddp else args.batch_size
    sampler = BalancedBatchSampler(real_indices, pseudo_indices_list, batch_size=sampler_batch_size)

    optimal_config = find_optimal_dataloader_config(combined_dataset, sampler, is_ddp, rank)

    loader_kwargs = {"pin_memory": True}
    if optimal_config["num_workers"] > 0:
        loader_kwargs["num_workers"] = optimal_config["num_workers"]
        loader_kwargs["persistent_workers"] = optimal_config["persistent_workers"]
        loader_kwargs["prefetch_factor"] = optimal_config["prefetch_factor"]

    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, **loader_kwargs)

    _, testloader, stl_min, stl_max = get_stl10_dataloaders(args.data_root, batch_size=64)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    # Clean up temp file
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
    if rank == 0:
        temp_file = os.path.join(ckpt_dir, 'temp_pseudo_labels.pth')
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # ── 4. Instantiate RHAN-v10 model ────────────────────────────
    model = RHANv10(
        max_foraging_steps=args.max_foraging_steps,
        fovea_size=args.fovea_size,
        metabolic_cost=args.metabolic_cost,
    ).to(device, memory_format=torch.channels_last)

    # ── 5. Load base checkpoint (Large pseudolabel) ──────────────
    best_target_ckpt = args.target_ckpt if args.target_ckpt else os.path.join(ckpt_dir, 'rhan_stl10_large_pseudolabel_best.pth')
    best_target_ckpt = ensure_checkpoint_exists(best_target_ckpt)
    if os.path.exists(best_target_ckpt):
        ckpt = torch.load(best_target_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"Loaded base checkpoint: {best_target_ckpt}", flush=True)
            print(f"  Missing keys (new v10 components): {len(missing)}", flush=True)
            print(f"  Unexpected keys: {len(unexpected)}", flush=True)
            if unexpected:
                print(f"  WARNING unexpected keys: {unexpected[:5]}", flush=True)
    else:
        if rank == 0:
            print(f"Warning: Base checkpoint {best_target_ckpt} not found! Initializing model randomly.", flush=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model instantiated with {total_params:,} parameters", flush=True)

    # Enable torch.compile if selected
    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile()...", flush=True)
        model = torch.compile(model, mode="default")

    # Multi-GPU / DDP support
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # v10 has conditional computation paths
            broadcast_buffers=False
        )
    elif torch.cuda.device_count() > 1:
        if rank == 0:
            print(f"Using {torch.cuda.device_count()} GPUs for training (DataParallel)", flush=True)
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, 'module') else model

    # ── 6. Curriculum Setup (60 epochs total) ────────────────────
    # format: (start_epoch, end_epoch, eps, beta, steps, lr)
    curriculum = [
        (1,  20, 0.031, 2.0, 7,  0.003),
        (21, 40, 0.062, 2.0, 10, 0.002),
        (41, 60, 0.094, 2.5, 10, 0.001),
    ]

    scaler = GradScaler('cuda')
    best_acc = 0.0
    start_epoch = 1
    current_phase_start = None
    optimizer = None
    scheduler = None

    best_path = os.path.join(ckpt_dir, 'rhan_stl10_v10_best.pth')
    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_v10_rolling.pth')

    # ── 7. Automatic Resume Check ────────────────────────────────
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

    local_epoch = -1
    checkpoint_data = None
    if os.path.exists(rolling_path):
        try:
            local_data = torch.load(rolling_path, map_location='cpu')
            local_epoch = local_data.get('epoch', -1)
        except Exception:
            pass

    if rank == 0:
        try:
            from huggingface_hub import hf_hub_download
            print("Checking for a newer checkpoint on Hugging Face...", flush=True)
            temp_rolling_path = hf_hub_download(
                repo_id='FerrariKazu/rhan-checkpoints',
                filename='rhan_stl10_v10_rolling.pth',
                repo_type='dataset',
                token=hf_token
            )
            remote_data = torch.load(temp_rolling_path, map_location='cpu')
            remote_epoch = remote_data.get('epoch', -1)
            if remote_epoch > local_epoch:
                print(f"Hugging Face has a newer checkpoint (Epoch {remote_epoch}) than local (Epoch {local_epoch}). Synchronizing...", flush=True)
                os.makedirs(os.path.dirname(rolling_path), exist_ok=True)
                shutil.copy(temp_rolling_path, rolling_path)
        except Exception as e:
            print(f"Hugging Face sync check skipped/failed: {e}", flush=True)

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    if os.path.exists(rolling_path):
        if rank == 0:
            print(f"\nFound rolling checkpoint at {rolling_path}. Attempting to resume...", flush=True)
        checkpoint_data = torch.load(rolling_path, map_location=device)
        raw_model.load_state_dict(checkpoint_data['model'])
        best_acc = checkpoint_data.get('best_acc', 0.0)
        start_epoch = checkpoint_data['epoch'] + 1
        if rank == 0:
            print(f"Resuming from Epoch {start_epoch} (Best validation accuracy so far: {best_acc:.2f}%)")

    # ── 8. Training loop ─────────────────────────────────────────
    diagnostics = EpochDiagnostics()

    for epoch in range(start_epoch, 61):
        t0 = time.time()
        diagnostics.reset()

        # Determine current phase parameters
        for p_start, p_end, eps, beta, steps, lr in curriculum:
            if p_start <= epoch <= p_end:
                phase_params = (eps, beta, steps)
                phase_lr = lr
                if current_phase_start != p_start:
                    current_phase_start = p_start
                    optimizer = optim.SGD(
                        model.parameters(), lr=phase_lr,
                        momentum=0.9, weight_decay=1e-4,
                        foreach=True
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=p_end - p_start + 1, eta_min=phase_lr * 0.1
                    )
                    # Restore state dicts if resuming
                    if epoch == start_epoch and checkpoint_data is not None and 'optimizer' in checkpoint_data:
                        optimizer.load_state_dict(checkpoint_data['optimizer'])
                        scheduler.load_state_dict(checkpoint_data['scheduler'])
                        if rank == 0:
                            print("Restored optimizer and scheduler state dicts.")
                    if rank == 0:
                        print(f"\n--- Epoch {epoch}: New optimizer phase {p_start}-{p_end} (lr={phase_lr}) ---")
                break

        eps, beta, steps = phase_params

        # Training loop
        model.train()
        total_loss = n_total = correct = 0

        total_batch_size = args.batch_size * world_size if is_ddp else args.batch_size
        if args.fixed_samples_per_epoch > 0:
            num_batches = max(1, args.fixed_samples_per_epoch // total_batch_size)
        else:
            num_batches = min(len(trainloader), 600)

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (imgs, lbls, weights) in enumerate(trainloader):
            if batch_idx >= num_batches:
                break

            is_accumulating = (batch_idx + 1) % args.accum_steps != 0 and (batch_idx + 1) < num_batches
            if is_ddp and is_accumulating:
                sync_context = model.no_sync()
            else:
                from contextlib import nullcontext
                sync_context = nullcontext()

            with sync_context:
                imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)

                # ── PGD adversarial example generation ───────────
                raw_model.eval()
                with torch.no_grad():
                    with autocast('cuda'):
                        logits_c_pgd = raw_model(imgs)
                probs_c = F.softmax(logits_c_pgd.float(), dim=1)

                x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
                x_adv = torch.clamp(x_adv, stl_min, stl_max)
                for _ in range(steps):
                    x_adv.requires_grad_(True)
                    with torch.enable_grad():
                        with autocast('cuda'):
                            logits_a_pgd = raw_model(x_adv)
                            loss_adv = F.kl_div(
                                F.log_softmax(logits_a_pgd.float(), dim=1),
                                probs_c, reduction='batchmean'
                            )
                    grad = torch.autograd.grad(loss_adv, x_adv)[0]
                    x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                    delta = torch.clamp(x_adv - imgs, -eps, eps)
                    x_adv = torch.clamp(imgs + delta, stl_min, stl_max).detach()

                model.train()

                # ── Combined loss computation ────────────────────
                with autocast('cuda'):
                    # Dynamic TRADES loss (core innovation)
                    l_trades, traj_c, traj_a, beta_dyn = dynamic_trades_loss(
                        raw_model, imgs, lbls, weights, x_adv, beta)

                    # Auxiliary losses
                    with autocast('cuda'):
                        logits_c_aux = raw_model(imgs)
                    l_foraging, l_precision_cal, l_halt = compute_auxiliary_losses(
                        raw_model, traj_c, traj_a, logits_c_aux, lbls)

                    # Total combined loss
                    loss = (args.w_trades * l_trades +
                            args.w_foraging * l_foraging +
                            args.w_precision * l_precision_cal +
                            args.w_halt * l_halt) / args.accum_steps

                scaler.scale(loss).backward()

                # Collect diagnostics
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
                      f"Loss: {l_trades.item():.4f} | "
                      f"β_dyn: {beta_dyn.mean():.3f} | "
                      f"Steps: {traj_c['steps']}", flush=True)

            if args.dry_run:
                if rank == 0:
                    print("Dry-run mode active. Successfully completed 1 training step.", flush=True)
                break

        # Flush remaining accumulated gradients
        if num_batches % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ── Validation (clean test) ──────────────────────────────
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
            val_acc = 100. * val_correct / val_total

        if is_ddp:
            val_acc_tensor = torch.tensor([val_acc], device=device)
            import torch.distributed as dist
            dist.broadcast(val_acc_tensor, src=0)
            val_acc = val_acc_tensor.item()

        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            marker = ' ★'
            if rank == 0:
                torch.save(raw_model.state_dict(), best_path)
                sync_to_hf(best_path)

        if rank == 0:
            t_epoch = time.time() - t0
            total_images = n_total * world_size if is_ddp else n_total
            images_per_sec = total_images / t_epoch if t_epoch > 0 else 0
            epochs_per_hour = 3600.0 / t_epoch if t_epoch > 0 else 0
            print(
                f"Epoch {epoch:03d}/060 (ε={eps:.3f}) | Loss:{total_loss/max(n_total,1):.3f} | "
                f"TrAcc:{100.*correct/max(n_total,1):.1f}% TeAcc:{val_acc:.1f}% | "
                f"Throughput:{images_per_sec:.2f} img/sec ({epochs_per_hour:.2f} epochs/hour) | "
                f"{t_epoch:.0f}s{marker}", flush=True
            )

            # Print diagnostic telemetry
            diagnostics.report(epoch, eps)

            # Save rolling checkpoint
            torch.save({
                'epoch': epoch,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
            }, rolling_path)
            sync_to_hf(rolling_path)

        if args.dry_run:
            if is_ddp:
                import torch.distributed as dist
                dist.barrier()
            break

        if is_ddp:
            import torch.distributed as dist
            dist.barrier()

    if rank == 0:
        print(f"\n{'═'*60}")
        print(f"  Training Complete. Best Model saved to {best_path}")
        print(f"  Best validation accuracy: {best_acc:.2f}%")
        print(f"{'═'*60}")

        if not args.dry_run:
            print("\nLaunching final evaluation...")
            eval_script = os.path.join(script_dir, '..', 'run_eval_stl10.py')
            try:
                subprocess.run([
                    sys.executable, eval_script,
                    "--model-size", "v10",
                    "--checkpoint", best_path,
                    "--samples", "1000",
                    "--batch-size", "64"
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Evaluation script failed: {e}")
            except FileNotFoundError:
                print("Evaluation script not found. Skipping automatic evaluation.")
        else:
            print("Dry-run complete. Skipping evaluation script execution.")


if __name__ == '__main__':
    main()

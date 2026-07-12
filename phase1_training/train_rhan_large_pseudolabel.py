#!/usr/bin/env python3
"""
Experiment: Large model + Pseudo-label training on STL-10
=========================================================

1. Generates pseudo-labels using the BEST available model (rhan_stl10_pseudolabel_best.pth) at confidence threshold 0.65.
2. Combines 5K real labeled STL-10 images + the generated pseudo-labels.
3. Initializes the Large model (RHANLargeSTL10) from checkpoints/rhan_stl10_large_video_tdv.pth.
4. Trains under a 120-epoch curriculum with standard TRADES loss (no TDV loss).
5. Employs gradient checkpointing and FP16 mixed precision to fit in memory.
6. Saves the best model to checkpoints/rhan_stl10_large_pseudolabel_best.pth.
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
from model_rhan_stl10_large import RHANLargeSTL10
from train_rhan_stl10_tdv import get_stl10_dataloaders

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION & BALANCED LOADING
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
        
        # Cache normalized pseudo-labeled images in RAM using pre-allocated tensor to avoid OOM
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
    Uses the loaded checkpoint's predictions.
    Saves only the indices and predictions to minimize RAM footprint.
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
    
    # Per-class distribution and mean confidence of pseudo-labels
    print("\nPseudo-label class distribution and mean confidence:")
    classes = ['airplane','bird','car','cat','deer',
               'dog','horse','monkey','ship','truck']
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
# WEIGHTED TRADES LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def trades_loss_weighted(model, imgs, lbls, weights, 
                          x_adv, beta):
    """
    TRADES loss with per-sample weights.
    """
    ce = nn.CrossEntropyLoss(reduction='none')
    
    logits_c = model(imgs)
    logits_a = model(x_adv)
    
    l_ce = ce(logits_c, lbls)
    
    l_kl = F.kl_div(
        F.log_softmax(logits_a.float(), dim=1),
        F.softmax(logits_c.float().detach(), dim=1),
        reduction='none'
    ).sum(dim=1)
    
    l_total = (l_ce + beta * l_kl) * weights.to(l_ce.device)
    return l_total.mean()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HUGGING FACE DOWNLOAD HELPER
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
                from kaggle_secrets import UserSecretsClient
                hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
        if not hf_token:
            try:
                from google.colab import userdata
                hf_token = userdata.get('HF_TOKEN')
            except Exception:
                pass
        filename = os.path.basename(ckpt_path)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        print(f"Downloading {filename} from FerrariKazu/rhan-checkpoints...", flush=True)
        downloaded_path = hf_hub_download(
            repo_id='FerrariKazu/rhan-checkpoints',
            filename=filename,
            repo_type='dataset',
            local_dir=os.path.dirname(ckpt_path),
            token=hf_token
        )
        print(f"Successfully downloaded to: {downloaded_path}", flush=True)
        return downloaded_path
    except Exception as e:
        err_str = str(e)
        if "404" in err_str:
            print(f"Checkpoint not found on Hugging Face (404). Starting from scratch.", flush=True)
            return ckpt_path
        else:
            print(f"\n[FATAL ERROR]: Hugging Face download failed: {e}", flush=True)
            print("To prevent accidentally overwriting your training progress, the script is aborting.", flush=True)
            print("Please check your HF_TOKEN, Google Drive storage space, or internet connection, then try again.", flush=True)
            sys.exit(1)

def sync_to_hf(file_path):
    if not os.path.exists(file_path):
        return
    import shutil
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

    # Launch in a daemon thread so it doesn't block the training loop
    threading.Thread(target=_async_sync, args=(sync_path, os.path.basename(file_path)), daemon=True).start()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_optimal_dataloader_config(dataset, sampler, is_ddp=False, rank=0):
    import time
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
        kwargs = {
            "pin_memory": True,
        }
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
                print(f"  num_workers={config['num_workers']}, persistent={config['persistent_workers']}, prefetch={config['prefetch_factor']} -> time: {t_total:.4f}s")
            
            if t_total < min_time:
                min_time = t_total
                best_config = config
        except Exception as e:
            if rank == 0:
                print(f"  Config num_workers={config['num_workers']} failed: {e}")
                
    if rank == 0:
        print(f"Optimal DataLoader Config: num_workers={best_config['num_workers']}, persistent_workers={best_config['persistent_workers']}, prefetch_factor={best_config['prefetch_factor']}\n")
    return best_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training combined loader (default: 16 for T4)')
    parser.add_argument('--unlabeled-batch-size', type=int, default=256, help='Batch size for pseudo-label generation (default: 256 for T4)')
    parser.add_argument('--accum-steps', type=int, default=16, help='Gradient accumulation steps (default: 16 for effective batch size 256)')
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--labeling-ckpt', type=str, default='')
    parser.add_argument('--target-ckpt', type=str, default='')
    parser.add_argument('--fixed-samples-per-epoch', type=int, default=0, help='If > 0, normalizes epoch length to this many images')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile()')
    parser.add_argument('--dry-run', action='store_true', help='Runs a single epoch step and exits')
    args, _ = parser.parse_known_args()

    # DDP Initialization check
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
    
    # CUDA Optimization flags
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if rank == 0:
        print(f"Device: {device} | DDP: {is_ddp} (world_size={world_size})", flush=True)

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Generate new pseudo-labels using the BEST labeling model (Only on Rank 0 to save VRAM and communication overhead)
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

        # Free memory immediately
        del labeling_model
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        # Save to temp file for other ranks to load
        if is_ddp:
            torch.save({'indices': pseudo_indices, 'labels': pseudo_lbls}, os.path.join(ckpt_dir, 'temp_pseudo_labels.pth'))

    if is_ddp:
        dist.barrier() # wait for rank 0 to finish generation

    if rank != 0:
        # Load from temp file and keep on CPU to avoid device mismatch during DataLoader collation
        temp_data = torch.load(os.path.join(ckpt_dir, 'temp_pseudo_labels.pth'), map_location='cpu')
        pseudo_indices = temp_data['indices']
        pseudo_lbls = temp_data['labels']

    if len(pseudo_indices) == 0:
        if rank == 0:
            print("Error: No pseudo-labels generated. Exiting.", flush=True)
        sys.exit(1)

    # 2. Load raw real labeled training data
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

    # 3. Combined dataset & balanced loader
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
    ])
    combined_dataset = CombinedSTL10Dataset(real_imgs, real_labels, unlabeled_dataset, pseudo_indices, pseudo_lbls, transform=train_transform)
    
    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(real_imgs) + len(pseudo_indices)))
    
    # Shard indices across DDP ranks
    if is_ddp:
        import random
        random.Random(args.seed + rank).shuffle(real_indices)
        random.Random(args.seed + rank).shuffle(pseudo_indices_list)
        real_indices = real_indices[rank::world_size]
        pseudo_indices_list = pseudo_indices_list[rank::world_size]
        
    sampler_batch_size = args.batch_size // world_size if is_ddp else args.batch_size
    sampler = BalancedBatchSampler(real_indices, pseudo_indices_list, batch_size=sampler_batch_size)
    
    # Auto-tune DataLoader configurations
    optimal_config = find_optimal_dataloader_config(combined_dataset, sampler, is_ddp, rank)
    
    loader_kwargs = {
        "pin_memory": True,
    }
    if optimal_config["num_workers"] > 0:
        loader_kwargs["num_workers"] = optimal_config["num_workers"]
        loader_kwargs["persistent_workers"] = optimal_config["persistent_workers"]
        loader_kwargs["prefetch_factor"] = optimal_config["prefetch_factor"]
        
    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, **loader_kwargs)

    _, testloader, stl_min, stl_max = get_stl10_dataloaders(args.data_root, batch_size=64)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    # Clean up temp file
    if is_ddp:
        dist.barrier()
    if rank == 0:
        temp_file = os.path.join(ckpt_dir, 'temp_pseudo_labels.pth')
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # 4. Instantiate target model (Large)
    model = RHANLargeSTL10().to(device, memory_format=torch.channels_last)

    # 5. Load Target (TDV pretrained) checkpoint
    best_target_ckpt = args.target_ckpt if args.target_ckpt else os.path.join(ckpt_dir, 'rhan_stl10_large_video_tdv.pth')
    best_target_ckpt = ensure_checkpoint_exists(best_target_ckpt)
    if os.path.exists(best_target_ckpt):
        ckpt = torch.load(best_target_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            if rank == 0:
                print(f"Loaded target model checkpoint: {best_target_ckpt} (from wrapped state dict)", flush=True)
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            if rank == 0:
                print(f"Loaded target model checkpoint: {best_target_ckpt} (from state_dict)", flush=True)
        else:
            model.load_state_dict(ckpt)
            if rank == 0:
                print(f"Loaded target model checkpoint: {best_target_ckpt} (from raw state dict)", flush=True)
    else:
        if rank == 0:
            print(f"Warning: Pretrained target checkpoint {best_target_ckpt} not found! Initializing model randomly.", flush=True)

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
            find_unused_parameters=False,
            broadcast_buffers=False
        )
    elif torch.cuda.device_count() > 1:
        if rank == 0:
            print(f"Using {torch.cuda.device_count()} GPUs for training (DataParallel)", flush=True)
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, 'module') else model

    # Curriculum Setup (120 epochs total)
    # format: (start_epoch, end_epoch, eps, beta, steps, lr)
    curriculum = [
        (1,  40,  0.031, 2.0, 7,  0.003),
        (41, 80,  0.062, 2.0, 10, 0.002),
        (81, 120, 0.094, 2.5, 10, 0.001),
    ]

    scaler = GradScaler('cuda')
    best_acc = 0.0
    start_epoch = 1
    current_phase_start = None
    optimizer = None
    scheduler = None

    best_path = os.path.join(ckpt_dir, 'rhan_stl10_large_pseudolabel_best.pth')
    rolling_path = os.path.join(ckpt_dir, 'rhan_stl10_large_pseudolabel_rolling.pth')

    # 6. Automatic Resume Check
    # Fetch HF Token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
        except Exception:
            pass

    # Compare local checkpoint with remote checkpoint to always resume from the newest state
    local_epoch = -1
    if os.path.exists(rolling_path):
        try:
            local_data = torch.load(rolling_path, map_location='cpu')
            local_epoch = local_data.get('epoch', -1)
        except Exception:
            pass

    if rank == 0:
        remote_checkpoint_data = None
        try:
            from huggingface_hub import hf_hub_download
            print("Checking for a newer checkpoint on Hugging Face...", flush=True)
            temp_rolling_path = hf_hub_download(
                repo_id='FerrariKazu/rhan-checkpoints',
                filename='rhan_stl10_large_pseudolabel_rolling.pth',
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
        dist.barrier() # wait for rank 0 to download newer checkpoint

    if os.path.exists(rolling_path):
        if rank == 0:
            print(f"\nFound rolling checkpoint at {rolling_path}. Attempting to resume...", flush=True)
        checkpoint_data = torch.load(rolling_path, map_location=device)
        raw_model.load_state_dict(checkpoint_data['model'])
        best_acc = checkpoint_data.get('best_acc', 0.0)
        start_epoch = checkpoint_data['epoch'] + 1
        if rank == 0:
            print(f"Resuming from Epoch {start_epoch} (Best validation accuracy so far: {best_acc:.2f}%)")

    # 7. Training loop
    for epoch in range(start_epoch, 121):
        t0 = time.time()
        
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
                    # Restore state dicts if we are resuming right at the start of this session
                    if epoch == start_epoch and os.path.exists(rolling_path) and 'optimizer' in checkpoint_data:
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
                
            # Determine if this step accumulates gradients without synchronization
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
                
                # TRADES Adversarial PGD generation
                model.eval()
                with torch.no_grad():
                    with autocast('cuda'):
                        logits_c = model(imgs)
                probs_c = F.softmax(logits_c.float(), dim=1)

                x_adv = imgs.clone().detach() + 0.001 * torch.randn_like(imgs)
                x_adv = torch.clamp(x_adv, stl_min, stl_max)
                for _ in range(steps):
                    x_adv.requires_grad_(True)
                    with torch.enable_grad():
                        with autocast('cuda'):
                            logits_a = model(x_adv)
                            loss_adv = F.kl_div(
                                F.log_softmax(logits_a.float(), dim=1),
                                probs_c, reduction='batchmean'
                            )
                    grad = torch.autograd.grad(loss_adv, x_adv)[0]
                    x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                    delta = torch.clamp(x_adv - imgs, -eps, eps)
                    x_adv = torch.clamp(imgs + delta, stl_min, stl_max).detach()
                
                model.train()

                with autocast('cuda'):
                    l_trades = trades_loss_weighted(model, imgs, lbls, weights, x_adv, beta)
                    loss = l_trades / args.accum_steps

                scaler.scale(loss).backward()
            
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
                    logits_c = model(imgs)
            correct += logits_c.argmax(1).eq(lbls).sum().item()
            n_total += B
            if rank == 0 and batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches} | Loss: {l_trades.item():.4f}", flush=True)

            if args.dry_run:
                if rank == 0:
                    print("Dry-run mode active. Successfully completed 1 training step.", flush=True)
                break

        if num_batches % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Validation (clean test) - run only on Rank 0
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

        # Broadcast validation score to all processes in DDP mode
        if is_ddp:
            val_acc_tensor = torch.tensor([val_acc], device=device)
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
            images_per_sec = total_images / t_epoch
            epochs_per_hour = 3600.0 / t_epoch
            print(
                f"Epoch {epoch:03d}/120 (ε={eps:.3f}) | Loss:{total_loss/n_total:.3f} | "
                f"TrAcc:{100.*correct/n_total:.1f}% TeAcc:{val_acc:.1f}% | "
                f"Throughput:{images_per_sec:.2f} img/sec ({epochs_per_hour:.2f} epochs/hour) | "
                f"{t_epoch:.0f}s{marker}", flush=True
            )

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
                dist.barrier()
            break

        if is_ddp:
            dist.barrier() # Sync before starting next epoch

    if rank == 0:
        print(f"Training Complete. Best Model saved to {best_path}")

        if not args.dry_run:
            # 8. Launch full evaluation (AutoAttack + PGD sweep) automatically on the trained model
            print("\nLaunching final evaluation (AutoAttack + PGD sweep)...")
            eval_script = os.path.join(script_dir, '..', 'run_eval_stl10.py')
            try:
                subprocess.run([
                    sys.executable, eval_script,
                    "--model-size", "large",
                    "--checkpoint", "checkpoints/rhan_stl10_large_pseudolabel_best.pth",
                    "--samples", "1000",
                    "--batch-size", "128"
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Evaluation script failed: {e}")
        else:
            print("Dry-run complete. Skipping evaluation script execution.")

if __name__ == '__main__':
    main()

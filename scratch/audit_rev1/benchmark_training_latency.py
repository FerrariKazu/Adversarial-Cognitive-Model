import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from gen2_foundation.backbone import CompactViTGen2
from gen2_foundation.model import Gen2FoundationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0)})")

# Dummy synthetic dataset matching ImageNet-100 batch dimensions
class FakeDataset(Dataset):
    def __init__(self, size=(112, 112), num_samples=1200):
        self.size = size
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return torch.randn(3, self.size[0], self.size[1]), torch.randint(0, 100, (1,)).item()

# 1. Feedforward DINOv2-S model at arbitrary resolution (divisible by 14)
class FeedforwardDINOv2(nn.Module):
    def __init__(self, img_size=112, num_classes=100, ckpt_path="checkpoints/dinov2_vits14_pretrain.pth"):
        super().__init__()
        self.img_size = img_size
        self.backbone = CompactViTGen2(img_size=img_size, num_refine_iters=2)
        full_ckpt = os.path.join(REPO_ROOT, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
        if os.path.exists(full_ckpt):
            self.backbone.load_dino_warm_start(full_ckpt, strict_band=0.90)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        # Clean feedforward pass through ViT trunk (no recurrence/refinement)
        x_prep = self.backbone._prep(x)
        feat = self.backbone._trunk_forward(x_prep)
        cls_token = feat[:, 0]
        return self.head(cls_token)

# Corrected TRADES PGD-4 function
def run_pgd4_trades_corrected(model, x, eps=0.094, steps=4):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probs_clean = F.softmax(model(x).float(), dim=1)
        # Random start in [-eps, eps]
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        # Corrected step size: alpha = 2.5 * eps / steps
        alpha = (2.5 * eps) / steps

        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits_adv = model(x_adv)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv.float(), dim=1),
                    probs_clean,
                    reduction="batchmean",
                )
            grad = torch.autograd.grad(loss_kl, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = x_adv.clamp(-4.0, 4.0)
        return x_adv.detach()
    finally:
        if was_training:
            model.train()

# Compute trades loss
def compute_trades_loss(model, x, y, x_adv, beta=2.5, w_trades=0.55):
    logits_clean = model(x)
    loss_ce = F.cross_entropy(logits_clean, y)
    logits_adv = model(x_adv)
    loss_kl = F.kl_div(
        F.log_softmax(logits_adv.float(), dim=1),
        F.softmax(logits_clean.float(), dim=1),
        reduction="batchmean",
    )
    loss = (1.0 - w_trades) * loss_ce + w_trades * beta * loss_kl
    return loss

def benchmark_loop(model, loader, is_adversarial=True, name="config", num_batches=25):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    
    # Warmup
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if i >= 3: break
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    t0 = time.time()
    batch_count = 0
    for i, (x, y) in enumerate(loader):
        if i >= num_batches: break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if is_adversarial:
            x_adv = run_pgd4_trades_corrected(model, x)
            with torch.amp.autocast("cuda"):
                loss = compute_trades_loss(model, x, y, x_adv)
        else:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_count += 1
    
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    sec_per_batch = elapsed / batch_count
    total_epoch_batches = 126689 // loader.batch_size
    est_epoch_sec = total_epoch_batches * sec_per_batch
    est_epoch_min = est_epoch_sec / 60.0
    print(f"[{name}] {sec_per_batch:.3f} s/batch | Est 1 full epoch (126k): {est_epoch_min:.1f} min ({est_epoch_min/60.0:.2f} GPU-h)")
    return sec_per_batch, est_epoch_min

if __name__ == "__main__":
    batch_size = 48
    print(f"Running benchmarks with batch_size={batch_size}...")
    
    # 1. Feedforward 112x112 DINOv2-S
    print("\n--- 1. Feedforward 112x112 DINOv2-S ---")
    loader_112 = DataLoader(FakeDataset(size=(112, 112), num_samples=1500), batch_size=batch_size, shuffle=False)
    ff_112 = FeedforwardDINOv2(img_size=112).to(device)
    benchmark_loop(ff_112, loader_112, is_adversarial=False, name="FF 112px Clean")
    benchmark_loop(ff_112, loader_112, is_adversarial=True, name="FF 112px TRADES-PGD4")
    
    # 2. Feedforward 126x126 DINOv2-S (81 tokens, divisible by patch 14)
    print("\n--- 2. Feedforward 126x126 DINOv2-S (81 tokens) ---")
    loader_126 = DataLoader(FakeDataset(size=(126, 126), num_samples=1500), batch_size=batch_size, shuffle=False)
    ff_126 = FeedforwardDINOv2(img_size=126).to(device)
    benchmark_loop(ff_126, loader_126, is_adversarial=True, name="FF 126px TRADES-PGD4")
    
    # 3. Gen-2 Recurrent Loop 56x56
    print("\n--- 3. Gen-2 Recurrent 56x56 (recurrence_only) ---")
    loader_56 = DataLoader(FakeDataset(size=(56, 56), num_samples=1500), batch_size=batch_size, shuffle=False)
    rec_56 = Gen2FoundationModel(phase="recurrence_only", arm="v1").to(device)
    benchmark_loop(rec_56, loader_56, is_adversarial=True, name="Gen-2 Recurrent 56px TRADES-PGD4")

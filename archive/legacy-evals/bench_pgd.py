#!/usr/bin/env python3
"""Benchmark PGD timing for RHAN-STL10."""
import sys, time, torch, torch.nn as nn
sys.path.insert(0, 'phase1_training')
from model_rhan_stl10 import RHANSTL10

device = torch.device('cuda')
ckpt = torch.load('checkpoints/rhan_stl10_best.pth', map_location=device, weights_only=False)
model = RHANSTL10(head_type='linear').to(device)
model.load_state_dict(ckpt)
model.eval()

clip_min = torch.tensor([-3.3487, -3.3289, -3.1331], device=device).view(1,3,1,1)
clip_max = torch.tensor([2.9870, 3.0136, 3.2104], device=device).view(1,3,1,1)

def bench_pgd(batch_size, recurrent_steps, n_steps=10):
    model.feedback.num_recurrent_steps = recurrent_steps
    x = torch.randn(batch_size, 3, 96, 96, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    eps = 0.05
    alpha = eps / 4
    
    # Warmup
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
    for _ in range(2):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x + torch.clamp(x_adv - x, -eps, eps), clip_min, clip_max).detach()
    
    torch.cuda.synchronize()
    t0 = time.time()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x + torch.clamp(x_adv - x, -eps, eps), clip_min, clip_max).detach()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed

for bs in [64, 128]:
    for rec in [0, 2]:
        t = bench_pgd(bs, rec, n_steps=10)
        per_step = t / 10
        total_batches = (8000 + bs - 1) // bs
        est_total = per_step * 100 * total_batches * 5  # 5 epsilons
        print(f"BS={bs} rec={rec}: {per_step*1000:.0f}ms/step, est total for 5 eps: {est_total:.0f}s = {est_total/60:.1f}min")

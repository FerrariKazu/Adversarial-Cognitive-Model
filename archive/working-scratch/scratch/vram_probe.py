"""Probe peak VRAM for the Stage 2 C_hpc_only smoke at a given batch size.

The local RTX 4060 has only ~1.6 GB free (display holds the rest). This
measures forward+backward+optimizer-step peak so we can pick a batch size /
accum-steps pair that keeps the protocol's effective batch 256 without OOM.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_training"))

import torch

from rhan_core.ablation.matrix import get_entry
from rhan_core.model import RHANNext

cfg = get_entry("C_hpc_only")["config"]
print(f"config: enable_hpc={cfg.enable_hpc} hpc_num_levels={cfg.hpc_num_levels} "
      f"w_hpc={cfg.hpc_error_weight} enable_ais={cfg.enable_ais}")

model = RHANNext(cfg)
base = "checkpoints/rhan_next_ais_v1_halting_only_best.pth"
state = torch.load(base, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(state["model"], strict=False)
print(f"base ckpt load: missing={len(missing)} unexpected={len(unexpected)}")
model = model.cuda().train()

opt = torch.optim.AdamW(
    [{"params": model.parameters(), "lr": 3e-4}], weight_decay=0.05)

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 16
img = torch.randn(batch, 3, 96, 96, device="cuda")
gaze = torch.randint(0, 96, (batch, 2), device="cuda")
t = torch.randn(batch, 3, 48, 48, device="cuda")

torch.cuda.reset_peak_memory_stats()
for step in range(3):
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    logits, traj = model(img, return_trajectory=True)
    # mimic the trainer's loss at w_hpc=0.10 / w_trades=0.55
    l_hpc = model.get_hpc_loss(img, (logits, traj))
    loss = (0.55 * torch.nn.functional.cross_entropy(
        logits, torch.randint(0, 10, (batch,), device="cuda"))
        + 0.10 * l_hpc)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"step {step}: peak_alloc={peak:.2f} GB, free_now={free:.2f} GB, "
          f"{time.time()-t0:.1f}s/step")
print(f"FINAL peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB @ batch={batch}")

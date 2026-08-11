#!/usr/bin/env python3
"""
SBR feasibility probe — does a lightweight, FROZEN-backbone Slot Attention head
recover object-part-like clusters from a VALIDATED checkpoint's existing
features?  (Pillar 3 feasibility only — NOT a Pillar 3 launch.)
================================================================================

Scope (read this before running):
  * STANDALONE, READ-ONLY probe — not a training pipeline, not part of the
    training loop, never scheduled with the ablation matrix. It loads a
    validated checkpoint, runs it ONCE in eval/no_grad, and asks ONLY:
    "do object-part-like clusters (attention masks) emerge from the frozen
    features, evaluated QUALITATIVELY (visualized masks + stats)?" It does
    NOT ask "does it improve accuracy".
  * Which checkpoint: whichever pillar is validated by the time this runs —
    HPC's (rhan_next_hpc_only_best.pth) if Stage 2 succeeds, AIS-v1's
    (rhan_next_ais_v1_halting_only_best.pth) otherwise. Default is the
    AIS-v1 checkpoint (the currently-validated pillar); it is downloaded from
    HuggingFace when not present locally.
  * HARD CONSTRAINTS (enforced):
      - This probe NEVER imports or modifies RHANNextConfig. It constructs the
        frozen model as RHANNext(**embedded_config_dict) — the config dataclass
        is built internally by the model, never named here — and it explicitly
        asserts the embedded config keeps enable_sbr=False (validate() would
        reject True anyway; the gate stays locked exactly as Stage 0 built it).
      - It never sets enable_sbr=True anywhere.
      - It never touches train_rhan_next.py or the main training loop.
      - Output goes to report/sbr_feasibility/ — never into the ablation matrix.
  * The Slot Attention head is freshly (randomly) initialized and UNTRAINED:
    this is a feasibility probe of the MACHINERY on frozen features, not a
    learned-binding result. Interpret nothing until HPC (Stage 2) is validated,
    per the agreed sequencing.

Usage:
    python3 rhan_core/beliefs/experimental/sbr_feasibility.py \
        [--checkpoint checkpoints/rhan_next_ais_v1_halting_only_best.pth] \
        [--n-samples 8] [--slots 8] [--iters 3] [--seed 0] \
        [--out report/sbr_feasibility] [--data-root ./data/stl10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

# The repo root + phase1_training on sys.path (mirrors rhan_core/model.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "phase1_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CKPT = "checkpoints/rhan_next_ais_v1_halting_only_best.pth"
HF_REPO = "FerrariKazu/rhan-checkpoints"
EXPERIMENTAL_BANNER = (
    "FEASIBILITY PROBE — NOT a Pillar 3 launch. Frozen-backbone Slot Attention "
    "on validated features, qualitatively evaluated. The slot head is "
    "UNTRAINED (random init). No accuracy claim. Do not schedule or cite "
    "beyond feasibility. enable_sbr stays locked at False.")


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _ensure_checkpoint(path: str) -> str:
    """Resolve the checkpoint locally; download from HF when missing."""
    if os.path.exists(path):
        return path
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            f"checkpoint {path} not present locally and HF_TOKEN is unset — "
            "set HF_TOKEN to download it, or pass --checkpoint with a local file.")
    print(f"[sbr-probe] downloading {os.path.basename(path)} from {HF_REPO} ...",
          flush=True)
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                    filename=os.path.basename(path), local_dir=os.path.dirname(path) or ".",
                    token=os.environ["HF_TOKEN"])
    assert os.path.exists(path), "download finished but file missing"
    return path


def _load_frozen_model(ckpt_path: str, device):
    """Load the frozen backbone WITHOUT importing RHANNextConfig.

    The checkpoint carries its own serialized config dict (written by the
    trainer). We pass that dict straight into RHANNext(**dict) — the config
    dataclass is constructed internally by the model, never imported here —
    which reproduces the EXACT trained architecture. A plain v12-style
    checkpoint (no 'config' key) falls back to RHANv12.
    """
    from rhan_core.model import RHANNext

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in state:
                weights = state[k]
                break
        else:
            weights = state
        cfg_dict = state.get("config")
    else:
        weights, cfg_dict = state, None

    if isinstance(cfg_dict, dict):
        # HARD CONSTRAINT: the SBR gate stays locked. from_dict/validate would
        # raise on enable_sbr=True anyway; assert it explicitly for the record.
        assert cfg_dict.get("enable_sbr") is False, (
            "embedded config has enable_sbr=True — the probe must never run "
            "with SBR enabled (gate stays locked)")
        model = RHANNext(**cfg_dict)      # config built internally, not imported
        print(f"[sbr-probe] frozen backbone = RHANNext({cfg_dict.get('enable_ais') and 'AIS' or ''}"
              f"{cfg_dict.get('enable_hpc') and '+HPC' or ''}) from embedded config",
              flush=True)
    else:
        from model_rhan_v12 import RHANv12
        model = RHANv12()
        print("[sbr-probe] frozen backbone = RHANv12 (no embedded config)", flush=True)

    model = model.to(device).eval()
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"[sbr-probe] weights: {len(weights) - len(missing)}/{len(weights)} "
          f"keys loaded ({len(missing)} missing, {len(unexpected)} unexpected)",
          flush=True)
    return model


def _load_images(data_root: str, n: int, device, seed: int = 0):
    """STL-10 test images when the dataset is present; else synthetic blobs.

    The probe must run anywhere (the smoke test is 'imports and runs'), so a
    missing dataset falls back to simple synthetic images that still exercise
    edge structure (the HPC target) — clearly labeled as synthetic in the
    summary.
    """
    norm = torch.tensor([[0.4467, 0.4398, 0.4066]], device=device).view(1, 3, 1, 1)
    std = torch.tensor([[0.2603, 0.2566, 0.2713]], device=device).view(1, 3, 1, 1)
    try:
        import torchvision
        import torchvision.transforms as T
        # Read-only probe: never download a dataset. If STL-10 is absent,
        # fall back to synthetic blobs (clearly labeled as such).
        ds = torchvision.datasets.STL10(
            data_root, split="test", download=False,
            transform=T.Compose([T.ToTensor()]))
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
        imgs = torch.stack([ds[i][0].to(device) for i in idx])
        print(f"[sbr-probe] data source = STL-10 test ({n} samples)", flush=True)
        return (imgs - norm) / std, "stl10_test"
    except Exception as e:
        print(f"[sbr-probe] STL-10 unavailable ({e}); using SYNTHETIC blobs "
              f"(smoke-only, not interpretable)", flush=True)
        g = torch.Generator(device=device).manual_seed(seed)
        base = torch.randn(n, 3, 96, 96, device=device, generator=g) * 0.1
        for i in range(n):
            cx, cy = (g.normal_(0, 1).item() * 0.2 + 0.5), (g.normal_(0, 1).item() * 0.2 + 0.5)
            y, x = torch.meshgrid(torch.linspace(0, 1, 96, device=device),
                                  torch.linspace(0, 1, 96, device=device), indexing="ij")
            blob = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.06 ** 2))
            base[i, :] += 0.8 * blob.unsqueeze(0)
        return base, "synthetic_blobs"


class _SlotAttention(nn.Module):
    """Minimal Slot Attention (Locatello et al. 2020) — frozen, UNTRAINED."""

    def __init__(self, dim: int, num_slots: int = 8, iters: int = 3,
                 num_heads: int = 4):
        super().__init__()
        self.dim, self.num_slots, self.iters = dim, num_slots, iters
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(),
                                 nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor):
        """inputs: (B, N, D) -> (slots (B, K, D), masks (B, K, N))."""
        B, N, D = inputs.shape
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_sigma.expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        masks = None
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm(slots)
            attn = torch.einsum("bkd,bnd->bkn", self.to_q(slots), k) * self.scale
            attn = attn - attn.max(dim=1, keepdim=True).values
            attn = attn.softmax(dim=1)                    # (B, K, N) masks
            masks = attn
            updates = torch.einsum("bkn,bnd->bkd", attn, v)
            slots = self.gru(updates.reshape(-1, D),
                             slots_prev.reshape(-1, D)).reshape(B, -1, D)
            slots = slots + self.mlp(self.norm(slots))
        return slots, masks


def _pos_enc(h: int, w: int, dim: int, device) -> torch.Tensor:
    """Fixed 2D sinusoidal positional encoding, (1, H*W, dim)."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h, device=device),
                            torch.linspace(0, 1, w, device=device), indexing="ij")
    pe = torch.zeros(1, h * w, dim, device=device)
    freqs = 10.0 ** torch.linspace(0, 1, dim // 4, device=device)
    i = 0
    for f in freqs:
        pe[0, :, i] = (xs.flatten() * f).sin(); i += 1
        pe[0, :, i] = (ys.flatten() * f).sin(); i += 1
        pe[0, :, i] = (xs.flatten() * f).cos(); i += 1
        pe[0, :, i] = (ys.flatten() * f).cos(); i += 1
    return pe


def _mask_stats(masks: torch.Tensor):
    """masks: (B, K, N) -> per-sample stats dict (entropy, coverage, IoU, centroids)."""
    B, K, N = masks.shape
    eps = 1e-8
    ent = -(masks * (masks + eps).log()).sum(dim=1).mean(dim=1)  # (B,) mean over positions
    maxp, argmax = masks.max(dim=1)                              # (B, N), (B, N)
    coverage = (maxp > 0.5).float().mean(dim=1)                  # (B,)
    # Pairwise mask IoU per sample (thresholded at the per-position argmax).
    hard = F.one_hot(argmax, num_classes=K).permute(0, 2, 1).float()  # (B, K, N)
    used = (hard.sum(dim=2) > 0).sum(dim=1)                            # (B,) slots used
    ious = []
    for b in range(B):
        row = []
        for i in range(K):
            for j in range(i + 1, K):
                a, c = hard[b, i].bool(), hard[b, j].bool()
                row.append(float((a & c).sum() / (a | c).sum().clamp(min=1)))
        ious.append(row)
    return {"entropy_mean": [round(float(e), 4) for e in ent],
            "coverage": [round(float(c), 4) for c in coverage],
            "num_slots_used": [int(u) for u in used],
            "pairwise_iou": [[round(v, 4) for v in r] for r in ious]}


def _visualize(imgs, masks, h, w, out_dir, device):
    """Save per-sample montages (input + K upsampled slot masks). Best-effort."""
    try:
        import torchvision
        from torchvision.utils import make_grid, save_image
    except Exception as e:
        print(f"[sbr-probe] torchvision unavailable — skipping PNG montages ({e})",
              flush=True)
        return False
    os.makedirs(out_dir, exist_ok=True)
    B, K, _ = masks.shape
    for b in range(min(B, 16)):
        # Denormalize for display.
        # Per-channel view (3,1,1) so broadcasting NEVER pads a leading batch
        # dim onto the (3,96,96) image.
        norm = torch.tensor([0.4467, 0.4398, 0.4066]).view(3, 1, 1).to(device)
        std = torch.tensor([0.2603, 0.2566, 0.2713]).view(3, 1, 1).to(device)
        img = (imgs[b].detach() * std + norm).clamp(0, 1)
        cells = [img]
        for k in range(K):
            m = masks[b, k].view(h, w).unsqueeze(0).unsqueeze(0)      # (1,1,H,W)
            m = F.interpolate(m, size=(96, 96), mode="bilinear",
                              align_corners=False).squeeze(0)
            cells.append(m.expand(3, -1, -1))
        grid = make_grid(cells, nrow=K + 1, padding=2, normalize=False)
        save_image(grid, os.path.join(out_dir, f"slot_masks_sample_{b:03d}.png"))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/sbr_feasibility")
    ap.add_argument("--data-root", default="./data/stl10")
    args = ap.parse_args()

    print("=" * 72)
    print(EXPERIMENTAL_BANNER)
    print("=" * 72)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    ckpt = _ensure_checkpoint(args.checkpoint)
    model = _load_frozen_model(ckpt, device)

    # Tap the full-image stem: the spatial conv features that feed the
    # tokeniser/transformer — the single lowest-level spatial map of the frozen
    # backbone (documented tap point for this probe).
    captured = {}
    handle = model.stem.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("feats", out.detach()))

    imgs, src = _load_images(args.data_root, args.n_samples, device, args.seed)
    with torch.no_grad():
        logits = model(imgs)
    feats = captured["feats"]
    handle.remove()
    assert feats.dim() == 4, f"expected spatial stem features, got {tuple(feats.shape)}"
    B, C, H, W = feats.shape
    print(f"[sbr-probe] stem tap: {tuple(feats.shape)} "
          f"(B={B}, C={C}, grid {H}x{W})", flush=True)
    sm = F.softmax(logits, dim=1)
    top1 = sm.max(dim=1).values
    print(f"[sbr-probe] frozen logits sanity: mean TOP-1 softmax prob "
          f"{top1.mean():.3f} (uniform baseline = 0.100)", flush=True)

    grid_feats = feats.flatten(2).transpose(1, 2)                 # (B, H*W, C)
    grid_feats = grid_feats + _pos_enc(H, W, C, device)           # positional info
    sa = _SlotAttention(dim=C, num_slots=args.slots, iters=args.iters).to(device)
    with torch.no_grad():
        slots, masks = sa(grid_feats)
    print(f"[sbr-probe] slots {tuple(slots.shape)} | attention masks "
          f"{tuple(masks.shape)}", flush=True)

    stats = _mask_stats(masks)
    vis_ok = _visualize(imgs, masks, H, W, args.out, device)

    summary = {
        "schema": "sbr_feasibility_probe_v1",
        "banner": EXPERIMENTAL_BANNER,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": ckpt,
        "checkpoint_sha256": _sha256(ckpt),
        "data_source": src,
        "n_samples": B,
        "stem_tap_shape": list(feats.shape),
        "slots": args.slots,
        "iters": args.iters,
        "slot_head": "UNTRAINED random-init Slot Attention on frozen features",
        "mask_stats": stats,
        "visualizations": vis_ok,
        "interpretation": (
            "FEASIBILITY ONLY — qualitative. Slot masks reveal whether the "
            "frozen features partition into compact, object-part-like regions "
            "(low mean mask entropy / low pairwise IoU / few unused slots). "
            "Do NOT interpret results until HPC (Stage 2) is validated, per "
            "the agreed sequencing. This probe never trains and never touches "
            "enable_sbr (gate stays locked)."),
    }
    with open(os.path.join(args.out, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write(EXPERIMENTAL_BANNER + "\n\n")
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"tap: model.stem ({H}x{W} spatial grid, {C} channels)\n")
        f.write(f"slot head: UNTRAINED, {args.slots} slots, {args.iters} iters\n\n")
        f.write("Interpretation deferred until HPC (Stage 2) validates.\n")
    print(f"[sbr-probe] done in {time.time() - t0:.0f}s -> {args.out}/ "
          f"(probe_summary.json, README.txt{', PNG montages' if vis_ok else ''})",
          flush=True)
    print("=" * 72)
    print("FEASIBILITY PROBE COMPLETE — outputs are preliminary, untrained, "
          "and NOT scheduled. HPC (Stage 2) verdict gates any interpretation.")
    print("=" * 72)


if __name__ == "__main__":
    main()

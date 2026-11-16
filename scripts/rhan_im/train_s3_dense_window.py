"""Stage S3: Dense window fine-tuning of the top trunk blocks.

Pre-registered remedy for Gate 1 failure on 'looking_closer_pays'.
Unfreezes trunk.blocks[9:12] and trunk.norm (5,326,464 parameters);
trains with dense window cross-entropy on both native and gist looks;
early-stops on held-out native-over-gist ratio at size 4, alpha 1.0 from train/ (never val/);
discards the training head and saves the tuned trunk for Gate 1 v2 evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2  # noqa: E402
from rhan_im import gate1 as g1  # noqa: E402
from rhan_im.layout import UnitLayout  # noqa: E402
from rhan_im.trunk_adapter import native_crop  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

EXPECTED_TRUNK_TRAINABLE_PARAMS = 5326464  # blocks[9:12] (5,325,696) + norm (768)


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class S3FolderBank:
    """Loads images from a pinned list of relative paths within data_root/train."""

    def __init__(self, data_root: pathlib.Path, file_list: List[str], tcls: List[str], bcls: List[str]):
        self.data_root = data_root
        self.tcls = tcls
        self.bcls = bcls
        self.cls2target = {c: i for i, c in enumerate(tcls)}
        self.cls2bg = {c: i for i, c in enumerate(bcls)}
        self.target_files: Dict[int, List[pathlib.Path]] = {i: [] for i in range(len(tcls))}
        self.bg_files: Dict[int, List[pathlib.Path]] = {i: [] for i in range(len(bcls))}

        for rel in file_list:
            p = data_root / "train" / rel
            dirname = p.parent.name
            if dirname in self.cls2target:
                self.target_files[self.cls2target[dirname]].append(p)
            elif dirname in self.cls2bg:
                self.bg_files[self.cls2bg[dirname]].append(p)

        for c_idx, fl in self.target_files.items():
            if len(fl) == 0:
                raise ValueError(f"Target class index {c_idx} ({tcls[c_idx]}) has 0 images in partition")
        for c_idx, fl in self.bg_files.items():
            if len(fl) == 0:
                raise ValueError(f"Background class index {c_idx} ({bcls[c_idx]}) has 0 images in partition")

        self.n_classes = len(tcls)
        self._cache: Dict[pathlib.Path, torch.Tensor] = {}

    def get_image(self, p: pathlib.Path) -> torch.Tensor:
        if p not in self._cache:
            from PIL import Image
            import torchvision.transforms.functional as TF
            with Image.open(p) as img:
                img = img.convert("RGB")
                t = TF.to_tensor(img)
            self._cache[p] = t
        return self._cache[p]

    def sample_target(self, c: int, g: torch.Generator) -> torch.Tensor:
        fl = self.target_files[c]
        idx = int(torch.randint(0, len(fl), (1,), generator=g).item())
        return self.get_image(fl[idx])

    def sample_bg(self, g: torch.Generator) -> torch.Tensor:
        bg_c = int(torch.randint(0, len(self.bg_files), (1,), generator=g).item())
        fl = self.bg_files[bg_c]
        idx = int(torch.randint(0, len(fl), (1,), generator=g).item())
        return self.get_image(fl[idx])


def partition_train_split(
    data_root: pathlib.Path,
    tcls: List[str],
    bcls: List[str],
    heldout_per_class: int = 50,
    out_manifest: Optional[pathlib.Path] = None,
) -> Tuple[List[str], List[str]]:
    """Partitions data_root/train into disjoint (train_files, heldout_files)."""
    train_dir = data_root / "train"
    all_classes = sorted(tcls + bcls)
    heldout_files: List[str] = []
    train_files: List[str] = []

    for c in all_classes:
        cdir = train_dir / c
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing directory {cdir}")
        files = sorted(f.name for f in cdir.iterdir() if f.is_file() and not f.name.startswith("."))
        if len(files) <= heldout_per_class:
            raise ValueError(f"Class {c} has {len(files)} files, cannot hold out {heldout_per_class}")
        for fn in files[:heldout_per_class]:
            heldout_files.append(f"{c}/{fn}")
        for fn in files[heldout_per_class:]:
            train_files.append(f"{c}/{fn}")

    heldout_files.sort()
    train_files.sort()

    if out_manifest:
        h_sha = hashlib.sha256("\n".join(heldout_files).encode("utf-8")).hexdigest()
        t_sha = hashlib.sha256("\n".join(train_files).encode("utf-8")).hexdigest()
        manifest = {
            "status": "FINAL",
            "heldout_per_class": heldout_per_class,
            "heldout_count": len(heldout_files),
            "train_count": len(train_files),
            "heldout_listing_sha256": h_sha,
            "train_listing_sha256": t_sha,
            "heldout_files": heldout_files,
            "train_files": train_files,
        }
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps(manifest, indent=2))

    return train_files, heldout_files


def make_scene_batch(
    bank: S3FolderBank,
    lay: UnitLayout,
    batch_size: int,
    sizes: List[int],
    alphas: List[float],
    p_null: float,
    g: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Generates a batch of composited scenes on canvas 224x224."""
    import torchvision.transforms.functional as TF

    canvases = []
    labels = []
    overlaps = []

    for _ in range(batch_size):
        # sample bg
        bg_raw = bank.sample_bg(g)
        # centre crop bg to canvas x canvas
        h, w = bg_raw.shape[1], bg_raw.shape[2]
        side = min(h, w)
        bg = TF.center_crop(bg_raw, [side, side])
        bg = F.interpolate(bg.unsqueeze(0), size=(lay.canvas, lay.canvas), mode="bilinear", align_corners=False).squeeze(0)

        is_null = torch.rand(1, generator=g).item() < p_null
        overlap_map = torch.zeros((lay.canvas_units, lay.canvas_units), dtype=torch.float32)

        if is_null:
            canvases.append(bg)
            labels.append(-1)
            overlaps.append(overlap_map)
        else:
            c = int(torch.randint(0, bank.n_classes, (1,), generator=g).item())
            s = sizes[int(torch.randint(0, len(sizes), (1,), generator=g).item())]
            a = alphas[int(torch.randint(0, len(alphas), (1,), generator=g).item())]
            tgt_raw = bank.sample_target(c, g)
            tgt_h, tgt_w = tgt_raw.shape[1], tgt_raw.shape[2]
            tgt_side = min(tgt_h, tgt_w)
            tgt = TF.center_crop(tgt_raw, [tgt_side, tgt_side])
            tgt_px = s * lay.patch
            tgt = F.interpolate(tgt.unsqueeze(0), size=(tgt_px, tgt_px), mode="bilinear", align_corners=False).squeeze(0)

            # random unit placement
            max_r = lay.canvas_units - s
            max_c = lay.canvas_units - s
            r = int(torch.randint(0, max_r + 1, (1,), generator=g).item())
            col = int(torch.randint(0, max_c + 1, (1,), generator=g).item())

            top_px = r * lay.patch
            left_px = col * lay.patch

            canvas = bg.clone()
            canvas[:, top_px:top_px + tgt_px, left_px:left_px + tgt_px] = (
                a * tgt + (1.0 - a) * canvas[:, top_px:top_px + tgt_px, left_px:left_px + tgt_px]
            )
            overlap_map[r:r + s, col:col + s] = 1.0

            canvases.append(canvas)
            labels.append(c)
            overlaps.append(overlap_map)

    return {
        "canvas": torch.stack(canvases),
        "label": torch.tensor(labels, dtype=torch.long),
        "overlap": torch.stack(overlaps),
    }


def sample_windows(
    scenes: Dict[str, torch.Tensor],
    lay: UnitLayout,
    candidate_windows: torch.Tensor,
    g: torch.Generator,
) -> torch.Tensor:
    """Samples one native window per scene, 70% containing target centre if target present."""
    b = scenes["canvas"].shape[0]
    out = torch.zeros((b, 2), dtype=torch.long)
    for i in range(b):
        lbl = scenes["label"][i].item()
        ov = scenes["overlap"][i]
        if lbl >= 0 and torch.rand(1, generator=g).item() < 0.7:
            # find candidate windows covering target centre
            pos = torch.nonzero(ov)
            if len(pos) > 0:
                ctr_r = pos[:, 0].float().mean().item()
                ctr_c = pos[:, 1].float().mean().item()
                good = []
                for w_idx in range(candidate_windows.shape[0]):
                    wr, wc = candidate_windows[w_idx, 0].item(), candidate_windows[w_idx, 1].item()
                    if wr <= ctr_r < wr + lay.window_units and wc <= ctr_c < wc + lay.window_units:
                        good.append(w_idx)
                if good:
                    pick = good[int(torch.randint(0, len(good), (1,), generator=g).item())]
                    out[i] = candidate_windows[pick]
                    continue
        pick = int(torch.randint(0, candidate_windows.shape[0], (1,), generator=g).item())
        out[i] = candidate_windows[pick]
    return out


def load_trunk_for_s3(
    ckpt_path: pathlib.Path,
    window: int = 112,
    device: str = "cuda",
) -> Tuple[CompactViTGen2, Dict[str, object]]:
    trunk = CompactViTGen2(img_size=window, num_refine_iters=2)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw.get("model_state_dict", raw.get("state_dict", raw)))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    own = trunk.state_dict()
    used = {k: v for k, v in bb.items() if k in own and own[k].shape == v.shape and not k.startswith("refinement.")}
    trunk.load_state_dict({**own, **used}, strict=True)

    # Freeze entire trunk first
    trunk.eval().requires_grad_(False)

    # Unfreeze only blocks[9:12] and norm
    for b in trunk.blocks[9:12]:
        b.train().requires_grad_(True)
    trunk.norm.train().requires_grad_(True)

    trainable_params = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    if trainable_params != EXPECTED_TRUNK_TRAINABLE_PARAMS:
        raise ValueError(
            f"Trainable trunk params mismatch: got {trainable_params}, expected {EXPECTED_TRUNK_TRAINABLE_PARAMS} "
            f"(blocks[9:12] + norm)"
        )

    trunk.to(device)
    info = {
        "checkpoint": ckpt_path.name,
        "sha256": sha256_file(ckpt_path),
        "trainable_trunk_params": trainable_params,
        "blocks_tuned": [9, 10, 11],
        "norm_tuned": True,
    }
    return trunk, info


def evaluate_heldout_ratio(
    trunk: CompactViTGen2,
    head: nn.Module,
    bank_heldout: S3FolderBank,
    lay: UnitLayout,
    cand: torch.Tensor,
    device: str,
    n_scenes: int = 600,
    seed: int = 1000,
) -> Dict[str, float]:
    """Computes held-out d_native_centre, d_gist_full, and their ratio at size 4, alpha 1.0."""
    trunk.eval()
    head.eval()
    g = torch.Generator().manual_seed(seed)

    # Generate fixed evaluation scenes (size 4, alpha 1.0 only for ratio estimation)
    sc = make_scene_batch(bank_heldout, lay, n_scenes, sizes=[4], alphas=[1.0], p_null=0.3, g=g)
    tl = sample_windows(sc, lay, cand, g)

    canvas = sc["canvas"]
    labels = sc["label"]
    overlaps = sc["overlap"]

    # Extract native tokens and gist tokens
    with torch.no_grad():
        c_dev = canvas.to(device)
        crops = native_crop(c_dev, (tl * lay.patch).to(device), lay.window)
        gists = F.interpolate(c_dev, size=(lay.window, lay.window), mode="area")

        tn = trunk(normalise(crops))[1].float()
        tg = trunk(normalise(gists))[1].float()

        # Score through head
        # head: Linear(384, 51) -> score = logit_c - logit_bg
        out_n = head(tn)  # (B, 64, 51)
        out_g = head(tg)

        scores_n = out_n[:, :, :50] - out_n[:, :, 50:51]  # (B, 64, 50)
        scores_g = out_g[:, :, :50] - out_g[:, :, 50:51]

    # Calculate overlaps
    cn = g1.window_token_overlap(overlaps, tl, lay)  # (B, 64)
    cg = g1.gist_token_overlap(overlaps, lay)  # (B, 64)

    # Split into cal (first half) and test (second half)
    half = n_scenes // 2
    cal_scores_n, tst_scores_n = scores_n[:half], scores_n[half:]
    cal_scores_g, tst_scores_g = scores_g[:half], scores_g[half:]
    cal_labels, tst_labels = labels[:half], labels[half:]
    cal_cn, tst_cn = cn[:half], cn[half:]
    cal_cg, tst_cg = cg[:half], cg[half:]

    # Compute null mean & sd on cal null scenes
    null_mask_cal = cal_labels < 0
    if null_mask_cal.sum() < 10:
        return {"d_native_centre": 0.0, "d_gist_full": 0.0, "ratio": 0.0}

    null_n = cal_scores_n[null_mask_cal]  # (N_null, 64, 50)
    mu_n = null_n.mean(dim=0, keepdim=True)
    sd_n = null_n.std(dim=0, keepdim=True).clamp(min=1e-4)

    null_g = cal_scores_g[null_mask_cal]
    mu_g = null_g.mean(dim=0, keepdim=True)
    sd_g = null_g.std(dim=0, keepdim=True).clamp(min=1e-4)

    # Standardise test scores
    z_n = (tst_scores_n - mu_n) / sd_n  # (N_tst, 64, 50)
    z_g = (tst_scores_g - mu_g) / sd_g

    # Compute d_native_centre (tokens covering target centre)
    tgt_mask_tst = tst_labels >= 0
    d_native_centre = 0.0
    if tgt_mask_tst.sum() > 0:
        pos_z_n = []
        for i in range(half):
            if tst_labels[i] >= 0:
                c = tst_labels[i].item()
                # covered tokens >= 0.5
                cov = tst_cn[i] >= 0.5
                if cov.sum() > 0:
                    pos_z_n.append(z_n[i, cov, c].mean().item())
        if len(pos_z_n) > 0:
            d_native_centre = float(sum(pos_z_n) / len(pos_z_n))

    # Compute d_gist_full (fully covered gist tokens: cg == 1.0)
    d_gist_full = 0.0
    if tgt_mask_tst.sum() > 0:
        pos_z_g = []
        for i in range(half):
            if tst_labels[i] >= 0:
                c = tst_labels[i].item()
                cov = tst_cg[i] >= 0.99
                if cov.sum() > 0:
                    pos_z_g.append(z_g[i, cov, c].mean().item())
        if len(pos_z_g) > 0:
            d_gist_full = float(sum(pos_z_g) / len(pos_z_g))

    ratio = d_native_centre / max(d_gist_full, 1e-4)
    return {
        "d_native_centre": d_native_centre,
        "d_gist_full": d_gist_full,
        "ratio": ratio,
    }


def train_s3(
    ckpt_path: pathlib.Path,
    data_root: pathlib.Path,
    class_split_path: pathlib.Path,
    pools_path: pathlib.Path,
    out_ckpt_path: pathlib.Path,
    out_manifest_path: pathlib.Path,
    device: str = "cuda",
    epochs: int = 5,
    steps_per_epoch: int = 300,
    batch_size: int = 48,
    lr_trunk: float = 1e-5,
    lr_head: float = 1e-3,
    heldout_per_class: int = 50,
    seed: int = 17,
) -> Dict[str, object]:
    """Executes S3 fine-tuning."""
    torch.manual_seed(seed)
    g_train = torch.Generator().manual_seed(seed)

    # 1. Read class split
    sp = g1.read_class_split(class_split_path)
    tcls, bcls = sp["target_classes"], sp["background_classes"]
    C = len(tcls)

    # Verify pools file exists and matches val listing
    pl = g1.read_pools(pools_path, sp["class_split_sha256"])

    # 2. Partition data_root/train into s3_train and s3_heldout
    train_files, heldout_files = partition_train_split(
        data_root, tcls, bcls, heldout_per_class=heldout_per_class, out_manifest=out_manifest_path
    )

    # 3. Create FolderBanks
    bank_train = S3FolderBank(data_root, train_files, tcls, bcls)
    bank_heldout = S3FolderBank(data_root, heldout_files, tcls, bcls)

    lay = UnitLayout(canvas=224, window=112, patch=14, stride_units=2)
    cand = lay.candidate_window_toplefts()
    sizes = [2, 4, 6]
    alphas = [1.0, 0.6]

    # 4. Load trunk and set up trainable parameters
    trunk, trunk_info = load_trunk_for_s3(ckpt_path, window=lay.window, device=device)

    # Temporary training head: Linear(384, 51)
    head = nn.Linear(384, C + 1).to(device)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)

    total_trainable = sum(p.numel() for p in trunk.parameters() if p.requires_grad) + sum(
        p.numel() for p in head.parameters() if p.requires_grad
    )
    print(f"S3 setup: trunk blocks[9:12] + norm ({trunk_info['trainable_trunk_params']} params) + head ({sum(p.numel() for p in head.parameters())} params)")
    print(f"Total trainable parameters: {total_trainable}")

    # Optimizer
    trunk_params = [p for p in trunk.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": lr_trunk, "weight_decay": 0.05},
            {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
        ]
    )
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # Evaluate baseline ratio on heldout before training
    print("Evaluating initial heldout ratio before training...")
    init_metrics = evaluate_heldout_ratio(trunk, head, bank_heldout, lay, cand, device, seed=seed)
    print(
        f"Initial heldout: d_native={init_metrics['d_native_centre']:.3f}, "
        f"d_gist={init_metrics['d_gist_full']:.3f}, ratio={init_metrics['ratio']:.3f}"
    )

    best_ratio = init_metrics["ratio"]
    best_epoch = -1
    best_trunk_state = {k: v.cpu().clone() for k, v in trunk.state_dict().items()}

    log: List[Dict[str, object]] = []

    for epoch in range(epochs):
        trunk.train()
        head.train()
        epoch_losses = []

        t0 = time.time()
        for step in range(steps_per_epoch):
            # Sample scenes
            sc = make_scene_batch(bank_train, lay, batch_size, sizes, alphas, p_null=0.3, g=g_train)
            tl = sample_windows(sc, lay, cand, g_train)

            c_dev = sc["canvas"].to(device)
            crops = native_crop(c_dev, (tl * lay.patch).to(device), lay.window)
            gists = F.interpolate(c_dev, size=(lay.window, lay.window), mode="area")

            # Forward passes
            tn = trunk(normalise(crops))[1]  # (B, 64, 384)
            tg = trunk(normalise(gists))[1]  # (B, 64, 384)

            cn = g1.window_token_overlap(sc["overlap"], tl, lay).to(device)
            cg = g1.gist_token_overlap(sc["overlap"], lay).to(device)

            # Targets & weights
            # Native targets:
            yn = torch.where(
                (cn >= 0.5) & (sc["label"].to(device) >= 0)[:, None],
                sc["label"].to(device)[:, None].expand_as(cn),
                torch.full_like(cn, C, dtype=torch.long),
            )
            wn = torch.where(yn == C, torch.full_like(cn, 0.25), torch.ones_like(cn))

            # Gist targets:
            yg = torch.where(
                (cg >= 0.5) & (sc["label"].to(device) >= 0)[:, None],
                sc["label"].to(device)[:, None].expand_as(cg),
                torch.full_like(cg, C, dtype=torch.long),
            )
            # Skip ambiguous partial overlap tokens (0 < overlap < 0.5)
            wg = ((cg == 0) | (cg >= 0.5)).float() * torch.where(yg == C, torch.full_like(cg, 0.25), torch.ones_like(cg))

            logits_n = head(tn)  # (B, 64, C+1)
            logits_g = head(tg)

            loss_n = (F.cross_entropy(logits_n.flatten(0, 1), yn.flatten(), reduction="none") * wn.flatten()).sum() / wn.sum().clamp(min=1e-6)
            loss_g = (F.cross_entropy(logits_g.flatten(0, 1), yg.flatten(), reduction="none") * wg.flatten()).sum() / wg.sum().clamp(min=1e-6)

            # Equal loss weights
            loss = 0.5 * loss_n + 0.5 * loss_g

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trunk_params + head_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        elapsed = time.time() - t0
        mean_loss = sum(epoch_losses) / len(epoch_losses)

        # Heldout validation
        metrics = evaluate_heldout_ratio(trunk, head, bank_heldout, lay, cand, device, seed=seed + epoch + 1)
        ratio = metrics["ratio"]
        print(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.1f}s): loss={mean_loss:.4f}, "
            f"heldout d_native={metrics['d_native_centre']:.3f}, d_gist={metrics['d_gist_full']:.3f}, "
            f"ratio={ratio:.3f}"
        )

        entry = {
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "heldout_metrics": metrics,
            "elapsed_seconds": elapsed,
        }
        log.append(entry)

        if ratio > best_ratio:
            best_ratio = ratio
            best_epoch = epoch + 1
            best_trunk_state = {k: v.cpu().clone() for k, v in trunk.state_dict().items()}
            print(f" -> New best held-out ratio: {best_ratio:.3f} (saving trunk checkpoint)")

    # Save the tuned trunk checkpoint (discarding head)
    # The output checkpoint follows the standard format so load_trunk can read it cleanly.
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # Read original checkpoint metadata to preserve phase/arm/seeds
    raw_orig = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out_payload = {
        "phase": "ff_112_s3",
        "arm": "s3_tuned",
        "epoch": best_epoch,
        "base_checkpoint": ckpt_path.name,
        "base_checkpoint_sha256": trunk_info["sha256"],
        "best_heldout_ratio": best_ratio,
        "s3_log": log,
        "model": {f"backbone.{k}": v for k, v in best_trunk_state.items()},
    }
    torch.save(out_payload, out_ckpt_path)
    out_sha = sha256_file(out_ckpt_path)
    print(f"Saved tuned trunk to {out_ckpt_path} (sha256: {out_sha})")
    print(f"S3 completed: best_epoch={best_epoch}, best_ratio={best_ratio:.3f}")

    return {
        "output_checkpoint": str(out_ckpt_path),
        "output_checkpoint_sha256": out_sha,
        "best_epoch": best_epoch,
        "best_ratio": best_ratio,
        "initial_ratio": init_metrics["ratio"],
        "trainable_trunk_params": EXPECTED_TRUNK_TRAINABLE_PARAMS,
        "log": log,
    }


def main():
    ap = argparse.ArgumentParser(description="Stage S3 dense window tuning")
    ap.add_argument("--ckpt", default="checkpoints/foundation_ff_112_seed17_final_d00fb750.pth")
    ap.add_argument("--data-root", default="data/imagenet100")
    ap.add_argument("--class-split", default="prereg/class_split.json")
    ap.add_argument("--pools", default="prereg/pools.json")
    ap.add_argument("--out-ckpt", default="checkpoints/foundation_ff_112_s3_tuned.pth")
    ap.add_argument("--out-split", default="prereg/s3_split.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--heldout-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    res = train_s3(
        ckpt_path=pathlib.Path(args.ckpt),
        data_root=pathlib.Path(args.data_root),
        class_split_path=pathlib.Path(args.class_split),
        pools_path=pathlib.Path(args.pools),
        out_ckpt_path=pathlib.Path(args.out_ckpt),
        out_manifest_path=pathlib.Path(args.out_split),
        device=args.device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch,
        lr_trunk=args.lr_trunk,
        lr_head=args.lr_head,
        heldout_per_class=args.heldout_per_class,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=2))


if __name__ == "__main__":
    main()


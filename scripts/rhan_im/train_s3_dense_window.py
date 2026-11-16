"""Stage S3: Dense window fine-tuning of the top trunk blocks.

Pre-registered remedy for Gate 1 failure on 'looking_closer_pays'.
Unfreezes trunk.blocks[9:12] and trunk.norm (5,326,464 parameters);
trains with dense window cross-entropy on both native and gist looks;
evaluates early-stopping on held-out native-over-gist ratio at size 4, alpha 1.0
from train/ (strictly disjoint from val/);
discards the training head and saves the tuned trunk for Gate 1 v2 evaluation.

Total training steps: 5 epochs x 300 steps/epoch x batch 48 = 1,500 steps.
Rule for no improvement: If no epoch beats the baseline held-out ratio, the untuned
trunk is retained (best_epoch = -1) and saved for Gate 1 v2.
Evidence boundary note: Held-out ratio computed on train/ images is an internal
selection signal only and is NOT comparable to Gate 1 v1's 1.06 on val/, because
all train/ images were seen during foundation self-supervised pre-training.
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
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2  # noqa: E402
from rhan_im import gate1 as g1  # noqa: E402
from rhan_im.layout import UnitLayout  # noqa: E402
from rhan_im.trunk_adapter import ff_token_fn, native_crop  # noqa: E402
from rhan_im.window_head import EvidenceHead  # noqa: E402

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
    """Loads images on demand from a pinned list of relative paths within data_root/train.

    No unbounded caching: images are loaded with PIL, converted, and immediately discarded
    after batch creation to strictly prevent RAM exhaustion.
    """

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

    def _load_image(self, p: pathlib.Path) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms.functional as TF
        with Image.open(p) as img:
            img = img.convert("RGB")
            return TF.to_tensor(img)

    def sample_target(self, c: int, g: torch.Generator) -> torch.Tensor:
        fl = self.target_files[c]
        idx = int(torch.randint(0, len(fl), (1,), generator=g).item())
        return self._load_image(fl[idx])

    def sample_bg(self, g: torch.Generator) -> torch.Tensor:
        bg_c = int(torch.randint(0, len(self.bg_files), (1,), generator=g).item())
        fl = self.bg_files[bg_c]
        idx = int(torch.randint(0, len(fl), (1,), generator=g).item())
        return self._load_image(fl[idx])


def partition_train_split(
    data_root: pathlib.Path,
    tcls: List[str],
    bcls: List[str],
    heldout_per_class: int = 50,
    out_manifest: Optional[pathlib.Path] = None,
    pools_path: Optional[pathlib.Path] = None,
) -> Tuple[List[str], List[str]]:
    """Partitions data_root/train into disjoint (train_files, heldout_files) and asserts disjointness from val."""
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

    # Assert internal disjointness
    disjoint = set(heldout_files).isdisjoint(set(train_files))
    if not disjoint:
        raise ValueError("heldout_files and train_files are not disjoint")

    # Assert strict disjointness from validation pools if pools_path provided
    if pools_path and pathlib.Path(pools_path).exists():
        p_raw = json.loads(pathlib.Path(pools_path).read_text())
        val_files = set()
        for part_name in ("calibration_pool_part_0_of_2", "test_pool_part_1_of_2"):
            if part_name in p_raw:
                for c_name, fl in p_raw[part_name].items():
                    for fn in fl:
                        val_files.add(f"{c_name}/{fn}")
        overlap_train = set(train_files) & val_files
        overlap_held = set(heldout_files) & val_files
        if overlap_train or overlap_held:
            raise ValueError(
                f"S3 partition leaks val images! {len(overlap_train)} train overlaps, {len(overlap_held)} held-out overlaps"
            )

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
        bg_raw = bank.sample_bg(g)
        h, w = bg_raw.shape[1], bg_raw.shape[2]
        side = min(h, w)
        bg = TF.center_crop(bg_raw, [side, side])
        bg = F.interpolate(bg.unsqueeze(0), size=(lay.canvas, lay.canvas), mode="bilinear", align_corners=False).squeeze(0)

        is_null = torch.rand(1, generator=g).item() < p_null
        overlap_map = torch.zeros((lay.g, lay.g), dtype=torch.float32)

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

            max_r = lay.g - s
            max_c = lay.g - s
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
            pos = torch.nonzero(ov)
            if len(pos) > 0:
                ctr_r = pos[:, 0].float().mean().item()
                ctr_c = pos[:, 1].float().mean().item()
                good = []
                for w_idx in range(candidate_windows.shape[0]):
                    wr, wc = candidate_windows[w_idx, 0].item(), candidate_windows[w_idx, 1].item()
                    if wr <= ctr_r < wr + lay.tok and wc <= ctr_c < wc + lay.tok:
                        good.append(w_idx)
                if good:
                    pick = good[int(torch.randint(0, len(good), (1,), generator=g).item())]
                    out[i] = candidate_windows[pick]
                    continue
        pick = int(torch.randint(0, candidate_windows.shape[0], (1,), generator=g).item())
        out[i] = candidate_windows[pick]
    return out


@torch.no_grad()
def trunk_tokens(fn: Callable, canvas: torch.Tensor, tl_units: Optional[torch.Tensor], lay: UnitLayout, device: str, chunk: int = 64) -> torch.Tensor:
    """Native look (tl_units given) or gist look (None): (n, T, D) patch tokens on CPU float32."""
    outs = []
    for i in range(0, canvas.shape[0], chunk):
        c = canvas[i:i + chunk].to(device)
        if tl_units is None:
            x = F.interpolate(c, size=(lay.window, lay.window), mode="area")
        else:
            x = native_crop(c, (tl_units[i:i + chunk] * lay.patch).to(device), lay.window)
        outs.append(fn(normalise(x))[1].float().cpu())
    return torch.cat(outs)


def set_trunk_train_mode(trunk: CompactViTGen2) -> None:
    """Sets trunk to training mode while strictly isolating frozen blocks in eval mode.

    Blocks 0..8 and patch_embed remain in eval mode (deterministic, no dropout/drop-path);
    blocks 9..11 and norm are in train mode.
    """
    trunk.train()
    trunk.patch_embed.eval()
    for b in trunk.blocks[:9]:
        b.eval()
    if hasattr(trunk, "refinement"):
        trunk.refinement.eval()


def load_trunk_for_s3(
    ckpt_path: pathlib.Path,
    window: int = 112,
    device: str = "cuda",
) -> Tuple[CompactViTGen2, Dict[str, object]]:
    """Loads CompactViTGen2 from checkpoint and unfreezes only blocks[9:12] and norm."""
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
        b.requires_grad_(True)
    trunk.norm.requires_grad_(True)

    set_trunk_train_mode(trunk)

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


def generate_fixed_eval_scenes(
    bank_heldout: S3FolderBank,
    lay: UnitLayout,
    n_cal: int,
    n_test: int,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Generates fixed calibration and test scenes from held-out train partition.

    Canvases are stored in uint8 on CPU to bound memory (~380 MB for 2,500 scenes).
    Fixed scenes are shared across baseline and all epochs for zero scene-sampling variance.
    """
    cand = lay.candidates(stride_units=2)
    g_cal = torch.Generator().manual_seed(seed + 100)
    g_tst = torch.Generator().manual_seed(seed + 200)

    # Calibration scenes (size 4, alpha 1.0, 30% null)
    sc_cal_raw = make_scene_batch(bank_heldout, lay, n_cal, sizes=[4], alphas=[1.0], p_null=0.3, g=g_cal)
    tl_cal = sample_windows(sc_cal_raw, lay, cand, g_cal)
    sc_cal = {
        "canvas_uint8": (sc_cal_raw["canvas"] * 255.0).clamp(0, 255).to(torch.uint8),
        "label": sc_cal_raw["label"],
        "overlap": sc_cal_raw["overlap"],
        "tl": tl_cal,
        "cover_native": g1.window_token_overlap(sc_cal_raw["overlap"], tl_cal, lay) > 0,
        "ov_gist": g1.gist_token_overlap(sc_cal_raw["overlap"], lay),
        "cover_gist": g1.gist_token_overlap(sc_cal_raw["overlap"], lay) > 0,
    }

    # Test scenes (size 4, alpha 1.0, 30% null)
    sc_tst_raw = make_scene_batch(bank_heldout, lay, n_test, sizes=[4], alphas=[1.0], p_null=0.3, g=g_tst)
    tl_tst = sample_windows(sc_tst_raw, lay, cand, g_tst)
    sc_tst = {
        "canvas_uint8": (sc_tst_raw["canvas"] * 255.0).clamp(0, 255).to(torch.uint8),
        "label": sc_tst_raw["label"],
        "overlap": sc_tst_raw["overlap"],
        "tl": tl_tst,
        "cover_native": g1.window_token_overlap(sc_tst_raw["overlap"], tl_tst, lay) > 0,
        "ov_gist": g1.gist_token_overlap(sc_tst_raw["overlap"], lay),
        "cover_gist": g1.gist_token_overlap(sc_tst_raw["overlap"], lay) > 0,
    }

    return sc_cal, sc_tst


@torch.no_grad()
def extract_split_scores(
    trunk: CompactViTGen2,
    head: nn.Module,
    sc: Dict[str, torch.Tensor],
    lay: UnitLayout,
    device: str,
    chunk_size: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs trunk and head on sc in chunks: returns (scores_native, scores_gist) on CPU float32."""
    trunk.eval()
    head.eval()
    fn = ff_token_fn(trunk)
    n = sc["canvas_uint8"].shape[0]

    native_scores = []
    gist_scores = []

    for i in range(0, n, chunk_size):
        c_chunk = (sc["canvas_uint8"][i:i + chunk_size].float() / 255.0).to(device)
        tl_chunk = sc["tl"][i:i + chunk_size]

        # Native crops
        crop = native_crop(c_chunk, (tl_chunk * lay.patch).to(device), lay.window)
        tn = fn(normalise(crop))[1]
        sn = head(tn)
        native_scores.append(sn.cpu().float())

        # Gist view
        gist = F.interpolate(c_chunk, size=(lay.window, lay.window), mode="area")
        tg = fn(normalise(gist))[1]
        sg = head(tg)
        gist_scores.append(sg.cpu().float())

    return torch.cat(native_scores, dim=0), torch.cat(gist_scores, dim=0)


def evaluate_heldout_ratio(
    trunk: CompactViTGen2,
    head: nn.Module,
    sc_cal: Dict[str, torch.Tensor],
    sc_tst: Dict[str, torch.Tensor],
    lay: UnitLayout,
    device: str,
    min_pos: int = 30,
) -> Dict[str, float]:
    """Computes held-out d_native_centre, d_gist, d_gist_full, and their ratio using Gate 1 estimators."""
    trunk.eval()
    head.eval()

    # 1. Extract scores for cal and tst splits
    cal_native, cal_gist = extract_split_scores(trunk, head, sc_cal, lay, device)
    tst_native, tst_gist = extract_split_scores(trunk, head, sc_tst, lay, device)

    C = cal_native.shape[-1]

    # 2. Fit standardizer on calibration split
    mu_n, sd_n = g1.fit_standardizer(cal_native, sc_cal["cover_native"])
    mu_g, sd_g = g1.fit_standardizer(cal_gist, sc_cal["cover_gist"])

    # 3. Measure tokens on test split (Gate 1 exact estimator functions)
    lab = sc_tst["label"]
    m_native = g1.measure_tokens(tst_native, sc_tst["cover_native"], lab, mu_n, sd_n, lay.tok, min_pos=min_pos)
    d_native_centre = g1.region_means(m_native["d_shift"], m_native["n_pos"])["centre"]

    m_gist = g1.measure_tokens(tst_gist, sc_tst["cover_gist"], lab, mu_g, sd_g, lay.tok, min_pos=min_pos)
    d_gist = g1.region_means(m_gist["d_shift"], m_gist["n_pos"])["all"]

    # 4. Gist full on operating cell
    zg = (tst_gist - mu_g) / sd_g
    own = zg.gather(2, lab.clamp_min(0)[:, None, None].expand(-1, lay.n_coarse, 1)).squeeze(2)
    has = (lab >= 0)[:, None]
    full_m = (sc_tst["ov_gist"] >= 0.99) & has
    nullg = g1.null_mask(sc_tst["cover_gist"], C)
    null_mean = float(zg[nullg].mean()) if bool(nullg.any()) else 0.0
    d_gist_full = float(own[full_m].mean() - null_mean) if int(full_m.sum()) >= min_pos else float("nan")

    # 5. Native over gist ratio (Gate 1 exact comparator)
    gists = [v for v in (d_gist, d_gist_full) if math.isfinite(v)]
    dg = max(gists) if gists else float("nan")
    ratio = d_native_centre / max(dg, 1e-6) if (gists and math.isfinite(d_native_centre)) else float("nan")

    return {
        "d_native_centre": float(d_native_centre),
        "d_gist": float(d_gist),
        "d_gist_full": float(d_gist_full),
        "ratio": float(ratio),
    }


def refit_eval_head(
    trunk: CompactViTGen2,
    bank_train: S3FolderBank,
    lay: UnitLayout,
    device: str,
    steps: int = 600,
    batch_size: int = 48,
    lr: float = 2e-3,
    head_seed: int = 17,
) -> EvidenceHead:
    """Fits an EvidenceHead on frozen features from current trunk, matching Gate 1 v1 procedure."""
    trunk.eval()
    cand = lay.candidates(stride_units=2)
    fn = ff_token_fn(trunk)
    gtr = torch.Generator().manual_seed(head_seed + 10)
    sizes = [2, 4, 6]
    alphas = [1.0, 0.6]

    def train_batch(step: int):
        sc = make_scene_batch(bank_train, lay, batch_size, sizes, alphas, p_null=0.3, g=gtr)
        tl = sample_windows(sc, lay, cand, gtr)
        tn = trunk_tokens(fn, sc["canvas"], tl, lay, device)
        tg = trunk_tokens(fn, sc["canvas"], None, lay, device)
        cn = g1.window_token_overlap(sc["overlap"], tl, lay)
        cg = g1.gist_token_overlap(sc["overlap"], lay)
        cover = torch.cat([cn >= 0.5, cg >= 0.5])
        weight = torch.cat([torch.ones_like(cn), ((cg == 0) | (cg >= 0.5)).float()])
        return torch.cat([tn, tg]), cover, torch.cat([sc["label"], sc["label"]]), weight

    head, _ = g1.train_evidence_head(
        train_batch,
        d_model=384,
        n_classes=bank_train.n_classes,
        steps=steps,
        lr=lr,
        seed=head_seed,
    )
    head.eval()
    return head


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
    eval_head_steps: int = 600,
    n_eval_cal: int = 1000,
    n_eval_test: int = 1500,
    heldout_per_class: int = 50,
    seed: int = 17,
) -> Dict[str, object]:
    """Executes S3 fine-tuning."""
    torch.manual_seed(seed)
    g_train = torch.Generator().manual_seed(seed)

    # 1. Read class split
    sp = g1.read_class_split(class_split_path)
    tcls, bcls = sp["targets"], sp["backgrounds"]
    C = len(tcls)

    # Verify pools file exists and matches class split
    pl = g1.read_pools(pools_path, sp["class_split_sha256"])

    # 2. Partition data_root/train into s3_train and s3_heldout (asserting disjointness from val)
    train_files, heldout_files = partition_train_split(
        data_root, tcls, bcls, heldout_per_class=heldout_per_class, out_manifest=out_manifest_path, pools_path=pools_path
    )

    # 3. Create FolderBanks (on-demand loading, no unbounded cache)
    bank_train = S3FolderBank(data_root, train_files, tcls, bcls)
    bank_heldout = S3FolderBank(data_root, heldout_files, tcls, bcls)

    lay = UnitLayout(canvas=224, window=112, patch=14, target_units=4)
    cand = lay.candidates(stride_units=2)
    sizes = [2, 4, 6]
    alphas = [1.0, 0.6]

    # 4. Load trunk and set up trainable parameters
    trunk, trunk_info = load_trunk_for_s3(ckpt_path, window=lay.window, device=device)

    # 5. Pre-generate fixed held-out evaluation scenes (uint8 on CPU, zero scene-sampling variance)
    print(f"Pre-generating fixed held-out scenes ({n_eval_cal} cal, {n_eval_test} test) for early stopping...")
    sc_cal, sc_tst = generate_fixed_eval_scenes(bank_heldout, lay, n_eval_cal, n_eval_test, seed=seed)

    # Co-training head: Linear(384, C + 1)
    head_train = nn.Linear(384, C + 1).to(device)
    nn.init.normal_(head_train.weight, std=0.01)
    nn.init.zeros_(head_train.bias)

    total_trainable = sum(p.numel() for p in trunk.parameters() if p.requires_grad) + sum(
        p.numel() for p in head_train.parameters() if p.requires_grad
    )
    print(
        f"S3 setup: trunk blocks[9:12] + norm ({trunk_info['trainable_trunk_params']} params) + "
        f"co-training head ({sum(p.numel() for p in head_train.parameters())} params)"
    )
    print(f"Total trainable parameters: {total_trainable}")

    # Optimizer
    trunk_params = [p for p in trunk.parameters() if p.requires_grad]
    head_params = list(head_train.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": lr_trunk, "weight_decay": 0.05},
            {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
        ]
    )
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # Evaluate baseline ratio on untuned trunk with a freshly fitted head (matching Gate 1 procedure)
    print(f"Refitting baseline evaluation head ({eval_head_steps} steps) on untuned trunk...")
    baseline_head = refit_eval_head(trunk, bank_train, lay, device, steps=eval_head_steps, head_seed=seed)
    print("Evaluating initial held-out ratio on untuned trunk...")
    init_metrics = evaluate_heldout_ratio(trunk, baseline_head, sc_cal, sc_tst, lay, device)
    print(
        f"Initial held-out: d_native={init_metrics['d_native_centre']:.3f}, "
        f"d_gist={init_metrics['d_gist']:.3f}, d_gist_full={init_metrics['d_gist_full']:.3f}, "
        f"ratio={init_metrics['ratio']:.3f}"
    )

    best_ratio = init_metrics["ratio"]
    best_epoch = -1
    best_trunk_state = {k: v.cpu().clone() for k, v in trunk.state_dict().items()}

    log: List[Dict[str, object]] = []

    for epoch in range(epochs):
        set_trunk_train_mode(trunk)
        head_train.train()
        epoch_losses = []

        t0 = time.time()
        for step in range(steps_per_epoch):
            sc = make_scene_batch(bank_train, lay, batch_size, sizes, alphas, p_null=0.3, g=g_train)
            tl = sample_windows(sc, lay, cand, g_train)

            c_dev = sc["canvas"].to(device)
            crops = native_crop(c_dev, (tl * lay.patch).to(device), lay.window)
            gists = F.interpolate(c_dev, size=(lay.window, lay.window), mode="area")

            tn = trunk(normalise(crops))[1]  # (B, 64, 384)
            tg = trunk(normalise(gists))[1]  # (B, 64, 384)

            cn = g1.window_token_overlap(sc["overlap"], tl, lay).to(device)
            cg = g1.gist_token_overlap(sc["overlap"], lay).to(device)

            # Native targets and weights
            yn = torch.where(
                (cn >= 0.5) & (sc["label"].to(device) >= 0)[:, None],
                sc["label"].to(device)[:, None].expand_as(cn),
                torch.full_like(cn, C, dtype=torch.long),
            )
            wn = torch.where(yn == C, torch.full_like(cn, 0.25), torch.ones_like(cn))

            # Gist targets and weights
            yg = torch.where(
                (cg >= 0.5) & (sc["label"].to(device) >= 0)[:, None],
                sc["label"].to(device)[:, None].expand_as(cg),
                torch.full_like(cg, C, dtype=torch.long),
            )
            wg = ((cg == 0) | (cg >= 0.5)).float() * torch.where(yg == C, torch.full_like(cg, 0.25), torch.ones_like(cg))

            logits_n = head_train(tn)  # (B, 64, C+1)
            logits_g = head_train(tg)

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

        # Held-out validation: refit fresh head on current trunk checkpoint, matching Gate 1
        trunk.eval()
        eval_head = refit_eval_head(trunk, bank_train, lay, device, steps=eval_head_steps, head_seed=seed)
        metrics = evaluate_heldout_ratio(trunk, eval_head, sc_cal, sc_tst, lay, device)
        ratio = metrics["ratio"]
        print(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.1f}s): loss={mean_loss:.4f}, "
            f"held-out d_native={metrics['d_native_centre']:.3f}, d_gist={metrics['d_gist']:.3f}, "
            f"d_gist_full={metrics['d_gist_full']:.3f}, ratio={ratio:.3f}"
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
            print(f" -> New best held-out ratio: {best_ratio:.3f} (checkpoint staged)")

    # Save the tuned trunk checkpoint (discarding head)
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "phase": "ff_112_s3",
        "arm": "s3_tuned",
        "epoch": best_epoch,
        "base_checkpoint": ckpt_path.name,
        "base_checkpoint_sha256": trunk_info["sha256"],
        "best_heldout_ratio": best_ratio,
        "initial_ratio": init_metrics["ratio"],
        "s3_log": log,
        "model": {f"backbone.{k}": v for k, v in best_trunk_state.items()},
    }
    torch.save(out_payload, out_ckpt_path)
    out_sha = sha256_file(out_ckpt_path)
    print(f"Saved trunk to {out_ckpt_path} (sha256: {out_sha})")
    print(f"S3 completed: best_epoch={best_epoch}, best_ratio={best_ratio:.3f} (baseline was {init_metrics['ratio']:.3f})")

    return {
        "output_checkpoint": str(out_ckpt_path),
        "output_checkpoint_sha256": out_sha,
        "best_epoch": best_epoch,
        "best_ratio": best_ratio,
        "initial_ratio": init_metrics["ratio"],
        "trainable_trunk_params": EXPECTED_TRUNK_TRAINABLE_PARAMS,
        "total_steps": epochs * steps_per_epoch,
        "log": log,
    }


def main():
    ap = argparse.ArgumentParser(description="Stage S3 dense window tuning (1,500 total steps)")
    ap.add_argument("--ckpt", default="checkpoints/foundation_ff_112_seed17_final_d00fb750.pth")
    ap.add_argument("--data-root", default="data/imagenet100")
    ap.add_argument("--class-split", default="prereg/class_split.json")
    ap.add_argument("--pools", default="prereg/pools.json")
    ap.add_argument("--out-ckpt", default="checkpoints/foundation_ff_112_s3_tuned.pth")
    ap.add_argument("--out-split", default="prereg/s3_split.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5, help="Number of epochs (default 5; 5 x 300 = 1500 steps)")
    ap.add_argument("--steps-per-epoch", type=int, default=300, help="Steps per epoch (default 300)")
    ap.add_argument("--batch", type=int, default=48, help="Batch size (default 48)")
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--eval-head-steps", type=int, default=600, help="Steps to refit eval head at each epoch (default 600)")
    ap.add_argument("--n-eval-cal", type=int, default=1000, help="Fixed held-out cal scenes (default 1000)")
    ap.add_argument("--n-eval-test", type=int, default=1500, help="Fixed held-out test scenes (default 1500)")
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
        eval_head_steps=args.eval_head_steps,
        n_eval_cal=args.n_eval_cal,
        n_eval_test=args.n_eval_test,
        heldout_per_class=args.heldout_per_class,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=2))


if __name__ == "__main__":
    main()

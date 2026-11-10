#!/usr/bin/env python3
"""Gate 1 for RHAN-IM: MEASURE the sensor, do not assume it.

Question  Does the frozen ff_112 trunk's token-level evidence behave like the equal-variance Gaussian
          sensor that the RHAN-IM Bayes state assumes, and does looking closer pay?

Procedure (everything frozen except a 384 x C linear head)
  1. Scenes. A background canvas (default 224 x 224) gets, with probability 1 - p_null, one square image
     of a TARGET class pasted at a unit-aligned position (size in 14-px units, opacity alpha). Target
     images come from `--data-root/<split>/<class>/*`; backgrounds from classes that are NEVER targets
     (the class list is split in two), so clutter is natural but class-free for the target label set.
  2. Looks. A native look = a window of the trunk's own size (112 px) cut from the canvas WITHOUT resampling.
     The gist look = the whole canvas area-resampled to the window size. Same trunk, same head.
  3. Head. Linear(384, C + 1) trained with cross-entropy on token-level labels (class c on tokens that
     overlap a class-c target, background elsewhere), exported as score_c = logit_c - logit_bg.
  4. Calibration split: per (look type, token position, class) null mean / sd. Test split: per token position
     d (shift in null-sd units), d', z-ROC slope, null / positive sd, wrong-class leakage. Calibration and
     test scenes are built from DISJOINT halves of the val images (every second file of each class).
  5. Cross-look dependence: correlation of the standardised score of the same (scene, unit, class) between
     two different native windows -> "sum" (independent looks) vs "best" (nested deterministic views).
     Within-look dependence: variance-inflation factor (VIF) of the d-weighted sum over a block of tokens of one
     look (1 = independent tokens, as the Bayes state assumes) on null scenes and on target scenes.
     Spillover: does the own-class score rise on tokens that do NOT overlap the target (the footprint model says no)?
  6. Draft verdict (thresholds in rhan_im.gate1.DEFAULT_THRESHOLDS; to be frozen in a pre-registration
     BEFORE the first real run).

What this does NOT show: that natural objects behave like pasted squares; that d' is the same for other
sizes / backgrounds than the ones swept; anything about adversarial robustness (that is a later gate).

Usage
    python scripts/rhan_im/gate1_measure_dprime.py --self-test                    # no data, random weights, CPU
    python scripts/rhan_im/gate1_measure_dprime.py --ckpt foundation_ff_112_seed17_final_d00fb750.pth \\
        --data-root data/imagenet100 --device cuda --out docs/research/RHAN_IM_gate1_ff112_seed17.json
The checkpoint is only READ. The script never writes next to it, never reads credentials, uses no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2  # noqa: E402
from rhan_im import gate1 as g1  # noqa: E402
from rhan_im.layout import UnitLayout  # noqa: E402
from rhan_im.trunk_adapter import ff_token_fn, native_crop  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- image sources
class SyntheticBank:
    """Class-conditional procedural textures + smooth noise backgrounds: for the self-test only."""

    def __init__(self, n_classes: int, seed: int = 0) -> None:
        self.n_classes = n_classes
        g = torch.Generator().manual_seed(seed)
        self.hue = torch.rand(n_classes, 3, generator=g)
        self.theta = torch.rand(n_classes, generator=g) * math.pi
        self.freq = 2.0 + 6.0 * torch.rand(n_classes, generator=g)

    def target(self, c: int, size: int, gen: torch.Generator) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
        phase = torch.rand(1, generator=gen) * 6.283
        wave = torch.sin(6.283 * self.freq[c] * (xx * math.cos(float(self.theta[c])) + yy * math.sin(float(self.theta[c]))) + phase)
        img = 0.5 + 0.35 * self.hue[c].view(3, 1, 1) * wave[None] + 0.15 * (self.hue[c].view(3, 1, 1) - 0.5)
        return img.clamp(0, 1)

    def background(self, size: int, gen: torch.Generator) -> torch.Tensor:
        coarse = torch.randn(1, 3, 6, 6, generator=gen)
        fine = torch.randn(1, 3, size, size, generator=gen)
        bg = 0.5 + 0.12 * F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False) + 0.05 * fine
        return bg[0].clamp(0, 1)


class FolderBank:
    """ImageFolder-style root/<class>/<img>. Targets use `target_classes`, backgrounds `bg_classes`."""

    def __init__(self, root: pathlib.Path, split: str, target_classes: List[str], bg_classes: List[str], seed: int = 0,
                 part: Optional[Tuple[int, int]] = None) -> None:
        """`part = (i, n)` keeps every n-th file of each class's sorted list starting at i, so that two banks built
        from the same split with different `i` draw from DISJOINT image pools (calibration vs test)."""
        from PIL import Image  # local import: the self-test does not need PIL
        self.Image = Image
        self.rng = random.Random(seed)
        base = root / split
        self.files = {c: sorted(p for p in (base / c).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) for c in target_classes + bg_classes}
        if part is not None:
            i, n = part
            self.files = {c: f[i::n] for c, f in self.files.items()}
            empty = [c for c, f in self.files.items() if not f]
            if empty:
                raise SystemExit(f"class(es) {empty[:3]} have too few images for a {n}-way disjoint split")
        self.target_classes, self.bg_classes = target_classes, bg_classes
        self.n_classes = len(target_classes)

    def _load(self, path: pathlib.Path, size: int) -> torch.Tensor:
        im = self.Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), self.Image.BICUBIC)
        return torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1).float() / 255.0

    def target(self, c: int, size: int, gen: torch.Generator) -> torch.Tensor:
        return self._load(self.rng.choice(self.files[self.target_classes[c]]), size)

    def background(self, size: int, gen: torch.Generator) -> torch.Tensor:
        return self._load(self.rng.choice(self.files[self.rng.choice(self.bg_classes)]), size)


# ----------------------------------------------------------------------------- scenes
def scene_geometry(lay: UnitLayout, n: int, n_classes: int, sizes: List[int], alphas: List[float], p_null: float, gen: torch.Generator):
    """Which scenes contain which target where (no pixels): label (n,) (-1: no target), size_units, alpha, top/left px,
    overlap (n, g, g) = fraction of every fine unit covered by the target."""
    label = torch.where(torch.rand(n, generator=gen) < p_null, torch.full((n,), -1), torch.randint(0, n_classes, (n,), generator=gen))
    size_u = torch.tensor(sizes)[torch.randint(0, len(sizes), (n,), generator=gen)]
    alpha = torch.tensor(alphas)[torch.randint(0, len(alphas), (n,), generator=gen)]
    top = (torch.rand(n, generator=gen) * (lay.g - size_u + 1).float()).long() * lay.patch
    left = (torch.rand(n, generator=gen) * (lay.g - size_u + 1).float()).long() * lay.patch
    overlap = torch.zeros(n, lay.g, lay.g)
    for s in sorted(set(sizes)):
        idx = ((size_u == s) & (label >= 0)).nonzero().squeeze(1)
        if len(idx):
            overlap[idx] = g1.unit_overlap(top[idx], left[idx], s * lay.patch, lay.g, lay.patch)
    return {"label": label, "size_units": size_u, "alpha": alpha, "top": top, "left": left, "overlap": overlap}


def make_scenes(bank, lay: UnitLayout, n: int, sizes: List[int], alphas: List[float], p_null: float, gen: torch.Generator):
    """Returns canvas (n,3,H,W) in [0,1] plus the geometry of scene_geometry()."""
    canvas = torch.stack([bank.background(lay.canvas, gen) for _ in range(n)])
    geo = scene_geometry(lay, n, bank.n_classes, sizes, alphas, p_null, gen)
    label, size_u, top, left, alpha = geo["label"], geo["size_units"], geo["top"], geo["left"], geo["alpha"]
    for s in sorted(set(sizes)):
        idx = ((size_u == s) & (label >= 0)).nonzero().squeeze(1)
        if len(idx) == 0:
            continue
        patches = torch.stack([bank.target(int(label[i]), s * lay.patch, gen) for i in idx])
        canvas[idx] = g1.paste_squares(canvas[idx], patches, top[idx], left[idx], alpha[idx])
    return {"canvas": canvas, **geo}


def sample_windows(sc, lay: UnitLayout, cand: torch.Tensor, gen: torch.Generator, p_on_target: float = 0.7) -> torch.Tensor:
    """(n, 2) window top-left in fine units: with prob p_on_target the window contains the target centre."""
    n = sc["label"].shape[0]
    pick = torch.randint(0, cand.shape[0], (n,), generator=gen)
    tl = cand[pick].clone()
    centre_r = (sc["top"] // lay.patch) + sc["size_units"] // 2
    centre_c = (sc["left"] // lay.patch) + sc["size_units"] // 2
    for i in range(n):
        if sc["label"][i] < 0 or torch.rand(1, generator=gen).item() > p_on_target:
            continue
        ok = ((cand[:, 0] <= centre_r[i]) & (centre_r[i] < cand[:, 0] + lay.tok) & (cand[:, 1] <= centre_c[i]) & (centre_c[i] < cand[:, 1] + lay.tok)).nonzero().squeeze(1)
        if len(ok):
            tl[i] = cand[ok[torch.randint(0, len(ok), (1,), generator=gen)]]
    return tl


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


@torch.no_grad()
def trunk_tokens(fn, canvas: torch.Tensor, tl_units: Optional[torch.Tensor], lay: UnitLayout, device: str, chunk: int = 64) -> torch.Tensor:
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


# ----------------------------------------------------------------------------- trunk
def load_trunk(ckpt: Optional[str], window: int, device: str):
    trunk = CompactViTGen2(img_size=window, num_refine_iters=2)
    info: Dict[str, object] = {"checkpoint": None}
    if ckpt:
        p = pathlib.Path(ckpt)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        raw = torch.load(p, map_location="cpu", weights_only=False)
        meta = {k: raw[k] for k in ("phase", "arm", "epoch", "val_clean_acc", "val_rob_acc") if isinstance(raw, dict) and k in raw}
        sd = raw.get("model", raw.get("model_state_dict", raw.get("state_dict", raw)))
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        own = trunk.state_dict()
        used = {k: v for k, v in bb.items() if k in own and own[k].shape == v.shape and not k.startswith("refinement.")}
        need = [k for k in own if not k.startswith("refinement.")]
        frac = len(used) / max(1, len(need))
        if frac < 0.99:
            raise SystemExit(f"checkpoint matches only {frac:.1%} of the trunk keys; refusing to measure a half-loaded trunk")
        trunk.load_state_dict({**own, **used}, strict=True)
        info = {"checkpoint": p.name, "sha256": h.hexdigest(), "trunk_keys_loaded": len(used), "trunk_keys_needed": len(need), **meta}
    trunk.eval().requires_grad_(False).to(device)
    return trunk, info


# ----------------------------------------------------------------------------- analysis (pure function of scores + geometry)
def analyse(cal: Dict, tst: Dict, lay: UnitLayout, sizes: List[int], alphas: List[float], op_size: int, n_pair: int) -> Dict[str, object]:
    """Everything after the scores are collected: standardise on `cal`, measure on `tst`. A pure function of the
    scores and the scene geometry (no trunk, no data), so gate1_operating_characteristics.py can run exactly this
    code on synthetic scores with known properties."""
    C = tst["native"].shape[2]
    mu_n, sd_n = g1.fit_standardizer(cal["native"], cal["cover_native"])
    mu_g, sd_g = g1.fit_standardizer(cal["gist"], cal["cover_gist"])

    # ---- 3. d, d', slope, sds, leakage per token position, per (size, alpha) -------------------------------
    report: Dict[str, object] = {"native": {}, "gist": {}, "within_look": {}, "spillover": {}}
    for s in sizes:
        for a in alphas:
            sel = (tst["sc"]["size_units"] == s) & (tst["sc"]["alpha"] == a) | (tst["sc"]["label"] < 0)
            key = f"size{s}_alpha{a}"
            lab = tst["sc"]["label"][sel]
            m = g1.measure_tokens(tst["native"][sel], tst["cover_native"][sel], lab, mu_n, sd_n, lay.tok)
            mg = g1.measure_tokens(tst["gist"][sel], tst["cover_gist"][sel], lab, mu_g, sd_g, lay.tok, min_pos=20)
            report["native"][key] = {k: v.tolist() for k, v in m.items()}
            report["native"][key]["regions_d_shift"] = g1.region_means(m["d_shift"], m["n_pos"])
            report["gist"][key] = {k: v.tolist() for k, v in mg.items()}
            report["gist"][key]["d_shift_mean"] = g1.region_means(mg["d_shift"], mg["n_pos"])["all"]
            # within-look dependence of the tokens (the Bayes state adds token evidence as if independent)
            report["within_look"][key] = {
                "native": g1.block_vif((tst["native"][sel] - mu_n) / sd_n, tst["cover_native"][sel], lab, lay.tok, block=s, weights=m["d_shift"]),
                "gist": g1.block_vif((tst["gist"][sel] - mu_g) / sd_g, tst["cover_gist"][sel], lab, lay.tok, block=max(1, s // lay.coarse_block),
                                     weights=mg["d_shift"], min_pos=20),
            }
            # does evidence leak onto tokens outside the target (footprint check)?
            report["spillover"][key] = g1.spillover((tst["native"][sel] - mu_n) / sd_n, tst["cover_native"][sel], lab, lay.tok)
    # gist detectability by overlap fraction (the Bayes model uses one d for every overlapped coarse unit)
    full = tst["ov_gist"] >= 0.99
    zg = (tst["gist"] - mu_g) / sd_g
    own = zg.gather(2, tst["sc"]["label"].clamp_min(0)[:, None, None].expand(-1, lay.n_coarse, 1)).squeeze(2)
    has = (tst["sc"]["label"] >= 0)[:, None]
    nullg = g1.null_mask(tst["cover_gist"], C)
    report["gist_by_overlap"] = {
        "fully_covered_token": float(own[full & has].mean() - zg[nullg].mean()) if bool((full & has).any()) else None,
        "partially_covered_token": float(own[(tst["ov_gist"] > 0) & (tst["ov_gist"] < 0.99) & has].mean() - zg[nullg].mean()) if bool(((tst["ov_gist"] > 0) & (tst["ov_gist"] < 0.99) & has).any()) else None,
    }

    # ---- 4. cross-look dependence between two native windows of the same scene ---------------------------------
    if n_pair > 0:
        sc = {k: v[:n_pair] for k, v in tst["sc"].items()}
        tl2 = tst["tl2"]
        z1 = (tst["native"][:n_pair] - mu_n) / sd_n                           # first window: the scores already collected
        z2 = (tst["native2"] - mu_n) / sd_n                                   # second, independently sampled window
        cov1 = g1.window_token_overlap(sc["overlap"], tst["tl"][:n_pair], lay) > 0
        cov2 = g1.window_token_overlap(sc["overlap"], tl2, lay) > 0
        na, nb, pa, pb = [], [], [], []
        d_map = torch.tensor(report["native"][f"size{op_size if op_size in sizes else sizes[0]}_alpha{alphas[0]}"]["d_shift"]).flatten()
        for i in range(n_pair):
            m1 = torch.full((lay.g * lay.g,), -1, dtype=torch.long)
            m2 = torch.full((lay.g * lay.g,), -1, dtype=torch.long)
            i1, v1 = lay.window_units(tst["tl"][i:i + 1])
            i2, v2 = lay.window_units(tl2[i:i + 1])
            m1[i1[0][v1[0]]] = torch.arange(lay.tok ** 2)[v1[0]]
            m2[i2[0][v2[0]]] = torch.arange(lay.tok ** 2)[v2[0]]
            common = ((m1 >= 0) & (m2 >= 0)).nonzero().squeeze(1)
            if len(common) == 0:
                continue
            ta, tb = m1[common], m2[common]
            both_clear = (~cov1[i, ta]) & (~cov2[i, tb])
            if both_clear.any():
                na.append(z1[i, ta[both_clear]].flatten())
                nb.append(z2[i, tb[both_clear]].flatten())
            lab = int(sc["label"][i])
            both_cov = cov1[i, ta] & cov2[i, tb]
            if lab >= 0 and both_cov.any():
                pa.append(z1[i, ta[both_cov], lab] - torch.nan_to_num(d_map[ta[both_cov]], nan=0.0))
                pb.append(z2[i, tb[both_cov], lab] - torch.nan_to_num(d_map[tb[both_cov]], nan=0.0))
        rho_null = g1.paired_noise_correlation(torch.cat(na), torch.cat(nb)) if na else float("nan")
        rho_pos = g1.paired_noise_correlation(torch.cat(pa), torch.cat(pb)) if pa else float("nan")
        report["cross_look"] = {"rho_null": rho_null, "rho_signal_residual": rho_pos, "n_null_pairs": int(sum(len(x) for x in na)), "n_pos_pairs": int(sum(len(x) for x in pa)),
                                "effective_looks_of_8_at_rho_null": g1.effective_looks(8, rho_null) if rho_null == rho_null else None}
    else:                                                               # not measured (e.g. operating-characteristics simulation)
        rho_null = float("nan")
        report["cross_look"] = {"rho_null": rho_null, "rho_signal_residual": float("nan"), "n_null_pairs": 0, "n_pos_pairs": 0, "effective_looks_of_8_at_rho_null": None}

    # ---- 5. summary + draft verdict at the operating point -----------------------------------------------------
    op_s = op_size if op_size in sizes else sizes[0]
    op = f"size{op_s}_alpha{alphas[0]}"
    nat, gist = report["native"][op], report["gist"][op]

    def nanmedian(x):
        t = torch.tensor(x).flatten()
        t = t[torch.isfinite(t)]
        return float(t.median()) if t.numel() else float("nan")

    summary = {
        "operating_point": op,
        "d_native_centre": nat["regions_d_shift"]["centre"], "d_native_edge_ring": nat["regions_d_shift"]["edge_ring"], "d_gist": gist["d_shift_mean"],
        "zroc_slope_median": nanmedian(nat["zroc_slope"]), "null_sd_median": nanmedian(nat["null_sd"]), "pos_sd_median": nanmedian(nat["pos_sd"]),
        "leakage_max": float(torch.tensor(nat["leakage"]).nan_to_num(0).abs().max()), "rho_null": rho_null,
        "vif_null_op": report["within_look"][op]["native"]["vif_null"], "vif_pos_op": report["within_look"][op]["native"]["vif_pos"],
        "vif_pos_scenes_op": float(report["within_look"][op]["native"]["n_pos_scenes"]),
        "op_block_tokens": float(op_s * op_s),
        "d_gist_full": report["gist_by_overlap"]["fully_covered_token"] if report["gist_by_overlap"]["fully_covered_token"] is not None else float("nan"),
        "spill_adjacent_op": report["spillover"][op]["adjacent"], "spill_far_op": report["spillover"][op]["far"],
    }
    report["summary"] = summary
    report["verdict_draft"] = g1.verdict(summary)
    report["thresholds_draft"] = {k: list(v) if isinstance(v, tuple) else v for k, v in g1.DEFAULT_THRESHOLDS.items()}
    report["counts"] = {"n_cal": int(cal["native"].shape[0]), "n_test": int(tst["native"].shape[0]), "n_pairs": int(tst["native2"].shape[0]),
                        "n_test_null_scenes": int((tst["sc"]["label"] < 0).sum()), "n_test_target_scenes": int((tst["sc"]["label"] >= 0).sum())}
    return report


# ----------------------------------------------------------------------------- main experiment
def run(args) -> Dict[str, object]:
    t0 = time.time()
    torch.manual_seed(args.seed)
    dev = args.device
    lay = UnitLayout(canvas=args.canvas, patch=14, window=args.window, target_units=args.op_size)
    cand = lay.candidates(args.stride_units)
    trunk, info = load_trunk(args.ckpt, args.window, dev)
    fn = ff_token_fn(trunk)

    if args.self_test:
        bank_train = bank_cal = bank_test = SyntheticBank(args.classes, seed=1)
    else:
        root = pathlib.Path(args.data_root)
        classes = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
        half = len(classes) // 2
        tcls, bcls = classes[:half][:args.classes], classes[half:]
        bank_train = FolderBank(root, "train", tcls, bcls, seed=args.seed)
        bank_cal = FolderBank(root, "val", tcls, bcls, seed=args.seed + 1, part=(0, 2))        # calibration and test scenes use
        bank_test = FolderBank(root, "val", tcls, bcls, seed=args.seed + 2, part=(1, 2))       # DISJOINT halves of the val images
    C = bank_train.n_classes
    sizes, alphas = args.sizes, args.alphas

    # ---- 1. train the evidence head on frozen tokens (native + gist looks mixed) ------------------------
    gtr = torch.Generator().manual_seed(args.seed + 10)

    def train_batch(step):
        sc = make_scenes(bank_train, lay, args.batch, sizes, alphas, args.p_null, gtr)
        tl = sample_windows(sc, lay, cand, gtr)
        tn = trunk_tokens(fn, sc["canvas"], tl, lay, dev)
        tg = trunk_tokens(fn, sc["canvas"], None, lay, dev)
        cn = g1.window_token_overlap(sc["overlap"], tl, lay)
        cg = g1.gist_token_overlap(sc["overlap"], lay)
        cover = torch.cat([cn >= 0.5, cg >= 0.5])
        weight = torch.cat([torch.ones_like(cn), ((cg == 0) | (cg >= 0.5)).float()])       # skip ambiguous partial gist tokens
        return torch.cat([tn, tg]), cover, torch.cat([sc["label"], sc["label"]]), weight

    head, log = g1.train_evidence_head(train_batch, 384, C, steps=args.steps, lr=args.lr)
    head.eval()

    # ---- 2. calibration + test splits: scores only (n, T, C), never the tokens or the canvases ----------------------
    # Scenes are generated, looked at and DISCARDED chunk by chunk: one 224-px canvas is 0.6 MB, so keeping 6000 of
    # them (plus the calibration split) would not fit next to the model on a 12 GB Colab.
    def collect(bank, n, seed, n_second=0, seed_second=0, chunk=getattr(args, "chunk_scenes", 200)):
        g, g2 = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed_second)
        keep = ("label", "size_units", "alpha", "top", "left", "overlap")
        parts, done = [], 0
        while done < n:
            m = min(chunk, n - done)
            sc = make_scenes(bank, lay, m, sizes, alphas, args.p_null, g)
            tl = sample_windows(sc, lay, cand, g)
            part = {k: sc[k] for k in keep}
            part["tl"] = tl
            with torch.no_grad():
                part["native"] = head(trunk_tokens(fn, sc["canvas"], tl, lay, dev))
                part["gist"] = head(trunk_tokens(fn, sc["canvas"], None, lay, dev))
                k2 = max(0, min(m, n_second - done))                          # a second, independent window for the first n_second scenes
                if k2:
                    sc2 = {k: v[:k2] for k, v in sc.items()}
                    part["tl2"] = sample_windows(sc2, lay, cand, g2)
                    part["native2"] = head(trunk_tokens(fn, sc2["canvas"], part["tl2"], lay, dev))
            parts.append(part)
            del sc
            done += m
        cat = lambda k: torch.cat([q[k] for q in parts if k in q])
        out = {"sc": {k: cat(k) for k in keep}, "tl": cat("tl"), "native": cat("native"), "gist": cat("gist")}
        if n_second:
            out["tl2"], out["native2"] = cat("tl2"), cat("native2")
        out["cover_native"] = g1.window_token_overlap(out["sc"]["overlap"], out["tl"], lay) > 0
        out["ov_gist"] = g1.gist_token_overlap(out["sc"]["overlap"], lay)
        out["cover_gist"] = out["ov_gist"] > 0
        return out

    n_pair = min(args.n_pairs, args.n_test)
    cal = collect(bank_cal, args.n_cal, args.seed + 20)
    tst = collect(bank_test, args.n_test, args.seed + 30, n_second=n_pair, seed_second=args.seed + 40)
    report = analyse(cal, tst, lay, args.sizes, args.alphas, args.op_size, n_pair)
    report["config"] = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    report["trunk"] = info
    # what ran: the aggregate (gate1_prereg.py) compares these with the frozen file, so a run on other code or other data is a deviation
    report["code_sha256"] = {f: g1.file_sha256(ROOT / f) for f in g1.MEASUREMENT_CODE_FILES}
    report["data_listing"] = None if args.self_test else {f"{s}_path_listing": g1.listing_hash(root / s) for s in ("train", "val")}
    report["head_training_log"] = log
    report["seconds"] = round(time.time() - t0, 1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="synthetic textures, random-init trunk, 56-px windows on a 112-px canvas, CPU")
    ap.add_argument("--ckpt", default=None, help="Gen2FoundationModel checkpoint (ff_112); only read")
    ap.add_argument("--data-root", default="data/imagenet100", help="ImageFolder root with train/ and val/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--canvas", type=int, default=224)
    ap.add_argument("--window", type=int, default=112)
    ap.add_argument("--stride-units", type=int, default=2)
    ap.add_argument("--classes", type=int, default=50, help="number of TARGET classes (the other half of the classes are backgrounds only)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6], help="target side in 14-px units")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.6])
    ap.add_argument("--op-size", type=int, default=4, help="size (units) of the operating point reported in the verdict")
    ap.add_argument("--p-null", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-cal", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--chunk-scenes", type=int, default=200, help="scenes generated (and then discarded) at a time; bounds host memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.self_test:
        args.canvas, args.window, args.classes = 112, 56, 6
        args.sizes, args.alphas, args.op_size = [2, 3], [1.0], 2
        args.steps, args.batch, args.n_cal, args.n_test, args.n_pairs = 60, 24, 400, 600, 300
    torch.set_num_threads(2)
    res = run(args)
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2, default=lambda o: None))
    s = res["summary"]
    print(json.dumps({"summary": s, "verdict_draft": res["verdict_draft"], "cross_look": res["cross_look"], "gist_by_overlap": res["gist_by_overlap"], "seconds": res["seconds"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
diag_dataset_forensics.py — §4 DATASET FORENSICS (programmatic evidence).
================================================================================
Checks, with explicit numbers (never code-reading alone):
  * pinned revision (fingerprint.json vs the production manifest's value)
  * exact train/val sample counts, class counts
  * unique/min/max labels, contiguity, remap-once property
  * train/val class-order identity (the ImageFolder remap hazard)
  * label frequency histogram (min/max class count)
  * duplicate-file detection (size+mtime signature)
  * decode: every 47th image decodes; dims; near-constant fraction
  * tensor stats after the REAL val transform: dtype, range, mean/std,
    NaN/Inf fraction
  * glimpse-plane check: raw image -> foveal_sample at the 4 fixed gaze
    points -> per-crop stats (the plane the backbone actually sees)
"""
from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import (DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,  # noqa: E402
                         PINNED_FINGERPRINT_SHA, REPO_ROOT, save_json)

from PIL import Image  # noqa: E402


def main() -> int:
    ev: dict = {"section": "4_dataset_forensics"}
    import json

    # -- pinned revision -----------------------------------------------------
    fp_path = os.path.join(DATA_ROOT, "fingerprint.json")
    fp = json.load(open(fp_path))
    got = fp.get("aggregate_file_listing_sha256")
    ev["pinned_revision"] = {
        "expected": PINNED_FINGERPRINT_SHA, "found": got,
        "match": got == PINNED_FINGERPRINT_SHA,
        "classes_in_mapping": len(fp.get("class_names_to_dirs", {})),
    }

    # -- counts, labels, mapping identity -------------------------------------
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)),
                             transforms.CenterCrop(IMG_SIZE),
                             transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=tf)
    val = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
    tr_classes, va_classes = train.classes, val.classes
    ev["counts"] = {"train_samples": len(train), "val_samples": len(val),
                    "train_classes": len(tr_classes), "val_classes": len(va_classes)}
    ev["label_space"] = {
        "unique_train_labels": sorted(set(train.targets)),
        "labels_contiguous_0_99": set(train.targets) == set(range(100)),
        "val_labels_equal_train_mapping": va_classes == tr_classes,
    }

    # -- class-order identity in SEQUENCE terms (mapping hazard) --------------
    mism = [(i, a, b) for i, (a, b) in enumerate(zip(tr_classes, va_classes)) if a != b]
    ev["train_val_class_order"] = {
        "identical_order": not mism,
        "mismatches": mism[:10],
        "train_dir_names_sorted_equals": (tr_classes ==
                                          sorted(os.listdir(os.path.join(DATA_ROOT, "train")))),
    }

    # -- label frequency histogram --------------------------------------------
    import collections
    freq = collections.Counter(train.targets)
    counts = [freq[c] for c in range(100)]
    ev["train_label_histogram"] = {
        "min_class_count": min(counts), "max_class_count": max(counts),
        "mean": sum(counts) / 100, "n_classes_below_50": sum(c < 50 for c in counts),
        "histogram_head": [counts[i] for i in range(10)],
    }

    # -- duplicates (size+mtime signature) ------------------------------------
    def dup_scan(split: str):
        seen, dups = {}, 0
        root = os.path.join(DATA_ROOT, split)
        for c in sorted(os.listdir(root)):
            cd = os.path.join(root, c)
            if not os.path.isdir(cd):
                continue
            for fn in os.listdir(cd):
                p = os.path.join(cd, fn)
                if os.path.isfile(p):
                    st = os.stat(p)
                    sig = (st.st_size, int(st.st_mtime))
                    if sig in seen:
                        dups += 1
                    else:
                        seen[sig] = p
        return dups
    ev["duplicates"] = {"train_size_mtime_dups": dup_scan("train"),
                        "val_size_mtime_dups": dup_scan("val")}

    # -- decode + raw dims + near-constant scan (every 47th image) ------------
    random.seed(41)
    n_decode_fail = n_raw = near_const = 0
    raw_dim_hist: dict = {}
    root = os.path.join(DATA_ROOT, "train")
    class_dirs = sorted(os.listdir(root))
    for ci, c in enumerate(class_dirs):
        cd = os.path.join(root, c)
        files = sorted(f for f in os.listdir(cd)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for fi, fn in enumerate(files):
            if (ci * 1000 + fi) % 47 != 0:
                continue
            n_raw += 1
            try:
                im = Image.open(os.path.join(cd, fn)).convert("RGB")
            except Exception:
                n_decode_fail += 1
                continue
            w, h = im.size
            raw_dim_hist[f"{w}x{h}"] = raw_dim_hist.get(f"{w}x{h}", 0) + 1
            g = im.convert("L")
            lo, hi_ = g.getextrema()
            if hi_ - lo <= 2:
                near_const += 1
    ev["decode_scan"] = {"sampled": n_raw, "decode_failures": n_decode_fail,
                         "near_constant_gray": near_const,
                         "near_constant_frac": round(near_const / max(n_raw, 1), 4),
                         "raw_dim_hist_head": dict(sorted(raw_dim_hist.items(),
                                                          key=lambda kv: -kv[1])[:8])}

    # -- tensor stats through the REAL val transform --------------------------
    xs = [val[i][0] for i in range(0, 256, 8)]
    x = torch.stack(xs)
    ev["tensor_stats_val_transform"] = {
        "n": x.shape[0], "dtype": str(x.dtype), "shape": list(x.shape),
        "min": round(x.min().item(), 4), "max": round(x.max().item(), 4),
        "mean": round(x.mean().item(), 4), "std": round(x.std().item(), 4),
        "per_channel_mean": [round(v, 4) for v in x.mean(dim=(0, 2, 3)).tolist()],
        "per_channel_std": [round(v, 4) for v in x.std(dim=(0, 2, 3)).tolist()],
        "nan_frac": float(torch.isnan(x).float().mean().item()),
        "inf_frac": float(torch.isinf(x).float().mean().item()),
    }

    # -- glimpse-plane sanity (§6 hook): the 4 fixed gaze points ---------------
    from noesis_vision.models.foveation import foveal_sample
    gaze = torch.tensor([[-0.6, -0.6], [0.6, -0.6], [-0.6, 0.6], [0.6, 0.6]])
    crops = torch.cat([foveal_sample(x[:8], gaze[i].expand(8, 2),
                                     fovea_size=56) for i in range(4)])
    ev["glimpse_plane"] = {
        "crop_shape": list(crops.shape),
        "min": round(crops.min().item(), 4), "max": round(crops.max().item(), 4),
        "mean": round(crops.mean().item(), 4), "std": round(crops.std().item(), 4),
        "per_gaze_mean": [round(v, 4) for v in crops.view(4, -1).mean(dim=1).tolist()],
        "distinct_gazes_distinct_crops": bool(len({
            round(v, 4) for v in crops.view(4, -1).mean(dim=1).tolist()}) == 4),
    }

    save_json("04_dataset_forensics.json", ev)
    ok = (ev["pinned_revision"]["match"]
          and ev["label_space"]["labels_contiguous_0_99"]
          and ev["train_val_class_order"]["identical_order"]
          and ev["decode_scan"]["decode_failures"] == 0
          and ev["tensor_stats_val_transform"]["nan_frac"] == 0.0)
    print(f"SECTION 4 VERDICT: {'PASS' if ok else 'ATTENTION'} "
          f"(revision_match={ev['pinned_revision']['match']}, "
          f"labels_ok={ev['label_space']['labels_contiguous_0_99']}, "
          f"order_ok={ev['train_val_class_order']['identical_order']}, "
          f"decode_fail={ev['decode_scan']['decode_failures']})")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

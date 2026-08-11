#!/usr/bin/env python3
"""Verify best.pth == rolling.pth for rhan_next_ais_v1_halting_only.

Downloads both artifacts from HF (public repos) into a scratch dir, then
compares:
  1. file-level sha256;
  2. per-tensor state-dict hash (covers the 'model' payload regardless of
     extra metadata keys like epoch/best_acc/optimizer);
  3. recorded metadata (epoch, best_acc, arch) for provenance.

Read-only against HF. Run: python3 scratch/verify_best_vs_rolling.py
"""
import hashlib
import json
import os
import shutil
import sys

import torch

BEST_REPO = "FerrariKazu/rhan-checkpoints"
ROLLING_REPO = "FerrariKazu/rhan-checkpoints-rolling"
NAME = "rhan_next_ais_v1_halting_only"
SCRATCH = "scratch/_ckpt_verify"
BEST = os.path.join(SCRATCH, f"{NAME}_best.pth")
ROLLING = os.path.join(SCRATCH, f"{NAME}_rolling.pth")

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _state_dict_hash(ckpt):
    """Stable hash over the model state dict (sorted keys, per-tensor sha256)."""
    model = ckpt.get("model")
    if model is None:
        for k in ("model_state_dict", "state_dict"):
            if isinstance(ckpt, dict) and k in ckpt:
                model = ckpt[k]
                break
    if model is None:
        raise KeyError("no 'model' state dict found in checkpoint")
    h = hashlib.sha256()
    for k in sorted(model.keys()):
        t = model[k]
        if torch.is_tensor(t):
            h.update(k.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(k.encode())
            h.update(repr(t).encode())
    return h.hexdigest()


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from huggingface_hub import hf_hub_download

    if not os.path.exists(BEST):
        print(f"downloading {NAME}_best.pth from {BEST_REPO} ...", flush=True)
        p = hf_hub_download(repo_id=BEST_REPO, filename=f"{NAME}_best.pth",
                            repo_type="dataset")
        shutil.copy(p, BEST)
    if not os.path.exists(ROLLING):
        print(f"downloading {NAME}_rolling.pth from {ROLLING_REPO} ...", flush=True)
        p = hf_hub_download(repo_id=ROLLING_REPO, filename=f"{NAME}_rolling.pth",
                            repo_type="dataset")
        shutil.copy(p, ROLLING)

    best = torch.load(BEST, map_location="cpu", weights_only=False)
    rolling = torch.load(ROLLING, map_location="cpu", weights_only=False)

    print(f"best.pth    size={os.path.getsize(BEST)/1e6:.1f} MB  "
          f"sha256={_sha256_file(BEST)[:16]}...")
    print(f"rolling.pth size={os.path.getsize(ROLLING)/1e6:.1f} MB  "
          f"sha256={_sha256_file(ROLLING)[:16]}...")
    print(f"file-level sha256 equal: {_sha256_file(BEST) == _sha256_file(ROLLING)}")

    hb, hr = _state_dict_hash(best), _state_dict_hash(rolling)
    print(f"state-dict hash best   = {hb}")
    print(f"state-dict hash rolling= {hr}")
    print(f"STATE-DICT EQUAL: {hb == hr}")

    def meta(d, name):
        print(f"  {name}: keys={sorted(d.keys())}")
        for k in ("epoch", "best_acc", "arch"):
            if k in d:
                v = d[k]
                print(f"    {k} = {v:.2f}" if isinstance(v, float) else
                      f"    {k} = {v}")

    meta(best, "best")
    meta(rolling, "rolling")

    result = {
        "name": NAME,
        "file_sha256_equal": _sha256_file(BEST) == _sha256_file(ROLLING),
        "state_dict_equal": hb == hr,
        "best_sha256": _sha256_file(BEST),
        "rolling_sha256": _sha256_file(ROLLING),
        "best_state_hash": hb,
        "rolling_state_hash": hr,
        "best_epoch": best.get("epoch"),
        "rolling_epoch": rolling.get("epoch"),
        "best_best_acc": best.get("best_acc"),
        "rolling_best_acc": rolling.get("best_acc"),
        "arch_best": best.get("arch"),
        "arch_rolling": rolling.get("arch"),
        "config_match": best.get("config") == rolling.get("config"),
    }
    out = os.path.join(SCRATCH, "comparison.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    print(f"\nWrote {out}")
    print(json.dumps({k: (str(v)[:80] if not isinstance(v, bool) else v)
                      for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    sys.exit(main())

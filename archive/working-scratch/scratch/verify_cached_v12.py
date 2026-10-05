#!/usr/bin/env python3
"""Verify a v12 rolling checkpoint from the local HF blob cache is restorable
(loads, has epoch/optimizer keys, and loads into the RHANv12 architecture)."""
import os
import sys
import json
import torch

BLOB = ("/home/ferrarikazu/.cache/huggingface/hub/"
        "datasets--FerrariKazu--rhan-checkpoints-rolling/blobs/"
        "f8c3b9e7ed8dde0076dfd03ee7298cf14060f26ea6b2580408b7433a441beb38")

out = {}
try:
    d = torch.load(BLOB, map_location="cpu", weights_only=False)
    out["epoch"] = d.get("epoch")
    out["best_acc"] = d.get("best_acc")
    out["keys"] = sorted(d.keys())
    out["has_optimizer"] = "optimizer" in d
    out["has_scheduler"] = "scheduler" in d
    out["has_scaler"] = "scaler" in d

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from phase1_training.model_rhan_v12 import RHANv12

    m = RHANv12()
    missing, unexpected = m.load_state_dict(d["model"], strict=False)
    out["missing"] = len(missing)
    out["unexpected"] = len(unexpected)
    out["restorable"] = "OK"
except Exception as e:
    out["error"] = str(e)[:500]
    import traceback
    out["traceback"] = traceback.format_exc(limit=3)

with open("/tmp/v12_restore_check.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

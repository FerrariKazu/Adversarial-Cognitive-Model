#!/usr/bin/env python3
"""
T4c: Learning curve slope, epochs 10-15, flagship vs ff_112.
Source: gen2_epoch_log.jsonl from HF rolling repo.
Output: evaluation/stage0_results/t4c_learning_curve.json
"""
import os, json
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

token = os.environ.get("HF_TOKEN")
OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect all epoch logs from both HF repos
all_entries = []
for repo_id in ["FerrariKazu/rhan-nxa-g2-checkpoints", "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"]:
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        for fn in files:
            if fn.endswith(".jsonl"):
                p = hf_hub_download(repo_id=repo_id, filename=fn,
                                    repo_type="dataset", token=token)
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            try:
                                e = json.loads(line)
                                if any(k in ["val_acc_clean", "val_acc_robust"] for k in e):
                                    all_entries.append(e)
                            except Exception:
                                pass
    except Exception as e:
        print(f"Warning {repo_id}: {e}")

# De-duplicate by (phase, epoch)
seen = set()
entries = []
for e in all_entries:
    key = (e.get("phase", ""), e.get("epoch", -1))
    if key not in seen and key[0] and key[1] >= 0:
        seen.add(key)
        entries.append(e)

entries.sort(key=lambda x: (x.get("phase",""), x.get("epoch", 0)))

# Build per-arm epoch series
arms = {}
for e in entries:
    phase = e.get("phase", "")
    if phase not in ("consolidated_112", "ff_112"):
        continue
    arm = arms.setdefault(phase, [])
    arm.append({
        "epoch": e.get("epoch"),
        "val_acc_clean": e.get("val_acc_clean"),
        "val_acc_robust": e.get("val_acc_robust"),
    })

# Compute slope for epochs 10-15
def slope(series, key, e_min=10, e_max=15):
    pts = [(p["epoch"], p[key]) for p in series
           if p.get("epoch") and e_min <= p["epoch"] <= e_max and p.get(key) is not None]
    if len(pts) < 2:
        return None
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    n = len(xs)
    xm, ym = sum(xs)/n, sum(ys)/n
    num = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
    den = sum((x-xm)**2 for x in xs)
    return num / den if den > 0 else 0.0

result = {"protocol_id": "consolidated-flagship-112/1", "arms": {}}
for phase, series in arms.items():
    s_clean = slope(series, "val_acc_clean")
    s_rob = slope(series, "val_acc_robust")
    result["arms"][phase] = {
        "slope_clean_ep10_15_per_epoch": s_clean,
        "slope_robust_ep10_15_per_epoch": s_rob,
        "n_epochs": len(series),
        "series": sorted(series, key=lambda x: x["epoch"] or 0),
    }
    print(f"{phase}: slope_clean={s_clean:.4f}/ep  slope_robust={s_rob:.4f}/ep")

out = OUT_DIR / "t4c_learning_curve.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out}")

#!/usr/bin/env python3
"""Gen0 evidence pull: aggregate every sweep CSV on HF + roadmap state."""
import os, io, csv, json, statistics as st, sys
from collections import defaultdict

def load_token():
    for line in open(".env"):
        line = line.strip()
        if line.startswith("HF_TOKEN"):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            return v
    return None

TOK = load_token()
if not TOK:
    sys.exit("no HF_TOKEN in .env")
os.environ["HF_TOKEN"] = TOK

from huggingface_hub import HfApi
api = HfApi(token=TOK)

SWEEP_REPO = "FerrariKazu/rhan-eval-sweep"
OUT = {}

# ---------- 1. discover every sweep subdir ----------
files = api.list_repo_files(SWEEP_REPO, repo_type="dataset")
per_seed = sorted({f.split("/")[0] for f in files
                   if f.endswith("epsilon_sweep_per_seed.csv")})
print(f"[discover] {len(per_seed)} sweep dirs with per-seed CSVs:")
for d in per_seed:
    print("   ", d)

# ---------- 2. download + aggregate each ----------
from huggingface_hub import hf_hub_download

def agg(rows):
    """rows: list of dicts with keys like ckpt_label, eps, acc, dprime"""
    cells = defaultdict(list)
    dcells = defaultdict(list)
    for r in rows:
        lab = r.get("ckpt_label") or r.get("label") or r.get("model") or "UNKNOWN"
        if not lab:
            continue
        eps = r.get("eps_pixel")
        acc = r.get("acc_pct")
        dp = r.get("macro_dprime")
        if eps in (None, "") or acc in (None, ""):
            continue
        key = (lab, float(eps))
        cells[key].append(float(acc))
        if dp not in (None, ""):
            dcells[key].append(float(dp))
    out = {}
    for key, accs in sorted(cells.items()):
        n = len(accs)
        dps = dcells.get(key, [])
        out[f"{key[0]} @ eps={key[1]}"] = {
            "n_seeds": n,
            "acc_mean": round(st.mean(accs), 2),
            "acc_std": round(st.stdev(accs), 2) if n > 1 else None,
            "dp_mean": round(st.mean(dps), 4) if dps else None,
            "dp_std": round(st.stdev(dps), 4) if len(dps) > 1 else None,
        }
    return out

all_sweeps = {}
for d in per_seed:
    try:
        p = hf_hub_download(SWEEP_REPO, f"{d}/epsilon_sweep_per_seed.csv",
                            repo_type="dataset", token=TOK)
        rows = list(csv.DictReader(open(p)))
        all_sweeps[d] = {"n_rows": len(rows), "agg": agg(rows),
                         "columns": list(rows[0].keys()) if rows else []}
    except Exception as e:
        all_sweeps[d] = {"error": str(e)}

OUT["sweeps"] = all_sweeps

# ---------- 3. roadmap state ----------
for rid, fn in [("FerrariKazu/rhan-checkpoints-rolling", "rhan_next_roadmap.json"),
                ("FerrariKazu/rhan-checkpoints", "rhan_next_roadmap.json"),
                ("FerrariKazu/rhan-checkpoints-rolling", "roadmap.json")]:
    try:
        p = hf_hub_download(rid, fn, repo_type="dataset", token=TOK)
        OUT["roadmap_repo"] = rid
        OUT["roadmap"] = json.load(open(p))
        break
    except Exception:
        continue

with open("report/_gen0_pull.json", "w") as f:
    json.dump(OUT, f, indent=1, default=str)

# ---------- 4. print compact digest ----------
print("\n================ ROADMAP ================")
rm = OUT.get("roadmap", {})
stages = rm.get("rhan_nx", {}).get("stages", rm.get("stages", {}))
for name, s in stages.items():
    vd = s.get("verdict") or {}
    passed = vd.get("passed", "n/a")
    print(f"  {name:12s} {s.get('status','?'):14s} gate={passed}  ckpt={s.get('ckpt','?')}")

print("\n================ SWEEP AGGREGATES ================")
for d, info in all_sweeps.items():
    print(f"\n--- {d} (rows={info.get('n_rows')}) ---")
    if "error" in info:
        print("   ERROR:", info["error"]); continue
    for k, v in info["agg"].items():
        print(f"   {k:45s} n={v['n_seeds']:2d}  acc={v['acc_mean']}±{v['acc_std']}  d'={v['dp_mean']}±{v['dp_std']}")

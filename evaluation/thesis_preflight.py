#!/usr/bin/env python3
"""
Thesis pre-flight checks for ff_56 result.

Check 1: ff_56 clean on ALL 5,000 val images (both final and best_robust ckpt).
          Confirms the epoch-15 checkpoint identity and diagnoses the 7-pp harness gap.

Check 2: APGD-DLR on ff_112 seeds 17, 42, 99.
          Makes the attack set like-for-like with ff_56.

Check 3: Full per-sample re-evaluation of ff_56, recurrence_only, C1, ff_112-s17.
          Enables proper McNemar tests ff_56 vs every other arm.

Writes: report/audit_rev1/thesis_preflight.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchattacks
from scipy.stats import chi2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS    = 5.417 / 255.0
SEED   = 42
VAL    = "data/imagenet100/val"
OUT    = "report/audit_rev1/thesis_preflight.json"
print(f"Device: {DEVICE}  EPS={EPS*255:.4f}/255")

# ── shared helpers ─────────────────────────────────────────────────────────────
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from gen2_foundation.model import Gen2FoundationModel

class NormalizedModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        m = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
        s = torch.tensor(IMAGENET_STD).view(1,3,1,1)
        self.register_buffer("mean", m)
        self.register_buffer("std",  s)
    def forward(self, x):
        return self.base((x - self.mean) / self.std)

def load_model(phase, fovea, ckpt_path):
    base = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=fovea).to(DEVICE)
    sd   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    base.load_state_dict(sd.get("model", sd), strict=False)
    base.eval()
    m = NormalizedModel(base).to(DEVICE)
    m.eval()
    meta = {k: v for k, v in sd.items() if k not in ("model","optimizer","scheduler")}
    return m, meta

def make_loader(img_size, indices=None, batch=128):
    # CANONICAL: matches build_raw_loader in Kaggle_ATTACK_RECONCILIATION.py
    # which uses Resize((img_size, img_size)) — NOT Resize(64)+CenterCrop(56).
    # The 7-pp gap in eval_ff56_reconciled.py was likely due to this mismatch.
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    ds = ImageFolder(VAL, transform=tf)
    if indices is not None:
        ds = Subset(ds, indices)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

def eval_clean(model, loader):
    corr = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            corr.extend((model(x).argmax(-1) == y).cpu().numpy())
    return np.array(corr, dtype=np.int8)

def pgd50(model, x, y):
    alpha = EPS * 2.5 / 50
    x_adv = (x + torch.zeros_like(x).uniform_(-EPS, EPS)).clamp(0,1)
    bl = torch.full((len(x),), -1e9, device=DEVICE)
    ba = x_adv.clone()
    for _ in range(50):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y, reduction="none")
        g = torch.autograd.grad(loss.sum(), x_adv)[0]
        with torch.no_grad():
            x_adv = (x_adv + alpha * g.sign()).clamp(x-EPS, x+EPS).clamp(0,1)
        mask = loss.detach() > bl
        bl[mask] = loss.detach()[mask]
        ba[mask] = x_adv[mask]
    return ba

def eval_attacks(model, loader, attacks):
    """attacks: dict name->callable.  Returns dict name->int8 array."""
    results = {}
    for atk_name, atk_fn in attacks.items():
        corr, t0 = [], time.time()
        print(f"    {atk_name:<14}", end="", flush=True)
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if atk_fn is None:
                x_adv = x
            else:
                x_adv = atk_fn(x, y)
            with torch.no_grad():
                corr.extend((model(x_adv).argmax(-1) == y).cpu().numpy())
        results[atk_name] = np.array(corr, dtype=np.int8)
        acc = results[atk_name].mean()*100
        print(f"  {acc:.2f}%  ({time.time()-t0:.1f}s)")
    return results

def mcnemar(a, b):
    """a, b: int8 arrays of correct/wrong.  Returns dict with stat/p/b/c."""
    disc_b = ((a == 1) & (b == 0)).sum()   # a right, b wrong
    disc_c = ((a == 0) & (b == 1)).sum()   # a wrong, b right
    n = disc_b + disc_c
    if n == 0:
        return {"b": 0, "c": 0, "stat": 0.0, "p": 1.0}
    stat = (abs(disc_b - disc_c) - 1)**2 / n   # with continuity correction
    from scipy.stats import chi2
    p = 1 - chi2.cdf(stat, df=1)
    return {"b": int(disc_b), "c": int(disc_c), "stat": float(stat), "p": float(p)}

# ── fixed 1000-image indices (same as all prior evaluations) ──────────────────
rng1000 = np.random.default_rng(SEED)
full_ds  = ImageFolder(VAL, transform=T.ToTensor())
idx1000  = rng1000.choice(len(full_ds), size=1000, replace=False)
all_idx  = np.arange(len(full_ds))  # 5000 images

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: ff_56 clean on ALL 5,000 val images + checkpoint identity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 1: ff_56 clean accuracy — full 5,000-image val set")
print("="*70)

ckpts_to_check = [
    ("ff_56_final",       "checkpoints/foundation_ff_56_final.pth"),
    ("ff_56_best_robust", "checkpoints/foundation_ff_56_best_robust.pth"),
    ("ff_56_best_clean",  "checkpoints/foundation_ff_56_best_clean.pth"),
]

check1 = {}
for label, ckpt in ckpts_to_check:
    if not os.path.exists(ckpt):
        print(f"  SKIP {label}: not found")
        continue
    model, meta = load_model("ff_56", 56, ckpt)
    print(f"\n  [{label}]  epoch={meta.get('epoch','?')}  "
          f"val_clean_acc={meta.get('val_clean_acc','?')}  "
          f"val_rob_acc={meta.get('val_rob_acc','?')}")

    # Full 5,000 images
    loader5k = make_loader(56, indices=None, batch=256)
    t0 = time.time()
    corr5k = eval_clean(model, loader5k)
    acc5k = corr5k.mean() * 100
    print(f"  Full 5k clean: {acc5k:.2f}%  ({time.time()-t0:.1f}s)")

    # Seed-42 N=1000 slice (reproducibility check)
    loader1k = make_loader(56, indices=idx1000, batch=256)
    corr1k = eval_clean(model, loader1k)
    acc1k = corr1k.mean() * 100
    print(f"  Seed-42 1k clean: {acc1k:.2f}%")

    check1[label] = {
        "epoch": meta.get("epoch"),
        "ckpt_val_clean_acc": meta.get("val_clean_acc"),
        "ckpt_val_rob_acc":   meta.get("val_rob_acc"),
        "full5k_clean_acc":   float(acc5k),
        "seed42_1k_clean_acc": float(acc1k),
    }

results["check1_ff56_clean"] = check1

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: APGD-DLR on ff_112 seeds 17, 42, 99 (like-for-like completion)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 2: APGD-DLR on ff_112 seeds 17/42/99")
print("="*70)

FF112_CKPTS = {
    "seed17": "checkpoints/foundation_ff_112_final.pth",
    "seed42": "checkpoints/foundation_ff_112_seed42_final.pth",
    "seed99": "checkpoints/foundation_ff_112_seed99_final.pth",
}
loader1k_112 = make_loader(112, indices=idx1000, batch=64)

check2 = {}
for seed_label, ckpt in FF112_CKPTS.items():
    if not os.path.exists(ckpt):
        # Try alternate naming
        alt = ckpt.replace(f"_{seed_label}_final", f"_final_{seed_label}")
        alt2 = f"checkpoints/foundation_ff_112_{seed_label[-2:]}_final.pth"
        for candidate in [alt, alt2]:
            if os.path.exists(candidate):
                ckpt = candidate
                break
        else:
            print(f"  SKIP {seed_label}: checkpoint not found (tried {ckpt})")
            check2[seed_label] = {"error": "checkpoint not found"}
            continue
    model, meta = load_model("ff_112", 112, ckpt)
    print(f"\n  ff_112 {seed_label}  epoch={meta.get('epoch','?')}")
    apgd_dlr = torchattacks.APGD(model, norm="Linf", eps=EPS, steps=100,
                                  n_restarts=1, loss="dlr", verbose=False)
    r = eval_attacks(model, loader1k_112, {"apgd_dlr": lambda x,y,a=apgd_dlr: a(x,y)})
    check2[seed_label] = {
        "apgd_dlr_acc": float(r["apgd_dlr"].mean()),
        "outcomes_apgd_dlr": r["apgd_dlr"].tolist(),
    }

results["check2_ff112_dlr"] = check2

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Full per-sample re-eval for McNemar (ff_56, rec_only, C1, ff_112-s17)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CHECK 3: Per-sample full re-eval for McNemar pairs")
print("="*70)

ARMS = [
    ("ff_56",           "ff_56",          56,  "checkpoints/foundation_ff_56_final.pth"),
    ("recurrence_only", "recurrence_only", 56,  "checkpoints/foundation_recurrence_only_final.pth"),
    ("c1",              "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_cont_final.pth"),
    ("ff_112_s17",      "ff_112",          112, "checkpoints/foundation_ff_112_final.pth"),
]

# (already set above)

arm_outcomes = {}   # arm_label -> {"clean":int8, "pgd50":int8, "apgd_ce":int8, "apgd_dlr":int8, "worst":int8}

for arm_label, phase, fovea, ckpt in ARMS:
    if ckpt is None or not os.path.exists(ckpt):
        print(f"  SKIP {arm_label}: checkpoint not found")
        continue
    print(f"\n  Evaluating {arm_label} ({fovea}px)  ckpt={ckpt}")
    model, meta = load_model(phase, fovea, ckpt)
    loader = make_loader(fovea, indices=idx1000, batch=(64 if fovea==112 else 128))
    apgd_ce  = torchattacks.APGD(model, norm="Linf", eps=EPS, steps=100,
                                  n_restarts=1, loss="ce",  verbose=False)
    apgd_dlr = torchattacks.APGD(model, norm="Linf", eps=EPS, steps=100,
                                  n_restarts=1, loss="dlr", verbose=False)
    r = eval_attacks(model, loader, {
        "clean":    None,
        "pgd50":    lambda x, y, m=model: pgd50(m, x, y),
        "apgd_ce":  lambda x, y, a=apgd_ce:  a(x, y),
        "apgd_dlr": lambda x, y, a=apgd_dlr: a(x, y),
    })
    worst = np.bitwise_and.reduce([r[k] for k in ["pgd50","apgd_ce","apgd_dlr"]])
    r["worst"] = worst
    print(f"    worst_case: {worst.mean()*100:.2f}%")
    arm_outcomes[arm_label] = {k: v.tolist() for k,v in r.items()}

# ── McNemar table ──────────────────────────────────────────────────────────────
print("\n  McNemar tests:")
pairs = [
    ("ff_56",      "recurrence_only", "worst"),
    ("ff_56",      "c1",              "worst"),
    ("ff_56",      "ff_112_s17",      "worst"),
    ("ff_56",      "recurrence_only", "clean"),
    ("ff_56",      "c1",              "clean"),
    ("recurrence_only", "c1",         "worst"),
]

mcnemar_results = {}
for a_lbl, b_lbl, metric in pairs:
    if a_lbl not in arm_outcomes or b_lbl not in arm_outcomes:
        print(f"  SKIP {a_lbl} vs {b_lbl} ({metric}): outcomes not available")
        continue
    a_arr = np.array(arm_outcomes[a_lbl][metric])
    b_arr = np.array(arm_outcomes[b_lbl][metric])
    mn = mcnemar(a_arr, b_arr)
    diff = (a_arr.mean() - b_arr.mean()) * 100
    key = f"{a_lbl}_vs_{b_lbl}_{metric}"
    mcnemar_results[key] = {
        "acc_a": float(a_arr.mean()*100),
        "acc_b": float(b_arr.mean()*100),
        "diff_pp": round(diff, 2),
        **mn,
    }
    dir_str = ">" if diff > 0 else "<"
    sig = "***" if mn["p"] < 0.001 else ("**" if mn["p"] < 0.01 else ("*" if mn["p"] < 0.05 else "ns"))
    print(f"  {a_lbl} vs {b_lbl} ({metric}): "
          f"{a_arr.mean()*100:.1f}% {dir_str} {b_arr.mean()*100:.1f}%  "
          f"diff={diff:+.1f}pp  b={mn['b']} c={mn['c']}  p={mn['p']:.4f} {sig}")

results["check3_per_sample"] = {
    "arm_summary": {lbl: {k: float(np.array(v).mean()*100) for k,v in outs.items()}
                    for lbl, outs in arm_outcomes.items()},
    "mcnemar": mcnemar_results,
    "arm_outcomes": arm_outcomes,
}

# ── Save ───────────────────────────────────────────────────────────────────────
import datetime
results["timestamp"] = datetime.datetime.now().isoformat()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {OUT}")

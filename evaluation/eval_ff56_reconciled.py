#!/usr/bin/env python3
"""
Evaluate the retrained ff_56 checkpoint on the reconciled N=1000 harness.
eps = 5.417/255, seed=42 — same slice used for all other arms.

Attacks: PGD-50, APGD-CE, APGD-DLR → worst-case = bitwise-AND.
Writes results to report/audit_rev1/ff56_reconciled_eval.json.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPS      = 5.417 / 255.0
STEPS_PGD = 50
N        = 1000
SEED     = 42
CKPT     = "checkpoints/foundation_ff_56_final.pth"
# ImageNet-100 val — same dataset used by all Kaggle eval scripts
IMAGENET = os.environ.get("IMAGENET_VAL", "data/imagenet100/val")
OUT      = "report/audit_rev1/ff56_reconciled_eval.json"

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT}")
print(f"eps = {EPS*255:.4f}/255  N={N}  seed={SEED}")

# ── Model ──────────────────────────────────────────────────────────────────────
from gen2_foundation.model import Gen2FoundationModel

class NormalizedModel(torch.nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    def __init__(self, base):
        super().__init__()
        self.base = base
        m = torch.tensor(self.MEAN).view(1, 3, 1, 1)
        s = torch.tensor(self.STD).view(1, 3, 1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)
    def forward(self, x):
        return self.base((x - self.mean) / self.std)

base = Gen2FoundationModel(phase="ff_56", arm="v1", fovea_size=56).to(DEVICE)
sd   = torch.load(CKPT, map_location=DEVICE, weights_only=False)
base.load_state_dict(sd.get("model", sd), strict=False)
base.eval()
model = NormalizedModel(base).to(DEVICE)
model.eval()

print(f"Loaded: clean val acc from ckpt = {sd.get('val_clean_acc', '?'):.4f}")

# ── Dataset ────────────────────────────────────────────────────────────────────
# Match the build_raw_loader in CONTROLS: Resize(64), CenterCrop(56)
tf = T.Compose([T.Resize(64), T.CenterCrop(56), T.ToTensor()])
ds = ImageFolder(IMAGENET, transform=tf)
rng = np.random.default_rng(SEED)
idx = rng.choice(len(ds), size=N, replace=False)
loader = DataLoader(Subset(ds, idx), batch_size=100, shuffle=False,
                    num_workers=4, pin_memory=True)

# ── PGD-50 helper (pixel space) ────────────────────────────────────────────────
def pgd50(m, x, y):
    alpha   = EPS * 2.5 / STEPS_PGD
    x_adv   = x + torch.zeros_like(x).uniform_(-EPS, EPS)
    x_adv   = x_adv.clamp(0, 1)
    best_loss = torch.full((len(x),), -1e9, device=DEVICE)
    best_adv  = x_adv.clone()
    for _ in range(STEPS_PGD):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(m(x_adv), y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), x_adv)[0]
        with torch.no_grad():
            x_adv = (x_adv + alpha * grad.sign()).clamp(x - EPS, x + EPS).clamp(0, 1)
        mask = loss.detach() > best_loss
        best_loss[mask] = loss.detach()[mask]
        best_adv[mask]  = x_adv[mask]
    return best_adv

# ── Evaluation ─────────────────────────────────────────────────────────────────
try:
    import torchattacks
    apgd_ce  = torchattacks.APGD(model, norm="Linf", eps=EPS, steps=100,
                                  n_restarts=1, loss="ce",  verbose=False)
    apgd_dlr = torchattacks.APGD(model, norm="Linf", eps=EPS, steps=100,
                                  n_restarts=1, loss="dlr", verbose=False)
    HAS_TA = True
except ImportError:
    print("torchattacks not found — skipping APGD; will use PGD only")
    HAS_TA = False

outcomes = {}
for atk in (["clean", "pgd50"] + (["apgd_ce", "apgd_dlr"] if HAS_TA else [])):
    preds_all, labels_all = [], []
    t0 = time.time()
    print(f"  {atk:<14}", end="", flush=True)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad() if atk == "clean" else torch.enable_grad():
            if atk == "clean":
                x_adv = x
            elif atk == "pgd50":
                x_adv = pgd50(model, x, y)
            elif atk == "apgd_ce":
                x_adv = apgd_ce(x, y)
            elif atk == "apgd_dlr":
                x_adv = apgd_dlr(x, y)
        with torch.no_grad():
            preds = model(x_adv).argmax(-1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    corr = np.array(preds_all) == np.array(labels_all)
    acc  = corr.mean() * 100
    outcomes[atk] = corr.astype(np.int8).tolist()
    print(f"  acc={acc:.2f}%  ({time.time()-t0:.1f}s)")

adv_keys = [k for k in outcomes if k != "clean"]
worst = np.bitwise_and.reduce([np.array(outcomes[k]) for k in adv_keys])
print(f"  {'worst_case':<14}  acc={worst.mean()*100:.2f}%")

result = {
    "checkpoint": CKPT,
    "phase": "ff_56",
    "eval_n": N,
    "eval_seed": SEED,
    "eps_pixel_255": round(EPS * 255, 4),
    "clean_acc":       float(np.array(outcomes["clean"]).mean()),
    "pgd50_acc":       float(np.array(outcomes["pgd50"]).mean()),
    "apgd_ce_acc":     float(np.array(outcomes.get("apgd_ce", [])).mean()) if HAS_TA else None,
    "apgd_dlr_acc":    float(np.array(outcomes.get("apgd_dlr", [])).mean()) if HAS_TA else None,
    "worst_case_acc":  float(worst.mean()),
    "outcomes":        {k: v for k, v in outcomes.items()},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {OUT}")

# ── Decision Rule ──────────────────────────────────────────────────────────────
print("\n[Decision Rule (preregistered)]")
rec_worst   = 0.238   # recurrence_only from paired analysis
c1_worst    = 0.225   # g2_gist_only_cont from paired analysis
ff56_worst  = worst.mean()
best_ctrl   = max(c1_worst, ff56_worst)
gap_pp      = (rec_worst - best_ctrl) * 100.0
sigma_train = 0.5508  # ff_112 inter-seed σ for worst-case (from seed variance JSON)
two_sigma   = 2 * sigma_train
print(f"  recurrence_only worst:     {rec_worst*100:.2f}%")
print(f"  C1 (g2_gist_only_cont):    {c1_worst*100:.2f}%")
print(f"  ff_56 (true, retrained):   {ff56_worst*100:.2f}%")
print(f"  best control worst:        {best_ctrl*100:.2f}%")
print(f"  gap = {gap_pp:+.2f} pp  vs  2σ threshold = {two_sigma:.2f} pp")
if gap_pp > two_sigma:
    print("  → CLAIM PASSES (gap > 2σ)")
else:
    print("  → CLAIM FAILS or inconclusive (gap ≤ 2σ)")

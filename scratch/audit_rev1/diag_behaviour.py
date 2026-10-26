#!/usr/bin/env python3
"""RHAN-NXA Revision-1 audit diagnostics #2 (READ-ONLY, exploratory).

Real-checkpoint behavioural probes on 8 val images:
  * determinism at eval (stochastic gaze?)
  * predictor bypass / evidential bypass interventions
  * belief-update magnitude relative to the state
  * which parameters actually receive gradient
  * learned precision range, refinement LayerScale gamma
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

OUT = {}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load(phase, path, arm="v1"):
    m = Gen2FoundationModel(phase=phase, arm=arm).to(DEV)
    sd = torch.load(path, map_location=DEV, weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def batch(n=8):
    tf = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(os.path.join(REPO_ROOT, "data/imagenet100/val"), transform=tf)
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    x, y = next(iter(DataLoader(Subset(ds, idx), batch_size=n)))
    return x.to(DEV), y.to(DEV)


def main():
    x, y = batch()
    print(f"device={DEV} batch={tuple(x.shape)}")

    models = {
        "recurrence_only": load("recurrence_only",
                                "checkpoints/foundation_recurrence_only_final.pth"),
        "gen1_core": load("gen1_core",
                          "checkpoints/foundation_gen1_core_best_clean.pth"),
    }

    # ------------------------------------------------------------------
    hr("A. Determinism at eval (two identical forward passes)")
    for name, m in models.items():
        with torch.no_grad():
            a = m(x)
            b = m(x)
        same = torch.equal(a, b)
        mx = (a - b).abs().max().item()
        print(f"  {name}: bitwise identical={same}  max|Δlogit|={mx:.3e}")
        OUT.setdefault("determinism", {})[name] = {"identical": bool(same),
                                                   "max_abs_delta": mx}

    # ------------------------------------------------------------------
    hr("B. Predictor bypass (remove prediction error from the update)")
    for name, m in models.items():
        if not getattr(m, "belief_dynamics", False):
            print(f"  {name}: no belief dynamics — skipped")
            continue
        with torch.no_grad():
            base = m(x)
        orig = m.predictor.predict_features

        def zero_pred(belief, gaze, _orig=orig):
            p = _orig(belief, gaze)
            return torch.zeros_like(p)

        m.predictor.predict_features = zero_pred
        try:
            with torch.no_grad():
                ablated = m(x)
        finally:
            m.predictor.predict_features = orig
        d = (base - ablated).abs().max().item()
        flips = int((base.argmax(-1) != ablated.argmax(-1)).sum())
        print(f"  {name}: max|Δlogit|={d:.4f}  argmax flips={flips}/{len(y)}")
        OUT.setdefault("predictor_bypass", {})[name] = {
            "max_abs_delta": d, "argmax_flips": flips, "n": int(len(y))}

    # ------------------------------------------------------------------
    hr("C. Evidential-evidence bypass (zero the Dirichlet evidence)")
    for name, m in models.items():
        if not getattr(m, "carry_belief", False):
            print(f"  {name}: no evidential head — skipped")
            continue
        from noesis_vision.uncertainty.evidential_head import DirichletParams
        with torch.no_grad():
            base = m(x)
        orig = m.evidential_head.forward

        def const_ev(features, _orig=orig):
            dp = _orig(features)
            return DirichletParams(evidence=torch.full_like(dp.evidence, 1.0))

        m.evidential_head.forward = const_ev
        try:
            with torch.no_grad():
                ablated = m(x)
        finally:
            m.evidential_head.forward = orig
        d = (base - ablated).abs().max().item()
        flips = int((base.argmax(-1) != ablated.argmax(-1)).sum())
        print(f"  {name}: max|Δlogit|={d:.4f}  argmax flips={flips}/{len(y)}")
        OUT.setdefault("evidential_bypass", {})[name] = {
            "max_abs_delta": d, "argmax_flips": flips, "n": int(len(y))}

    # ------------------------------------------------------------------
    hr("D. Belief-update magnitude vs state magnitude (belief phases)")
    m = models["gen1_core"]
    z = None
    from types import SimpleNamespace
    from noesis_vision.models.foveation import foveal_sample

    with torch.no_grad():
        B = x.shape[0]
        gist = m.gist_encoder(x)
        z = gist.mean(dim=1)
        d = m.evidential_head(z)
        sched = [(0.0, 0.0), (0.3, 0.3), (-0.3, 0.3), (0.0, -0.3)]
        rows = []
        for t, (gx, gy) in enumerate(sched):
            gaze = torch.tensor([[gx, gy]], device=DEV).expand(B, -1)
            crop = foveal_sample(x, gaze, fovea_size=m.fovea_size)
            pooled, toks = m.backbone(crop)
            prec = m.precision(d).unsqueeze(-1)
            belief = SimpleNamespace(z=z, uncertainty=d.uncertainty, evidence=d.evidence)
            pf = m.predictor.predict_features(belief, gaze)
            err = pf - toks
            dz = m.update_net(z, err)
            step = prec * dz
            rows.append({
                "t": t,
                "z_norm": float(z.norm(dim=-1).mean()),
                "update_norm": float(step.norm(dim=-1).mean()),
                "rel_update": float((step.norm(dim=-1) / z.norm(dim=-1).clamp_min(1e-9)).mean()),
                "err_norm": float(err.norm(dim=-1).mean()),
                "prec_mean": float(prec.mean()),
            })
            z = z + step
            d = m.evidential_head(z)
        for r in rows:
            print(f"  t={r['t']}  ||z||={r['z_norm']:.2f}  ||Δz||={r['update_norm']:.4f}  "
                  f"rel={r['rel_update']*100:.3f}%  ||err||={r['err_norm']:.3f}  "
                  f"prec={r['prec_mean']:.4f}")
        OUT["update_magnitude"] = rows

    # ------------------------------------------------------------------
    hr("E. Learned precision range (batch of 32)")
    x32, y32 = batch(32)
    for name in ("gen1_core",):
        mm = models[name]
        with torch.no_grad():
            gist = mm.gist_encoder(x32)
            z0 = gist.mean(dim=1)
            dd = mm.evidential_head(z0)
            p = mm.precision(dd)
            u = dd.uncertainty
        print(f"  {name}: precision min={p.min():.4f} mean={p.mean():.4f} "
              f"max={p.max():.4f}  (registry claims clamp [1e-4,1])")
        print(f"  {name}: uncertainty C/sum(alpha) min={u.min():.5f} "
              f"mean={u.mean():.5f} max={u.max():.5f}")
        OUT["precision"] = {
            "min": float(p.min()), "mean": float(p.mean()), "max": float(p.max()),
            "u_min": float(u.min()), "u_mean": float(u.mean()), "u_max": float(u.max()),
            "exceeds_1": bool(p.max() > 1.0),
        }

    # ------------------------------------------------------------------
    hr("F. Refinement LayerScale gamma (mean |gamma|) from checkpoints")
    for name, path in [
        ("recurrence_only", "checkpoints/foundation_recurrence_only_final.pth"),
        ("gen1_core", "checkpoints/foundation_gen1_core_best_clean.pth"),
    ]:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd["model"] if "model" in sd else sd
        for k in sd:
            if k.startswith("backbone.refinement") and k.endswith("gamma"):
                t = sd[k].float()
                print(f"  {name} {k}: mean|γ|={t.abs().mean():.3e} "
                      f"max|γ|={t.abs().max():.3e} std={t.std():.3e}")
        trunk = [sd[k].float().abs().mean().item() for k in sd
                 if k.startswith("backbone.blocks") and k.endswith("gamma")]
        print(f"  {name}: trunk LayerScale mean|γ| across 24 tensors "
              f"= {sum(trunk)/len(trunk):.3e}")
        OUT.setdefault("gamma", {})[name] = {"trunk_mean_abs": sum(trunk)/len(trunk)}

    # ------------------------------------------------------------------
    hr("G. Gradient reachability per parameter group (one CE backward)")
    for name in ("gen1_core",):
        mm = models[name]
        mm.zero_grad(set_to_none=True)
        out = mm(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        covered = {}
        for pname, p in mm.named_parameters():
            grp = pname.split(".")[1] if pname.startswith("backbone.") else pname.split(".")[0]
            g = 0.0 if p.grad is None else float(p.grad.norm())
            covered.setdefault(grp, []).append(g)
        for grp, vals in sorted(covered.items()):
            nz = sum(1 for v in vals if v > 0)
            print(f"  {name}: group={grp:<22} tensors={len(vals):3d} "
                  f"with_grad={nz:3d}  sum|grad|={sum(vals):.3e}")
        OUT.setdefault("grad_groups", {})[name] = {
            g: {"n": len(v), "n_nonzero": sum(1 for z_ in v if z_ > 0),
                "sum": sum(v)} for g, v in covered.items()}

    print("\n" + "=" * 78)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()

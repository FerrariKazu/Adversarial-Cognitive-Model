"""Stage 2 pre-flight (2026-08-13): per-step |dW| for the HPC predictor.

Question, before burning a 15-epoch smoke: after the optimizer-attention fix
(two SGD param groups: backbone lr=0.003, hpc_stack lr=0.02 = 0.003*6.67,
per-group grad clip; w_hpc=0.1 UNCHANGED), does the head's per-step weight
movement land in the regime where learning is visible above noise?

References:
  * starved (pre-fix real loop):  last-conv per-step |dW| ~ 1.36e-5 on |W|~0.005
    (rel ~0.27%/step) -> hpc_error_mean froze at the predict-zero baseline
    (0.6904 -> 0.6911, ratio 1.00) across 15 epochs (smoke #3).
  * isolated learnability recipe: SGD(hpc_params, lr=0.05), w_hpc=1.0 (no cut),
    no clip -> 28% error drop in 10 steps (test_hpc_head_learns_under_optimization).
  * real-recipe regression test:   27% drop in 60 steps under the NEW recipe
    (test_hpc_head_learns_under_real_recipe, committed).

This script measures BOTH recipes on the SAME model/inputs (the real Stage 2
base checkpoint weights, so the backbone features are the ones the smoke will
see), and reports the |dW| statistics + error trend per recipe, plus verdicts.

Usage: python3 scratch/measure_hpc_dw.py
"""
import gc
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "phase1_training")

import torch

from checkpoint_utils import compat_load
from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from train_rhan_next import build_next_optimizer, clip_grad_per_group

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "checkpoints/rhan_next_ais_v1_halting_only_best.pth"
ACCUM = 8            # micro-batches per optimizer step (real loop: 16)
MICRO_B = 8          # batch per micro-batch (real loop: 16)
W_HPC = 0.10         # pre-registered loss weight — UNCHANGED
STEPS_A = 24         # real-recipe optimizer steps
STEPS_B = 10         # isolated-recipe steps (matches the 28%/10-step reference test)
STARVED_DW = 1.36e-5  # measured pre-fix per-step |dW| on the output conv

# The model was trained on STL10-normalized inputs (dataset_stl10.py stats).
# Raw [0,1] tensors push the backbone off-manifold and it diverges to NaN —
# the same failure seen in the earlier 2026-08-13 diagnostic runs.
_MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2242, 0.2215, 0.2239]).view(1, 3, 1, 1)


def build_model():
    cfg = RHANNextConfig(enable_ais=False, enable_hpc=True,
                         hpc_num_levels=1, hpc_error_weight=W_HPC)
    m = RHANNext(config=cfg)
    try:
        ckpt = compat_load(BASE, map_location="cpu")
    except Exception:
        ckpt = torch.load(BASE, map_location="cpu", weights_only=False)
    missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)
    n_missing_hpc = sum(1 for k in missing if "hpc" in k)
    print(f"  base ckpt: epoch={ckpt.get('epoch')} "
          f"best_acc={ckpt.get('best_acc')} | missing={len(missing)} "
          f"(hpc keys fresh-init: {n_missing_hpc}) "
          f"unexpected={len(unexpected)}")
    assert n_missing_hpc == len(missing), "unexpected missing keys beyond hpc"
    m = m.to(DEV).train()
    # Measure the HPC group's per-step movement ONLY: freeze the backbone,
    # exactly like the committed real-recipe reference test
    # (test_hpc_head_learns_under_real_recipe). In the real loop the backbone's
    # update is dominated by L_trades + L_recon; backprop'ing only l_hpc
    # through the whole backbone would apply a pure-HPC update the loop never
    # makes and knock the trained backbone off-manifold (-> NaN). The HPC
    # group's per-step |dW| depends only on the features at that step, so
    # freezing the backbone is faithful for what we measure.
    for name, p in m.named_parameters():
        p.requires_grad = "hpc" in name
    return m


def make_batch():
    # Random STL10-like images, normalized exactly as the training loader does.
    x = torch.rand(MICRO_B, 3, 96, 96, device=DEV)
    m = _MEAN.to(DEV)
    s = _STD.to(DEV)
    return (x - m) / s


def last_conv(m):
    return m.hpc_stack.levels[0].decoder[-2]


def hpc_err(m, x):
    with torch.no_grad():
        logits, traj = m(x, return_trajectory=True)
    return float(torch.stack(traj["hpc_errors"]).mean())


def run_recipe(m, mode):
    """mode='real': real loop recipe (w_hpc=0.1, lr=0.02, /accum, per-group
    clip). mode='isolated': SGD(hpc, lr=0.05), w_hpc=1.0, no clip."""
    n_steps = STEPS_A if mode == "real" else STEPS_B
    if mode == "real":
        opt = build_next_optimizer(m, phase_lr=0.003, hpc_lr_mult=6.67)
        g = [g for g in opt.param_groups if g["lr"] > 0.003][0]
        hpc_params = g["params"]
        assert len(hpc_params) > 0
    else:
        hpc_params = [p for n, p in m.named_parameters() if "hpc" in n]
        opt = torch.optim.SGD(hpc_params, lr=0.05)

    lc = last_conv(m)
    w_norm = float(lc.weight.detach().norm())
    w_ref = lc.weight.detach().clone()
    dws, rels = [], []
    errs = [hpc_err(m, make_batch())]
    t0 = time.time()
    for s in range(n_steps):
        opt.zero_grad(set_to_none=True)
        if mode == "real":
            for _ in range(ACCUM):          # accumulate micro-batches (real loop)
                x = make_batch()
                with torch.enable_grad():
                    logits, traj = m(x, return_trajectory=True)
                    l_hpc = m.get_hpc_loss(x, (logits, traj))
                loss = (W_HPC * l_hpc) / ACCUM
                loss.backward()
            clip_grad_per_group(opt, 1.0)   # real loop: per-group clip 1.0
        else:
            x = make_batch()
            with torch.enable_grad():
                logits, traj = m(x, return_trajectory=True)
                l_hpc = m.get_hpc_loss(x, (logits, traj))
            l_hpc.backward()                # isolated: w_hpc=1.0, no clip
        opt.step()
        dw = float((lc.weight.detach() - w_ref).norm())
        dws.append(dw)
        rels.append(dw / w_norm)
        w_ref = lc.weight.detach().clone()
        errs.append(hpc_err(m, make_batch()))
    dt = time.time() - t0
    return {
        "mode": mode,
        "steps": n_steps,
        "secs": dt,
        "dws": dws, "rels": rels,
        "err0": errs[0], "errN": errs[-1],
        "errs_mid": errs[1:-1],
    }


def summarize(r):
    drop = 1.0 - r["errN"] / r["err0"]
    return (f"  {r['mode']:<9} {r['steps']:>3} steps ({r['secs']:.0f}s) | "
            f"last-conv |dW|/step mean={sum(r['dws'])/len(r['dws']):.3e} "
            f"(max {max(r['dws']):.3e}) | rel={100*sum(r['rels'])/len(r['rels']):.2f}%/step "
            f"| HPC err {r['err0']:.4f} -> {r['errN']:.4f} "
            f"(drop {100*drop:.1f}%)")


def main():
    torch.manual_seed(0)
    print(f"device={DEV} | base={BASE} | accum={ACCUM} micro_b={MICRO_B} "
          f"w_hpc={W_HPC}")
    m = build_model()
    print("\n--- recipe A: REAL loop (w_hpc=0.1, head lr=0.02, per-group clip) ---")
    ra = run_recipe(m, "real")
    print(summarize(ra))
    print("\n--- recipe B: ISOLATED (lr=0.05, w_hpc=1.0, no clip) ---")
    rb = run_recipe(m, "isolated")
    print(summarize(rb))
    print("\n--- comparison ---")
    dwA = sum(ra["dws"]) / len(ra["dws"])
    dwB = sum(rb["dws"]) / len(rb["dws"])
    relA = sum(ra["rels"]) / len(ra["rels"])
    dropA = 1.0 - ra["errN"] / ra["err0"]
    dropB = 1.0 - rb["errN"] / rb["err0"]
    per_step_drop_A = dropA / ra["steps"]
    per_step_drop_B = dropB / rb["steps"]
    ratio = (per_step_drop_B / per_step_drop_A) if per_step_drop_A > 0 else float("inf")

    print(f"  starved (pre-fix)  : last-conv |dW|/step = 1.36e-5 (rel 0.27%/step), "
          f"err ratio 1.00 over 15 epochs")
    print(f"  real recipe (A)    : last-conv |dW|/step = {dwA:.3e} "
          f"(rel {100*relA:.2f}%/step), err drop {100*dropA:.1f}% over {ra['steps']} steps")
    print(f"  isolated (B)       : last-conv |dW|/step = {dwB:.3e}, "
          f"err drop {100*dropB:.1f}% over {rb['steps']} steps")
    print(f"  A-vs-starved |dW|  : {dwA/STARVED_DW:.1f}x larger per step "
          f"(theoretical from w_hpc 0.1 x lr 0.02 vs 0.003 ~= 6.7x; global-clip "
          f"dilution removed)")
    print(f"  A-vs-B per-step err drop ratio : {ratio:.2f}x (theory ~ (0.1*0.02)/(1.0*0.05) = 0.04)")

    ok = True
    print("\n--- VERDICT ---")
    c1 = dwA >= 5 * STARVED_DW
    c2 = relA >= 0.01          # >= 1%/step relative movement (starved: 0.27%)
    c3 = dropA > 0.0           # error actually declines over the window
    c4 = 0.005 <= ratio <= 0.5 # within an order of magnitude of isolated
    for name, passed, detail in [
        ("C1 |dW| >= 5x starved", c1, f"{dwA:.3e} vs {STARVED_DW:.2e}"),
        ("C2 rel movement >= 1%/step", c2, f"{100*relA:.2f}%/step"),
        ("C3 error declines", c3, f"{100*dropA:.1f}% over {ra['steps']} steps"),
        ("C4 within ~10x of isolated", c4, f"per-step drop ratio {ratio:.2f}x"),
    ]:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} — {detail}")
        ok = ok and passed
    print("\n  =>", "PROCEED: head movement is in the learnable regime; "
                   "re-run the 15-epoch smoke." if ok
          else "STOP: movement still too small; diagnose before the 15-epoch run.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

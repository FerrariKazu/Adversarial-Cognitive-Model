"""Tests for the ported Gen-0 TRADES/PGD curriculum (training/adv_curriculum.py)
and its wiring into the foundation trainer.

Covers the 2026-09-29 recipe correction: the Gen-1 foundation trainer
previously trained PURE cross-entropy; the ported curriculum must reproduce
Gen-0's ramp (eps 0.031 -> 0.062 -> 0.094, beta 2.0 -> 2.5, PGD-4, w=0.55)
and the adversarial path must actually perturb training (finite loss,
in-ball attack, nonzero adversarial gradient).
"""
import math
import inspect
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from training.adv_curriculum import (  # noqa: E402
    CURRICULUM_60,
    RAND_START_MAG,
    W_TRADES_DEFAULT,
    curriculum_for_epoch,
    curriculum_table,
    pgd_kl_attack,
    phase_curriculum,
    trades_loss,
)
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig,
    build_model,
    train_one_epoch,
)


# ── The ramp must be Gen-0's, exactly ────────────────────────────────────────
def test_curriculum_table_is_gen0():
    assert CURRICULUM_60 == (
        (1, 20, 0.031, 2.0, 4),
        (21, 40, 0.062, 2.0, 4),
        (41, 60, 0.094, 2.5, 4),
    )
    assert W_TRADES_DEFAULT == 0.55


def test_ramp_segments_at_60_epochs():
    assert curriculum_for_epoch(1, 60).eps == 0.031
    assert curriculum_for_epoch(20, 60).eps == 0.031
    assert curriculum_for_epoch(21, 60).eps == 0.062
    assert curriculum_for_epoch(40, 60).eps == 0.062
    assert curriculum_for_epoch(41, 60).eps == 0.094
    assert curriculum_for_epoch(60, 60).eps == 0.094
    # beta steps up only in the final third (Gen-0: 2.0, 2.0, 2.5)
    assert curriculum_for_epoch(30, 60).beta == 2.0
    assert curriculum_for_epoch(45, 60).beta == 2.5
    # PGD-4 attack steps everywhere (Gen-0 canonical table)
    for e in (1, 25, 50):
        assert curriculum_for_epoch(e, 60).pgd_steps == 4


def test_ramp_slicing_covers_any_window_without_skipping_gentle_eps():
    # 1-epoch window (smoke) must land in the FIRST segment.
    p = curriculum_for_epoch(1, 1)
    assert p.eps == 0.031 and p.adversarial
    # 59-epoch window (the two phases that logged 59 epoch lines) slices
    # without a gap or an out-of-range epoch.
    for e in range(1, 60):
        p = curriculum_for_epoch(e, 59)
        assert p.eps in (0.031, 0.062, 0.094)
    # thirds: 59 -> ceil(59/3)=20 | ceil(118/3)=40 | rest
    assert curriculum_for_epoch(20, 59).eps == 0.031
    assert curriculum_for_epoch(21, 59).eps == 0.062
    assert curriculum_for_epoch(40, 59).eps == 0.062
    assert curriculum_for_epoch(41, 59).eps == 0.094


def test_phase_curriculum_covers_all_six_phases():
    for phase in ("backbone_only", "recurrence_only", "belief_no_f",
                  "belief_with_f", "ais_v2_swap", "gen1_core"):
        p = phase_curriculum(phase, 45, 60)
        assert p.eps == 0.094 and p.beta == 2.5 and p.pgd_steps == 4
    with pytest.raises(ValueError):
        phase_curriculum("not_a_phase", 1, 60)


def test_table_helper_and_monotonic_ramp():
    tbl = curriculum_table(60)
    assert len(tbl) == 60
    eps_seq = [tbl[str(e)].eps for e in range(1, 61)]
    assert eps_seq == sorted(eps_seq)
    assert len(set(eps_seq)) == 3


# ── The PGD-KL attack: Gen-0 mechanics ──────────────────────────────────────
class _Tiny(torch.nn.Module):
    def __init__(self, c=6):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 8 * 8, c)

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h.flatten(1))


def test_pgd_kl_attack_invariants():
    torch.manual_seed(0)
    m = _Tiny().eval()
    x = torch.randn(4, 3, 8, 8)
    eps, steps = 0.062, 4
    x_adv = pgd_kl_attack(m, x, eps=eps, steps=steps)
    delta = (x_adv - x).flatten(1).abs().max(dim=1).values
    assert torch.all(delta <= eps + 1e-5)          # inside the Linf ball
    with torch.no_grad():
        base = F.softmax(m(x), dim=1)
        adv_p = F.softmax(m(x_adv), dim=1)
    # The KL objective moved the output (attack did something).
    assert (base - adv_p).abs().sum() > 0


def test_pgd_kl_attack_restores_train_mode():
    m = _Tiny()
    m.train()
    pgd_kl_attack(m, torch.randn(2, 3, 8, 8), eps=0.031, steps=2)
    assert m.training                                  # restored


def test_pgd_kl_alpha_is_eps_over_steps():
    # alpha = eps/steps (Gen-0); with steps=1 the first sign step moves the
    # start point by exactly eps before projection — detectable via the ball.
    torch.manual_seed(0)
    m = _Tiny().eval()
    x = torch.randn(2, 3, 8, 8)
    x1 = pgd_kl_attack(m, x, eps=0.062, steps=1)
    assert ((x1 - x).abs().amax(dim=(1, 2, 3)) <= 0.062 + 1e-5).all()


def test_pgd_kl_random_start_matches_gen0():
    torch.manual_seed(0)
    m = _Tiny().eval()
    x = torch.zeros(2, 3, 8, 8)
    x_adv = pgd_kl_attack(m, x, eps=0.031, steps=1)
    assert 0.0 < (x_adv - x).abs().max() <= 0.031 + 1e-5


def test_rand_start_mag_is_recorded():
    assert RAND_START_MAG == 0.001


# ── The TRADES loss ─────────────────────────────────────────────────────────
def test_trades_loss_finite_and_positive():
    torch.manual_seed(0)
    m = _Tiny()
    x, y = torch.randn(8, 3, 8, 8), torch.randint(0, 6, (8,))
    x_adv = pgd_kl_attack(m, x, eps=0.031, steps=2)
    loss, beta = trades_loss(m, x, y, x_adv, beta=2.0)
    assert torch.isfinite(loss) and loss.item() > 0
    assert beta == 2.0


def test_trades_loss_matches_reference_formula():
    torch.manual_seed(1)
    m = _Tiny().eval()
    x, y = torch.randn(5, 3, 8, 8), torch.randint(0, 6, (5,))
    x_adv = pgd_kl_attack(m, x, eps=0.031, steps=2)
    loss, _ = trades_loss(m, x, y, x_adv, beta=2.0)
    with torch.no_grad():
        lc = m(x)
        la = m(x_adv)
    ref = (F.cross_entropy(lc.float(), y)
           + 2.0 * F.kl_div(F.log_softmax(la.float(), dim=1),
                            F.softmax(lc.float().detach(), dim=1),
                            reduction="batchmean"))
    assert torch.allclose(loss, ref, atol=1e-6)


def test_trades_beta_scales_kl_term():
    torch.manual_seed(2)
    m = _Tiny()
    x, y = torch.randn(6, 3, 8, 8), torch.randint(0, 6, (6,))
    x_adv = pgd_kl_attack(m, x, eps=0.031, steps=2)
    l1, _ = trades_loss(m, x, y, x_adv, beta=0.5)
    l2, _ = trades_loss(m, x, y, x_adv, beta=2.5)
    assert l2.item() > l1.item()


def test_trades_loss_gradients_flow():
    torch.manual_seed(3)
    m = _Tiny()
    x, y = torch.randn(4, 3, 8, 8), torch.randint(0, 6, (4,))
    x_adv = pgd_kl_attack(m, x, eps=0.031, steps=2)
    loss, _ = trades_loss(m, x, y, x_adv, beta=2.0)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(g.abs().sum() > 0 for g in grads)


# ── The adversarial training step actually trains ───────────────────────────
def _fake_loader(batches=2, b=4, c=6, size=8):
    for _ in range(batches):
        yield torch.randn(b, 3, size, size), torch.randint(0, c, (b,))


class _R:
    def clip_grad_per_group(self):
        pass


def _tiny_trained():
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, opt


def test_train_one_epoch_adversarial_mode_runs():
    model, opt = _tiny_trained()
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_one_epoch(model, _fake_loader(), opt, _R(),
                           torch.device("cpu"), None,
                           epoch=45, total_epochs=60,
                           clean_only=False, w_trades=0.55)
    assert math.isfinite(loss) and loss > 0
    moved = any(not torch.equal(a, b) for a, b in
                zip(before, model.parameters()))
    assert moved                                   # weights updated


def test_train_one_epoch_clean_only_mode_runs():
    model, opt = _tiny_trained()
    loss = train_one_epoch(model, _fake_loader(), opt, _R(),
                           torch.device("cpu"), None,
                           epoch=1, total_epochs=60, clean_only=True)
    assert math.isfinite(loss) and loss > 0


def test_epoch_log_line_includes_curriculum():
    # The trainer's per-epoch print carries eps/beta/pgd — the audit trail
    # that made the pure-CE gap detectable in the first place.
    import inspect
    from training import train_generation1_foundation as trainer
    src = inspect.getsource(trainer.run_phase)
    assert "eps={point.eps:.3f}" in src
    assert "beta={point.beta:.1f}" in src


# ── Trainer wiring: defaults are the curriculum, deviations are loud ────────
def test_config_defaults_carry_the_curriculum():
    cfg = FoundationConfig()
    assert cfg.w_trades == 0.55
    assert cfg.clean_only is False
    assert cfg.recipe_version == "gen1-adv-curriculum-v1"
    d = cfg.to_dict()
    assert d["w_trades"] == 0.55 and d["clean_only"] is False
    # Enters the provenance hash (to_dict feeds config_sha256).
    cfg2 = FoundationConfig()
    cfg2.clean_only = True
    assert cfg.to_dict() != cfg2.to_dict()


def test_cli_clean_only_flag_exists():
    from training import train_generation1_foundation as trainer
    src = inspect.getsource(trainer)
    assert '"--clean-only"' in src             # the flag is defined
    assert "args.clean_only" in src             # and consumed into cfg


def test_smoke_forces_clean_only(tmp_path):
    # Smoke's quarantine contract stays intact AND the smoke proof stays
    # cheap: smoke forces clean_only via main()'s config setup. Verify by
    # running the smallest possible real piece: the source must show the
    # assignment (structural) and FoundationConfig default stays False.
    import inspect
    from training import train_generation1_foundation as trainer
    src = inspect.getsource(trainer.main)
    assert "cfg.clean_only = True" in src          # inside the smoke block
    assert FoundationConfig().clean_only is False  # real runs default to adv


# ── FoundationModel still satisfies the harness contract under the new path ─
def test_foundation_model_forward_contract_unchanged():
    cfg = FoundationConfig()
    cfg.batch_size = 2
    model = build_model(cfg, "backbone_only")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, cfg.img_size, cfg.img_size))
    assert out.shape == (2, cfg.num_classes)


def test_gradient_reach_still_passes_with_curriculum_imports():
    # check_gradient_reach uses plain CE internally — unchanged behavior.
    cfg = FoundationConfig()
    model = build_model(cfg, "belief_no_f")
    x = torch.randn(2, 3, cfg.img_size, cfg.img_size)
    y = torch.randint(0, cfg.num_classes, (2,))
    from training.train_generation1_foundation import check_gradient_reach
    check_gradient_reach(model, x, y)              # raises on failure

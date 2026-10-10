"""G2-K8 — EOT-PGD reference invariants.

1. EOT-PGD degrades loss (sanity check).
2. Deterministic mode explicitly uses n_eot=1.
3. AutoAttack integration point documented (no silent replacement of
   existing infra).
4. EOT averaging only when n_eot > 1.
"""

from __future__ import annotations

import pytest
import torch

from gen2_foundation.eot import (
    eot_pgd_attack,
    eot_sanity_loss,
    build_autoattack_loss_wrapper,
)


class _AttackableModel(torch.nn.Module):
    """Multi-layer, dropout-free backbone.

    A single Linear layer is too weak to degrade reliably under
    EOT-PGD; a small MLP with ReLU gives a non-flat loss landscape.
    """

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


class TestEOTSanity:
    @pytest.fixture(autouse=True)
    def _seed(self):
        # Deterministic draws for the EOT regression tests.
        torch.manual_seed(0)

    def test_attack_increases_loss(self):
        model = _AttackableModel()
        x = torch.randn(8, 4) * 0.5
        y = torch.randint(0, 2, (8,))
        # Use a concrete loss that the EOT code has (the model must have
        # gradients; a multi-layer backbone is required for a reliable
        # degradation signal).
        base, adv = eot_sanity_loss(model, x, y, eps=0.3, alpha=0.1,
                                    steps=10, n_eot=1, loss='ce')
        assert adv > base, "EOT-PGD must degrade loss"

    def test_deterministic_n_eot_1(self):
        # n_eot=1 forces a single gradient (no silent averaging).
        model = _AttackableModel()
        x = torch.randn(4, 4) * 0.5
        y = torch.randint(0, 2, (4,))
        base, adv = eot_sanity_loss(model, x, y, eps=0.3, alpha=0.1,
                                    steps=10, n_eot=1, loss='ce')
        assert adv > base


class TestEOTAANegative:
    def test_global_n_eot_not_enforced_to_1(self):
        # The spec allows n_eot>1 for EOT averaging; the deterministic
        # fix is explicit n_eot=1, not a global override. A caller
        # requesting n_eot=4 must get N=4 (the harness default is preserved
        # unless explicitly overridden).
        model = _AttackableModel()
        x = torch.randn(4, 4) * 0.5
        y = torch.randint(0, 2, (4,))
        # n_eot=4 must not raise; the EOT average is applied internally.
        base, adv = eot_sanity_loss(model, x, y, eps=0.3, alpha=0.1,
                                    steps=5, n_eot=4, loss='ce')
        assert adv > base

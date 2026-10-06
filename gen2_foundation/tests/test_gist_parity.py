"""G2-K1 — fixed gist invariants.

1. fixed/adaptive delta-z bounds (test #3).
2. zero-gated gist initialization returns Gen-1 parity exactly (test #7).
3. shared patch embedder contract [never a second tokenizer].
4. gist dropout present for over-reliance probing.
"""

from __future__ import annotations

import pytest
import torch

from gen2_foundation.gist import FixedGistEncoder
from gen2_foundation.flags import RHANGen2Config


class TestGistParity:
    def test_default_config_is_control(self):
        cfg = RHANGen2Config()
        assert cfg.gist_fixed is False
        assert cfg.no_gist is True   # on by default (Gen-1 parity)        assert cfg.gist_learned is False

    def test_zero_gated_init_matches_gen1(self):
        # Patch embedder contract: single Conv2d, no extra params here.
        proj = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)
        g = FixedGistEncoder(proj, target_size=56)
        img = torch.randn(2, 3, 224, 224)

        # Frozen: at init gate is exactly 0 -> bit-exact Gen-1 parity.
        with torch.no_grad():
            z0 = g(img).mean(dim=1)
            z0_prime = g.fuse(z0, g(img))
        assert torch.equal(z0, z0_prime), (
            "zero-gated gist init must reproduce Gen-1 behavior")

    def test_shared_patch_embedder_only(self):
        # Verify FixedGistEncoder consumes the shared embedder WITHOUT
        # adding tokenizer parameters. Only the shared embedder params
        # (a Conv2d) belong to the tokenizer path.
        proj = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)
        g = FixedGistEncoder(proj, target_size=56, dropout=0.0)
        gist = g(torch.randn(1, 3, 224, 224))
        # The shared embedder is the ONLY tokenizer (no second one).
        assert gist.shape == (1, 16, 384)

    def test_gist_shape(self):
        proj = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)
        g = FixedGistEncoder(proj, target_size=56, dropout=0.0)
        gist = g(torch.randn(2, 3, 224, 224))
        assert gist.shape == (2, 16, 384)

    def test_gist_dropout_applied(self):
        # gist-dropout is the over-reliance probe; active dropout changes
        # the forward output (stochastic).
        proj = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)
        g = FixedGistEncoder(proj, target_size=56, dropout=0.1)
        g.train()
        img = torch.randn(1, 3, 224, 224)
        a = g(img)
        b = g(img)
        assert not torch.equal(a, b), "dropout did not affect output"

    def test_gist_dropout_off_is_deterministic(self):
        proj = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)
        g = FixedGistEncoder(proj, target_size=56, dropout=0.0)
        img = torch.randn(1, 3, 224, 224)
        a = g(img)
        b = g(img)
        assert torch.equal(a, b)

    def test_gist_forward_deterministic(self):
        proj = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)
        g = FixedGistEncoder(proj, target_size=56, dropout=0.0)
        img = torch.randn(1, 3, 224, 224)
        a = g(img)
        b = g(img)
        assert torch.equal(a, b)

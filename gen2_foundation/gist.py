"""G2-K1 — fixed (and learned) gist encoder with bit-exact Gen-1 parity gate.

Implementation concept (per spec):
  * Full-frame 224x224 -> 56x56 gist passed through the SHARED patch
    embedder (`CompactViT.patch_embed`) — never a second tokenizer.
    This is a hard architectural invariant: constructing a second
    independent patch embedder re-introduces the exact confound K1 is
    meant to remove.
  * Gist is 16 tokens (224/14 = 16).
  * Gist fusion happens at t=0 only: z_0' = GistFusion(z_0, gist_vector).
  * Fusion gate initialized at EXACTLY zero -> at initialization the arm
    is bit-exact Gen-1 behavior; gist usage must be LEARNED.
  * gist_dropout available for the over-reliance probe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Full-frame downscale factor for the fixed gist (224 -> 56).
GIST_TARGET_SIZE = 56
#: Patch size used by CompactViT (14x14). 56 / 14 = 16 gist tokens.
GIST_NUM_TOKENS = 16


class FixedGistEncoder(nn.Module):
    """Downscale the full input frame to a 56x56 gist token bank.

    Args:
        patch_embed: the shared `CompactViT.patch_embed` (the single
            patch embedder of the whole substrate). Passed in, never
            constructed here. Hard invariant: this module owns no
            tokenizer parameters of its own.
        target_size: (default 56) - full-frame downscale target.
        num_tokens: (default 16) - 56 / 14 = 16 tokens.
    """

    def __init__(
        self,
        patch_embed: nn.Module,
        target_size: int = GIST_TARGET_SIZE,
        num_tokens: int = GIST_NUM_TOKENS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if target_size % 14 != 0:
            raise ValueError(
                f"target_size {target_size} must be divisible by patch size "
                f"14 so the downscale maps cleanly onto patch_embed"
            )
        self.patch_embed = patch_embed      # shared, hard invariant
        self.target_size = int(target_size)
        self.num_tokens = int(num_tokens)
        self.dropout = dropout

        # No parameters of our own: the gist is produced by the shared
        # embedder. This is the zero-new-tokenizer-params invariant.

        # Fixed averaging pool for the gist (no learnable params).
        self.gist_pool = nn.AdaptiveAvgPool2d((target_size, target_size))

        # Fusion gate: initialized at EXACTLY zero -> bit-exact Gen-1
        # parity at initialization; learns gist usage during training.
        self.gate = nn.Parameter(torch.zeros(1))

    # ---- forward ----

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode the full-frame image via the shared patch embedder.

        Args:
            image: (B, 3, 224, 224) full RGB frame.

        Returns:
            gist_vector: (B, num_tokens, D) — 16 gist tokens produced by
            the shared patch embedder over the downscale gist.
        """
        # 224 -> 56: keep the shared tokenizer contract; no second embedder.
        gist = self.gist_pool(image)              # (B, C, 56, 56)

        # Shared patch embedder: the ONLY tokenizer in the system.
        # patch_embed is nn.ModuleDict({"proj": Conv2d(...)}) in the
        # substrate (CompactViT). Take the 'proj' key if available.
        proj = self.patch_embed["proj"] if isinstance(self.patch_embed, dict) else self.patch_embed
        B, C, _, _ = gist.shape
        h = proj(gist)        # (B, D, 4, 4)
        tokens = h.flatten(2).transpose(1, 2)     # (B, 16, D)

        if self.training and self.dropout > 0.0:
            tokens = nn.functional.dropout(tokens, p=self.dropout, training=True)

        return tokens

    # ---- fusion (t=0 only) ----

    def fuse(self, z0: torch.Tensor, gist: torch.Tensor) -> torch.Tensor:
        """Fuse gist into the t=0 belief state.

        z_0' = z_0 + gate * gist_attention
        with gate initialized at exactly 0 -> z_0' == z_0 at init
        (bit-exact Gen-1 parity), and gist usage is learned.

        Args:
            z0: (B, D_z) the current belief before t=0 fusion.
            gist: (B, 16, D) — the gist token bank.

        Returns:
            z0_p: (B, D_z) fused belief state at t=0.
        """
        # Projection: gist token dim (16) -> single global gist vector.
        # Mean-pool over the token dimension -> (B, D), a single scene prior.
        gist_global = gist.mean(dim=1)            # (B, D)

        # Gate is scalar; broadcast to (B, 1) to broadcast over D.
        w = torch.tanh(self.gate)                 # (1,) -> scalar
        return z0 + w * gist_global


# ---- parity check ----

def _power_on_gate(cfg: FixedGistEncoder, z0: torch.Tensor,
                    gist: torch.Tensor) -> torch.Tensor:
    """Power ON the gist gate from exactly 0 and check parity."""
    init_gate = cfg.gate.detach().clone()
    cfg.gate.mul_(0.0).add_(1.0)                # leave at exactly 0.0
    z0_prime = cfg.fuse(z0, gist)
    cfg.gate.copy_(init_gate)                   # restore
    if not torch.equal(z0, z0_prime):
        raise AssertionError("zero-gated gist init does not match Gen-1")
    return z0_prime


if __name__ == "__main__":
    # Sanity: 32-bit float, 224x224, shared patch embedder contract.
    class _DummyPatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._proj = nn.Conv2d(3, 384, kernel_size=14, stride=14)

        def __getitem__(self, key):
            return self._proj

    emb = _DummyPatchEmbed()
    g = FixedGistEncoder(emb, target_size=56)
    img = torch.randn(2, 3, 224, 224)
    gist = g(img)
    assert gist.shape == (2, 16, 384), f"gist shape {gist.shape}"
    # zero gate => parity
    z0 = gist.mean(dim=1)
    z0p = g.fuse(z0, gist)
    assert torch.equal(z0, z0p), "parity violated"
    print("FixedGistEncoder OK:", gist.shape)

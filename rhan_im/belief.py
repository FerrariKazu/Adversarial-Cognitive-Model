"""The recurrent state of RHAN-IM: an exact Bayesian posterior over scene content.

Hypotheses
    H_0      : no target in the scene
    H_(b,c)  : a target of class c anchored at cell b      (b < N, c < C)

Evidence units
    Hypotheses are anchored at N cells; evidence arrives in V units (V == N by default; V != N
    lets coarse peripheral tokens and fine native-window tokens feed the same posterior). A target
    anchored at b is seen by the units with footprint[b, u] = 1 (footprint is (N, V); default
    identity). A look gives every unit u a score vector z[u] in R^C and a believed detectability d[u]:
        z[u, c] ~ N(d[u] * footprint[b, u], 1)   if the target is H_(b, c)
        z[u, c] ~ N(0, 1)                         otherwise
    so the log-likelihood ratio of H_(b,c) against H_0 is exactly
        sum_u footprint[b, u] * ( d[u] * z[u, c] - d[u]^2 / 2 ).

State (per batch element)
    log_prior (N, C)   log prior mass of every H_(b,c); with p_null it sums to 1
    u_contrib (V, C)   per-unit accumulated evidence  sum (d z - d^2/2)   (or best view only)
    d_best    (V)      best detectability with which unit u has been seen
    D2        (V)      accumulated unit precision   (sum d^2  |  d_best^2)
    S1        (V)      accumulated unit influence   (sum d    |  d_best)
    n_fix              number of measurements absorbed

Evidence-combination mode (it must match how the sensor's noise behaves)
    "sum"   looks are conditionally independent given the scene (fresh sensor noise per look)
    "best"  looks are nested deterministic views of the same pixels: only the view of a unit
            with the largest d counts; a better view *replaces* the old one.
Optional robustness knobs
    kappa      clip every score to [-kappa, kappa] (per look, at update time)
    unit_cap   clip every unit's accumulated evidence to [-unit_cap, unit_cap]

There are no recurrent weights: all learning lives in the measurement model (the evidence
head that produces z, and the measured visibility d).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from rhan_im.sdt import bayes_error_equal_prior

MODES = ("sum", "best")


class SearchPosterior:
    def __init__(
        self,
        n_cells: int,
        n_classes: int,
        batch: int,
        p_null: float = 0.4,
        log_prior_cells: Optional[torch.Tensor] = None,
        mode: str = "sum",
        footprint: Optional[torch.Tensor] = None,
        unit_cap: Optional[float] = None,
        n_units: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not (0.0 < p_null < 1.0):
            raise ValueError("p_null must lie in (0, 1)")
        self.N, self.C, self.B = int(n_cells), int(n_classes), int(batch)
        self.V = int(n_units) if n_units is not None else self.N
        self.mode = mode
        self.p_null0 = float(p_null)
        self.unit_cap = unit_cap
        kw = dict(device=device, dtype=dtype)
        if footprint is None and self.V != self.N:
            raise ValueError("n_units != n_cells needs an explicit footprint (N, V)")
        if footprint is not None:
            if footprint.shape != (self.N, self.V):
                raise ValueError("footprint must have shape (N, V)")
            if not bool(((footprint == 0) | (footprint == 1)).all()):
                raise ValueError("footprint must be binary (the exact LLR is linear in it only then)")
            self.F = footprint.to(**kw)
        else:
            self.F = None
        if log_prior_cells is None:
            lp = torch.full((self.B, self.N, self.C), (1.0 - p_null) / (self.N * self.C), **kw).log()
        else:
            if log_prior_cells.shape != (self.B, self.N, self.C):
                raise ValueError("log_prior_cells must have shape (B, N, C)")
            lp = log_prior_cells.to(**kw)
            lp = lp - torch.logsumexp(lp.flatten(1), dim=1)[:, None, None]
            lp = lp + torch.log(torch.tensor(1.0 - p_null, **kw))
        self.log_prior = lp
        self.u_contrib = torch.zeros(self.B, self.V, self.C, **kw)
        self.log_cells = self.log_prior.clone()
        self.log_null = torch.full((self.B,), p_null, **kw).log()
        self.d_best = torch.zeros(self.B, self.V, **kw)
        self.D2 = torch.zeros(self.B, self.V, **kw)
        self.S1 = torch.zeros(self.B, self.V, **kw)
        self.n_fix = torch.zeros(self.B, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ helpers
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-level (B, V, C) quantity -> hypothesis-level (B, N, C) via the footprint."""
        return x if self.F is None else torch.einsum("hu,buc->bhc", self.F, x)

    def hyp_precision(self, d2_units: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(..., N) total precision with which each hypothesis can be told from H_0."""
        d2 = self.D2 if d2_units is None else d2_units
        return d2 if self.F is None else torch.einsum("...u,hu->...h", d2, self.F)

    @property
    def contrib(self) -> torch.Tensor:
        return self.project(self.u_contrib)

    def repeat(self, n: int) -> "SearchPosterior":
        """Branch every batch element into n copies (element i -> rows i*n .. i*n+n-1)."""
        out = SearchPosterior.__new__(SearchPosterior)
        out.N, out.C, out.V, out.mode, out.p_null0 = self.N, self.C, self.V, self.mode, self.p_null0
        out.F, out.unit_cap = self.F, self.unit_cap
        out.B = self.B * n
        for name in ("log_prior", "u_contrib", "log_cells", "log_null", "d_best", "D2", "S1", "n_fix"):
            setattr(out, name, getattr(self, name).repeat_interleave(n, dim=0).clone())
        return out

    def slice(self, i0: int, i1: int) -> "SearchPosterior":
        out = SearchPosterior.__new__(SearchPosterior)
        out.N, out.C, out.V, out.mode, out.p_null0 = self.N, self.C, self.V, self.mode, self.p_null0
        out.F, out.unit_cap = self.F, self.unit_cap
        out.B = i1 - i0
        for name in ("log_prior", "u_contrib", "log_cells", "log_null", "d_best", "D2", "S1", "n_fix"):
            setattr(out, name, getattr(self, name)[i0:i1])
        return out

    # ------------------------------------------------------------------ update
    def update(
        self,
        z: torch.Tensor,
        d: torch.Tensor,
        kappa: Optional[float] = None,
        active: Optional[torch.Tensor] = None,
        temper: float = 1.0,
    ) -> None:
        """Absorb one measurement. `active` (B,) bool freezes finished trials."""
        if z.dim() != 3 or z.shape[-1] != self.C or z.shape[0] != self.B or z.shape[1] not in (1, self.V):
            raise ValueError(f"z must be (B, V|1, C); got {tuple(z.shape)}")
        if d.shape != (self.B, self.V):
            raise ValueError(f"d must be (B, V); got {tuple(d.shape)}")
        if (d < 0).any():
            raise ValueError("detectability must be non-negative")
        if kappa is not None:
            z = z.clamp(-kappa, kappa)
        dd = d.unsqueeze(-1)
        llr = temper * (dd * z - 0.5 * dd * dd)                      # (B, V, C) per unit
        if active is None:
            gate = torch.ones(self.B, dtype=torch.bool, device=d.device)
        else:
            gate = active.to(torch.bool)
        g2 = gate[:, None]
        if self.mode == "sum":
            self.u_contrib = self.u_contrib + llr * g2[..., None].to(llr.dtype)
            self.D2 = self.D2 + (d * d) * g2.to(d.dtype)
            self.S1 = self.S1 + d * g2.to(d.dtype)
        else:
            better = (d > self.d_best) & g2
            self.u_contrib = torch.where(better[..., None], llr, self.u_contrib)
            self.d_best = torch.where(better, d, self.d_best)
            self.D2 = self.d_best ** 2
            self.S1 = self.d_best.clone()
        if self.unit_cap is not None:
            self.u_contrib = self.u_contrib.clamp(-self.unit_cap, self.unit_cap)
        self.n_fix = self.n_fix + gate.long()
        self.log_cells = self.log_prior + self.project(self.u_contrib)

    # --------------------------------------------------------------- read-outs
    def _log_norm(self) -> torch.Tensor:
        flat = torch.cat([self.log_null[:, None], self.log_cells.flatten(1)], dim=1)
        return torch.logsumexp(flat, dim=1)

    def p_null(self) -> torch.Tensor:
        return (self.log_null - self._log_norm()).exp()

    def p_cells(self) -> torch.Tensor:
        return (self.log_cells - self._log_norm()[:, None, None]).exp()

    def p_class(self) -> torch.Tensor:
        return self.p_cells().sum(1)

    def p_loc(self) -> torch.Tensor:
        return self.p_cells().sum(2)

    def outcome_log_weights(self) -> torch.Tensor:
        """(B, 1 + C) unnormalised log weight of [absent, class 0, ..., class C-1]."""
        return torch.cat([self.log_null[:, None], torch.logsumexp(self.log_cells, dim=1)], dim=1)

    def outcome_probs(self) -> torch.Tensor:
        return torch.softmax(self.outcome_log_weights(), dim=1)

    def decision(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bayes decision over {absent, class c}: (index, its posterior probability)."""
        p, idx = self.outcome_probs().max(dim=1)
        return idx, p

    def risk(self) -> torch.Tensor:
        return 1.0 - self.decision()[1]

    def ignorance_mass(self) -> torch.Tensor:
        """sum_b P(target anchored at b) * P(a target there would still be missed), where
        'missed' is the equal-prior Bayes error Phi(-sqrt(precision)/2) of telling H_b from H_0."""
        return (self.p_loc() * bayes_error_equal_prior(self.hyp_precision().sqrt())).sum(1)

    def joint_log_weights(self) -> torch.Tensor:
        flat = torch.cat([self.log_null[:, None], self.log_cells.flatten(1)], dim=1)
        return flat - self._log_norm()[:, None]

    def entropy(self) -> torch.Tensor:
        lw = self.joint_log_weights()
        return -(lw.exp() * lw).sum(1)

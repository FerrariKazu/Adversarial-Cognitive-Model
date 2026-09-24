"""
Center-bias measurement — Agent F. Gate 7 infrastructure, NOT a gate.
================================================================================

What this module is: the empirical measurement the plan REQUIRES before
Gate 7 can even be evaluated — "track and report the empirical
distribution of chosen gaze locations across a validation batch" (docs
10_AIS_v2 engineering rule 2; Part 1.E: center-bias = FAILURE CONDITION).

What this module is NOT: a threshold. Gate 7's numeric criterion is
PENDING DECISION (Part 6) — this module deliberately ships the
measurement, the per-batch summary, and the DEGENERATE flag's
infrastructure, and nothing that pretends to decide the gate.

Definition of center (LOCKED conventions, Part 1.A): a gaze location is
"at center" when its normalized coordinates lie within `radius` of
(0, 0) in the [-1, +1] image frame (Euclidean distance). The default
radius 0.25 matches the fovea's half-width at the default full-frame
foveation (56/56 / 2 = 0.5 half-width in normalized coords → a centered
crop spans the middle half of the image); the radius is a parameter of
the MEASUREMENT, never silently a pass/fail criterion.

The degenerate-policy signal (the thing the measurement must catch, per
the Agent H1 responsiveness-guard pattern): a policy that always
produces the same location has a near-zero spatial spread. `fraction_at_center`
and `mean_radius` together flag it: fraction_at_center -> 1 AND
mean_radius -> 0. A genuinely exploratory policy scatters locations and
produces a usable histogram; the MEASUREMENT NEVER returns a verdict on
Gate 7 itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

#: Measurement default only — a reporting parameter, NOT a gate threshold.
DEFAULT_CENTER_RADIUS = 0.25
#: Grid resolution of the empirical gaze-location histogram.
DEFAULT_HIST_BINS = 8


@dataclass
class CenterBiasReport:
    """The empirical gaze distribution over a batch (measurement only).

    degenerate is the MEASUREMENT's own diagnostic flag (always-center /
    zero-spread detection). It is infrastructure for Gate 7's future
    threshold decision — not the gate verdict, which stays PENDING.
    """

    n_locations: int
    fraction_at_center: float        # share within `radius` of (0, 0)
    mean_radius: float               # mean Euclidean distance from center
    radius_std: float                # spatial spread of the radius
    hist_edges: List[float]          # shared bin edges, [-1, +1]
    hist_x: List[float]              # marginal over x (normalized)
    hist_y: List[float]              # marginal over y (normalized)
    center_radius: float
    degenerate: bool
    reason: Optional[str] = None


class CenterBiasMeter:
    """Accumulates chosen gaze locations and reports the distribution.

    Feed every policy-selected (B, 2) location of a validation pass via
    update(); read one CenterBiasReport via report(). The meter detaches
    everything at the boundary — a measurement never alters the graph.
    """

    def __init__(self, center_radius: float = DEFAULT_CENTER_RADIUS,
                 hist_bins: int = DEFAULT_HIST_BINS,
                 degenerate_center_fraction: float = 0.999,
                 degenerate_radius: float = 1e-3):
        if not 0.0 < center_radius <= 1.0:
            raise ValueError(
                f"center_radius must lie in (0, 1]; got {center_radius}")
        if hist_bins < 2:
            raise ValueError("hist_bins must be >= 2")
        self.center_radius = float(center_radius)
        self.hist_bins = int(hist_bins)
        # Diagnostic-flag parameters (measurement-internal; NOT a gate).
        self._deg_frac = float(degenerate_center_fraction)
        self._deg_radius = float(degenerate_radius)
        self._locs: List[torch.Tensor] = []

    def update(self, locations: torch.Tensor) -> None:
        """Accumulate one batch of chosen (B, 2) locations."""
        if locations.dim() != 2 or locations.shape[1] != 2:
            got = tuple(locations.shape) if torch.is_tensor(locations) \
                else type(locations).__name__
            raise ValueError(f"locations must be (B, 2); got {got}")
        self._locs.append(locations.detach().reshape(-1, 2).clone())

    def reset(self) -> None:
        self._locs.clear()

    def _stacked(self) -> torch.Tensor:
        if not self._locs:
            raise ValueError(
                "no locations accumulated — call update() before report()")
        return torch.cat(self._locs, dim=0)

    def report(self) -> CenterBiasReport:
        locs = self._stacked()                            # (M, 2)
        M = locs.shape[0]
        radius = locs.norm(dim=-1)                        # (M,)
        frac_center = float((radius <= self.center_radius)
                            .float().mean().item())
        mean_radius = float(radius.mean().item())
        std_radius = float(radius.std(unbiased=False).item())

        # Marginal histograms over a shared [-1, +1] grid.
        edges = torch.linspace(-1.0, 1.0, self.hist_bins + 1)
        hx = torch.histc(locs[:, 0], bins=self.hist_bins,
                         min=-1.0, max=1.0)
        hy = torch.histc(locs[:, 1], bins=self.hist_bins,
                         min=-1.0, max=1.0)
        hx, hy = hx / M, hy / M

        # Degenerate detection: every location essentially identical AND
        # pinned at the center. Two conditions, both from the data — a
        # non-centered constant policy (e.g. always top-left) must NOT be
        # flagged by the center test; it shows up as a zero-spread
        # histogram instead (still reportable, honestly labeled).
        degenerate = frac_center >= self._deg_frac \
            and mean_radius <= self._deg_radius
        reason = None
        if degenerate:
            reason = (
                f"all {M} sampled locations lie within "
                f"{self.center_radius} of center (fraction_at_center="
                f"{frac_center:.4f}, mean_radius={mean_radius:.2e}) — "
                f"always-center policy signature")
        return CenterBiasReport(
            n_locations=M,
            fraction_at_center=frac_center,
            mean_radius=mean_radius,
            radius_std=std_radius,
            hist_edges=edges.tolist(),
            hist_x=hx.tolist(),
            hist_y=hy.tolist(),
            center_radius=self.center_radius,
            degenerate=bool(degenerate),
            reason=reason,
        )

    def __repr__(self) -> str:
        return (f"CenterBiasMeter(radius={self.center_radius}, "
                f"bins={self.hist_bins}, n={sum(l.shape[0] for l in self._locs)}"
                f") — Gate 7 measurement, threshold PENDING DECISION")


def gaze_location_histogram(locations: Sequence[torch.Tensor],
                            hist_bins: int = DEFAULT_HIST_BINS
                            ) -> torch.Tensor:
    """Joint 2-D histogram of chosen locations, (bins, bins).

    Rows index y, columns index x, bins span [-1, +1]; normalized to sum
    to 1. The raw artifact behind CenterBiasReport's marginals.
    """
    if not locations:
        raise ValueError("no locations given")
    locs = torch.cat([l.detach().reshape(-1, 2) for l in locations], dim=0)
    joint = torch.zeros(hist_bins, hist_bins)
    inside = locs[(locs.abs() <= 1.0 + 1e-6).all(dim=-1)]
    if inside.numel() == 0:
        raise ValueError("no locations inside [-1, +1]")
    idx = ((inside + 1.0) / 2.0 * hist_bins).long().clamp(0, hist_bins - 1)
    for x_i, y_i in idx.tolist():
        joint[y_i, x_i] += 1.0
    return joint / joint.sum()

"""Gate-1 measurement: does a trunk's token-level evidence behave like the equal-variance SDT sensor that
the RHAN-IM Bayes state assumes?  (Library; scripts/rhan_im/gate1_measure_dprime.py runs it on a trunk.)

Pipeline
    1. paste_squares / unit_overlap / window_token_overlap / gist_token_overlap  build scenes with a KNOWN
       footprint and say which tokens of which look overlap the target.
    2. train_evidence_head   fits a linear head on frozen trunk tokens: class c on covered tokens, background
       elsewhere; exports  score_c = logit_c - logit_bg  (the log-odds of H_(.,c) against H_0).
    3. fit_standardizer      null mean / std of every (token position, class) score on a calibration split.
    4. measure_tokens        on a held-out split: per token position the measured detectability d (shift of
                             the standardised score, in null-sd units), the z-ROC slope, the null / positive
                             sd, the wrong-class leakage.
    5. paired_noise_correlation  how correlated is the score of the same (scene, unit, class) between two
                             different looks? It decides between "sum" and "best" evidence combination.
    5b. block_vif            how dependent are the tokens WITHIN one look? The Bayes state adds token
                             evidence as if the tokens were independent; the variance-inflation factor of
                             the d-weighted block sum says how wrong that is (1 = independent) and gives
                             the closed-form correction for equicorrelated tokens (divide z and d by sqrt(VIF)).
    5c. spillover            does the own-class score rise on tokens that do not overlap the target? (the
                             footprint F of the Bayes state says it must not)
    6. verdict               draft pass / fail thresholds (to be frozen in a pre-registration before use).
"""
from __future__ import annotations

import hashlib
import math
import pathlib
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhan_im.layout import UnitLayout
from rhan_im.sdt import dprime_from_samples, zroc_slope
from rhan_im.window_head import EvidenceHead

# Every repository file whose content can change what the gate-1 measurement computes: the script, this library, the
# layout / adapter / head / SDT code it calls, the trunk definition and the modules the trunk imports its blocks from.
# tests/rhan_im/test_gate1_prereg.py fails if the measurement script loads a repository module that is neither listed
# here nor on that test's explicit "not part of the measurement" list, so an added import cannot be forgotten.
MEASUREMENT_CODE_FILES = (
    "scripts/rhan_im/gate1_measure_dprime.py",
    "rhan_im/gate1.py",
    "rhan_im/layout.py",
    "rhan_im/trunk_adapter.py",
    "rhan_im/window_head.py",
    "rhan_im/sdt.py",
    "gen2_foundation/backbone.py",
    "noesis_vision/models/recurrent_block.py",
    "noesis_vision/models/foveation.py",
)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def listing_hash(folder) -> Dict[str, object]:
    """sha256 of the sorted relative file paths under `folder` (names only: the practice of full_val_path_listing)."""
    folder = pathlib.Path(folder)
    rel = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    return {"sha256": hashlib.sha256("\n".join(rel).encode()).hexdigest(), "n_files": len(rel)}


# --------------------------------------------------------------------------- scene geometry
def paste_squares(bg: torch.Tensor, patches: torch.Tensor, top_px: torch.Tensor, left_px: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """bg (B,3,H,W); patches (B,3,s,s) pasted at integer pixel offsets with opacity alpha (B,)."""
    out = bg.clone()
    s = patches.shape[-1]
    for i in range(bg.shape[0]):
        t, l, a = int(top_px[i]), int(left_px[i]), float(alpha[i])
        out[i, :, t:t + s, l:l + s] = (1.0 - a) * out[i, :, t:t + s, l:l + s] + a * patches[i]
    return out


def unit_overlap(top_px: torch.Tensor, left_px: torch.Tensor, size_px: int, g: int, patch: int) -> torch.Tensor:
    """(B, g, g): fraction of every fine unit's area that lies inside the pasted square."""
    edges = torch.arange(g + 1, dtype=torch.float32) * patch
    lo, hi = edges[:-1], edges[1:]
    t, l = top_px.float()[:, None], left_px.float()[:, None]
    b, r = t + size_px, l + size_px
    oy = (torch.minimum(hi[None], b) - torch.maximum(lo[None], t)).clamp_min(0.0) / patch
    ox = (torch.minimum(hi[None], r) - torch.maximum(lo[None], l)).clamp_min(0.0) / patch
    return oy[:, :, None] * ox[:, None, :]


def window_token_overlap(overlap: torch.Tensor, topleft_units: torch.Tensor, layout: UnitLayout) -> torch.Tensor:
    """(B,g,g) overlap + (B,2) window top-left (fine units) -> (B, tok*tok) overlap of every window token."""
    idx, valid = layout.window_units(topleft_units)
    return overlap.flatten(1).gather(1, idx) * valid.to(overlap.dtype)


def gist_token_overlap(overlap: torch.Tensor, layout: UnitLayout) -> torch.Tensor:
    """(B,g,g) -> (B, tok*tok): mean overlap of every coarse (gist) token."""
    return F.avg_pool2d(overlap[:, None], layout.coarse_block).flatten(1)


# --------------------------------------------------------------------------- evidence head
def train_evidence_head(
    batches: Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    d_model: int,
    n_classes: int,
    steps: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    bg_weight: float = 0.25,
    seed: int = 0,
) -> Tuple[EvidenceHead, list]:
    """batches(step) -> tokens (b,T,D), covered (b,T) bool, label (b,) (-1 = no target), weight (b,T) float.
    Trains Linear(D, C+1) with cross-entropy (class c on covered tokens of a class-c scene, background
    otherwise) and exports score_c = logit_c - logit_bg as an EvidenceHead with C outputs."""
    g = torch.Generator().manual_seed(seed)
    lin = nn.Linear(d_model, n_classes + 1)
    nn.init.normal_(lin.weight, std=0.01, generator=g)
    nn.init.zeros_(lin.bias)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    log = []
    for step in range(steps):
        tokens, covered, label, weight = batches(step)
        y = torch.where(covered & (label >= 0)[:, None], label[:, None].expand_as(covered), torch.full_like(covered, n_classes, dtype=torch.long))
        w = weight * torch.where(y == n_classes, torch.full_like(weight, bg_weight), torch.ones_like(weight))
        logits = lin(tokens)                                          # (b, T, C+1)
        loss = (F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none") * w.flatten()).sum() / w.sum().clamp_min(1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            log.append((step, float(loss.detach())))
    head = EvidenceHead(d_model, n_classes)
    with torch.no_grad():
        head.proj.weight.copy_(lin.weight[:n_classes] - lin.weight[n_classes:n_classes + 1])
        head.proj.bias.copy_(lin.bias[:n_classes] - lin.bias[n_classes:n_classes + 1])
    return head, log


# --------------------------------------------------------------------------- SDT measurements
def null_mask(cover: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(n, T, C) True where the score is a clean NULL score: the token overlaps no target (any class).

    Wrong-class scores on a token that DOES overlap a target are deliberately not null: if they are shifted
    the model "z ~ N(0, 1) otherwise" is violated, and measure_tokens reports that shift as `leakage`."""
    return (~cover)[:, :, None].expand(-1, -1, n_classes)


def fit_standardizer(scores: torch.Tensor, cover: torch.Tensor, label: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Null mean / std of every (token position, class): scores (n, T, C) -> mu0, sd0 of shape (T, C)."""
    w = null_mask(cover, scores.shape[-1]).to(scores.dtype)
    n = w.sum(0).clamp_min(2.0)
    mu = (scores * w).sum(0) / n
    var = (((scores - mu) ** 2) * w).sum(0) / (n - 1.0)
    return mu, var.sqrt().clamp_min(1e-6)


def measure_tokens(scores: torch.Tensor, cover: torch.Tensor, label: torch.Tensor, mu0: torch.Tensor, sd0: torch.Tensor, tok: int, min_pos: int = 30) -> Dict[str, torch.Tensor]:
    """Per token position (returned as (tok, tok) maps; NaN where fewer than `min_pos` positives):
        d_shift     mean standardised score of the scene's own class on covered tokens minus the null mean.
                    This is the d that enters the LLR  d*z - d^2/2  (it presumes unit variance, see null_sd/pos_sd)
        d_prime     the same shift divided by the pooled sd (the textbook d')
        zroc_slope  slope of the z-ROC (equal-variance Gaussian evidence gives 1)
        null_sd / pos_sd / null_mean   spread and offset of the standardised scores
        leakage     mean standardised score of the WRONG classes on covered tokens, minus the null mean
        n_pos       number of positive samples"""
    z = (scores - mu0) / sd0
    n, T, C = z.shape
    null = null_mask(cover, C)
    has = label >= 0
    lab = label.clamp_min(0)
    own = z.gather(2, lab[:, None, None].expand(n, T, 1)).squeeze(2)            # (n, T)
    names = ("d_shift", "d_prime", "zroc_slope", "null_sd", "pos_sd", "null_mean", "leakage")
    out = {k: torch.full((T,), float("nan")) for k in names}
    out["n_pos"] = torch.zeros(T)
    cls = torch.arange(C)
    for t in range(T):
        zn = z[:, t, :][null[:, t, :]]
        out["null_sd"][t], out["null_mean"][t] = zn.std(), zn.mean()
        pm = cover[:, t] & has
        npos = int(pm.sum())
        out["n_pos"][t] = npos
        if npos < min_pos:
            continue
        zp = own[pm, t]
        out["pos_sd"][t] = zp.std()
        out["d_shift"][t] = zp.mean() - zn.mean()
        out["d_prime"][t] = dprime_from_samples(zp, zn)
        out["zroc_slope"][t] = zroc_slope(zp, zn)[0]
        wrong = pm[:, None] & (cls[None, :] != lab[:, None])
        out["leakage"][t] = z[:, t, :][wrong].mean() - zn.mean()
    return {k: v.reshape(tok, tok) for k, v in out.items()}


def paired_noise_correlation(za: torch.Tensor, zb: torch.Tensor) -> float:
    """Pearson correlation of two looks' standardised scores of the same (scene, unit, class) entries."""
    za, zb = za.double().flatten(), zb.double().flatten()
    za, zb = za - za.mean(), zb - zb.mean()
    return float((za * zb).sum() / (za.norm() * zb.norm()).clamp_min(1e-12))


def effective_looks(n_looks: int, rho: float) -> float:
    """Effective number of independent looks when the noise of K looks is equicorrelated (rho)."""
    return n_looks / (1.0 + (n_looks - 1.0) * rho)


def block_vif(z: torch.Tensor, cover: torch.Tensor, label: torch.Tensor, tok: int, block: int,
              weights: Optional[torch.Tensor] = None, min_pos: int = 30, min_scenes: int = 100,
              full_only: bool = True, min_pos_scenes: int = 50) -> Dict[str, float]:
    """Within-look dependence of token evidence: variance-inflation factor (VIF) of the weighted block sum.

    The Bayes state adds, for a hypothesis whose footprint spans several tokens of ONE look,
    sum_u (d_u z_u - d_u^2 / 2) as if the z_u were independent. If they are equicorrelated (rho) the exact
    log-likelihood ratio for equal d is that sum divided by VIF = 1 + (n - 1) rho, and the information
    D = sum d_u^2 shrinks by the same factor. VIF = Var(sum_u w_u z_u) / sum_u w_u^2 Var(z_u): 1 for
    independent tokens, n for fully dependent ones. Dividing the standardised scores and the detectabilities
    by sqrt(VIF) reproduces the exact equicorrelated likelihood ratio inside the existing machinery.

    z (n, T, C) standardised scores of one look type; cover (n, T) bool token overlaps a target; label (n,)
    class of the scene's target, -1 = no target; weights (T,) per-position detectability d (NaN = unusable;
    default all ones).
      vif_null  block x block token blocks of scenes WITHOUT a target, every class channel (a clean H_0 estimate)
      vif_pos   covered tokens of scenes WITH a target, own-class channel, residual around the per-position
                mean: pooled sum S_i^2 / sum N_i. Also reflects shared image difficulty (a target that is
                easy at one token is easy at its neighbours), which is exactly what the model ignores.
                With full_only (default) only scenes whose look covers ALL block x block tokens of the target
                count: a VIF grows with the number of tokens summed, so pooling windows that cover different
                numbers of tokens would make vif_pos incomparable with vif_null (which is a block x block
                quantity) and bias the shared-difficulty ratio and the spread estimate low.
    NaN where there are too few scenes / positives (min_scenes, min_pos, min_pos_scenes), or block < 2."""
    n, T, C = z.shape
    w = torch.ones(T) if weights is None else torch.nan_to_num(weights.flatten().double(), nan=0.0)
    z = z.double()
    out = {"vif_null": float("nan"), "n_null_scenes": 0, "vif_pos": float("nan"), "n_pos_scenes": 0}

    null_scene = label < 0
    m = int(null_scene.sum())
    out["n_null_scenes"] = m
    if block >= 2 and m >= min_scenes:
        z0 = z[null_scene]
        num = den = 0.0
        for r in range(tok - block + 1):
            for c in range(tok - block + 1):
                idx = torch.tensor([(r + i) * tok + (c + j) for i in range(block) for j in range(block)])
                wb = w[idx]
                if float((wb ** 2).sum()) == 0.0:
                    continue
                zb = z0[:, idx, :]                                               # (m, b*b, C)
                s = (zb * wb[None, :, None]).sum(1)                              # (m, C)
                num += float(s.var(0, unbiased=True).sum())
                den += float(((wb[:, None] ** 2) * zb.var(0, unbiased=True)).sum())
        if den > 0:
            out["vif_null"] = num / den

    has = label >= 0
    if block >= 2 and bool(has.any()):
        own = z.gather(2, label.clamp_min(0)[:, None, None].expand(n, T, 1)).squeeze(2)         # (n, T)
        cov_all = cover & has[:, None]
        cnt = cov_all.sum(0)
        ok = (cnt >= min_pos) & (w != 0)
        cov = cov_all & ok[None, :]
        need = block * block
        keep = ((cov_all.sum(1) == need) & (cov.sum(1) == need)) if full_only else (cov.sum(1) >= 2)
        out["n_pos_scenes"] = int(keep.sum())
        if out["n_pos_scenes"] >= min_pos_scenes:
            covf = cov.double()
            mean = (own * covf).sum(0) / covf.sum(0).clamp_min(1.0)                              # per-position mean over ALL covered positives
            var = (((own - mean[None, :]) ** 2) * covf).sum(0) / (covf.sum(0) - 1.0).clamp_min(1.0)
            resid = (own - mean[None, :]) * covf * w[None, :]
            s = resid.sum(1)[keep]
            nom = (covf * (w ** 2 * var)[None, :]).sum(1)[keep]
            if float(nom.sum()) > 0:
                out["vif_pos"] = float((s ** 2).sum() / nom.sum())
    return out


def difficulty_cv_from_vif(vif_pos: float, n_tokens: int, d: float, vif_null: float = 1.0) -> float:
    """Spread of a scene-level difficulty multiplier implied by the positive-scene VIF, for equal d (null-sd units, NOT VIF-corrected).

    z_u | kappa ~ N(kappa d, Sigma), kappa with mean 1 and coefficient of variation cv, Sigma equicorrelated with
    VIF_null = 1 + (n - 1) rho (VIF_null = 1: independent noise), n tokens, c = d^2 cv^2:
        VIF_pos = (VIF_null + n c) / (1 + c)   =>   cv^2 = (VIF_pos - VIF_null) / ((n - VIF_pos) d^2).
    vif_null below 1 is treated as 1 (sampling noise). Returns 0.0 when VIF_pos <= VIF_null (no evidence of heterogeneity) and NaN
    when the inputs are unusable (VIF_pos >= n means the tokens are fully dependent, which this model cannot express)."""
    if not (math.isfinite(vif_pos) and math.isfinite(d)) or d <= 0 or vif_pos >= n_tokens:
        return float("nan")
    v0 = max(vif_null, 1.0) if math.isfinite(vif_null) else 1.0
    if vif_pos <= v0:
        return 0.0
    return math.sqrt((vif_pos - v0) / ((n_tokens - vif_pos) * d * d))


def spillover(z: torch.Tensor, cover: torch.Tensor, label: torch.Tensor, tok: int, min_null: int = 100, min_tokens: int = 200) -> Dict[str, float]:
    """Does target evidence leak onto tokens that do NOT overlap the target? (footprint check)

    The Bayes state says: under H_(b,c) only the units in the footprint F_b carry signal; every other unit is N(0, 1).
    A ViT window mixes all of its tokens, so the own-class score may rise on uncovered tokens too. For windows that
    contain part of the target, report the mean own-class standardised score at uncovered tokens relative to the same
    (token position, class) in scenes WITHOUT a target, in null-sd units:
      adjacent  uncovered tokens within one token (Chebyshev) of a covered token
      far       uncovered tokens with no covered token within one token
    z (n, T, C) standardised scores; cover (n, T) bool; label (n,) -1 = no target. NaN if too few samples."""
    n, T, C = z.shape
    has = label >= 0
    out = {"adjacent": float("nan"), "far": float("nan"), "n_adjacent": 0, "n_far": 0}
    if int((~has).sum()) < min_null or not bool(has.any()):
        return out
    mu_null = z[~has].mean(0)                                                          # (T, C) null-scene mean per position and class
    ref = mu_null[None].expand(n, T, C).gather(2, label.clamp_min(0)[:, None, None].expand(n, T, 1)).squeeze(2)
    delta = z.gather(2, label.clamp_min(0)[:, None, None].expand(n, T, 1)).squeeze(2) - ref   # (n, T) own-class shift vs target-free scenes
    near = F.max_pool2d(cover.float().reshape(n, 1, tok, tok), 3, stride=1, padding=1).reshape(n, T) > 0
    in_view = has & cover.any(1)                                                       # windows that contain part of the target
    adj = (near & ~cover) & in_view[:, None]
    far = (~near) & in_view[:, None]
    out["n_adjacent"], out["n_far"] = int(adj.sum()), int(far.sum())
    if out["n_adjacent"] >= min_tokens:
        out["adjacent"] = float(delta[adj].mean())
    if out["n_far"] >= min_tokens:
        out["far"] = float(delta[far].mean())
    return out


def region_means(m: torch.Tensor, n_pos: torch.Tensor) -> Dict[str, float]:
    """Positive-count-weighted means of a (tok, tok) map over the centre block and the outer ring."""
    tok = m.shape[0]
    a, b = tok // 4, tok - tok // 4
    centre = torch.zeros(tok, tok, dtype=torch.bool)
    centre[a:b, a:b] = True
    ring = torch.zeros(tok, tok, dtype=torch.bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True

    def wmean(mask):
        ok = mask & torch.isfinite(m)
        w = n_pos[ok]
        return float((m[ok] * w).sum() / w.sum()) if ok.any() and float(w.sum()) > 0 else float("nan")

    return {"centre": wmean(centre), "edge_ring": wmean(ring), "all": wmean(torch.ones(tok, tok, dtype=torch.bool))}


DEFAULT_THRESHOLDS = {
    "zroc_slope": (0.8, 1.25),            # equal-variance Gaussian evidence <=> slope 1
    "null_sd": (0.85, 1.2),
    "d_centre_min": 1.5,                  # centre-of-window detectability of the operating target size
    "native_over_gist_min": 1.5,          # looking closer must pay: d_native_centre / d_gist
    "rho_sum_max": 0.2,                   # independent-looks regime  -> "sum"
    "rho_best_min": 0.6,                  # nested deterministic views -> "best"
    "vif_independent_max": 1.25,          # within-look token dependence: VIF_null above this -> correct z, d by 1/sqrt(VIF_null); VIF_pos / VIF_null above this -> shared difficulty flagged
    "d_block_min": 2.0,                   # effective detectability of one clean look at the operating size: Phi(-2/2) = 15.9 % equal-prior error
}


def verdict(summary: Dict[str, float], thr: Optional[Dict] = None) -> Dict[str, object]:
    """Draft decision rule over the summary of one operating point. Thresholds are conventions to be
    frozen in a pre-registration BEFORE the measurement is run on the real checkpoint."""
    thr = DEFAULT_THRESHOLDS if thr is None else thr
    summary = {k: (float("nan") if v is None else v) for k, v in summary.items()}    # strict JSON stores NaN as null; both mean 'not measured'
    lo, hi = thr["zroc_slope"]
    nlo, nhi = thr["null_sd"]
    slope = summary["zroc_slope_median"]
    if not math.isfinite(slope):                                   # d' too high for the z-ROC: use sd_null / sd_pos (equal to the slope)
        slope = summary["null_sd_median"] / max(summary["pos_sd_median"], 1e-6)
    adequate = lo <= slope <= hi and nlo <= summary["null_sd_median"] <= nhi
    # within-look token dependence (measured, not a pass/fail item). Noise dependence under H_0 (VIF_null) is corrected by
    # 1/sqrt(VIF) (exact for equicorrelated tokens). Dependence only on TARGET scenes (VIF_pos well above VIF_null) is shared image
    # difficulty: shrinking does not fix it (simulated), a scale mixture does, so it is flagged, not corrected.

    def finite_or_none(x):
        return float(x) if (x is not None and math.isfinite(float(x))) else None

    vn, vp = finite_or_none(summary.get("vif_null_op")), finite_or_none(summary.get("vif_pos_op"))
    n_op = finite_or_none(summary.get("op_block_tokens"))                 # tokens of one operating-size target in a window
    n_op = n_op if (n_op is not None and n_op > 0) else None
    # FAIL CLOSED: the block-level detectability needs both VIF_null and the block size. A missing or non-finite one
    # used to be read as "independent tokens" / "no block test", which can only help a pass (a reviewer showed d_block
    # 7.2 instead of 2.08 for d_centre 1.8 and VIF 12). Now `informative` fails and the quantity is listed as unmeasured;
    # the aggregate (scripts/rhan_im/gate1_prereg.py) reports INCONCLUSIVE rather than a verdict on the trunk.
    unmeasured = [name for name, val in (("vif_null_op", vn), ("op_block_tokens", n_op)) if val is None]
    noise_dependent = vn is not None and vn > thr["vif_independent_max"]
    token_rule = "correct z and d by 1/sqrt(VIF_null)" if noise_dependent else ("independent" if vn is not None else "unmeasured")
    vif_used = vn if noise_dependent else 1.0
    shared_assessed = vn is not None and vp is not None                    # unassessed is not the same as absent
    shared_difficulty = shared_assessed and vp / max(vn, 1.0) > thr["vif_independent_max"]
    d_c = summary["d_native_centre"]
    d_block = d_c * math.sqrt(n_op / vif_used) if (not unmeasured and math.isfinite(d_c)) else float("nan")
    informative = bool(math.isfinite(d_c) and d_c >= thr["d_centre_min"] and math.isfinite(d_block) and d_block >= thr["d_block_min"])
    # comparator for "looking closer pays": the BEST case gist (pooled mean over overlapped tokens, or tokens fully covered
    # by the target, whichever is larger); NaN if neither is measurable, which counts as "does not pay"
    gists = [v for v in (summary["d_gist"], summary.get("d_gist_full")) if v is not None and math.isfinite(v)]
    dg = max(gists) if gists else float("nan")
    ratio = summary["d_native_centre"] / max(dg, 1e-6) if gists else float("nan")
    pays = bool(gists) and ratio >= thr["native_over_gist_min"]
    rho = summary["rho_null"]
    rule = "sum" if rho <= thr["rho_sum_max"] else ("best" if rho >= thr["rho_best_min"] else "best (conservative; sum is not justified)")
    return {
        "sdt_adequate": bool(adequate), "informative": informative, "looking_closer_pays": bool(pays),
        "native_over_gist": ratio, "combination_rule": rule,
        "token_evidence": token_rule, "vif_used": vif_used, "vif_measured": vn is not None, "shared_difficulty": bool(shared_difficulty),
        "shared_difficulty_assessed": bool(shared_assessed),
        "d_block_effective": d_block, "unmeasured": unmeasured,
        "pass_all": bool(adequate and informative and pays),
    }

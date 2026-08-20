"""
dashboards/lens_app.py — the Lens dashboard (real checkpoints, no mocks).
==========================================================================

Streamlit front-end for the read-only introspection layer (rhan_core/lens).
Loads any NOESIS / RHAN-Next checkpoint (A/B/C/D-matrix or local .pth) and
visualises its internal state on a single image, live, step by step:

  * foveal gaze trajectory overlaid on the input (the v11 diagnostic plot
    mapping), growing step by step
  * foveal crop at the current fixation vs the generative prior's
    reconstruction
  * Π_D (sensory precision) per step + β_dynamic readout
  * HPC prediction-error map heatmap (only when the checkpoint has HPC)
  * AIS halting gauge (per-step continuation probability)
  * final classification + confidence + ground truth
  * side-by-side "different eyes" mode: 2-4 checkpoints on the SAME image
    (TRADES baseline vs AIS-v1 vs HPC …), from the ABLATION_MATRIX registry

Run (no GPU required for a single image; the 4060 or CPU both work):

    streamlit run dashboards/lens_app.py

Reuse contract (nothing duplicated here): checkpoint config auto-detection +
loading -> eval_rhan.py; HF download/cache -> eval_rhan_v11.py; PGD +
STL-10 bounds/normalization -> eval_full_epsilon_sweep.py; ablation registry
-> rhan_core/ablation/matrix.py. This file is UI only.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Repo-root importability (same convention as cognitive_vision_lab/app.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "phase1_training"),
           os.path.join(_ROOT, "phase2_attacks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

# Reused canonical logic (imported, never reimplemented):
import eval_full_epsilon_sweep as _sweep  # noqa: E402  (PGD + MEAN/STD + loader)
from eval_rhan_v11 import download_checkpoint_from_hf  # noqa: E402
from dataset_stl10 import STL10_CLASSES  # noqa: E402

from rhan_core.ablation.matrix import ABLATION_MATRIX, matrix_keys  # noqa: E402
from rhan_core.lens.capture import ForwardResult, StepCapture  # noqa: E402
from rhan_core.lens.session import LensSession  # noqa: E402

CKPT_DIR = os.path.join(_ROOT, "checkpoints")

st.set_page_config(
    page_title="RHAN-Next Lens — introspect any checkpoint",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint options (ABLATION_MATRIX + local scan + HF fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _local_checkpoints() -> List[Tuple[str, str, str]]:
    """(label, path, source) for every .pth under checkpoints/."""
    out = []
    if not os.path.isdir(CKPT_DIR):
        return out
    for fn in sorted(os.listdir(CKPT_DIR)):
        if not fn.endswith(".pth") or "Zone.Identifier" in fn:
            continue
        full = os.path.join(CKPT_DIR, fn)
        out.append((f"[local] {fn}", full, "local"))
    return out


def checkpoint_options() -> List[Dict]:
    """Selector entries: matrix registry first, then the local scan.

    C_hpc_only has no checkpoint yet (Stage 2 not finished) — it is listed
    with a status note but is not runnable, and nothing blocks on it.
    """
    opts: List[Dict] = []
    seen: set = set()
    for key in matrix_keys():
        e = ABLATION_MATRIX[key]
        path = e.get("checkpoint")
        label = f"[{key}] {e['label']} — {e['status']}"
        if path is None:
            opts.append({"label": f"{label} (not trained yet)", "path": None,
                         "arch": e.get("arch"), "source": "matrix-none",
                         "key": key})
            continue
        full = path if os.path.isabs(path) else os.path.join(_ROOT, path)
        if os.path.exists(full):
            seen.add(os.path.abspath(full))
            opts.append({"label": label, "path": full, "arch": e.get("arch"),
                         "source": "matrix-local", "key": key})
        else:
            opts.append({"label": f"{label} (HF)", "path": full,
                         "arch": e.get("arch"), "source": "matrix-hf",
                         "key": key})
    for label, full, src in _local_checkpoints():
        if os.path.abspath(full) in seen:
            continue
        opts.append({"label": label, "path": full, "arch": None,
                     "source": src, "key": None})
    return opts


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers (normalization constants come from the canonical sweep module)
# ─────────────────────────────────────────────────────────────────────────────
def denorm(t: torch.Tensor) -> np.ndarray:
    """Normalized (C,H,W) tensor -> (H,W,3) display array in [0,1]."""
    mean = _sweep.MEAN.squeeze(0).reshape(3).numpy()
    std = _sweep.STD.squeeze(0).reshape(3).numpy()
    a = t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(a * std + mean, 0.0, 1.0)


def normalize_pil(pil_img) -> torch.Tensor:
    """PIL image -> normalized (3, 96, 96) tensor (mirrors load_test_samples)."""
    img = pil_img.convert("RGB").resize((96, 96))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - _sweep.MEAN.squeeze(0)) / _sweep.STD.squeeze(0)


@st.cache_data(show_spinner="Loading STL-10 test batch…")
def load_test_batch(n: int, seed: int = 42):
    """Reuses the canonical sweep loader (HF 'mteb/stl10', normalized)."""
    return _sweep.load_test_samples(n_samples=n, seed=seed)


@st.cache_resource(show_spinner="Loading checkpoint…")
def get_session(path: str, device: str) -> LensSession:
    return LensSession(path, arch=None, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders (gaze overlay = the exact v11 diagnostic mapping)
# ─────────────────────────────────────────────────────────────────────────────
def fig_gaze_overlay(x_img: torch.Tensor, captures: List[StepCapture],
                     upto: int) -> plt.Figure:
    """Input image with the gaze trajectory overlaid up to step `upto`.

    Pixel mapping (a+1)*48 and the yellow 'T=i' annotation style are exactly
    the ones eval_rhan_v11 used for v11_foraging_trajectory.png.
    """
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(denorm(x_img))
    pts = [(c.gaze_x_px, c.gaze_y_px) for c in captures[:upto + 1]
           if c.gaze_x_px is not None]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color="yellow", markersize=8, linewidth=2,
                label="Gaze trajectory")
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x + 2, y - 2, f"T={i}", color="yellow", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Gaze trajectory — up to step {upto}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_recon(cap: StepCapture) -> plt.Figure:
    """Generative prior: actual foveal crop vs predicted crop."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    if cap.foveal_crop is not None:
        a1.imshow(denorm(cap.foveal_crop))
    a1.set_title(f"Actual crop @ T={cap.step}", fontsize=9)
    a1.axis("off")
    if cap.predicted_crop is not None:
        a2.imshow(denorm(cap.predicted_crop))
    a2.set_title("Predicted (generative prior)", fontsize=9)
    a2.axis("off")
    fig.tight_layout()
    return fig


def fig_hpc(cap: StepCapture) -> Optional[plt.Figure]:
    """HPC |prediction − target| edge-map error heatmap (Stage-2 style)."""
    if cap.hpc_error_map is None:
        return None
    m = cap.hpc_error_map.detach().cpu().numpy().squeeze()
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    im = ax.imshow(m, cmap="inferno")
    ax.set_title(f"HPC error map @ T={cap.step}\nerr={cap.hpc_error:.4f}",
                 fontsize=9)
    ax.axis("off")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_pi_d_bars(captures: List[StepCapture], upto: int) -> plt.Figure:
    """Π_D per step (bars) + β_dynamic (line), up to the current step."""
    steps = [c.step for c in captures[:upto + 1] if c.pi_d is not None]
    pi = [c.pi_d for c in captures[:upto + 1] if c.pi_d is not None]
    beta = [c.beta_dynamic for c in captures[:upto + 1] if c.beta_dynamic is not None]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    if steps:
        ax.bar(steps, pi, color="#2563EB", alpha=0.85, label="Π_D")
        ax.set_xlabel("Recurrent step")
        ax.set_ylabel("Π_D (sensory precision)", color="#2563EB")
        ax.set_ylim(0, 1)
        ax2 = ax.twinx()
        ax2.plot(steps, beta, "-o", color="#e11d48", label="β_dynamic")
        ax2.set_ylabel("β_dynamic", color="#e11d48")
        ax2.set_ylim(0, max(beta) * 1.2 if beta else 1.0)
        ax.set_title("Π_D per step + β_dynamic", fontsize=10)
    else:
        ax.text(0.5, 0.5, "no foraging state (static model)",
                ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_class_probs(result: ForwardResult) -> plt.Figure:
    """Final classifier confidence per class."""
    names = STL10_CLASSES
    probs = result.class_probs.detach().cpu().numpy()
    order = np.argsort(probs)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh([names[i] for i in order], probs[order], color="#10b981")
    ax.set_xlim(0, 1)
    ax.set_title("Classifier confidence per class", fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Panel renderers
# ─────────────────────────────────────────────────────────────────────────────
def render_step_metrics(cap: StepCapture) -> None:
    m = st.columns(6)
    m[0].metric("Π_D", f"{cap.pi_d:.3f}" if cap.pi_d is not None else "—")
    m[1].metric("β_dynamic", f"{cap.beta_dynamic:.3f}" if cap.beta_dynamic is not None else "—")
    m[2].metric("Error mag", f"{cap.error_mag:.3f}" if cap.error_mag is not None else "—")
    m[3].metric("Gate α", f"{cap.gate_alpha:.3f}" if cap.gate_alpha is not None else "—")
    m[4].metric("Uncertainty", f"{cap.uncertainty:.3f}" if cap.uncertainty is not None else "—")
    m[5].metric("Recon MSE", f"{cap.recon_error:.4f}" if cap.recon_error is not None else "—")


def render_step_panel(cap: StepCapture, x_img: torch.Tensor,
                      captures: List[StepCapture]) -> None:
    render_step_metrics(cap)

    c1, c2, c3 = st.columns([1.15, 1.15, 1.0])
    with c1:
        st.pyplot(fig_gaze_overlay(x_img, captures, cap.step))
    with c2:
        if cap.foveal_crop is not None or cap.predicted_crop is not None:
            st.pyplot(fig_recon(cap))
        else:
            st.caption("Generative prior not present in this checkpoint.")
    with c3:
        hpc_fig = fig_hpc(cap)
        if hpc_fig is not None:
            st.pyplot(hpc_fig)
        else:
            st.caption("No HPC head in this checkpoint — panel omitted.")
        if cap.continuation is not None:
            st.markdown("**AIS halting gauge** — per-step continuation "
                        "(~1 keep gathering, ~0 stop)")
            st.progress(min(max(cap.continuation, 0.0), 1.0),
                        text=f"continuation = {cap.continuation:.3f}")
            st.caption(f"soft-halt triggered: {'✓' if cap.halted else '✗'}")
        else:
            st.caption("No AIS halting in this checkpoint — gauge omitted.")

    st.pyplot(fig_pi_d_bars(captures, cap.step))


def render_final(result: ForwardResult, gt_name: Optional[str]) -> None:
    st.markdown("---")
    st.subheader(f"Final — {result.label}")
    c1, c2, c3, c4 = st.columns(4)
    pred_name = STL10_CLASSES[result.top_class] \
        if result.top_class < len(STL10_CLASSES) else str(result.top_class)
    c1.metric("Prediction", pred_name,
              delta=f"{result.top_confidence * 100:.1f}% confidence")
    c2.metric("Ground truth", gt_name or "unknown",
              delta=("correct ✓" if result.correct else
                     ("wrong ✗" if result.correct is False else None)))
    c3.metric("Evidence steps", f"{result.steps_effective:.2f} / "
                                f"{result.steps_total}")
    c4.metric("Frac. halted", f"{result.frac_halting:.2f}"
              if result.frac_halting is not None else "—")
    st.pyplot(fig_class_probs(result))
    st.caption(f"arch={result.arch} · device={result.device} · "
               f"config={result.config_summary or 'n/a'} · "
               f"AIS={'on' if result.ais_active else 'off'} · "
               f"HPC={'on' if result.hpc_active else 'off'}")


def run_checkpoint(session: LensSession, x_img: torch.Tensor,
                   gt: Optional[int]) -> Tuple[ForwardResult, List[StepCapture]]:
    caps: List[StepCapture] = []
    result: Optional[ForwardResult] = None
    for item in session.run(x_img, step_by_step=True, ground_truth=gt):
        if isinstance(item, StepCapture):
            caps.append(item)
        else:
            result = item
    assert result is not None
    return result, caps


def render_drilldown(session: LensSession, x_img: torch.Tensor,
                     gt: Optional[int]) -> None:
    """Single-checkpoint mode: scrub / animate through the steps."""
    st.markdown(f"### 🔬 {session.label}")
    with st.spinner("Running forward pass…"):
        result, caps = run_checkpoint(session, x_img, gt)
    gt_name = STL10_CLASSES[gt] if gt is not None else None

    if not caps:
        st.warning("No captures produced.")
        render_final(result, gt_name)
        return

    if len(caps) == 1 and not caps[0].has_pillars:
        st.caption("Static (feed-forward) checkpoint — no recurrent foraging "
                   "state to step through; showing the final panel only.")
        render_final(result, gt_name)
        return

    st.caption(f"{len(caps)} recurrent steps — scrub or animate.")
    step_idx = st.slider("Step", 0, len(caps) - 1, 0, key=f"slider_{session.label}")
    if st.button("▶ Animate steps", key=f"anim_{session.label}"):
        holder = st.empty()
        for i in range(len(caps)):
            with holder.container():
                st.markdown(f"**Step {i}**")
                render_step_panel(caps[i], x_img, caps)
            time.sleep(0.45)
        holder.empty()
    render_step_panel(caps[step_idx], x_img, caps)
    render_final(result, gt_name)


def render_side_by_side(sessions: List[LensSession], x_img: torch.Tensor,
                        gt: Optional[int]) -> None:
    """2-4 checkpoints on the SAME image — 'looking through different eyes'."""
    st.markdown("### 👀 Side-by-side — same image, different eyes")
    results = []
    for s in sessions:
        with st.spinner(f"Running {s.label}…"):
            results.append(run_checkpoint(s, x_img, gt))

    gt_name = STL10_CLASSES[gt] if gt is not None else None
    per_row = 2
    for row in range(0, len(sessions), per_row):
        cols = st.columns(per_row)
        for col, (s, (res, caps)) in zip(cols, list(zip(sessions, results))[row:row + per_row]):
            with col:
                last = caps[-1] if caps else None
                st.pyplot(fig_gaze_overlay(x_img, caps, len(caps) - 1))
                pred_name = STL10_CLASSES[res.top_class] \
                    if res.top_class < len(STL10_CLASSES) else str(res.top_class)
                st.markdown(f"**{s.label}**")
                c1, c2 = st.columns(2)
                c1.metric("Pred", pred_name,
                          delta=f"{res.top_confidence * 100:.1f}%")
                c2.metric("Correct", "✓" if res.correct else
                          ("✗" if res.correct is False else "—"))
                m1, m2, m3 = st.columns(3)
                m1.metric("Steps", f"{res.steps_effective:.1f}/{res.steps_total}")
                m2.metric("Halt frac", f"{res.frac_halting:.2f}"
                          if res.frac_halting is not None else "—")
                m3.metric("HPC err", f"{last.hpc_error:.4f}"
                          if last is not None and last.hpc_error is not None
                          else "—")

    st.markdown("### Final panels")
    for s, (res, caps) in zip(sessions, results):
        with st.expander(f"Final — {s.label}"):
            render_final(res, gt_name)


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
def render() -> None:
    st.title("🔬 RHAN-Next Lens")
    st.caption("Read-only introspection: load any NOESIS/RHAN-Next checkpoint "
               "and watch its internal state on ONE image, step by step. "
               "No training, no checkpoint writes — hooks only.")

    # ── Sidebar: device + checkpoint selection ───────────────────────────────
    device = st.sidebar.radio("Device", ["cpu", "cuda"], horizontal=True,
                              help="Single-image inference is CPU-friendly "
                                   "for STL-10-sized inputs.")
    opts = checkpoint_options()
    runnable = [o for o in opts if o["path"] is not None]
    labels = [o["label"] for o in runnable]
    sel = st.sidebar.multiselect(
        "Checkpoints (1 = drill-down, 2-4 = side-by-side)",
        labels,
        default=[labels[0]] if labels else [],
    )
    not_ready = [o for o in opts if o["path"] is None]
    if not_ready:
        st.sidebar.caption("Skipped (no checkpoint yet): "
                           + ", ".join(o["label"] for o in not_ready))
    hf_missing = [o for o in runnable if o["source"] == "matrix-hf"
                  and not os.path.exists(o["path"])]
    if hf_missing:
        st.sidebar.caption("Entries marked (HF) are downloaded from "
                           "HuggingFace automatically when selected.")

    if not sel:
        st.info("Select at least one checkpoint from the sidebar.")
        return

    sessions: List[LensSession] = []
    for o in runnable:
        if o["label"] in sel:
            try:
                sessions.append(get_session(o["path"], device))
            except Exception as e:  # surface load failures per checkpoint
                st.error(f"Failed to load {o['label']}: {e}")

    if not sessions:
        st.error("No checkpoints could be loaded.")
        return

    # ── Image source: upload OR STL-10 test set ──────────────────────────────
    st.markdown("### 1 · Image")
    src = st.radio("Source", ["STL-10 test set", "Upload"], horizontal=True)

    x_img: Optional[torch.Tensor] = None
    gt: Optional[int] = None
    if src == "Upload":
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is None:
            st.info("Upload an image to continue.")
            return
        from PIL import Image
        x_img = normalize_pil(Image.open(up))
        st.image(denorm(x_img), caption="Upload (normalized)", width=220)
    else:
        try:
            xs, ys = load_test_batch(12, seed=42)
        except Exception as e:
            st.error(f"Could not load the STL-10 test set (network/datasets "
                     f"lib needed): {e}")
            return
        idx = st.selectbox("Test image", range(len(xs)),
                           format_func=lambda i:
                           f"#{i} — {STL10_CLASSES[int(ys[i])]}")
        x_img = xs[idx]
        gt = int(ys[idx])
        c1, c2 = st.columns([1, 3])
        with c1:
            st.image(denorm(x_img), caption=f"GT: {STL10_CLASSES[gt]}",
                     width=220)
        with c2:
            st.caption("Normalized STL-10 tensor ready.")

    assert x_img is not None

    # ── PGD (delegated: canonical norm-space attack, per-checkpoint model) ──
    st.markdown("### 2 · Adversarial view (optional)")
    eps = st.slider("ε (norm space, Finding-17 convention)", 0.0, 0.2, 0.094,
                    step=0.001)
    pgd_steps = st.slider("PGD steps", 5, 50, 20)
    adv_images: Dict[str, torch.Tensor] = st.session_state.setdefault(
        "lens_adv", {})
    if st.button("⚔ Generate PGD-perturbed image(s)", type="primary"):
        for s in sessions:
            with st.spinner(f"Attacking {s.label} ({eps}, {pgd_steps} steps)…"):
                adv_images[s.checkpoint_path] = s.pgd(
                    x_img, eps=eps, steps=pgd_steps)
        st.success(f"PGD done for {len(sessions)} checkpoint(s).")

    for o in runnable:
        if o["label"] in sel and o["path"] in adv_images:
            adv = adv_images[o["path"]]
            cc1, cc2 = st.columns(2)
            cc1.image(denorm(x_img), caption="clean", width=200)
            cc2.image(denorm(adv[0]), caption=f"PGD ε={eps}", width=200)

    # ── Per-checkpoint rendering ─────────────────────────────────────────────
    st.markdown("### 3 · Internal state")
    active = [s for s in sessions]  # use the clean image for the run
    if len(active) == 1:
        render_drilldown(active[0], x_img, gt)
    else:
        render_side_by_side(active, x_img, gt)

    # ── Belief drift analysis ──────────────────────────────────────────────
    st.markdown("### 4 · Belief drift (clean vs adversarial)")
    st.caption(
        "Falsifiable claim: if HPC stabilises representation, "
        "belief_drift(HPC-only) < belief_drift(baseline) at matched "
        "recurrent steps — the HPC error-correction loop pulls the "
        "adversarial belief trajectory back toward the clean one."
    )
    drift_eps = st.slider(
        "Drift ε (norm space)", 0.0, 0.2, 0.094, step=0.001,
        key="drift_eps",
    )
    drift_pgd_steps = st.slider(
        "Drift PGD steps", 5, 50, 50, key="drift_pgd_steps",
    )
    if st.button("📊 Compute belief drift", type="primary",
                 key="btn_drift"):
        from rhan_core.lens.session import run_captures
        from rhan_core.lens.capture import compute_belief_drift

        drift_results: Dict[str, Any] = {}
        for sess in sessions:
            with st.spinner(f"Running clean + PGD on {sess.label}…"):
                _, clean_caps = run_captures(sess, x_img, gt)
                adv_img = sess.pgd(x_img, eps=drift_eps,
                                   steps=drift_pgd_steps)
                _, adv_caps = run_captures(sess, adv_img[0], gt)
                drift_rows = compute_belief_drift(clean_caps, adv_caps)
                drift_results[sess.label] = {
                    "clean_caps": clean_caps,
                    "adv_caps": adv_caps,
                    "rows": drift_rows,
                }

        if not drift_results:
            st.warning("No checkpoints loaded.")
        else:
            # ── Per-step drift table ────────────────────────────────────────
            n_steps = max(len(r["rows"]) for r in drift_results.values())
            st.markdown("**Per-step belief drift (cosine distance + L2)**")
            header_cols = st.columns(len(drift_results) + 1)
            header_cols[0].markdown("**Step**")
            for ci, (label, d) in enumerate(drift_results.items(), 1):
                has_b = any(r["has_belief"] for r in d["rows"])
                tag = "✓" if has_b else "✗ (no belief)"
                header_cols[ci].markdown(f"**{label}** {tag}")

            for t in range(n_steps):
                cols = st.columns(len(drift_results) + 1)
                cols[0].markdown(f"`T={t}`")
                for ci, (label, d) in enumerate(drift_results.items(), 1):
                    if t < len(d["rows"]):
                        r = d["rows"][t]
                        if r["drift_cosine"] is not None:
                            cos_v = r["drift_cosine"]
                            l2_v = r["drift_l2"]
                            pi_c = r["pi_d_clean"]
                            cols[ci].markdown(
                                f"cos={cos_v:.4f}<br>"
                                f"L2={l2_v:.2f}<br>"
                                f"π_clean={pi_c:.3f}",
                                unsafe_allow_html=True,
                            )
                        else:
                            cols[ci].markdown("—")
                    else:
                        cols[ci].markdown("—")

            # ── Summary row ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Aggregate drift (mean cosine distance across "
                        "all steps)**")
            sum_cols = st.columns(len(drift_results))
            for ci, (label, d) in enumerate(drift_results.items()):
                cos_all = [r["drift_cosine"] for r in d["rows"]
                           if r["drift_cosine"] is not None]
                mean_cos = float(np.mean(cos_all)) if cos_all else None
                l2_all = [r["drift_l2"] for r in d["rows"]
                          if r["drift_l2"] is not None]
                mean_l2 = float(np.mean(l2_all)) if l2_all else None
                has_b = any(r["has_belief"] for r in d["rows"])
                if mean_cos is not None:
                    sum_cols[ci].metric(
                        label,
                        f"cos={mean_cos:.4f}",
                        delta=f"L2={mean_l2:.2f}",
                        help=f"{'Has per-step belief' if has_b else 'No per-step belief (single-pass model)'}. "
                             f"Mean cosine distance and mean L2 across all recurrent steps.",
                    )
                else:
                    sum_cols[ci].metric(label, "—",
                                        help="No belief state (static feed-forward model).")

            # ── Visual comparison bar chart ──────────────────────────────────
            has_any_belief = any(
                any(r["has_belief"] for r in d["rows"])
                for d in drift_results.values()
            )
            if has_any_belief:
                fig_drift, ax_d = plt.subplots(figsize=(7, 3.2))
                labels_list = list(drift_results.keys())
                cos_means = []
                for label in labels_list:
                    cos_all = [r["drift_cosine"] for r in drift_results[label]["rows"]
                               if r["drift_cosine"] is not None]
                    cos_means.append(float(np.mean(cos_all)) if cos_all else 0)
                colors = ["#6b7280", "#2563EB", "#10b981"][:len(labels_list)]
                ax_d.barh(labels_list, cos_means, color=colors, alpha=0.85)
                ax_d.set_xlabel("Mean cosine distance (clean vs adv)")
                ax_d.set_title("Belief drift: lower = more stable")
                ax_d.invert_yaxis()
                fig_drift.tight_layout()
                st.pyplot(fig_drift)
            else:
                st.info("None of the selected checkpoints have per-step "
                        "belief state (all are static feed-forward models).")

            # ── Step-scrubber: side-by-side clean vs adv at each step ────────
            if has_any_belief:
                st.markdown("---")
                st.markdown("**Step-by-step belief state comparison** "
                            "(clean vs adversarial, per checkpoint)")
                max_caps = max(
                    max((len(d["clean_caps"]) for d in drift_results.values()), default=0),
                    max((len(d["adv_caps"]) for d in drift_results.values()), default=0),
                )
                if max_caps > 1:
                    sel_step = st.slider(
                        "Scrub step", 0, max_caps - 1, 0,
                        key="drift_step_scrub",
                    )
                else:
                    sel_step = 0

                for label, d in drift_results.items():
                    caps_c = d["clean_caps"]
                    caps_a = d["adv_caps"]
                    has_b = any(r["has_belief"] for r in d["rows"])
                    if not has_b or sel_step >= len(d["rows"]):
                        continue
                    r = d["rows"][sel_step]
                    if r["drift_cosine"] is None:
                        continue

                    st.markdown(f"**{label}** — step `T={sel_step}`")
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.pyplot(fig_gaze_overlay(
                            x_img, caps_c, min(sel_step, len(caps_c) - 1)))
                        st.caption("Clean gaze")
                    with sc2:
                        if sel_step < len(caps_a):
                            st.pyplot(fig_gaze_overlay(
                                x_img, caps_a, min(sel_step, len(caps_a) - 1)))
                        st.caption("Adversarial gaze")
                    with sc3:
                        st.metric("Cosine distance",
                                  f"{r['drift_cosine']:.4f}")
                        st.metric("L2 distance",
                                  f"{r['drift_l2']:.2f}")
                        st.metric("Π_D clean",
                                  f"{r['pi_d_clean']:.3f}"
                                  if r['pi_d_clean'] is not None else "—")
                        st.metric("Π_D adv",
                                  f"{r['pi_d_adv']:.3f}"
                                  if r['pi_d_adv'] is not None else "—")

    st.markdown("---")
    st.caption("Lens is read-only: it never writes a checkpoint, never "
               "fine-tunes, and never touches the resume-gate system.")


if st.runtime.exists():
    render()

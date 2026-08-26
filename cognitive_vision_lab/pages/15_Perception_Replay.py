"""15 — Perception Replay: watch RHAN-NX D perceive an image step-by-step.

Uses the real LensSession + StepCapture infrastructure from rhan_core/lens/
to instrument the actual model forward pass.  Every visual element is backed
by tensors from the real D checkpoint (rhan_next_ais_hpc_best.pth).

Run with:
    streamlit run cognitive_vision_lab/app.py
Then navigate to "Perception Replay" in the sidebar.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from cognitive_vision_lab.config import (
    CHECKPOINTS_DIR,
    STL10_CLASSES,
    STL10_MEAN,
    STL10_STD,
    STL10_SIZE,
)
from cognitive_vision_lab.components.layout import footer, hero, section

# ── STL-10 denormalization ──────────────────────────────────────────────────
_MEAN_3 = torch.tensor(STL10_MEAN).view(3, 1, 1)   # (3,1,1) for (C,H,W)
_STD_3 = torch.tensor(STL10_STD).view(3, 1, 1)     # (3,1,1) for (C,H,W)
_MEAN_4 = torch.tensor(STL10_MEAN).view(1, 3, 1, 1)  # (1,3,1,1) for (B,C,H,W)
_STD_4 = torch.tensor(STL10_STD).view(1, 3, 1, 1)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised -> (H,W,3) uint8 numpy."""
    img = (t.cpu().float() * _STD_3 + _MEAN_3).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _denorm_batch(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) -> (H,W,3) uint8."""
    return _denorm(t[0])


# ── Image loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_stl10_test(n: int = 200, seed: int = 42) -> list[tuple[Image.Image, int]]:
    """Load n STL-10 test images (cached)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("mteb/stl10", split="test", streaming=True)
        ds = ds.shuffle(seed=seed)
        items = []
        for i, item in enumerate(ds):
            if i >= n:
                break
            items.append((item["image"].convert("RGB"), item["label"]))
        return items
    except Exception:
        return []


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> (1,3,96,96) normalised tensor."""
    img = img.resize((STL10_SIZE, STL10_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 96, 96)
    return ((t - _MEAN_3) / _STD_3).unsqueeze(0)  # (1, 3, 96, 96)


# ── Session state helpers ───────────────────────────────────────────────────
def _get_session():
    """Return (or build) the LensSession, cached in Streamlit session state."""
    from rhan_core.lens.session import LensSession

    ckpt_name = st.session_state.get("pr_checkpoint", "rhan_next_ais_hpc_best.pth")
    device = st.session_state.get("pr_device", "cpu")
    cache_key = f"lens_session_{ckpt_name}_{device}"

    if cache_key not in st.session_state:
        ckpt_path = str(CHECKPOINTS_DIR / ckpt_name)
        with st.spinner(f"Loading checkpoint {ckpt_name}…"):
            sess = LensSession(ckpt_path, device=device, label=ckpt_name)
        st.session_state[cache_key] = sess

    return st.session_state[cache_key]


def _run_inference(sess, image_tensor: torch.Tensor, gt: int | None = None):
    """Run the model and return (ForwardResult, list[StepCapture])."""
    from rhan_core.lens.capture import ForwardResult, StepCapture

    caps = []
    result = None
    for item in sess.run(image_tensor, step_by_step=True, ground_truth=gt):
        if isinstance(item, StepCapture):
            caps.append(item)
        else:
            result = item
    return result, caps


def _run_pgd(sess, image_tensor: torch.Tensor, eps: float = 0.094, steps: int = 100):
    """Generate PGD adversarial example."""
    return sess.pgd(image_tensor, eps=eps, steps=steps)


# ── Visualization helpers ───────────────────────────────────────────────────
def _gaze_heatmap(caps, size: int = 96) -> list[np.ndarray]:
    """Build a gaze accumulation heatmap per step."""
    heatmaps = []
    acc = np.zeros((size, size), dtype=np.float32)
    for cap in caps:
        if cap.gaze_x_px is not None and cap.gaze_y_px is not None:
            cx = int(np.clip(round(cap.gaze_x_px), 0, size - 1))
            cy = int(np.clip(round(cap.gaze_y_px), 0, size - 1))
            # Gaussian blob
            yy, xx = np.mgrid[0:size, 0:size]
            sigma = 6.0
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            acc += blob
        # Normalize to [0, 1] and store cumulative
        if acc.max() > 0:
            heatmaps.append(acc / acc.max())
        else:
            heatmaps.append(acc)
    return heatmaps


def _hpc_error_map_image(cap) -> np.ndarray | None:
    """Convert an HPC error map tensor to a visualizable numpy array."""
    if cap.hpc_error_map is None:
        return None
    m = cap.hpc_error_map.float()
    if m.ndim == 3:
        m = m.mean(dim=0)  # average over channels
    m = m.cpu().numpy()
    # Normalize
    mn, mx = m.min(), m.max()
    if mx - mn > 1e-8:
        m = (m - mn) / (mx - mn)
    else:
        m = np.zeros_like(m)
    return m


def _hpc_prediction_image(cap) -> np.ndarray | None:
    """Convert HPC prediction tensor to a visualizable RGB image."""
    if cap.hpc_prediction is None:
        return None
    p = cap.hpc_prediction.float()
    if p.ndim == 4:
        p = p[0]  # (C, H, W) or (1, H, W)
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]  # (H, W) grayscale
    elif p.ndim == 3 and p.shape[0] == 3:
        p = p.permute(1, 2, 0)  # (H, W, 3)
    p = p.cpu().numpy()
    mn, mx = p.min(), p.max()
    if mx - mn > 1e-8:
        p = (p - mn) / (mx - mn)
    else:
        p = np.zeros_like(p)
    return (p * 255).astype(np.uint8)


def _foveal_image(cap) -> np.ndarray | None:
    """Extract the foveal crop from a StepCapture."""
    if cap.foveal_crop is None:
        return None
    return _denorm(cap.foveal_crop)


def _predicted_crop_image(cap) -> np.ndarray | None:
    """Extract the predicted crop from a StepCapture."""
    if cap.predicted_crop is None:
        return None
    return _denorm(cap.predicted_crop)


def _render_timeline(caps, current_step: int):
    """Render a step indicator: ● t1 → ● t2 → … → ◉ HALT."""
    parts = []
    for i, cap in enumerate(caps):
        is_current = i == current_step
        is_halted = cap.halted
        if is_halted:
            marker = "◉"
            label = "HALT"
        else:
            marker = "●" if not is_current else "▶"
            label = f"t{i + 1}"
        style = "font-weight:bold;" if is_current else ""
        parts.append(f'<span style="{style}">{marker} {label}</span>')
        if i < len(caps) - 1:
            parts.append('<span style="color:#888;">→</span>')
    st.markdown(" ".join(parts), unsafe_allow_html=True)


def _render_step_explain(cap, step_idx: int, total_steps: int):
    """Generate an 'Explain the Step' narrative from actual tensor values."""
    lines = []
    lines.append(f"**At timestep {step_idx + 1}** of {total_steps}:")

    if cap.gaze_x_norm is not None:
        lines.append(
            f"- AIS selected gaze position ({cap.gaze_x_norm:.3f}, {cap.gaze_y_norm:.3f}) "
            f"in normalized coordinates (pixel: {cap.gaze_x_px:.0f}, {cap.gaze_y_px:.0f})."
        )

    if cap.pi_d is not None:
        lines.append(f"- Sensory precision Π_D = {cap.pi_d:.4f}.")
        lines.append(f"- Uncertainty u = {cap.uncertainty:.4f}." if cap.uncertainty else "")

    if cap.gate_alpha is not None:
        lines.append(f"- Foveal/parafoveal gate α = {cap.gate_alpha:.4f}.")

    if cap.hpc_error is not None:
        lines.append(f"- HPC Level-1 prediction error = {cap.hpc_error:.6f}.")
        if cap.hpc_error_map is not None:
            lines.append(
                f"  Error map: min={cap.hpc_error_map.min():.6f}, "
                f"max={cap.hpc_error_map.max():.6f}, "
                f"std={cap.hpc_error_map.std():.6f}."
            )

    if cap.continuation is not None:
        lines.append(f"- Soft continuation weight = {cap.continuation:.4f}.")
        if cap.halted:
            lines.append(
                "- **HALTED** — continuation < 0.5, uncertainty below threshold."
            )
        else:
            lines.append("- Continued — model needs more evidence.")

    if cap.recon_error is not None:
        lines.append(f"- Reconstruction MSE = {cap.recon_error:.6f}.")

    return "\n".join(lines)


# ── Main page render ────────────────────────────────────────────────────────
def render() -> None:
    hero(
        "Perception Replay",
        "Watch RHAN-NX D (AIS-v1 + HPC) perceive an image step by step. "
        "Every value comes from the real model forward pass.",
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Perception Replay Controls")

        # Checkpoint selection
        ckpt_options = [
            "rhan_next_ais_hpc_best.pth",          # D: AIS + HPC (Stage 3-D)
            "rhan_next_ais_v1_halting_only_best.pth", # B: AIS-v1
            "rhan_next_hpc_only_best.pth",           # C: HPC-only
            "rhan_stl10_large_pseudolabel_best.pth", # A: TRADES baseline
        ]
        ckpt_labels = [
            "D: AIS + HPC ",
            "B: AIS-v1 halting-only",
            "C: HPC-only",
            "A: TRADES baseline",
        ]
        sel_idx = st.selectbox(
            "Checkpoint",
            range(len(ckpt_labels)),
            format_func=lambda i: ckpt_labels[i],
            index=0,
            key="pr_checkpoint_idx",
        )
        st.session_state["pr_checkpoint"] = ckpt_options[sel_idx]

        device = st.radio("Device", ["cpu", "cuda"], horizontal=True, key="pr_device")

        # Adversarial toggle
        adv_mode = st.toggle(
            "Adversarial comparison (PGD-100)",
            value=False,
            key="pr_adv_mode",
            help="Run clean and PGD-ε=0.094 side-by-side.",
        )
        if adv_mode:
            adv_eps = st.slider("PGD ε", 0.01, 0.15, 0.094, 0.001, key="pr_adv_eps")
            adv_steps = st.slider("PGD steps", 20, 200, 100, 10, key="pr_adv_steps")

    # ── Image source ─────────────────────────────────────────────────────────
    section("Input Image", "Upload an image or select from STL-10 test set.")

    src = st.radio("Image source", ["STL-10 test set", "Upload"], horizontal=True)

    image_pil = None
    gt_label = None

    if src == "STL-10 test set":
        stl10_samples = _load_stl10_test(200)
        if not stl10_samples:
            st.warning("STL-10 not available — using procedural fallback.")
            from cognitive_vision_lab.utils.io import procedural_sample
            image_pil = procedural_sample("airplane", 96)
        else:
            c1, c2 = st.columns([1, 3])
            with c1:
                sample_idx = st.slider("Sample index", 0, len(stl10_samples) - 1, 0)
            image_pil, gt_idx = stl10_samples[sample_idx]
            gt_label = STL10_CLASSES[gt_idx]
            st.caption(f"Ground truth: **{gt_label}** (index {gt_idx})")
    else:
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            image_pil = Image.open(up).convert("RGB")

    if image_pil is None:
        st.info("Select or upload an image to begin.")
        footer()
        return

    st.image(image_pil, caption="Input image", width=200)

    # ── Run inference ────────────────────────────────────────────────────────
    st.markdown("---")

    if st.button("▶  START PERCEPTION", type="primary", use_container_width=True):
        sess = _get_session()
        x = _pil_to_tensor(image_pil)

        with st.spinner("Running model forward pass…"):
            result_clean, caps_clean = _run_inference(sess, x, gt=gt_idx if gt_label else None)

        st.session_state["pr_result_clean"] = result_clean
        st.session_state["pr_caps_clean"] = caps_clean
        st.session_state["pr_image_tensor"] = x
        st.session_state["pr_image_pil"] = image_pil
        st.session_state["pr_gt_label"] = gt_label
        st.session_state["pr_gt_idx"] = gt_idx if gt_label else None

        # Adversarial run
        if adv_mode:
            with st.spinner(f"Generating PGD-{adv_steps} adversarial (ε={adv_eps})…"):
                x_adv = _run_pgd(sess, x, eps=adv_eps, steps=adv_steps)
            with st.spinner("Running adversarial forward pass…"):
                result_adv, caps_adv = _run_inference(
                    sess, x_adv[0].unsqueeze(0) if x_adv.ndim == 3 else x_adv,
                    gt=gt_idx if gt_label else None,
                )
            st.session_state["pr_result_adv"] = result_adv
            st.session_state["pr_caps_adv"] = caps_adv
            st.session_state["pr_adv_tensor"] = x_adv

        st.rerun()

    # ── Display results ──────────────────────────────────────────────────────
    if "pr_caps_clean" not in st.session_state:
        footer()
        return

    result = st.session_state["pr_result_clean"]
    caps = st.session_state["pr_caps_clean"]
    image_pil = st.session_state.get("pr_image_pil", image_pil)
    gt_label = st.session_state.get("pr_gt_label")

    if not caps:
        st.warning("Model produced no recurrent steps (static checkpoint?).")
        footer()
        return

    # ── Final Perception Report ──────────────────────────────────────────────
    section("Final Perception Report")

    class_probs = result.class_probs.numpy()
    top3_idx = np.argsort(class_probs)[-3:][::-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction", STL10_CLASSES[result.top_class])
    c2.metric("Confidence", f"{result.top_confidence * 100:.1f}%")
    c3.metric("Iterations", f"{result.steps_effective:.1f} / {result.steps_total}")
    if gt_label:
        correct = result.top_class == (st.session_state.get("pr_gt_idx") or -1)
        c4.metric("Correct?", "" if correct else "")

    # AIS metrics
    if caps[0].pi_d is not None:
        m1, m2, m3, m4 = st.columns(4)
        initial_entropy = caps[0].uncertainty if caps[0].uncertainty else 0
        final_cap = caps[-1]
        final_entropy = final_cap.uncertainty if final_cap.uncertainty else 0
        m1.metric("Initial Π_D", f"{caps[0].pi_d:.4f}")
        m2.metric("Final Π_D", f"{final_cap.pi_d:.4f}" if final_cap.pi_d else "—")
        m3.metric("Initial uncertainty", f"{initial_entropy:.4f}")
        m4.metric("Final uncertainty", f"{final_entropy:.4f}")

    # HPC metrics
    if caps[0].hpc_error is not None:
        m1, m2 = st.columns(2)
        m1.metric("Initial HPC error", f"{caps[0].hpc_error:.6f}")
        final_hpc = final_cap.hpc_error if final_cap.hpc_error else 0
        m2.metric("Final HPC error", f"{final_hpc:.6f}")

    # ── Timeline ─────────────────────────────────────────────────────────────
    section("Recurrent Perception Timeline")
    _render_timeline(caps, 0)

    # ── Playback controls ────────────────────────────────────────────────────
    st.markdown("---")

    if "pr_step" not in st.session_state:
        st.session_state["pr_step"] = 0

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 1, 3])
    with ctrl1:
        if st.button("⏮ Reset"):
            st.session_state["pr_step"] = 0
            st.rerun()
    with ctrl2:
        if st.button("◀ Prev") and st.session_state["pr_step"] > 0:
            st.session_state["pr_step"] -= 1
            st.rerun()
    with ctrl3:
        if st.button("▶ Play"):
            # Play through all steps
            for i in range(len(caps)):
                st.session_state["pr_step"] = i
            st.rerun()
    with ctrl4:
        if st.button("Next ▶") and st.session_state["pr_step"] < len(caps) - 1:
            st.session_state["pr_step"] += 1
            st.rerun()
    with ctrl5:
        step = st.slider(
            "Step",
            0,
            len(caps) - 1,
            st.session_state["pr_step"],
            key="pr_step_slider",
        )
        if step != st.session_state["pr_step"]:
            st.session_state["pr_step"] = step
            st.rerun()

    current_step = st.session_state["pr_step"]
    cap = caps[current_step]

    # ── Main visualization: 3-column layout ──────────────────────────────────
    col_left, col_center, col_right = st.columns(3)

    # LEFT: Input image with gaze overlay
    with col_left:
        st.markdown(f"**Input + Gaze** (step {current_step + 1}/{len(caps)})")
        img_np = np.array(image_pil.resize((96, 96)))
        # Build gaze overlay up to current step
        gheat = _gaze_heatmap(caps[: current_step + 1])
        overlay = img_np.copy().astype(np.float32)
        if gheat[-1].max() > 0:
            import matplotlib.cm as cm
            heatmap_rgba = cm.hot(gheat[-1])
            overlay = overlay * 0.6 + heatmap_rgba[..., :3] * 255 * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Draw gaze crosshair for current step
        if cap.gaze_x_px is not None:
            cx, cy = int(cap.gaze_x_px), int(cap.gaze_y_px)
            # Draw crosshair without cv2 dependency
            cross_size = 5
            if 0 <= cy < overlay.shape[0] and 0 <= cx < overlay.shape[1]:
                overlay[ max(0, cy-cross_size):cy, cx] = [0, 255, 0]
                overlay[ cy+1:min(overlay.shape[0], cy+cross_size+1), cx] = [0, 255, 0]
                overlay[ cy, max(0, cx-cross_size):cx] = [0, 255, 0]
                overlay[ cy, cx+1:min(overlay.shape[1], cx+cross_size+1)] = [0, 255, 0]

        st.image(overlay, use_container_width=True)

        if cap.gaze_x_norm is not None:
            st.caption(
                f"Gaze: ({cap.gaze_x_norm:.3f}, {cap.gaze_y_norm:.3f}) "
                f"= pixel ({cap.gaze_x_px:.0f}, {cap.gaze_y_px:.0f})"
            )

        # Foveal crop
        if cap.foveal_crop is not None:
            fov = _foveal_image(cap)
            if fov is not None:
                st.image(fov, caption="Foveal crop (actual)", width=120)

    # CENTER: HPC visualization
    with col_center:
        st.markdown("**HPC Level 1**")

        if cap.hpc_error is not None:
            # HPC target, prediction, error
            hpc_cols = st.columns(3)
            with hpc_cols[0]:
                st.caption("Target (edge map)")
                # The HPC target is derived from the foveal crop edges
                if cap.foveal_crop is not None:
                    fov_t = cap.foveal_crop.float().unsqueeze(0)
                    # Simple edge detection as target visualization
                    gray = fov_t.mean(dim=1, keepdim=True)
                    edges = F.conv2d(
                        gray,
                        torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float(),
                        padding=1,
                    )
                    edges_np = edges[0, 0].cpu().numpy()
                    edges_np = np.clip((edges_np - edges_np.min()) / (edges_np.max() - edges_np.min() + 1e-8), 0, 1)
                    st.image(edges_np, use_container_width=True, clamp=True)
                else:
                    st.caption("N/A")

            with hpc_cols[1]:
                st.caption("HPC prediction")
                pred_img = _hpc_prediction_image(cap)
                if pred_img is not None:
                    st.image(pred_img, use_container_width=True, clamp=True)
                else:
                    st.caption("N/A")

            with hpc_cols[2]:
                st.caption("Prediction error")
                err_img = _hpc_error_map_image(cap)
                if err_img is not None:
                    import matplotlib.cm as cm
                    err_color = (cm.hot(err_img) * 255).astype(np.uint8)
                    st.image(err_color, use_container_width=True)
                else:
                    st.caption("N/A")

            st.metric("HPC error", f"{cap.hpc_error:.6f}")
        else:
            st.info("HPC not active for this checkpoint.")

        # Predicted crop (generative prior)
        if cap.predicted_crop is not None:
            st.markdown("**Generative Prior**")
            pred_crop = _predicted_crop_image(cap)
            if pred_crop is not None:
                st.image(pred_crop, caption="Predicted crop (generative prior)", width=120)

    # RIGHT: Belief / prediction
    with col_right:
        st.markdown("**Belief & Prediction**")

        # Top-3 class probabilities at this step
        if result is not None:
            probs = class_probs
            top3 = np.argsort(probs)[-3:][::-1]

            bar_data = []
            for idx in top3:
                bar_data.append({
                    "class": STL10_CLASSES[idx],
                    "probability": float(probs[idx]),
                })

            import altair as alt
            bar_df = []
            for item in bar_data:
                bar_df.append({"class": item["class"], "probability": item["probability"]})

            chart = alt.Chart(alt.Data(values=bar_df)).mark_bar().encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                y=alt.Y("class:N", sort="-x", title=""),
                color=alt.value("#000"),
            ).properties(height=150)
            st.altair_chart(chart, use_container_width=True)

        # AIS metrics
        if cap.pi_d is not None:
            st.markdown("**AIS State**")
            ais_data = {
                "Π_D": f"{cap.pi_d:.4f}",
                "Uncertainty": f"{cap.uncertainty:.4f}" if cap.uncertainty else "—",
                "Gate α": f"{cap.gate_alpha:.4f}" if cap.gate_alpha else "—",
                "Continuation": f"{cap.continuation:.4f}" if cap.continuation else "—",
                "β_dynamic": f"{cap.beta_dynamic:.4f}" if cap.beta_dynamic else "—",
            }
            for k, v in ais_data.items():
                st.text(f"  {k}: {v}")

            # Halting decision
            if cap.halted:
                st.error(" HALTED")
            else:
                st.success("▶ Continuing")

    # ── Explain the Step ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(f" Explain Step {current_step + 1}", expanded=True):
        explanation = _render_step_explain(cap, current_step, len(caps))
        st.markdown(explanation)

    # ── Belief trajectory chart ──────────────────────────────────────────────
    section("Belief Trajectory Over Steps")
    traj_data = []
    for i, c in enumerate(caps):
        # Use the final result's probabilities (we don't have per-step probs
        # from the trajectory directly, but we have the belief vectors)
        traj_data.append({
            "step": i + 1,
            "Π_D": c.pi_d if c.pi_d is not None else 0,
            "uncertainty": c.uncertainty if c.uncertainty else 0,
            "continuation": c.continuation if c.continuation is not None else 1.0,
        })

    import pandas as pd
    traj_df = pd.DataFrame(traj_data)

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(
            traj_df.set_index("step")[["Π_D", "uncertainty"]],
            use_container_width=True,
        )
        st.caption("Π_D (sensory precision) and uncertainty over recurrent steps.")
    with c2:
        st.line_chart(
            traj_df.set_index("step")[["continuation"]],
            use_container_width=True,
        )
        st.caption("Soft continuation weight (AIS halting signal).")

    # ── HPC error trajectory ─────────────────────────────────────────────────
    if any(c.hpc_error is not None for c in caps):
        section("HPC Error Trajectory")
        hpc_data = []
        for i, c in enumerate(caps):
            if c.hpc_error is not None:
                hpc_data.append({"step": i + 1, "HPC error": c.hpc_error})
        hpc_df = pd.DataFrame(hpc_data)
        st.line_chart(hpc_df.set_index("step"), use_container_width=True)
        st.caption("HPC Level-1 prediction error over recurrent steps.")

    # ── Adversarial comparison ───────────────────────────────────────────────
    if "pr_caps_adv" in st.session_state and st.session_state["pr_caps_adv"]:
        section("Adversarial Comparison (Clean vs PGD-100)")

        caps_adv = st.session_state["pr_caps_adv"]
        result_adv = st.session_state["pr_result_adv"]
        adv_tensor = st.session_state.get("pr_adv_tensor")

        # Side-by-side final predictions
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Clean**")
            st.metric("Prediction", STL10_CLASSES[result.top_class],
                       f"{result.top_confidence * 100:.1f}%")
            st.metric("HPC error", f"{caps[-1].hpc_error:.6f}" if caps[-1].hpc_error else "—")
        with cc2:
            st.markdown("**Adversarial**")
            st.metric("Prediction", STL10_CLASSES[result_adv.top_class],
                       f"{result_adv.top_confidence * 100:.1f}%")
            st.metric("HPC error", f"{caps_adv[-1].hpc_error:.6f}" if caps_adv[-1].hpc_error else "—")

        # Belief drift over steps
        if caps[0].step_belief is not None and caps_adv[0].step_belief is not None:
            drift_data = []
            n_steps = max(len(caps), len(caps_adv))
            for t in range(n_steps):
                c_belief = caps[t].step_belief if t < len(caps) else None
                a_belief = caps_adv[t].step_belief if t < len(caps_adv) else None
                if c_belief is not None and a_belief is not None:
                    diff = (c_belief.float() - a_belief.float()).norm().item()
                    cos_sim = F.cosine_similarity(
                        c_belief.float().unsqueeze(0), a_belief.float().unsqueeze(0)
                    ).item()
                    drift_data.append({
                        "step": t + 1,
                        "belief_drift_L2": diff,
                        "belief_drift_cosine": 1.0 - cos_sim,
                    })
            if drift_data:
                drift_df = pd.DataFrame(drift_data)
                st.line_chart(drift_df.set_index("step"), use_container_width=True)
                st.caption("Belief drift: distance between clean and adversarial belief states.")

        # Gaze comparison
        st.markdown("**Gaze Trajectory Comparison**")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("Clean gaze path")
            clean_gaze = []
            for c in caps:
                if c.gaze_x_px is not None:
                    clean_gaze.append({"x": c.gaze_x_px, "y": c.gaze_y_px})
            if clean_gaze:
                gaze_df = pd.DataFrame(clean_gaze)
                st.scatter_chart(gaze_df, x="x", y="y", use_container_width=True)
        with gc2:
            st.markdown("Adversarial gaze path")
            adv_gaze = []
            for c in caps_adv:
                if c.gaze_x_px is not None:
                    adv_gaze.append({"x": c.gaze_x_px, "y": c.gaze_y_px})
            if adv_gaze:
                gaze_df = pd.DataFrame(adv_gaze)
                st.scatter_chart(gaze_df, x="x", y="y", use_container_width=True)

    # ── Developer / Debug section ────────────────────────────────────────────
    st.markdown("---")
    with st.expander(" Raw Step Data (Developer)", expanded=False):
        st.markdown(f"**Step {current_step + 1}** — raw tensor information:")

        debug_info = {}
        debug_info["step"] = cap.step
        debug_info["halted"] = cap.halted
        debug_info["gaze_x_norm"] = cap.gaze_x_norm
        debug_info["gaze_y_norm"] = cap.gaze_y_norm
        debug_info["gaze_x_px"] = cap.gaze_x_px
        debug_info["gaze_y_px"] = cap.gaze_y_px
        debug_info["pi_d"] = cap.pi_d
        debug_info["error_mag"] = cap.error_mag
        debug_info["beta_dynamic"] = cap.beta_dynamic
        debug_info["gate_alpha"] = cap.gate_alpha
        debug_info["uncertainty"] = cap.uncertainty
        debug_info["continuation"] = cap.continuation
        debug_info["recon_error"] = cap.recon_error
        debug_info["hpc_error"] = cap.hpc_error

        if cap.foveal_crop is not None:
            debug_info["foveal_crop_shape"] = list(cap.foveal_crop.shape)
        if cap.predicted_crop is not None:
            debug_info["predicted_crop_shape"] = list(cap.predicted_crop.shape)
        if cap.hpc_error_map is not None:
            debug_info["hpc_error_map_shape"] = list(cap.hpc_error_map.shape)
            debug_info["hpc_error_map_min"] = float(cap.hpc_error_map.min())
            debug_info["hpc_error_map_max"] = float(cap.hpc_error_map.max())
            debug_info["hpc_error_map_std"] = float(cap.hpc_error_map.std())
        if cap.hpc_prediction is not None:
            debug_info["hpc_prediction_shape"] = list(cap.hpc_prediction.shape)
        if cap.step_belief is not None:
            debug_info["step_belief_shape"] = list(cap.step_belief.shape)
            debug_info["step_belief_norm"] = float(cap.step_belief.norm())
            debug_info["step_belief_mean"] = float(cap.step_belief.mean())

        st.json(debug_info)

        # Session metadata
        if result:
            st.markdown("**Session metadata:**")
            st.json({
                "label": result.label,
                "arch": result.arch,
                "device": result.device,
                "ais_active": result.ais_active,
                "hpc_active": result.hpc_active,
                "config": result.config_summary,
                "steps_total": result.steps_total,
                "steps_effective": result.steps_effective,
            })

    footer()


if st.runtime.exists():
    render()

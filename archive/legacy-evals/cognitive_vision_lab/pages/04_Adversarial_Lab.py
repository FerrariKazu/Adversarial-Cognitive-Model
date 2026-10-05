"""04 — Adversarial Lab: a virtual attack simulator."""
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from cognitive_vision_lab.backend.attacks import ATTACKS, distance_metrics, run_attack
from cognitive_vision_lab.backend.models import class_names, list_models, load_model
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.utils.io import denormalize, pil_to_tensor, procedural_sample
from cognitive_vision_lab.utils.math_helpers import ATTACK_NOTES

STL_MEAN = (0.4467, 0.4398, 0.4066)
STL_STD = (0.2603, 0.2566, 0.2713)


@st.cache_resource(show_spinner="Loading model…")
def _load(model_id: str, device: str = "cpu"):
    return load_model(model_id, device=device)


def _to_pil(x: torch.Tensor, mean, std):
    img = denormalize(x.detach().cpu().float(), mean, std).clamp(0, 1).squeeze(0).permute(1, 2, 0)
    return (img.numpy() * 255).astype(np.uint8)


def render() -> None:
    hero("Adversarial Laboratory", "A virtual attack simulator — observe models fail, "
                                   "quantify every perturbation.")

    models = [m for m in list_models() if m["source"] != "ckpt" or m["available"]]
    names = [m["name"] for m in models]
    sel = st.selectbox("Model", names, index=names.index("ResNet-18") if "ResNet-18" in names else 0)
    entry = next(m for m in models if m["name"] == sel)
    device = st.radio("Device", ["cpu", "cuda"], horizontal=True)

    if not entry["available"] and entry["source"] == "checkpoint":
        st.warning("Checkpoint not present on this host; select an available model.")
        footer()
        return

    handle = _load(entry["id"], device=device)
    mean, std = (STL_MEAN, STL_STD) if handle.is_stl10 else (
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    size = 96 if handle.is_stl10 else 224

    c1, c2 = st.columns([1, 2])
    with c1:
        image = procedural_sample("car" if handle.is_stl10 else "demo", size=size)
        st.image(image, caption="Clean input", width=240)

    with c2:
        attack = st.selectbox("Attack", list(ATTACKS.keys()))
        a1, a2, a3 = st.columns(3)
        eps = a1.number_input("ε (norm-space)", value=0.031, min_value=0.001,
                              max_value=0.5, format="%.4f", step=0.005)
        steps = a2.number_input("Steps", value=50, min_value=1, max_value=500, step=1)
        targeted = a3.toggle("Targeted")
        st.caption(ATTACK_NOTES.get(attack, ""))

    x = pil_to_tensor(image, mean, std, size).to(next(handle.model.parameters()).device)
    with torch.no_grad():
        clean_probs = handle.predict(x).squeeze(0)
    labels = class_names(handle)
    y = clean_probs.argmax().unsqueeze(0)

    run_btn = st.button(" Generate attack", type="primary", width="stretch")
    if run_btn:
        t0 = time.perf_counter()
        with st.spinner(f"Running {attack} ({steps} steps)…"):
            x_adv = run_attack(attack, handle.model, x, y, eps=eps, steps=steps,
                               mean=mean, std=std)
        elapsed = (time.perf_counter() - t0) * 1000.0
        with torch.no_grad():
            adv_probs = handle.predict(x_adv).squeeze(0)
        success = int(adv_probs.argmax().item() != y.item())

        clean_idx = int(y.item())
        adv_idx = int(adv_probs.argmax().item())
        diff = (x_adv - x).squeeze(0).abs()
        diff_max = diff.max().item()

        st.markdown("### Attack result")
        mrow = st.columns(5)
        mrow[0].metric("Attack success", " fooled" if success else " failed")
        mrow[1].metric("Runtime", f"{elapsed:.0f} ms")
        mrow[2].metric("Δ confidence", f"{(clean_probs[clean_idx]-adv_probs[clean_idx]).item()*100:.1f} pp")
        mrow[3].metric("Prediction", labels[adv_idx] if adv_idx < len(labels) else str(adv_idx),
                       delta=f"from {labels[clean_idx] if clean_idx < len(labels) else clean_idx}")
        mrow[4].metric("Max |Δ| (px)", f"{diff_max*255:.2f}")

        st.markdown("### Perturbation view")
        col1, col2, col3, col4 = st.columns(4)
        col1.image(_to_pil(x, mean, std), caption="Original", width=180)
        col2.image(_to_pil(x_adv, mean, std), caption="Perturbed", width=180)
        col3.image((diff.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 8).astype(np.uint8),
                   caption="Difference (×8)", width=180)
        col4.image(_to_pil(x_adv, mean, std), caption="Magnified (×4 zoom)", width=180)

        st.markdown("### Distance metrics")
        metrics = distance_metrics(x, x_adv, mean, std)
        st.dataframe(pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()]),
                     hide_index=True, width="stretch")

        st.markdown("### Probability distribution")
        d1, d2 = st.columns(2)
        d1.caption("Clean")
        st.bar_chart(pd.Series(clean_probs.cpu().numpy(), name="p"), height=220)
        d2.caption("Adversarial")
        st.bar_chart(pd.Series(adv_probs.cpu().numpy(), name="p"), height=220)

        section("Educational context")
        with st.expander(" Attack mathematics", expanded=False):
            from cognitive_vision_lab.components.equations import render_equation

            if attack in ("pgd", "apgd"):
                render_equation("pgd")
                render_equation("apgd")
            elif attack == "fgsm":
                render_equation("fgsm")
            elif attack == "cw":
                render_equation("cw")
            elif attack == "deepfool":
                render_equation("deepfool")
            elif attack == "square":
                render_equation("square")
            elif attack == "fab":
                render_equation("fab")
            render_equation("ssim")
            render_equation("psnr")
    else:
        st.info("Configure the attack and press **Generate attack**.")

    footer()


if st.runtime.exists():
    render()

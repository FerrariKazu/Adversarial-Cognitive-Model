import io
from pathlib import Path

import streamlit as st
from PIL import Image

from cognitive_vision_lab.backend.model_registry import (
    list_available_models, load_model, predict,
)


def render():
    st.subheader("Live Image Classification")
    st.caption("Upload an image and compare how different models perceive it.")

    models = list_available_models()
    model_names = [m["id"] for m in models]

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_model = st.selectbox("Select model", model_names)
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg"],
            help="Upload any natural image (will be resized to 224×224 or 96×96 for STL-10 models)",
        )

    with col2:
        attack_eps = st.slider(
            "PGD attack strength (ε=0 = clean)",
            min_value=0.0, max_value=0.0313, value=0.0, step=0.001,
            format="%.4f",
        )
        attack_steps = st.slider("PGD steps", 10, 100, 40) if attack_eps > 0 else 0
        run_btn = st.button("Classify", type="primary")

    if uploaded_file is not None and run_btn:
        pil_img = Image.open(uploaded_file).convert("RGB")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.image(pil_img, caption="Input image", width=300)

        try:
            model, transform, is_stl10 = load_model(selected_model, use_cpu=True)
            input_tensor = transform(pil_img).to(next(model.parameters()).device)

            result = predict(model, input_tensor, stl10=is_stl10)

            with col_b:
                st.markdown(f"**Model:** {selected_model}")
                st.markdown(f"**Predicted:** `{result['predicted_class']}`")
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")

                if attack_eps > 0:
                    st.markdown(f"**Attack:** PGD-{attack_steps} at ε={attack_eps}")
                    from cognitive_vision_lab.backend.attacks import pgd_attack
                    adv_tensor = pgd_attack(
                        model, input_tensor,
                        label_idx=result["predicted_idx"],
                        eps=attack_eps, steps=attack_steps,
                    )
                    adv_result = predict(model, adv_tensor, stl10=is_stl10)
                    st.markdown(f"**Post-attack:** `{adv_result['predicted_class']}`")
                    st.markdown(f"**Post-attack confidence:** {adv_result['confidence']:.1%}")
                    flipped = adv_result["predicted_idx"] != result["predicted_idx"]
                    if flipped:
                        st.error("⚠ Classification flipped under attack!")
                    else:
                        st.success("✓ Prediction survived attack")

            import numpy as np
            st.markdown("**Top-5 predictions:**")
            labels = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"] if is_stl10 else []
            if not labels:
                labels = [f"class_{i}" for i in range(1000)]
            probs = result["all_probs"]
            top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
            prob_df = {
                "Rank": list(range(1, 6)),
                "Class": [labels[i] if i < len(labels) else str(i) for i, _ in top5],
                "Probability": [f"{p:.1%}" for _, p in top5],
            }
            st.dataframe(prob_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error running inference: {e}")
            st.info(
                "This model requires a checkpoint not found locally. "
                "RHAN models need their .pth files in checkpoints/ . "
                "Standard torchvision models should work out of the box."
            )

    elif uploaded_file is None:
        st.info("Upload an image and click **Classify** to begin.")

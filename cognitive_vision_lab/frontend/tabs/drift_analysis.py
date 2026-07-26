import streamlit as st
from PIL import Image

from cognitive_vision_lab.backend.model_registry import list_available_models, load_model
from cognitive_vision_lab.backend.explainability import compute_representation


def render():
    st.subheader("Representation Drift Analysis")
    st.caption(
        "Compare model internal representations between clean and adversarially "
        "perturbed inputs. Measures the geometric distortion in latent space."
    )

    models = list_available_models()
    model_names = [m["id"] for m in models]
    selected_model = st.selectbox("Select model", model_names, key="drift_model")

    col1, col2 = st.columns(2)
    with col1:
        clean_file = st.file_uploader("Clean image", type=["png", "jpg", "jpeg"], key="drift_clean")
    with col2:
        adv_file = st.file_uploader("Adversarial image", type=["png", "jpg", "jpeg"], key="drift_adv")
        st.caption("Upload two images to compare their representations.")

    if clean_file is not None and adv_file is not None:
        clean_img = Image.open(clean_file).convert("RGB")
        adv_img = Image.open(adv_file).convert("RGB")

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(clean_img, caption="Clean input", use_container_width=True)
        with col_b:
            st.image(adv_img, caption="Adversarial input", use_container_width=True)

        try:
            model, transform, is_stl10 = load_model(selected_model, use_cpu=True)
            clean_tensor = transform(clean_img).to(next(model.parameters()).device)
            adv_tensor = transform(adv_img).to(next(model.parameters()).device)

            clean_reps = compute_representation(model, clean_tensor)
            adv_reps = compute_representation(model, adv_tensor)

            import numpy as np
            from scipy.spatial.distance import cosine as cosine_dist

            st.markdown("**Representation drift by layer:**")
            rows = []
            for name in clean_reps:
                if name in adv_reps:
                    c_rep = clean_reps[name].cpu().numpy().flatten()
                    a_rep = adv_reps[name].cpu().numpy().flatten()
                    if c_rep.size > 0 and a_rep.size > 0:
                        cos_sim = 1 - cosine_dist(c_rep, a_rep)
                        l2 = np.linalg.norm(c_rep - a_rep)
                        rows.append({
                            "Layer": name,
                            "Cosine similarity": f"{cos_sim:.4f}",
                            "L2 drift": f"{l2:.4f}",
                            "Clean norm": f"{np.linalg.norm(c_rep):.4f}",
                            "Adv norm": f"{np.linalg.norm(a_rep):.4f}",
                        })

            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.warning("No shared representation layers found between the two inputs.")

        except Exception as e:
            st.error(f"Error: {e}")

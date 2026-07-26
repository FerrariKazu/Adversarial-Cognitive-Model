import streamlit as st
from PIL import Image

from cognitive_vision_lab.backend.model_registry import list_available_models, load_model, predict
from cognitive_vision_lab.backend.explainability import extract_attention_maps


def render():
    st.subheader("Attention Map Visualization")
    st.caption("Extract and compare attention patterns across model layers.")

    models = list_available_models()
    model_names = [m["id"] for m in models]
    selected_model = st.selectbox("Select model", model_names, key="attn_model")
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="attn_file")

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Input", width=300)

        try:
            model, transform, is_stl10 = load_model(selected_model, use_cpu=True)
            input_tensor = transform(pil_img).to(next(model.parameters()).device)

            attn_maps = extract_attention_maps(model, input_tensor)

            if not attn_maps:
                st.warning(
                    "No attention layers detected for this model. "
                    "ViT, Swin, and RHAN models have attention mechanisms; "
                    "standard CNNs do not expose attention maps."
                )
            else:
                st.markdown(f"Found **{len(attn_maps)}** attention maps across layers:")

                for i, (name, attn) in enumerate(attn_maps.items()):
                    with st.expander(f"Layer: {name}  —  shape {list(attn.shape)}"):
                        attn_np = attn.cpu().numpy()
                        if attn_np.ndim == 3:
                            import numpy as np
                            avg_attn = attn_np.mean(axis=0)
                            st.image(avg_attn, caption=f"Average attention — {name}",
                                     use_container_width=True, clamp=True)
                            st.caption(f"Attention mean: {float(attn.mean()):.4f}, "
                                       f"std: {float(attn.std()):.4f}")

            result = predict(model, input_tensor, stl10=is_stl10)
            st.markdown(f"**Predicted:** `{result['predicted_class']}` ({result['confidence']:.1%})")

        except Exception as e:
            st.error(f"Error: {e}")

import streamlit as st
from PIL import Image

from cognitive_vision_lab.backend.model_registry import list_available_models, load_model, predict
from cognitive_vision_lab.backend.explainability import GradCAM


def render():
    st.subheader("Saliency Maps (Grad-CAM)")
    st.caption("Visualize which image regions drive model decisions.")

    models = list_available_models()
    model_names = [m["id"] for m in models]
    selected_model = st.selectbox("Select model", model_names, key="sal_model")
    uploaded_file = st.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg"], key="sal_file",
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")

        try:
            model, transform, is_stl10 = load_model(selected_model, use_cpu=True)
            input_tensor = transform(pil_img).to(next(model.parameters()).device)

            result = predict(model, input_tensor, stl10=is_stl10)

            cam = GradCAM(model)
            heatmap = cam.generate(input_tensor, class_idx=result["predicted_idx"])

            import numpy as np
            heatmap_colored = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
            heatmap_colored[..., 0] = (heatmap * 255).astype(np.uint8)

            overlay = np.array(pil_img.resize((heatmap.shape[1], heatmap.shape[0])))
            if overlay.ndim == 3 and overlay.shape[2] == 3:
                overlay = (0.6 * overlay + 0.4 * heatmap_colored).astype(np.uint8)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(pil_img, caption="Input", use_container_width=True)
            with col2:
                st.image(heatmap_colored, caption="Grad-CAM heatmap", use_container_width=True)
            with col3:
                st.image(overlay, caption="Overlay", use_container_width=True)

            st.markdown(f"**Predicted:** `{result['predicted_class']}` ({result['confidence']:.1%})")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info(
                "Grad-CAM requires a model with convolutional layers. "
                "Standard torchvision models work best. RHAN models may "
                "need a target_layer specified in the config."
            )

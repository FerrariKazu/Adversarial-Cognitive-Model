from pathlib import Path
import streamlit as st
from cognitive_vision_lab.config import FIGURES_DIR


def render():
    st.subheader("Published Research Figures")

    figure_groups = [
        {
            "title": "Class Robustness Heatmap",
            "desc": "Per-class accuracy breakdown under increasing perturbation.",
            "path": FIGURES_DIR / "evaluation" / "figure_e4_class_robustness_heatmap_light.png",
        },
        {
            "title": "t-SNE Representation Drift",
            "desc": "Feature space distortion under adversarial attack.",
            "path": FIGURES_DIR / "representation" / "figure_a2_tsne_representation_drift_light.png",
        },
        {
            "title": "Attention Overlay — ViT vs RHAN",
            "desc": "Attention map comparison: ViT scatters, RHAN stays focused.",
            "path": FIGURES_DIR / "attention" / "figure_b1_attention_overlay_light.png",
        },
        {
            "title": "d′ vs ε (Publication Figure)",
            "desc": "Static d′ sensitivity chart with human reference.",
            "path": FIGURES_DIR / "evaluation" / "figure_e2_dprime_vs_epsilon_light.png",
        },
        {
            "title": "Accuracy vs ε Decay",
            "desc": "Raw accuracy decay across perturbation budgets.",
            "path": FIGURES_DIR / "evaluation" / "figure_e1_accuracy_vs_epsilon_light.png",
        },
        {
            "title": "Robustness Threshold Comparison",
            "desc": "Bar chart of εthresh across all evaluated models.",
            "path": FIGURES_DIR / "evaluation" / "figure_e3_robustness_threshold_comparison_light.png",
        },
        {
            "title": "Grad-CAM Explainability",
            "desc": "Saliency maps for clean and adversarial inputs.",
            "path": FIGURES_DIR / "explainability" / "figure_j1_explainability_gradcam_light.png",
        },
        {
            "title": "UMAP Feature Space",
            "desc": "UMAP projection of model feature representations.",
            "path": FIGURES_DIR / "representation" / "figure_a1_umap_feature_space_light.png",
        },
        {
            "title": "Representation Drift Histogram",
            "desc": "Distribution of per-sample representation drift magnitudes.",
            "path": FIGURES_DIR / "representation" / "figure_a3_representation_drift_histogram_light.png",
        },
        {
            "title": "Decision Boundary Slices",
            "desc": "2D slices through the decision boundary along attack directions.",
            "path": FIGURES_DIR / "representation" / "figure_h1_decision_boundary_slices_light.png",
        },
        {
            "title": "Loss Landscape",
            "desc": "Loss landscape visualization along random attack directions.",
            "path": FIGURES_DIR / "representation" / "figure_h2_loss_landscape_light.png",
        },
        {
            "title": "Attention Evolution",
            "desc": "How RHAN attention evolves across recurrent foraging steps.",
            "path": FIGURES_DIR / "attention" / "figure_b2_attention_evolution_light.png",
        },
        {
            "title": "Feedback Correction",
            "desc": "Predictive coding feedback correction visualization.",
            "path": FIGURES_DIR / "attention" / "figure_b3_feedback_correction_light.png",
        },
    ]

    filter_col, _ = st.columns([1, 3])
    with filter_col:
        categories = list(set(
            p["path"].parent.name for p in figure_groups if p["path"].exists()
        ))
        selected_cat = st.selectbox("Filter by category", ["All"] + sorted(categories))

    for group in figure_groups:
        if selected_cat != "All" and group["path"].parent.name != selected_cat:
            continue
        if not group["path"].exists():
            continue
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{group['title']}**")
                st.caption(group["desc"])
                st.caption(f"*{group['path'].parent.name}*")
            with col2:
                st.image(str(group["path"]), use_container_width=True)
            st.markdown("---")

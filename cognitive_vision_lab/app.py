import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Cognitive Vision Lab — Intelligent Benchmarking Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from cognitive_vision_lab.frontend.tabs import (
    dprime_chart,
    live_inference,
    saliency,
    attention,
    drift_analysis,
    figures,
    methodology,
    roadmap,
    benchmark,
)

PAGES = {
    "📈 d′ Sensitivity": dprime_chart,
    "🔬 Live Inference": live_inference,
    "🔥 Attack Lab": live_inference,
    "🌡️ Saliency Maps": saliency,
    "👁️ Attention": attention,
    "📊 Representation Drift": drift_analysis,
    "⚡ Benchmark": benchmark,
    "🖼️ Published Figures": figures,
    "📐 Methodology": methodology,
    "🗺️ Roadmap": roadmap,
}


def sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
        st.markdown("## Cognitive Vision Lab")
        st.caption("Intelligent Benchmarking Platform for Human-Like AI Vision")

        st.markdown("---")
        st.markdown("### Navigation")

        page = st.radio(
            "Go to", list(PAGES.keys()),
            label_visibility="collapsed",
            index=0,
        )

        st.markdown("---")
        st.markdown("### System Status")

        col1, col2 = st.columns(2)
        with col1:
            import torch
            gpu_avail = torch.cuda.is_available()
            st.metric("GPU", "✅" if gpu_avail else "❌ CPU", delta=None)
        with col2:
            from cognitive_vision_lab.config import SWEEP_PATH
            data_avail = SWEEP_PATH.exists()
            st.metric("Sweep Data", "✅" if data_avail else "❌ Missing", delta=None)

        import os
        st.caption(f"PyTorch {torch.__version__}")

        st.markdown("---")
        st.markdown(
            "**Cognitive Vision Lab** v1.0  \n"
            "Research prototype — not for clinical use.  \n"
            "[GitHub](https://github.com/FerrariKazu/Adversarial-Cognitive-Model)"
        )

    return page


def main():
    page = sidebar()

    st.title("🧠 Cognitive Vision Lab")
    st.markdown(
        "##### Intelligent Benchmarking Platform for Human-Like AI Vision"
    )
    st.markdown("---")

    tab_module = PAGES[page]
    tab_module.render()

    st.markdown("---")
    st.caption(
        "Cognitive Vision Lab — Research Prototype v1.0  |  "
        "Built with Streamlit + Plotly + PyTorch  |  "
        "No GPU required for basic functionality"
    )


if __name__ == "__main__":
    main()

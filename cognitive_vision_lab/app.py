"""Cognitive Vision Lab v2.0 — entry point.

Multipage application built on `st.navigation`. Pages live in `pages/` and are
declared explicitly here (no automatic discovery), giving full control over
ordering, titles, and icons.

Run with:
    streamlit run cognitive_vision_lab/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repository root importable so `cognitive_vision_lab` is a package
# regardless of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from cognitive_vision_lab.config import APP_TITLE  # noqa: E402
from cognitive_vision_lab.utils.theme import brand_header, inject_css  # noqa: E402

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
brand_header()

PAGES = [
    st.Page("pages/01_Home.py", title="Home", icon="🏛️", default=True),
    st.Page("pages/02_Model_Zoo.py", title="Model Zoo", icon="🧬"),
    st.Page("pages/03_Interactive_Inference.py", title="Interactive Inference", icon="🔬"),
    st.Page("pages/04_Adversarial_Lab.py", title="Adversarial Lab", icon="⚔️"),
    st.Page("pages/05_Attention_Explorer.py", title="Attention Explorer", icon="👁️"),
    st.Page("pages/06_Representation_Drift.py", title="Representation Drift", icon="🌊"),
    st.Page("pages/07_GradCAM_and_Saliency.py", title="GradCAM & Saliency", icon="🔥"),
    st.Page("pages/08_Human_vs_AI.py", title="Human vs AI", icon="🧑‍🔬"),
    st.Page("pages/09_RobustBench_Comparison.py", title="RobustBench", icon="🏆"),
    st.Page("pages/10_RHAN_Architecture.py", title="RHAN Architecture", icon="🕸️"),
    st.Page("pages/11_Benchmark_Results.py", title="Benchmark Results", icon="📊"),
    st.Page("pages/12_Experiment_Manager.py", title="Experiment Manager", icon="🧪"),
    st.Page("pages/13_Dataset_Explorer.py", title="Dataset Explorer", icon="🗃️"),
    st.Page("pages/14_Report_Generator.py", title="Report Generator", icon="📄"),
    st.Page("pages/15_Perception_Replay.py", title="Perception Replay", icon="🎬"),
]

nav = st.navigation(PAGES)
nav.run()

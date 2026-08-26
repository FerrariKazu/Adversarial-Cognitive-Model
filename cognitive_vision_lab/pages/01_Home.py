"""01 — Home: laboratory landing page with live system status."""
import streamlit as st

from cognitive_vision_lab import config
from cognitive_vision_lab.backend.benchmark import curated_profiles
from cognitive_vision_lab.backend.datasets import list_datasets
from cognitive_vision_lab.backend.experiments import list_runs
from cognitive_vision_lab.backend.models import list_models
from cognitive_vision_lab.backend.robustbench import available_threats
from cognitive_vision_lab.components.layout import footer, hero, metric_grid, section
from cognitive_vision_lab.utils.hardware import get_hardware


def render() -> None:
    hero(
        "Cognitive Vision Lab",
        "How does a vision model perceive, reason about, and ultimately fail compared to humans?",
    )

    profiles = curated_profiles()
    models = list_models()
    available = sum(1 for m in models if m["available"])
    runs = list_runs()
    done_runs = sum(1 for r in runs if r.status == "done")
    hw = get_hardware()

    metric_grid([
        {"label": "Registered models", "value": len(models),
         "note": f"{available} checkpoints available"},
        {"label": "Curated benchmarks", "value": len(profiles),
         "note": "STL-10 · CIFAR-10 · ImageNet"},
        {"label": "Datasets", "value": len(list_datasets()),
         "note": "clean + corrupted variants"},
        {"label": "Attack library", "value": 7,
         "note": "PGD · FGSM · CW · DeepFool · Square · FAB · APGD"},
        {"label": "RobustBench threats", "value": len(available_threats()),
         "note": "auto-loaded leaderboards"},
        {"label": "Experiments", "value": len(runs),
         "note": f"{done_runs} completed"},
    ])

    st.markdown("")
    section("System status", "Live hardware and data availability — refreshed every interaction.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Device", hw.device.upper(), delta=hw.gpu_name if hw.cuda_available else None)
    c2.metric("GPU memory", f"{hw.gpu_mem_used_gb:.2f} / {hw.gpu_vram_gb:.1f} GB" if hw.cuda_available
              else "n/a", delta=f"{hw.gpu_util_pct:.0f}% util" if hw.cuda_available else None)
    c3.metric("Runtime", f"PyTorch {hw.torch_version}", delta=f"Python {hw.python_version}")

    with st.expander(" Quick links", expanded=True):
        st.markdown(
            "| Page | Purpose |\n"
            "|---|---|\n"
            "| **02 Model Zoo** | Profiles, params, FLOPs, robustness for every model |\n"
            "| **03 Interactive Inference** | Single-image analysis incl. RHAN dynamics |\n"
            "| **04 Adversarial Lab** | Live attack simulator (PGD, CW, DeepFool, …) |\n"
            "| **06 Representation Drift** | PCA / t-SNE / UMAP drift animations |\n"
            "| **08 Human vs AI** | Psychophysics comparison with real human data |\n"
            "| **10 RHAN Architecture** | Interactive module explorer |\n"
            "| **12 Experiment Manager** | Launch and track benchmarks |\n"
            "| **14 Report Generator** | Markdown / LaTeX / PDF exports |\n"
        )

    section("Recent experiments", "Latest entries in the experiment store.")
    if runs:
        st.dataframe(
            [{"name": r.name, "status": r.status, "progress": f"{r.progress:.0f}%",
              "created": r.created_at[:19]} for r in runs[:5]],
            width="stretch", hide_index=True,
        )
    else:
        st.info("No experiments yet — launch one from the Experiment Manager (page 12).")

    st.caption(
        f"Cognitive Vision Lab v{config.APP_VERSION} — a research platform for "
        "explainable, robust, biologically inspired computer vision."
    )
    footer()


if st.runtime.exists():
    render()

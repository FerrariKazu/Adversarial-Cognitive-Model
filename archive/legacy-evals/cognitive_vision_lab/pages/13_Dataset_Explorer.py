"""13 — Dataset Explorer: browse datasets, class distributions, corruptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cognitive_vision_lab.backend.benchmark import load_comparison_table
from cognitive_vision_lab.backend.datasets import get_dataset, list_datasets, sample_images
from cognitive_vision_lab.components.charts import bar_fig
from cognitive_vision_lab.components.layout import footer, hero, metric_grid, section


IMAGENET_CORRUPTION_TYPES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def _distribution_fig(info) -> None:
    # Real label distribution is unknown without the dataset; show class list
    # counts as a synthetic uniform baseline, annotated honestly.
    n = len(info.classes)
    fig = bar_fig(list(range(min(n, 10))),
                  [100.0 / n] * min(n, 10),
                  names=info.classes[: min(n, 10)],
                  ylabel="% of labeled set",
                  title=f"{info.name} — class distribution (illustrative)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Illustrative uniform distribution. Connect a dataset cache to show "
               "the true empirical class balance.")


def render() -> None:
    hero("Dataset Explorer",
         "Every benchmark we evaluate against, documented and browsable.")

    name = st.selectbox("Dataset", list_datasets())
    info = get_dataset(name)

    metric_grid([
        {"label": "Classes", "value": info.n_classes},
        {"label": "Resolution", "value": info.resolution},
        {"label": "Train", "value": f"{info.train_size:,}"},
        {"label": "Test", "value": f"{info.test_size:,}"},
    ])

    st.markdown(info.notes)

    # ── Example images ────────────────────────────────────────────────────────
    section("Example images", "Procedural stand-ins until real tensors are cached.")
    imgs = sample_images(name, n=8)
    cols = st.columns(4)
    for i, (img, label) in enumerate(imgs):
        with cols[i % 4]:
            st.image(img, caption=label, width="stretch")

    # ── Class distribution ────────────────────────────────────────────────────
    section("Distribution")
    _distribution_fig(info)
    with st.expander(f"Class list ({info.n_classes})", expanded=False):
        st.write(", ".join(info.classes[:200]))

    # ── Corruptions ───────────────────────────────────────────────────────────
    if info.corruptions:
        section("Corruptions", "ImageNet-C style common corruptions.")
        for cat, corrs in IMAGENET_CORRUPTION_TYPES.items():
            present = [c for c in corrs if c in info.corruptions]
            if present:
                st.markdown(f"**{cat.title()}** — " + ", ".join(present))
    else:
        st.caption("No corruption benchmark defined for this dataset.")

    # ── Cross-dataset table (if available) ────────────────────────────────────
    section("Comparison table", "tier1/results/comparison_table.csv when present.")
    tbl = load_comparison_table()
    if tbl is not None:
        st.dataframe(tbl, width="stretch", height=300, hide_index=True)
    else:
        st.info("comparison_table.csv not found — embed curated profiles from FINDINGS.md "
                "instead (see Benchmark Results page).")

    footer()


if st.runtime.exists():
    render()

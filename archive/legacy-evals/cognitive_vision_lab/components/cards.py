"""Reusable cards and status elements."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.backend.benchmark import ModelProfile


def model_profile_card(profile: ModelProfile) -> None:
    """Compact profile card for the Model Zoo."""
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{profile.name}**")
            st.caption(f"{profile.family} · {profile.dataset}")
        with c2:
            if profile.ethresh and profile.ethresh > 0.05:
                label = "High robustness"
            elif profile.ethresh and profile.ethresh > 0.01:
                label = "Moderate"
            else:
                label = "Fragile"
            st.markdown(f"`{label}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clean %", f"{profile.clean_acc:.1f}")
        m2.metric("εthresh", f"{profile.ethresh:.3f}" if profile.ethresh else "n/a")
        m3.metric("Params", f"{profile.params_m:.1f}M")
        m4.metric("d′", f"{profile.dprime:.2f}" if profile.dprime else "n/a")


def stat_grid(stats: dict[str, str]) -> None:
    cols = st.columns(len(stats))
    for col, (k, v) in zip(cols, stats.items()):
        col.metric(k, v)


def status_badge(status: str) -> str:
    return f"`{status}`"

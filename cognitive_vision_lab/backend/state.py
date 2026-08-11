"""Session-state helpers — typed get/set with defaults."""
from __future__ import annotations

from typing import Any

import streamlit as st


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:  # noqa: A001
    st.session_state[key] = value


def get_or_set(key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def selected_model() -> str:
    return get_or_set("cvl_selected_model", "RHAN-Large (Pseudolabel)")


def set_selected_model(model_id: str) -> None:
    st.session_state["cvl_selected_model"] = model_id

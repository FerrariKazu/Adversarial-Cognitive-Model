import streamlit as st


def section_header(title: str, description: str = ""):
    st.markdown(f"### {title}")
    if description:
        st.caption(description)
    st.markdown("")


def info_box(text: str, kind: str = "info"):
    fn = {"info": st.info, "warning": st.warning, "success": st.success, "error": st.error}
    fn.get(kind, st.info)(text)


def disabled_button(label: str, key: str, help_text: str = "Not yet implemented"):
    st.button(label, disabled=True, key=key, help=help_text)

# components/sidebar.py
import streamlit as st

def render_sidebar():
    st.sidebar.title("Omega Agent Dashboard")
    st.sidebar.markdown("Control Center")

    if st.sidebar.button("Refresh Page"):
        st.rerun()      # 🔥 new API

    st.sidebar.markdown("---")
    st.sidebar.info("Use tabs to view different modules.")

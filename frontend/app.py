"""
HrMind — Streamlit Frontend (Phase 1 Placeholder)

Full implementation arrives in Phase 7.
This placeholder confirms the frontend service starts correctly.
"""

import os

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="HrMind",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 HrMind — HR Intelligence Platform")
st.caption("Multi-Agent Orchestration · RAG · Text2SQL · Document Parser")

st.info(
    f"**Phase 1 in progress** — Full UI coming in Phase 7.\n\n"
    f"Backend API: [{BACKEND_URL}/docs]({BACKEND_URL}/docs)",
    icon="🔧",
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RAG Agent", "Pending", delta="Phase 2")
with col2:
    st.metric("SQL Agent", "Pending", delta="Phase 3")
with col3:
    st.metric("Doc Parser", "Pending", delta="Phase 4")

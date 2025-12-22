"""
🛰️ SLCCI Satellite Altimetry Dashboard - Main Application
=========================================================
Streamlit entry point using modular architecture.
"""

import streamlit as st

# Configure page FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="🛰️ SLCCI Satellite Altimetry",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import the app
from app.main import run_app

if __name__ == "__main__":
    run_app()

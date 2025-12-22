"""
Main Streamlit Application
==========================
Entry point for the SLCCI Satellite Altimetry Dashboard.
"""

import streamlit as st
from .components.sidebar import render_sidebar
from .components.tabs import render_tabs
from .state import init_session_state
from .styles import apply_custom_css


def run_app():
    """Main application entry point."""
    
    # Initialize
    apply_custom_css()
    init_session_state()
    
    # Header
    st.markdown(
        '<div class="main-header">🛰️ SLCCI Satellite Altimetry Analysis</div>',
        unsafe_allow_html=True,
    )
    
    # Sidebar - data loading and settings
    config = render_sidebar()
    
    # Check if data is loaded
    if not st.session_state.get("datasets"):
        render_welcome_message()
        return
    
    # Main content tabs
    render_tabs(config)


def render_welcome_message():
    """Show welcome message when no data is loaded."""
    st.markdown("""
    <div class="info-box">
    <h3>👋 Welcome to the SLCCI Satellite Altimetry Analysis Dashboard!</h3>
    <p>To get started:</p>
    <ul>
        <li>📂 Use <b>Local Files</b> to load NetCDF files from your workspace</li>
        <li>📤 Use <b>Upload Files</b> to drag and drop your NetCDF files</li>
    </ul>
    <p><b>Supported formats:</b> SLCCI Altimeter Database NetCDF files (*.nc)</p>
    
    <h4>📊 Features:</h4>
    <ul>
        <li>📈 <b>Slope Timeline</b> - DOT slope evolution with error bars</li>
        <li>🌊 <b>DOT Profiles</b> - Compare profiles across cycles</li>
        <li>📅 <b>Monthly Analysis</b> - 12-subplot seasonal analysis</li>
        <li>🗺️ <b>Spatial View</b> - Interactive map visualization</li>
        <li>📊 <b>Data Explorer</b> - Raw data inspection</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

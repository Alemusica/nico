"""
Main Streamlit Application
==========================
Entry point for the SLCCI Satellite Altimetry Dashboard.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from app.components.sidebar import render_sidebar
from app.components.tabs import render_tabs
from app.state import init_session_state
from app.styles import apply_custom_css


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
    
    # Check if data is loaded - but always show catalog tab
    if not st.session_state.get("datasets"):
        # Show catalog-only view when no data loaded
        render_catalog_only_view()
        return
    
    # Main content tabs (all tabs when data is loaded)
    render_tabs(config)


def render_catalog_only_view():
    """Show catalog and welcome when no local data is loaded."""
    from app.components.catalog_tab import render_catalog_tab
    
    tab1, tab2 = st.tabs(["🗃️ Dataset Catalog", "👋 Welcome"])
    
    with tab1:
        render_catalog_tab()
    
    with tab2:
        render_welcome_message()


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


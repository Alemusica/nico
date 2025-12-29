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

# Setup logging FIRST
from src.core.logging_config import setup_streamlit_logging, get_logger, LogContext
logger = setup_streamlit_logging(level="DEBUG")

import streamlit as st
from app.components.sidebar import render_sidebar
from app.components.tabs import render_tabs
from app.state import init_session_state
from app.styles import apply_custom_css

# New unified data selector
try:
    from app.components.data_selector import (
        render_data_selector,
        is_data_load_requested,
        clear_load_request,
        get_current_selection
    )
    DATA_SELECTOR_AVAILABLE = True
    logger.info("Data selector loaded successfully")
except ImportError as e:
    DATA_SELECTOR_AVAILABLE = False
    logger.warning(f"Data selector not available: {e}")


def run_app():
    """Main application entry point."""
    logger.info("Starting Streamlit app")
    
    # Initialize
    apply_custom_css()
    init_session_state()
    
    # Header
    st.markdown(
        '<div class="main-header">🛰️ SLCCI Satellite Altimetry Analysis</div>',
        unsafe_allow_html=True,
    )
    
    # === NEW: Unified Data Selector ===
    if DATA_SELECTOR_AVAILABLE:
        logger.debug("Rendering data selector")
        try:
            selection = render_data_selector()
            logger.debug(f"Selection: source={selection.source}, gate={selection.gate_id}")
        except Exception as e:
            logger.error(f"Data selector error: {e}", exc_info=True)
            st.error(f"❌ Data selector error: {e}")
            # Fallback to old sidebar
            config = render_sidebar()
            selection = None
        
        # Handle data load request
        if selection and is_data_load_requested():
            _handle_data_load(selection)
            clear_load_request()
    else:
        # Fallback to old sidebar
        logger.info("Using fallback sidebar")
        config = render_sidebar()
    
    # Check if data is loaded - but always show catalog tab
    if not st.session_state.get("datasets"):
        # Show catalog-only view when no data loaded
        render_catalog_only_view()
        return
    
    # Main content tabs (all tabs when data is loaded)
    config = st.session_state.get("app_config")
    render_tabs(config)


def _handle_data_load(selection):
    """Handle data loading from the new selector."""
    
    if not selection.confirmed:
        st.warning("Please confirm the data request")
        return
    
    with st.spinner("Loading data..."):
        try:
            from src.services import DataService
            
            data_service = DataService()
            
            # Build request from selection
            datasets_loaded = []
            
            for ds_id in selection.selected_datasets:
                variables = selection.selected_variables.get(ds_id, [])
                
                # Load dataset
                data = data_service.load_dataset(
                    dataset_id=ds_id,
                    bbox=selection.bbox,
                    time_range=selection.time_range,
                    variables=variables if variables else None
                )
                
                if data is not None:
                    datasets_loaded.append({
                        "id": ds_id,
                        "data": data,
                        "variables": variables
                    })
            
            if datasets_loaded:
                st.session_state.datasets = datasets_loaded
                st.session_state.current_selection = selection
                st.success(f"✅ Loaded {len(datasets_loaded)} dataset(s)")
                st.rerun()
            else:
                st.error("No data loaded. Check your selection.")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")


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


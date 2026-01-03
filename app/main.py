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
    
    # === SIDEBAR: Data loading + Settings ===
    # The sidebar handles local file loading and analysis parameters
    config = render_sidebar()
    
    # Store config for tabs
    st.session_state.app_config = config
    
    # Check if any data is loaded (SLCCI, CMEMS, DTU, or generic)
    slcci_data = st.session_state.get("slcci_pass_data") or st.session_state.get("dataset_slcci")
    cmems_data = st.session_state.get("dataset_cmems")
    dtu_data = st.session_state.get("dataset_dtu")
    datasets = st.session_state.get("datasets")
    
    has_data = any([slcci_data, cmems_data, dtu_data, datasets])
    
    if not has_data:
        # Show catalog-only view when no data loaded
        render_catalog_only_view()
        return
    
    # Main content tabs (all tabs when data is loaded)
    render_tabs(config)


def _handle_data_load(selection):
    """Handle data loading from the new selector."""
    logger.info(f"Data load requested: {selection.selected_datasets}")
    
    if not selection.confirmed:
        st.warning("Please confirm the data request")
        return
    
    # For now, if no datasets selected, just show message
    if not selection.selected_datasets:
        st.info("📁 Use **Local Files** or **Upload Files** in the sidebar to load NetCDF data for visualization.")
        return
    
    with st.spinner("Loading data..."):
        try:
            from src.services import DataService
            
            data_service = DataService()
            
            # Build request from selection
            datasets_loaded = []
            cycle_info = []
            
            for ds_id in selection.selected_datasets:
                variables = selection.selected_variables.get(ds_id, [])
                logger.debug(f"Loading dataset {ds_id} with variables {variables}")
                
                # Load dataset
                data = data_service.load_dataset(
                    dataset_id=ds_id,
                    bbox=selection.bbox,
                    time_range=selection.time_range,
                    variables=variables if variables else None
                )
                
                if data is not None:
                    # If it's an xarray dataset, add directly
                    import xarray as xr
                    if isinstance(data, xr.Dataset):
                        datasets_loaded.append(data)
                        cycle_info.append({
                            "filename": ds_id,
                            "cycle": len(datasets_loaded),
                            "path": ds_id,
                        })
                    else:
                        datasets_loaded.append({
                            "id": ds_id,
                            "data": data,
                            "variables": variables
                        })
            
            if datasets_loaded:
                st.session_state.datasets = datasets_loaded
                st.session_state.cycle_info = cycle_info
                st.session_state.current_selection = selection
                logger.info(f"Loaded {len(datasets_loaded)} dataset(s)")
                st.success(f"✅ Loaded {len(datasets_loaded)} dataset(s)")
                st.rerun()
            else:
                st.warning("No data loaded. Try loading local files from the sidebar.")
                
        except Exception as e:
            logger.error(f"Data load error: {e}", exc_info=True)
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


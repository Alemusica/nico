"""
BACKUP of tabs.py - 2026-01-03
Main content tabs for the dashboard.
Following SLCCI PLOTTER notebook workflow exactly.
Supports comparison mode with SLCCI/CMEMS overlay.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
import io

from .sidebar import AppConfig
from ..state import get_slcci_data, get_cmems_data, is_comparison_mode

# Comparison mode colors (from COMPARISON_BATCH notebook)
COLOR_SLCCI = "darkorange"
COLOR_CMEMS = "steelblue"


def render_tabs(config: AppConfig):
    """Render main content tabs based on loaded data type and comparison mode."""
    slcci_data = get_slcci_data()
    cmems_data = get_cmems_data()
    comparison_mode = is_comparison_mode()
    
    # Legacy support
    legacy_slcci = st.session_state.get("slcci_pass_data")
    datasets = st.session_state.get("datasets", {})
    selected_dataset_type = st.session_state.get("selected_dataset_type", "SLCCI")
    
    # Comparison mode: overlay SLCCI and CMEMS
    if comparison_mode and slcci_data is not None and cmems_data is not None:
        _render_comparison_tabs(slcci_data, cmems_data, config)
    # Single SLCCI mode
    elif slcci_data is not None:
        _render_slcci_tabs(slcci_data, config)
    # Single CMEMS mode  
    elif cmems_data is not None:
        _render_cmems_tabs(cmems_data, config)
    # Legacy SLCCI support
    elif selected_dataset_type == "SLCCI" and legacy_slcci is not None:
        _render_slcci_tabs(legacy_slcci, config)
    elif datasets:
        _render_generic_tabs(datasets, config)
    else:
        _render_empty_tabs(config)


def _render_empty_tabs(config: AppConfig):
    """Render welcome tabs when no data is loaded."""
    tab1, tab2 = st.tabs(["Welcome", "Help"])
    
    with tab1:
        st.markdown("## Welcome to NICO Dashboard")
        st.info("""
        **Getting Started:**
        
        1. **Select a Region** from the sidebar
        2. **Choose a Gate** for analysis
        3. **Load Data** using:
           - SLCCI local files (J2 satellite passes)
           - CMEMS/ERA5 via API
        
        Data will appear in the appropriate tabs once loaded.
        """)
        
        # Quick status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Datasets Loaded", len(st.session_state.get("datasets", {})))
        with col2:
            st.metric("Selected Gate", st.session_state.get("selected_gate", "None"))
        with col3:
            st.metric("Data Type", st.session_state.get("selected_dataset_type", "None"))
    
    with tab2:
        st.markdown("## Help")
        st.markdown("""
        ### SLCCI Data
        - Uses local NetCDF files from J2 satellite
        - Select pass number and cycle range
        - Shows slope, DOT profiles, and spatial maps
        
        ### Other Datasets
        - CMEMS: Ocean data via API
        - ERA5: Atmospheric reanalysis
        - Monthly aggregations and profiles
        """)


def _render_slcci_tabs(slcci_data: Dict[str, Any], config: AppConfig):
    """Render tabs for SLCCI satellite data."""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Slope Timeline", 
        "DOT Profile", 
        "Spatial Map", 
        "Monthly Analysis",
        "Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_slope_timeline(slcci_data, config)
    with tab2:
        _render_dot_profile(slcci_data, config)
    with tab3:
        _render_spatial_map(slcci_data, config)
    with tab4:
        _render_slcci_monthly_analysis(slcci_data, config)
    with tab5:
        _render_geostrophic_velocity(slcci_data, config)
    with tab6:
        _render_export_tab(slcci_data, None, config)


def _render_cmems_tabs(cmems_data, config: AppConfig):
    """Render tabs for CMEMS data only."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Slope Timeline",
        "DOT Profile",
        "Spatial Map",
        "Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_slope_timeline(cmems_data, config)
    with tab2:
        _render_dot_profile(cmems_data, config)
    with tab3:
        _render_spatial_map(cmems_data, config)
    with tab4:
        _render_geostrophic_velocity(cmems_data, config)
    with tab5:
        _render_export_tab(None, cmems_data, config)


# NOTE: This is a backup file. The full implementation continues in the original tabs.py
# This backup was created on 2026-01-03 during the audit session.

"""
Tabs Component
==============
Main content tabs for the dashboard.

Tab Structure:
- SLCCI Analysis tabs (3 separate): Slope Timeline, DOT Profile, Spatial Map
- Generic tabs: Dataset Catalog, Map View, Data Explorer
"""

import streamlit as st
import numpy as np
import pandas as pd

from .sidebar import AppConfig
from .analysis_tab import render_slope_timeline_tab
from .profiles_tab import render_profiles_tab
from .monthly_tab import render_monthly_tab
from .spatial_tab import render_spatial_tab
from .explorer_tab import render_explorer_tab
from .catalog_tab import render_catalog_tab
from .map_tab import render_map_tab

# SLCCI-specific tabs
from .slcci_slope_tab import render_slcci_slope_timeline_tab
from .slcci_profile_tab import render_slcci_dot_profile_tab
from .slcci_spatial_tab import render_slcci_spatial_map_tab


def render_tabs(config: AppConfig):
    """Render main content tabs."""
    
    # Check if SLCCI data is loaded
    slcci_data = st.session_state.get("slcci_pass_data")
    
    # Create tabs - SLCCI tabs are separate as requested
    tab_slcci_slope, tab_slcci_profile, tab_slcci_spatial, tab_catalog, tab_map, tab_explorer = st.tabs([
        "📈 Slope Timeline",
        "🌊 DOT Profile", 
        "�️ Spatial Map",
        "�️ Dataset Catalog",
        "🗺️ Map View",
        "📊 Data Explorer",
    ])
    
    datasets = st.session_state.datasets
    cycle_info = st.session_state.cycle_info
    
    # === SLCCI TABS (3 separate) ===
    with tab_slcci_slope:
        render_slcci_slope_timeline_tab(slcci_data)
    
    with tab_slcci_profile:
        render_slcci_dot_profile_tab(slcci_data)
    
    with tab_slcci_spatial:
        render_slcci_spatial_map_tab(slcci_data)
    
    # === GENERIC TABS ===
    with tab_catalog:
        render_catalog_tab()
    
    with tab_map:
        render_map_tab()
    
    with tab_explorer:
        render_explorer_tab(datasets, cycle_info, config)

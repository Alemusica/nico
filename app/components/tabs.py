"""
Tabs Component
==============
Main content tabs for the dashboard.
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


def render_tabs(config: AppConfig):
    """Render main content tabs."""
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗃️ Dataset Catalog",
        "🗺️ Map View",
        "📈 Slope Timeline",
        "🌊 DOT Profiles",
        "📅 Monthly Analysis",
        "🗺️ Spatial View",
        "📊 Data Explorer",
    ])
    
    datasets = st.session_state.datasets
    cycle_info = st.session_state.cycle_info
    
    with tab1:
        render_catalog_tab()
    
    with tab2:
        render_map_tab()
    
    with tab3:
        render_slope_timeline_tab(datasets, cycle_info, config)
    
    with tab4:
        render_profiles_tab(datasets, cycle_info, config)
    
    with tab5:
        render_monthly_tab(datasets, cycle_info, config)
    
    with tab6:
        render_spatial_tab(datasets, cycle_info, config)
    
    with tab7:
        render_explorer_tab(datasets, cycle_info, config)

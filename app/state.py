"""
Session State Management
========================
"""

import streamlit as st
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AppConfig:
    """Application configuration from sidebar."""
    # Gate selection
    selected_gate: Optional[str] = None
    gate_geometry: Any = None
    gate_buffer_km: float = 50.0
    
    # Data source
    selected_dataset_type: str = "SLCCI"
    data_source_mode: str = "local"  # "local" or "api"
    
    # === SLCCI SETTINGS ===
    slcci_base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
    slcci_geoid_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"
    
    # Pass selection (SLCCI only)
    pass_mode: str = "manual"
    pass_number: int = 248
    
    # Cycle range (SLCCI only)
    cycle_start: int = 1
    cycle_end: int = 10  # Default: primi 10 cicli per test veloci
    
    # Processing parameters (from SLCCI PLOTTER notebook)
    use_flag: bool = True           # Quality flag filter
    lon_bin_size: float = 0.01      # Longitude binning size (degrees)
    lat_buffer_deg: float = 2.0     # Latitude buffer for spatial filter
    lon_buffer_deg: float = 5.0     # Longitude buffer for spatial filter
    
    # === CMEMS SETTINGS ===
    cmems_base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/CMEMS_L3_1Hz"
    cmems_start_date: Any = None    # datetime.date
    cmems_end_date: Any = None      # datetime.date
    cmems_lon_bin_size: float = 0.10  # Coarser than SLCCI (0.05-0.50)
    cmems_buffer_deg: float = 0.5   # Buffer around gate


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "datasets": [],
        "cycle_info": [],
        "analysis_results": None,
        "config": {
            "mss_var": "mean_sea_surface",
            "bin_size": 0.01,
            "lat_range": None,
            "lon_range": None,
        },
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_datasets(datasets: list, cycle_info: list):
    """Update loaded datasets in session state."""
    st.session_state.datasets = datasets
    st.session_state.cycle_info = cycle_info


def get_datasets():
    """Get currently loaded datasets."""
    return st.session_state.get("datasets", [])


def get_cycle_info():
    """Get cycle information."""
    return st.session_state.get("cycle_info", [])


def clear_data():
    """Clear all loaded data."""
    st.session_state.datasets = []
    st.session_state.cycle_info = []
    st.session_state.analysis_results = None

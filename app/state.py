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
    selected_gate: Optional[str] = None
    gate_geometry: Any = None
    gate_buffer_km: float = 50.0
    selected_dataset_type: str = "SLCCI"
    slcci_base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
    slcci_geoid_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"
    pass_mode: str = "manual"
    pass_number: int = 248
    cycle_start: int = 1
    cycle_end: int = 10  # Default: primi 10 cicli per test veloci


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

"""
Sidebar Component
=================
Data loading and configuration sidebar.
"""

import os
import glob
import tempfile
from dataclasses import dataclass

import streamlit as st
import numpy as np
import xarray as xr

from ..state import update_datasets
from src.core.helpers import extract_cycle_number


@dataclass
class AppConfig:
    """Application configuration from sidebar."""
    mss_var: str = "mean_sea_surface"
    bin_size: float = 0.01
    lat_range: tuple | None = None
    lon_range: tuple | None = None
    use_spatial_filter: bool = False


def render_sidebar() -> AppConfig:
    """Render sidebar and return configuration."""
    
    st.sidebar.title("⚙️ Settings")
    
    # Data source
    data_source = st.sidebar.radio(
        "Data Source",
        ["📂 Local Files", "📤 Upload Files"],
    )
    
    if data_source == "📂 Local Files":
        _handle_local_files()
    else:
        _handle_uploaded_files()
    
    # Analysis parameters (only if data loaded)
    config = AppConfig()
    
    if st.session_state.get("datasets"):
        config = _render_analysis_params()
    
    return config


def _handle_local_files():
    """Handle loading files from local directory."""
    
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="/Users/alessioivoycazzaniga/nico",
    )
    
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading local files..."):
            datasets, cycle_info = _load_local_files(data_dir)
            
            if datasets:
                update_datasets(datasets, cycle_info)
                st.sidebar.success(f"✅ Loaded {len(datasets)} files")
            else:
                st.sidebar.error("❌ No files found")


def _handle_uploaded_files():
    """Handle drag & drop file upload."""
    
    uploaded_files = st.sidebar.file_uploader(
        "🎯 Drag & Drop NetCDF Files",
        type=["nc", "nc4", "netcdf"],
        accept_multiple_files=True,
        help="Upload SLCCI NetCDF satellite altimetry files",
    )
    
    if uploaded_files:
        with st.spinner(f"Loading {len(uploaded_files)} files..."):
            datasets = []
            cycle_info = []
            
            for f in uploaded_files:
                ds = _load_uploaded_file(f)
                if ds is not None:
                    cycle_num = extract_cycle_number(f.name)
                    datasets.append(ds)
                    cycle_info.append({
                        "filename": f.name,
                        "cycle": cycle_num,
                    })
            
            if datasets:
                update_datasets(datasets, cycle_info)
                st.sidebar.success(f"✅ Loaded {len(datasets)} files")


def _render_analysis_params() -> AppConfig:
    """Render analysis parameter controls."""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Parameters")
    
    # Reference surface
    ds_sample = st.session_state.datasets[0]
    available_vars = list(ds_sample.data_vars)
    
    mss_options = ["mean_sea_surface"]
    if "geoid" in available_vars:
        mss_options.append("geoid")
    
    mss_var = st.sidebar.selectbox("Reference Surface", mss_options)
    
    # Bin size
    bin_size = st.sidebar.slider(
        "Longitude Bin Size (°)",
        min_value=0.005,
        max_value=0.1,
        value=0.01,
        step=0.005,
    )
    
    # Spatial filter
    st.sidebar.subheader("🗺️ Spatial Filter")
    use_filter = st.sidebar.checkbox("Apply Spatial Filter")
    
    lat_range = None
    lon_range = None
    
    if use_filter:
        lat_data = ds_sample["latitude"].values.flatten()
        lon_data = ds_sample["longitude"].values.flatten()
        
        lat_min, lat_max = float(np.nanmin(lat_data)), float(np.nanmax(lat_data))
        lon_min, lon_max = float(np.nanmin(lon_data)), float(np.nanmax(lon_data))
        
        lat_range = st.sidebar.slider(
            "Latitude Range",
            lat_min, lat_max, (lat_min, lat_max),
        )
        lon_range = st.sidebar.slider(
            "Longitude Range",
            lon_min, lon_max, (lon_min, lon_max),
        )
    
    return AppConfig(
        mss_var=mss_var,
        bin_size=bin_size,
        lat_range=lat_range,
        lon_range=lon_range,
        use_spatial_filter=use_filter,
    )


def _load_local_files(data_dir: str) -> tuple[list, list]:
    """Load NetCDF files from local directory."""
    
    pattern = os.path.join(data_dir, "SLCCI_ALTDB_*.nc")
    files = sorted(glob.glob(pattern))
    
    if not files:
        return [], []
    
    datasets = []
    cycle_info = []
    
    for f in files:
        try:
            ds = xr.open_dataset(f)
            cycle_num = extract_cycle_number(os.path.basename(f))
            datasets.append(ds)
            cycle_info.append({
                "filename": os.path.basename(f),
                "cycle": cycle_num,
            })
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    
    return datasets, cycle_info


def _load_uploaded_file(file_obj) -> xr.Dataset | None:
    """Load NetCDF from uploaded file object."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            tmp.write(file_obj.getvalue())
            tmp_path = tmp.name
        
        ds = xr.open_dataset(tmp_path)
        os.unlink(tmp_path)
        return ds
    except Exception:
        return None

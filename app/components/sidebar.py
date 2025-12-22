"""
Sidebar Component
=================
Data loading and configuration sidebar.
"""

import os
import glob
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
import numpy as np
import xarray as xr

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

from ..state import update_datasets
from src.core.helpers import extract_cycle_number


# Gate definitions with human-readable names and coordinates
GATE_CATALOG = {
    "fram_strait": {
        "name": "🧊 Fram Strait",
        "file": "fram_strait_S3_pass_481.shp",
        "description": "Main Arctic-Atlantic exchange",
        "region": "Atlantic Sector",
    },
    "bering_strait": {
        "name": "🌊 Bering Strait", 
        "file": "bering_strait_TPJ_pass_076.shp",
        "description": "Pacific-Arctic gateway",
        "region": "Pacific Sector",
    },
    "davis_strait": {
        "name": "❄️ Davis Strait",
        "file": "davis_strait.shp",
        "description": "Baffin Bay - Labrador Sea",
        "region": "Atlantic Sector",
    },
    "denmark_strait": {
        "name": "🌀 Denmark Strait",
        "file": "denmark_strait_TPJ_pass_246.shp",
        "description": "Iceland-Greenland overflow",
        "region": "Atlantic Sector",
    },
    "nares_strait": {
        "name": "🏔️ Nares Strait",
        "file": "nares_strait.shp",
        "description": "Greenland-Ellesmere Island",
        "region": "Canadian Archipelago",
    },
    "lancaster_sound": {
        "name": "🚢 Lancaster Sound",
        "file": "lancaster_sound.shp",
        "description": "Northwest Passage entrance",
        "region": "Canadian Archipelago",
    },
    "barents_opening": {
        "name": "🌡️ Barents Opening",
        "file": "barents_sea_opening_S3_pass_481.shp",
        "description": "Atlantic water inflow",
        "region": "Atlantic Sector",
    },
    "norwegian_boundary": {
        "name": "🇳🇴 Norwegian Sea Boundary",
        "file": "norwegian_sea_boundary_TPJ_pass_220.shp",
        "description": "Atlantic-Nordic Seas",
        "region": "Atlantic Sector",
    },
}


@dataclass
class AppConfig:
    """Application configuration from sidebar."""
    mss_var: str = "mean_sea_surface"
    bin_size: float = 0.01
    lat_range: tuple | None = None
    lon_range: tuple | None = None
    use_spatial_filter: bool = False
    sample_fraction: float = 1.0  # 1.0 = no sampling
    # Gate configuration
    selected_gate: str | None = None
    gate_geometry: Any = None  # GeoDataFrame when loaded
    gate_buffer_km: float = 50.0  # Buffer around gate for data selection


def render_sidebar() -> AppConfig:
    """Render sidebar and return configuration."""
    
    st.sidebar.title("⚙️ Settings")
    
    # === GATE SELECTION (TOP PRIORITY) ===
    gate_info = _render_gate_selector()
    
    st.sidebar.divider()
    
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


def _render_gate_selector() -> dict | None:
    """Render gate selection UI with visual cards."""
    
    st.sidebar.subheader("🗺️ Select Ocean Gate")
    
    gates_dir = Path("/Users/alessioivoycazzaniga/nico/gates")
    
    if not gates_dir.exists():
        st.sidebar.warning("⚠️ Gates directory not found")
        return None
    
    # Group gates by region
    regions = {}
    for gate_id, info in GATE_CATALOG.items():
        region = info["region"]
        if region not in regions:
            regions[region] = []
        regions[region].append((gate_id, info))
    
    # Current selection
    current_gate = st.session_state.get("selected_gate")
    
    # Create selection with nice formatting
    gate_options = ["🌍 None (Global Analysis)"]
    gate_ids = [None]
    
    for region, gates in sorted(regions.items()):
        for gate_id, info in gates:
            gate_options.append(f"{info['name']}")
            gate_ids.append(gate_id)
    
    # Find current index
    current_idx = 0
    if current_gate and current_gate in gate_ids:
        current_idx = gate_ids.index(current_gate)
    
    selected_idx = st.sidebar.selectbox(
        "Choose a strait/gate",
        range(len(gate_options)),
        format_func=lambda i: gate_options[i],
        index=current_idx,
        help="Select an ocean strait to analyze"
    )
    
    selected_gate_id = gate_ids[selected_idx]
    
    # Show gate info card
    if selected_gate_id:
        gate_info = GATE_CATALOG[selected_gate_id]
        
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                    padding: 12px; border-radius: 8px; margin: 8px 0;
                    border-left: 4px solid #4fc3f7;">
            <div style="color: #4fc3f7; font-size: 0.8em; margin-bottom: 4px;">
                📍 {gate_info['region']}
            </div>
            <div style="color: white; font-size: 0.9em;">
                {gate_info['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Buffer slider
        buffer_km = st.sidebar.slider(
            "Buffer (km)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Area around the gate to include in analysis"
        )
        st.session_state["gate_buffer_km"] = buffer_km
        
        # Load gate geometry
        if selected_gate_id != current_gate:
            _load_gate_geometry(selected_gate_id, gates_dir)
    else:
        st.session_state["gate_geometry"] = None
    
    st.session_state["selected_gate"] = selected_gate_id
    
    return {"gate_id": selected_gate_id}


def _load_gate_geometry(gate_id: str, gates_dir: Path):
    """Load gate shapefile geometry."""
    
    if not HAS_GEOPANDAS:
        st.sidebar.warning("⚠️ Install geopandas for gate support: pip install geopandas")
        return
    
    gate_info = GATE_CATALOG.get(gate_id)
    if not gate_info:
        return
    
    gate_file = gates_dir / gate_info["file"]
    
    if not gate_file.exists():
        st.sidebar.warning(f"⚠️ Gate file not found: {gate_info['file']}")
        return
    
    try:
        # Set GDAL option to restore missing .shx files
        import os
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        
        # Try reading with pyogrio engine (default in geopandas 1.0+)
        try:
            gdf = gpd.read_file(gate_file, engine='pyogrio')
        except:
            # Fallback to fiona engine
            try:
                gdf = gpd.read_file(gate_file, engine='fiona')
            except:
                gdf = gpd.read_file(gate_file)
        
        st.session_state["gate_geometry"] = gdf
        
        # Show bounds
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy
        st.sidebar.caption(f"📐 Bounds: {bounds[0]:.1f}°E to {bounds[2]:.1f}°E, {bounds[1]:.1f}°N to {bounds[3]:.1f}°N")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading gate: {e}")


def _handle_local_files():
    """Handle loading files from local directory."""
    
    # Default to data/slcci subfolder - use absolute path
    default_dir = "/Users/alessioivoycazzaniga/nico/data/slcci"
    
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=default_dir,
    )
    
    # Show available files count
    pattern = os.path.join(data_dir, "*.nc")
    available_files = glob.glob(pattern)
    if available_files:
        st.sidebar.info(f"📁 Found {len(available_files)} NetCDF files")
    else:
        st.sidebar.warning(f"⚠️ No .nc files in: {data_dir}")
    
    if st.sidebar.button("🔄 Load All Data", type="primary"):
        with st.spinner(f"Loading {len(available_files)} files..."):
            datasets, cycle_info = _load_local_files(data_dir)
            
            if datasets:
                update_datasets(datasets, cycle_info)
                st.sidebar.success(f"✅ Loaded {len(datasets)} cycles")
            else:
                st.sidebar.error("❌ No NetCDF files found")


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
    
    # Performance settings
    st.sidebar.subheader("⚡ Performance")
    sample_pct = st.sidebar.slider(
        "Data Sampling %",
        min_value=1,
        max_value=100,
        value=10,
        help="Use less data for faster analysis (10% recommended for exploration)"
    )
    sample_fraction = sample_pct / 100.0
    
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
    
    # Get gate info from session state
    selected_gate = st.session_state.get("selected_gate")
    gate_geometry = st.session_state.get("gate_geometry")
    gate_buffer = st.session_state.get("gate_buffer_km", 50.0)
    
    return AppConfig(
        mss_var=mss_var,
        bin_size=bin_size,
        lat_range=lat_range,
        lon_range=lon_range,
        use_spatial_filter=use_filter,
        sample_fraction=sample_fraction,
        selected_gate=selected_gate,
        gate_geometry=gate_geometry,
        gate_buffer_km=gate_buffer,
    )


def _load_local_files(data_dir: str) -> tuple[list, list]:
    """Load NetCDF files from local directory."""
    
    # Try multiple patterns
    patterns = [
        os.path.join(data_dir, "SLCCI_ALTDB_*.nc"),
        os.path.join(data_dir, "*.nc"),
    ]
    
    files = []
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            break
    
    if not files:
        return [], []
    
    datasets = []
    cycle_info = []
    
    progress = st.progress(0)
    for i, f in enumerate(files):
        progress.progress((i + 1) / len(files))
        try:
            ds = xr.open_dataset(f)
            cycle_num = extract_cycle_number(os.path.basename(f))
            datasets.append(ds)
            cycle_info.append({
                "filename": os.path.basename(f),
                "cycle": cycle_num,
                "path": f,
            })
        except Exception as e:
            st.warning(f"⚠️ Could not load {os.path.basename(f)}: {e}")
    
    progress.empty()
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

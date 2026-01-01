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


# Initialize GateService once
try:
    from src.services import GateService
    _gate_service = GateService()
    GATE_SERVICE_AVAILABLE = True
except ImportError:
    _gate_service = None
    GATE_SERVICE_AVAILABLE = False


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
        key="data_source_radio_unique"
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
    
    # Use GateService if available
    if GATE_SERVICE_AVAILABLE and _gate_service:
        return _render_gate_selector_v2()
    
    # Fallback to static GATE_CATALOG
    gates_dir = Path(__file__).parent.parent.parent / "gates"
    
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
        help="Select an ocean strait to analyze",
        key="main_gate_selector_unique"
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


def _render_gate_selector_v2() -> dict | None:
    """
    Render gate selection using the new GateService.
    
    This version uses:
    - config/gates.yaml for gate metadata
    - Relative paths
    - GateService for all operations
    """
    # Get regions
    regions = _gate_service.get_regions()
    
    # Region filter (optional)
    selected_region = st.sidebar.selectbox(
        "🌍 Filter by Region",
        ["All Regions"] + regions,
        help="Filter gates by ocean region",
        key="gate_region_filter"
    )
    
    # Get gates
    if selected_region == "All Regions":
        gates = _gate_service.list_gates()
    else:
        gates = _gate_service.list_gates_by_region(selected_region)
    
    # Build options
    gate_options = ["🌍 None (Global Analysis)"]
    gate_ids = [None]
    
    for gate in gates:
        # Icon based on gate type
        icon = "🚪" if "strait" in gate.name.lower() else "🌊"
        gate_options.append(f"{icon} {gate.name}")
        gate_ids.append(gate.id)
    
    # Current selection
    current_gate = st.session_state.get("selected_gate")
    current_idx = 0
    if current_gate and current_gate in gate_ids:
        current_idx = gate_ids.index(current_gate)
    
    # Selector
    selected_idx = st.sidebar.selectbox(
        "Choose a Gate",
        range(len(gate_options)),
        format_func=lambda i: gate_options[i],
        index=current_idx,
        help="Select an ocean strait/gate to analyze",
        key="main_gate_selector_v2"
    )
    
    selected_gate_id = gate_ids[selected_idx]
    
    # Show gate info card
    if selected_gate_id:
        gate = _gate_service.get_gate(selected_gate_id)
        
        if gate:
            # Nice info card
            st.sidebar.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                        padding: 12px; border-radius: 8px; margin: 8px 0;
                        border-left: 4px solid #4fc3f7;">
                <div style="color: #4fc3f7; font-size: 0.8em; margin-bottom: 4px;">
                    📍 {gate.region}
                </div>
                <div style="color: white; font-size: 0.9em;">
                    {gate.description}
                </div>
                <div style="color: #90caf9; font-size: 0.75em; margin-top: 6px;">
                    Datasets: {', '.join(gate.datasets) if gate.datasets else 'Any'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Buffer slider
            buffer_km = st.sidebar.slider(
                "Buffer (km)",
                min_value=10,
                max_value=200,
                value=int(gate.default_buffer_km) if gate.default_buffer_km else 50,
                step=10,
                help="Area around the gate to include in analysis",
                key="gate_buffer_slider"
            )
            st.session_state["gate_buffer_km"] = buffer_km
            
            # Load geometry if available
            try:
                geometry = _gate_service.get_gate_geometry(selected_gate_id)
                st.session_state["gate_geometry"] = geometry
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load geometry: {e}")
                st.session_state["gate_geometry"] = None
            
            # === LOAD DATA BUTTON ===
            st.sidebar.markdown("---")
            if st.sidebar.button("🚀 Load Data for Gate", type="primary", key="load_gate_data_btn"):
                _load_data_for_gate(selected_gate_id, buffer_km)
            
            # === SLCCI DATA LOADER ===
            _render_slcci_loader(selected_gate_id, gate)
                
    else:
        st.session_state["gate_geometry"] = None
    
    st.session_state["selected_gate"] = selected_gate_id
    
    return {"gate_id": selected_gate_id}


def _load_data_for_gate(gate_id: str, buffer_km: float):
    """
    Load data for the selected gate using DataService.
    
    This wires the UI to the Services Layer as per NICO Unified Architecture:
    UI → DataService → Provider (CMEMS/ERA5/Local) → Data
    
    The DataService decides WHERE to get data based on:
    1. User selection (dataset_id in session_state)
    2. Gate's recommended datasets (gate.datasets)
    3. config/datasets.yaml provider configuration
    """
    from src.services import DataService, GateService
    from src.core.models import BoundingBox, TimeRange, DataRequest
    from ..state import update_datasets
    from datetime import datetime, timedelta
    
    try:
        gs = GateService()
        data_service = DataService()
        
        gate = gs.get_gate(gate_id)
        if not gate:
            st.sidebar.error(f"❌ Gate not found: {gate_id}")
            return
        
        bbox = gate.bbox
        if not bbox:
            st.sidebar.error(f"❌ Gate has no bounding box: {gate_id}")
            return
        
        # Get user's dataset selection (from catalog tab or default)
        selected_dataset = st.session_state.get("selected_dataset_id")
        
        # If no selection, use gate's recommended datasets
        if not selected_dataset and gate.datasets:
            selected_dataset = gate.datasets[0]  # First recommended
            st.sidebar.info(f"📊 Using gate's recommended dataset: {selected_dataset}")
        
        # Fallback to demo (no API required)
        if not selected_dataset:
            selected_dataset = "demo"
            st.sidebar.info(f"📊 Using demo data (no API key needed)")
        
        # Time range (default: last 100 days for multiple cycles)
        time_start = st.session_state.get("time_start", datetime.now() - timedelta(days=100))
        time_end = st.session_state.get("time_end", datetime.now())
        
        # Build request following architecture
        request = data_service.build_request(
            gate=gate,
            bbox=bbox,
            time_range=TimeRange(start=time_start, end=time_end),
            variables=st.session_state.get("selected_variables", []),
            dataset_id=selected_dataset
        )
        
        # Show loading state
        with st.spinner(f"🔄 Loading {selected_dataset} for {gate.name}..."):
            # For demo data, use multi-cycle loader for proper timeline
            if "demo" in selected_dataset.lower():
                datasets, cycle_info = data_service.load_multi_cycle_demo(request, n_cycles=10)
                if datasets:
                    update_datasets(datasets, cycle_info)
                    st.sidebar.success(f"✅ Loaded {len(datasets)} demo cycles for {gate.name}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Could not generate demo data")
                return
            
            # DataService routes to correct provider based on dataset_id
            data = data_service.load(request)
            
            if data is not None:
                # Convert to list format expected by tabs
                if hasattr(data, 'dims'):  # xarray Dataset
                    datasets = [data]
                    cycle_info = [{
                        "filename": f"{selected_dataset}_{gate_id}",
                        "cycle": data.attrs.get("cycle", 1),
                        "pass": data.attrs.get("pass", None),
                        "path": selected_dataset,
                        "n_points": data.sizes.get("time", 0)
                    }]
                else:
                    datasets = [data]
                    cycle_info = [{"filename": selected_dataset, "cycle": 1}]
                
                update_datasets(datasets, cycle_info)
                st.sidebar.success(f"✅ Loaded data for {gate.name}")
                st.rerun()
            else:
                # No data from provider - try multi-cycle demo
                st.sidebar.warning(f"⚠️ No data from {selected_dataset}. Generating demo cycles...")
                datasets, cycle_info = data_service.load_multi_cycle_demo(request, n_cycles=10)
                if datasets:
                    update_datasets(datasets, cycle_info)
                    st.sidebar.success(f"✅ Generated {len(datasets)} demo cycles for {gate.name}")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Could not load or generate data")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())


def _render_slcci_loader(gate_id: str, gate):
    """
    Render SLCCI (ESA Sea Level CCI) data loader section.
    
    This loads data from local SLCCI NetCDF files using the SLCCIService.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛰️ SLCCI Data (ESA CCI)")
    
    # Check if SLCCI service is available
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig
        slcci_available = True
    except ImportError as e:
        st.sidebar.warning(f"⚠️ SLCCI Service not available: {e}")
        slcci_available = False
        return
    
    # Configuration
    base_dir = st.sidebar.text_input(
        "SLCCI Data Directory",
        value="/Users/nicolocaron/Desktop/ARCFRESH/J2",
        key="slcci_base_dir",
        help="Path to folder containing SLCCI_ALTDB_*.nc files"
    )
    
    geoid_path = st.sidebar.text_input(
        "Geoid File (TUM_ogmoc.nc)",
        value="/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc",
        key="slcci_geoid_path",
        help="Path to TUM geoid file for DOT calculation"
    )
    
    # Pass selection
    st.sidebar.markdown("**Pass Selection**")
    
    pass_mode = st.sidebar.radio(
        "Pass Mode",
        ["🔍 Auto-detect", "📝 Manual"],
        key="slcci_pass_mode",
        horizontal=True,
    )
    
    # Get gate shapefile path
    gates_dir = Path(__file__).parent.parent.parent / "gates"
    gate_shp = None
    
    # Try to find gate shapefile
    if gate and hasattr(gate, 'shapefile'):
        gate_shp = gates_dir / gate.shapefile
    elif gate_id:
        # Search for matching shapefile
        import glob
        patterns = [
            f"*{gate_id}*.shp",
            f"*{gate_id.replace('_', '-')}*.shp",
            f"*{gate_id.replace('_', ' ')}*.shp",
        ]
        for pattern in patterns:
            matches = list(gates_dir.glob(pattern))
            if matches:
                gate_shp = matches[0]
                break
    
    if gate_shp and gate_shp.exists():
        st.sidebar.caption(f"📁 Gate file: {gate_shp.name}")
    else:
        st.sidebar.warning("⚠️ Gate shapefile not found")
        gate_shp = st.sidebar.text_input(
            "Gate Shapefile Path",
            key="slcci_gate_path",
            help="Full path to gate shapefile"
        )
        if gate_shp:
            gate_shp = Path(gate_shp)
    
    # Pass number input
    if pass_mode == "📝 Manual":
        pass_number = st.sidebar.number_input(
            "Pass Number",
            min_value=1,
            max_value=500,
            value=248,
            key="slcci_pass_number",
            help="Satellite pass number to load"
        )
    else:
        pass_number = None
        st.sidebar.caption("Will find closest pass to gate automatically")
    
    # Cycle range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cycle_start = st.number_input("Start Cycle", min_value=1, max_value=300, value=1, key="slcci_cycle_start")
    with col2:
        cycle_end = st.number_input("End Cycle", min_value=1, max_value=300, value=100, key="slcci_cycle_end")
    
    # Load button
    if st.sidebar.button("🚀 Load SLCCI Data", type="primary", key="load_slcci_btn"):
        _load_slcci_data(
            gate_path=str(gate_shp) if gate_shp else None,
            pass_number=pass_number,
            base_dir=base_dir,
            geoid_path=geoid_path,
            cycles=list(range(int(cycle_start), int(cycle_end) + 1)),
        )


def _load_slcci_data(
    gate_path: str,
    pass_number: int | None,
    base_dir: str,
    geoid_path: str,
    cycles: list,
):
    """Load SLCCI data using SLCCIService."""
    from src.services.slcci_service import SLCCIService, SLCCIConfig
    
    if not gate_path or not Path(gate_path).exists():
        st.sidebar.error("❌ Gate shapefile path is required")
        return
    
    if not Path(base_dir).exists():
        st.sidebar.error(f"❌ SLCCI data directory not found: {base_dir}")
        return
    
    if not Path(geoid_path).exists():
        st.sidebar.error(f"❌ Geoid file not found: {geoid_path}")
        return
    
    # Initialize service
    config = SLCCIConfig(
        base_dir=base_dir,
        geoid_path=geoid_path,
        cycles=cycles,
        use_flag=True,
    )
    service = SLCCIService(config)
    
    with st.spinner("🔄 Loading SLCCI data..."):
        try:
            # Auto-detect pass if not specified
            if pass_number is None:
                st.sidebar.info("🔍 Finding closest passes to gate...")
                closest_passes = service.find_closest_pass(gate_path, n_passes=5)
                
                if closest_passes:
                    st.sidebar.success(f"✅ Found {len(closest_passes)} passes")
                    
                    # Show dropdown to select pass
                    pass_options = [f"Pass {p[0]} ({p[1]:.1f} km)" for p in closest_passes]
                    selected_pass_idx = st.sidebar.selectbox(
                        "Select Pass",
                        range(len(pass_options)),
                        format_func=lambda i: pass_options[i],
                        key="slcci_closest_pass_select"
                    )
                    pass_number = closest_passes[selected_pass_idx][0]
                else:
                    st.sidebar.error("❌ No passes found near gate")
                    return
            
            # Load pass data
            st.sidebar.info(f"📊 Loading pass {pass_number}...")
            pass_data = service.load_pass_data(
                gate_path=gate_path,
                pass_number=pass_number,
                cycles=cycles,
            )
            
            if pass_data is None:
                st.sidebar.error(f"❌ No data found for pass {pass_number}")
                return
            
            # Store in session state
            st.session_state["slcci_pass_data"] = pass_data
            st.session_state["slcci_service"] = service
            
            # Success message
            st.sidebar.success(
                f"✅ Loaded {len(pass_data.df):,} observations\n"
                f"• Pass: {pass_data.pass_number}\n"
                f"• Cycles: {pass_data.df['cycle'].nunique()}\n"
                f"• Strait: {pass_data.strait_name}"
            )
            
            # Rerun to update tabs
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading SLCCI data: {e}")
            import traceback
            st.sidebar.code(traceback.format_exc())


def _generate_demo_data(gate, buffer_km: float):
    """Generate demo data for testing when no real data is available."""
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    bbox = gate.bbox
    if not bbox:
        return [], []
    
    # Generate 5 demo cycles
    datasets = []
    cycle_info = []
    
    for cycle in range(1, 6):
        # Random points in bbox
        n_points = 500
        
        buffer_deg = buffer_km / 111.0
        lats = np.random.uniform(
            bbox.lat_min - buffer_deg,
            bbox.lat_max + buffer_deg,
            n_points
        )
        lons = np.random.uniform(
            bbox.lon_min - buffer_deg,
            bbox.lon_max + buffer_deg,
            n_points
        )
        
        # Generate DOT-like values
        base_dot = -0.5 + 0.1 * np.sin(2 * np.pi * lons / 30)
        noise = np.random.randn(n_points) * 0.05
        corssh = base_dot + noise
        mss = np.zeros(n_points)  # Reference surface
        
        # Create times
        base_time = pd.Timestamp("2020-01-01") + pd.Timedelta(days=10 * cycle)
        times = [base_time + pd.Timedelta(seconds=i) for i in range(n_points)]
        
        ds = xr.Dataset(
            {
                "corssh": (["time"], corssh),
                "mean_sea_surface": (["time"], mss),
                "latitude": (["time"], lats),
                "longitude": (["time"], lons),
            },
            coords={"time": times},
            attrs={
                "cycle": cycle,
                "source": "demo_data",
                "gate": gate.id,
            }
        )
        
        datasets.append(ds)
        cycle_info.append({
            "filename": f"demo_cycle_{cycle}.nc",
            "cycle": cycle,
            "path": "demo",
            "n_points": n_points
        })
    
    return datasets, cycle_info


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
    
    # Get gate info from session state
    selected_gate = st.session_state.get("selected_gate")
    gate_geometry = st.session_state.get("gate_geometry")
    gate_buffer = st.session_state.get("gate_buffer_km", 50.0)
    
    # Auto-apply gate bbox if a gate is selected
    gate_bbox = None
    if selected_gate and GATE_SERVICE_AVAILABLE and _gate_service:
        try:
            gate = _gate_service.get_gate(selected_gate)
            if gate and gate.bbox:
                gate_bbox = gate.bbox
                st.sidebar.success(f"🎯 Using gate bounds: {gate.name}")
        except Exception:
            pass
    
    # If gate provides bbox, use it as default filter
    if gate_bbox:
        # Apply buffer (approximate: 1 degree ≈ 111 km)
        buffer_deg = gate_buffer / 111.0
        
        default_lat_min = gate_bbox.lat_min - buffer_deg
        default_lat_max = gate_bbox.lat_max + buffer_deg
        default_lon_min = gate_bbox.lon_min - buffer_deg
        default_lon_max = gate_bbox.lon_max + buffer_deg
        
        st.sidebar.info(f"📍 Gate bbox + {gate_buffer}km buffer")
        
        # Show gate filter is active
        use_filter = st.sidebar.checkbox("Apply Gate Filter", value=True, key="gate_filter_checkbox")
        
        if use_filter:
            lat_range = (default_lat_min, default_lat_max)
            lon_range = (default_lon_min, default_lon_max)
            
            # Show the applied ranges
            st.sidebar.caption(f"Lat: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°")
            st.sidebar.caption(f"Lon: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°")
        else:
            lat_range = None
            lon_range = None
            use_filter = False
    else:
        # Manual spatial filter (when no gate selected)
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
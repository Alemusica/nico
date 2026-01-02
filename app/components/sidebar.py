"""
Sidebar Component - SLCCI Analysis
===================================
Controls following SLCCI PLOTTER notebook workflow:
1. Gate Selection (region + gate)
2. Data Paths (SLCCI files + geoid)
3. Pass Selection (auto/manual)
4. Cycle Range
5. Processing Parameters (binning, flags)
"""

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from ..state import AppConfig


# Initialize GateService
try:
    from src.services import GateService
    _gate_service = GateService()
    GATE_SERVICE_AVAILABLE = True
except ImportError:
    _gate_service = None
    GATE_SERVICE_AVAILABLE = False


def render_sidebar() -> AppConfig:
    """
    Render sidebar following SLCCI PLOTTER notebook workflow.
    
    Returns AppConfig with all settings.
    """
    st.sidebar.title("Settings")
    
    config = AppConfig()
    
    # === 1. GATE SELECTION ===
    st.sidebar.subheader("1. Gate Selection")
    config = _render_gate_selection(config)
    
    st.sidebar.divider()
    
    # === 2. DATA SOURCE ===
    st.sidebar.subheader("2. Data Source")
    config = _render_data_source(config)
    
    # Only show SLCCI options if SLCCI selected
    if config.selected_dataset_type == "SLCCI":
        st.sidebar.divider()
        
        # === 3. DATA PATHS ===
        st.sidebar.subheader("3. Data Paths")
        config = _render_data_paths(config)
        
        st.sidebar.divider()
        
        # === 4. PASS SELECTION ===
        st.sidebar.subheader("4. Pass Selection")
        config = _render_pass_selection(config)
        
        st.sidebar.divider()
        
        # === 5. CYCLE RANGE ===
        st.sidebar.subheader("5. Cycle Range")
        config = _render_cycle_range(config)
        
        st.sidebar.divider()
        
        # === 6. PROCESSING PARAMETERS ===
        with st.sidebar.expander("6. Processing Parameters", expanded=False):
            config = _render_processing_params(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("Load SLCCI Data", type="primary", use_container_width=True):
            _load_slcci_data(config)
    else:
        # Non-SLCCI data sources
        st.sidebar.divider()
        if st.sidebar.button("Load Data", type="primary", use_container_width=True):
            _load_generic_data(config)
    
    return config


def _render_gate_selection(config: AppConfig) -> AppConfig:
    """Render region filter and gate selector."""
    
    if not GATE_SERVICE_AVAILABLE:
        st.sidebar.warning("Gate service not available")
        return config
    
    # Region filter
    regions = _gate_service.get_regions()
    selected_region = st.sidebar.selectbox(
        "Region",
        ["All Regions"] + regions,
        key="sidebar_region"
    )
    
    # Get gates for selected region
    if selected_region == "All Regions":
        gates = _gate_service.list_gates()
    else:
        gates = _gate_service.list_gates_by_region(selected_region)
    
    # Gate selector
    gate_options = ["None (Global)"] + [g.name for g in gates]
    gate_ids = [None] + [g.id for g in gates]
    
    selected_idx = st.sidebar.selectbox(
        "Gate",
        range(len(gate_options)),
        format_func=lambda i: gate_options[i],
        key="sidebar_gate"
    )
    
    config.selected_gate = gate_ids[selected_idx]
    
    # Show gate info
    if config.selected_gate:
        gate = _gate_service.get_gate(config.selected_gate)
        if gate:
            st.sidebar.caption(f"📍 {gate.region} - {gate.description}")
    
    # Buffer
    config.gate_buffer_km = st.sidebar.slider(
        "Buffer (km)", 10, 200, 50, 10,
        key="sidebar_buffer"
    )
    
    return config


def _render_data_source(config: AppConfig) -> AppConfig:
    """Render data source selector."""
    
    config.selected_dataset_type = st.sidebar.radio(
        "Source",
        ["SLCCI", "CMEMS", "ERA5"],
        horizontal=True,
        key="sidebar_datasource",
        help="SLCCI=Local NetCDF files, CMEMS/ERA5=API"
    )
    
    return config


def _render_data_paths(config: AppConfig) -> AppConfig:
    """Render SLCCI data path inputs."""
    
    config.slcci_base_dir = st.sidebar.text_input(
        "SLCCI Data Directory",
        value="/Users/nicolocaron/Desktop/ARCFRESH/J2",
        key="sidebar_slcci_dir",
        help="Folder containing SLCCI_ALTDB_J2_CycleXXX_V2.nc files"
    )
    
    config.slcci_geoid_path = st.sidebar.text_input(
        "Geoid File",
        value="/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc",
        key="sidebar_geoid",
        help="TUM_ogmoc.nc for DOT calculation"
    )
    
    return config


def _render_pass_selection(config: AppConfig) -> AppConfig:
    """Render pass selection mode and number."""
    
    config.pass_mode = st.sidebar.radio(
        "Mode",
        ["manual", "auto"],
        format_func=lambda x: "Manual" if x == "manual" else "Auto-detect",
        horizontal=True,
        key="sidebar_pass_mode",
        help="Auto finds closest pass to gate"
    )
    
    if config.pass_mode == "manual":
        config.pass_number = st.sidebar.number_input(
            "Pass Number",
            min_value=1,
            max_value=500,
            value=248,
            key="sidebar_pass_num",
            help="J2 satellite pass number (e.g., 248 for Davis)"
        )
    
    return config


def _render_cycle_range(config: AppConfig) -> AppConfig:
    """Render cycle range inputs."""
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        config.cycle_start = st.number_input(
            "Start",
            min_value=1,
            max_value=300,
            value=1,
            key="sidebar_cycle_start"
        )
    
    with col2:
        config.cycle_end = st.number_input(
            "End",
            min_value=1,
            max_value=300,
            value=10,  # Default 10 for fast testing
            key="sidebar_cycle_end"
        )
    
    # Show info
    n_cycles = config.cycle_end - config.cycle_start + 1
    st.sidebar.caption(f"📊 {n_cycles} cycles selected")
    
    return config


def _render_processing_params(config: AppConfig) -> AppConfig:
    """Render advanced processing parameters from SLCCI PLOTTER."""
    
    # Quality flag filter
    config.use_flag = st.checkbox(
        "Use Quality Flags",
        value=True,
        key="sidebar_use_flag",
        help="Filter data using SLCCI quality flags"
    )
    
    # Longitude binning size - SLIDER da 0.01 a 0.1
    config.lon_bin_size = st.slider(
        "Lon Bin Size (°)",
        min_value=0.01,
        max_value=0.10,
        value=0.01,
        step=0.01,
        format="%.2f",
        key="sidebar_lon_bin",
        help="Binning resolution for slope calculation (0.01° - 0.10°)"
    )
    
    return config


def _load_slcci_data(config: AppConfig):
    """Load SLCCI data using SLCCIService."""
    
    # Validate paths
    if not Path(config.slcci_base_dir).exists():
        st.sidebar.error(f"❌ Path not found: {config.slcci_base_dir}")
        return
    
    if not Path(config.slcci_geoid_path).exists():
        st.sidebar.error(f"❌ Geoid not found: {config.slcci_geoid_path}")
        return
    
    # Get gate shapefile
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ Select a gate first")
        return
    
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig
        
        cycles = list(range(config.cycle_start, config.cycle_end + 1))
        
        slcci_config = SLCCIConfig(
            base_dir=config.slcci_base_dir,
            geoid_path=config.slcci_geoid_path,
            cycles=cycles,
            use_flag=config.use_flag,
            lat_buffer_deg=config.lat_buffer_deg,
            lon_buffer_deg=config.lon_buffer_deg,
        )
        
        service = SLCCIService(slcci_config)
        
        with st.spinner(f"Loading {len(cycles)} cycles..."):
            
            # Determine pass number
            pass_number = config.pass_number
            
            if config.pass_mode == "auto":
                st.sidebar.info("🔍 Finding closest pass...")
                closest = service.find_closest_pass(gate_path, n_passes=1)
                if closest:
                    pass_number = closest[0][0]
                    st.sidebar.success(f"Found pass {pass_number}")
                else:
                    st.sidebar.error("No passes found near gate")
                    return
            
            # Load pass data
            pass_data = service.load_pass_data(
                gate_path=gate_path,
                pass_number=pass_number,
                cycles=cycles,
            )
            
            if pass_data is None:
                st.sidebar.error(f"❌ No data for pass {pass_number}")
                return
            
            # Store in session state
            st.session_state["slcci_pass_data"] = pass_data
            st.session_state["slcci_service"] = service
            st.session_state["slcci_config"] = config
            st.session_state["datasets"] = {}  # Clear generic
            
            # Success message
            n_obs = len(pass_data.df) if hasattr(pass_data, 'df') else 0
            n_cyc = pass_data.df['cycle'].nunique() if hasattr(pass_data, 'df') and 'cycle' in pass_data.df.columns else 0
            
            st.sidebar.success(f"""
            ✅ Data Loaded!
            - Pass: {pass_number}
            - Observations: {n_obs:,}
            - Cycles: {n_cyc}
            """)
            
            st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"❌ Service not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


def _load_generic_data(config: AppConfig):
    """Load data from CMEMS or ERA5 API."""
    
    if not config.selected_gate:
        st.sidebar.warning("Select a gate first")
        return
    
    st.sidebar.info(f"Loading {config.selected_dataset_type}... (not implemented yet)")


def _get_gate_shapefile(gate_id: Optional[str]) -> Optional[str]:
    """Get path to gate shapefile."""
    
    if not gate_id:
        return None
    
    gates_dir = Path(__file__).parent.parent.parent / "gates"
    
    if not gates_dir.exists():
        return None
    
    # Try GateService first
    if GATE_SERVICE_AVAILABLE and _gate_service:
        gate = _gate_service.get_gate(gate_id)
        if gate and hasattr(gate, 'shapefile') and gate.shapefile:
            shp_path = gates_dir / gate.shapefile
            if shp_path.exists():
                return str(shp_path)
    
    # Fallback: search by pattern
    patterns = [
        f"*{gate_id}*.shp",
        f"*{gate_id.replace('_', '-')}*.shp",
        f"*{gate_id.replace('_', ' ')}*.shp",
    ]
    
    for pattern in patterns:
        matches = list(gates_dir.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None

"""
Sidebar Component - SLCCI & CMEMS Analysis
===========================================
Controls following SLCCI PLOTTER notebook workflow:
1. Gate Selection (region + gate)
2. Data Source Selection (SLCCI/CMEMS with comparison mode)
3. Data Paths (SLCCI files + geoid / CMEMS folders)
4. Pass Selection (auto/manual for SLCCI, from filename for CMEMS)
5. Cycle Range (SLCCI only)
6. Processing Parameters (binning, flags)

Comparison Mode:
- Load SLCCI and CMEMS separately
- Toggle comparison to overlay plots
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import streamlit as st

from ..state import (
    AppConfig, 
    store_slcci_data, 
    store_cmems_data, 
    is_comparison_mode, 
    set_comparison_mode,
    store_dtu_data,
    get_dtu_data
)


# ============================================================
# PASS/TRACK EXTRACTION HELPERS
# ============================================================

def _extract_pass_from_gate_name(gate_name: str) -> Optional[int]:
    """
    Extract pass/track number from gate shapefile name.
    
    Patterns supported:
    - *_TPJ_pass_XXX  (J1/J2/J3 = TOPEX/Poseidon-Jason)
    - *_S3_pass_XXX   (Sentinel-3)
    - Trailing _XXX   (generic fallback)
    
    Args:
        gate_name: Gate name or shapefile path
        
    Returns:
        Pass number as int, or None if not found
    """
    if not gate_name:
        return None
    
    # Get just the filename stem
    name = Path(gate_name).stem if "/" in gate_name or "\\" in gate_name else gate_name
    
    # Pattern 1: _TPJ_pass_XXX (J1/J2/J3)
    match = re.search(r"_TPJ_pass_(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 2: _S3_pass_XXX (Sentinel-3)
    match = re.search(r"_S3_pass_(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 3: Generic trailing _XXX (3+ digits at end)
    match = re.search(r"_(\d{3,})$", name)
    if match:
        return int(match.group(1))
    
    return None


def _extract_satellite_from_gate_name(gate_name: str) -> Optional[str]:
    """
    Extract satellite type from gate name.
    
    Returns:
        'J2' for TPJ (Jason-2), 'S3' for Sentinel-3, or None
    """
    if not gate_name:
        return None
    
    name = Path(gate_name).stem if "/" in gate_name or "\\" in gate_name else gate_name
    
    if "_TPJ_" in name.upper():
        return "J2"  # TOPEX/Poseidon-Jason series
    elif "_S3_" in name.upper():
        return "S3"  # Sentinel-3
    
    return None


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
        
        # === 66°N WARNING ===
        _render_latitude_warning(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("Load SLCCI Data", type="primary", use_container_width=True):
            _load_slcci_data(config)
    
    elif config.selected_dataset_type == "CMEMS":
        # === CMEMS-SPECIFIC OPTIONS ===
        st.sidebar.divider()
        
        # === 3. DATA PATHS (includes source mode) ===
        st.sidebar.subheader("3. CMEMS Data Paths")
        config = _render_cmems_paths(config)
        
        st.sidebar.divider()
        
        # === 4. PROCESSING PARAMETERS ===
        with st.sidebar.expander("4. Processing Parameters", expanded=False):
            config = _render_cmems_params(config)
        
        # === 66°N WARNING ===
        _render_latitude_warning(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("Load CMEMS Data", type="primary", use_container_width=True):
            _load_cmems_data(config)
    
    elif config.selected_dataset_type == "DTUSpace":
        # === DTUSpace-SPECIFIC OPTIONS (ISOLATED - GRIDDED PRODUCT) ===
        st.sidebar.divider()
        
        # === 3. DATA PATH ===
        st.sidebar.subheader("3. DTUSpace Data Path")
        config = _render_dtu_paths(config)
        
        st.sidebar.divider()
        
        # === 4. TIME RANGE ===
        st.sidebar.subheader("4. Time Range")
        config = _render_dtu_time_range(config)
        
        st.sidebar.divider()
        
        # === LOAD BUTTON ===
        if st.sidebar.button("🟢 Load DTUSpace Data", type="primary", use_container_width=True):
            _load_dtu_data(config)
    
    else:
        # Non-SLCCI/CMEMS data sources (ERA5, etc.)
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
    """Render data source selector with comparison mode option."""
    
    # Main dataset selector
    config.selected_dataset_type = st.sidebar.radio(
        "Dataset",
        ["SLCCI", "CMEMS", "DTUSpace", "ERA5"],
        horizontal=True,
        key="sidebar_datasource",
        help="SLCCI=Sea Level CCI (J2), CMEMS=Copernicus L3 1Hz, DTUSpace=Gridded DOT"
    )
    
    # Store in session state for tabs
    st.session_state["selected_dataset_type"] = config.selected_dataset_type
    
    # === COMPARISON MODE ===
    st.sidebar.divider()
    
    # Check if both datasets are loaded
    slcci_loaded = st.session_state.get("dataset_slcci") is not None
    cmems_loaded = st.session_state.get("dataset_cmems") is not None
    
    if slcci_loaded or cmems_loaded:
        st.sidebar.markdown("**📊 Loaded Data:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if slcci_loaded:
                slcci_data = st.session_state.get("dataset_slcci")
                st.success(f"✅ SLCCI\nPass {getattr(slcci_data, 'pass_number', '?')}")
            else:
                st.info("⬜ SLCCI")
        with col2:
            if cmems_loaded:
                cmems_data = st.session_state.get("dataset_cmems")
                pass_num = getattr(cmems_data, 'pass_number', None)
                pass_str = f"Pass {pass_num}" if pass_num else "Synthetic"
                st.success(f"✅ CMEMS\n{pass_str}")
            else:
                st.info("⬜ CMEMS")
        
        # Comparison toggle (only if both loaded)
        if slcci_loaded and cmems_loaded:
            comparison_enabled = st.sidebar.checkbox(
                "🔀 **Comparison Mode**",
                value=is_comparison_mode(),
                key="sidebar_comparison_mode",
                help="Overlay SLCCI and CMEMS plots for comparison"
            )
            set_comparison_mode(comparison_enabled)
            config.comparison_mode = comparison_enabled
            
            if comparison_enabled:
                st.sidebar.success("✅ Comparison mode: Plots will overlay both datasets")
    
    # For SLCCI, add LOCAL/API selector
    if config.selected_dataset_type == "SLCCI":
        config.data_source_mode = st.sidebar.radio(
            "Source Mode",
            ["local", "api"],
            format_func=lambda x: "📁 LOCAL Files" if x == "local" else "🌐 CEDA API",
            horizontal=True,
            key="sidebar_source_mode",
            help="LOCAL=NetCDF files on disk, API=CEDA OPeNDAP"
        )
        
        if config.data_source_mode == "api":
            st.sidebar.caption("⚡ Faster downloads with bbox filtering")
    
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
    """Render pass selection mode and number with suggested pass from gate."""
    
    # Try to extract suggested pass from gate shapefile name
    suggested_pass = None
    suggested_satellite = None
    gate_path = None  # Initialize here for use later
    
    if config.selected_gate:
        gate_path = _get_gate_shapefile(config.selected_gate)
        if gate_path:
            suggested_pass = _extract_pass_from_gate_name(gate_path)
            suggested_satellite = _extract_satellite_from_gate_name(gate_path)
    
    # Show suggested pass if found
    if suggested_pass:
        sat_label = f" ({suggested_satellite})" if suggested_satellite else ""
        st.sidebar.success(f"🎯 **Suggested Pass: {suggested_pass}**{sat_label}")
        st.sidebar.caption("Extracted from gate shapefile name")
    
    # Build mode options based on what's available
    mode_options = ["suggested", "closest", "manual"] if suggested_pass else ["closest", "manual"]
    
    config.pass_mode = st.sidebar.radio(
        "Mode",
        mode_options,
        format_func=lambda x: {
            "suggested": f"🎯 Suggested ({suggested_pass})" if suggested_pass else "Suggested",
            "closest": "🔍 5 Closest",
            "manual": "✏️ Manual"
        }.get(x, x),
        horizontal=True,
        key="sidebar_pass_mode",
        help="Suggested=from gate name, Closest=5 nearest passes, Manual=enter number"
    )
    
    if config.pass_mode == "suggested" and suggested_pass:
        config.pass_number = suggested_pass
        st.sidebar.info(f"Using pass **{suggested_pass}** from gate definition")
        
    elif config.pass_mode == "closest" and gate_path:
        # Find and show 5 closest passes
        try:
            from src.services.slcci_service import SLCCIService, SLCCIConfig
            temp_config = SLCCIConfig(base_dir=config.data_dir)
            temp_service = SLCCIService(temp_config)
            
            # Cache closest passes in session state
            cache_key = f"slcci_closest_{config.selected_gate}"
            if cache_key not in st.session_state:
                with st.spinner("Finding closest passes..."):
                    st.session_state[cache_key] = temp_service.find_closest_pass(gate_path, n_passes=5)
            
            closest_passes = st.session_state[cache_key]
            
            if closest_passes and closest_passes[0][1] != float('inf'):
                # Build options: list of (pass_num, distance_km)
                pass_options = [p[0] for p in closest_passes]
                pass_labels = [f"Pass {p[0]} ({p[1]:.1f} km)" for p in closest_passes]
                
                selected_idx = st.sidebar.selectbox(
                    "Select Pass",
                    range(len(pass_options)),
                    format_func=lambda i: pass_labels[i],
                    key="sidebar_slcci_closest_pass",
                    help="Passes sorted by distance to gate centroid"
                )
                
                config.pass_number = pass_options[selected_idx]
                st.sidebar.success(f"🎯 Pass {config.pass_number} selected")
            else:
                st.sidebar.warning("No passes found near this gate")
                config.pass_number = 248  # Default fallback
        except Exception as e:
            st.sidebar.error(f"Error finding passes: {e}")
            config.pass_number = 248
            
    elif config.pass_mode == "manual":
        config.pass_number = st.sidebar.number_input(
            "Pass Number",
            min_value=1,
            max_value=500,
            value=suggested_pass if suggested_pass else 248,
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


# ============================================================
# CMEMS-SPECIFIC FUNCTIONS
# ============================================================

def _render_cmems_paths(config: AppConfig) -> AppConfig:
    """Render CMEMS data path inputs and source mode."""
    
    # Source mode (like SLCCI)
    config.cmems_source_mode = st.sidebar.radio(
        "Source Mode",
        ["local", "api"],
        format_func=lambda x: "📁 LOCAL Files" if x == "local" else "🌐 CMEMS API",
        horizontal=True,
        key="sidebar_cmems_source_mode",
        help="LOCAL=NetCDF files on disk, API=Copernicus Marine Service (requires login)"
    )
    
    if config.cmems_source_mode == "api":
        st.sidebar.info(
            "🌐 API mode downloads directly from Copernicus Marine. "
            "Set CMEMS_USERNAME and CMEMS_PASSWORD environment variables."
        )
    
    # Only show path input for local mode
    if config.cmems_source_mode == "local":
        config.cmems_base_dir = st.sidebar.text_input(
            "CMEMS Data Directory",
            value="/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA",
            key="sidebar_cmems_dir",
            help="Folder containing J1_netcdf/, J2_netcdf/, J3_netcdf/ subfolders"
        )
        
        # Show available files info using CMEMSService
        from pathlib import Path
        try:
            from src.services.cmems_service import CMEMSService, CMEMSConfig
            temp_config = CMEMSConfig(base_dir=config.cmems_base_dir)
            temp_service = CMEMSService(temp_config)
            file_counts = temp_service.count_files()
            date_info = temp_service.get_date_range()
            
            if file_counts["total"] > 0:
                st.sidebar.caption(f"📁 {file_counts['total']:,} NetCDF files found")
                # Show per-satellite breakdown
                sat_info = " | ".join([f"{k}: {v}" for k, v in file_counts.items() if k != "total"])
                st.sidebar.caption(f"   {sat_info}")
                if date_info["years"]:
                    st.sidebar.caption(f"📅 Years: {date_info['years'][0]} - {date_info['years'][-1]}")
            else:
                st.sidebar.warning("⚠️ 0 NetCDF files found")
        except Exception as e:
            cmems_path = Path(config.cmems_base_dir)
            if not cmems_path.exists():
                st.sidebar.warning("⚠️ Directory not found")
            else:
                st.sidebar.warning(f"⚠️ Error checking files: {e}")
    else:
        # API mode: use default path (won't be used anyway)
        config.cmems_base_dir = "/tmp/cmems_api_cache"
    
    # Show pass number from gate filename (if available)
    gate_path = _get_gate_shapefile(config.selected_gate)
    if gate_path:
        try:
            from src.services.cmems_service import _extract_pass_from_gate_name
            strait_name, pass_number = _extract_pass_from_gate_name(gate_path)
            
            if pass_number is not None:
                st.sidebar.success(f"🎯 **Pass {pass_number}** found in gate filename")
                st.sidebar.caption(f"Gate: {strait_name}")
            else:
                st.sidebar.info("ℹ️ No pass number in gate filename (synthetic pass)")
                st.sidebar.caption(f"Gate: {strait_name}")
        except Exception as e:
            pass  # Silently continue if extraction fails
    
    return config


def _render_cmems_date_range(config: AppConfig) -> AppConfig:
    """
    CMEMS date range - REMOVED.
    We load ALL data from the folder, no date filtering.
    """
    st.sidebar.info("📅 Loading ALL data from folder (no date filter)")
    return config


def _render_cmems_params(config: AppConfig) -> AppConfig:
    """Render CMEMS-specific processing parameters."""
    
    # Track selection (equivalent to SLCCI pass)
    st.sidebar.markdown("### 🛤️ Track Selection")
    
    # Get gate path for track discovery
    gate_path = _get_gate_shapefile(config.selected_gate)
    
    # Try to extract suggested track from gate name
    suggested_track = None
    if config.selected_gate:
        suggested_track = _extract_pass_from_gate_name(config.selected_gate)
    
    # Show suggested track if found
    if suggested_track:
        st.sidebar.success(f"🎯 **Suggested Track: {suggested_track}**")
        st.sidebar.caption("Extracted from gate shapefile name")
    
    # Track selection mode
    track_mode = st.sidebar.radio(
        "Track Mode",
        ["all", "suggested", "closest", "manual"] if suggested_track else ["all", "closest", "manual"],
        format_func=lambda x: {
            "all": "📊 All Tracks",
            "suggested": f"🎯 Suggested ({suggested_track})" if suggested_track else "Suggested",
            "closest": "🔍 5 Closest",
            "manual": "✏️ Manual"
        }.get(x, x),
        horizontal=True,
        key="sidebar_cmems_track_mode",
        help="all=merge all tracks, suggested=from gate name, closest=5 nearest to gate, manual=enter number"
    )
    
    if track_mode == "all":
        config.cmems_track_number = None
        st.sidebar.info("Using ALL tracks (merged)")
        
    elif track_mode == "suggested" and suggested_track:
        config.cmems_track_number = suggested_track
        st.sidebar.info(f"Using track **{suggested_track}** from gate definition")
        
    elif track_mode == "closest" and gate_path:
        # Find and show 5 closest tracks
        try:
            from src.services.cmems_service import CMEMSService, CMEMSConfig
            temp_config = CMEMSConfig(
                base_dir=config.cmems_base_dir,
                source_mode=config.cmems_source_mode
            )
            temp_service = CMEMSService(temp_config)
            
            # Cache closest tracks in session state
            cache_key = f"cmems_closest_{config.selected_gate}"
            if cache_key not in st.session_state:
                with st.spinner("Finding closest tracks..."):
                    st.session_state[cache_key] = temp_service.find_closest_tracks(gate_path, n_tracks=5)
            
            closest_tracks = st.session_state[cache_key]
            
            if closest_tracks:
                # Build options: list of (track_num, distance_km)
                track_options = [t[0] for t in closest_tracks]
                track_labels = [f"Track {t[0]} ({t[1]:.1f} km)" for t in closest_tracks]
                
                selected_idx = st.sidebar.selectbox(
                    "Select Track",
                    range(len(track_options)),
                    format_func=lambda i: track_labels[i],
                    key="sidebar_cmems_closest_track",
                    help="Tracks sorted by distance to gate centroid"
                )
                
                config.cmems_track_number = track_options[selected_idx]
                st.sidebar.success(f"🎯 Track {config.cmems_track_number} selected")
            else:
                st.sidebar.warning("No tracks found near this gate")
                config.cmems_track_number = None
        except Exception as e:
            st.sidebar.error(f"Error finding tracks: {e}")
            config.cmems_track_number = None
            
    elif track_mode == "manual":
        config.cmems_track_number = st.sidebar.number_input(
            "Track Number",
            min_value=1,
            max_value=500,
            value=suggested_track if suggested_track else 100,
            key="sidebar_cmems_manual_track",
            help="Enter a specific track number"
        )
        st.sidebar.info(f"Using manual track **{config.cmems_track_number}**")
    
    # Performance options (collapsible)
    with st.expander("⚡ Performance", expanded=False):
        # Parallel processing toggle
        config.cmems_use_parallel = st.checkbox(
            "Use Parallel Loading",
            value=True,
            key="sidebar_cmems_parallel",
            help="Load files in parallel (faster for large datasets)"
        )
        
        # Cache toggle
        config.cmems_use_cache = st.checkbox(
            "Use Cache",
            value=True,
            key="sidebar_cmems_cache",
            help="Cache processed data (instant reload on second run)"
        )
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", key="sidebar_clear_cmems_cache"):
            try:
                from src.services.cmems_service import CACHE_DIR
                import shutil
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    st.success("Cache cleared!")
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
    
    # Longitude binning size - SLIDER da 0.05 a 0.50 (lower res than SLCCI)
    config.cmems_lon_bin_size = st.slider(
        "Lon Bin Size (°)",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        format="%.2f",
        key="sidebar_cmems_lon_bin",
        help="Binning resolution for CMEMS (0.05° - 0.50°, coarser than SLCCI)"
    )
    
    # Buffer around gate - default 5.0° (from Copernicus notebook)
    config.cmems_buffer_deg = st.slider(
        "Gate Buffer (°)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,  # Changed from 0.5 to 5.0 as per Copernicus notebook
        step=0.5,
        format="%.1f",
        key="sidebar_cmems_buffer",
        help="Buffer around gate for data extraction (default 5.0° from notebook)"
    )
    
    return config


def _render_latitude_warning(config: AppConfig) -> AppConfig:
    """
    Show warning if gate is above 66°N (Jason satellite coverage limit).
    Applies to both SLCCI and CMEMS.
    """
    if not GATE_SERVICE_AVAILABLE or not config.selected_gate:
        return config
    
    gate = _gate_service.get_gate(config.selected_gate)
    if not gate:
        return config
    
    # Try to get gate latitude from shapefile
    gate_path = _get_gate_shapefile(config.selected_gate)
    if gate_path:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gate_path)
            max_lat = gdf.geometry.bounds['maxy'].max()
            
            if max_lat > 66.0:
                st.sidebar.warning(f"""
                ⚠️ **Latitude Warning**
                
                Gate extends to {max_lat:.2f}°N.
                
                Jason satellites (J1/J2/J3) coverage is limited to ±66°.
                Data beyond 66°N may be sparse or unavailable.
                """)
        except Exception:
            pass
    
    return config


def _load_slcci_data(config: AppConfig):
    """Load SLCCI data using SLCCIService (local or API)."""
    
    # Validate geoid path (always needed)
    if not Path(config.slcci_geoid_path).exists():
        st.sidebar.error(f"❌ Geoid not found: {config.slcci_geoid_path}")
        return
    
    # Validate local paths if using local source
    source_mode = getattr(config, 'data_source_mode', 'local')
    if source_mode == "local" and not Path(config.slcci_base_dir).exists():
        st.sidebar.error(f"❌ Path not found: {config.slcci_base_dir}")
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
            lon_bin_size=getattr(config, 'lon_bin_size', 0.01),  # From sidebar slider
            source=source_mode,  # "local" or "api"
            satellite="J2",
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
            
            # Store in session state using dedicated function
            store_slcci_data(pass_data)
            st.session_state["slcci_service"] = service
            st.session_state["slcci_config"] = config
            st.session_state["datasets"] = {}  # Clear generic
            
            # Success message
            n_obs = len(pass_data.df) if hasattr(pass_data, 'df') else 0
            n_cyc = pass_data.df['cycle'].nunique() if hasattr(pass_data, 'df') and 'cycle' in pass_data.df.columns else 0
            
            st.sidebar.success(f"""
            ✅ SLCCI Data Loaded!
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
    """Load data from ERA5 or other APIs."""
    
    if not config.selected_gate:
        st.sidebar.warning("Select a gate first")
        return
    
    st.sidebar.info(f"Loading {config.selected_dataset_type}... (not implemented yet)")


def _load_cmems_data(config: AppConfig):
    """Load CMEMS L3 1Hz data using CMEMSService."""
    from pathlib import Path
    
    # For API mode, no path validation needed
    if getattr(config, 'cmems_source_mode', 'local') == 'local':
        # Validate CMEMS path
        cmems_path = Path(config.cmems_base_dir)
        if not cmems_path.exists():
            st.sidebar.error(f"❌ Path not found: {config.cmems_base_dir}")
            return
    else:
        cmems_path = Path(config.cmems_base_dir)
    
    # Get gate shapefile
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ Select a gate first")
        return
    
    try:
        from src.services.cmems_service import CMEMSService, CMEMSConfig
        
        cmems_config = CMEMSConfig(
            base_dir=str(cmems_path),
            source_mode=getattr(config, 'cmems_source_mode', 'local'),
            lon_bin_size=getattr(config, 'cmems_lon_bin_size', 0.1),
            buffer_deg=getattr(config, 'cmems_buffer_deg', 5.0),
            # Track filtering (like SLCCI pass)
            track_number=getattr(config, 'cmems_track_number', None),
            # Performance options
            use_parallel=getattr(config, 'cmems_use_parallel', True),
            use_cache=getattr(config, 'cmems_use_cache', True),
        )
        
        service = CMEMSService(cmems_config)
        
        # Show info based on source mode
        if cmems_config.source_mode == "api":
            st.sidebar.info("🌐 Connecting to CMEMS API...")
        else:
            # Show file count
            file_counts = service.count_files()
            total_files = file_counts['total']
            
            # Show performance info
            perf_info = []
            if cmems_config.use_cache:
                perf_info.append("📦 Cache ON")
            if cmems_config.use_parallel:
                perf_info.append("🚀 Parallel ON")
            if cmems_config.track_number:
                perf_info.append(f"🛤️ Track {cmems_config.track_number}")
            perf_str = " | ".join(perf_info) if perf_info else ""
            
            st.sidebar.info(f"📁 Found {total_files:,} files to process... {perf_str}")
        
        # Create progress bar
        progress_bar = st.sidebar.progress(0, text="Preparing...")
        status_text = st.sidebar.empty()
        
        def update_progress(processed: int, total: int):
            """Callback to update progress bar."""
            pct = processed / total if total > 0 else 0
            progress_bar.progress(pct, text=f"Processing: {processed:,}/{total:,} files ({pct*100:.0f}%)")
        
        # Check gate coverage
        coverage_info = service.check_gate_coverage(gate_path)
        if coverage_info.get("warning"):
            st.sidebar.warning(f"⚠️ {coverage_info['warning']}")
        
        status_text.text("Loading CMEMS data... (this may take several minutes)")
        
        # Load pass data with progress callback
        pass_data = service.load_pass_data(
            gate_path=gate_path, 
            progress_callback=update_progress
        )
        
        # Clear progress bar
        progress_bar.empty()
        status_text.empty()
        
        if pass_data is None:
            st.sidebar.error("❌ No data found for this gate")
            return
        
        # Store in session state using dedicated function
        store_cmems_data(pass_data)
        st.session_state["cmems_service"] = service
        st.session_state["cmems_config"] = config
        st.session_state["datasets"] = {}  # Clear generic
        
        # Success message
        n_obs = len(pass_data.df) if hasattr(pass_data, 'df') else 0
        n_months = pass_data.df['year_month'].nunique() if hasattr(pass_data, 'df') and 'year_month' in pass_data.df.columns else 0
        time_range = ""
        if hasattr(pass_data, 'df') and 'time' in pass_data.df.columns:
            time_range = f"\n- Period: {pass_data.df['time'].min().strftime('%Y-%m')} → {pass_data.df['time'].max().strftime('%Y-%m')}"
        
        # Show pass number properly
        pass_num = pass_data.pass_number
        pass_display = f"Pass {pass_num}" if pass_num else "Synthetic pass"
        
        st.sidebar.success(f"""
        ✅ CMEMS Data Loaded!
        - Gate: {pass_data.strait_name}
        - {pass_display}
        - Observations: {n_obs:,}
        - Monthly periods: {n_months}{time_range}
        - Satellites: J1+J2+J3 merged
        """)
        
        st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"❌ CMEMSService not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CMEMS: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


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


# ==============================================================================
# DTUSpace-SPECIFIC FUNCTIONS (ISOLATED - does not affect SLCCI/CMEMS)
# ==============================================================================

def _render_dtu_paths(config: AppConfig) -> AppConfig:
    """Render DTUSpace NetCDF file path input."""
    
    st.sidebar.info("🟢 **DTUSpace v4** is a gridded DOT product (not along-track)")
    
    config.dtu_nc_path = st.sidebar.text_input(
        "NetCDF File Path",
        value="/Users/nicolocaron/Desktop/ARCFRESH/arctic_ocean_prod_DTUSpace_v4.0.nc/arctic_ocean_prod_DTUSpace_v4.0.nc",
        key="sidebar_dtu_nc_path",
        help="Full path to DTUSpace NetCDF file (arctic_ocean_prod_DTUSpace_v4.0.nc)"
    )
    
    # Validate path
    if config.dtu_nc_path:
        nc_path = Path(config.dtu_nc_path)
        if nc_path.exists():
            st.sidebar.success(f"✅ File found: {nc_path.name}")
        else:
            st.sidebar.warning("⚠️ File not found at specified path")
    
    # Info about DTUSpace
    with st.sidebar.expander("ℹ️ About DTUSpace", expanded=False):
        st.markdown("""
        **DTUSpace v4** is a gridded Dynamic Ocean Topography (DOT) product.
        
        **Key differences from SLCCI/CMEMS:**
        - 📊 **Gridded** (lat × lon × time), not along-track
        - 🚫 **No satellite passes** - gate defines the "synthetic pass"
        - 📁 **Local only** - no API access
        - 🗓️ **Monthly** resolution
        
        **Variables:**
        - `dot`: Dynamic Ocean Topography (m)
        - `lat`, `lon`, `date`: coordinates
        """)
    
    return config


def _render_dtu_time_range(config: AppConfig) -> AppConfig:
    """Render DTUSpace time range selector."""
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        config.dtu_start_year = st.number_input(
            "Start Year",
            min_value=1993,
            max_value=2020,
            value=2006,
            key="sidebar_dtu_start_year"
        )
    
    with col2:
        config.dtu_end_year = st.number_input(
            "End Year",
            min_value=1993,
            max_value=2020,
            value=2017,
            key="sidebar_dtu_end_year"
        )
    
    # Show time range
    years = config.dtu_end_year - config.dtu_start_year + 1
    st.sidebar.caption(f"📅 Period: {config.dtu_start_year}–{config.dtu_end_year} ({years} years, ~{years*12} monthly steps)")
    
    # Processing options
    with st.sidebar.expander("⚙️ Processing Options", expanded=False):
        config.dtu_n_gate_pts = st.slider(
            "Gate interpolation points",
            min_value=100,
            max_value=800,
            value=400,
            step=50,
            key="sidebar_dtu_n_gate_pts",
            help="Number of points to interpolate along the gate line"
        )
    
    return config


def _load_dtu_data(config: AppConfig):
    """Load DTUSpace data for the selected gate."""
    
    # Validate
    if not config.dtu_nc_path or not Path(config.dtu_nc_path).exists():
        st.sidebar.error("❌ DTUSpace NetCDF file not found. Check the path.")
        return
    
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("❌ No gate selected. Select a gate first.")
        return
    
    try:
        from src.services.dtu_service import DTUService
        
        service = DTUService()
        
        with st.sidebar.status("🟢 Loading DTUSpace data...", expanded=True) as status:
            st.write(f"📁 File: {Path(config.dtu_nc_path).name}")
            st.write(f"🚪 Gate: {config.selected_gate}")
            st.write(f"📅 Period: {config.dtu_start_year}–{config.dtu_end_year}")
            
            pass_data = service.load_gate_data(
                nc_path=config.dtu_nc_path,
                gate_path=gate_path,
                start_year=config.dtu_start_year,
                end_year=config.dtu_end_year,
                n_gate_pts=config.dtu_n_gate_pts
            )
            
            status.update(label="✅ DTUSpace loaded!", state="complete", expanded=False)
        
        if pass_data is None:
            st.sidebar.error("❌ No data found for this gate/period")
            return
        
        # Store in session state using dedicated DTU function
        store_dtu_data(pass_data)
        st.session_state["dtu_service"] = service
        st.session_state["dtu_config"] = config
        
        # Verify storage
        stored = get_dtu_data()
        if stored is None:
            st.sidebar.error("❌ Failed to store DTU data in session state!")
            return
        
        # Success message
        n_time = pass_data.n_time
        n_valid_slopes = sum(~np.isnan(pass_data.slope_series))
        gate_length = pass_data.x_km[-1] if len(pass_data.x_km) > 0 else 0
        
        st.sidebar.success(f"""
        ✅ DTUSpace Data Loaded!
        - Gate: {pass_data.strait_name}
        - Dataset: {pass_data.dataset_name}
        - Period: {config.dtu_start_year}–{config.dtu_end_year}
        - Time steps: {n_time} monthly
        - Valid slopes: {n_valid_slopes}/{n_time}
        - Gate length: {gate_length:.1f} km
        """)
        
        # Rerun to display DTU tabs (data verified above)
        st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"❌ DTUService not available: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading DTUSpace: {e}")
        import traceback
        with st.sidebar.expander("Traceback"):
            st.code(traceback.format_exc())


# Import numpy for DTU functions
import numpy as np

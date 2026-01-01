"""
Sidebar Component - Simplified
"""

import os
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from ..state import AppConfig


try:
    from src.services import GateService
    _gate_service = GateService()
    GATE_SERVICE_AVAILABLE = True
except ImportError:
    _gate_service = None
    GATE_SERVICE_AVAILABLE = False


def render_sidebar() -> AppConfig:
    """Render simplified sidebar and return configuration."""
    st.sidebar.title("NICO Settings")
    config = AppConfig()
    
    config = _render_region_filter(config)
    config = _render_gate_selector(config)
    config = _render_buffer_slider(config)
    
    st.sidebar.divider()
    config = _render_data_source(config)
    
    if config.selected_dataset_type == "SLCCI":
        config = _render_slcci_paths(config)
        config = _render_pass_selection(config)
        config = _render_cycle_range(config)
        st.sidebar.divider()
        if st.sidebar.button("Load SLCCI Data", type="primary", use_container_width=True):
            _load_slcci_data(config)
    else:
        st.sidebar.divider()
        if st.sidebar.button("Load Data", type="primary", use_container_width=True):
            _load_generic_data(config)
    
    return config


def _render_region_filter(config: AppConfig) -> AppConfig:
    if not GATE_SERVICE_AVAILABLE:
        return config
    regions = _gate_service.get_regions()
    selected_region = st.sidebar.selectbox("Region", ["All Regions"] + regions, key="region_filter")
    st.session_state["selected_region"] = selected_region
    return config


def _render_gate_selector(config: AppConfig) -> AppConfig:
    if not GATE_SERVICE_AVAILABLE:
        st.sidebar.warning("Gate service not available")
        return config
    
    selected_region = st.session_state.get("selected_region", "All Regions")
    if selected_region == "All Regions":
        gates = _gate_service.list_gates()
    else:
        gates = _gate_service.list_gates_by_region(selected_region)
    
    gate_options = ["None (Global)"]
    gate_ids = [None]
    for gate in gates:
        gate_options.append(gate.name)
        gate_ids.append(gate.id)
    
    current_gate = st.session_state.get("selected_gate")
    current_idx = gate_ids.index(current_gate) if current_gate in gate_ids else 0
    
    selected_idx = st.sidebar.selectbox(
        "Gate", range(len(gate_options)),
        format_func=lambda i: gate_options[i],
        index=current_idx, key="gate_selector"
    )
    
    config.selected_gate = gate_ids[selected_idx]
    st.session_state["selected_gate"] = config.selected_gate
    
    if config.selected_gate:
        gate = _gate_service.get_gate(config.selected_gate)
        if gate:
            st.sidebar.caption(f"{gate.region} - {gate.description}")
            try:
                config.gate_geometry = _gate_service.get_gate_geometry(config.selected_gate)
                st.session_state["gate_geometry"] = config.gate_geometry
            except:
                pass
    return config


def _render_buffer_slider(config: AppConfig) -> AppConfig:
    config.gate_buffer_km = st.sidebar.slider("Buffer (km)", 10, 200, 50, 10, key="buffer_slider")
    st.session_state["gate_buffer_km"] = config.gate_buffer_km
    return config


def _render_data_source(config: AppConfig) -> AppConfig:
    config.selected_dataset_type = st.sidebar.radio(
        "Data Source", ["SLCCI", "CMEMS", "ERA5"],
        key="dataset_type", horizontal=True
    )
    st.session_state["selected_dataset_type"] = config.selected_dataset_type
    return config


def _render_slcci_paths(config: AppConfig) -> AppConfig:
    config.slcci_base_dir = st.sidebar.text_input(
        "SLCCI Data Path",
        value=st.session_state.get("slcci_base_dir", config.slcci_base_dir),
        key="slcci_path"
    )
    st.session_state["slcci_base_dir"] = config.slcci_base_dir
    
    config.slcci_geoid_path = st.sidebar.text_input(
        "Geoid Path",
        value=st.session_state.get("slcci_geoid_path", config.slcci_geoid_path),
        key="geoid_path"
    )
    st.session_state["slcci_geoid_path"] = config.slcci_geoid_path
    return config


def _render_pass_selection(config: AppConfig) -> AppConfig:
    config.pass_mode = st.sidebar.radio(
        "Pass Mode", ["auto", "manual"],
        format_func=lambda x: "Auto-detect" if x == "auto" else "Manual",
        key="pass_mode", horizontal=True
    )
    if config.pass_mode == "manual":
        config.pass_number = st.sidebar.number_input(
            "Pass Number", 1, 500,
            st.session_state.get("pass_number", 248),
            key="pass_number"
        )
        st.session_state["pass_number"] = config.pass_number
    return config


def _render_cycle_range(config: AppConfig) -> AppConfig:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        config.cycle_start = st.number_input("Start", 1, 300, st.session_state.get("cycle_start", 1), key="cycle_start")
        st.session_state["cycle_start"] = config.cycle_start
    with col2:
        config.cycle_end = st.number_input("End", 1, 300, st.session_state.get("cycle_end", 10), key="cycle_end")  # Default 10 per test
        st.session_state["cycle_end"] = config.cycle_end
    return config


def _load_slcci_data(config: AppConfig):
    if not Path(config.slcci_base_dir).exists():
        st.sidebar.error(f"SLCCI path not found: {config.slcci_base_dir}")
        return
    if not Path(config.slcci_geoid_path).exists():
        st.sidebar.error(f"Geoid not found: {config.slcci_geoid_path}")
        return
    
    gate_path = _get_gate_shapefile(config.selected_gate)
    if not gate_path:
        st.sidebar.error("Gate shapefile required")
        return
    
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig
        
        cycles = list(range(config.cycle_start, config.cycle_end + 1))
        slcci_config = SLCCIConfig(
            base_dir=config.slcci_base_dir,
            geoid_path=config.slcci_geoid_path,
            cycles=cycles, use_flag=True,
        )
        service = SLCCIService(slcci_config)
        
        with st.spinner("Loading SLCCI data..."):
            pass_number = config.pass_number
            if config.pass_mode == "auto":
                st.sidebar.info("Finding closest passes...")
                closest = service.find_closest_pass(gate_path, n_passes=3)
                if closest:
                    pass_number = closest[0][0]
                else:
                    st.sidebar.error("No passes found")
                    return
            
            pass_data = service.load_pass_data(gate_path=gate_path, pass_number=pass_number, cycles=cycles)
            if pass_data is None:
                st.sidebar.error(f"No data for pass {pass_number}")
                return
            
            st.session_state["slcci_pass_data"] = pass_data
            st.session_state["slcci_service"] = service
            st.session_state["datasets"] = {}
            
            st.sidebar.success(f"Loaded {len(pass_data.df):,} points")
            st.rerun()
            
    except ImportError as e:
        st.sidebar.error(f"SLCCI service not available: {e}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


def _load_generic_data(config: AppConfig):
    if not config.selected_gate:
        st.sidebar.warning("Select a gate first")
        return
    
    try:
        from src.services import DataService, GateService
        from src.core.models import TimeRange
        from datetime import datetime, timedelta
        from ..state import update_datasets
        
        gs = GateService()
        ds = DataService()
        gate = gs.get_gate(config.selected_gate)
        if not gate or not gate.bbox:
            st.sidebar.error("Gate has no bounding box")
            return
        
        with st.spinner(f"Loading {config.selected_dataset_type}..."):
            request = ds.build_request(
                gate=gate, bbox=gate.bbox,
                time_range=TimeRange(start=datetime.now()-timedelta(days=100), end=datetime.now()),
                dataset_id=config.selected_dataset_type.lower()
            )
            data = ds.load(request)
            if data is not None:
                datasets = {config.selected_dataset_type: data}
                update_datasets(datasets, {"dataset": config.selected_dataset_type})
                st.session_state["slcci_pass_data"] = None
                st.sidebar.success(f"Loaded {config.selected_dataset_type}")
                st.rerun()
            else:
                st.sidebar.warning("No data - using demo")
                datasets, cycle_info = ds.load_multi_cycle_demo(request, n_cycles=10)
                if datasets:
                    update_datasets(datasets, cycle_info)
                    st.session_state["slcci_pass_data"] = None
                    st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


def _get_gate_shapefile(gate_id: str) -> Optional[str]:
    if not gate_id:
        return None
    gates_dir = Path(__file__).parent.parent.parent / "gates"
    if not gates_dir.exists():
        return None
    
    if GATE_SERVICE_AVAILABLE:
        gate = _gate_service.get_gate(gate_id)
        if gate and hasattr(gate, 'shapefile') and gate.shapefile:
            shp_path = gates_dir / gate.shapefile
            if shp_path.exists():
                return str(shp_path)
    
    import glob
    patterns = [f"*{gate_id}*.shp", f"*{gate_id.replace('_', '-')}*.shp"]
    for pattern in patterns:
        matches = list(gates_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return None

"""
Sidebar Component (Refactored)
==============================
Streamlit sidebar using unified services layer.

This is the new modular version that uses:
- GateService for gate selection
- DataService for data loading
- AnalysisService for computations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import logging

import streamlit as st

try:
    from src.services import GateService, DataService
    from src.core.models import BoundingBox, TimeRange, GateModel
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    GateService = None
    DataService = None

logger = logging.getLogger(__name__)


@dataclass
class SidebarConfig:
    """Configuration returned from sidebar."""
    # Gate selection
    selected_gate_id: Optional[str] = None
    selected_gate: Optional[Any] = None  # GateModel
    gate_buffer_km: float = 50.0
    
    # Bounding box (from gate or manual)
    bbox: Optional[Any] = None  # BoundingBox
    
    # Time range
    time_range: Optional[Any] = None  # TimeRange
    
    # Analysis parameters
    mss_var: str = "mean_sea_surface"
    bin_size: float = 0.01
    sample_fraction: float = 1.0
    
    # Satellite passes
    selected_passes: list = field(default_factory=list)


def render_gate_sidebar() -> SidebarConfig:
    """
    Render the gate selection sidebar.
    
    Uses GateService for all gate operations.
    
    Returns:
        SidebarConfig with user selections
    """
    config = SidebarConfig()
    
    if not SERVICES_AVAILABLE:
        st.sidebar.warning("⚠️ Services layer not available")
        return config
    
    st.sidebar.title("🗺️ Gate Selection")
    
    # Initialize service
    try:
        gate_service = GateService()
    except Exception as e:
        st.sidebar.error(f"Could not initialize gate service: {e}")
        return config
    
    # Get regions
    regions = gate_service.get_regions()
    
    # Region filter
    selected_region = st.sidebar.selectbox(
        "Region",
        ["All"] + regions,
        help="Filter gates by region"
    )
    
    # Get gates
    if selected_region == "All":
        gates = gate_service.list_gates()
    else:
        gates = gate_service.list_gates_by_region(selected_region)
    
    # Create options
    gate_options = {"🌍 None (Global)": None}
    for gate in gates:
        icon = _get_gate_icon(gate.id)
        gate_options[f"{icon} {gate.name}"] = gate.id
    
    # Gate selector
    selected_label = st.sidebar.selectbox(
        "Ocean Gate",
        list(gate_options.keys()),
        help="Select a strait or gate to analyze"
    )
    
    selected_gate_id = gate_options[selected_label]
    config.selected_gate_id = selected_gate_id
    
    # Show gate info
    if selected_gate_id:
        gate = gate_service.select_gate(selected_gate_id)
        config.selected_gate = gate
        
        if gate:
            _render_gate_card(gate)
            
            # Buffer slider
            config.gate_buffer_km = st.sidebar.slider(
                "Buffer (km)",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Area around the gate to include"
            )
            
            # Get bbox
            bbox = gate_service.get_bbox(selected_gate_id, buffer_km=config.gate_buffer_km)
            config.bbox = bbox
            
            # Show satellite passes
            passes = gate_service.get_closest_passes(selected_gate_id, n=5)
            if passes:
                st.sidebar.markdown("**🛰️ Closest Satellite Passes**")
                config.selected_passes = st.sidebar.multiselect(
                    "Select passes",
                    options=passes,
                    default=[passes[0]] if passes else [],
                    help="Satellite passes closest to this gate"
                )
    
    st.sidebar.divider()
    
    # Analysis parameters
    st.sidebar.subheader("⚙️ Analysis Settings")
    
    config.bin_size = st.sidebar.slider(
        "Bin Size (°)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Spatial binning resolution"
    )
    
    config.sample_fraction = st.sidebar.slider(
        "Sample Fraction",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Fraction of data to use (for faster testing)"
    )
    
    # Store in session state
    st.session_state["sidebar_config"] = config
    st.session_state["selected_gate_id"] = selected_gate_id
    
    return config


def _render_gate_card(gate: Any) -> None:
    """Render a styled card for gate information."""
    
    region = getattr(gate, 'region', 'Unknown')
    description = getattr(gate, 'description', '')
    importance = getattr(gate, 'importance', '')
    
    st.sidebar.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
        padding: 12px; 
        border-radius: 8px; 
        margin: 8px 0;
        border-left: 4px solid #4fc3f7;
    ">
        <div style="color: #4fc3f7; font-size: 0.8em; margin-bottom: 4px;">
            📍 {region}
        </div>
        <div style="color: white; font-size: 0.9em;">
            {description}
        </div>
        {f'<div style="color: #81d4fa; font-size: 0.75em; margin-top: 8px; font-style: italic;">{importance}</div>' if importance else ''}
    </div>
    """, unsafe_allow_html=True)


def _get_gate_icon(gate_id: str) -> str:
    """Get icon for gate based on ID."""
    icons = {
        "fram_strait": "🧊",
        "bering_strait": "🌊",
        "davis_strait": "❄️",
        "denmark_strait": "🌀",
        "nares_strait": "🏔️",
        "lancaster_sound": "🚢",
        "barents_opening": "🌡️",
        "norwegian_boundary": "🇳🇴",
    }
    return icons.get(gate_id, "🌊")


# Optional: Compatibility wrapper for existing code
def render_sidebar():
    """
    Compatibility wrapper for existing code.
    
    Calls new render_gate_sidebar and converts to old AppConfig format.
    """
    config = render_gate_sidebar()
    
    # Try to import old AppConfig for compatibility
    try:
        from app.components.sidebar import AppConfig
        
        old_config = AppConfig(
            mss_var=config.mss_var,
            bin_size=config.bin_size,
            sample_fraction=config.sample_fraction,
            selected_gate=config.selected_gate_id,
            gate_buffer_km=config.gate_buffer_km,
        )
        
        if config.bbox:
            old_config.lat_range = (config.bbox.lat_min, config.bbox.lat_max)
            old_config.lon_range = (config.bbox.lon_min, config.bbox.lon_max)
        
        return old_config
        
    except ImportError:
        return config

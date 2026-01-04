"""
3D Globe Component for Landing Page
====================================
Interactive 3D globe showing all available gates.
Click on a gate to select it for analysis.

Uses Plotly's scattergeo for Streamlit compatibility.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Optional, List, Dict, Any

# Try to import GateService
try:
    from src.services import GateService
    _gate_service = GateService()
    GATE_SERVICE_AVAILABLE = True
except ImportError:
    _gate_service = None
    GATE_SERVICE_AVAILABLE = False


def render_globe_landing(on_gate_select: Optional[callable] = None):
    """
    Render the 3D globe landing page with all gates.
    
    Args:
        on_gate_select: Callback when a gate is clicked (receives gate_name)
    """
    st.markdown("## 🌍 NICO - Arctic Ocean Analysis")
    st.markdown("*Navigate the Arctic: Select a gate to begin your analysis*")
    
    # Dataset preview selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        preview_dataset = st.selectbox(
            "Preview Dataset",
            ["All Gates", "SLCCI Coverage", "CMEMS Coverage", "DTUSpace Coverage"],
            key="globe_preview_dataset",
            help="Highlight gates by dataset availability"
        )
    
    with col2:
        show_labels = st.checkbox("Show Labels", value=True, key="globe_show_labels")
    
    with col3:
        projection = st.selectbox(
            "Projection",
            ["orthographic", "natural earth", "equirectangular"],
            key="globe_projection"
        )
    
    # Get all gates
    gates_data = _get_all_gates_positions()
    
    if not gates_data:
        # Use demo/fallback gates if service not available
        st.info("🔧 Using demo gates (GateService not configured)")
        gates_data = _get_demo_gates()
    
    if not gates_data:
        st.error("No gates available and no demo data. Check configuration.")
        return
    
    # Get currently selected gate from session state
    selected_gate = st.session_state.get("selected_gate", None)
    
    # Create the globe figure
    fig = _create_globe_figure(
        gates_data=gates_data,
        selected_gate=selected_gate,
        show_labels=show_labels,
        projection=projection
    )
    
    # Render with click events
    clicked = st.plotly_chart(
        fig, 
        use_container_width=True,
        key="globe_chart",
        on_select="rerun",  # Enable selection events
        selection_mode="points"
    )
    
    # Handle click events
    if clicked and clicked.selection and clicked.selection.points:
        point = clicked.selection.points[0]
        if "customdata" in point:
            clicked_gate = point["customdata"]
            if clicked_gate != selected_gate:
                st.session_state["selected_gate"] = clicked_gate
                st.session_state["sidebar_gate"] = clicked_gate
                if on_gate_select:
                    on_gate_select(clicked_gate)
                st.rerun()
    
    # Quick stats
    st.divider()
    _render_quick_stats(gates_data, selected_gate)
    
    # Selected gate info
    if selected_gate:
        st.divider()
        _render_selected_gate_info(selected_gate)


def _get_all_gates_positions() -> List[Dict[str, Any]]:
    """
    Get all gates with their centroid positions by scanning the gates/ folder.
    
    This function directly reads shapefiles from the gates/ directory,
    independent of GateService configuration.
    """
    import geopandas as gpd
    import os
    from pathlib import Path
    
    # Suppress shapefile warnings
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    # Find gates folder
    gates_folder = Path(__file__).parent.parent.parent / "gates"
    
    if not gates_folder.exists():
        st.warning(f"Gates folder not found: {gates_folder}")
        return []
    
    # Find all shapefiles
    shp_files = list(gates_folder.glob("*.shp"))
    
    if not shp_files:
        st.warning(f"No shapefiles found in {gates_folder}")
        return []
    
    gates = []
    
    for shp_path in shp_files:
        try:
            # Read shapefile
            gdf = gpd.read_file(shp_path)
            
            # Convert to WGS84 if needed
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:3413")  # Assume polar stereographic
            
            if not gdf.crs.is_geographic:
                gdf = gdf.to_crs("EPSG:4326")
            
            # Get centroid
            centroid = gdf.geometry.unary_union.centroid
            lon, lat = centroid.x, centroid.y
            
            # Extract name from filename
            gate_name = shp_path.stem
            
            # Determine region from name
            region = _infer_region_from_name(gate_name)
            
            gates.append({
                "name": gate_name,
                "lon": lon,
                "lat": lat,
                "region": region,
                "path": str(shp_path)
            })
            
        except Exception as e:
            # Skip problematic files
            continue
    
    return gates


def _infer_region_from_name(gate_name: str) -> str:
    """Infer the region from gate name."""
    name_lower = gate_name.lower()
    
    if "fram" in name_lower:
        return "Nordic Seas"
    elif "bering" in name_lower:
        return "Pacific-Arctic"
    elif "davis" in name_lower:
        return "Labrador Sea"
    elif "denmark" in name_lower:
        return "Nordic Seas"
    elif "barents" in name_lower:
        return "Barents Sea"
    elif "kara" in name_lower:
        return "Kara Sea"
    elif "laptev" in name_lower:
        return "Laptev Sea"
    elif "east_siberian" in name_lower or "siberian" in name_lower:
        return "East Siberian Sea"
    elif "beaufort" in name_lower:
        return "Beaufort Sea"
    elif "canadian" in name_lower or "nares" in name_lower or "lancaster" in name_lower or "jones" in name_lower:
        return "Canadian Arctic"
    elif "norwegian" in name_lower:
        return "Norwegian Sea"
    elif "central_arctic" in name_lower:
        return "Central Arctic"
    else:
        return "Arctic Ocean"


def _get_demo_gates() -> List[Dict[str, Any]]:
    """
    Return demo gates for display when shapefiles not available.
    These are the main Arctic straits used in NICO analysis.
    """
    return [
        {"name": "fram_strait", "lon": 0.0, "lat": 79.0, "region": "Nordic Seas", "path": None},
        {"name": "bering_strait", "lon": -168.5, "lat": 65.8, "region": "Pacific-Arctic", "path": None},
        {"name": "davis_strait", "lon": -57.0, "lat": 66.5, "region": "Labrador Sea", "path": None},
        {"name": "denmark_strait", "lon": -27.0, "lat": 66.0, "region": "Nordic Seas", "path": None},
        {"name": "barents_sea_opening", "lon": 20.0, "lat": 74.0, "region": "Barents Sea", "path": None},
        {"name": "nares_strait", "lon": -70.0, "lat": 80.5, "region": "Canadian Arctic", "path": None},
        {"name": "lancaster_sound", "lon": -85.0, "lat": 74.0, "region": "Canadian Arctic", "path": None},
        {"name": "hudson_strait", "lon": -70.0, "lat": 62.0, "region": "Hudson Bay", "path": None},
    ]


def _create_globe_figure(
    gates_data: List[Dict],
    selected_gate: Optional[str] = None,
    show_labels: bool = True,
    projection: str = "orthographic"
) -> go.Figure:
    """Create the 3D globe figure with gates."""
    
    # Separate selected gate from others
    other_gates = [g for g in gates_data if g["name"] != selected_gate]
    selected_gates = [g for g in gates_data if g["name"] == selected_gate]
    
    fig = go.Figure()
    
    # Add non-selected gates (blue markers)
    if other_gates:
        fig.add_trace(go.Scattergeo(
            lon=[g["lon"] for g in other_gates],
            lat=[g["lat"] for g in other_gates],
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=12,
                color="steelblue",
                symbol="circle",
                line=dict(width=1, color="white")
            ),
            text=[g["name"].replace("_", " ").title() for g in other_gates] if show_labels else None,
            textposition="top center",
            textfont=dict(size=10, color="white"),
            customdata=[g["name"] for g in other_gates],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
            name="Available Gates"
        ))
    
    # Add selected gate (highlighted - orange/gold)
    if selected_gates:
        sg = selected_gates[0]
        fig.add_trace(go.Scattergeo(
            lon=[sg["lon"]],
            lat=[sg["lat"]],
            mode="markers+text",
            marker=dict(
                size=18,
                color="darkorange",
                symbol="star",
                line=dict(width=2, color="gold")
            ),
            text=[sg["name"].replace("_", " ").title()],
            textposition="top center",
            textfont=dict(size=12, color="gold", family="Arial Black"),
            customdata=[sg["name"]],
            hovertemplate="<b>%{text}</b> ⭐ SELECTED<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>",
            name="Selected Gate"
        ))
    
    # Configure the globe
    fig.update_geos(
        projection_type=projection,
        showland=True,
        landcolor="rgb(40, 40, 40)",
        showocean=True,
        oceancolor="rgb(20, 50, 80)",
        showcoastlines=True,
        coastlinecolor="rgb(100, 100, 100)",
        showlakes=True,
        lakecolor="rgb(20, 50, 80)",
        showcountries=True,
        countrycolor="rgb(80, 80, 80)",
        
        # Focus on Arctic
        projection_rotation=dict(lon=0, lat=70, roll=0),
        
        # Lat/lon grid lines
        lataxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.2)",
            dtick=10,
            range=[50, 90]
        ),
        lonaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.2)",
            dtick=30,
            range=[-180, 180]
        ),
    )
    
    # Layout
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        ),
        title=dict(
            text="🌊 Arctic Ocean Gates",
            font=dict(size=16, color="white"),
            x=0.5
        )
    )
    
    return fig


def _render_quick_stats(gates_data: List[Dict], selected_gate: Optional[str]):
    """Render quick statistics about available gates."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gates", len(gates_data))
    
    with col2:
        # Count by region
        regions = set(g.get("region", "Unknown") for g in gates_data)
        st.metric("Regions", len(regions))
    
    with col3:
        status = "✅ Selected" if selected_gate else "❌ None"
        st.metric("Current Gate", status)
    
    with col4:
        if selected_gate:
            st.metric("Gate Name", selected_gate.replace("_", " ").title()[:20])
        else:
            st.metric("Gate Name", "Click to select")


def _render_selected_gate_info(gate_name: str):
    """Render detailed info about the selected gate."""
    
    if not GATE_SERVICE_AVAILABLE:
        return
    
    gate_info = _gate_service.get_gate(gate_name)
    if not gate_info:
        return
    
    st.markdown(f"### 🎯 Selected: **{gate_name.replace('_', ' ').title()}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Gate Info:**")
        st.markdown(f"- Region: `{gate_info.get('region', 'Unknown')}`")
        
        # Try to get coordinates
        try:
            gate_path = gate_info.get("path") or _gate_service.get_gate_path(gate_name)
            if gate_path:
                import geopandas as gpd
                import os
                os.environ['SHAPE_RESTORE_SHX'] = 'YES'
                gdf = gpd.read_file(gate_path)
                if gdf.crs and not gdf.crs.is_geographic:
                    gdf = gdf.to_crs("EPSG:4326")
                bounds = gdf.total_bounds
                st.markdown(f"- Lon: `{bounds[0]:.2f}°` to `{bounds[2]:.2f}°`")
                st.markdown(f"- Lat: `{bounds[1]:.2f}°` to `{bounds[3]:.2f}°`")
        except:
            pass
    
    with col2:
        st.markdown("**Next Steps:**")
        st.markdown("1. Select a **Data Source** in the sidebar")
        st.markdown("2. Configure **parameters**")
        st.markdown("3. Click **Load Data**")
        
        # Quick action buttons
        if st.button("📊 Load SLCCI", key="globe_load_slcci"):
            st.session_state["selected_dataset_type"] = "SLCCI"
            st.session_state["sidebar_datasource"] = "SLCCI"
            st.rerun()
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🌊 Load CMEMS", key="globe_load_cmems"):
                st.session_state["selected_dataset_type"] = "CMEMS"
                st.session_state["sidebar_datasource"] = "CMEMS"
                st.rerun()
        with col_b:
            if st.button("🟢 Load DTU", key="globe_load_dtu"):
                st.session_state["selected_dataset_type"] = "DTUSpace"
                st.session_state["sidebar_datasource"] = "DTUSpace"
                st.rerun()

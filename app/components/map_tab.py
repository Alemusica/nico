"""
🗺️ Map Tab - pydeck visualization per multi-sensor data

Usa pydeck (nativo Streamlit) invece di Kepler.gl per semplicità.
Supporta:
- Multi-layer visualization
- Time animation (via slider)
- Dataset overlay
- Causal chain arrows
"""

import streamlit as st
import pydeck as pdk
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# ==============================================================================
# LAYER BUILDERS
# ==============================================================================

def build_dataset_coverage_layer(
    datasets: List[Dict[str, Any]],
    opacity: float = 0.3
) -> List[pdk.Layer]:
    """
    Build GeoJsonLayer for each dataset's spatial coverage.
    """
    layers = []
    colors = {
        "live": [0, 200, 100, 150],      # 🟢 Green
        "daily": [100, 150, 255, 150],   # 🔵 Blue
        "monthly": [255, 200, 100, 150], # 🟡 Yellow
        "historical": [150, 150, 150, 100],  # ⚪ Gray
    }
    
    for ds in datasets:
        bbox = ds.get("bbox", {})
        if not bbox:
            continue
            
        west = bbox.get("west", -180)
        east = bbox.get("east", 180)
        south = bbox.get("south", -90)
        north = bbox.get("north", 90)
        
        latency = ds.get("latency_class", "daily")
        color = colors.get(latency, [100, 100, 100, 100])
        
        # Create GeoJSON polygon
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south]
                    ]]
                },
                "properties": {
                    "name": ds.get("id", "unknown"),
                    "provider": ds.get("provider", ""),
                    "latency": latency
                }
            }]
        }
        
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson,
            opacity=opacity,
            stroked=True,
            filled=True,
            get_fill_color=color,
            get_line_color=[255, 255, 255, 200],
            get_line_width=2,
            pickable=True
        )
        layers.append(layer)
    
    return layers


def build_causal_arrow_layer(
    causal_chains: List[Dict[str, Any]],
    dataset_locations: Dict[str, tuple]
) -> Optional[pdk.Layer]:
    """
    Build ArcLayer showing causal relationships between datasets.
    
    dataset_locations: {dataset_id: (lon, lat)}
    """
    if not causal_chains or not dataset_locations:
        return None
    
    arcs = []
    for chain in causal_chains:
        source_ds = chain.get("source", "")
        target_ds = chain.get("target", "")
        
        if source_ds in dataset_locations and target_ds in dataset_locations:
            source_loc = dataset_locations[source_ds]
            target_loc = dataset_locations[target_ds]
            
            physics_score = chain.get("physics_score", 0.5)
            
            arcs.append({
                "source_position": source_loc,
                "target_position": target_loc,
                "source_color": [255, 100, 100],
                "target_color": [100, 255, 100],
                "width": max(1, int(physics_score * 5))
            })
    
    if not arcs:
        return None
        
    return pdk.Layer(
        "ArcLayer",
        data=arcs,
        get_source_position="source_position",
        get_target_position="target_position",
        get_source_color="source_color",
        get_target_color="target_color",
        get_width="width",
        pickable=True
    )


def build_scatter_layer(
    points: List[Dict[str, Any]],
    radius_scale: float = 1.0
) -> pdk.Layer:
    """
    Build ScatterplotLayer for point data (e.g., events, stations).
    """
    df = pd.DataFrame(points)
    
    if df.empty:
        return None
    
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        radius_scale=radius_scale,
        radius_min_pixels=5,
        radius_max_pixels=50,
    )


# ==============================================================================
# MAP COMPONENT
# ==============================================================================

def render_map_view(
    center: tuple = (0, 45),
    zoom: float = 4,
    datasets: List[Dict[str, Any]] = None,
    causal_chains: List[Dict[str, Any]] = None,
    event_bbox: Dict[str, float] = None,
    points: List[Dict[str, Any]] = None,
):
    """
    Render interactive map with pydeck.
    
    Args:
        center: (lon, lat) map center
        zoom: Initial zoom level
        datasets: List of datasets with bbox metadata
        causal_chains: Causal relationships to show as arcs
        event_bbox: Optional event bounding box to highlight
        points: Optional point data to scatter
    """
    
    layers = []
    
    # 1. Dataset coverage layers
    if datasets:
        coverage_layers = build_dataset_coverage_layer(datasets)
        layers.extend(coverage_layers)
    
    # 2. Causal chain arcs
    if causal_chains:
        # Simple dataset->centroid mapping
        dataset_locations = {
            "era5_reanalysis": (10, 50),
            "cmems_sealevel": (-20, 60),
            "cmems_sst": (-30, 45),
            "slcci_altimetry": (0, 70),
            "noaa_climate_indices": (-40, 35),
            "cygnss_wind": (50, 20),
            "hydrognss": (15, 55),
            "flood": (8.5, 46),  # Lago Maggiore
            "hydrology": (10, 48),
        }
        arc_layer = build_causal_arrow_layer(causal_chains, dataset_locations)
        if arc_layer:
            layers.append(arc_layer)
    
    # 3. Event bbox highlight
    if event_bbox:
        event_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [event_bbox["west"], event_bbox["south"]],
                        [event_bbox["east"], event_bbox["south"]],
                        [event_bbox["east"], event_bbox["north"]],
                        [event_bbox["west"], event_bbox["north"]],
                        [event_bbox["west"], event_bbox["south"]]
                    ]]
                },
                "properties": {"type": "event_area"}
            }]
        }
        
        event_layer = pdk.Layer(
            "GeoJsonLayer",
            data=event_geojson,
            opacity=0.5,
            stroked=True,
            filled=True,
            get_fill_color=[255, 0, 0, 80],
            get_line_color=[255, 0, 0, 255],
            get_line_width=3,
        )
        layers.append(event_layer)
    
    # 4. Point scatter
    if points:
        scatter = build_scatter_layer(points)
        if scatter:
            layers.append(scatter)
    
    # Build deck
    view_state = pdk.ViewState(
        latitude=center[1],
        longitude=center[0],
        zoom=zoom,
        pitch=0,
        bearing=0
    )
    
    # Use a free map style that doesn't require Mapbox token
    # Options: "light", "dark", "road", "satellite", "streets"
    # Or use Carto: "carto-positron", "carto-darkmatter"
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "text": "{name}\n{provider}\nLatency: {latency}"
        }
    )
    
    return deck


# ==============================================================================
# STREAMLIT TAB
# ==============================================================================

def render_map_tab():
    """
    Render the Map tab in Streamlit.
    """
    st.header("🗺️ Multi-Sensor Map")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Map Controls")
        
        show_coverage = st.checkbox("Show Dataset Coverage", value=True)
        show_causal = st.checkbox("Show Causal Chains", value=False)
        show_event = st.checkbox("Highlight Event Area", value=True)
        
        st.divider()
        
        # Event selector
        event_options = {
            "Lago Maggiore 2000": {
                "center": (8.5, 46),
                "zoom": 6,
                "bbox": {"west": 8.0, "south": 45.0, "east": 10.0, "north": 47.0}
            },
            "North Atlantic": {
                "center": (-30, 55),
                "zoom": 3,
                "bbox": {"west": -60, "south": 40, "east": 0, "north": 70}
            },
            "Arctic Ocean": {
                "center": (0, 80),
                "zoom": 2,
                "bbox": {"west": -180, "south": 60, "east": 180, "north": 90}
            },
            "Global": {
                "center": (0, 30),
                "zoom": 1,
                "bbox": None
            }
        }
        
        selected_event = st.selectbox(
            "Region",
            options=list(event_options.keys()),
            index=0
        )
        
        event_config = event_options[selected_event]
    
    # Load data
    datasets = []
    causal_chains = []
    
    if show_coverage:
        try:
            from src.data_manager.intake_bridge import IntakeCatalogBridge
            bridge = IntakeCatalogBridge()
            
            for ds_id in bridge.list_datasets():
                try:
                    meta = bridge.get_metadata(ds_id)
                    datasets.append({
                        "id": ds_id,
                        "provider": meta.get("provider", "Unknown"),
                        "latency_class": meta.get("latency_class", "daily"),
                        "bbox": meta.get("bbox", {"west": -180, "east": 180, "south": -90, "north": 90})
                    })
                except:
                    pass
        except ImportError:
            st.warning("IntakeCatalogBridge not available")
    
    if show_causal:
        try:
            from src.data_manager.causal_graph import CausalGraphDB
            db = CausalGraphDB()
            causal_chains = db.get_all_edges()
        except Exception as e:
            st.warning(f"Causal chains not available: {e}")
    
    # Render map
    deck = render_map_view(
        center=event_config["center"],
        zoom=event_config["zoom"],
        datasets=datasets if show_coverage else None,
        causal_chains=causal_chains if show_causal else None,
        event_bbox=event_config["bbox"] if show_event else None,
    )
    
    st.pydeck_chart(deck)
    
    # Legend
    with st.expander("Legend"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Dataset Latency:**
            - 🟢 Live (< 1h)
            - 🔵 Daily (~1d)
            - 🟡 Monthly (~1mo)
            - ⚪ Historical
            """)
        
        with col2:
            st.markdown("""
            **Map Elements:**
            - 🔴 Event area (red box)
            - ➡️ Causal arcs (red→green)
            - 📍 Data points (scatter)
            """)
    
    # Stats
    if datasets:
        st.caption(f"Showing {len(datasets)} datasets")
    if causal_chains:
        st.caption(f"Showing {len(causal_chains)} causal relationships")


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Map Test", layout="wide")
    render_map_tab()

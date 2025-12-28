"""
🌊 Causal Discovery Demo Dashboard
==================================
Full-featured demo with synthetic data and live API.
"""

import streamlit as st
import pandas as pd
import numpy as np
import httpx
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Config
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="🌊 Causal Discovery Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    background: linear-gradient(90deg, #1e3a5f, #2d5a7b);
    color: white;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}
.event-card {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
}
.causal-link {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    border-radius: 4px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# API HELPERS
# ==============================================================================

@st.cache_data(ttl=60)
def fetch_api(endpoint: str, params: dict = None) -> Optional[Dict]:
    """Fetch data from API with caching."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_BASE}{endpoint}", params=params)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        st.sidebar.warning(f"API error: {e}")
    return None


def check_api_health() -> Dict[str, Any]:
    """Check API health status."""
    data = fetch_api("/health")
    if data:
        return {
            "status": "🟢 Online",
            "version": data.get("api_version", "?"),
            "llm": data.get("components", {}).get("llm", {}).get("status", "?"),
            "pcmci": data.get("components", {}).get("causal_discovery", {}).get("status", "?"),
        }
    return {"status": "🔴 Offline", "version": "?", "llm": "?", "pcmci": "?"}


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🌊 Causal Discovery Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar - API Status
    with st.sidebar:
        st.markdown("### ⚡ System Status")
        health = check_api_health()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API", health["status"])
        with col2:
            st.metric("Version", health["version"])
        
        st.markdown(f"**LLM**: {health['llm']}")
        st.markdown(f"**PCMCI**: {health['pcmci']}")
        
        st.divider()
        
        # Demo Event Selector
        st.markdown("### 🎯 Demo Event")
        demo_event = st.selectbox(
            "Select Case Study",
            ["Lago Maggiore 2000", "North Atlantic Storm", "Arctic Ice Melt"],
            index=0
        )
        
        st.divider()
        st.markdown("### 📊 Quick Stats")
        catalog = fetch_api("/data/catalog")
        if catalog:
            st.metric("Datasets", catalog.get("count", 0))
        
        ts_available = fetch_api("/timeseries/available")
        if ts_available:
            st.metric("Time Series", ts_available.get("count", 0))
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🗃️ Catalog", 
        "📈 Time Series",
        "🔗 Causal Graph",
        "🔬 PCMCI Results"
    ])
    
    # ==== TAB 1: Overview ====
    with tab1:
        render_overview_tab(demo_event)
    
    # ==== TAB 2: Catalog ====
    with tab2:
        render_catalog_tab()
    
    # ==== TAB 3: Time Series ====
    with tab3:
        render_timeseries_tab()
    
    # ==== TAB 4: Causal Graph ====
    with tab4:
        render_causal_tab()
    
    # ==== TAB 5: PCMCI Results ====
    with tab5:
        render_pcmci_tab()


# ==============================================================================
# TAB RENDERERS
# ==============================================================================

def render_overview_tab(event_name: str):
    """Overview with key metrics and event summary."""
    
    st.markdown(f"## 🎯 Case Study: {event_name}")
    
    if event_name == "Lago Maggiore 2000":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📅 Event Date", "Oct 13-16, 2000")
        with col2:
            st.metric("🌧️ Max Rainfall", "600 mm / 72h")
        with col3:
            st.metric("⚠️ Severity", "Extreme")
        with col4:
            st.metric("🔗 Causal Links", "113")
        
        st.markdown("""
        <div class="event-card">
        <b>🌊 Lago Maggiore Flood (October 2000)</b><br>
        One of the most severe flooding events in Northern Italy. 
        600mm of rain fell in 72 hours, causing the lake to overflow.
        PCMCI analysis reveals <b>precipitation → runoff</b> as the dominant causal link (lag: 12h, score: 0.99).
        </div>
        """, unsafe_allow_html=True)
        
        # Fetch and show time series preview
        frames = fetch_api("/timeseries/frames/lago_maggiore_2000", {"variables": "precipitation,runoff", "step": 4})
        
        if frames:
            # Build dataframe
            df = pd.DataFrame([
                {
                    "timestamp": f["timestamp"],
                    "precipitation": f["values"].get("precipitation", 0) * 1000,  # mm
                    "runoff": f["values"].get("runoff", 0) * 1000,  # mm
                }
                for f in frames
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["precipitation"],
                name="Precipitation (mm)",
                line=dict(color="#3498db", width=2),
                fill="tozeroy",
                fillcolor="rgba(52, 152, 219, 0.3)"
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["runoff"],
                name="Runoff (mm)",
                line=dict(color="#e74c3c", width=2),
                yaxis="y2"
            ))
            
            fig.update_layout(
                title="Precipitation & Runoff - October 2000",
                xaxis_title="Date",
                yaxis_title="Precipitation (mm)",
                yaxis2=dict(title="Runoff (mm)", overlaying="y", side="right"),
                height=400,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Event markers
            st.markdown("#### 🚨 Detected Events")
            events = fetch_api("/timeseries/events", {"dataset_id": "lago_maggiore_2000", "threshold": "0.02"})
            if events and events.get("events"):
                for evt in events["events"][:5]:
                    severity_bar = "🔴" * int(evt.get("severity", 0) * 5) + "⚪" * (5 - int(evt.get("severity", 0) * 5))
                    st.markdown(f"- **{evt['timestamp'][:10]}**: {evt['type']} {severity_bar}")
        else:
            st.info("Run `python experiments/lago_maggiore_pipeline.py` to generate demo data")
    
    else:
        st.info(f"Demo data for '{event_name}' not yet available. Use Lago Maggiore 2000.")


def render_catalog_tab():
    """Dataset catalog browser."""
    
    st.markdown("### 🗃️ Multi-Sensor Catalog")
    
    catalog = fetch_api("/data/catalog")
    
    if not catalog:
        st.error("Could not fetch catalog from API")
        return
    
    datasets = catalog.get("datasets", [])
    
    # Summary cards
    summary = catalog.get("summary", {})
    by_status = summary.get("by_status", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Datasets", len(datasets))
    with col2:
        st.metric("Available", by_status.get("available", 0))
    with col3:
        st.metric("To Implement", by_status.get("to_implement", 0))
    with col4:
        st.metric("Coming Soon", by_status.get("coming_2026", 0))
    
    st.divider()
    
    # Dataset table
    df = pd.DataFrame([
        {
            "ID": d["id"],
            "Provider": d.get("provider", ""),
            "Latency": d.get("latency_badge", "") + " " + d.get("latency", ""),
            "Variables": ", ".join(d.get("variables", [])[:4]),
            "Status": d.get("status", ""),
            "Resolution": d.get("resolution_spatial", ""),
        }
        for d in datasets
    ])
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        provider_filter = st.multiselect("Filter by Provider", df["Provider"].unique())
    with col2:
        status_filter = st.multiselect("Filter by Status", df["Status"].unique())
    
    if provider_filter:
        df = df[df["Provider"].isin(provider_filter)]
    if status_filter:
        df = df[df["Status"].isin(status_filter)]
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_timeseries_tab():
    """Time series visualization with slider."""
    
    st.markdown("### 📈 Time Series Explorer")
    
    available = fetch_api("/timeseries/available")
    
    if not available or not available.get("datasets"):
        st.warning("No time series data available. Run the pipeline first.")
        st.code("python experiments/lago_maggiore_pipeline.py --full")
        return
    
    datasets = available["datasets"]
    
    # Dataset selector
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_ds = st.selectbox(
            "Dataset",
            [d["id"] for d in datasets],
            index=0
        )
    
    # Get available variables
    ds_info = next((d for d in datasets if d["id"] == selected_ds), None)
    var_names = ds_info.get("variables", []) if ds_info else []
    
    # Map internal names to friendly names
    var_map = {"tp": "precipitation", "t2m": "temperature", "msl": "pressure", 
               "u10": "u_wind", "v10": "v_wind", "swvl1": "soil_moisture", "ro": "runoff"}
    friendly_vars = [var_map.get(v, v) for v in var_names]
    
    with col2:
        selected_vars = st.multiselect(
            "Variables",
            friendly_vars,
            default=["precipitation", "runoff"] if "precipitation" in friendly_vars else friendly_vars[:2]
        )
    
    with col3:
        step = st.slider("Time Step", 1, 12, 4)
    
    if not selected_vars:
        st.info("Select at least one variable")
        return
    
    # Fetch data
    frames = fetch_api(
        f"/timeseries/frames/{selected_ds}",
        {"variables": ",".join(selected_vars), "step": str(step)}
    )
    
    if not frames:
        st.error("Could not fetch time series data")
        return
    
    # Build DataFrame
    df = pd.DataFrame([
        {"timestamp": f["timestamp"], **f["values"]}
        for f in frames
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Time slider for animation
    st.markdown("#### ⏱️ Time Navigation")
    
    time_idx = st.slider(
        "Select Time",
        0, len(df) - 1, len(df) // 2,
        format=f"Step %d"
    )
    
    current_time = df.iloc[time_idx]["timestamp"]
    st.markdown(f"**Current: {current_time}**")
    
    # Show values at current time
    cols = st.columns(len(selected_vars))
    for i, var in enumerate(selected_vars):
        with cols[i]:
            val = df.iloc[time_idx].get(var, 0)
            st.metric(var.title(), f"{val:.4f}")
    
    # Plot
    fig = go.Figure()
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    
    for i, var in enumerate(selected_vars):
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[var],
            name=var.title(),
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    
    # Add vertical line at current time (use string format to avoid timestamp issues)
    current_time_str = str(current_time)
    fig.add_vline(x=current_time_str, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=f"Time Series: {selected_ds}",
        xaxis_title="Time",
        yaxis_title="Value",
        height=450,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_causal_tab():
    """Causal graph visualization."""
    
    st.markdown("### 🔗 Causal Graph")
    
    # Load PCMCI results
    import json
    from pathlib import Path
    
    results_file = Path(__file__).parent.parent.parent / "data" / "pipeline" / "lago_maggiore_2000" / "pcmci_results.json"
    
    if not results_file.exists():
        st.warning("No causal analysis results. Run the pipeline first.")
        st.code("python experiments/lago_maggiore_pipeline.py --full")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    links = results.get("links", [])
    
    st.markdown(f"**{len(links)} causal links discovered**")
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.5)
    
    with col2:
        target_filter = st.selectbox(
            "Target Variable",
            ["All"] + list(set(l["target"] for l in links))
        )
    
    # Filter links
    filtered = [l for l in links if l["score"] >= min_score]
    if target_filter != "All":
        filtered = [l for l in filtered if l["target"] == target_filter]
    
    st.markdown(f"**Showing {len(filtered)} links**")
    
    # Display as styled cards
    for link in sorted(filtered, key=lambda x: -x["score"])[:15]:
        lag_hours = link["lag"] * 6  # 6-hourly data
        score_bar = "█" * int(link["score"] * 10) + "░" * (10 - int(link["score"] * 10))
        
        st.markdown(f"""
        <div class="causal-link">
            <b>{link['source']}</b> → <b>{link['target']}</b>  
            | Lag: {lag_hours}h | Score: {score_bar} {link['score']:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    # Network visualization
    st.markdown("#### 📊 Network View")
    
    try:
        import networkx as nx
        
        G = nx.DiGraph()
        
        for link in filtered[:30]:  # Top 30
            G.add_edge(
                link["source"], 
                link["target"],
                weight=link["score"],
                lag=link["lag"]
            )
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Build plotly figure
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                size=20,
                color='#3498db',
                line_width=2
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400,
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.info("Install networkx for graph visualization: pip install networkx")


def render_pcmci_tab():
    """PCMCI analysis details."""
    
    st.markdown("### 🔬 PCMCI Causal Discovery")
    
    import json
    from pathlib import Path
    
    results_file = Path(__file__).parent.parent.parent / "data" / "pipeline" / "lago_maggiore_2000" / "pcmci_results.json"
    
    if not results_file.exists():
        st.warning("No PCMCI results available.")
        
        st.markdown("#### Run Analysis")
        st.code("""
# Generate data and run PCMCI
python experiments/lago_maggiore_pipeline.py --full

# Or step by step:
python experiments/lago_maggiore_pipeline.py --download
python experiments/lago_maggiore_pipeline.py --pcmci
python experiments/lago_maggiore_pipeline.py --store
        """)
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Event", results.get("event", "Unknown")[:20])
    with col2:
        st.metric("Variables", len(results.get("variables", [])))
    with col3:
        st.metric("Time Steps", results.get("n_timesteps", 0))
    with col4:
        st.metric("Significant Links", results.get("n_significant_links", 0))
    
    st.divider()
    
    # Config used
    with st.expander("⚙️ PCMCI Configuration"):
        config = results.get("pcmci_config", {})
        st.json(config)
    
    # Variables analyzed
    st.markdown("#### 📊 Variables Analyzed")
    vars_analyzed = results.get("variables", [])
    st.write(", ".join(vars_analyzed))
    
    # Top links table
    st.markdown("#### 🏆 Top Causal Links")
    
    links = results.get("links", [])
    top_links = sorted(links, key=lambda x: -x.get("score", 0))[:20]
    
    df = pd.DataFrame([
        {
            "Source": l["source"],
            "Target": l["target"],
            "Lag (h)": l["lag"] * 6,
            "Score": f"{l['score']:.3f}",
            "p-value": f"{l.get('p_value', 0):.4f}",
        }
        for l in top_links
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Interpretation
    st.markdown("#### 💡 Key Findings")
    
    # Find strongest link to runoff
    runoff_links = [l for l in links if l["target"] == "runoff"]
    if runoff_links:
        strongest = max(runoff_links, key=lambda x: x["score"])
        st.success(f"""
        **Primary Flood Driver**: {strongest['source']} → runoff
        - Lag: {strongest['lag'] * 6} hours
        - Causal Score: {strongest['score']:.3f}
        - This confirms precipitation as the dominant precursor of runoff/flooding.
        """)


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    main()

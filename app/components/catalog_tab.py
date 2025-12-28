"""
Dataset Catalog Tab
===================
Browse multi-provider datasets with latency badges.
Uses the Intake catalog via API or direct bridge.
"""

import streamlit as st
import pandas as pd
import httpx
from typing import Optional, Dict, List, Any

# API base URL
API_BASE = "http://localhost:8000/api/v1"


def get_catalog_data() -> Optional[Dict[str, Any]]:
    """Fetch catalog data from API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/data/catalog")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.warning(f"API not available: {e}. Using direct bridge.")
    
    # Fallback to direct import
    try:
        from src.data_manager.intake_bridge import get_catalog
        cat = get_catalog()
        datasets = []
        for ds_id in cat.list_datasets():
            meta = cat.get_metadata(ds_id)
            datasets.append({
                "id": ds_id,
                "description": meta.get("description", ds_id),
                "provider": meta.get("provider"),
                "variables": meta.get("variables", []),
                "latency": meta.get("latency"),
                "latency_badge": meta.get("latency_badge"),
                "status": meta.get("status"),
                "resolution_spatial": meta.get("resolution_spatial"),
                "resolution_temporal": meta.get("resolution_temporal"),
            })
        return {"datasets": datasets, "count": len(datasets)}
    except ImportError:
        return None


def search_catalog(
    provider: Optional[str] = None,
    variables: Optional[List[str]] = None,
    latency: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Search catalog with filters."""
    try:
        params = {}
        if provider:
            params["provider"] = provider
        if variables:
            params["variables"] = ",".join(variables)
        if latency:
            params["latency"] = latency
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE}/data/catalog/search", params=params)
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    
    # Fallback to direct search
    try:
        from src.data_manager.intake_bridge import get_catalog
        cat = get_catalog()
        results = cat.search(
            variables=variables,
            latency_badge=latency,
            provider=provider,
        )
        return {"matches": results, "count": len(results)}
    except ImportError:
        return None


def render_catalog_tab():
    """Render the Dataset Catalog tab."""
    
    st.markdown("### 🗃️ Multi-Provider Dataset Catalog")
    st.markdown("""
    Browse available datasets from multiple providers. 
    **Latency badges** indicate data freshness for operational use.
    """)
    
    # Filters in columns
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        provider_filter = st.selectbox(
            "Provider",
            ["All", "Copernicus Marine", "ECMWF", "NOAA", "NASA", "ESA CCI", "EUMETSAT"],
            index=0,
        )
    
    with col2:
        latency_filter = st.selectbox(
            "Latency",
            ["All", "🟢 Real-time (<24h)", "🟡 Near real-time (1-7d)", "🔴 Delayed (>7d)", "⚫ Unknown"],
            index=0,
        )
    
    with col3:
        variable_input = st.text_input(
            "Variables (comma-sep)",
            placeholder="e.g., sla, sst, wind_speed",
        )
    
    with col4:
        st.write("")  # Spacer
        search_btn = st.button("🔍 Search", use_container_width=False)
    
    # Parse filters
    provider = None if provider_filter == "All" else provider_filter
    latency = None
    if latency_filter.startswith("🟢"):
        latency = "🟢"
    elif latency_filter.startswith("🟡"):
        latency = "🟡"
    elif latency_filter.startswith("🔴"):
        latency = "🔴"
    elif latency_filter.startswith("⚫"):
        latency = "⚫"
    
    variables = [v.strip() for v in variable_input.split(",") if v.strip()] or None
    
    # Fetch data
    if search_btn or provider or latency or variables:
        data = search_catalog(provider=provider, variables=variables, latency=latency)
        key = "matches"
    else:
        data = get_catalog_data()
        key = "datasets"
    
    if not data:
        st.error("❌ Could not load catalog. Check API or intake_bridge installation.")
        return
    
    datasets = data.get(key, [])
    
    if not datasets:
        st.info("No datasets match your filters.")
        return
    
    # Convert to DataFrame for display
    df = pd.DataFrame(datasets)
    
    # Format variables as comma-separated
    if "variables" in df.columns:
        df["variables"] = df["variables"].apply(
            lambda x: ", ".join(x[:4]) + ("..." if len(x) > 4 else "") if isinstance(x, list) else str(x)
        )
    
    # Display columns
    display_cols = ["id", "provider", "latency_badge", "variables", "status", "resolution_temporal"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Rename for display
    col_names = {
        "id": "Dataset",
        "provider": "Provider",
        "latency_badge": "Latency",
        "variables": "Variables",
        "status": "Status",
        "resolution_temporal": "Resolution",
    }
    
    df_display = df[display_cols].rename(columns=col_names)
    
    # Show count
    st.markdown(f"**Found {len(datasets)} datasets**")
    
    # Interactive table
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Latency": st.column_config.TextColumn(
                "Latency",
                help="🟢 <24h, 🟡 1-7d, 🔴 >7d, ⚫ unknown",
                width="small",
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                width="small",
            ),
        },
    )
    
    # Dataset selection for details
    st.markdown("---")
    st.markdown("### 📋 Dataset Details")
    
    selected_id = st.selectbox(
        "Select dataset for details",
        options=[d["id"] for d in datasets],
        index=0,
    )
    
    if selected_id:
        selected = next((d for d in datasets if d["id"] == selected_id), None)
        if selected:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Provider:** {selected.get('provider', 'N/A')}")
                st.markdown(f"**Latency:** {selected.get('latency', 'N/A')} {selected.get('latency_badge', '')}")
                st.markdown(f"**Status:** {selected.get('status', 'N/A')}")
                st.markdown(f"**Spatial Resolution:** {selected.get('resolution_spatial', 'N/A')}")
                st.markdown(f"**Temporal Resolution:** {selected.get('resolution_temporal', 'N/A')}")
            
            with col2:
                st.markdown("**Variables:**")
                vars_list = selected.get("variables", [])
                if isinstance(vars_list, list):
                    for v in vars_list:
                        st.markdown(f"- `{v}`")
                else:
                    st.markdown(f"- `{vars_list}`")
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Create Briefing", key=f"briefing_{selected_id}"):
                    st.session_state["selected_dataset"] = selected_id
                    st.info(f"TODO: Open briefing dialog for {selected_id}")
            
            with col2:
                if st.button("🔗 Show Causal Chains", key=f"causal_{selected_id}"):
                    _show_causal_chains(selected_id)
            
            with col3:
                if st.button("📊 Quick Preview", key=f"preview_{selected_id}"):
                    st.info(f"TODO: Load preview for {selected_id}")


def _show_causal_chains(dataset_id: str):
    """Show causal chains for a dataset."""
    try:
        from src.data_manager.causal_graph import CausalGraphDB
        
        db = CausalGraphDB()
        db.connect()
        
        # Get precursors (what causes this)
        precursors = db.get_precursors(dataset_id)
        
        # Get effects (what this causes)
        effects = db.get_effects(dataset_id)
        
        db.close()
        
        if not precursors and not effects:
            st.info(f"No causal chains found for {dataset_id}")
            return
        
        st.markdown(f"#### Causal Chains for `{dataset_id}`")
        
        if precursors:
            st.markdown("**Precursors (what drives this):**")
            for p in precursors:
                lag = p.get("lag_days", "?")
                source = p.get("source", "?")
                source_var = p.get("source_var", "?")
                mechanism = p.get("physics_mechanism", "unknown")
                score = p.get("physics_score", 0)
                st.markdown(f"- `{source}.{source_var}` → (lag {lag}d, {mechanism}, score {score:.1%})")
        
        if effects:
            st.markdown("**Effects (what this influences):**")
            for e in effects:
                lag = e.get("lag_days", "?")
                target = e.get("target", "?")
                target_var = e.get("target_var", "?")
                mechanism = e.get("physics_mechanism", "unknown")
                st.markdown(f"- → `{target}.{target_var}` (lag {lag}d, {mechanism})")
    
    except Exception as e:
        st.warning(f"Could not load causal chains: {e}")
        st.info("Make sure SurrealDB is running: `docker start surrealdb`")

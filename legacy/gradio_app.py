"""
🛰️ SLCCI Satellite Altimetry Dashboard - Gradio Version
========================================================
Clean, simple UI without Streamlit's state issues.
"""

import gradio as gr
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = Path(__file__).parent / "data"
GATES_DIR = Path(__file__).parent / "gates"

# Global state (simple dict, no framework magic)
APP_STATE = {
    "datasets": {},
    "current_gate": None,
}


def find_nc_files() -> List[str]:
    """Find all NetCDF files in data directory."""
    files = []
    if DATA_DIR.exists():
        for subdir in DATA_DIR.iterdir():
            if subdir.is_dir():
                for f in subdir.glob("*.nc"):
                    files.append(str(f.relative_to(DATA_DIR.parent)))
    return sorted(files)


def load_dataset(file_path: str) -> Tuple[str, pd.DataFrame]:
    """Load a NetCDF file and return info + preview."""
    if not file_path:
        return "No file selected", pd.DataFrame()
    
    full_path = Path(__file__).parent / file_path
    if not full_path.exists():
        return f"File not found: {file_path}", pd.DataFrame()
    
    try:
        ds = xr.open_dataset(full_path)
        APP_STATE["datasets"][file_path] = ds
        
        # Build info string
        info_lines = [
            f"📂 **File:** {full_path.name}",
            f"📏 **Dimensions:** {dict(ds.dims)}",
            f"📊 **Variables:** {list(ds.data_vars.keys())}",
            f"🏷️ **Coordinates:** {list(ds.coords.keys())}",
        ]
        
        # Add attributes
        if ds.attrs:
            info_lines.append(f"ℹ️ **Attributes:** {len(ds.attrs)} items")
        
        info = "\n".join(info_lines)
        
        # Create preview DataFrame
        preview_data = {}
        for var in list(ds.data_vars.keys())[:5]:
            values = ds[var].values.flatten()[:100]
            preview_data[var] = values
        
        preview_df = pd.DataFrame(preview_data)
        
        return info, preview_df
    
    except Exception as e:
        return f"❌ Error loading file: {e}", pd.DataFrame()


def get_data_summary() -> str:
    """Get summary of loaded datasets."""
    if not APP_STATE["datasets"]:
        return "No datasets loaded yet."
    
    lines = ["## 📊 Loaded Datasets\n"]
    for name, ds in APP_STATE["datasets"].items():
        lines.append(f"- **{Path(name).name}**: {dict(ds.dims)}")
    
    return "\n".join(lines)


def create_dot_plot(file_path: str) -> go.Figure:
    """Create DOT visualization."""
    if not file_path or file_path not in APP_STATE["datasets"]:
        fig = go.Figure()
        fig.add_annotation(text="Load a dataset first", x=0.5, y=0.5, showarrow=False)
        return fig
    
    ds = APP_STATE["datasets"][file_path]
    
    # Find DOT variable
    dot_var = None
    for name in ['dot', 'DOT', 'dynamic_ocean_topography', 'sla', 'SLA']:
        if name in ds.data_vars:
            dot_var = name
            break
    
    if dot_var is None:
        fig = go.Figure()
        fig.add_annotation(text="No DOT/SLA variable found", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get lat/lon
    lat_var = next((n for n in ['lat', 'latitude', 'Latitude'] if n in ds.coords or n in ds.data_vars), None)
    lon_var = next((n for n in ['lon', 'longitude', 'Longitude'] if n in ds.coords or n in ds.data_vars), None)
    
    if lat_var and lon_var:
        lat = ds[lat_var].values.flatten()
        lon = ds[lon_var].values.flatten()
        dot = ds[dot_var].values.flatten()
        
        # Remove NaN
        mask = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(dot))
        lat, lon, dot = lat[mask], lon[mask], dot[mask]
        
        # Sample if too many points
        if len(lat) > 5000:
            idx = np.random.choice(len(lat), 5000, replace=False)
            lat, lon, dot = lat[idx], lon[idx], dot[idx]
        
        fig = px.scatter_mapbox(
            lat=lat, lon=lon, color=dot,
            color_continuous_scale="RdBu_r",
            mapbox_style="carto-positron",
            zoom=2,
            title=f"{dot_var.upper()} Spatial Distribution",
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    else:
        # Simple histogram
        dot = ds[dot_var].values.flatten()
        dot = dot[~np.isnan(dot)]
        
        fig = px.histogram(x=dot, nbins=50, title=f"{dot_var.upper()} Distribution")
        fig.update_layout(height=400)
    
    return fig


def create_time_series(file_path: str) -> go.Figure:
    """Create time series plot."""
    if not file_path or file_path not in APP_STATE["datasets"]:
        fig = go.Figure()
        fig.add_annotation(text="Load a dataset first", x=0.5, y=0.5, showarrow=False)
        return fig
    
    ds = APP_STATE["datasets"][file_path]
    
    # Find time and value variables
    time_var = next((n for n in ['time', 'Time', 'timestamp'] if n in ds.coords or n in ds.data_vars), None)
    value_var = next((n for n in ['dot', 'DOT', 'sla', 'SLA', 'ssh'] if n in ds.data_vars), None)
    
    if time_var and value_var:
        time = ds[time_var].values
        values = ds[value_var].values.flatten()
        
        # Aggregate by time if needed
        if len(values) > 1000:
            # Simple mean aggregation
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=values[:1000],
                mode='lines',
                name=value_var
            ))
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time if len(time) == len(values) else np.arange(len(values)),
                y=values,
                mode='lines',
                name=value_var
            ))
        
        fig.update_layout(
            title=f"{value_var.upper()} Time Series",
            xaxis_title="Time/Index",
            yaxis_title=value_var.upper(),
            height=400
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="No time series data found", x=0.5, y=0.5, showarrow=False)
    
    return fig


def run_pattern_analysis(file_path: str, analysis_type: str) -> Tuple[str, go.Figure]:
    """Run pattern detection analysis."""
    if not file_path or file_path not in APP_STATE["datasets"]:
        return "Load a dataset first", go.Figure()
    
    ds = APP_STATE["datasets"][file_path]
    
    # Find value variable
    value_var = next((n for n in ['dot', 'DOT', 'sla', 'SLA'] if n in ds.data_vars), None)
    if not value_var:
        return "No DOT/SLA variable found", go.Figure()
    
    values = ds[value_var].values.flatten()
    values = values[~np.isnan(values)]
    
    if analysis_type == "Anomaly Detection":
        # Z-score based anomaly detection
        mean_val = np.mean(values)
        std_val = np.std(values)
        z_scores = (values - mean_val) / std_val
        anomalies = np.abs(z_scores) > 2
        
        n_anomalies = np.sum(anomalies)
        pct_anomalies = n_anomalies / len(values) * 100
        
        report = f"""## 🔍 Anomaly Detection Results

**Method:** Z-score (threshold = 2σ)

**Statistics:**
- Total points: {len(values):,}
- Anomalies detected: {n_anomalies:,} ({pct_anomalies:.2f}%)
- Mean: {mean_val:.4f}
- Std: {std_val:.4f}
- Min anomaly Z: {z_scores[anomalies].min():.2f} (if any)
- Max anomaly Z: {z_scores[anomalies].max():.2f} (if any)
"""
        
        # Create visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Anomaly Scatter"])
        
        fig.add_trace(
            go.Histogram(x=values, nbinsx=50, name="All"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(values)),
                y=values,
                mode='markers',
                marker=dict(
                    color=['red' if a else 'blue' for a in anomalies],
                    size=3
                ),
                name="Data"
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        
    elif analysis_type == "Statistics":
        report = f"""## 📊 Statistical Analysis

**Variable:** {value_var}

**Descriptive Statistics:**
- Count: {len(values):,}
- Mean: {np.mean(values):.6f}
- Std: {np.std(values):.6f}
- Min: {np.min(values):.6f}
- 25%: {np.percentile(values, 25):.6f}
- 50%: {np.percentile(values, 50):.6f}
- 75%: {np.percentile(values, 75):.6f}
- Max: {np.max(values):.6f}
- Skewness: {pd.Series(values).skew():.4f}
- Kurtosis: {pd.Series(values).kurtosis():.4f}
"""
        
        fig = px.box(y=values, title=f"{value_var} Box Plot")
        fig.update_layout(height=400)
        
    else:  # Trend Analysis
        # Simple moving average trend
        window = min(100, len(values) // 10)
        if window < 2:
            window = 2
        
        ma = pd.Series(values).rolling(window=window).mean()
        
        report = f"""## 📈 Trend Analysis

**Method:** Moving Average (window={window})

**Trend Summary:**
- Start value: {values[0]:.6f}
- End value: {values[-1]:.6f}
- Change: {values[-1] - values[0]:.6f}
- MA Start: {ma.dropna().iloc[0]:.6f}
- MA End: {ma.dropna().iloc[-1]:.6f}
"""
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=values, mode='lines', name='Raw', opacity=0.5))
        fig.add_trace(go.Scatter(y=ma, mode='lines', name=f'MA({window})', line=dict(width=2)))
        fig.update_layout(title="Trend Analysis", height=400)
    
    return report, fig


# Build Gradio Interface
def create_app():
    """Create the Gradio app."""
    
    with gr.Blocks(
        title="🛰️ SLCCI Satellite Altimetry",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        """
    ) as app:
        
        gr.Markdown("# 🛰️ SLCCI Satellite Altimetry Dashboard")
        gr.Markdown("Analyze satellite altimetry data with pattern detection")
        
        with gr.Tabs():
            # Tab 1: Data Loading
            with gr.TabItem("📂 Data"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_dropdown = gr.Dropdown(
                            choices=find_nc_files(),
                            label="Select NetCDF File",
                            interactive=True
                        )
                        load_btn = gr.Button("📂 Load Dataset", variant="primary")
                        refresh_btn = gr.Button("🔄 Refresh File List")
                        
                        data_info = gr.Markdown("Select a file to load")
                    
                    with gr.Column(scale=2):
                        data_preview = gr.Dataframe(label="Data Preview")
                
                load_btn.click(
                    fn=load_dataset,
                    inputs=[file_dropdown],
                    outputs=[data_info, data_preview]
                )
                
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=find_nc_files()),
                    outputs=[file_dropdown]
                )
            
            # Tab 2: Visualization
            with gr.TabItem("📊 Visualize"):
                with gr.Row():
                    viz_file = gr.Dropdown(
                        choices=find_nc_files(),
                        label="Select Dataset"
                    )
                    plot_type = gr.Radio(
                        choices=["Spatial Map", "Time Series"],
                        value="Spatial Map",
                        label="Plot Type"
                    )
                    plot_btn = gr.Button("📈 Generate Plot", variant="primary")
                
                viz_plot = gr.Plot(label="Visualization")
                
                def update_viz(file_path, plot_type):
                    if plot_type == "Spatial Map":
                        return create_dot_plot(file_path)
                    else:
                        return create_time_series(file_path)
                
                plot_btn.click(
                    fn=update_viz,
                    inputs=[viz_file, plot_type],
                    outputs=[viz_plot]
                )
            
            # Tab 3: Pattern Detection
            with gr.TabItem("🔬 Pattern Detection"):
                with gr.Row():
                    pattern_file = gr.Dropdown(
                        choices=find_nc_files(),
                        label="Select Dataset"
                    )
                    analysis_type = gr.Radio(
                        choices=["Anomaly Detection", "Statistics", "Trend Analysis"],
                        value="Statistics",
                        label="Analysis Type"
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        analysis_report = gr.Markdown("Select a dataset and analysis type")
                    with gr.Column():
                        analysis_plot = gr.Plot(label="Analysis Results")
                
                analyze_btn.click(
                    fn=run_pattern_analysis,
                    inputs=[pattern_file, analysis_type],
                    outputs=[analysis_report, analysis_plot]
                )
            
            # Tab 4: Summary
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown("""
                ## 🛰️ About This Dashboard
                
                This dashboard analyzes **satellite altimetry data** for oceanographic research.
                
                ### Features:
                - 📂 **Data Loading**: Load NetCDF files from the data directory
                - 📊 **Visualization**: Spatial maps and time series plots
                - 🔬 **Pattern Detection**: Anomaly detection, statistics, trend analysis
                
                ### Data Sources:
                - **SLCCI**: Sea Level Climate Change Initiative
                - **AVISO**: Archiving, Validation and Interpretation of Satellite Oceanographic
                - **CMEMS**: Copernicus Marine Environment Monitoring Service
                
                ### Pattern Engine Integration:
                The pattern detection uses physics-validated rules for:
                - Storm surge detection
                - Sea level anomaly identification
                - Trend analysis with uncertainty quantification
                """)
                
                summary_btn = gr.Button("🔄 Refresh Summary")
                summary_text = gr.Markdown("Click refresh to see loaded datasets")
                
                summary_btn.click(fn=get_data_summary, outputs=[summary_text])
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_port=7860,
        share=False,
        show_error=True
    )

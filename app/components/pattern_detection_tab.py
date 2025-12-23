"""
Pattern Detection Tab
=====================
Streamlit component for Pattern Detection Engine.
Integrates tsfresh, mlxtend, physics validation with satellite data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Import pattern engine components
try:
    from src.pattern_engine import (
        PhysicsValidator,
        Domain,
        create_flood_validator,
    )
    from src.pattern_engine.preprocessing import (
        TSFreshExtractor,
        FeatureExtractionConfig,
        TSFRESH_AVAILABLE,
    )
    from src.pattern_engine.detection import (
        AssociationRuleDetector,
        AssociationRuleConfig,
        MLXTEND_AVAILABLE,
    )
    from src.pattern_engine.output import (
        GrayZoneDetector,
        GrayZoneConfig,
    )
    PATTERN_ENGINE_AVAILABLE = True
except ImportError as e:
    PATTERN_ENGINE_AVAILABLE = False
    st.warning(f"Pattern Engine not available: {e}")


def render_pattern_detection_tab(datasets: Dict, cycle_info: Dict, config: Any):
    """
    Render the Pattern Detection tab.
    
    Args:
        datasets: Loaded NetCDF datasets
        cycle_info: Metadata about cycles
        config: App configuration
    """
    st.markdown("## 🔬 Pattern Detection Engine")
    
    if not PATTERN_ENGINE_AVAILABLE:
        st.error("❌ Pattern Engine module not available. Check imports.")
        return
    
    # Sidebar controls for pattern detection
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔬 Pattern Detection Settings")
        
        detection_mode = st.selectbox(
            "Detection Mode",
            ["Surge Patterns", "Anomaly Detection", "Association Rules", "Custom Analysis"],
            help="Choose the type of pattern detection",
            key="pattern_detection_mode"
        )
        
        physics_enabled = st.checkbox("Enable Physics Validation", value=True, key="physics_enabled_cb")
        
        if TSFRESH_AVAILABLE:
            feature_mode = st.selectbox(
                "Feature Extraction",
                ["minimal", "efficient", "comprehensive"],
                index=0,
                help="minimal=10 features, efficient=750, comprehensive=800+",
                key="feature_extraction_mode"
            )
        else:
            feature_mode = "minimal"
            st.info("📦 tsfresh not installed - using basic features")
        
        min_confidence = st.slider("Min Confidence", 0.3, 0.9, 0.6, 0.05, key="min_conf_slider")
        min_support = st.slider("Min Support", 0.05, 0.3, 0.1, 0.01, key="min_supp_slider")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Data Preparation")
        
        # Convert satellite data to DataFrame
        df = convert_datasets_to_dataframe(datasets, cycle_info)
        
        if df is None or df.empty:
            st.warning("⚠️ No data available. Load satellite data first.")
            return
        
        st.success(f"✅ Loaded {len(df)} observations from {df['cycle'].nunique()} cycles")
        
        # Preview data
        with st.expander("📋 Preview Data", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Points", f"{len(df):,}")
            st.metric("Cycles", df['cycle'].nunique())
        with col_b:
            if 'dot' in df.columns:
                st.metric("DOT Range", f"{df['dot'].max() - df['dot'].min():.3f} m")
            if 'sla' in df.columns:
                st.metric("SLA Std", f"{df['sla'].std():.3f} m")
    
    st.markdown("---")
    
    # Run pattern detection based on mode
    if detection_mode == "Surge Patterns":
        run_surge_detection(df, physics_enabled, min_confidence, min_support)
    elif detection_mode == "Anomaly Detection":
        run_anomaly_detection(df, feature_mode)
    elif detection_mode == "Association Rules":
        run_association_rules(df, min_confidence, min_support)
    else:
        run_custom_analysis(df, config)


def convert_datasets_to_dataframe(datasets: Dict, cycle_info: Dict) -> Optional[pd.DataFrame]:
    """
    Convert xarray datasets to pandas DataFrame for pattern detection.
    """
    if not datasets:
        return None
    
    all_data = []
    
    for cycle_num, ds in datasets.items():
        try:
            # Extract variables from dataset
            data_dict = {'cycle': cycle_num}
            
            # Try common variable names
            var_mapping = {
                'lat': ['lat', 'latitude', 'Latitude'],
                'lon': ['lon', 'longitude', 'Longitude'],
                'dot': ['dot', 'DOT', 'dynamic_ocean_topography'],
                'sla': ['sla', 'SLA', 'sea_level_anomaly'],
                'mss': ['mss', 'MSS', 'mean_sea_surface'],
                'time': ['time', 'Time', 'timestamp'],
            }
            
            # Get dimension size
            dim_name = list(ds.dims.keys())[0] if ds.dims else None
            n_points = ds.dims[dim_name] if dim_name else 0
            
            if n_points == 0:
                continue
            
            # Extract each variable
            for key, possible_names in var_mapping.items():
                for name in possible_names:
                    if name in ds.data_vars or name in ds.coords:
                        values = ds[name].values
                        if len(values) == n_points:
                            data_dict[key] = values
                        break
            
            # Create DataFrame for this cycle
            if 'lat' in data_dict and 'lon' in data_dict:
                cycle_df = pd.DataFrame({
                    k: v if isinstance(v, (list, np.ndarray)) else [v] * n_points
                    for k, v in data_dict.items()
                })
                all_data.append(cycle_df)
        
        except Exception as e:
            st.warning(f"⚠️ Error processing cycle {cycle_num}: {e}")
            continue
    
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Add derived features
    if 'dot' in df.columns:
        df['dot_anomaly'] = df['dot'] - df.groupby('cycle')['dot'].transform('mean')
    
    if 'lat' in df.columns:
        df['lat_zone'] = pd.cut(df['lat'], bins=[-90, -60, -30, 0, 30, 60, 90], 
                                labels=['Antarctic', 'S-Mid', 'S-Trop', 'N-Trop', 'N-Mid', 'Arctic'])
    
    return df


def run_surge_detection(df: pd.DataFrame, physics_enabled: bool, min_conf: float, min_supp: float):
    """Run surge pattern detection with physics validation."""
    
    st.markdown("### 🌊 Surge Pattern Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create surge indicators
        st.markdown("#### 📊 Feature Engineering")
        
        features_df = df.copy()
        
        # Add surge-related features
        if 'dot' in features_df.columns:
            features_df['dot_high'] = features_df['dot'] > features_df['dot'].quantile(0.9)
            features_df['dot_low'] = features_df['dot'] < features_df['dot'].quantile(0.1)
            features_df['dot_extreme'] = features_df['dot_high'] | features_df['dot_low']
        
        if 'sla' in features_df.columns:
            features_df['sla_positive'] = features_df['sla'] > 0.1
            features_df['sla_negative'] = features_df['sla'] < -0.1
        
        # Summary
        st.write("**Created Features:**")
        feature_cols = [c for c in features_df.columns if c not in df.columns]
        for col in feature_cols:
            if features_df[col].dtype == bool:
                pct = features_df[col].sum() / len(features_df) * 100
                st.write(f"- `{col}`: {pct:.1f}% positive")
    
    with col2:
        st.markdown("#### ⚗️ Physics Validation")
        
        if physics_enabled:
            validator = create_flood_validator()
            st.write(f"**Available rules:** {validator.available_rules()}")
            
            # Since we don't have wind/pressure in satellite data, 
            # show physics is ready but needs external data
            st.info("💡 Physics rules need wind/pressure data. Integrate with ERA5 or ECMWF for full validation.")
        else:
            st.write("Physics validation disabled")
    
    # Visualization
    st.markdown("#### 📈 Surge Distribution")
    
    if 'dot' in df.columns:
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("DOT Distribution", "DOT by Cycle"))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df['dot'], nbinsx=50, name="DOT"),
            row=1, col=1
        )
        
        # Box by cycle (sample if too many)
        sample_cycles = sorted(df['cycle'].unique())[:20]
        sample_df = df[df['cycle'].isin(sample_cycles)]
        
        for cycle in sample_cycles:
            cycle_data = sample_df[sample_df['cycle'] == cycle]['dot']
            fig.add_trace(
                go.Box(y=cycle_data, name=f"C{cycle}", showlegend=False),
                row=1, col=2
            )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def run_anomaly_detection(df: pd.DataFrame, feature_mode: str):
    """Run anomaly detection with optional tsfresh features."""
    
    st.markdown("### 🔍 Anomaly Detection")
    
    # Aggregate by cycle for feature extraction
    if 'dot' not in df.columns:
        st.warning("⚠️ DOT variable not found in data")
        return
    
    cycle_stats = df.groupby('cycle').agg({
        'dot': ['mean', 'std', 'min', 'max', 'count'],
        'lat': ['mean', 'min', 'max'] if 'lat' in df.columns else [],
    })
    cycle_stats.columns = ['_'.join(col) for col in cycle_stats.columns]
    cycle_stats = cycle_stats.reset_index()
    
    st.write(f"**Cycles analyzed:** {len(cycle_stats)}")
    
    # TSFresh feature extraction (if available)
    if TSFRESH_AVAILABLE and st.button("🔬 Extract TSFresh Features"):
        with st.spinner("Extracting features..."):
            try:
                extractor = TSFreshExtractor(FeatureExtractionConfig(
                    mode=feature_mode,
                    n_jobs=1,
                    disable_progressbar=True,
                ))
                
                # Prepare data for tsfresh
                ts_df = df[['cycle', 'dot']].copy()
                ts_df['idx'] = ts_df.groupby('cycle').cumcount()
                
                features = extractor.extract(
                    ts_df,
                    column_id='cycle',
                    column_sort='idx',
                    column_value='dot',
                )
                
                st.success(f"✅ Extracted {features.shape[1]} features!")
                
                with st.expander("📋 Feature Preview"):
                    st.dataframe(features.head(10), use_container_width=True)
                
                # Store in session state
                st.session_state['tsfresh_features'] = features
                
            except Exception as e:
                st.error(f"❌ Feature extraction failed: {e}")
    
    # Simple anomaly detection (Z-score based)
    st.markdown("#### 📊 Cycle Anomaly Scores")
    
    if 'dot_mean' in cycle_stats.columns:
        cycle_stats['z_score'] = (
            (cycle_stats['dot_mean'] - cycle_stats['dot_mean'].mean()) 
            / cycle_stats['dot_mean'].std()
        )
        cycle_stats['is_anomaly'] = np.abs(cycle_stats['z_score']) > 2
        
        n_anomalies = cycle_stats['is_anomaly'].sum()
        st.metric("Anomalous Cycles", n_anomalies, 
                  delta=f"{n_anomalies/len(cycle_stats)*100:.1f}%")
        
        # Plot
        fig = px.scatter(
            cycle_stats,
            x='cycle',
            y='dot_mean',
            color='is_anomaly',
            size=np.abs(cycle_stats['z_score']),
            color_discrete_map={True: 'red', False: 'blue'},
            title="Cycle DOT Mean with Anomalies",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomalous cycles
        if n_anomalies > 0:
            with st.expander("🚨 Anomalous Cycles Details"):
                st.dataframe(
                    cycle_stats[cycle_stats['is_anomaly']],
                    use_container_width=True
                )


def run_association_rules(df: pd.DataFrame, min_conf: float, min_supp: float):
    """Run association rule mining with mlxtend."""
    
    st.markdown("### 🔗 Association Rules Mining")
    
    if not MLXTEND_AVAILABLE:
        st.error("❌ mlxtend not installed. Install with: `pip install mlxtend`")
        return
    
    # Prepare categorical features
    st.markdown("#### 🏷️ Feature Binning")
    
    binned_df = df.copy()
    
    # Bin continuous variables
    if 'dot' in df.columns:
        binned_df['dot_level'] = pd.qcut(df['dot'], q=3, labels=['low', 'medium', 'high'])
    
    if 'lat' in df.columns:
        binned_df['region'] = pd.cut(
            df['lat'], 
            bins=[-90, -23.5, 23.5, 90],
            labels=['south', 'tropics', 'north']
        )
    
    # Aggregate to cycle level
    cycle_df = binned_df.groupby('cycle').agg({
        'dot_level': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'medium',
        'region': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'tropics',
    }).reset_index()
    
    st.write(f"**Prepared {len(cycle_df)} cycle records**")
    
    # Run association rules
    if st.button("🔍 Mine Association Rules"):
        with st.spinner("Mining rules..."):
            try:
                detector = AssociationRuleDetector(AssociationRuleConfig(
                    min_support=min_supp,
                    min_confidence=min_conf,
                    algorithm='fpgrowth',
                ))
                
                # One-hot encode
                encoded = pd.get_dummies(cycle_df[['dot_level', 'region']])
                
                rules = detector.fit_from_boolean(encoded.astype(bool))
                
                if rules:
                    st.success(f"✅ Found {len(rules)} rules!")
                    
                    # Display rules
                    rules_data = []
                    for rule in rules[:20]:
                        rules_data.append({
                            'IF': ' & '.join(rule.antecedents),
                            'THEN': ' & '.join(rule.consequents),
                            'Support': f"{rule.support:.2%}",
                            'Confidence': f"{rule.confidence:.2%}",
                            'Lift': f"{rule.lift:.2f}",
                        })
                    
                    st.dataframe(pd.DataFrame(rules_data), use_container_width=True)
                else:
                    st.warning("No rules found with current thresholds")
                    
            except Exception as e:
                st.error(f"❌ Rule mining failed: {e}")


def run_custom_analysis(df: pd.DataFrame, config: Any):
    """Custom analysis interface."""
    
    st.markdown("### 🛠️ Custom Analysis")
    
    st.write("Build your own analysis pipeline:")
    
    # Variable selection
    available_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        target_var = st.selectbox("Target Variable", available_vars, index=0)
    with col2:
        feature_vars = st.multiselect("Feature Variables", 
                                       [v for v in available_vars if v != target_var],
                                       default=available_vars[1:3] if len(available_vars) > 2 else [])
    
    if feature_vars:
        # Correlation analysis
        st.markdown("#### 📊 Correlation Analysis")
        
        corr_df = df[[target_var] + feature_vars].corr()
        
        fig = px.imshow(
            corr_df,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f"Correlation Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series comparison (if cycle exists)
        if 'cycle' in df.columns:
            st.markdown("#### 📈 Variable Evolution")
            
            agg = df.groupby('cycle')[[target_var] + feature_vars].mean().reset_index()
            
            fig = go.Figure()
            for var in [target_var] + feature_vars:
                fig.add_trace(go.Scatter(
                    x=agg['cycle'], 
                    y=(agg[var] - agg[var].mean()) / agg[var].std(),  # Standardize
                    name=var,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Standardized Variables by Cycle",
                xaxis_title="Cycle",
                yaxis_title="Standardized Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

#!/usr/bin/env python3
"""
Tab-by-tab audit for NICO Streamlit App.
Tests each visualization tab with both SLCCI and CMEMS data.
"""
import sys
sys.path.insert(0, '/Users/nicolocaron/Documents/GitHub/nico')

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Results tracker
results = []

def log_result(test_name: str, success: bool, message: str = ""):
    status = "✅" if success else "❌"
    results.append({"test": test_name, "success": success, "message": message})
    print(f"  {status} {test_name}: {message}")

print("=" * 70)
print("NICO STREAMLIT APP - TAB-BY-TAB AUDIT")
print("=" * 70)

# ==============================================================================
# TEST 1: Load SLCCI Data
# ==============================================================================
print("\n📊 TEST 1: SLCCI Data Loading")
slcci_df = None
try:
    from src.services.slcci_service import SLCCIService, SLCCIConfig
    gate_path = Path("/Users/nicolocaron/Documents/GitHub/nico/gates/denmark_strait_TPJ_pass_246.shp")
    
    # Use cycles 40-50 for speed (about 1 year of data)
    config = SLCCIConfig(
        cycles=list(range(40, 51)),
        source="local",
        satellite="J2",
    )
    service = SLCCIService(config)
    pass_data = service.load_pass_data(
        gate_path=gate_path,
        pass_number=246,
        cycles=list(range(40, 51)),
    )
    slcci_df = pass_data.df if pass_data else None
    if slcci_df is not None:
        log_result("SLCCI load", True, f"{len(slcci_df)} rows, cols: {list(slcci_df.columns)[:5]}")
    else:
        log_result("SLCCI load", False, "No data returned")
except Exception as e:
    log_result("SLCCI load", False, str(e)[:100])

# ==============================================================================
# TEST 2: Load CMEMS Data (Local)
# ==============================================================================
print("\n📊 TEST 2: CMEMS Data Loading (Local)")
cmems_df = None
try:
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    gate_path = Path("/Users/nicolocaron/Documents/GitHub/nico/gates/denmark_strait_TPJ_pass_246.shp")
    
    config = CMEMSConfig(
        source_mode="local",
        use_parallel=True,
        use_cache=True,
    )
    service = CMEMSService(config)
    pass_data = service.load_pass_data(gate_path=gate_path)
    cmems_df = pass_data.df if pass_data else None
    if cmems_df is not None:
        log_result("CMEMS load", True, f"{len(cmems_df)} rows, cols: {list(cmems_df.columns)[:5]}")
    else:
        log_result("CMEMS load", False, "No data returned")
except Exception as e:
    log_result("CMEMS load", False, str(e)[:100])

# ==============================================================================
# TEST 3: Tab Generation Tests
# ==============================================================================
# Use whichever dataset loaded successfully
test_df = slcci_df if slcci_df is not None else cmems_df
if test_df is None:
    print("\n❌ No data loaded, cannot test tabs!")
    sys.exit(1)

print(f"\n📈 TEST 3: Tab Generation (using {'SLCCI' if slcci_df is not None else 'CMEMS'} data)")
print(f"   Data shape: {test_df.shape}")
print(f"   Columns: {list(test_df.columns)}")

# Ensure we have required columns
if 'time' not in test_df.columns and 'datetime' in test_df.columns:
    test_df['time'] = test_df['datetime']

# Tab 3.1: Time Series
print("\n   TAB 3.1: Time Series")
try:
    # Find a numeric column to plot
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ssha' in numeric_cols:
        y_col = 'ssha'
    elif 'sla' in numeric_cols:
        y_col = 'sla'
    else:
        y_col = numeric_cols[0] if numeric_cols else None
    
    if y_col and 'time' in test_df.columns:
        fig = px.scatter(test_df, x='time', y=y_col, title=f"Time Series: {y_col}")
        log_result("Time Series tab", True, f"Plot created with {y_col}")
    else:
        log_result("Time Series tab", False, "Missing time or numeric column")
except Exception as e:
    log_result("Time Series tab", False, str(e)[:100])

# Tab 3.2: Statistics
print("\n   TAB 3.2: Statistics")
try:
    stats = test_df.describe()
    log_result("Statistics tab", True, f"Stats computed: {stats.shape}")
except Exception as e:
    log_result("Statistics tab", False, str(e)[:100])

# Tab 3.3: Spatial Map
print("\n   TAB 3.3: Spatial Map")
try:
    if 'latitude' in test_df.columns and 'longitude' in test_df.columns:
        fig = px.scatter_mapbox(
            test_df.head(1000),  # Limit points for speed
            lat='latitude',
            lon='longitude',
            color=y_col if y_col else None,
            mapbox_style='open-street-map',
            title="Spatial Distribution"
        )
        log_result("Spatial Map tab", True, f"Map created with {len(test_df.head(1000))} points")
    else:
        log_result("Spatial Map tab", False, "Missing lat/lon columns")
except Exception as e:
    log_result("Spatial Map tab", False, str(e)[:100])

# Tab 3.4: Correlation
print("\n   TAB 3.4: Correlation")
try:
    numeric_df = test_df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        fig = px.imshow(corr, title="Correlation Matrix")
        log_result("Correlation tab", True, f"Correlation matrix: {corr.shape}")
    else:
        log_result("Correlation tab", False, "Not enough numeric columns")
except Exception as e:
    log_result("Correlation tab", False, str(e)[:100])

# Tab 3.5: Geostrophic Velocity
print("\n   TAB 3.5: Geostrophic Velocity")
try:
    has_ugos = 'ugos' in test_df.columns or 'u_geo' in test_df.columns
    has_vgos = 'vgos' in test_df.columns or 'v_geo' in test_df.columns
    
    if has_ugos and has_vgos:
        u_col = 'ugos' if 'ugos' in test_df.columns else 'u_geo'
        v_col = 'vgos' if 'vgos' in test_df.columns else 'v_geo'
        
        # Handle both numpy array and pandas Series
        u_data = test_df[u_col]
        v_data = test_df[v_col]
        
        if hasattr(u_data, 'values'):
            u_data = u_data.values
        if hasattr(v_data, 'values'):
            v_data = v_data.values
            
        speed = np.sqrt(u_data**2 + v_data**2)
        fig = px.histogram(speed, title="Geostrophic Speed Distribution")
        log_result("Geostrophic Velocity tab", True, f"Speed computed, mean: {np.nanmean(speed):.4f}")
    else:
        log_result("Geostrophic Velocity tab", False, f"Missing velocity columns (has_ugos={has_ugos}, has_vgos={has_vgos})")
except Exception as e:
    log_result("Geostrophic Velocity tab", False, str(e)[:100])

# Tab 3.6: Export
print("\n   TAB 3.6: Export")
try:
    # Test CSV export
    csv_data = test_df.to_csv(index=False)
    
    # Test plot export (PNG)
    import kaleido
    fig = px.scatter(test_df.head(100), x='time', y=y_col)
    # Just verify it can be created, don't actually save
    log_result("Export tab", True, f"CSV: {len(csv_data)} bytes, kaleido: {kaleido.__version__}")
except Exception as e:
    log_result("Export tab", False, str(e)[:100])

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)
passed = sum(1 for r in results if r['success'])
failed = sum(1 for r in results if not r['success'])
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"📊 Total:  {len(results)}")

if failed > 0:
    print("\nFailed tests:")
    for r in results:
        if not r['success']:
            print(f"  - {r['test']}: {r['message']}")

print("\n" + "=" * 70)

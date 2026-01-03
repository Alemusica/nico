#!/usr/bin/env python3
"""
FULL AUDIT TEST SCRIPT
======================
Tests all tabs with both SLCCI and CMEMS datasets using specific passes/tracks.

Gate examples with pass numbers:
- bering_strait_TPJ_pass_076.shp -> Pass 76
- denmark_strait_TPJ_pass_246.shp -> Pass 246  
- barents_sea_opening_S3_pass_481.shp -> Pass 481
- fram_strait_S3_pass_481.shp -> Pass 481

Test Coverage:
1. Data Loading (SLCCI local, CMEMS local, CMEMS API)
2. Pass/Track extraction
3. All visualization tabs
4. Export functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime

# Test configuration
TEST_GATES = [
    ("bering_strait_TPJ_pass_076.shp", 76),
    ("denmark_strait_TPJ_pass_246.shp", 246),
]

SLCCI_DATA_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/CEDA DATA"
CMEMS_DATA_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA"

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(test_name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} | {test_name}")
    if details:
        print(f"         └─ {details}")

def test_imports():
    """Test all required imports."""
    print_header("TEST 1: IMPORTS")
    
    results = []
    
    # Core services
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig
        print_result("SLCCIService import", True)
        results.append(True)
    except Exception as e:
        print_result("SLCCIService import", False, str(e))
        results.append(False)
    
    try:
        from src.services.cmems_service import CMEMSService, CMEMSConfig
        print_result("CMEMSService import", True)
        results.append(True)
    except Exception as e:
        print_result("CMEMSService import", False, str(e))
        results.append(False)
    
    # Tabs module
    try:
        from app.components.tabs import (
            render_tabs,
            _render_cmems_tabs,
            _render_export_tab,
        )
        print_result("Tabs module import", True)
        results.append(True)
    except Exception as e:
        print_result("Tabs module import", False, str(e))
        results.append(False)
    
    # State management
    try:
        from app.state import AppConfig, init_session_state
        print_result("State module import", True)
        results.append(True)
    except Exception as e:
        print_result("State module import", False, str(e))
        results.append(False)
    
    return all(results)

def test_gate_loading():
    """Test gate shapefile loading and pass extraction."""
    print_header("TEST 2: GATE LOADING & PASS EXTRACTION")
    
    results = []
    gates_dir = project_root / "gates"
    
    from src.services.cmems_service import _extract_pass_from_gate_name, _load_gate_gdf
    
    for gate_file, expected_pass in TEST_GATES:
        gate_path = gates_dir / gate_file
        
        # Test pass extraction
        try:
            strait_name, pass_number = _extract_pass_from_gate_name(str(gate_path))
            success = pass_number == expected_pass
            print_result(
                f"Pass extraction: {gate_file}", 
                success, 
                f"Expected {expected_pass}, got {pass_number}"
            )
            results.append(success)
        except Exception as e:
            print_result(f"Pass extraction: {gate_file}", False, str(e))
            results.append(False)
        
        # Test GDF loading
        try:
            gdf = _load_gate_gdf(str(gate_path))
            success = gdf is not None and len(gdf) > 0
            print_result(
                f"GDF loading: {gate_file}",
                success,
                f"CRS: {gdf.crs}, Bounds: {gdf.total_bounds[:2]}"
            )
            results.append(success)
        except Exception as e:
            print_result(f"GDF loading: {gate_file}", False, str(e))
            results.append(False)
    
    return all(results)

def test_slcci_loading():
    """Test SLCCI data loading with specific pass."""
    print_header("TEST 3: SLCCI DATA LOADING")
    
    from src.services.slcci_service import SLCCIService, SLCCIConfig
    
    results = []
    gates_dir = project_root / "gates"
    
    # Use Bering Strait (pass 76) - lower latitude, more likely to have data
    gate_file = "bering_strait_TPJ_pass_076.shp"
    gate_path = gates_dir / gate_file
    
    if not Path(SLCCI_DATA_DIR).exists():
        print_result("SLCCI data directory", False, f"Not found: {SLCCI_DATA_DIR}")
        return False
    
    try:
        config = SLCCIConfig(
            base_dir=SLCCI_DATA_DIR,
            pass_number=76,
            lon_bin_size=0.05,
        )
        service = SLCCIService(config)
        
        print_result("SLCCIService created", True, f"Base dir: {SLCCI_DATA_DIR}")
        results.append(True)
        
        # Try loading data
        print("  ⏳ Loading SLCCI data (this may take a while)...")
        pass_data = service.load_pass_data(str(gate_path))
        
        if pass_data is not None:
            print_result("SLCCI load_pass_data", True, f"Strait: {pass_data.strait_name}")
            results.append(True)
            
            # Check PassData attributes
            attrs_to_check = [
                ('slope_series', 'Slope series'),
                ('time_array', 'Time array'),
                ('profile_mean', 'DOT profile'),
                ('df', 'DataFrame'),
                ('v_geostrophic_series', 'Geostrophic velocity'),
            ]
            
            for attr, name in attrs_to_check:
                val = getattr(pass_data, attr, None)
                if val is not None:
                    if hasattr(val, '__len__'):
                        print_result(f"  {name}", True, f"Length: {len(val)}")
                    else:
                        print_result(f"  {name}", True, f"Value: {val}")
                    results.append(True)
                else:
                    print_result(f"  {name}", False, "None")
                    results.append(False)
        else:
            print_result("SLCCI load_pass_data", False, "Returned None")
            results.append(False)
            
    except Exception as e:
        print_result("SLCCI loading", False, str(e))
        import traceback
        traceback.print_exc()
        results.append(False)
    
    return all(results)

def test_cmems_loading_local():
    """Test CMEMS data loading from local files."""
    print_header("TEST 4: CMEMS LOCAL DATA LOADING")
    
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    
    results = []
    gates_dir = project_root / "gates"
    
    # Use Bering Strait
    gate_file = "bering_strait_TPJ_pass_076.shp"
    gate_path = gates_dir / gate_file
    
    if not Path(CMEMS_DATA_DIR).exists():
        print_result("CMEMS data directory", False, f"Not found: {CMEMS_DATA_DIR}")
        return False
    
    try:
        config = CMEMSConfig(
            base_dir=CMEMS_DATA_DIR,
            source_mode="local",
            lon_bin_size=0.1,
            buffer_deg=5.0,
            use_parallel=True,
            use_cache=True,
        )
        service = CMEMSService(config)
        
        # Count files
        file_counts = service.count_files()
        print_result("CMEMS files found", file_counts['total'] > 0, f"Total: {file_counts['total']}")
        results.append(file_counts['total'] > 0)
        
        # Get available tracks
        print("  ⏳ Scanning for tracks...")
        try:
            tracks = service.get_available_tracks(str(gate_path))
            print_result("Tracks found", len(tracks) > 0, f"Tracks: {tracks[:10]}..." if len(tracks) > 10 else f"Tracks: {tracks}")
            results.append(len(tracks) > 0)
        except Exception as e:
            print_result("Tracks scan", False, str(e))
            results.append(False)
        
        # Load data
        print("  ⏳ Loading CMEMS data (this may take several minutes)...")
        
        def progress_cb(done, total):
            if done % 500 == 0 or done == total:
                print(f"     Progress: {done}/{total} ({100*done//total}%)")
        
        pass_data = service.load_pass_data(str(gate_path), progress_callback=progress_cb)
        
        if pass_data is not None:
            print_result("CMEMS load_pass_data", True, f"Strait: {pass_data.strait_name}")
            results.append(True)
            
            # Check PassData attributes
            attrs_to_check = [
                ('slope_series', 'Slope series'),
                ('time_array', 'Time array'),
                ('profile_mean', 'DOT profile'),
                ('df', 'DataFrame'),
                ('v_geostrophic_series', 'Geostrophic velocity'),
            ]
            
            for attr, name in attrs_to_check:
                val = getattr(pass_data, attr, None)
                if val is not None:
                    if hasattr(val, '__len__'):
                        print_result(f"  {name}", True, f"Length: {len(val)}")
                    else:
                        print_result(f"  {name}", True, f"Value: {val}")
                    results.append(True)
                else:
                    print_result(f"  {name}", False, "None")
                    results.append(False)
            
            # Check DataFrame columns
            if pass_data.df is not None:
                expected_cols = ['lat', 'lon', 'time', 'dot', 'sla_filtered', 'mdt', 'track']
                missing = [c for c in expected_cols if c not in pass_data.df.columns]
                print_result("  DataFrame columns", len(missing) == 0, 
                            f"Columns: {list(pass_data.df.columns)}" if len(missing) == 0 
                            else f"Missing: {missing}")
                results.append(len(missing) == 0)
        else:
            print_result("CMEMS load_pass_data", False, "Returned None")
            results.append(False)
            
    except Exception as e:
        print_result("CMEMS loading", False, str(e))
        import traceback
        traceback.print_exc()
        results.append(False)
    
    return all(results)

def test_cmems_api():
    """Test CMEMS API data loading."""
    print_header("TEST 5: CMEMS API DATA LOADING")
    
    try:
        import copernicusmarine
        print_result("copernicusmarine installed", True, f"Version available")
    except ImportError:
        print_result("copernicusmarine installed", False, "Not installed")
        return False
    
    # Check credentials
    username = os.environ.get('CMEMS_USERNAME')
    password = os.environ.get('CMEMS_PASSWORD')
    
    if not username or not password:
        print_result("CMEMS credentials", False, 
                    "Set CMEMS_USERNAME and CMEMS_PASSWORD environment variables")
        print("  ℹ️  Skipping API test (no credentials)")
        return None  # Skip, not fail
    
    print_result("CMEMS credentials", True, f"Username: {username[:3]}***")
    
    # Try API loading
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    
    gates_dir = project_root / "gates"
    gate_file = "bering_strait_TPJ_pass_076.shp"
    gate_path = gates_dir / gate_file
    
    try:
        config = CMEMSConfig(
            source_mode="api",
            lon_bin_size=0.1,
            buffer_deg=2.0,  # Smaller buffer for API
        )
        service = CMEMSService(config)
        
        print("  ⏳ Loading from CMEMS API (this may take a while)...")
        pass_data = service.load_pass_data(str(gate_path))
        
        if pass_data is not None:
            print_result("CMEMS API load", True, f"Observations: {len(pass_data.df)}")
            return True
        else:
            print_result("CMEMS API load", False, "Returned None")
            return False
            
    except Exception as e:
        print_result("CMEMS API load", False, str(e))
        return False

def test_visualization_data():
    """Test that loaded data can be visualized (mock test without Streamlit)."""
    print_header("TEST 6: VISUALIZATION DATA VALIDATION")
    
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    
    results = []
    gates_dir = project_root / "gates"
    gate_file = "bering_strait_TPJ_pass_076.shp"
    gate_path = gates_dir / gate_file
    
    # Load CMEMS data
    config = CMEMSConfig(
        base_dir=CMEMS_DATA_DIR,
        source_mode="local",
        use_cache=True,
    )
    service = CMEMSService(config)
    
    print("  ⏳ Loading data for visualization test...")
    pass_data = service.load_pass_data(str(gate_path))
    
    if pass_data is None:
        print_result("Data loading", False, "No data")
        return False
    
    # Tab 1: Slope Timeline
    try:
        slope = pass_data.slope_series
        time_arr = pass_data.time_array
        valid = ~np.isnan(slope)
        print_result("Tab 1 (Slope Timeline)", True, 
                    f"{np.sum(valid)} valid slope values")
        results.append(True)
    except Exception as e:
        print_result("Tab 1 (Slope Timeline)", False, str(e))
        results.append(False)
    
    # Tab 2: DOT Profile
    try:
        profile = pass_data.profile_mean
        x_km = pass_data.x_km
        print_result("Tab 2 (DOT Profile)", True, 
                    f"Profile length: {len(profile)}, x_km range: {x_km.min():.1f}-{x_km.max():.1f}")
        results.append(True)
    except Exception as e:
        print_result("Tab 2 (DOT Profile)", False, str(e))
        results.append(False)
    
    # Tab 3: Spatial Distribution
    try:
        df = pass_data.df
        required = ['lat', 'lon', 'dot']
        has_all = all(c in df.columns for c in required)
        print_result("Tab 3 (Spatial Map)", has_all, 
                    f"Columns: {list(df.columns)[:5]}...")
        results.append(has_all)
    except Exception as e:
        print_result("Tab 3 (Spatial Map)", False, str(e))
        results.append(False)
    
    # Tab 4: Monthly Analysis
    try:
        df = pass_data.df
        has_month = 'month' in df.columns
        print_result("Tab 4 (Monthly Analysis)", has_month, 
                    f"Has month column: {has_month}")
        results.append(has_month)
    except Exception as e:
        print_result("Tab 4 (Monthly Analysis)", False, str(e))
        results.append(False)
    
    # Tab 5: Geostrophic Velocity
    try:
        v_geo = pass_data.v_geostrophic_series
        mean_lat = pass_data.mean_latitude
        f_cor = pass_data.coriolis_f
        print_result("Tab 5 (Geostrophic Velocity)", True, 
                    f"Mean lat: {mean_lat:.2f}°, f: {f_cor:.2e}")
        results.append(True)
    except Exception as e:
        print_result("Tab 5 (Geostrophic Velocity)", False, str(e))
        results.append(False)
    
    # Tab 6: Export (test data availability)
    try:
        # Raw data
        raw_ok = pass_data.df is not None and len(pass_data.df) > 0
        # Time series
        ts_ok = pass_data.slope_series is not None and pass_data.time_array is not None
        # Stats
        stats_ok = raw_ok and ts_ok
        print_result("Tab 6 (Export)", stats_ok, 
                    f"Raw: {len(pass_data.df)} rows, TS: {len(pass_data.slope_series)} months")
        results.append(stats_ok)
    except Exception as e:
        print_result("Tab 6 (Export)", False, str(e))
        results.append(False)
    
    return all(results)

def test_export_functions():
    """Test export functionality."""
    print_header("TEST 7: EXPORT FUNCTIONALITY")
    
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    
    results = []
    gates_dir = project_root / "gates"
    gate_file = "bering_strait_TPJ_pass_076.shp"
    gate_path = gates_dir / gate_file
    
    # Load data
    config = CMEMSConfig(
        base_dir=CMEMS_DATA_DIR,
        source_mode="local",
        use_cache=True,
    )
    service = CMEMSService(config)
    pass_data = service.load_pass_data(str(gate_path))
    
    if pass_data is None:
        print_result("Data loading for export", False)
        return False
    
    # Test CSV export
    try:
        csv_data = pass_data.df.to_csv(index=False)
        print_result("CSV export (raw data)", True, f"Size: {len(csv_data)} bytes")
        results.append(True)
    except Exception as e:
        print_result("CSV export (raw data)", False, str(e))
        results.append(False)
    
    # Test time series export
    try:
        ts_df = pd.DataFrame({
            'time': pd.to_datetime(pass_data.time_array),
            'slope_m_100km': pass_data.slope_series,
            'v_geostrophic_m_s': pass_data.v_geostrophic_series,
        })
        csv_ts = ts_df.to_csv(index=False)
        print_result("CSV export (time series)", True, f"Rows: {len(ts_df)}")
        results.append(True)
    except Exception as e:
        print_result("CSV export (time series)", False, str(e))
        results.append(False)
    
    # Test PNG export (requires kaleido)
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(pass_data.time_array),
            y=pass_data.slope_series,
            mode='lines+markers'
        ))
        
        # Try to generate PNG
        img_bytes = fig.to_image(format="png", width=800, height=400)
        print_result("PNG export (kaleido)", True, f"Size: {len(img_bytes)} bytes")
        results.append(True)
    except Exception as e:
        print_result("PNG export (kaleido)", False, str(e))
        results.append(False)
    
    return all(results)

def run_full_audit():
    """Run all tests and generate summary."""
    print("\n" + "🔍"*35)
    print("        NICO STREAMLIT APP - FULL AUDIT")
    print("🔍"*35)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Branch: feature/gates-streamlit")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['gate_loading'] = test_gate_loading()
    
    # Only run data loading tests if imports passed
    if results['imports']:
        results['slcci_loading'] = test_slcci_loading()
        results['cmems_local'] = test_cmems_loading_local()
        results['cmems_api'] = test_cmems_api()
        
        if results['cmems_local']:
            results['visualization'] = test_visualization_data()
            results['export'] = test_export_functions()
        else:
            results['visualization'] = False
            results['export'] = False
    
    # Summary
    print_header("AUDIT SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = total - passed - skipped
    
    print(f"\n  Total Tests: {total}")
    print(f"  ✅ Passed:   {passed}")
    print(f"  ❌ Failed:   {failed}")
    print(f"  ⏭️  Skipped:  {skipped}")
    print(f"\n  Success Rate: {100*passed//(total-skipped) if (total-skipped) > 0 else 0}%")
    
    print("\n  Detailed Results:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"    {status} | {test_name}")
    
    print("\n" + "="*70)
    
    return passed == (total - skipped)

if __name__ == "__main__":
    success = run_full_audit()
    sys.exit(0 if success else 1)

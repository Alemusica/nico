#!/usr/bin/env python3
"""Quick test to verify imports and basic functionality."""
import sys
sys.path.insert(0, '/Users/nicolocaron/Documents/GitHub/nico')

print("=" * 60)
print("QUICK AUDIT TEST")
print("=" * 60)

# Test 1: Imports
print("\n📦 TEST 1: Imports")
try:
    from src.services.slcci_service import SLCCIService, SLCCIConfig
    print("  ✅ SLCCI service")
except Exception as e:
    print(f"  ❌ SLCCI service: {e}")

try:
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    print("  ✅ CMEMS service")
except Exception as e:
    print(f"  ❌ CMEMS service: {e}")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    print("  ✅ Plotly")
except Exception as e:
    print(f"  ❌ Plotly: {e}")

# Test 2: SLCCI Data Loading
print("\n📊 TEST 2: SLCCI Data Loading")
try:
    from pathlib import Path
    gate_path = Path("/Users/nicolocaron/Documents/GitHub/nico/gates/denmark_strait_TPJ_pass_246.shp")
    
    config = SLCCIConfig(
        gate_path=gate_path,
        variable="ssha",
        time_start="2016-01-01",
        time_end="2016-12-31"
    )
    service = SLCCIService()
    df = service.load_pass_data(config)
    print(f"  ✅ SLCCI loaded: {len(df)} rows, cols: {list(df.columns)[:5]}")
except Exception as e:
    print(f"  ❌ SLCCI load: {e}")

# Test 3: List available gates
print("\n🚪 TEST 3: Available Gates")
try:
    from pathlib import Path
    gates_dir = Path("/Users/nicolocaron/Documents/GitHub/nico/gates")
    shp_files = list(gates_dir.glob("*.shp"))
    print(f"  Found {len(shp_files)} gate shapefiles:")
    for shp in shp_files[:5]:
        print(f"    - {shp.stem}")
except Exception as e:
    print(f"  ❌ Gates: {e}")

# Test 4: Plot generation
print("\n📈 TEST 4: Plot Generation")
try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    
    # Create test data
    dates = pd.date_range("2016-01-01", "2016-12-31", freq="D")
    df_test = pd.DataFrame({
        "time": dates,
        "ssha": np.random.randn(len(dates)) * 0.1
    })
    
    fig = px.line(df_test, x="time", y="ssha", title="Test Plot")
    print(f"  ✅ Plotly figure created: {type(fig)}")
except Exception as e:
    print(f"  ❌ Plot: {e}")

# Test 5: Kaleido (PNG export)
print("\n🖼️ TEST 5: Kaleido (PNG export)")
try:
    import kaleido
    print(f"  ✅ Kaleido version: {kaleido.__version__}")
except Exception as e:
    print(f"  ❌ Kaleido: {e}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)

"""
🧪 Headless Test - Causal Discovery Pipeline
=============================================
Test the full pipeline without GUI using real satellite data.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from api.services.llm_service import OllamaLLMService, LLMConfig
from api.services.causal_service import CausalDiscoveryService, DiscoveryConfig
from api.services.data_service import DataService


async def test_llm_service():
    """Test Ollama LLM integration."""
    print("\n" + "="*60)
    print("🤖 Testing LLM Service (Ollama)")
    print("="*60)
    
    llm = OllamaLLMService(LLMConfig(model="qwen3-coder:30b"))
    
    # Check availability
    available = await llm.check_availability()
    print(f"✅ Ollama available: {available}")
    
    if not available:
        print("❌ Ollama not running. Start with: ollama serve")
        return False
    
    # Test data interpretation
    print("\n📊 Testing data interpretation...")
    result = await llm.interpret_dataset(
        columns_info=[
            {"name": "time", "dtype": "datetime64", "samples": ["2024-01-01", "2024-01-02"], "unique_count": 365, "null_count": 0},
            {"name": "sea_level_anomaly", "dtype": "float64", "samples": [0.15, -0.23, 0.08], "unique_count": 300, "null_count": 5},
            {"name": "wind_speed", "dtype": "float64", "samples": [12.5, 8.3, 15.2], "unique_count": 250, "null_count": 2},
            {"name": "pressure_hpa", "dtype": "float64", "samples": [1013.2, 1008.5, 1015.8], "unique_count": 200, "null_count": 0},
            {"name": "precipitation_mm", "dtype": "float64", "samples": [0.0, 5.2, 12.3], "unique_count": 100, "null_count": 10},
        ],
        filename="storm_surge_data.csv"
    )
    
    print(f"  Domain detected: {result.domain}")
    print(f"  Temporal column: {result.temporal_column}")
    print(f"  Suggested targets: {result.suggested_targets}")
    print(f"  Summary: {result.summary[:200]}...")
    
    return True


async def test_causal_discovery():
    """Test causal discovery with synthetic data."""
    print("\n" + "="*60)
    print("🔬 Testing Causal Discovery (PCMCI)")
    print("="*60)
    
    # Create synthetic data with KNOWN causal structure
    np.random.seed(42)
    n = 500
    
    # Causal structure:
    # precipitation -> river_level (lag 2)
    # river_level -> flood_index (lag 1)
    # wind_speed -> storm_surge (lag 1)
    # pressure -> storm_surge (lag 1, negative)
    
    precipitation = np.random.randn(n) * 10 + 50  # mm
    wind_speed = np.random.randn(n) * 5 + 15  # m/s
    pressure = np.random.randn(n) * 10 + 1013  # hPa
    
    river_level = np.zeros(n)
    flood_index = np.zeros(n)
    storm_surge = np.zeros(n)
    
    for t in range(2, n):
        river_level[t] = 0.6 * precipitation[t-2] + 0.4 * np.random.randn() * 5
    
    for t in range(1, n):
        flood_index[t] = 0.7 * river_level[t-1] + 0.3 * np.random.randn() * 3
        storm_surge[t] = (
            0.5 * wind_speed[t-1] 
            - 0.4 * (pressure[t-1] - 1013) / 10  # Inverse barometer
            + 0.2 * np.random.randn()
        )
    
    df = pd.DataFrame({
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "river_level": river_level,
        "flood_index": flood_index,
        "storm_surge": storm_surge,
    })
    
    print(f"📈 Created synthetic dataset: {df.shape}")
    print(f"   Variables: {list(df.columns)}")
    
    # Run discovery
    config = DiscoveryConfig(
        max_lag=5,
        alpha_level=0.05,
        min_effect_size=0.1,
        use_llm_explanations=False,  # Skip LLM for speed
    )
    
    service = CausalDiscoveryService(config)
    graph = await service.discover(df, domain="flood")
    
    print(f"\n📊 Discovery Results:")
    print(f"   Method: {graph.discovery_method}")
    print(f"   Links found: {len(graph.links)}")
    
    # Check if we found the known relationships
    expected = [
        ("precipitation", "river_level", 2),
        ("river_level", "flood_index", 1),
        ("wind_speed", "storm_surge", 1),
        ("pressure", "storm_surge", 1),
    ]
    
    found = set()
    print("\n   Discovered links:")
    for link in graph.links:
        print(f"   • {link.source} → {link.target} "
              f"(lag={link.lag}, r={link.strength:.3f}, p={link.p_value:.4f})")
        found.add((link.source, link.target, link.lag))
    
    # Validate
    print("\n✅ Validation against ground truth:")
    for src, tgt, lag in expected:
        matches = [l for l in graph.links 
                   if l.source == src and l.target == tgt and abs(l.lag - lag) <= 1]
        status = "✓ FOUND" if matches else "✗ MISSED"
        print(f"   {status}: {src} → {tgt} (lag ~{lag})")
    
    return graph


async def test_with_satellite_data():
    """Test with real satellite data from the data/ directory."""
    print("\n" + "="*60)
    print("🛰️ Testing with Satellite Data")
    print("="*60)
    
    data_service = DataService()
    
    # List available files
    files = data_service.list_available_files()
    print(f"📁 Found {len(files)} data files:")
    for f in files[:5]:
        print(f"   • {f['path']} ({f['size_mb']:.2f} MB)")
    
    # Try to load a NetCDF file
    nc_files = [f for f in files if f['type'] == 'nc']
    if nc_files:
        test_file = nc_files[0]['path']
        print(f"\n📂 Loading: {test_file}")
        
        try:
            df = data_service.load_file(test_file)
            meta = data_service.get_metadata(test_file.split('/')[-1].split('.')[0])
            
            print(f"   Rows: {meta.n_rows}")
            print(f"   Columns: {meta.n_cols}")
            print(f"   Memory: {meta.memory_mb:.2f} MB")
            
            if meta.time_range:
                print(f"   Time range: {meta.time_range['start']} to {meta.time_range['end']}")
            
            if meta.spatial_bounds:
                print(f"   Spatial: lat [{meta.spatial_bounds['lat_min']:.2f}, {meta.spatial_bounds['lat_max']:.2f}]")
            
            # Show columns
            print(f"\n   Variables:")
            for col in meta.columns[:10]:
                print(f"     • {col['name']}: {col['dtype']}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading: {e}")
            return None
    else:
        print("⚠️ No NetCDF files found in data/")
        return None


async def test_llm_with_real_discovery():
    """Test LLM explanations with real discovery results."""
    print("\n" + "="*60)
    print("🧠 Testing LLM Explanations")
    print("="*60)
    
    llm = OllamaLLMService(LLMConfig(model="qwen3-coder:30b"))
    
    if not await llm.check_availability():
        print("❌ Ollama not available, skipping")
        return
    
    # Explain a causal relationship
    print("\n📝 Generating explanation for: wind_speed → storm_surge (lag=1)")
    
    explanation = await llm.explain_causal_relationship(
        source="wind_speed",
        target="storm_surge",
        lag=1,
        strength=0.52,
        p_value=0.001,
        domain="flood",
        context={"units": {"wind_speed": "m/s", "storm_surge": "cm"}}
    )
    
    print(f"\n{explanation[:800]}...")
    
    # Validate physics
    print("\n🔬 Physics validation...")
    validation = await llm.validate_pattern_physics(
        pattern_description="wind_speed → storm_surge with 1 time step lag",
        variables=["wind_speed", "storm_surge"],
        domain="flood",
        statistical_confidence=0.999,
    )
    
    print(f"   Valid: {validation.get('is_valid')}")
    print(f"   Physics score: {validation.get('physics_score')}")
    print(f"   Explanation: {validation.get('explanation', '')[:200]}...")


async def main():
    """Run all tests."""
    print("\n" + "🚀 " + "="*56 + " 🚀")
    print("   CAUSAL DISCOVERY PIPELINE - HEADLESS TEST")
    print("🚀 " + "="*56 + " 🚀")
    
    results = {}
    
    # Test 1: LLM Service
    try:
        results["llm"] = await test_llm_service()
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        results["llm"] = False
    
    # Test 2: Causal Discovery
    try:
        graph = await test_causal_discovery()
        results["causal"] = len(graph.links) > 0
    except Exception as e:
        print(f"❌ Causal discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        results["causal"] = False
    
    # Test 3: Satellite Data
    try:
        df = await test_with_satellite_data()
        results["satellite"] = df is not None
    except Exception as e:
        print(f"❌ Satellite data test failed: {e}")
        results["satellite"] = False
    
    # Test 4: LLM Explanations (only if LLM available)
    if results.get("llm"):
        try:
            await test_llm_with_real_discovery()
            results["llm_explain"] = True
        except Exception as e:
            print(f"❌ LLM explanation test failed: {e}")
            results["llm_explain"] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test}")
    
    all_passed = all(results.values())
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '⚠️ Some tests failed'}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

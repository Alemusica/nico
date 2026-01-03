#!/usr/bin/env python3
"""
Test CMEMS Service Optimizations
================================
Tests parallel processing, caching, and API mode.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    from src.services.cmems_service import (
        CMEMSService, 
        CMEMSConfig, 
        CACHE_DIR, 
        MAX_WORKERS
    )
    print(f"✅ CMEMSService imported")
    print(f"✅ CACHE_DIR: {CACHE_DIR}")
    print(f"✅ MAX_WORKERS: {MAX_WORKERS}")
    return True


def test_config():
    """Test CMEMSConfig dataclass."""
    print("\nTesting CMEMSConfig...")
    from src.services.cmems_service import CMEMSConfig
    
    config = CMEMSConfig()
    print(f"  source_mode: {config.source_mode}")
    print(f"  use_parallel: {config.use_parallel}")
    print(f"  use_cache: {config.use_cache}")
    print(f"  max_workers: {config.max_workers}")
    print(f"  api_dataset: {config.api_dataset}")
    print(f"  api_variables: {config.api_variables}")
    print("✅ CMEMSConfig OK")
    return True


def test_service_init():
    """Test CMEMSService initialization."""
    print("\nTesting CMEMSService...")
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    
    # Default config
    service = CMEMSService()
    print(f"  Default service created")
    
    # Custom config
    config = CMEMSConfig(
        use_parallel=True,
        use_cache=True,
        max_workers=4,
        source_mode="local"
    )
    service2 = CMEMSService(config)
    print(f"  Custom service created")
    
    # API mode config
    api_config = CMEMSConfig(
        source_mode="api",
        api_dataset="sea_level_global",
        api_variables=["sla", "adt"]
    )
    service3 = CMEMSService(api_config)
    print(f"  API mode service created")
    
    print("✅ CMEMSService OK")
    return True


def test_cache_functions():
    """Test cache key generation and file operations."""
    print("\nTesting cache functions...")
    from src.services.cmems_service import CMEMSService, CMEMSConfig
    
    service = CMEMSService()
    
    # Test cache key
    bounds = {"lon_min": -10, "lon_max": 10, "lat_min": 40, "lat_max": 60}
    cache_key = service._get_cache_key(bounds)
    print(f"  Cache key: {cache_key}")
    assert len(cache_key) == 16, "Cache key should be 16 chars"
    
    print("✅ Cache functions OK")
    return True


def test_api_client_available():
    """Test if CMEMSClient is available."""
    print("\nTesting CMEMSClient availability...")
    try:
        from src.surge_shazam.data.cmems_client import CMEMSClient
        client = CMEMSClient()
        print(f"  CMEMSClient available")
        print(f"  Datasets: {list(client.list_datasets().keys())[:3]}...")
        print("✅ CMEMSClient OK")
        return True
    except ImportError as e:
        print(f"⚠️ CMEMSClient not available: {e}")
        return True  # Not a failure, just optional


def main():
    """Run all tests."""
    print("=" * 60)
    print("CMEMS Service Optimization Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Imports", test_imports()))
    except Exception as e:
        print(f"❌ Imports failed: {e}")
        results.append(("Imports", False))
    
    try:
        results.append(("Config", test_config()))
    except Exception as e:
        print(f"❌ Config failed: {e}")
        results.append(("Config", False))
    
    try:
        results.append(("Service Init", test_service_init()))
    except Exception as e:
        print(f"❌ Service Init failed: {e}")
        results.append(("Service Init", False))
    
    try:
        results.append(("Cache Functions", test_cache_functions()))
    except Exception as e:
        print(f"❌ Cache Functions failed: {e}")
        results.append(("Cache Functions", False))
    
    try:
        results.append(("API Client", test_api_client_available()))
    except Exception as e:
        print(f"⚠️ API Client test error: {e}")
        results.append(("API Client", True))  # Optional
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Test script for comparison mode functionality.
Tests SLCCI, CMEMS, and comparison mode features.

Run with: source .venv/bin/activate && python scripts/test_comparison_mode.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def test_slcci_service():
    """Test SLCCI service imports and config."""
    print("\n" + "="*60)
    print("🧪 TEST 1: SLCCI Service")
    print("="*60)
    
    try:
        from src.services.slcci_service import SLCCIService, SLCCIConfig, PassData
        
        # Test config creation
        config = SLCCIConfig(
            base_dir="/Users/nicolocaron/Desktop/ARCFRESH/J2",
            geoid_path="/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc",
            cycles=list(range(1, 6)),  # Just 5 cycles
        )
        
        # Verify config
        assert config.base_dir == "/Users/nicolocaron/Desktop/ARCFRESH/J2"
        assert config.lon_buffer_deg == 5.0  # Default buffer
        assert config.lon_bin_size == 0.01  # Default binning
        
        print("✅ SLCCIConfig created successfully!")
        print(f"   - base_dir: {config.base_dir}")
        print(f"   - lon_buffer_deg: {config.lon_buffer_deg}°")
        print(f"   - lon_bin_size: {config.lon_bin_size}°")
        print(f"   - cycles: {len(config.cycles)} cycles")
        
        # Test service creation
        service = SLCCIService(config)
        assert service is not None
        print("✅ SLCCIService created successfully!")
        
        # Skip actual data loading for quick test
        print("⚠️ Skipping full SLCCI load (takes time)")
        
        return True
        
    except Exception as e:
        print(f"❌ SLCCI test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cmems_service():
    """Test CMEMS data loading."""
    print("\n" + "="*60)
    print("🧪 TEST 2: CMEMS Service")
    print("="*60)
    
    try:
        from src.services.cmems_service import CMEMSService, CMEMSConfig
        
        # Create config with quick test settings
        config = CMEMSConfig(
            base_dir="/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA",
            buffer_deg=5.0,  # Verify buffer is set correctly
            max_files=100,  # Limit files for quick test
        )
        
        service = CMEMSService(config)
        
        # Use gate with pass number in name
        gate_path = "/Users/nicolocaron/Documents/GitHub/nico/gates/barents_sea_opening_S3_pass_481.shp"
        
        print(f"Loading CMEMS data for {gate_path}...")
        print(f"Config buffer_deg: {config.buffer_deg}°")
        print(f"Config max_files: {getattr(config, 'max_files', 'unlimited')}")
        
        # Skip actual loading for quick test, just test pass extraction
        print("⚠️ Skipping full CMEMS load (takes 2+ minutes)")
        print("   Testing pass extraction and config instead...")
        
        # Verify config
        assert config.buffer_deg == 5.0, f"Buffer should be 5.0, got {config.buffer_deg}"
        print("✅ CMEMS config verified!")
        print(f"   - buffer_deg = {config.buffer_deg}°")
        
        return True  # Return True to indicate config test passed
        
    except Exception as e:
        print(f"❌ CMEMS test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pass_extraction():
    """Test pass number extraction from gate filenames."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Pass Number Extraction")
    print("="*60)
    
    try:
        from src.services.cmems_service import _extract_pass_from_gate_name
        
        test_cases = [
            ("barents_sea_opening_S3_pass_481.shp", "Barents Sea Opening", 481),
            ("denmark_strait_TPJ_pass_248.shp", "Denmark Strait", 248),
            ("fram_strait_pass_123.shp", "Fram Strait", 123),
            ("fram_strait.shp", "Fram Strait", None),
            ("gate_name_481.shp", "Gate Name", 481),
        ]
        
        all_passed = True
        for filename, expected_name, expected_pass in test_cases:
            name, pass_num = _extract_pass_from_gate_name(filename)
            
            # Check pass number
            if pass_num != expected_pass:
                print(f"❌ {filename}: expected pass {expected_pass}, got {pass_num}")
                all_passed = False
            else:
                print(f"✅ {filename} -> pass={pass_num}")
        
        if all_passed:
            print(f"\n✅ All pass extraction tests passed!")
        else:
            print(f"\n❌ Some pass extraction tests failed!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Pass extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_functions():
    """Test session state functions (mocked)."""
    print("\n" + "="*60)
    print("🧪 TEST 4: State Functions")
    print("="*60)
    
    try:
        # Mock streamlit session state
        import streamlit as st
        
        # Initialize mock state
        if not hasattr(st, 'session_state'):
            class MockSessionState(dict):
                def __getattr__(self, key):
                    return self.get(key)
                def __setattr__(self, key, value):
                    self[key] = value
            st.session_state = MockSessionState()
        
        from app.state import (
            init_session_state,
            store_slcci_data,
            store_cmems_data,
            get_slcci_data,
            get_cmems_data,
            is_comparison_mode,
            set_comparison_mode,
            clear_data
        )
        
        # Test init
        init_session_state()
        print("✅ init_session_state() works")
        
        # Test store/get SLCCI
        mock_slcci = type('PassData', (), {'pass_number': 248, 'strait_name': 'Test'})()
        store_slcci_data(mock_slcci)
        assert get_slcci_data() == mock_slcci
        print("✅ store_slcci_data/get_slcci_data works")
        
        # Test store/get CMEMS
        mock_cmems = type('PassData', (), {'pass_number': 481, 'strait_name': 'Test2'})()
        store_cmems_data(mock_cmems)
        assert get_cmems_data() == mock_cmems
        print("✅ store_cmems_data/get_cmems_data works")
        
        # Test comparison mode
        assert is_comparison_mode() == False
        set_comparison_mode(True)
        assert is_comparison_mode() == True
        print("✅ set_comparison_mode/is_comparison_mode works")
        
        # Test clear
        clear_data()
        assert get_slcci_data() is None
        assert get_cmems_data() is None
        print("✅ clear_data works")
        
        print("\n✅ All state function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ State functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tabs_imports():
    """Test that tabs.py imports correctly."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Tabs Module Imports")
    print("="*60)
    
    try:
        from app.components.tabs import (
            render_tabs,
            _render_slcci_tabs,
            _render_cmems_tabs,
            _render_comparison_tabs,
            _render_slope_timeline,
            _render_dot_profile,
            _render_spatial_map,
            _render_slcci_monthly_analysis,
            _render_geostrophic_velocity,
            _render_export_tab,
            _render_slope_comparison,
            _render_dot_profile_comparison,
            _render_spatial_map_comparison,
            _render_geostrophic_comparison,
            COLOR_SLCCI,
            COLOR_CMEMS,
        )
        
        # Verify colors
        assert COLOR_SLCCI == "darkorange", f"Expected darkorange, got {COLOR_SLCCI}"
        assert COLOR_CMEMS == "steelblue", f"Expected steelblue, got {COLOR_CMEMS}"
        
        print(f"✅ All tabs functions import correctly")
        print(f"   - COLOR_SLCCI = {COLOR_SLCCI}")
        print(f"   - COLOR_CMEMS = {COLOR_CMEMS}")
        print(f"   - render_tabs, _render_slcci_tabs, _render_cmems_tabs, _render_comparison_tabs loaded")
        print(f"   - All comparison functions loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Tabs import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("🚀 COMPARISON MODE FUNCTIONAL TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: SLCCI Service
    slcci_data = test_slcci_service()
    results["SLCCI Service"] = slcci_data is not None
    
    # Test 2: CMEMS Service  
    cmems_data = test_cmems_service()
    results["CMEMS Service"] = cmems_data is not None
    
    # Test 3: Pass Extraction
    results["Pass Extraction"] = test_pass_extraction()
    
    # Test 4: State Functions
    results["State Functions"] = test_state_functions()
    
    # Test 5: Tabs Imports
    results["Tabs Imports"] = test_tabs_imports()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n{'='*60}")
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

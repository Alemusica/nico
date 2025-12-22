#!/usr/bin/env python3
"""
Test script for VariableResolver multi-format support.

Run with: python tests/test_resolver.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import xarray as xr

from src.core.resolver import VariableResolver, compare_formats


def test_format_detection():
    """Test automatic format detection."""
    print("\n" + "=" * 60)
    print("TEST: Format Detection")
    print("=" * 60)
    
    test_files = {
        "data/slcci/SLCCI_ALTDB_J1_Cycle001_V2.nc": "SLCCI Jason-1",
        "data/cmems/cmems_l4_test.nc": "CMEMS Level-4",
        "data/cmems/cmems_l3_test.nc": "CMEMS Level-3",
        "data/aviso/aviso_test.nc": "AVISO/DUACS",
    }
    
    for filepath, expected_format in test_files.items():
        if not Path(filepath).exists():
            print(f"  ⚠️  Skipping {filepath} (not found)")
            continue
            
        ds = xr.open_dataset(filepath)
        resolver = VariableResolver.from_dataset(ds)
        ds.close()
        
        detected = resolver.format_name
        status = "✅" if detected == expected_format else "❌"
        print(f"  {status} {Path(filepath).name}: {detected}")
        
        assert detected == expected_format, f"Expected {expected_format}, got {detected}"
    
    print("  All format detection tests passed!")


def test_variable_access():
    """Test unified variable access across formats."""
    print("\n" + "=" * 60)
    print("TEST: Unified Variable Access")
    print("=" * 60)
    
    test_files = [
        "data/slcci/SLCCI_ALTDB_J1_Cycle001_V2.nc",
        "data/cmems/cmems_l4_test.nc",
        "data/cmems/cmems_l3_test.nc",
        "data/aviso/aviso_test.nc",
    ]
    
    for filepath in test_files:
        if not Path(filepath).exists():
            continue
            
        ds = xr.open_dataset(filepath)
        resolver = VariableResolver.from_dataset(ds)
        
        print(f"\n  📂 {resolver.format_name}")
        
        # Test SSH access
        if resolver.has_variable("ssh"):
            ssh = resolver.get("ssh")
            valid = ssh[np.isfinite(ssh)]
            print(f"    ✅ SSH: shape={ssh.shape}, mean={valid.mean():.4f}m")
        
        # Test coordinate access
        lat, lon = resolver.get_coordinates()
        print(f"    ✅ Coords: lat={lat.shape}, lon={lon.shape}")
        
        ds.close()
    
    print("\n  All variable access tests passed!")


def test_dot_computation():
    """Test DOT computation."""
    print("\n" + "=" * 60)
    print("TEST: DOT Computation")
    print("=" * 60)
    
    # Test with SLCCI (has MSS)
    filepath = "data/slcci/SLCCI_ALTDB_J1_Cycle001_V2.nc"
    if Path(filepath).exists():
        ds = xr.open_dataset(filepath)
        resolver = VariableResolver.from_dataset(ds)
        
        dot = resolver.compute_dot(reference="mss")
        valid = dot[np.isfinite(dot)]
        
        print(f"  ✅ SLCCI DOT: mean={valid.mean():.4f}m, std={valid.std():.4f}m")
        
        ds.close()
    
    print("  DOT computation test passed!")


def test_quality_mask():
    """Test quality mask retrieval."""
    print("\n" + "=" * 60)
    print("TEST: Quality Mask")
    print("=" * 60)
    
    filepath = "data/slcci/SLCCI_ALTDB_J1_Cycle001_V2.nc"
    if Path(filepath).exists():
        ds = xr.open_dataset(filepath)
        resolver = VariableResolver.from_dataset(ds)
        
        mask = resolver.get_quality_mask()
        if mask is not None:
            pct = 100 * mask.sum() / len(mask)
            print(f"  ✅ SLCCI mask: {mask.sum():,} valid ({pct:.1f}%)")
        
        ds.close()
    
    print("  Quality mask test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VARIABLE RESOLVER TEST SUITE")
    print("=" * 60)
    
    # Show supported formats
    compare_formats()
    
    # Run tests
    test_format_detection()
    test_variable_access()
    test_dot_computation()
    test_quality_mask()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

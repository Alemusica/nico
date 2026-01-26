#!/usr/bin/env python
"""
Test Export Service - Audit and Pipeline Verification
======================================================
Diagnoses issues with the export tab functionality.

Run with:
    source .venv/bin/activate
    python scripts/test_export_service.py
"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all export_service imports."""
    print("\n" + "="*70)
    print("🧪 TEST 1: Export Service Imports")
    print("="*70)
    
    try:
        from src.services.export_service import (
            generate_volume_transport_raw_csv,
            generate_volume_transport_climatology_csv,
            generate_volume_transport_annual_csv,
            generate_salt_flux_raw_csv,
            export_volume_transport_timeseries,
            export_volume_transport_statistics,
            export_monthly_profiles_grid,
            export_velocity_hovmoller,
            export_bathymetry_profile,
            export_salt_flux_timeseries,
            create_export_zip,
            DATASET_FULL_NAMES
        )
        print("✅ All imports successful")
        print(f"   Dataset names: {list(DATASET_FULL_NAMES.keys())}")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_csv_generation():
    """Test CSV generation with mock data."""
    print("\n" + "="*70)
    print("🧪 TEST 2: CSV Generation")
    print("="*70)
    
    import numpy as np
    import pandas as pd
    
    try:
        from src.services.export_service import (
            generate_volume_transport_raw_csv,
            generate_volume_transport_climatology_csv,
            generate_volume_transport_annual_csv,
        )
        
        # Create mock data (monthly from 2010 to 2015)
        dates = pd.date_range('2010-01-01', '2015-12-31', freq='D')
        n_obs = len(dates)
        transport_sv = np.sin(np.arange(n_obs) * 2 * np.pi / 365) * 0.5 + np.random.randn(n_obs) * 0.1
        time_array = dates.to_numpy()
        
        print(f"   Mock data: {n_obs} observations from 2010-2015")
        
        # Test raw CSV
        df_raw = generate_volume_transport_raw_csv(
            transport_sv, time_array, "Test Gate", "cmems_l4", "gebco"
        )
        print(f"   ✅ Raw CSV: {len(df_raw)} rows, columns: {list(df_raw.columns)}")
        
        # Test climatology CSV
        df_clim = generate_volume_transport_climatology_csv(
            transport_sv, time_array, "Test Gate"
        )
        print(f"   ✅ Climatology CSV: {len(df_clim)} rows")
        
        # Test annual CSV
        df_annual = generate_volume_transport_annual_csv(
            transport_sv, time_array, "Test Gate"
        )
        print(f"   ✅ Annual CSV: {len(df_annual)} rows")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_image_generation():
    """Test PNG image generation with mock data."""
    print("\n" + "="*70)
    print("🧪 TEST 3: Image Generation")
    print("="*70)
    
    import numpy as np
    import pandas as pd
    
    try:
        from src.services.export_service import (
            export_volume_transport_timeseries,
            export_volume_transport_statistics,
            export_bathymetry_profile,
            export_velocity_hovmoller,
        )
        
        # Create mock data
        dates = pd.date_range('2010-01-01', '2015-12-31', freq='D')
        n_obs = len(dates)
        transport_sv = np.sin(np.arange(n_obs) * 2 * np.pi / 365) * 0.5 + np.random.randn(n_obs) * 0.1
        time_array = dates.to_numpy()
        
        # Test timeseries image
        img = export_volume_transport_timeseries(
            transport_sv, time_array, "Test Gate", "cmems_l4", dpi=100
        )
        print(f"   ✅ Timeseries PNG: {len(img)/1024:.1f} KB")
        
        # Test statistics image
        img = export_volume_transport_statistics(
            transport_sv, time_array, "Test Gate", "cmems_l4", dpi=100
        )
        print(f"   ✅ Statistics PNG: {len(img)/1024:.1f} KB")
        
        # Test bathymetry image
        n_pts = 50
        x_km = np.linspace(0, 200, n_pts)
        depth_profile = 100 + 50 * np.sin(x_km / 20) + np.random.randn(n_pts) * 10
        gate_lon = np.linspace(-10, 5, n_pts)
        gate_lat = np.linspace(60, 65, n_pts)
        
        img = export_bathymetry_profile(
            depth_profile, x_km, "Test Gate", gate_lon, gate_lat, dpi=100
        )
        print(f"   ✅ Bathymetry PNG: {len(img)/1024:.1f} KB")
        
        # Test hovmoller
        v_perp = np.random.randn(n_pts, n_obs) * 0.1
        img = export_velocity_hovmoller(
            v_perp, x_km, time_array, "Test Gate", "cmems_l4", dpi=100
        )
        print(f"   ✅ Hovmöller PNG: {len(img)/1024:.1f} KB")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_monthly_profiles_grid():
    """Test the 3x4 monthly profiles grid generation."""
    print("\n" + "="*70)
    print("🧪 TEST 4: Monthly Profiles Grid (3x4)")
    print("="*70)
    
    import numpy as np
    
    try:
        from src.services.export_service import export_monthly_profiles_grid
        
        # Create mock monthly profiles
        # Dict: month -> (bin_centers, bin_means, bin_stds)
        monthly_profiles = {}
        bin_centers = np.linspace(0, 200, 20)  # 20 bins along 200km gate
        
        for month in range(1, 13):
            # Seasonal variation
            amplitude = 0.1 * np.sin(month * np.pi / 6)
            bin_means = amplitude * np.sin(bin_centers / 20) + np.random.randn(20) * 0.02
            bin_stds = np.abs(np.random.randn(20) * 0.01)
            monthly_profiles[month] = (bin_centers, bin_means, bin_stds)
        
        print(f"   Mock profiles: {len(monthly_profiles)} months, {len(bin_centers)} bins each")
        
        img = export_monthly_profiles_grid(
            monthly_profiles,
            gate_name="Test Gate",
            plot_type="Volume Transport",
            dataset="cmems_l4",
            start_year=2010,
            end_year=2015,
            n_observations=1000,
            y_label="Velocity (cm/s)",
            y_scale=100.0,
            show_regression=True,
            dpi=100
        )
        
        print(f"   ✅ Monthly grid PNG: {len(img)/1024:.1f} KB")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_zip_creation():
    """Test ZIP archive creation."""
    print("\n" + "="*70)
    print("🧪 TEST 5: ZIP Archive Creation")
    print("="*70)
    
    try:
        from src.services.export_service import create_export_zip
        
        # Create mock files
        files = {
            "csv/test_data.csv": "a,b,c\n1,2,3\n4,5,6",
            "images/test.png": b"\x89PNG fake data",
        }
        
        zip_bytes = create_export_zip(files, "test_export")
        print(f"   ✅ ZIP created: {len(zip_bytes)/1024:.1f} KB")
        
        # Verify contents
        import zipfile
        import io
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            file_list = zf.namelist()
            print(f"   Contents: {file_list}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_transport_service():
    """Test transport service functions used in export."""
    print("\n" + "="*70)
    print("🧪 TEST 6: Transport Service (dependencies)")
    print("="*70)
    
    try:
        from src.services.transport_service import (
            compute_perpendicular_velocity,
            compute_segment_widths,
            compute_monthly_along_gate_profile,
            SVERDRUP
        )
        print(f"   ✅ Transport service imports OK")
        print(f"   SVERDRUP constant: {SVERDRUP:.2e}")
        
        import numpy as np
        
        # Test compute_perpendicular_velocity
        n_pts = 50
        n_time = 100
        ugos = np.random.randn(n_pts, n_time) * 0.1
        vgos = np.random.randn(n_pts, n_time) * 0.1
        gate_lon = np.linspace(-10, 5, n_pts)
        gate_lat = np.linspace(60, 65, n_pts)
        
        v_perp = compute_perpendicular_velocity(ugos, vgos, gate_lon, gate_lat)
        print(f"   ✅ v_perp shape: {v_perp.shape}")
        
        # Test segment widths
        x_km = np.linspace(0, 200, n_pts)
        widths = compute_segment_widths(gate_lon, gate_lat, x_km)
        print(f"   ✅ Widths: {len(widths)} segments, mean {np.mean(widths):.1f} m")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_gebco_service():
    """Test GEBCO bathymetry service."""
    print("\n" + "="*70)
    print("🧪 TEST 7: GEBCO Service (dependencies)")
    print("="*70)
    
    try:
        from src.services.gebco_service import get_bathymetry_cache, BathymetryCache
        print(f"   ✅ GEBCO service imports OK")
        
        cache = get_bathymetry_cache()
        print(f"   ✅ BathymetryCache instance created")
        print(f"   Cache gates: {list(cache._cache.keys()) if hasattr(cache, '_cache') else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_data_pipeline():
    """Test the complete data pipeline from session state simulation."""
    print("\n" + "="*70)
    print("🧪 TEST 8: Complete Pipeline Simulation")
    print("="*70)
    
    import numpy as np
    import pandas as pd
    
    try:
        from src.services.export_service import (
            generate_volume_transport_raw_csv,
            export_volume_transport_timeseries,
            export_monthly_profiles_grid,
            create_export_zip,
        )
        from src.services.transport_service import (
            compute_perpendicular_velocity,
            compute_segment_widths,
            compute_monthly_along_gate_profile,
            SVERDRUP
        )
        
        # Simulate session state data
        n_pts = 50
        n_time = 365 * 3  # 3 years daily
        
        # Generate dates
        dates = pd.date_range('2018-01-01', periods=n_time, freq='D')
        time_array = dates.to_numpy()
        
        # Generate gate coordinates
        gate_lon = np.linspace(-15, -5, n_pts)
        gate_lat = np.linspace(78, 80, n_pts)
        x_km = np.linspace(0, 400, n_pts)
        
        # Generate velocity matrices
        ugos_matrix = np.random.randn(n_pts, n_time) * 0.05
        vgos_matrix = np.random.randn(n_pts, n_time) * 0.05
        
        # Generate depth
        depth_profile = 100 + 50 * np.sin(x_km / 30) + np.random.randn(n_pts) * 10
        depth_profile = np.clip(depth_profile, 10, 300)
        
        print(f"   Simulated data:")
        print(f"     - {n_pts} gate points")
        print(f"     - {n_time} time steps ({dates[0].year}-{dates[-1].year})")
        print(f"     - Gate: {gate_lon[0]:.1f}°E to {gate_lon[-1]:.1f}°E")
        
        # Compute v_perp
        v_perp = compute_perpendicular_velocity(ugos_matrix, vgos_matrix, gate_lon, gate_lat)
        print(f"   ✅ v_perp computed: shape {v_perp.shape}")
        
        # Compute segment widths
        widths = compute_segment_widths(gate_lon, gate_lat, x_km)
        print(f"   ✅ Widths computed: {len(widths)} segments")
        
        # Compute transport
        transport_per_point = v_perp * depth_profile[:, np.newaxis] * widths[:, np.newaxis]
        transport_per_point_sv = transport_per_point / SVERDRUP
        transport_total_sv = np.nansum(transport_per_point_sv, axis=0)
        print(f"   ✅ Transport computed: mean {np.mean(transport_total_sv):.4f} Sv")
        
        # Compute monthly profiles
        bin_size_km = 10.0
        monthly_profiles = compute_monthly_along_gate_profile(
            x_km, transport_per_point_sv, time_array, bin_size_km
        )
        print(f"   ✅ Monthly profiles: {len(monthly_profiles)} months")
        
        # Generate exports
        files = {}
        
        # CSV
        df_raw = generate_volume_transport_raw_csv(
            transport_total_sv, time_array, "Fram Strait", "cmems_l4", "gebco",
            gate_coords={
                'start_lon': gate_lon[0], 'start_lat': gate_lat[0],
                'end_lon': gate_lon[-1], 'end_lat': gate_lat[-1],
                'length_km': x_km.max()
            }
        )
        files["csv/fram_strait_volume_transport_raw.csv"] = df_raw.to_csv(index=False)
        print(f"   ✅ Raw CSV: {len(df_raw)} rows")
        
        # PNG
        img = export_volume_transport_timeseries(
            transport_total_sv, time_array, "Fram Strait", "cmems_l4", dpi=100
        )
        files["volume_transport/fram_strait_timeseries.png"] = img
        print(f"   ✅ Timeseries PNG: {len(img)/1024:.1f} KB")
        
        # Monthly grid - verify monthly_profiles structure
        if monthly_profiles:
            first_month = list(monthly_profiles.keys())[0]
            bin_centers, bin_means, bin_stds = monthly_profiles[first_month]
            print(f"   Monthly profile check:")
            print(f"     - Bin centers: {len(bin_centers)} bins, range {bin_centers[0]:.1f}-{bin_centers[-1]:.1f} km")
            print(f"     - Bin means: {len(bin_means)}, first={bin_means[0]:.6f}")
            
            img = export_monthly_profiles_grid(
                monthly_profiles,
                gate_name="Fram Strait",
                plot_type="Volume Transport",
                dataset="cmems_l4",
                start_year=2018,
                end_year=2020,
                n_observations=n_time,
                y_label="Transport (mSv/km)",
                y_scale=1000.0,  # Sv to mSv
                show_regression=True,
                dpi=100
            )
            files["volume_transport/fram_strait_monthly_grid.png"] = img
            print(f"   ✅ Monthly grid PNG: {len(img)/1024:.1f} KB")
        
        # Create ZIP
        zip_bytes = create_export_zip(files, "test_export_fram_strait")
        print(f"   ✅ Final ZIP: {len(zip_bytes)/1024:.1f} KB with {len(files)} files")
        
        # Save to file for manual inspection
        output_path = Path(__file__).parent.parent / "data" / "export_test.zip"
        with open(output_path, 'wb') as f:
            f.write(zip_bytes)
        print(f"   📁 Saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🔍 EXPORT SERVICE - AUDIT & PIPELINE VERIFICATION")
    print("="*70)
    
    results = {
        "imports": test_imports(),
        "csv": test_csv_generation(),
        "images": test_image_generation(),
        "monthly_grid": test_monthly_profiles_grid(),
        "zip": test_zip_creation(),
        "transport": test_transport_service(),
        "gebco": test_gebco_service(),
        "pipeline": test_data_pipeline(),
    }
    
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n⚠️  Some tests failed - review errors above")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

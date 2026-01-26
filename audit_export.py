#!/usr/bin/env python
"""Audit della pipeline Export per verificare che funzioni."""
import sys
import numpy as np

def main():
    print("=" * 60)
    print("AUDIT: Export Pipeline")
    print("=" * 60)

    # 1. Test import funzioni
    try:
        from src.services.export_service import (
            export_mean_dot_profile,
            export_gate_spatial_map,
            export_velocity_comparison_timeseries,
            export_monthly_velocity_grid,
            export_monthly_dot_analysis,
            export_bathymetry_profile_fixed
        )
        print("OK 1. All 7 new export functions imported")
        
        # 2. Test che le funzioni siano callable
        import inspect
        funcs = [
            ("export_mean_dot_profile", export_mean_dot_profile),
            ("export_gate_spatial_map", export_gate_spatial_map),
            ("export_velocity_comparison_timeseries", export_velocity_comparison_timeseries),
            ("export_monthly_velocity_grid", export_monthly_velocity_grid),
            ("export_monthly_dot_analysis", export_monthly_dot_analysis),
            ("export_bathymetry_profile_fixed", export_bathymetry_profile_fixed)
        ]
        for name, f in funcs:
            sig = inspect.signature(f)
            print(f"   - {name}: {len(sig.parameters)} params")
            
    except Exception as e:
        print(f"FAIL 1. Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 3. Test con dati sintetici
    print("\n" + "=" * 60)
    print("AUDIT: Test with synthetic data")
    print("=" * 60)

    # Dati sintetici
    n_pts = 50
    n_time = 100
    x_km = np.linspace(0, 300, n_pts)
    gate_lon = np.linspace(-20, -10, n_pts)
    gate_lat = np.linspace(70, 75, n_pts)
    time_array = np.arange('2020-01', '2028-05', dtype='datetime64[M]')[:n_time]
    v_perp = np.random.randn(n_pts, n_time) * 0.1
    dot_matrix = np.random.randn(n_pts, n_time) * 0.5
    depth_profile = np.abs(np.random.randn(n_pts) * 500 + 200)

    errors = []
    
    # Test export_mean_dot_profile
    try:
        dot_mean = np.nanmean(dot_matrix, axis=1)
        dot_std = np.nanstd(dot_matrix, axis=1)
        img = export_mean_dot_profile(dot_mean, x_km, "Test Gate", "cmems_l4", dot_std, 2020, 2028, n_time, 150)
        print(f"OK 2. export_mean_dot_profile: {len(img)/1024:.1f} KB")
    except Exception as e:
        print(f"FAIL 2. export_mean_dot_profile failed: {e}")
        errors.append(("export_mean_dot_profile", e))

    # Test export_gate_spatial_map
    try:
        img = export_gate_spatial_map(gate_lon, gate_lat, "Test Gate", 150)
        print(f"OK 3. export_gate_spatial_map: {len(img)/1024:.1f} KB")
    except Exception as e:
        print(f"FAIL 3. export_gate_spatial_map failed: {e}")
        errors.append(("export_gate_spatial_map", e))

    # Test export_velocity_comparison_timeseries
    try:
        v_perp_mean_ts = np.nanmean(v_perp, axis=0) * 100
        img = export_velocity_comparison_timeseries(v_perp_mean_ts, time_array, "Test Gate", None, "cmems_l4", 150)
        print(f"OK 4. export_velocity_comparison_timeseries: {len(img)/1024:.1f} KB")
    except Exception as e:
        print(f"FAIL 4. export_velocity_comparison_timeseries failed: {e}")
        errors.append(("export_velocity_comparison_timeseries", e))

    # Test export_monthly_velocity_grid
    try:
        img = export_monthly_velocity_grid(v_perp, x_km, time_array, "Test Gate", "cmems_l4", 150)
        print(f"OK 5. export_monthly_velocity_grid: {len(img)/1024:.1f} KB")
    except Exception as e:
        print(f"FAIL 5. export_monthly_velocity_grid failed: {e}")
        errors.append(("export_monthly_velocity_grid", e))

    # Test export_monthly_dot_analysis
    try:
        img = export_monthly_dot_analysis(dot_matrix, x_km, time_array, "Test Gate", "cmems_l4", 150)
        print(f"OK 6. export_monthly_dot_analysis: {len(img)/1024:.1f} KB")
    except Exception as e:
        print(f"FAIL 6. export_monthly_dot_analysis failed: {e}")
        errors.append(("export_monthly_dot_analysis", e))

    # Test export_bathymetry_profile_fixed
    try:
        img = export_bathymetry_profile_fixed(depth_profile, x_km, "Test Gate", gate_lon, gate_lat, 150)
        print(f"OK 7. export_bathymetry_profile_fixed: {len(img)/1024:.1f} KB")
    except Exception as e:
        print(f"FAIL 7. export_bathymetry_profile_fixed failed: {e}")
        errors.append(("export_bathymetry_profile_fixed", e))

    print("\n" + "=" * 60)
    if errors:
        print(f"AUDIT FAILED: {len(errors)} errors")
        for name, e in errors:
            print(f"  - {name}: {e}")
        return 1
    else:
        print("AUDIT PASSED: All functions work correctly!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

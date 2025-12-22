"""
Test Suite for Data Loading
============================
Tests for NetCDF loading, chunking, and partial file handling.

Run with: pytest tests/test_loaders.py -v
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
import os
from pathlib import Path


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_slcci_dataset():
    """Create a sample SLCCI-like dataset for testing."""
    n_points = 1000
    
    ds = xr.Dataset({
        "corssh": (["time"], np.random.randn(n_points) * 0.5 + 10),
        "mean_sea_surface": (["time"], np.random.randn(n_points) * 0.3 + 10),
        "latitude": (["time"], np.random.uniform(60, 80, n_points)),
        "longitude": (["time"], np.random.uniform(-50, 20, n_points)),
        "TimeDay": (["time"], np.arange(n_points) * 0.001 + 1000),
        "validation_flag": (["time"], np.zeros(n_points, dtype=np.int8)),
        "swh": (["time"], np.random.uniform(0.5, 5, n_points)),
    })
    ds.attrs["source"] = "SLCCI Test"
    return ds


@pytest.fixture
def sample_cmems_dataset():
    """Create a sample CMEMS-like dataset for testing."""
    n_points = 500
    
    ds = xr.Dataset({
        "sla_filtered": (["time"], np.random.randn(n_points) * 0.1),
        "adt": (["time"], np.random.randn(n_points) * 0.5 + 0.8),
        "latitude": (["time"], np.random.uniform(-60, 60, n_points)),
        "longitude": (["time"], np.random.uniform(-180, 180, n_points)),
        "time": (["time"], np.arange(n_points)),
    })
    ds.attrs["source"] = "CMEMS Test"
    return ds


@pytest.fixture
def temp_nc_file(sample_slcci_dataset, tmp_path):
    """Create a temporary NetCDF file."""
    filepath = tmp_path / "test_SLCCI_ALTDB_J1_Cycle001_V2.nc"
    sample_slcci_dataset.to_netcdf(filepath)
    return filepath


@pytest.fixture
def temp_nc_files(sample_slcci_dataset, tmp_path):
    """Create multiple temporary NetCDF files."""
    files = []
    for i in range(1, 4):
        filepath = tmp_path / f"SLCCI_ALTDB_J1_Cycle{i:03d}_V2.nc"
        # Modify data slightly for each cycle
        ds = sample_slcci_dataset.copy()
        ds["TimeDay"] = ds["TimeDay"] + (i * 10)  # Different time ranges
        ds.to_netcdf(filepath)
        files.append(filepath)
    return files


@pytest.fixture
def real_data_dir():
    """Path to real data directory (skip if not available)."""
    data_dir = Path("/Users/alessioivoycazzaniga/nico/data/slcci")
    if not data_dir.exists():
        pytest.skip("Real data directory not available")
    return data_dir


# =============================================================================
# BASIC LOADING TESTS
# =============================================================================

class TestBasicLoading:
    """Tests for basic file loading functionality."""
    
    def test_load_single_file(self, temp_nc_file):
        """Test loading a single NetCDF file."""
        ds = xr.open_dataset(temp_nc_file)
        
        assert ds is not None
        assert "corssh" in ds.data_vars
        assert "latitude" in ds.data_vars
        assert len(ds["corssh"]) == 1000
    
    def test_load_with_decode_times_false(self, temp_nc_file):
        """Test loading with decode_times=False."""
        ds = xr.open_dataset(temp_nc_file, decode_times=False)
        
        assert ds is not None
        # TimeDay should remain numeric
        assert np.issubdtype(ds["TimeDay"].dtype, np.number)
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            xr.open_dataset("/nonexistent/path/file.nc")
    
    def test_load_corrupted_file(self, tmp_path):
        """Test handling of corrupted files."""
        bad_file = tmp_path / "corrupted.nc"
        bad_file.write_text("not a netcdf file")
        
        with pytest.raises(Exception):  # Could be various exceptions
            xr.open_dataset(bad_file)


# =============================================================================
# CHUNKED LOADING TESTS
# =============================================================================

class TestChunkedLoading:
    """Tests for chunked/lazy loading with dask."""
    
    def test_load_with_chunks(self, temp_nc_file):
        """Test loading with chunking enabled."""
        ds = xr.open_dataset(temp_nc_file, chunks={"time": 100})
        
        assert ds is not None
        # Should be dask arrays
        assert hasattr(ds["corssh"].data, "dask")
    
    def test_chunk_sizes(self, temp_nc_file):
        """Test different chunk sizes."""
        for chunk_size in [10, 100, 500]:
            ds = xr.open_dataset(temp_nc_file, chunks={"time": chunk_size})
            assert ds is not None
            # Compute a small operation to verify chunks work
            mean_val = ds["corssh"].mean().compute()
            assert np.isfinite(mean_val)
    
    def test_compute_from_chunks(self, temp_nc_file):
        """Test computing values from chunked dataset."""
        ds = xr.open_dataset(temp_nc_file, chunks={"time": 100})
        
        # Should be able to compute
        corssh_mean = ds["corssh"].mean().compute()
        assert np.isfinite(corssh_mean)
        
        # Full array computation
        corssh_vals = ds["corssh"].values
        assert len(corssh_vals) == 1000


# =============================================================================
# PARTIAL LOADING TESTS
# =============================================================================

class TestPartialLoading:
    """Tests for loading subsets of data."""
    
    def test_load_single_variable(self, temp_nc_file):
        """Test loading only specific variables."""
        ds = xr.open_dataset(temp_nc_file)
        
        # Select only corssh
        corssh = ds["corssh"]
        assert corssh is not None
        assert len(corssh) == 1000
    
    def test_load_subset_by_index(self, temp_nc_file):
        """Test loading a subset by index."""
        ds = xr.open_dataset(temp_nc_file)
        
        subset = ds.isel(time=slice(0, 100))
        assert len(subset["corssh"]) == 100
    
    def test_load_subset_by_condition(self, temp_nc_file):
        """Test loading subset by condition (lat/lon filter)."""
        ds = xr.open_dataset(temp_nc_file)
        
        # Filter by latitude
        mask = ds["latitude"] > 70
        subset = ds.where(mask, drop=True)
        
        assert len(subset["corssh"]) < 1000
        assert all(subset["latitude"].values > 70)
    
    def test_spatial_filter(self, temp_nc_file):
        """Test spatial filtering like in the app."""
        ds = xr.open_dataset(temp_nc_file)
        
        lat = ds["latitude"].values.flatten()
        lon = ds["longitude"].values.flatten()
        
        # Create mask
        lat_range = (65, 75)
        lon_range = (-30, 10)
        
        mask = (
            (lat >= lat_range[0]) & (lat <= lat_range[1]) &
            (lon >= lon_range[0]) & (lon <= lon_range[1])
        )
        
        filtered_lat = lat[mask]
        assert len(filtered_lat) <= len(lat)
        if len(filtered_lat) > 0:
            assert all(filtered_lat >= lat_range[0])
            assert all(filtered_lat <= lat_range[1])


# =============================================================================
# MULTIPLE FILE LOADING TESTS
# =============================================================================

class TestMultipleFileLoading:
    """Tests for loading multiple files."""
    
    def test_load_multiple_files(self, temp_nc_files):
        """Test loading multiple NetCDF files."""
        datasets = []
        for f in temp_nc_files:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        
        assert len(datasets) == 3
        for ds in datasets:
            assert "corssh" in ds.data_vars
    
    def test_combine_datasets(self, temp_nc_files):
        """Test combining multiple datasets."""
        datasets = [xr.open_dataset(f) for f in temp_nc_files]
        
        # Concatenate along time dimension
        combined = xr.concat(datasets, dim="time")
        
        assert len(combined["corssh"]) == 3000  # 3 files x 1000 points
    
    def test_load_with_progress(self, temp_nc_files):
        """Test loading with progress tracking."""
        loaded = []
        total = len(temp_nc_files)
        
        for i, f in enumerate(temp_nc_files):
            ds = xr.open_dataset(f)
            loaded.append(ds)
            progress = (i + 1) / total
            assert 0 < progress <= 1
        
        assert len(loaded) == 3


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Tests for data validation after loading."""
    
    def test_required_variables_present(self, temp_nc_file):
        """Test that required variables are present."""
        ds = xr.open_dataset(temp_nc_file)
        
        required_vars = ["corssh", "mean_sea_surface", "latitude", "longitude"]
        for var in required_vars:
            assert var in ds.data_vars, f"Missing required variable: {var}"
    
    def test_data_ranges_valid(self, temp_nc_file):
        """Test that data values are within expected ranges."""
        ds = xr.open_dataset(temp_nc_file)
        
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        
        # Latitude should be -90 to 90
        assert np.all((lat >= -90) & (lat <= 90))
        
        # Longitude can be 0-360 or -180-180
        assert np.all((lon >= -180) & (lon <= 360))
    
    def test_no_all_nan_variables(self, temp_nc_file):
        """Test that variables are not all NaN."""
        ds = xr.open_dataset(temp_nc_file)
        
        for var in ["corssh", "latitude", "longitude"]:
            data = ds[var].values
            assert not np.all(np.isnan(data)), f"Variable {var} is all NaN"
    
    def test_minimum_points(self, temp_nc_file):
        """Test minimum number of valid points."""
        ds = xr.open_dataset(temp_nc_file)
        
        corssh = ds["corssh"].values
        valid_count = np.sum(np.isfinite(corssh))
        
        # Need at least some valid points for analysis
        assert valid_count >= 10, f"Only {valid_count} valid points"


# =============================================================================
# RESOLVER INTEGRATION TESTS
# =============================================================================

class TestResolverIntegration:
    """Tests for VariableResolver with loaded data."""
    
    def test_resolver_auto_detect_slcci(self, temp_nc_file):
        """Test resolver auto-detects SLCCI format."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.core.resolver import VariableResolver
        
        ds = xr.open_dataset(temp_nc_file)
        resolver = VariableResolver.from_dataset(ds)
        
        assert "slcci" in resolver.format_name.lower()
    
    def test_resolver_get_ssh(self, temp_nc_file):
        """Test resolver can get SSH with canonical name."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.core.resolver import VariableResolver
        
        ds = xr.open_dataset(temp_nc_file)
        resolver = VariableResolver.from_dataset(ds)
        
        ssh = resolver.get("ssh")
        assert ssh is not None
        assert len(ssh) == 1000
    
    def test_resolver_compute_dot(self, temp_nc_file):
        """Test resolver DOT computation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from src.core.resolver import VariableResolver
        
        ds = xr.open_dataset(temp_nc_file)
        resolver = VariableResolver.from_dataset(ds)
        
        dot = resolver.compute_dot()
        assert dot is not None
        assert len(dot) == 1000
        assert np.any(np.isfinite(dot))


# =============================================================================
# REAL DATA TESTS (Skip if data not available)
# =============================================================================

class TestRealData:
    """Tests with real SLCCI data (skipped if not available)."""
    
    def test_load_real_slcci_file(self, real_data_dir):
        """Test loading a real SLCCI file."""
        files = list(real_data_dir.glob("*.nc"))
        assert len(files) > 0, "No NetCDF files found"
        
        ds = xr.open_dataset(files[0])
        assert "corssh" in ds.data_vars
    
    def test_real_data_has_valid_points(self, real_data_dir):
        """Test real data has enough valid points."""
        files = list(real_data_dir.glob("*.nc"))
        ds = xr.open_dataset(files[0])
        
        corssh = ds["corssh"].values.flatten()
        valid = np.sum(np.isfinite(corssh))
        
        # Real files should have many valid points
        assert valid > 100000, f"Only {valid} valid points in real data"
    
    def test_load_all_real_files(self, real_data_dir):
        """Test loading all real files."""
        files = sorted(real_data_dir.glob("*.nc"))
        
        for f in files:
            ds = xr.open_dataset(f)
            assert ds is not None
            assert "corssh" in ds.data_vars
            ds.close()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_large_file_chunked_memory(self, tmp_path):
        """Test that chunked loading doesn't load everything to memory."""
        # Create a larger test file
        n_points = 100000
        ds = xr.Dataset({
            "corssh": (["time"], np.random.randn(n_points)),
            "latitude": (["time"], np.random.uniform(-90, 90, n_points)),
            "longitude": (["time"], np.random.uniform(0, 360, n_points)),
        })
        
        filepath = tmp_path / "large_test.nc"
        ds.to_netcdf(filepath)
        
        # Load with chunks - should not load all data immediately
        ds_chunked = xr.open_dataset(filepath, chunks={"time": 1000})
        
        # Just accessing should be fast (lazy loading)
        assert ds_chunked["corssh"] is not None
        
        # Compute mean - processes chunks
        mean_val = ds_chunked["corssh"].mean().compute()
        assert np.isfinite(mean_val)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

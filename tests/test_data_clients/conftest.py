"""
Shared Fixtures for Data Client Tests
=====================================

Provides common fixtures and synthetic data generators
for testing data clients without real API calls.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# Common Test Parameters
# ============================================================================

@dataclass
class TestRegion:
    """Test region definition."""
    name: str
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    description: str


# Standard test regions for CTW
TEST_REGIONS = {
    "lago_maggiore": TestRegion(
        name="Lago Maggiore",
        lat_range=(45.5, 46.5),
        lon_range=(8.0, 9.5),
        description="Alpine lake region, Italy/Switzerland",
    ),
    "north_sea": TestRegion(
        name="North Sea",
        lat_range=(51.0, 56.0),
        lon_range=(2.0, 8.0),
        description="Storm surge prone area",
    ),
    "mediterranean": TestRegion(
        name="Mediterranean",
        lat_range=(35.0, 45.0),
        lon_range=(5.0, 20.0),
        description="Mediterranean Sea region",
    ),
    "global_ocean": TestRegion(
        name="Global Ocean",
        lat_range=(-60.0, 60.0),
        lon_range=(-180.0, 180.0),
        description="Global ocean coverage",
    ),
}


# ============================================================================
# Synthetic Data Generators
# ============================================================================

@pytest.fixture
def test_region():
    """Default test region (Lago Maggiore)."""
    return TEST_REGIONS["lago_maggiore"]


@pytest.fixture
def all_test_regions():
    """All test regions."""
    return TEST_REGIONS


@pytest.fixture
def synthetic_time_range():
    """Standard time range for testing."""
    return ("2020-01-01", "2020-01-31")


@pytest.fixture
def synthetic_coords():
    """Generate synthetic coordinate grid."""
    def _generate(
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        spatial_res: float = 0.25,
        temporal_res: str = "1D",
    ) -> Dict[str, np.ndarray]:
        lats = np.arange(lat_range[0], lat_range[1], spatial_res)
        lons = np.arange(lon_range[0], lon_range[1], spatial_res)
        
        if HAS_PANDAS:
            times = pd.date_range(time_range[0], time_range[1], freq=temporal_res)
        else:
            times = np.arange(
                np.datetime64(time_range[0]),
                np.datetime64(time_range[1]),
                np.timedelta64(1, 'D')
            )
        
        return {
            "latitude": lats,
            "longitude": lons,
            "time": times,
        }
    
    return _generate


@pytest.fixture
def synthetic_xarray_dataset():
    """Generate synthetic xarray Dataset."""
    def _generate(
        variables: List[str],
        lat_range: Tuple[float, float] = (45.0, 47.0),
        lon_range: Tuple[float, float] = (8.0, 10.0),
        time_range: Tuple[str, str] = ("2020-01-01", "2020-01-15"),
        spatial_res: float = 0.25,
        attrs: Dict[str, Any] = None,
    ):
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        lats = np.arange(lat_range[0], lat_range[1], spatial_res)
        lons = np.arange(lon_range[0], lon_range[1], spatial_res)
        times = np.arange(
            np.datetime64(time_range[0]),
            np.datetime64(time_range[1]),
            np.timedelta64(1, 'D')
        )
        
        shape = (len(times), len(lats), len(lons))
        
        data_vars = {}
        for var in variables:
            data = np.random.randn(*shape).astype(np.float32)
            data_vars[var] = (['time', 'latitude', 'longitude'], data)
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons,
            },
            attrs=attrs or {'source': 'synthetic_test_data'},
        )
        
        return ds
    
    return _generate


@pytest.fixture
def synthetic_dataframe():
    """Generate synthetic pandas DataFrame."""
    def _generate(
        columns: List[str],
        n_rows: int = 100,
        start_date: str = "2020-01-01",
        include_coords: bool = True,
    ):
        if not HAS_PANDAS:
            pytest.skip("pandas not installed")
        
        data = {}
        
        if include_coords:
            data['latitude'] = np.random.uniform(40, 50, n_rows)
            data['longitude'] = np.random.uniform(5, 15, n_rows)
            data['time'] = pd.date_range(start_date, periods=n_rows, freq='H')
        
        for col in columns:
            if col not in data:
                data[col] = np.random.randn(n_rows)
        
        return pd.DataFrame(data)
    
    return _generate


# ============================================================================
# Mock Response Fixtures
# ============================================================================

@pytest.fixture
def mock_http_response():
    """Factory for mock HTTP responses."""
    class MockResponse:
        def __init__(self, status=200, json_data=None, text_data=None):
            self.status = status
            self._json = json_data
            self._text = text_data
        
        async def json(self):
            return self._json
        
        async def text(self):
            return self._text
        
        async def read(self):
            return (self._text or "").encode()
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
    
    return MockResponse


@pytest.fixture
def mock_aiohttp_session(mock_http_response):
    """Factory for mock aiohttp session."""
    class MockSession:
        def __init__(self, responses=None):
            self.responses = responses or {}
            self.requests = []
        
        def get(self, url, **kwargs):
            self.requests.append(('GET', url, kwargs))
            
            # Find matching response
            for pattern, response in self.responses.items():
                if pattern in url:
                    return response
            
            # Default response
            return mock_http_response(status=200, json_data={})
        
        def post(self, url, **kwargs):
            self.requests.append(('POST', url, kwargs))
            return mock_http_response(status=200, json_data={})
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
    
    return MockSession


# ============================================================================
# Data Validation Helpers
# ============================================================================

@pytest.fixture
def validate_dataset():
    """Validator for xarray datasets."""
    def _validate(ds, expected_vars=None, expected_dims=None):
        if not HAS_XARRAY:
            return True
        
        assert ds is not None, "Dataset is None"
        
        if expected_vars:
            for var in expected_vars:
                assert var in ds.data_vars, f"Missing variable: {var}"
        
        if expected_dims:
            for dim in expected_dims:
                assert dim in ds.dims, f"Missing dimension: {dim}"
        
        # Check for NaN values
        for var in ds.data_vars:
            nan_ratio = float(np.isnan(ds[var].values).sum()) / ds[var].values.size
            assert nan_ratio < 0.5, f"Too many NaN values in {var}: {nan_ratio:.1%}"
        
        return True
    
    return _validate


@pytest.fixture
def validate_dataframe():
    """Validator for pandas DataFrames."""
    def _validate(df, expected_cols=None, min_rows=1):
        if not HAS_PANDAS:
            return True
        
        assert df is not None, "DataFrame is None"
        assert len(df) >= min_rows, f"Too few rows: {len(df)}"
        
        if expected_cols:
            for col in expected_cols:
                assert col in df.columns, f"Missing column: {col}"
        
        return True
    
    return _validate


# ============================================================================
# Temporary Files & Caching
# ============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory for tests."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_data_file(tmp_path):
    """Factory for temporary data files."""
    def _create(filename: str, content: str = "") -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    
    return _create

"""
🌍 Test GRACE Client
====================

Tests for GRACE/GRACE-FO satellite gravity data client.
GRACE measures:
- Terrestrial Water Storage (TWS) changes
- Ice mass balance
- Ocean mass variations
- Groundwater depletion

All tests use synthetic data to avoid API dependencies.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock, patch, AsyncMock

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
# GRACE Client Implementation (stub for testing)
# ============================================================================

class GRACEMission(Enum):
    """GRACE satellite missions."""
    GRACE = "grace"  # 2002-2017
    GRACE_FO = "grace-fo"  # 2018-present


class GRACEProduct(Enum):
    """GRACE data products."""
    MASCON = "mascon"  # Mass concentration (JPL, CSR, GSFC)
    SPHERICAL_HARMONICS = "sh"  # Spherical harmonics
    TELLUS = "tellus"  # NASA TELLUS products


@dataclass
class GRACEDataset:
    """GRACE dataset definition."""
    name: str
    provider: str  # JPL, CSR, GFZ, GSFC
    product: GRACEProduct
    resolution_deg: float
    temporal_resolution_days: int
    description: str
    variables: List[str] = field(default_factory=list)
    time_range: Tuple[str, str] = ("2002-04", "present")


GRACE_DATASETS = {
    # JPL Mascons
    "jpl_mascon_rl06": GRACEDataset(
        name="JPL GRACE/GRACE-FO RL06.1 Mascon",
        provider="JPL",
        product=GRACEProduct.MASCON,
        resolution_deg=0.5,
        temporal_resolution_days=30,
        description="JPL mascon solution, 0.5° grid",
        variables=["lwe_thickness", "uncertainty"],
    ),
    
    # CSR Mascons
    "csr_mascon_rl06": GRACEDataset(
        name="CSR GRACE/GRACE-FO RL06 Mascon",
        provider="CSR",
        product=GRACEProduct.MASCON,
        resolution_deg=0.25,
        temporal_resolution_days=30,
        description="CSR mascon solution, 0.25° grid",
        variables=["lwe_thickness", "uncertainty"],
    ),
    
    # GSFC Mascons
    "gsfc_mascon_rl06": GRACEDataset(
        name="GSFC GRACE/GRACE-FO RL06 Mascon",
        provider="GSFC",
        product=GRACEProduct.MASCON,
        resolution_deg=0.5,
        temporal_resolution_days=30,
        description="GSFC mascon solution, 0.5° grid",
        variables=["lwe_thickness", "uncertainty", "land_mask"],
    ),
    
    # TELLUS Land
    "tellus_land": GRACEDataset(
        name="GRACE TELLUS Land Grid",
        provider="NASA",
        product=GRACEProduct.TELLUS,
        resolution_deg=1.0,
        temporal_resolution_days=30,
        description="NASA TELLUS terrestrial water storage",
        variables=["lwe_thickness", "scale_factor"],
    ),
    
    # TELLUS Ocean
    "tellus_ocean": GRACEDataset(
        name="GRACE TELLUS Ocean Grid",
        provider="NASA",
        product=GRACEProduct.TELLUS,
        resolution_deg=1.0,
        temporal_resolution_days=30,
        description="NASA TELLUS ocean bottom pressure",
        variables=["obp", "obp_uncertainty"],
    ),
}


class GRACEClient:
    """
    Client for GRACE/GRACE-FO satellite gravity data.
    
    Provides terrestrial water storage (TWS) anomalies for:
    - Drought monitoring
    - Groundwater depletion
    - Lake/reservoir changes
    - Ice sheet mass balance
    
    Usage:
        client = GRACEClient()
        
        # Get water storage anomaly
        ds = await client.get_water_storage(
            lat_range=(30, 50),
            lon_range=(-10, 30),
            time_range=("2020-01", "2020-12"),
        )
        
        # Get time series for region
        ts = await client.get_timeseries(
            lat_range=(45, 47),
            lon_range=(8, 10),
        )
    """
    
    def __init__(
        self,
        dataset: str = "jpl_mascon_rl06",
        cache_dir: str = None,
    ):
        self.dataset_key = dataset
        self.dataset = GRACE_DATASETS.get(dataset)
        if not self.dataset:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        self.cache_dir = cache_dir
    
    def list_datasets(self) -> Dict[str, str]:
        """List available GRACE datasets."""
        return {k: v.description for k, v in GRACE_DATASETS.items()}
    
    def get_dataset_info(self, dataset: str = None) -> Optional[GRACEDataset]:
        """Get dataset metadata."""
        key = dataset or self.dataset_key
        return GRACE_DATASETS.get(key)
    
    async def download(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
    ) -> Optional[Any]:  # xr.Dataset
        """
        Download GRACE data.
        
        Args:
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude
            time_range: (start, end) as "YYYY-MM" strings
            variables: Variables to include (default: all)
            
        Returns:
            xarray Dataset with LWE thickness anomalies
        """
        # For testing, generate synthetic data
        return await self._generate_synthetic_data(
            lat_range, lon_range, time_range, variables
        )
    
    async def get_water_storage(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Convenience method for water storage anomaly."""
        return await self.download(
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            variables=["lwe_thickness"],
        )
    
    async def get_timeseries(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str] = ("2002-04", "2024-12"),
    ) -> Optional[Any]:  # pd.DataFrame
        """
        Get area-averaged time series.
        
        Returns:
            DataFrame with monthly TWS anomaly
        """
        ds = await self.download(lat_range, lon_range, time_range)
        
        if ds is None or not HAS_PANDAS:
            return None
        
        # Compute area average
        lwe = ds["lwe_thickness"]
        
        # Weight by cosine of latitude
        weights = np.cos(np.deg2rad(ds.latitude))
        weights = weights / weights.sum()
        
        # Weighted mean over space
        ts = (lwe * weights).sum(dim=['latitude', 'longitude'])
        
        df = pd.DataFrame({
            'time': ts.time.values,
            'lwe_thickness_cm': ts.values,
        })
        df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    async def compute_trend(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Dict[str, float]:
        """
        Compute linear trend in water storage.
        
        Returns:
            Dict with trend (cm/year) and uncertainty
        """
        ts = await self.get_timeseries(lat_range, lon_range, time_range)
        
        if ts is None or len(ts) < 12:
            return {"trend_cm_yr": 0.0, "uncertainty": 0.0}
        
        # Simple linear regression
        x = np.arange(len(ts))
        y = ts["lwe_thickness_cm"].values
        
        # Remove NaN
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]
        
        if len(x) < 3:
            return {"trend_cm_yr": 0.0, "uncertainty": 0.0}
        
        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Convert to cm/year (monthly data)
        trend_cm_yr = slope * 12
        
        # Estimate uncertainty (simplified)
        residuals = y - np.polyval(coeffs, x)
        uncertainty = np.std(residuals) * np.sqrt(12 / len(x))
        
        return {
            "trend_cm_yr": float(trend_cm_yr),
            "uncertainty": float(uncertainty),
        }
    
    async def _generate_synthetic_data(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
    ) -> Optional[Any]:
        """Generate synthetic GRACE data."""
        if not HAS_XARRAY:
            return None
        
        variables = variables or self.dataset.variables
        
        # Create coordinates
        res = self.dataset.resolution_deg
        lats = np.arange(lat_range[0], lat_range[1], res)
        lons = np.arange(lon_range[0], lon_range[1], res)
        
        # Monthly time steps
        start = datetime.strptime(time_range[0], "%Y-%m")
        end = datetime.strptime(time_range[1], "%Y-%m")
        
        months = []
        current = start
        while current <= end:
            months.append(current)
            # Next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        times = np.array(months, dtype='datetime64[M]')
        shape = (len(times), len(lats), len(lons))
        
        data_vars = {}
        
        for var in variables:
            if var == "lwe_thickness":
                # Liquid water equivalent thickness in cm
                # Add seasonal cycle + trend + noise
                seasonal = np.zeros(shape)
                for i, t in enumerate(range(len(times))):
                    # Seasonal cycle: wet winter, dry summer
                    seasonal[i] = 5 * np.sin(2 * np.pi * t / 12)
                
                # Spatial pattern
                lat_pattern = np.cos(np.deg2rad(lats))[:, np.newaxis]
                
                # Add trend (-1 cm/year for groundwater depletion)
                trend = -0.1 * np.arange(len(times))[:, np.newaxis, np.newaxis]
                
                # Combine
                data = seasonal + lat_pattern * 3 + trend + np.random.normal(0, 1, shape)
                
            elif var == "uncertainty":
                # Uncertainty typically 1-3 cm
                data = np.random.uniform(1, 3, shape)
                
            elif var == "scale_factor":
                # Scale factors for TELLUS ~1.0
                data = np.random.uniform(0.8, 1.2, shape)
                
            elif var == "obp":
                # Ocean bottom pressure anomaly in cm
                data = np.random.normal(0, 2, shape)
                
            else:
                data = np.random.randn(*shape)
            
            data_vars[var] = (['time', 'latitude', 'longitude'], data.astype(np.float32))
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': times.astype('datetime64[ns]'),
                'latitude': lats,
                'longitude': lons,
            },
            attrs={
                'source': 'synthetic_grace',
                'dataset': self.dataset_key,
                'provider': self.dataset.provider,
                'product': self.dataset.product.value,
                'units': 'cm equivalent water height',
            }
        )
        
        return ds


# ============================================================================
# Tests
# ============================================================================

class TestGRACEMission:
    """Test GRACEMission enum."""
    
    def test_missions_defined(self):
        """Both GRACE missions should be defined."""
        missions = [m.value for m in GRACEMission]
        
        assert "grace" in missions
        assert "grace-fo" in missions
    
    def test_mission_count(self):
        """Should have 2 missions."""
        assert len(GRACEMission) == 2


class TestGRACEProduct:
    """Test GRACEProduct enum."""
    
    def test_products_defined(self):
        """All product types should be defined."""
        products = [p.value for p in GRACEProduct]
        
        assert "mascon" in products
        assert "sh" in products
        assert "tellus" in products


class TestGRACEDataset:
    """Test GRACEDataset dataclass."""
    
    def test_create_dataset(self):
        """Should create dataset definition."""
        ds = GRACEDataset(
            name="Test Dataset",
            provider="TEST",
            product=GRACEProduct.MASCON,
            resolution_deg=0.5,
            temporal_resolution_days=30,
            description="Test description",
            variables=["lwe_thickness"],
        )
        
        assert ds.name == "Test Dataset"
        assert ds.resolution_deg == 0.5
        assert "lwe_thickness" in ds.variables


class TestGRACEDatasets:
    """Test predefined GRACE datasets."""
    
    def test_jpl_mascon_defined(self):
        """JPL mascon should be defined."""
        assert "jpl_mascon_rl06" in GRACE_DATASETS
        
        ds = GRACE_DATASETS["jpl_mascon_rl06"]
        assert ds.provider == "JPL"
        assert ds.product == GRACEProduct.MASCON
    
    def test_csr_mascon_defined(self):
        """CSR mascon should be defined."""
        assert "csr_mascon_rl06" in GRACE_DATASETS
    
    def test_tellus_land_defined(self):
        """TELLUS land should be defined."""
        assert "tellus_land" in GRACE_DATASETS
        
        ds = GRACE_DATASETS["tellus_land"]
        assert ds.product == GRACEProduct.TELLUS
    
    def test_all_datasets_have_lwe(self):
        """All datasets should have LWE or OBP variable."""
        for name, ds in GRACE_DATASETS.items():
            has_main_var = (
                "lwe_thickness" in ds.variables or
                "obp" in ds.variables
            )
            assert has_main_var, f"{name} missing main variable"


class TestGRACEClient:
    """Test GRACEClient class."""
    
    @pytest.fixture
    def client(self):
        return GRACEClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
        assert client.dataset_key == "jpl_mascon_rl06"
    
    def test_create_with_dataset(self):
        """Should create client with specific dataset."""
        client = GRACEClient(dataset="csr_mascon_rl06")
        
        assert client.dataset_key == "csr_mascon_rl06"
        assert client.dataset.provider == "CSR"
    
    def test_create_unknown_dataset_raises(self):
        """Should raise for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            GRACEClient(dataset="nonexistent")
    
    def test_list_datasets(self, client):
        """Should list available datasets."""
        datasets = client.list_datasets()
        
        assert isinstance(datasets, dict)
        assert len(datasets) >= 4
        assert "jpl_mascon_rl06" in datasets
    
    def test_get_dataset_info(self, client):
        """Should return dataset info."""
        info = client.get_dataset_info()
        
        assert info is not None
        assert info.provider == "JPL"
        assert info.resolution_deg == 0.5
    
    @pytest.mark.asyncio
    async def test_download_returns_dataset(self, client):
        """Download should return xarray Dataset."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            lat_range=(40, 50),
            lon_range=(0, 20),
            time_range=("2020-01", "2020-12"),
        )
        
        assert ds is not None
        assert "lwe_thickness" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_get_water_storage(self, client):
        """Should get water storage data."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.get_water_storage(
            lat_range=(40, 50),
            lon_range=(0, 20),
            time_range=("2020-01", "2020-06"),
        )
        
        assert ds is not None
        assert "lwe_thickness" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_get_timeseries(self, client):
        """Should get time series."""
        if not HAS_XARRAY or not HAS_PANDAS:
            pytest.skip("xarray/pandas not installed")
        
        ts = await client.get_timeseries(
            lat_range=(45, 47),
            lon_range=(8, 10),
            time_range=("2020-01", "2020-12"),
        )
        
        assert ts is not None
        assert "time" in ts.columns
        assert "lwe_thickness_cm" in ts.columns
        assert len(ts) == 12  # 12 months
    
    @pytest.mark.asyncio
    async def test_compute_trend(self, client):
        """Should compute linear trend."""
        if not HAS_XARRAY or not HAS_PANDAS:
            pytest.skip("xarray/pandas not installed")
        
        trend = await client.compute_trend(
            lat_range=(45, 47),
            lon_range=(8, 10),
            time_range=("2015-01", "2020-12"),
        )
        
        assert "trend_cm_yr" in trend
        assert "uncertainty" in trend
        # Synthetic data has variable trend, just check it's computed
        assert isinstance(trend["trend_cm_yr"], float)
        assert isinstance(trend["uncertainty"], float)


class TestGRACEDataQuality:
    """Test synthetic GRACE data quality."""
    
    @pytest.fixture
    def client(self):
        return GRACEClient()
    
    @pytest.mark.asyncio
    async def test_lwe_values_realistic(self, client):
        """LWE values should be realistic."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            lat_range=(40, 50),
            lon_range=(0, 20),
            time_range=("2020-01", "2020-12"),
        )
        
        lwe = ds["lwe_thickness"].values
        
        # LWE typically -30 to +30 cm
        assert np.abs(lwe).max() < 50, "LWE values unrealistic"
    
    @pytest.mark.asyncio
    async def test_has_seasonal_cycle(self, client):
        """Data should show seasonal variability."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            lat_range=(40, 50),
            lon_range=(0, 20),
            time_range=("2020-01", "2020-12"),
        )
        
        # Compute temporal std
        lwe = ds["lwe_thickness"]
        temporal_std = float(lwe.std(dim='time').mean())
        
        # Should have some seasonal variation
        assert temporal_std > 1, "Data lacks seasonal variability"
    
    @pytest.mark.asyncio
    async def test_dataset_has_metadata(self, client):
        """Dataset should have proper metadata."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            lat_range=(40, 50),
            lon_range=(0, 20),
            time_range=("2020-01", "2020-06"),
        )
        
        assert "source" in ds.attrs
        assert "provider" in ds.attrs
        assert "units" in ds.attrs


class TestGRACEMultipleDatasets:
    """Test switching between GRACE datasets."""
    
    @pytest.mark.asyncio
    async def test_jpl_vs_csr_resolution(self):
        """JPL and CSR should have different resolutions."""
        jpl = GRACEClient(dataset="jpl_mascon_rl06")
        csr = GRACEClient(dataset="csr_mascon_rl06")
        
        assert jpl.dataset.resolution_deg == 0.5
        assert csr.dataset.resolution_deg == 0.25
    
    @pytest.mark.asyncio
    async def test_tellus_ocean_product(self):
        """TELLUS ocean should have OBP variable."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        client = GRACEClient(dataset="tellus_ocean")
        
        ds = await client.download(
            lat_range=(30, 45),
            lon_range=(-10, 10),
            time_range=("2020-01", "2020-06"),
        )
        
        assert "obp" in ds.data_vars


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

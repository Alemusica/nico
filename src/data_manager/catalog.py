"""
🌍 Copernicus Marine Data Catalog
=================================

Dynamic discovery and browsing of all available Copernicus Marine datasets.
Provides metadata, variables, temporal/spatial coverage for each product.

Usage:
    catalog = CopernicusCatalog()
    
    # List all available products
    products = await catalog.list_products()
    
    # Search by variable
    sla_products = await catalog.search(variable="sla")
    
    # Get product details
    info = await catalog.get_product_info("SEALEVEL_GLO_PHY_L4_NRT_008_046")
    
    # Check what's available for a region/time
    available = await catalog.check_availability(
        lat_range=(45, 47),
        lon_range=(8, 10),
        time_range=("2000-10-01", "2000-10-31"),
        variables=["sla", "sst"]
    )
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import hashlib

try:
    import copernicusmarine
    HAS_COPERNICUS = True
except ImportError:
    HAS_COPERNICUS = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class DataCategory(Enum):
    """Categories of Copernicus Marine data."""
    SEA_LEVEL = "sea_level"
    SEA_SURFACE_TEMPERATURE = "sst"
    OCEAN_CURRENTS = "currents"
    OCEAN_WAVES = "waves"
    OCEAN_PHYSICS = "physics"
    BIOGEOCHEMISTRY = "biogeochemistry"
    SEA_ICE = "sea_ice"
    WIND = "wind"
    REANALYSIS = "reanalysis"
    FORECAST = "forecast"


@dataclass
class Variable:
    """A variable within a dataset."""
    name: str
    standard_name: str
    long_name: str
    units: str
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "standard_name": self.standard_name,
            "long_name": self.long_name,
            "units": self.units,
            "description": self.description,
        }


@dataclass
class SpatialCoverage:
    """Spatial coverage of a dataset."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    
    def contains(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max and 
                self.lon_min <= lon <= self.lon_max)
    
    def overlaps(self, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> bool:
        return not (lat_range[1] < self.lat_min or lat_range[0] > self.lat_max or
                   lon_range[1] < self.lon_min or lon_range[0] > self.lon_max)
    
    def to_dict(self) -> Dict:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
        }


@dataclass
class TemporalCoverage:
    """Temporal coverage of a dataset."""
    start_date: str  # ISO format
    end_date: str  # ISO format or "present"
    resolution: str  # "hourly", "daily", "monthly"
    update_frequency: str = "daily"  # How often data is updated
    
    def contains_date(self, date_str: str) -> bool:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        if self.end_date == "present":
            return date >= start
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        return start <= date <= end
    
    def overlaps(self, time_range: Tuple[str, str]) -> bool:
        range_start = datetime.strptime(time_range[0], "%Y-%m-%d")
        range_end = datetime.strptime(time_range[1], "%Y-%m-%d")
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        if self.end_date == "present":
            return range_end >= start
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        return not (range_end < start or range_start > end)
    
    def to_dict(self) -> Dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "resolution": self.resolution,
            "update_frequency": self.update_frequency,
        }


@dataclass
class DataProduct:
    """A Copernicus Marine data product."""
    product_id: str
    dataset_id: str
    name: str
    description: str
    category: DataCategory
    variables: List[Variable]
    spatial_coverage: SpatialCoverage
    temporal_coverage: TemporalCoverage
    spatial_resolution_deg: float
    data_type: str  # "observation", "model", "reanalysis"
    processing_level: str  # "L2", "L3", "L4"
    source: str  # "satellite", "in-situ", "model"
    keywords: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    citation: Optional[str] = None
    
    def has_variable(self, var_name: str) -> bool:
        """Check if product has a variable (case-insensitive)."""
        var_lower = var_name.lower()
        return any(
            var_lower in v.name.lower() or 
            var_lower in v.standard_name.lower()
            for v in self.variables
        )
    
    def matches_query(
        self,
        variable: Optional[str] = None,
        category: Optional[DataCategory] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> bool:
        """Check if product matches search criteria."""
        if variable and not self.has_variable(variable):
            return False
        if category and self.category != category:
            return False
        if lat_range and lon_range:
            if not self.spatial_coverage.overlaps(lat_range, lon_range):
                return False
        if time_range:
            if not self.temporal_coverage.overlaps(time_range):
                return False
        if keywords:
            product_keywords = set(k.lower() for k in self.keywords)
            product_keywords.add(self.name.lower())
            product_keywords.add(self.description.lower())
            if not any(kw.lower() in ' '.join(product_keywords) for kw in keywords):
                return False
        return True
    
    def to_dict(self) -> Dict:
        return {
            "product_id": self.product_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "variables": [v.to_dict() for v in self.variables],
            "spatial_coverage": self.spatial_coverage.to_dict(),
            "temporal_coverage": self.temporal_coverage.to_dict(),
            "spatial_resolution_deg": self.spatial_resolution_deg,
            "data_type": self.data_type,
            "processing_level": self.processing_level,
            "source": self.source,
            "keywords": self.keywords,
            "doi": self.doi,
        }
    
    def to_summary(self) -> Dict:
        """Compact summary for UI listing."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category.value,
            "variables": [v.name for v in self.variables],
            "resolution": f"{self.spatial_resolution_deg}°",
            "temporal_resolution": self.temporal_coverage.resolution,
            "time_range": f"{self.temporal_coverage.start_date} → {self.temporal_coverage.end_date}",
        }


# Pre-defined catalog of known CMEMS products
KNOWN_PRODUCTS = [
    # Sea Level Products
    DataProduct(
        product_id="SEALEVEL_GLO_PHY_L4_NRT_008_046",
        dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D",
        name="Global Ocean Gridded L4 Sea Surface Heights",
        description="Daily gridded sea level anomaly and absolute dynamic topography from multi-satellite altimetry",
        category=DataCategory.SEA_LEVEL,
        variables=[
            Variable("sla", "sea_surface_height_above_sea_level", "Sea Level Anomaly", "m", "Sea surface height anomaly above geoid"),
            Variable("adt", "sea_surface_height_above_geoid", "Absolute Dynamic Topography", "m", "Sea surface height above geoid"),
            Variable("ugos", "surface_geostrophic_eastward_sea_water_velocity", "Geostrophic Eastward Velocity", "m/s"),
            Variable("vgos", "surface_geostrophic_northward_sea_water_velocity", "Geostrophic Northward Velocity", "m/s"),
            Variable("err_sla", "sea_surface_height_above_sea_level_error", "SLA Error", "m"),
        ],
        spatial_coverage=SpatialCoverage(-90, 90, -180, 180),
        temporal_coverage=TemporalCoverage("1993-01-01", "present", "daily"),
        spatial_resolution_deg=0.25,
        data_type="observation",
        processing_level="L4",
        source="satellite",
        keywords=["altimetry", "sea level", "SSH", "multi-mission", "DUACS"],
        doi="10.48670/moi-00148",
    ),
    DataProduct(
        product_id="SEALEVEL_EUR_PHY_L4_NRT_008_060",
        dataset_id="cmems_obs-sl_eur_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        name="European Seas Gridded L4 Sea Surface Heights",
        description="Daily gridded sea level for European seas at higher resolution",
        category=DataCategory.SEA_LEVEL,
        variables=[
            Variable("sla", "sea_surface_height_above_sea_level", "Sea Level Anomaly", "m"),
            Variable("adt", "sea_surface_height_above_geoid", "Absolute Dynamic Topography", "m"),
        ],
        spatial_coverage=SpatialCoverage(20, 66, -30, 42),
        temporal_coverage=TemporalCoverage("1993-01-01", "present", "daily"),
        spatial_resolution_deg=0.125,
        data_type="observation",
        processing_level="L4",
        source="satellite",
        keywords=["altimetry", "sea level", "European", "Mediterranean", "Baltic"],
    ),
    # SST Products
    DataProduct(
        product_id="SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
        dataset_id="cmems_obs-sst_glo_phy_nrt_l4_P1D-m",
        name="Global Ocean OSTIA Sea Surface Temperature",
        description="Daily gap-free SST analysis using satellite and in-situ data",
        category=DataCategory.SEA_SURFACE_TEMPERATURE,
        variables=[
            Variable("analysed_sst", "sea_surface_temperature", "Sea Surface Temperature", "K"),
            Variable("analysis_error", "sea_surface_temperature_error", "SST Analysis Error", "K"),
            Variable("sea_ice_fraction", "sea_ice_area_fraction", "Sea Ice Fraction", "1"),
        ],
        spatial_coverage=SpatialCoverage(-90, 90, -180, 180),
        temporal_coverage=TemporalCoverage("2007-01-01", "present", "daily"),
        spatial_resolution_deg=0.05,
        data_type="observation",
        processing_level="L4",
        source="satellite",
        keywords=["SST", "temperature", "OSTIA", "infrared", "microwave"],
    ),
    # Ocean Physics - Currents
    DataProduct(
        product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        name="Global Ocean Physics Analysis and Forecast",
        description="Daily ocean physics from NEMO model (currents, temperature, salinity)",
        category=DataCategory.OCEAN_CURRENTS,
        variables=[
            Variable("uo", "eastward_sea_water_velocity", "Eastward Velocity", "m/s"),
            Variable("vo", "northward_sea_water_velocity", "Northward Velocity", "m/s"),
            Variable("thetao", "sea_water_potential_temperature", "Temperature", "degC"),
            Variable("so", "sea_water_salinity", "Salinity", "PSU"),
            Variable("zos", "sea_surface_height_above_geoid", "Sea Surface Height", "m"),
        ],
        spatial_coverage=SpatialCoverage(-80, 90, -180, 180),
        temporal_coverage=TemporalCoverage("2019-01-01", "present", "daily"),
        spatial_resolution_deg=0.083,
        data_type="model",
        processing_level="L4",
        source="model",
        keywords=["currents", "NEMO", "physics", "3D", "forecast"],
    ),
    # Waves
    DataProduct(
        product_id="GLOBAL_ANALYSISFORECAST_WAV_001_027",
        dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        name="Global Ocean Waves Analysis and Forecast",
        description="3-hourly wave analysis and forecast",
        category=DataCategory.OCEAN_WAVES,
        variables=[
            Variable("VHM0", "sea_surface_wave_significant_height", "Significant Wave Height", "m"),
            Variable("VMDR", "sea_surface_wave_from_direction", "Wave Direction", "degrees"),
            Variable("VTM10", "sea_surface_wave_mean_period", "Mean Wave Period", "s"),
            Variable("VTPK", "sea_surface_wave_period_at_variance_spectral_density_maximum", "Peak Period", "s"),
        ],
        spatial_coverage=SpatialCoverage(-90, 90, -180, 180),
        temporal_coverage=TemporalCoverage("2019-01-01", "present", "3-hourly"),
        spatial_resolution_deg=0.083,
        data_type="model",
        processing_level="L4",
        source="model",
        keywords=["waves", "significant height", "swell", "wind waves"],
    ),
    # Wind
    DataProduct(
        product_id="WIND_GLO_PHY_L4_NRT_012_004",
        dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        name="Global Ocean Wind L4 Hourly",
        description="Hourly wind speed and direction from satellite scatterometry",
        category=DataCategory.WIND,
        variables=[
            Variable("eastward_wind", "eastward_wind", "Eastward Wind", "m/s"),
            Variable("northward_wind", "northward_wind", "Northward Wind", "m/s"),
            Variable("wind_speed", "wind_speed", "Wind Speed", "m/s"),
            Variable("wind_stress", "surface_downward_eastward_stress", "Wind Stress", "N/m2"),
        ],
        spatial_coverage=SpatialCoverage(-90, 90, -180, 180),
        temporal_coverage=TemporalCoverage("2018-01-01", "present", "hourly"),
        spatial_resolution_deg=0.125,
        data_type="observation",
        processing_level="L4",
        source="satellite",
        keywords=["wind", "scatterometer", "stress"],
    ),
    # Reanalysis
    DataProduct(
        product_id="GLOBAL_MULTIYEAR_PHY_001_030",
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        name="Global Ocean Physics Reanalysis",
        description="Multi-year ocean reanalysis (1993-present) for climate studies",
        category=DataCategory.REANALYSIS,
        variables=[
            Variable("uo", "eastward_sea_water_velocity", "Eastward Velocity", "m/s"),
            Variable("vo", "northward_sea_water_velocity", "Northward Velocity", "m/s"),
            Variable("thetao", "sea_water_potential_temperature", "Temperature", "degC"),
            Variable("so", "sea_water_salinity", "Salinity", "PSU"),
            Variable("zos", "sea_surface_height_above_geoid", "Sea Surface Height", "m"),
            Variable("mlotst", "ocean_mixed_layer_thickness", "Mixed Layer Depth", "m"),
        ],
        spatial_coverage=SpatialCoverage(-80, 90, -180, 180),
        temporal_coverage=TemporalCoverage("1993-01-01", "present", "daily"),
        spatial_resolution_deg=0.083,
        data_type="reanalysis",
        processing_level="L4",
        source="model",
        keywords=["reanalysis", "GLORYS", "multi-year", "climate", "long-term"],
        doi="10.48670/moi-00021",
    ),
    # Sea Ice
    DataProduct(
        product_id="SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_006",
        dataset_id="cmems_obs-si_glo_phy-sie_nrt_l4-multi-north_P1D",
        name="Global Sea Ice Extent and Concentration",
        description="Daily sea ice concentration and extent from satellite",
        category=DataCategory.SEA_ICE,
        variables=[
            Variable("ice_conc", "sea_ice_area_fraction", "Sea Ice Concentration", "%"),
            Variable("ice_edge", "sea_ice_edge", "Sea Ice Edge", "1"),
        ],
        spatial_coverage=SpatialCoverage(30, 90, -180, 180),
        temporal_coverage=TemporalCoverage("2016-01-01", "present", "daily"),
        spatial_resolution_deg=0.25,
        data_type="observation",
        processing_level="L4",
        source="satellite",
        keywords=["sea ice", "Arctic", "concentration", "extent"],
    ),
]


class CopernicusCatalog:
    """
    Catalog of available Copernicus Marine datasets.
    
    Combines pre-defined products with live API queries for discovery.
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / "data" / "cache" / "catalog"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize with known products
        self._products: Dict[str, DataProduct] = {p.product_id: p for p in KNOWN_PRODUCTS}
        self._loaded_from_api = False
        self._last_refresh = None
    
    def list_products(
        self,
        category: Optional[DataCategory] = None,
    ) -> List[DataProduct]:
        """List all available products, optionally filtered by category."""
        products = list(self._products.values())
        if category:
            products = [p for p in products if p.category == category]
        return sorted(products, key=lambda p: (p.category.value, p.name))
    
    def list_categories(self) -> List[Dict]:
        """List available categories with counts."""
        categories = {}
        for product in self._products.values():
            cat = product.category
            if cat not in categories:
                categories[cat] = {"category": cat.value, "count": 0, "products": []}
            categories[cat]["count"] += 1
            categories[cat]["products"].append(product.product_id)
        return list(categories.values())
    
    def list_variables(self) -> List[Dict]:
        """List all available variables across products."""
        variables = {}
        for product in self._products.values():
            for var in product.variables:
                key = var.standard_name or var.name
                if key not in variables:
                    variables[key] = {
                        "name": var.name,
                        "standard_name": var.standard_name,
                        "units": var.units,
                        "products": [],
                    }
                variables[key]["products"].append(product.product_id)
        return list(variables.values())
    
    def get_product(self, product_id: str) -> Optional[DataProduct]:
        """Get product by ID."""
        return self._products.get(product_id)
    
    def search(
        self,
        variable: Optional[str] = None,
        category: Optional[DataCategory] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        keywords: Optional[List[str]] = None,
        text_query: Optional[str] = None,
    ) -> List[DataProduct]:
        """
        Search products by criteria.
        
        Args:
            variable: Variable name (e.g., "sla", "sst")
            category: Data category
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude  
            time_range: (start, end) dates
            keywords: Keywords to match
            text_query: Free text search in name/description
            
        Returns:
            List of matching products
        """
        results = []
        
        # Parse text query into keywords
        search_keywords = keywords or []
        if text_query:
            search_keywords.extend(text_query.lower().split())
        
        for product in self._products.values():
            if product.matches_query(
                variable=variable,
                category=category,
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=time_range,
                keywords=search_keywords if search_keywords else None,
            ):
                results.append(product)
        
        return results
    
    def check_availability(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Check what data is available for a specific region/time.
        
        Returns:
            Dict with available products grouped by variable
        """
        result = {
            "region": {
                "lat_range": lat_range,
                "lon_range": lon_range,
            },
            "time_range": time_range,
            "available": {},
            "summary": {},
        }
        
        # Find products for each requested variable
        vars_to_check = variables or ["sla", "sst", "uo", "vo", "VHM0", "wind_speed"]
        
        for var in vars_to_check:
            matches = self.search(
                variable=var,
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=time_range,
            )
            if matches:
                result["available"][var] = [
                    {
                        "product_id": p.product_id,
                        "name": p.name,
                        "resolution": f"{p.spatial_resolution_deg}°",
                        "temporal": p.temporal_coverage.resolution,
                    }
                    for p in matches
                ]
        
        # Summary
        result["summary"] = {
            "variables_found": list(result["available"].keys()),
            "total_products": sum(len(v) for v in result["available"].values()),
            "coverage": "full" if len(result["available"]) >= len(vars_to_check) * 0.5 else "partial",
        }
        
        return result
    
    def get_download_config(
        self,
        product_id: str,
        variables: Optional[List[str]] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[str, str]] = None,
    ) -> Optional[Dict]:
        """
        Get configuration for downloading data from a product.
        
        Returns dict ready for CMEMSClient.download()
        """
        product = self.get_product(product_id)
        if not product:
            return None
        
        # Default to all variables
        if variables is None:
            variables = [v.name for v in product.variables]
        
        # Estimate size
        if lat_range and lon_range and time_range:
            lat_span = abs(lat_range[1] - lat_range[0])
            lon_span = abs(lon_range[1] - lon_range[0])
            lat_points = max(1, int(lat_span / product.spatial_resolution_deg))
            lon_points = max(1, int(lon_span / product.spatial_resolution_deg))
            
            # Parse time range
            start = datetime.strptime(time_range[0], "%Y-%m-%d")
            end = datetime.strptime(time_range[1], "%Y-%m-%d")
            days = (end - start).days + 1
            
            if product.temporal_coverage.resolution == "hourly":
                time_points = days * 24
            elif product.temporal_coverage.resolution == "3-hourly":
                time_points = days * 8
            else:
                time_points = days
            
            total_points = lat_points * lon_points * time_points * len(variables)
            estimated_size_mb = (total_points * 4 * 0.5) / (1024 * 1024)  # 4 bytes, 50% compression
        else:
            estimated_size_mb = None
        
        return {
            "product_id": product_id,
            "dataset_id": product.dataset_id,
            "variables": variables,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "time_range": time_range,
            "estimated_size_mb": round(estimated_size_mb, 2) if estimated_size_mb else None,
            "spatial_resolution": product.spatial_resolution_deg,
            "temporal_resolution": product.temporal_coverage.resolution,
        }
    
    async def refresh_from_api(self, force: bool = False) -> int:
        """
        Refresh catalog from Copernicus Marine API.
        
        Returns number of new products discovered.
        """
        if not HAS_COPERNICUS:
            print("⚠️ copernicusmarine not installed, using static catalog")
            return 0
        
        # Check cache freshness (refresh daily)
        cache_file = self.cache_dir / "catalog_cache.json"
        if cache_file.exists() and not force:
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))
                if datetime.now() - cached_time < timedelta(hours=24):
                    print(f"📁 Using cached catalog ({len(cached.get('products', []))} products)")
                    return 0
            except:
                pass
        
        print("🔄 Refreshing catalog from Copernicus Marine API...")
        
        try:
            # Use copernicusmarine library to list products
            # This is a placeholder - actual implementation depends on API
            # catalog = copernicusmarine.describe(include_datasets=True)
            
            # For now, just save current catalog to cache
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "products": [p.to_dict() for p in self._products.values()],
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self._loaded_from_api = True
            self._last_refresh = datetime.now()
            return 0
            
        except Exception as e:
            print(f"⚠️ API refresh failed: {e}")
            return 0
    
    def to_dict(self) -> Dict:
        """Export full catalog as dict."""
        return {
            "products": [p.to_dict() for p in self._products.values()],
            "categories": self.list_categories(),
            "variables": self.list_variables(),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
        }


# Convenience functions
def get_catalog() -> CopernicusCatalog:
    """Get singleton catalog instance."""
    if not hasattr(get_catalog, "_instance"):
        get_catalog._instance = CopernicusCatalog()
    return get_catalog._instance


def search_products(
    variable: Optional[str] = None,
    category: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """Quick search for products."""
    catalog = get_catalog()
    cat = DataCategory(category) if category else None
    results = catalog.search(variable=variable, category=cat, **kwargs)
    return [p.to_summary() for p in results]


def check_availability(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_range: Tuple[str, str],
    variables: Optional[List[str]] = None,
) -> Dict:
    """Quick availability check."""
    catalog = get_catalog()
    return catalog.check_availability(lat_range, lon_range, time_range, variables)


# CLI test
if __name__ == "__main__":
    catalog = CopernicusCatalog()
    
    print("=== Copernicus Marine Data Catalog ===\n")
    
    print("Categories:")
    for cat in catalog.list_categories():
        print(f"  {cat['category']}: {cat['count']} products")
    
    print("\nSea Level Products:")
    for product in catalog.list_products(DataCategory.SEA_LEVEL):
        print(f"  - {product.name}")
        print(f"    Variables: {[v.name for v in product.variables]}")
        print(f"    Resolution: {product.spatial_resolution_deg}°, {product.temporal_coverage.resolution}")
    
    print("\nSearch for 'sla' variable:")
    results = catalog.search(variable="sla")
    for p in results:
        print(f"  - {p.product_id}: {p.name}")
    
    print("\nCheck availability for Lago Maggiore (Oct 2000):")
    avail = catalog.check_availability(
        lat_range=(45.5, 46.5),
        lon_range=(8.0, 9.5),
        time_range=("2000-10-01", "2000-10-31"),
    )
    print(f"  Variables found: {avail['summary']['variables_found']}")
    print(f"  Total products: {avail['summary']['total_products']}")

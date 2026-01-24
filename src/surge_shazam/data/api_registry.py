"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    API REGISTRY - Central Data Source Catalog                 ║
║                                                                              ║
║   Single source of truth for all data APIs used in the Early Warning System. ║
║   Each source has: endpoint, auth, variables, latency, status.               ║
║                                                                              ║
║   Categories:                                                                ║
║   - SATELLITE: CMEMS, ERA5, CYGNSS, GPM, GRACE, Sentinel, SLCCI              ║
║   - AIRCRAFT: AMDAR, Mode-S, OpenSky                                         ║
║   - IN-SITU: Tide Gauges, ARGO Floats, Weather Stations                      ║
║   - INDICES: Climate teleconnections (NAO, ENSO, AMO...)                     ║
║   - KNOWLEDGE: Papers, News archives, Historical events                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
from datetime import datetime


class DataCategory(Enum):
    """Data source categories."""
    SATELLITE = "satellite"
    AIRCRAFT = "aircraft"
    IN_SITU = "in_situ"
    INDICES = "indices"
    KNOWLEDGE = "knowledge"
    REANALYSIS = "reanalysis"


class Latency(Enum):
    """Data latency classification."""
    REALTIME = "realtime"      # < 1 hour
    NEAR_RT = "near_realtime"  # 1-6 hours
    DELAYED = "delayed"        # 6h - 24h
    ARCHIVE = "archive"        # > 24h
    HISTORICAL = "historical"  # No updates
    
    @property
    def badge(self) -> str:
        return {
            Latency.REALTIME: "🟢",
            Latency.NEAR_RT: "🟡",
            Latency.DELAYED: "🟠",
            Latency.ARCHIVE: "🔴",
            Latency.HISTORICAL: "⚫",
        }[self]
    
    @property
    def hours(self) -> int:
        """Typical latency in hours."""
        return {
            Latency.REALTIME: 1,
            Latency.NEAR_RT: 6,
            Latency.DELAYED: 24,
            Latency.ARCHIVE: 168,  # 7 days
            Latency.HISTORICAL: 9999,
        }[self]


class Status(Enum):
    """Implementation status."""
    AVAILABLE = "available"       # Fully implemented
    PARTIAL = "partial"          # Basic implementation
    STUB = "stub"                # Interface only
    TODO = "todo"                # Not started
    COMING = "coming"            # Future dataset


@dataclass
class AuthConfig:
    """Authentication configuration."""
    required: bool = False
    env_vars: List[str] = field(default_factory=list)
    url_signup: str = ""
    method: str = "env"  # env, oauth2, api_key, basic


@dataclass 
class DataSource:
    """Single data source definition."""
    id: str
    name: str
    category: DataCategory
    provider: str
    
    # API details
    base_url: str
    product_id: str = ""
    
    # Variables
    variables: List[str] = field(default_factory=list)
    
    # Resolution
    spatial_resolution: str = ""  # e.g., "0.25°", "25km"
    temporal_resolution: str = ""  # e.g., "hourly", "daily"
    
    # Coverage
    coverage_spatial: str = "global"  # or bbox
    coverage_temporal: tuple = ("1990-01-01", "present")
    
    # Latency
    latency: Latency = Latency.ARCHIVE
    
    # Auth
    auth: AuthConfig = field(default_factory=AuthConfig)
    
    # Implementation
    status: Status = Status.TODO
    client_module: str = ""  # e.g., "src.surge_shazam.data.cmems_client"
    
    # Metadata
    description: str = ""
    docs_url: str = ""
    priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    
    # Physics relevance
    physics_variables: List[str] = field(default_factory=list)  # Which SWE variables it provides
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "provider": self.provider,
            "latency": f"{self.latency.badge} {self.latency.value}",
            "status": self.status.value,
            "variables": self.variables,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SATELLITE DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════

CMEMS_SEALEVEL = DataSource(
    id="cmems_sealevel",
    name="CMEMS Sea Level L4",
    category=DataCategory.SATELLITE,
    provider="Copernicus Marine",
    base_url="https://data.marine.copernicus.eu/",
    product_id="SEALEVEL_GLO_PHY_L4_NRT_008_046",
    variables=["sla", "adt", "ugos", "vgos", "err_sla"],
    spatial_resolution="0.25°",
    temporal_resolution="daily",
    coverage_spatial="global",
    coverage_temporal=("1993-01-01", "present"),
    latency=Latency.DELAYED,
    auth=AuthConfig(
        required=True,
        env_vars=["CMEMS_USERNAME", "CMEMS_PASSWORD"],
        url_signup="https://data.marine.copernicus.eu/register",
    ),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.cmems_client",
    description="Global gridded sea surface heights from multi-satellite altimetry",
    docs_url="https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046",
    priority="HIGH",
    physics_variables=["η", "u_geo", "v_geo"],  # Sea level, geostrophic velocity
)

CMEMS_SST = DataSource(
    id="cmems_sst",
    name="CMEMS SST OSTIA",
    category=DataCategory.SATELLITE,
    provider="Copernicus Marine",
    base_url="https://data.marine.copernicus.eu/",
    product_id="SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
    variables=["analysed_sst", "analysis_error", "sea_ice_fraction"],
    spatial_resolution="0.05°",
    temporal_resolution="daily",
    latency=Latency.DELAYED,
    auth=AuthConfig(required=True, env_vars=["CMEMS_USERNAME", "CMEMS_PASSWORD"]),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.cmems_client",
    priority="HIGH",
    physics_variables=["T_surface"],  # For steric height
)

CMEMS_WAVES = DataSource(
    id="cmems_waves",
    name="CMEMS Ocean Waves",
    category=DataCategory.SATELLITE,
    provider="Copernicus Marine",
    base_url="https://data.marine.copernicus.eu/",
    product_id="GLOBAL_ANALYSISFORECAST_WAV_001_027",
    variables=["VHM0", "VMDR", "VTM10", "VTPK"],
    spatial_resolution="0.083°",
    temporal_resolution="3-hourly",
    latency=Latency.DELAYED,
    auth=AuthConfig(required=True, env_vars=["CMEMS_USERNAME", "CMEMS_PASSWORD"]),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.cmems_client",
    priority="MEDIUM",
    physics_variables=["H_wave"],  # Wave height for setup
)

ERA5_SURFACE = DataSource(
    id="era5_surface",
    name="ERA5 Surface Variables",
    category=DataCategory.REANALYSIS,
    provider="ECMWF",
    base_url="https://cds.climate.copernicus.eu/",
    product_id="reanalysis-era5-single-levels",
    variables=[
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "2m_temperature",
        "total_precipitation", "surface_pressure",
        "evaporation", "runoff", "soil_moisture"
    ],
    spatial_resolution="0.25°",
    temporal_resolution="hourly",
    coverage_temporal=("1940-01-01", "present"),
    latency=Latency.ARCHIVE,  # ~5 days
    auth=AuthConfig(
        required=True,
        env_vars=["CDS_API_KEY"],
        url_signup="https://cds.climate.copernicus.eu/user/register",
    ),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.era5_client",
    description="ERA5 hourly reanalysis - atmosphere surface variables",
    priority="HIGH",
    physics_variables=["τ_wind", "P_atm", "precip"],  # Wind stress, pressure, precipitation
)

CYGNSS = DataSource(
    id="cygnss",
    name="CYGNSS Wind Speed",
    category=DataCategory.SATELLITE,
    provider="NASA",
    base_url="https://podaac.jpl.nasa.gov/",
    product_id="CYGNSS_L3_GLOBAL_DAILY_V3.1",
    variables=["wind_speed", "wind_speed_uncertainty"],
    spatial_resolution="0.2°",
    temporal_resolution="daily",
    coverage_spatial="±38° latitude",
    coverage_temporal=("2017-03-01", "present"),
    latency=Latency.NEAR_RT,  # 2-24h
    auth=AuthConfig(
        required=True,
        env_vars=["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        url_signup="https://urs.earthdata.nasa.gov/users/new",
    ),
    status=Status.PARTIAL,
    client_module="src.surge_shazam.data.cygnss_client",
    description="GNSS-R wind speed from 8 microsatellites - fast revisit",
    docs_url="https://podaac.jpl.nasa.gov/CYGNSS",
    priority="HIGH",
    physics_variables=["U_wind"],  # Wind for stress calculation
)

GPM_IMERG = DataSource(
    id="gpm_imerg",
    name="GPM IMERG Precipitation",
    category=DataCategory.SATELLITE,
    provider="NASA",
    base_url="https://gpm.nasa.gov/",
    product_id="GPM_3IMERGDE.07",
    variables=["precipitation", "precipitation_quality"],
    spatial_resolution="0.1°",
    temporal_resolution="30-min",
    coverage_spatial="60°N to 60°S",
    coverage_temporal=("2000-06-01", "present"),
    latency=Latency.NEAR_RT,  # Early run: 4h, Late run: 14h
    auth=AuthConfig(
        required=True,
        env_vars=["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
    ),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.gpm_client",
    description="Global precipitation measurement - near real-time",
    docs_url="https://gpm.nasa.gov/data/imerg",
    priority="HIGH",
    physics_variables=["precip"],  # Precipitation forcing
)

GRACE_FO = DataSource(
    id="grace_fo",
    name="GRACE-FO Mass Change",
    category=DataCategory.SATELLITE,
    provider="NASA/GFZ",
    base_url="https://podaac.jpl.nasa.gov/",
    product_id="GRACE-FO_MONTHLY_MASCON",
    variables=["lwe_thickness", "uncertainty"],
    spatial_resolution="0.5°",
    temporal_resolution="monthly",
    coverage_temporal=("2018-06-01", "present"),
    latency=Latency.ARCHIVE,  # ~2 months
    auth=AuthConfig(required=True, env_vars=["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"]),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.grace_client",
    description="Terrestrial water storage - groundwater, soil moisture",
    priority="MEDIUM",
    physics_variables=["TWS"],  # Total water storage
)

SENTINEL1_SAR = DataSource(
    id="sentinel1_sar",
    name="Sentinel-1 SAR",
    category=DataCategory.SATELLITE,
    provider="ESA/Copernicus",
    base_url="https://dataspace.copernicus.eu/",
    product_id="SENTINEL-1",
    variables=["sigma0_vv", "sigma0_vh", "coherence", "owi_wind_speed"],
    spatial_resolution="10m",
    temporal_resolution="6-12 days",
    latency=Latency.DELAYED,
    auth=AuthConfig(
        required=True,
        env_vars=["COPERNICUS_USERNAME", "COPERNICUS_PASSWORD"],
        url_signup="https://dataspace.copernicus.eu/",
    ),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.sentinel_client",
    description="SAR imagery - flood mapping, sea state, wind",
    priority="MEDIUM",
)

SENTINEL3_OLCI = DataSource(
    id="sentinel3_olci",
    name="Sentinel-3 OLCI",
    category=DataCategory.SATELLITE,
    provider="ESA/Copernicus",
    base_url="https://dataspace.copernicus.eu/",
    product_id="SENTINEL-3-OLCI",
    variables=["chlorophyll", "turbidity", "sst", "tsm", "cdom"],
    spatial_resolution="300m",
    temporal_resolution="daily",
    latency=Latency.DELAYED,
    auth=AuthConfig(required=True, env_vars=["COPERNICUS_USERNAME", "COPERNICUS_PASSWORD"]),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.sentinel_client",
    description="Ocean color - chlorophyll, sediments (river discharge proxy)",
    priority="LOW",
)

SLCCI_ALTIMETRY = DataSource(
    id="slcci_altimetry",
    name="ESA Sea Level CCI",
    category=DataCategory.SATELLITE,
    provider="ESA CCI",
    base_url="local",
    product_id="SLCCI_ALTDB",
    variables=["corssh", "mean_sea_surface", "swh"],
    spatial_resolution="along-track",
    coverage_temporal=("1993-01-01", "2020-12-31"),
    latency=Latency.HISTORICAL,
    status=Status.AVAILABLE,
    client_module="src.data.loaders",
    description="Climate-quality altimetry reprocessed (Jason-1, TOPEX)",
    priority="HIGH",
)


# ══════════════════════════════════════════════════════════════════════════════
# AIRCRAFT DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════

AMDAR = DataSource(
    id="amdar",
    name="AMDAR Aircraft Weather",
    category=DataCategory.AIRCRAFT,
    provider="NOAA/WMO",
    base_url="https://madis.ncep.noaa.gov/",
    product_id="MADIS-AMDAR",
    variables=["temperature", "wind_speed", "wind_direction", "humidity", "altitude"],
    spatial_resolution="point",
    temporal_resolution="minutes",
    coverage_spatial="global (flight routes)",
    latency=Latency.REALTIME,
    auth=AuthConfig(
        required=True,
        env_vars=["MADIS_USER", "MADIS_PASSWORD"],
        url_signup="https://madis.ncep.noaa.gov/madis_acars.shtml",
    ),
    status=Status.PARTIAL,
    client_module="src.surge_shazam.data.aircraft_client",
    description="Aircraft meteorological data - vertical profiles",
    docs_url="https://madis.ncep.noaa.gov/",
    priority="HIGH",
    physics_variables=["T_air", "U_wind", "q_humidity"],
)

MODE_S_EHS = DataSource(
    id="mode_s_ehs",
    name="Mode-S EHS Derived Weather",
    category=DataCategory.AIRCRAFT,
    provider="OpenSky Network",
    base_url="https://opensky-network.org/",
    product_id="Mode-S-EHS",
    variables=["temperature", "wind_speed", "wind_direction", "mach", "altitude"],
    spatial_resolution="point",
    temporal_resolution="seconds",
    coverage_spatial="Europe, parts of N. America",
    latency=Latency.REALTIME,
    auth=AuthConfig(
        required=False,  # Free tier available
        env_vars=["OPENSKY_USERNAME", "OPENSKY_PASSWORD"],
        url_signup="https://opensky-network.org/",
    ),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.aircraft_client",
    description="Derived meteorological data from ADS-B/Mode-S",
    docs_url="https://opensky-network.org/data/impala",
    priority="HIGH",
    physics_variables=["T_air", "U_wind"],
)


# ══════════════════════════════════════════════════════════════════════════════
# IN-SITU DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════

TIDE_GAUGES = DataSource(
    id="tide_gauges",
    name="IOC Sea Level Station Monitoring",
    category=DataCategory.IN_SITU,
    provider="IOC/UNESCO",
    base_url="https://www.ioc-sealevelmonitoring.org/",
    product_id="IOC-SLSMF",
    variables=["sea_level", "quality_flag"],
    spatial_resolution="point",
    temporal_resolution="minutes",
    latency=Latency.REALTIME,
    auth=AuthConfig(required=False),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.tide_gauge_client",
    description="Real-time tide gauge network - ground truth for SLA",
    docs_url="https://www.ioc-sealevelmonitoring.org/",
    priority="HIGH",
    physics_variables=["η_obs"],  # Observed sea level
)

ARGO_FLOATS = DataSource(
    id="argo_floats",
    name="Argo Float Network",
    category=DataCategory.IN_SITU,
    provider="Argo Program",
    base_url="https://argovis.colorado.edu/",
    product_id="Argovis-API",
    variables=["temperature", "salinity", "pressure"],
    spatial_resolution="point",
    temporal_resolution="10-day",
    latency=Latency.DELAYED,
    auth=AuthConfig(required=False),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.argo_client",
    description="Deep ocean T/S profiles - steric height calculation",
    docs_url="https://argo.ucsd.edu/",
    priority="MEDIUM",
    physics_variables=["T_ocean", "S_ocean"],
)

DMI_WEATHER = DataSource(
    id="dmi_weather",
    name="DMI Open Data",
    category=DataCategory.IN_SITU,
    provider="DMI",
    base_url="https://opendatadocs.dmi.govcloud.dk/",
    product_id="DMI-OBS",
    variables=["temperature", "wind", "pressure", "precipitation", "sea_level"],
    spatial_resolution="point",
    temporal_resolution="10-min",
    coverage_spatial="Denmark",
    latency=Latency.REALTIME,
    auth=AuthConfig(
        required=True,
        env_vars=["DMI_API_KEY"],
        url_signup="https://opendatadocs.dmi.govcloud.dk/",
    ),
    status=Status.STUB,
    client_module="src.surge_shazam.data.loaders.dmi_api",
    description="Danish Meteorological Institute - dense Denmark network",
    priority="HIGH",
)


# ══════════════════════════════════════════════════════════════════════════════
# CLIMATE INDICES
# ══════════════════════════════════════════════════════════════════════════════

NOAA_INDICES = DataSource(
    id="noaa_indices",
    name="NOAA Climate Indices",
    category=DataCategory.INDICES,
    provider="NOAA",
    base_url="https://www.cpc.ncep.noaa.gov/",
    variables=["NAO", "AO", "ENSO", "AMO", "PDO", "PNA", "EA", "SCAND"],
    temporal_resolution="monthly",
    coverage_temporal=("1950-01-01", "present"),
    latency=Latency.ARCHIVE,
    auth=AuthConfig(required=False),
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.climate_indices",
    description="Teleconnection indices - NAO affects European floods",
    priority="HIGH",
    physics_variables=["NAO", "ENSO"],  # Teleconnection forcing
)


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE SOURCES
# ══════════════════════════════════════════════════════════════════════════════

SEMANTIC_SCHOLAR = DataSource(
    id="semantic_scholar",
    name="Semantic Scholar API",
    category=DataCategory.KNOWLEDGE,
    provider="Allen AI",
    base_url="https://api.semanticscholar.org/",
    variables=["papers", "citations", "embeddings", "tldr"],
    latency=Latency.REALTIME,
    auth=AuthConfig(required=False),  # Free tier
    status=Status.AVAILABLE,
    client_module="src.surge_shazam.data.knowledge.semantic_scholar",
    description="Scientific papers - find flood/surge research",
    priority="MEDIUM",
)

NEWS_ARCHIVES = DataSource(
    id="news_archives",
    name="Historical News Archives",
    category=DataCategory.KNOWLEDGE,
    provider="Various",
    base_url="",
    variables=["articles", "dates", "locations", "entities"],
    latency=Latency.HISTORICAL,
    auth=AuthConfig(required=True),
    status=Status.TODO,
    client_module="src.surge_shazam.data.knowledge.news_scraper",
    description="Newspaper archives - historical flood testimonies",
    priority="MEDIUM",
)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

API_REGISTRY: Dict[str, DataSource] = {
    # Satellite
    "cmems_sealevel": CMEMS_SEALEVEL,
    "cmems_sst": CMEMS_SST,
    "cmems_waves": CMEMS_WAVES,
    "era5_surface": ERA5_SURFACE,
    "cygnss": CYGNSS,
    "gpm_imerg": GPM_IMERG,
    "grace_fo": GRACE_FO,
    "sentinel1_sar": SENTINEL1_SAR,
    "sentinel3_olci": SENTINEL3_OLCI,
    "slcci_altimetry": SLCCI_ALTIMETRY,
    
    # Aircraft
    "amdar": AMDAR,
    "mode_s_ehs": MODE_S_EHS,
    
    # In-situ
    "tide_gauges": TIDE_GAUGES,
    "argo_floats": ARGO_FLOATS,
    "dmi_weather": DMI_WEATHER,
    
    # Indices
    "noaa_indices": NOAA_INDICES,
    
    # Knowledge
    "semantic_scholar": SEMANTIC_SCHOLAR,
    "news_archives": NEWS_ARCHIVES,
}


def get_sources_by_category(category: DataCategory) -> Dict[str, DataSource]:
    """Get all sources in a category."""
    return {k: v for k, v in API_REGISTRY.items() if v.category == category}


def get_sources_by_status(status: Status) -> Dict[str, DataSource]:
    """Get all sources with given status."""
    return {k: v for k, v in API_REGISTRY.items() if v.status == status}


def get_sources_by_latency(max_latency: Latency) -> Dict[str, DataSource]:
    """Get sources with latency <= max_latency."""
    return {k: v for k, v in API_REGISTRY.items() if v.latency.hours <= max_latency.hours}


def get_high_priority_sources() -> Dict[str, DataSource]:
    """Get HIGH priority sources."""
    return {k: v for k, v in API_REGISTRY.items() if v.priority == "HIGH"}


def get_physics_variables_map() -> Dict[str, List[str]]:
    """Map physics variable to data sources."""
    result = {}
    for source_id, source in API_REGISTRY.items():
        for pvar in source.physics_variables:
            if pvar not in result:
                result[pvar] = []
            result[pvar].append(source_id)
    return result


def print_registry_summary():
    """Print human-readable registry summary."""
    print("=" * 70)
    print("  API REGISTRY SUMMARY")
    print("=" * 70)
    
    for cat in DataCategory:
        sources = get_sources_by_category(cat)
        if sources:
            print(f"\n📂 {cat.value.upper()}")
            for sid, s in sources.items():
                print(f"   {s.latency.badge} [{s.status.value:10}] {s.name}")
                print(f"      Variables: {', '.join(s.variables[:5])}...")
    
    print("\n" + "=" * 70)
    print("  STATUS SUMMARY")
    print("=" * 70)
    for status in Status:
        count = len(get_sources_by_status(status))
        print(f"   {status.value:12}: {count}")
    
    print("\n  Physics Variables:")
    for pvar, sources in get_physics_variables_map().items():
        print(f"   {pvar:12}: {', '.join(sources)}")


if __name__ == "__main__":
    print_registry_summary()

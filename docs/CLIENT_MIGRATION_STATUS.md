# Data Client Migration Status

**Document:** Tracking migration to unified `DataClient` interface
**Created:** 2026-01-19
**Last Updated:** 2026-01-19

---

## ✅ Migration Complete! (10/10)

All data clients have been migrated to the unified `DataClient` interface.

---

## XArray Clients (Gridded Data)

### 1. CMEMS Client ✅
- **File:** `src/surge_shazam/data/cmems_client.py`
- **Inherits:** `XArrayClientMixin`, `DataClient`
- **source_id:** `"cmems_sealevel"`
- **output_format:** `DataFormat.XARRAY`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`
- **Legacy:** `download_legacy()` for backward compatibility

### 2. ERA5 Client ✅
- **File:** `src/surge_shazam/data/era5_client.py`
- **Inherits:** `XArrayClientMixin`, `DataClient`
- **source_id:** `"era5_surface"`
- **output_format:** `DataFormat.XARRAY`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`

### 3. GPM Client ✅
- **File:** `src/surge_shazam/data/gpm_client.py`
- **Inherits:** `XArrayClientMixin`, `DataClient`
- **source_id:** `"gpm_imerg"`
- **output_format:** `DataFormat.XARRAY`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`

### 4. GRACE Client ✅
- **File:** `src/surge_shazam/data/grace_client.py`
- **Inherits:** `XArrayClientMixin`, `DataClient`
- **source_id:** `"grace_fo"`
- **output_format:** `DataFormat.XARRAY`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`

### 5. CYGNSS Client ✅
- **File:** `src/surge_shazam/data/cygnss_client.py`
- **Inherits:** `XArrayClientMixin`, `DataClient`
- **source_id:** `"cygnss"`
- **output_format:** `DataFormat.XARRAY`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`

### 6. Sentinel Client ✅
- **File:** `src/surge_shazam/data/sentinel_client.py`
- **Inherits:** `XArrayClientMixin`, `DataClient`
- **source_id:** `"sentinel1_sar"`
- **output_format:** `DataFormat.XARRAY`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`
- **Legacy:** `get_sar_wind()`, `get_ocean_color()` for backward compatibility

---

## Point Data Clients (Observations)

### 7. Tide Gauge Client ✅
- **File:** `src/surge_shazam/data/tide_gauge_client.py`
- **Inherits:** `PointDataClientMixin`, `DataClient`
- **source_id:** `"tide_gauges"`
- **output_format:** `DataFormat.POINTS`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`
- **Legacy:** `find_stations()`, `get_data()`, `get_nearest_data()` for backward compatibility

### 8. ARGO Client ✅
- **File:** `src/surge_shazam/data/argo_client.py`
- **Inherits:** `PointDataClientMixin`, `DataClient`
- **source_id:** `"argo_floats"`
- **output_format:** `DataFormat.POINTS`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`
- **Legacy:** `get_profiles()`, `compute_steric_height()` for backward compatibility

### 9. Aircraft Client ✅
- **File:** `src/surge_shazam/data/aircraft_client.py`
- **Inherits:** `PointDataClientMixin`, `DataClient`
- **source_id:** `"mode_s_ehs"`
- **output_format:** `DataFormat.POINTS`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic_data()`
- **Legacy:** `get_current_observations()`, `get_observations()` for backward compatibility

---

## DataFrame Client (Time Series)

### 10. Climate Indices Client ✅
- **File:** `src/surge_shazam/data/climate_indices.py`
- **Inherits:** `DataFrameClientMixin`, `DataClient`
- **source_id:** `"noaa_indices"`
- **output_format:** `DataFormat.DATAFRAME`
- **Methods:** `download()`, `list_products()`, `health_check()`, `generate_synthetic()`
- **Legacy:** `get_index()`, `get_all_indices()`, `get_indices_for_flood()` for backward compatibility

---

## Interface Summary

All clients now implement the standard `DataClient` interface:

```python
# Required properties
source_id: str           # Unique identifier
output_format: DataFormat  # XARRAY, DATAFRAME, or POINTS

# Required methods
async def download(variables, bbox, time_range, **kwargs) -> Union[xr.Dataset, pd.DataFrame]
def list_products() -> Dict[str, str]
async def health_check() -> HealthCheckResult

# Optional methods (with default implementations)
async def generate_synthetic(variables, bbox, time_range, **kwargs) -> Union[xr.Dataset, pd.DataFrame]
def get_cache_key(variables, bbox, time_range, **kwargs) -> str
```

### Standard Types

```python
BoundingBox(lon_min, lat_min, lon_max, lat_max)  # Geographic bounds
TimeRange(start, end)                             # Time range
HealthCheckResult(status, message, latency_ms)   # Health status
DataClientError(source_id, operation, message)   # Unified exception
```

### Mixins Available

| Mixin | Output Format | Use Case |
|-------|---------------|----------|
| `XArrayClientMixin` | `DataFormat.XARRAY` | Gridded spatial-temporal data |
| `DataFrameClientMixin` | `DataFormat.DATAFRAME` | Time series / tabular data |
| `PointDataClientMixin` | `DataFormat.POINTS` | Point observations |

---

## Next Steps (Post-Migration)

New APIs to add using the same interface:

| API | Provider | Type | Priority |
|-----|----------|------|----------|
| Open-Meteo Flood | Open-Meteo | XArray | HIGH |
| GloFAS | Copernicus | XArray | HIGH |
| NOAA NDBC Buoys | NOAA | Points | HIGH |
| PSMSL Historical | PSMSL | Points | MEDIUM |
| AIS Ship Obs | Various | Points | MEDIUM |
| EFAS | Copernicus | XArray | MEDIUM |

---

*Migration completed: 2026-01-19*

# 🌊 Gates Catalog Documentation

> **Version**: 1.0  
> **Last Updated**: 2025-12-29  
> **Status**: 📋 Draft

---

## 📋 Overview

This document describes the ocean gates catalog used in the NICO project for analyzing Arctic Ocean exchanges.

---

## 🗺️ Available Gates

### Atlantic Sector

| Gate ID | Name | Description | Shapefile |
|---------|------|-------------|-----------|
| `fram_strait` | 🧊 Fram Strait | Main Arctic-Atlantic exchange | `fram_strait_S3_pass_481.shp` |
| `denmark_strait` | 🌀 Denmark Strait | Iceland-Greenland overflow | `denmark_strait_TPJ_pass_246.shp` |
| `davis_strait` | ❄️ Davis Strait | Baffin Bay - Labrador Sea | `davis_strait.shp` |
| `barents_opening` | 🌡️ Barents Opening | Atlantic water inflow | `barents_sea_opening_S3_pass_481.shp` |
| `norwegian_boundary` | 🇳🇴 Norwegian Sea Boundary | Atlantic-Nordic Seas | `norwegian_sea_boundary_TPJ_pass_220.shp` |

### Pacific Sector

| Gate ID | Name | Description | Shapefile |
|---------|------|-------------|-----------|
| `bering_strait` | 🌊 Bering Strait | Pacific-Arctic gateway | `bering_strait_TPJ_pass_076.shp` |

### Canadian Archipelago

| Gate ID | Name | Description | Shapefile |
|---------|------|-------------|-----------|
| `nares_strait` | 🏔️ Nares Strait | Greenland-Ellesmere Island | `nares_strait.shp` |
| `lancaster_sound` | 🚢 Lancaster Sound | Northwest Passage entrance | `lancaster_sound.shp` |

---

## 📍 Gate Details

### Fram Strait
- **Location**: Between Svalbard and Greenland
- **Latitude**: ~78°N - 80°N
- **Longitude**: ~20°W - 10°E
- **Significance**: Primary gateway for Arctic-Atlantic water exchange
- **Satellite Pass**: S3 Pass 481
- **Closest Passes**: [481, 254, 127, 308, 55]

### Bering Strait
- **Location**: Between Alaska and Russia
- **Latitude**: ~65°N - 66°N
- **Longitude**: ~168°W - 170°W
- **Significance**: Only Pacific-Arctic connection
- **Satellite Pass**: TPJ Pass 076
- **Closest Passes**: [76, 152, 228, 304, 380]

### Denmark Strait
- **Location**: Between Iceland and Greenland
- **Latitude**: ~66°N - 68°N
- **Longitude**: ~28°W - 24°W
- **Significance**: Deep overflow water path
- **Satellite Pass**: TPJ Pass 246
- **Closest Passes**: [246, 172, 320, 98, 394]

### Davis Strait
- **Location**: Between Baffin Island and Greenland
- **Latitude**: ~66°N - 68°N
- **Longitude**: ~58°W - 52°W
- **Significance**: Baffin Bay - Labrador Sea exchange

### Nares Strait
- **Location**: Between Greenland and Ellesmere Island
- **Latitude**: ~78°N - 82°N
- **Longitude**: ~70°W - 60°W
- **Significance**: High Arctic water exchange

### Lancaster Sound
- **Location**: Northern Canadian Archipelago
- **Latitude**: ~74°N
- **Longitude**: ~84°W - 80°W
- **Significance**: Eastern entrance to Northwest Passage

### Barents Opening
- **Location**: Between Norway and Svalbard
- **Latitude**: ~70°N - 76°N
- **Longitude**: ~15°E - 25°E
- **Significance**: Atlantic water inflow to Barents Sea

### Norwegian Sea Boundary
- **Location**: Norway to Iceland ridge
- **Latitude**: ~62°N - 66°N
- **Longitude**: ~10°W - 5°E
- **Significance**: Atlantic-Nordic Seas boundary

---

## 📁 Shapefile Structure

All gate shapefiles are stored in `gates/` directory with ESRI Shapefile format:
- `.shp` - Geometry
- `.shx` - Shape index
- `.dbf` - Attribute table
- `.prj` - Projection (EPSG:4326 - WGS84)

### Coordinate Reference System
- **EPSG**: 4326
- **Datum**: WGS84
- **Units**: Degrees

---

## 🔧 Usage

### Python (GateCatalog)
```python
from src.gates.catalog import GateCatalog

catalog = GateCatalog()

# List all gates
gates = catalog.list_all()

# Get specific gate
fram = catalog.get("fram_strait")
print(fram.name)  # "🧊 Fram Strait"
print(fram.closest_passes)  # [481, 254, 127, 308, 55]
```

### API Endpoints
```bash
# List all gates
GET /api/v1/gates/

# Get gate details
GET /api/v1/gates/fram_strait

# Get bounding box with buffer
GET /api/v1/gates/fram_strait/bbox?buffer_km=50

# Get closest passes
GET /api/v1/gates/fram_strait/passes
```

### Streamlit
```python
from src.services.gate_service import GateService

service = GateService()
bbox = service.get_bbox("fram_strait", buffer_km=50)
print(f"Lat: {bbox.lat_range}, Lon: {bbox.lon_range}")
```

---

## 🛰️ Satellite Pass Reference

### Pass Numbering
- **Jason/TOPEX**: 254 passes per 10-day cycle
- **Sentinel-3**: 385 passes per 27-day cycle

### Pre-computed Closest Passes
For each gate, we pre-compute the 5 closest satellite ground tracks:

```yaml
fram_strait:
  closest_passes: [481, 254, 127, 308, 55]
  
bering_strait:
  closest_passes: [76, 152, 228, 304, 380]
  
denmark_strait:
  closest_passes: [246, 172, 320, 98, 394]
```

---

## 📊 Data Flow

```
Gate Selection
      │
      ▼
┌─────────────────┐
│ Load Shapefile  │
│ (geopandas)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply Buffer    │
│ (50km default)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract BBox    │
│ (lat/lon range) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Filter Dataset  │
│ by coordinates  │
└─────────────────┘
```

---

## 🔗 Related Documents

- `config/gates.yaml` - Gate configuration file
- `src/gates/catalog.py` - GateCatalog implementation
- `src/services/gate_service.py` - Gate service layer
- `api/routers/gates_router.py` - REST API endpoints

---

*Last updated: 2025-12-29*

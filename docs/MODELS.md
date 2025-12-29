# 📐 Core Models Reference

> **Version**: 1.0  
> **Last Updated**: 2025-12-29  
> **Location**: `src/core/models.py`

---

## 📋 Overview

This document describes the Pydantic models shared across all branches and layers of the NICO project. These models provide:
- **Type safety** - Validated at runtime
- **Serialization** - JSON compatible for API
- **Documentation** - Self-documenting schemas

---

## 🔑 Core Models

### BoundingBox

Geographic bounding box with validation.

```python
from pydantic import BaseModel, Field
from typing import Tuple

class BoundingBox(BaseModel):
    """Geographic bounding box with validation."""
    lat_min: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    lat_max: float = Field(..., ge=-90, le=90, description="Maximum latitude")
    lon_min: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    lon_max: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    
    @property
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (lat_min, lat_max, lon_min, lon_max)."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
    
    @property 
    def lat_range(self) -> Tuple[float, float]:
        """Return latitude range as (min, max)."""
        return (self.lat_min, self.lat_max)
    
    @property
    def lon_range(self) -> Tuple[float, float]:
        """Return longitude range as (min, max)."""
        return (self.lon_min, self.lon_max)
    
    @property
    def as_list(self) -> list[float]:
        """Return as [lat_min, lat_max, lon_min, lon_max] for API compatibility."""
        return [self.lat_min, self.lat_max, self.lon_min, self.lon_max]
```

**Usage:**
```python
bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
print(bbox.lat_range)  # (78.0, 80.0)
print(bbox.as_list)    # [78.0, 80.0, -20.0, 10.0]
```

---

### GateModel

Ocean gate definition.

```python
from pydantic import BaseModel
from typing import Optional, List

class GateModel(BaseModel):
    """Ocean gate definition."""
    id: str
    name: str
    file: str
    description: str
    region: str
    emoji: Optional[str] = None
    closest_passes: Optional[List[int]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "fram_strait",
                "name": "🧊 Fram Strait",
                "file": "fram_strait_S3_pass_481.shp",
                "description": "Main Arctic-Atlantic exchange",
                "region": "Atlantic Sector",
                "closest_passes": [481, 254, 127, 308, 55]
            }
        }
```

**Usage:**
```python
gate = GateModel(
    id="fram_strait",
    name="🧊 Fram Strait",
    file="fram_strait_S3_pass_481.shp",
    description="Main Arctic-Atlantic exchange",
    region="Atlantic Sector"
)
```

---

### TimeRange

Temporal range for data queries.

```python
from pydantic import BaseModel, field_validator
from datetime import datetime

class TimeRange(BaseModel):
    """Temporal range for data queries."""
    start: str  # ISO format YYYY-MM-DD
    end: str    # ISO format YYYY-MM-DD
    
    @field_validator('start', 'end')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
        return v
    
    @property
    def start_date(self) -> datetime:
        return datetime.fromisoformat(self.start)
    
    @property
    def end_date(self) -> datetime:
        return datetime.fromisoformat(self.end)
```

**Usage:**
```python
time_range = TimeRange(start="2024-01-01", end="2024-12-31")
print(time_range.start_date)  # datetime(2024, 1, 1)
```

---

### DataRequest

Unified data request model for API and services.

```python
from pydantic import BaseModel
from typing import Optional, List

class DataRequest(BaseModel):
    """Unified data request model."""
    bbox: BoundingBox
    time_range: TimeRange
    variables: List[str]
    gate_id: Optional[str] = None
    pass_number: Optional[int] = None
    source: Optional[str] = None  # cmems, era5, etc.
    temporal_resolution: str = "daily"
    spatial_resolution: str = "0.25"
    
    class Config:
        json_schema_extra = {
            "example": {
                "bbox": {
                    "lat_min": 78.0,
                    "lat_max": 80.0,
                    "lon_min": -20.0,
                    "lon_max": 10.0
                },
                "time_range": {
                    "start": "2024-01-01",
                    "end": "2024-12-31"
                },
                "variables": ["sla", "adt"],
                "gate_id": "fram_strait",
                "source": "cmems"
            }
        }
```

**Usage:**
```python
request = DataRequest(
    bbox=BoundingBox(lat_min=78, lat_max=80, lon_min=-20, lon_max=10),
    time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
    variables=["sla", "adt"],
    gate_id="fram_strait"
)
```

---

### ResolutionConfig

Resolution configuration for data downloads.

```python
from pydantic import BaseModel
from enum import Enum

class TemporalResolution(str, Enum):
    HOURLY = "hourly"
    THREE_HOURLY = "3-hourly"
    SIX_HOURLY = "6-hourly"
    DAILY = "daily"
    MONTHLY = "monthly"

class SpatialResolution(str, Enum):
    HIGH = "0.1"
    MEDIUM = "0.25"
    LOW = "0.5"
    COARSE = "1.0"

class ResolutionConfig(BaseModel):
    """Resolution configuration."""
    temporal: TemporalResolution = TemporalResolution.DAILY
    spatial: SpatialResolution = SpatialResolution.MEDIUM
```

---

### AppConfig (Streamlit-specific)

Application configuration from sidebar. Extends core models for Streamlit use.

```python
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

@dataclass
class AppConfig:
    """Application configuration from sidebar."""
    # Analysis parameters
    mss_var: str = "mean_sea_surface"
    bin_size: float = 0.01
    sample_fraction: float = 1.0
    
    # Spatial filter (from BoundingBox)
    lat_range: Optional[Tuple[float, float]] = None
    lon_range: Optional[Tuple[float, float]] = None
    use_spatial_filter: bool = False
    
    # Gate configuration (from GateModel)
    selected_gate: Optional[str] = None
    gate_geometry: Any = None  # GeoDataFrame
    gate_buffer_km: float = 50.0
    selected_pass: Optional[int] = None
    
    @classmethod
    def from_bbox(cls, bbox: BoundingBox, **kwargs) -> "AppConfig":
        """Create from BoundingBox model."""
        return cls(
            lat_range=bbox.lat_range,
            lon_range=bbox.lon_range,
            use_spatial_filter=True,
            **kwargs
        )
```

---

## 🔄 Model Relationships

```
DataRequest
    │
    ├── BoundingBox
    │       │
    │       └── lat_range, lon_range → AppConfig
    │
    ├── TimeRange
    │
    └── gate_id → GateModel
                      │
                      └── file → Shapefile → GeoDataFrame
```

---

## 📊 API Response Models

### GateResponse
```python
class GateResponse(BaseModel):
    """API response for gate details."""
    gate: GateModel
    bbox: Optional[BoundingBox] = None
    available: bool = True
```

### DataResponse
```python
class DataResponse(BaseModel):
    """API response for data download."""
    request_id: str
    status: str  # pending, downloading, ready, error
    progress: float = 0.0
    data_url: Optional[str] = None
    metadata: Optional[dict] = None
```

---

## 🧪 Validation Examples

### BoundingBox Validation
```python
# Valid
bbox = BoundingBox(lat_min=-90, lat_max=90, lon_min=-180, lon_max=180)

# Invalid - raises ValidationError
bbox = BoundingBox(lat_min=100, lat_max=80, lon_min=-20, lon_max=10)
# Error: lat_min must be <= 90
```

### TimeRange Validation
```python
# Valid
tr = TimeRange(start="2024-01-01", end="2024-12-31")

# Invalid - raises ValidationError
tr = TimeRange(start="01-01-2024", end="2024-12-31")
# Error: Invalid date format
```

---

## 🔗 Related Documents

- `src/core/models.py` - Implementation (to be created)
- `docs/GATES_CATALOG.md` - Gate details
- `api/models.py` - API-specific models

---

*Last updated: 2025-12-29*

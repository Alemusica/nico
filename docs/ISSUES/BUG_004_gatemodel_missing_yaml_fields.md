# 🐛 GateModel Missing Fields from YAML

## Issue Type
Bug / Model Incomplete

## Summary
`GateModel` was missing fields that exist in `config/gates.yaml`:
- `datasets` - Recommended datasets for the gate
- `default_buffer_km` - Default buffer around gate
- `latitude_range` / `longitude_range` - Bounding box coordinates
- `bbox` property - Computed bounding box

## Error Message
```
AttributeError: 'GateModel' object has no attribute 'datasets'

File "sidebar.py", line 299, in _render_gate_selector_v2
    Datasets: {', '.join(gate.datasets) if gate.datasets else 'Any'}
```

## Root Cause
1. `GateModel` in `src/core/models.py` didn't have `datasets`, `default_buffer_km`, `latitude_range`, `longitude_range` fields
2. `GateCatalog._load_config()` in `src/gates/catalog.py` didn't parse these fields from YAML
3. YAML file `config/gates.yaml` had the fields but they weren't being used

## Solution

### 1. Updated `src/core/models.py` - GateModel
```python
class GateModel(BaseModel):
    # ... existing fields ...
    datasets: Optional[List[str]] = Field(default=None, description="Recommended datasets")
    default_buffer_km: Optional[float] = Field(default=50.0, description="Default buffer in km")
    latitude_range: Optional[List[float]] = Field(default=None, description="[lat_min, lat_max]")
    longitude_range: Optional[List[float]] = Field(default=None, description="[lon_min, lon_max]")
    
    @property
    def bbox(self) -> Optional[BoundingBox]:
        """Get bounding box from lat/lon fields or ranges."""
        if self.latitude_range and self.longitude_range:
            return BoundingBox(
                lat_min=self.latitude_range[0],
                lat_max=self.latitude_range[1],
                lon_min=self.longitude_range[0],
                lon_max=self.longitude_range[1]
            )
        return None
```

### 2. Updated `src/gates/catalog.py` - _load_config()
```python
self._gates[gate_id] = GateModel(
    id=gate_id,
    name=info.get("name", gate_id),
    file=info.get("file", f"{gate_id}.shp"),
    description=info.get("description", ""),
    region=info.get("region", "Unknown"),
    closest_passes=info.get("closest_passes"),
    datasets=info.get("datasets"),  # NEW
    default_buffer_km=info.get("default_buffer_km", default_buffer),  # NEW
    latitude_range=info.get("latitude_range"),  # NEW
    longitude_range=info.get("longitude_range"),  # NEW
)
```

### 3. Updated `config/gates.yaml`
Added `datasets` and `default_buffer_km` to all gates.

## Files Modified
- `src/core/models.py` - Added fields to GateModel
- `src/gates/catalog.py` - Parse new fields in _load_config()
- `config/gates.yaml` - Added datasets/buffer to all gates

## Testing
```python
from src.gates.catalog import GateCatalog
catalog = GateCatalog()
gate = catalog.get('fram_strait')
print(gate.datasets)  # ['SLCCI', 'ERA5', 'CMEMS-SST']
print(gate.bbox)  # BoundingBox(lat_min=78.0, lat_max=80.0, ...)
```

## Prevention
- Ensure model fields match YAML schema
- Add validation tests for config loading
- Document all GateModel fields with examples

## Status
✅ Fixed

## Related
- #12 Unified Architecture Refactoring
- #13 GateService missing methods

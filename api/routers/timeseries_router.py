"""
⏱️ Timeseries Router - Time animation and slicing endpoints
============================================================

Provides endpoints for:
- Time series slicing (for animation)
- Causal chain temporal propagation
- Event timeline data
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

router = APIRouter(prefix="/timeseries", tags=["timeseries"])


# ==============================================================================
# MODELS
# ==============================================================================

class TimeSliceRequest(BaseModel):
    """Request for a time slice of data."""
    dataset_id: str = Field(..., description="Dataset identifier")
    variables: List[str] = Field(..., description="Variables to retrieve")
    start_time: str = Field(..., description="Start time (ISO format)")
    end_time: str = Field(..., description="End time (ISO format)")
    lat_range: Optional[tuple] = Field(None, description="(min, max) latitude")
    lon_range: Optional[tuple] = Field(None, description="(min, max) longitude")
    aggregation: str = Field("mean", description="Spatial aggregation: mean, max, min")


class TimeSliceResponse(BaseModel):
    """Response with time series data."""
    dataset_id: str
    variables: List[str]
    timestamps: List[str]
    values: Dict[str, List[float]]  # variable -> values
    metadata: Dict[str, Any]


class AnimationFrame(BaseModel):
    """A single frame for animation."""
    timestamp: str
    values: Dict[str, float]
    events: List[Dict[str, Any]] = []


class CausalPropagationRequest(BaseModel):
    """Request for causal chain temporal propagation."""
    source_variable: str
    target_variable: str
    start_time: str
    duration_hours: int = 72


class CausalPropagationResponse(BaseModel):
    """Temporal propagation through causal chain."""
    source: str
    target: str
    total_lag_hours: float
    frames: List[AnimationFrame]
    causal_path: List[Dict[str, Any]]


# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================

def _load_lago_maggiore_data() -> dict:
    """Load Lago Maggiore pipeline data if available."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "pipeline" / "lago_maggiore_2000"
    
    # Try NetCDF
    nc_file = data_dir / "era5_lago_maggiore_2000.nc"
    if nc_file.exists():
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_file)
            return {
                "type": "netcdf",
                "path": str(nc_file),
                "dataset": ds,
                "variables": list(ds.data_vars),
                "time_range": (
                    str(ds.time.values[0])[:19],
                    str(ds.time.values[-1])[:19]
                ),
            }
        except ImportError:
            pass
    
    # Try JSON results
    json_file = data_dir / "pcmci_results.json"
    if json_file.exists():
        with open(json_file) as f:
            return {
                "type": "json",
                "path": str(json_file),
                "data": json.load(f),
            }
    
    return None


def _get_causal_chain(source: str, target: str) -> List[Dict]:
    """Get causal chain between source and target."""
    try:
        from src.data_manager.causal_graph import CausalGraphDB
        db = CausalGraphDB()
        
        # Direct link
        edges = db.get_all_edges()
        chain = []
        
        for edge in edges:
            if edge.get("source_var") == source and edge.get("target_var") == target:
                chain.append({
                    "source": edge.get("source_var"),
                    "target": edge.get("target_var"),
                    "lag_days": edge.get("lag_days", 0),
                    "strength": edge.get("correlation", 0),
                })
        
        return chain
    except Exception:
        return []


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@router.get("/available")
async def get_available_timeseries() -> Dict[str, Any]:
    """
    Get available time series data.
    """
    available = []
    
    # Check Lago Maggiore data
    data = _load_lago_maggiore_data()
    if data:
        available.append({
            "id": "lago_maggiore_2000",
            "name": "Lago Maggiore Flood 2000",
            "type": data["type"],
            "variables": data.get("variables", []),
            "time_range": data.get("time_range"),
        })
    
    return {
        "datasets": available,
        "count": len(available),
    }


@router.get("/slice/{dataset_id}")
async def get_time_slice(
    dataset_id: str,
    variables: str = Query(..., description="Comma-separated variable names"),
    start_time: Optional[str] = Query(None, description="Start time (ISO)"),
    end_time: Optional[str] = Query(None, description="End time (ISO)"),
    step: int = Query(1, description="Time step for subsampling"),
) -> TimeSliceResponse:
    """
    Get a time slice of data for animation.
    
    Example:
        GET /timeseries/slice/lago_maggiore_2000?variables=precipitation,runoff&step=4
    """
    if dataset_id == "lago_maggiore_2000":
        data = _load_lago_maggiore_data()
        if not data:
            raise HTTPException(404, "Lago Maggiore data not found. Run the pipeline first.")
        
        if data["type"] == "netcdf":
            ds = data["dataset"]
            var_list = [v.strip() for v in variables.split(",")]
            
            # Variable name mapping
            var_map = {
                "precipitation": "tp",
                "temperature": "t2m", 
                "pressure": "msl",
                "u_wind": "u10",
                "v_wind": "v10",
                "soil_moisture": "swvl1",
                "runoff": "ro",
            }
            
            # Time slicing
            if start_time:
                ds = ds.sel(time=slice(start_time, end_time or None))
            
            # Subsample
            ds = ds.isel(time=slice(None, None, step))
            
            # Spatial mean
            timestamps = [str(t)[:19] for t in ds.time.values]
            values = {}
            
            for var in var_list:
                nc_var = var_map.get(var, var)
                if nc_var in ds.data_vars:
                    vals = ds[nc_var].mean(dim=["latitude", "longitude"]).values
                    values[var] = [float(v) for v in vals]
            
            return TimeSliceResponse(
                dataset_id=dataset_id,
                variables=list(values.keys()),
                timestamps=timestamps,
                values=values,
                metadata={
                    "source": str(data["path"]),
                    "n_timesteps": len(timestamps),
                    "step": step,
                }
            )
    
    raise HTTPException(404, f"Dataset '{dataset_id}' not found")


@router.get("/frames/{dataset_id}")
async def get_animation_frames(
    dataset_id: str,
    variables: str = Query(..., description="Comma-separated variable names"),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    step: int = Query(4, description="Time step (4 = 6-hourly for ERA5)"),
) -> List[AnimationFrame]:
    """
    Get animation frames for time slider visualization.
    
    Returns frames suitable for frontend animation.
    """
    # Get slice
    slice_data = await get_time_slice(dataset_id, variables, start_time, end_time, step)
    
    # Convert to frames
    frames = []
    for i, timestamp in enumerate(slice_data.timestamps):
        frame_values = {
            var: slice_data.values[var][i] 
            for var in slice_data.variables
        }
        
        # Add events (flood peak detection)
        events = []
        if "precipitation" in frame_values:
            precip = frame_values["precipitation"]
            if precip > 0.05:  # High precipitation threshold
                events.append({
                    "type": "high_precipitation",
                    "severity": min(1.0, precip / 0.1),
                    "label": f"Heavy rain: {precip*1000:.1f} mm"
                })
        
        frames.append(AnimationFrame(
            timestamp=timestamp,
            values=frame_values,
            events=events,
        ))
    
    return frames


@router.post("/causal-propagation")
async def get_causal_propagation(
    request: CausalPropagationRequest
) -> CausalPropagationResponse:
    """
    Show how a signal propagates through causal chain over time.
    
    Example: precipitation spike → runoff increase (with lag)
    """
    # Get causal chain
    chain = _get_causal_chain(request.source_variable, request.target_variable)
    
    if not chain:
        # Return empty propagation
        return CausalPropagationResponse(
            source=request.source_variable,
            target=request.target_variable,
            total_lag_hours=0,
            frames=[],
            causal_path=[],
        )
    
    # Calculate total lag
    total_lag_days = sum(c.get("lag_days", 0) for c in chain)
    total_lag_hours = total_lag_days * 24
    
    # Generate propagation frames
    start = datetime.fromisoformat(request.start_time)
    frames = []
    
    for hour in range(0, request.duration_hours, 6):
        current = start + timedelta(hours=hour)
        
        # Simulate signal propagation
        source_signal = 1.0 if hour < 24 else 0.5  # Impulse response
        
        # Target signal appears after lag
        if hour >= total_lag_hours:
            target_signal = source_signal * chain[0].get("strength", 0.5)
        else:
            target_signal = 0.0
        
        frames.append(AnimationFrame(
            timestamp=current.isoformat(),
            values={
                request.source_variable: source_signal,
                request.target_variable: target_signal,
            },
            events=[],
        ))
    
    return CausalPropagationResponse(
        source=request.source_variable,
        target=request.target_variable,
        total_lag_hours=total_lag_hours,
        frames=frames,
        causal_path=chain,
    )


@router.get("/events")
async def get_event_timeline(
    dataset_id: str = Query("lago_maggiore_2000"),
    threshold: float = Query(0.05, description="Anomaly threshold"),
) -> Dict[str, Any]:
    """
    Get timeline of detected events (anomalies, peaks, etc.)
    """
    events = []
    
    if dataset_id == "lago_maggiore_2000":
        data = _load_lago_maggiore_data()
        if data and data["type"] == "netcdf":
            ds = data["dataset"]
            
            # Detect precipitation peaks
            precip = ds["tp"].mean(dim=["latitude", "longitude"]).values
            times = ds.time.values
            
            for i, (t, p) in enumerate(zip(times, precip)):
                if p > threshold:
                    events.append({
                        "timestamp": str(t)[:19],
                        "type": "precipitation_peak",
                        "variable": "precipitation",
                        "value": float(p),
                        "severity": min(1.0, float(p) / 0.1),
                    })
    
    # Sort by time
    events.sort(key=lambda x: x["timestamp"])
    
    return {
        "dataset_id": dataset_id,
        "events": events,
        "count": len(events),
        "threshold": threshold,
    }

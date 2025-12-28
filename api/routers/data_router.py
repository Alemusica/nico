"""
Data Management Router

Endpoints for data upload, download, caching, and briefing management.
Handles both direct data operations and investigation-related data workflows.
"""

import asyncio
import hashlib
import io
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.services.data_service import DataService, DatasetMetadata, get_data_service
from api.performance import cached, timed, get_cache


router = APIRouter(prefix="/data", tags=["data"])

# Service availability flags
DATA_MANAGER_AVAILABLE = False
try:
    from src.data_manager.manager import DataManager, InvestigationBriefing
    from src.data_manager.config import (
        ResolutionConfig,
        TemporalResolution,
        SpatialResolution,
    )
    DATA_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DataManager not available: {e}")


# ============== DEPENDENCY FUNCTIONS ==============

# Global data manager instance
_data_manager = None

def get_data_manager() -> "DataManager":
    """Get or create data manager instance."""
    global _data_manager
    if _data_manager is None and DATA_MANAGER_AVAILABLE:
        _data_manager = DataManager()
    return _data_manager



# ============== REQUEST/RESPONSE MODELS ==============

class ResolutionRequest(BaseModel):
    """Request model for resolution configuration."""
    temporal: str = "daily"  # hourly, 3-hourly, 6-hourly, daily, monthly
    spatial: str = "0.25"    # 0.1, 0.25, 0.5, 1.0


class BriefingRequest(BaseModel):
    """Request model for investigation briefing."""
    query: str
    location_name: str
    location_bbox: List[float]  # [lat_min, lat_max, lon_min, lon_max]
    event_type: str
    event_start: str  # YYYY-MM-DD
    event_end: str    # YYYY-MM-DD
    precursor_days: int = 30
    resolution: Optional[ResolutionRequest] = None


class BriefingConfirmation(BaseModel):
    """Request model for confirming a briefing."""
    briefing_id: str
    confirmed: bool = True
    modified_resolution: Optional[ResolutionRequest] = None


class DirectDownloadRequest(BaseModel):
    """Request for direct data download from Data Explorer."""
    source: str  # cmems, era5, climate_indices
    variables: List[str]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    start_date: str
    end_date: str
    temporal_resolution: str = "daily"
    spatial_resolution: str = "0.25"


# ============== FILE OPERATIONS ==============

@router.get("/files")
@cached(ttl_seconds=300, key_prefix="data_files")
async def list_files():
    """
    List all available data files in the data directory.
    
    Returns a list of files ready for loading and analysis.
    Cached for 5 minutes to improve performance.
    
    **Response Example:**
    ```json
    {
      "files": [
        "fram_strait_2020_2023.csv",
        "barents_opening_daily.nc",
        "nao_index_monthly.csv"
      ]
    }
    ```
    """
    service = get_data_service()
    return {"files": service.list_available_files()}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and load a data file (CSV or NetCDF).
    
    Accepts CSV and NetCDF (.nc) files. The file is loaded into memory
    and metadata is automatically extracted.
    
    **Supported Formats:**
    - CSV: Comma-separated values with headers
    - NetCDF: Network Common Data Form (Climate/Forecast conventions)
    
    **Request:**
    - Content-Type: multipart/form-data
    - Field: `file` (binary file content)
    
    **Response Example:**
    ```json
    {
      "success": true,
      "dataset": {
        "name": "fram_strait_temperature.csv",
        "rows": 1460,
        "columns": ["time", "temperature", "salinity"],
        "time_range": ["2020-01-01", "2023-12-31"],
        "variables": ["temperature", "salinity"]
      }
    }
    ```
    
    **Errors:**
    - 400: Unsupported file type
    - 500: File parsing error
    """
    service = get_data_service()
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = service.load_csv(io.BytesIO(content), name=file.filename)
        elif file.filename.endswith('.nc'):
            # Save temporarily for xarray
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
                f.write(content)
                df = service.load_netcdf(f.name, name=file.filename)
        else:
            raise HTTPException(400, f"Unsupported file type: {file.filename}")
        
        meta = service.get_metadata(file.filename)
        
        return {
            "success": True,
            "dataset": meta.__dict__ if meta else None,
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/load/{file_path:path}")
async def load_file(file_path: str):
    """Load a file from the data directory."""
    service = get_data_service()
    
    try:
        df = service.load_file(file_path)
        meta = service.get_metadata(file_path.split('/')[-1].split('.')[0])
        
        return {
            "success": True,
            "dataset": meta.__dict__ if meta else None,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ============== INTAKE CATALOG ENDPOINTS ==============
# NOTE: These must be BEFORE /{name} to avoid route conflicts

# Catalog availability flag
INTAKE_CATALOG_AVAILABLE = False
try:
    from src.data_manager.intake_bridge import get_catalog, IntakeCatalogBridge
    INTAKE_CATALOG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Intake Catalog not available: {e}")


@router.get("/catalog")
@cached(ttl_seconds=600, key_prefix="intake_catalog")
async def list_catalog():
    """
    List all datasets in the Intake catalog with latency metadata.
    
    Returns multi-provider catalog (CMEMS, ERA5, NOAA, NASA, ESA).
    """
    if not INTAKE_CATALOG_AVAILABLE:
        raise HTTPException(503, "Intake catalog not available")
    
    cat = get_catalog()
    datasets = []
    
    for ds_id in cat.list_datasets():
        meta = cat.get_metadata(ds_id)
        datasets.append({
            "id": ds_id,
            "description": meta.get("description", ds_id),
            "provider": meta.get("provider"),
            "variables": meta.get("variables", []),
            "latency": meta.get("latency"),
            "latency_badge": meta.get("latency_badge"),
            "status": meta.get("status"),
            "resolution_spatial": meta.get("resolution_spatial"),
            "resolution_temporal": meta.get("resolution_temporal"),
        })
    
    return {
        "datasets": datasets,
        "count": len(datasets),
        "summary": cat.summary(),
    }


@router.get("/catalog/search")
async def search_catalog(
    variables: Optional[str] = None,
    latency: Optional[str] = None,
    status: Optional[str] = None,
    provider: Optional[str] = None,
):
    """
    Search datasets by criteria.
    
    Args:
        variables: Comma-separated variables (e.g., "sla,sst")
        latency: Latency badge filter ("🟢", "🟡", "🔴", "⚫")
        status: Status filter ("available", "to_implement")
        provider: Provider filter ("Copernicus Marine", "ECMWF", etc.)
    """
    if not INTAKE_CATALOG_AVAILABLE:
        raise HTTPException(503, "Intake catalog not available")
    
    cat = get_catalog()
    var_list = variables.split(",") if variables else None
    
    results = cat.search(
        variables=var_list,
        latency_badge=latency,
        status=status,
        provider=provider,
    )
    
    return {
        "matches": results,
        "count": len(results),
    }


@router.get("/catalog/realtime")
async def get_realtime_datasets():
    """
    Get datasets with near real-time latency (🟢).
    
    Useful for operational monitoring.
    """
    if not INTAKE_CATALOG_AVAILABLE:
        raise HTTPException(503, "Intake catalog not available")
    
    cat = get_catalog()
    results = cat.search_by_latency(max_latency="🟢")
    
    return {
        "datasets": results,
        "count": len(results),
        "note": "Near real-time datasets (latency < 24h)",
    }


@router.get("/catalog/{dataset_id}")
async def get_catalog_dataset(dataset_id: str):
    """Get detailed metadata for a specific dataset."""
    if not INTAKE_CATALOG_AVAILABLE:
        raise HTTPException(503, "Intake catalog not available")
    
    cat = get_catalog()
    try:
        meta = cat.get_metadata(dataset_id)
        return {
            "id": dataset_id,
            **meta,
        }
    except KeyError:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")


# ============== DATASET OPERATIONS (catch-all routes last) ==============

@router.get("/{name}")
async def get_dataset_info(name: str):
    """Get dataset metadata and sample."""
    service = get_data_service()
    meta = service.get_metadata(name)
    
    if not meta:
        raise HTTPException(404, f"Dataset '{name}' not found")
    
    sample = service.get_sample_data(name, n_rows=5)
    
    return {
        "metadata": meta.__dict__,
        "sample": sample,
    }


@router.get("/{name}/preview")
async def get_data_preview(name: str, rows: int = 100):
    """Get data preview as JSON."""
    service = get_data_service()
    df = service.get_dataset(name)
    
    if df is None:
        raise HTTPException(404, f"Dataset '{name}' not found")
    
    return df.head(rows).to_dict(orient="records")


# ============== DATA SOURCES & RESOLUTION ==============

@router.get("/sources")
async def get_data_sources():
    """Get available data sources and their status."""
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    return {
        "sources": manager.get_available_sources(),
        "default_resolution": manager.config.investigation_resolution.to_dict(),
    }


@router.get("/resolutions")
async def get_available_resolutions():
    """Get available resolution options."""
    return {
        "temporal": [
            {"value": "hourly", "label": "Hourly (24/day)", "description": "Most detailed, largest downloads"},
            {"value": "3-hourly", "label": "3-hourly (8/day)", "description": "Good for sub-daily patterns"},
            {"value": "6-hourly", "label": "6-hourly (4/day)", "description": "Synoptic patterns, recommended"},
            {"value": "daily", "label": "Daily (1/day)", "description": "Fastest, good for multi-day events"},
            {"value": "monthly", "label": "Monthly", "description": "Climate indices only"},
        ],
        "spatial": [
            {"value": "0.1", "label": "High (0.1°, ~11km)", "description": "Small areas only"},
            {"value": "0.25", "label": "Medium (0.25°, ~28km)", "description": "Default, ERA5 native"},
            {"value": "0.5", "label": "Low (0.5°, ~55km)", "description": "Faster for large areas"},
            {"value": "1.0", "label": "Coarse (1.0°, ~111km)", "description": "Continental scale"},
        ],
    }


@router.put("/resolution")
async def update_default_resolution(resolution: ResolutionRequest):
    """Update default investigation resolution."""
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    manager.update_resolution(temporal=resolution.temporal, spatial=resolution.spatial)
    return {"status": "updated", "resolution": manager.config.investigation_resolution.to_dict()}


# ============== BRIEFING WORKFLOW ==============

# In-memory briefing storage (TODO: move to proper state management)
_briefings = {}

@router.post("/briefing")
async def create_investigation_briefing(request: BriefingRequest):
    """
    Create an investigation briefing for user confirmation.
    
    This is the Ishikawa-style planning step where LLM proposes
    what data to collect and user confirms before download.
    """
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    
    # Parse resolution if provided
    resolution = None
    if request.resolution:
        resolution = ResolutionConfig(
            temporal=TemporalResolution(request.resolution.temporal),
            spatial=SpatialResolution(request.resolution.spatial),
        )
    
    # Create briefing
    briefing = manager.create_briefing(
        query=request.query,
        location_name=request.location_name,
        location_bbox=tuple(request.location_bbox),
        event_type=request.event_type,
        event_time_range=(request.event_start, request.event_end),
        precursor_days=request.precursor_days,
        resolution=resolution,
    )
    
    # Store briefing for later confirmation
    briefing_id = hashlib.md5(f"{request.query}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    _briefings[briefing_id] = briefing
    
    return {
        "briefing_id": briefing_id,
        "briefing": briefing.to_dict(),
        "summary": briefing.summary(),
    }


# In-memory download job storage (TODO: move to proper state management)
_download_jobs = {}

@router.post("/briefing/{briefing_id}/confirm")
async def confirm_briefing(briefing_id: str, confirmation: BriefingConfirmation):
    """
    Confirm a briefing and start download.
    
    Returns immediately with download job ID. Use /data/download/{job_id}/status
    to check progress, or use WebSocket for real-time updates.
    """
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    if briefing_id not in _briefings:
        raise HTTPException(404, "Briefing not found")
    
    briefing = _briefings[briefing_id]
    
    if not confirmation.confirmed:
        del _briefings[briefing_id]
        return {"status": "cancelled"}
    
    # Update resolution if modified
    if confirmation.modified_resolution:
        for req in briefing.data_requests:
            req.resolution = ResolutionConfig(
                temporal=TemporalResolution(confirmation.modified_resolution.temporal),
                spatial=SpatialResolution(confirmation.modified_resolution.spatial),
            )
    
    briefing.confirmed = True
    
    # Start download in background
    job_id = str(uuid.uuid4())[:8]
    
    _download_jobs[job_id] = {
        "status": "queued",
        "briefing_id": briefing_id,
        "progress": {},
        "results": None,
    }
    
    # Launch background task
    asyncio.create_task(_run_download(job_id, briefing))
    
    return {
        "status": "started",
        "job_id": job_id,
        "briefing_id": briefing_id,
    }


async def _run_download(job_id: str, briefing: "InvestigationBriefing"):
    """Background task to run download."""
    manager = get_data_manager()
    job = _download_jobs[job_id]
    job["status"] = "downloading"
    
    def progress_callback(source: str, progress: float, message: str):
        job["progress"][source] = {"progress": progress, "message": message}
    
    try:
        results = await manager.download_briefing(briefing, progress_callback)
        job["results"] = {k: "downloaded" if v is not None else "failed" for k, v in results.items()}
        job["status"] = "completed"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# ============== DIRECT DOWNLOAD ==============

@router.post("/download")
async def direct_download(request: DirectDownloadRequest):
    """
    Direct data download from Data Explorer.
    
    Downloads data from specified source with given parameters.
    """
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    
    try:
        from src.data_manager.models import DataRequest
        
        lat_range = (request.lat_min, request.lat_max)
        lon_range = (request.lon_min, request.lon_max)
        time_range = (request.start_date, request.end_date)
        
        # Create resolution config
        temporal = TemporalResolution(request.temporal_resolution)
        spatial = SpatialResolution(request.spatial_resolution)
        resolution = ResolutionConfig(temporal=temporal, spatial=spatial)
        
        # Estimate size and time
        size_mb, time_sec = manager.estimate_request(
            request.source,
            request.variables,
            lat_range,
            lon_range,
            time_range,
            resolution,
        )
        
        # Create data request
        data_request = DataRequest(
            source=request.source,
            variables=request.variables,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            resolution=resolution,
            description=f"Direct download from {request.source}",
            estimated_size_mb=size_mb,
            estimated_time_sec=time_sec,
        )
        
        # Download using internal method
        data = await manager._download_source(data_request)
        
        if data is None:
            return {
                "status": "error",
                "message": f"No data available from {request.source}",
            }
        
        # Cache the data
        manager.cache.add(
            source=request.source,
            variables=request.variables,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            resolution_temporal=temporal.value,
            resolution_spatial=spatial.value,
            data=data,
        )
        
        # Return success with metadata
        if hasattr(data, 'dims'):  # xarray Dataset/DataArray
            return {
                "status": "success",
                "message": f"Downloaded {request.source} data ({len(request.variables)} variables)",
                "shape": {k: int(v) for k, v in data.dims.items()},
                "variables": list(data.data_vars) if hasattr(data, 'data_vars') else [],
                "size_mb": round(size_mb, 2),
                "cached": True,
            }
        elif isinstance(data, dict):  # Climate indices
            return {
                "status": "success",
                "message": f"Downloaded climate indices ({len(request.variables)} indices)",
                "indices": list(data.keys()),
                "cached": True,
            }
        else:
            return {
                "status": "success",
                "message": f"Downloaded {request.source} data",
                "cached": True,
            }
            
    except Exception as e:
        import traceback
        raise HTTPException(500, f"Download failed: {str(e)}\n{traceback.format_exc()}")


@router.get("/download/{job_id}/status")
async def get_download_status(job_id: str):
    """Get status of a download job."""
    if job_id not in _download_jobs:
        raise HTTPException(404, "Job not found")
    
    return _download_jobs[job_id]


# ============== CACHE MANAGEMENT ==============

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    return manager.get_cache_stats()


@router.get("/cache/entries")
async def list_cache_entries(source: Optional[str] = None):
    """List cached data entries."""
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    return {"entries": manager.list_cached_data(source)}


@router.delete("/cache")
async def clear_cache(source: Optional[str] = None, older_than_days: Optional[int] = None):
    """Clear cache entries."""
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    manager.clear_cache(source, older_than_days)
    return {"status": "cleared"}


@router.post("/cache/load_as_dataset")
async def load_cached_data_as_dataset(
    entry_id: str,
    dataset_name: Optional[str] = None,
):
    """
    Load cached data as a dataset for causal analysis.
    
    Converts cached xarray data to pandas DataFrame and loads into DataService.
    """
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    
    # Find cache entry - use cache.list_entries() which returns CacheEntry objects
    cache_entries = manager.cache.list_entries()
    entry = next((e for e in cache_entries if e.id == entry_id), None)
    
    if not entry:
        raise HTTPException(404, f"Cache entry {entry_id} not found")
    
    # Load data from cache (entry is now CacheEntry, not dict)
    cached_data = manager.cache.load(entry)
    
    if cached_data is None:
        raise HTTPException(500, "Failed to load data from cache")
    
    # Convert to DataFrame for analysis
    import pandas as pd
    import xarray as xr
    
    if isinstance(cached_data, xr.Dataset):
        # Convert xarray Dataset to DataFrame
        df = cached_data.to_dataframe().reset_index()
    elif isinstance(cached_data, dict):
        # Climate indices - already dict format
        df = pd.DataFrame(cached_data)
    else:
        raise HTTPException(400, "Unsupported data format")
    
    # Load into DataService
    data_service = get_data_service()
    name = dataset_name or f"{entry.source}_{entry_id[:8]}"
    
    # Store in data service
    data_service.datasets[name] = df
    data_service.metadata[name] = DatasetMetadata(
        name=name,
        file_type="cache",
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=[{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
    )
    
    return {
        "status": "loaded",
        "dataset_name": name,
        "rows": len(df),
        "columns": list(df.columns),
    }


# ========================
# LLM INTERPRETATION
# ========================
@router.post("/interpret")
async def interpret_dataset(dataset_name: str):
    """Use LLM to interpret dataset structure and meanings."""
    from api.models import InterpretationRequest, InterpretationResponse
    
    data_service = get_data_service()
    llm = get_llm_service()
    
    meta = data_service.get_metadata(dataset_name)
    if not meta:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    
    # Check LLM availability
    if not await llm.check_availability():
        return {
            "columns": meta.columns,
            "summary": "LLM not available for interpretation",
        }
    
    # Get sample data
    sample = data_service.get_sample_data(dataset_name, n_rows=10)
    
    # Run interpretation
    result = await llm.interpret_dataset(
        columns_info=meta.columns,
        filename=dataset_name,
        sample_data=sample,
    )
    
    return {
        "columns": [{
            "name": c.name,
            "dtype": c.dtype,
            "interpretation": c.interpretation,
            "is_temporal": c.is_temporal,
            "unit": c.unit,
        } for c in result.columns],
        "temporal_column": result.temporal_column,
        "suggested_targets": result.suggested_targets,
        "domain": result.domain,
        "summary": result.summary,
    }


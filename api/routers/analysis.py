"""
🔬 Advanced Analysis API Router
================================

Advanced endpoints for:
- Real PCMCI causal discovery (tigramite)
- Cross-region pattern matching
- Early warning alert system
- Dataset catalog & availability

Note: Basic analysis endpoints are in analysis_router.py
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from api.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analysis/advanced", tags=["Advanced Analysis"])


# ============== Request/Response Models ==============

class CausalLink(BaseModel):
    """A causal link between variables."""
    source: str
    target: str
    lag: int
    strength: float
    p_value: float
    score: float


class PCMCIRequest(BaseModel):
    """Request for PCMCI causal discovery."""
    data: Dict[str, List[float]]  # {variable_name: [values...]}
    target: Optional[str] = None
    max_lag: int = Field(default=30, ge=1, le=100)
    alpha: float = Field(default=0.05, ge=0.001, le=0.2)
    validate: bool = True


class PCMCIResponse(BaseModel):
    """Response from PCMCI causal discovery."""
    significant_links: List[CausalLink]
    var_names: List[str]
    method: str
    n_samples: int
    timestamp: str


class IshikawaRequest(BaseModel):
    """Request for Ishikawa diagram generation."""
    effect: str
    causes: List[Dict[str, Any]]  # [{source, score, lag}, ...]
    format: str = Field(default="json", pattern="^(json|svg|mermaid)$")


class RegionMatchRequest(BaseModel):
    """Request for cross-region pattern matching."""
    source_region: str
    effect: str
    causes: List[Dict[str, Any]]
    climate_drivers: List[str] = []
    max_distance_km: Optional[float] = 2000


class AlertCheckRequest(BaseModel):
    """Request for early warning check."""
    current_values: Dict[str, float]
    baselines: Optional[Dict[str, float]] = None
    pattern_ids: Optional[List[str]] = None


class AlertResponse(BaseModel):
    """An alert from the early warning system."""
    id: str
    region: str
    effect: str
    level: str
    level_emoji: str
    probability: float
    message: str
    recommendations: List[str]
    triggered_precursors: List[Dict[str, Any]]


# ============== PCMCI Endpoints ==============

@router.post("/causal/discover", response_model=PCMCIResponse)
async def discover_causal_links(request: PCMCIRequest):
    """
    Run PCMCI causal discovery on time series data.
    
    Discovers time-lagged causal relationships between variables.
    Requires tigramite to be installed.
    """
    try:
        import pandas as pd
        from src.pattern_engine.causal.pcmci_engine import PCMCIEngine, HAS_TIGRAMITE
        
        if not HAS_TIGRAMITE:
            raise HTTPException(
                status_code=503,
                detail="PCMCI requires tigramite. Install with: pip install tigramite"
            )
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        if len(df) < request.max_lag * 3:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least {request.max_lag * 3} samples, got {len(df)}"
            )
        
        # Run PCMCI
        engine = PCMCIEngine(
            max_lag=request.max_lag,
            alpha=request.alpha,
        )
        
        result = engine.discover(df, target=request.target)
        
        if request.validate:
            result = engine.validate_links(result, df)
        
        return PCMCIResponse(
            significant_links=[
                CausalLink(
                    source=l.source,
                    target=l.target,
                    lag=l.lag,
                    strength=l.strength,
                    p_value=l.p_value,
                    score=l.score,
                )
                for l in result.significant_links
            ],
            var_names=result.var_names,
            method=result.method,
            n_samples=len(df),
            timestamp=datetime.now().isoformat(),
        )
        
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"PCMCI error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal/methods")
async def list_causal_methods():
    """List available causal discovery methods."""
    try:
        from src.pattern_engine.causal.pcmci_engine import HAS_TIGRAMITE, HAS_CMI, HAS_GPDC
    except ImportError:
        HAS_TIGRAMITE = False
        HAS_CMI = False
        HAS_GPDC = False
    
    return {
        "methods": [
            {
                "id": "pcmci",
                "name": "PCMCI",
                "description": "Peter-Clark Momentary Conditional Independence",
                "available": HAS_TIGRAMITE,
                "tests": ["parcorr", "cmi" if HAS_CMI else None, "gpdc" if HAS_GPDC else None],
            },
            {
                "id": "granger",
                "name": "Granger Causality",
                "description": "Traditional Granger causality test",
                "available": True,
            },
            {
                "id": "correlation",
                "name": "Lagged Correlation",
                "description": "Simple lagged correlation analysis",
                "available": True,
            },
        ]
    }


# ============== Ishikawa Endpoints ==============

@router.post("/ishikawa/generate")
async def generate_ishikawa(request: IshikawaRequest):
    """
    Generate an Ishikawa (fishbone) diagram from causal analysis results.
    
    Formats:
    - json: Structured JSON with categories
    - svg: SVG image
    - mermaid: Mermaid diagram code
    """
    try:
        from src.pattern_engine.causal.ishikawa import IshikawaDiagram
        
        diagram = IshikawaDiagram.from_causal_links(
            effect=request.effect,
            causes=request.causes,
            auto_categorize=True,
        )
        
        if request.format == "svg":
            return {"format": "svg", "content": diagram.to_svg()}
        elif request.format == "mermaid":
            return {"format": "mermaid", "content": diagram.to_mermaid()}
        else:
            return {"format": "json", "content": diagram.to_dict()}
            
    except Exception as e:
        logger.error(f"Ishikawa error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Cross-Region Endpoints ==============

@router.post("/cross-region/match")
async def find_similar_regions(request: RegionMatchRequest):
    """
    Find regions where a discovered pattern might also occur.
    
    Compares climate zones, terrain, and structural features.
    """
    try:
        from src.pattern_engine.cross_region import CrossRegionMatcher, CausalPattern
        
        pattern = CausalPattern(
            id="query_pattern",
            name=f"{request.effect} pattern",
            causes=request.causes,
            effect=request.effect,
            source_region=request.source_region,
            climate_drivers=request.climate_drivers,
        )
        
        matcher = CrossRegionMatcher()
        
        matches = matcher.find_similar_regions(
            pattern=pattern,
            source_region=request.source_region,
            max_distance_km=request.max_distance_km,
        )
        
        return {
            "source_region": request.source_region,
            "effect": request.effect,
            "matches": [m.to_dict() for m in matches[:10]],
            "total_matches": len(matches),
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Cross-region error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-region/regions")
async def list_known_regions():
    """List all known regions in the database."""
    try:
        from src.pattern_engine.cross_region import KNOWN_REGIONS
        
        return {
            "regions": [
                {
                    "id": r.id,
                    "name": r.name,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "climate_zone": r.climate_zone.value,
                    "coastline_type": r.coastline_type.value if r.coastline_type else None,
                }
                for r in KNOWN_REGIONS
            ],
            "total": len(KNOWN_REGIONS),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Early Warning Endpoints ==============

# Global early warning system instance
_early_warning_system = None


def get_early_warning_system():
    """Get or create early warning system singleton."""
    global _early_warning_system
    if _early_warning_system is None:
        from src.pattern_engine.early_warning import create_default_patterns
        _early_warning_system = create_default_patterns()
    return _early_warning_system


@router.post("/early-warning/check", response_model=List[AlertResponse])
async def check_early_warning(request: AlertCheckRequest):
    """
    Check current conditions against monitored patterns.
    
    Returns alerts for each pattern based on precursor values.
    """
    try:
        system = get_early_warning_system()
        
        alerts = system.check_conditions(
            current_values=request.current_values,
            baselines=request.baselines,
            pattern_ids=request.pattern_ids,
        )
        
        return [
            AlertResponse(
                id=a.id,
                region=a.region,
                effect=a.effect,
                level=a.level.value,
                level_emoji=a.level.emoji,
                probability=a.probability,
                message=a.message,
                recommendations=a.recommendations,
                triggered_precursors=a.triggered_precursors,
            )
            for a in alerts
        ]
        
    except Exception as e:
        logger.error(f"Early warning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/early-warning/status")
async def get_early_warning_status():
    """Get current status of early warning system."""
    try:
        system = get_early_warning_system()
        return system.get_current_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/early-warning/patterns")
async def add_alert_pattern(
    pattern_id: str = Body(...),
    name: str = Body(...),
    precursors: List[Dict] = Body(...),
    effect: str = Body(...),
    region: str = Body(...),
    confidence: float = Body(0.5),
):
    """Add a new pattern to monitor."""
    try:
        system = get_early_warning_system()
        
        pattern = system.add_pattern(
            pattern_id=pattern_id,
            name=name,
            precursors=precursors,
            effect=effect,
            region=region,
            confidence=confidence,
        )
        
        return {"status": "created", "pattern": pattern.to_dict()}
        
    except Exception as e:
        logger.error(f"Add pattern error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/early-warning/patterns")
async def list_alert_patterns():
    """List all monitored patterns."""
    try:
        system = get_early_warning_system()
        return {
            "patterns": [p.to_dict() for p in system.patterns.values()],
            "total": len(system.patterns),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Timeline Endpoints ==============

@router.get("/timeline")
async def get_timeline(
    start_date: str = Query(..., description="Start date (ISO format)"),
    end_date: str = Query(..., description="End date (ISO format)"),
    entity_types: Optional[List[str]] = Query(None, description="Filter by entity types"),
    granularity: str = Query("day", description="Aggregation granularity"),
):
    """
    Get timeline data for a time range.
    
    Returns entries and aggregations for the specified period.
    """
    try:
        from api.services.timeline_service import (
            TimelineService, TimeWindow, TimeGranularity, TimelineEntityType
        )
        
        service = TimelineService()
        
        window = TimeWindow.from_strings(start_date, end_date)
        
        # Parse entity types
        types = None
        if entity_types:
            types = [TimelineEntityType(t) for t in entity_types]
        
        # Query entries
        entries = service.query(window, entity_types=types)
        
        # Aggregate
        gran = TimeGranularity(granularity)
        aggregation = service.aggregate(window, gran, types)
        
        return {
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "entries": [e.to_dict() for e in entries],
            "aggregation": aggregation.buckets,
            "total_entries": len(entries),
        }
        
    except Exception as e:
        logger.error(f"Timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Dataset Catalog Endpoints ==============

@router.get("/catalog/products")
async def list_catalog_products(
    category: Optional[str] = None,
    variable: Optional[str] = None,
):
    """List available Copernicus Marine datasets."""
    try:
        from src.data_manager.catalog import get_catalog, DataCategory
        
        catalog = get_catalog()
        
        if variable:
            products = catalog.search(variable=variable)
        elif category:
            cat = DataCategory(category)
            products = catalog.list_products(category=cat)
        else:
            products = catalog.list_products()
        
        return {
            "products": [p.to_summary() for p in products],
            "total": len(products),
        }
        
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Missing dependency: {e}")
    except Exception as e:
        logger.error(f"Catalog error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/catalog/check-availability")
async def check_data_availability(
    lat_range: List[float] = Body(..., min_length=2, max_length=2),
    lon_range: List[float] = Body(..., min_length=2, max_length=2),
    time_range: List[str] = Body(..., min_length=2, max_length=2),
    variables: Optional[List[str]] = Body(None),
):
    """Check what data is available for a specific region/time."""
    try:
        from src.data_manager.catalog import get_catalog
        
        catalog = get_catalog()
        
        result = catalog.check_availability(
            lat_range=tuple(lat_range),
            lon_range=tuple(lon_range),
            time_range=tuple(time_range),
            variables=variables,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Availability check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

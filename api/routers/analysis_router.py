"""
🔬 Root Cause Analysis API Router
===================================

API endpoints for:
- Ishikawa (fishbone) diagram generation
- FMEA (Failure Mode and Effects Analysis)
- 5-Why root cause drilling
- Hybrid scoring (physics + chain + experience)
- Multi-satellite data fusion
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

# Import services
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.llm_service import get_llm_service
from api.services.llm_root_cause import (
    RootCauseLLMExtension,
    RootCauseConfig,
    extend_llm_service,
)
from src.analysis.root_cause import (
    IshikawaCategory,
    create_flood_ishikawa_template,
    create_satellite_fmea_template,
    RootCauseAnalyzer,
    FloodPhysicsScore,
)
from src.data.satellite_fusion import (
    SatelliteFusionEngine,
    DynamicIndexCalculator,
    SATELLITE_CONFIGS,
    SatelliteStatus,
)
from api.services.causal_service import (
    get_discovery_service,
    CausalDiscoveryService,
    DiscoveryConfig,
)
from api.services.data_service import get_data_service


# Create router
router = APIRouter(prefix="/analysis", tags=["Root Cause Analysis"])


# ============== REQUEST/RESPONSE MODELS ==============

class IshikawaRequest(BaseModel):
    """Request for Ishikawa diagram generation."""
    event_description: str = Field(..., description="What happened")
    event_location: str = Field(..., description="Where (e.g., 'Fram Strait, 78°N 0°E')")
    event_time: str = Field(..., description="When (ISO format)")
    observed_data: Optional[Dict[str, Any]] = Field(None, description="Measurements at event time")
    domain: str = Field("flood", description="Analysis domain")


class IshikawaCauseResponse(BaseModel):
    """A single cause in the diagram."""
    category: str
    description: str
    evidence_level: str
    contributing_factors: List[str] = []


class IshikawaResponse(BaseModel):
    """Response containing Ishikawa diagram."""
    effect: str
    causes: List[IshikawaCauseResponse]
    summary: str


class FMEARequest(BaseModel):
    """Request for FMEA analysis."""
    component: str = Field(..., description="Component being analyzed")
    function: str = Field(..., description="What it's supposed to do")
    data_source: str = Field("satellite", description="Data source")
    known_issues: Optional[List[str]] = Field(None, description="Known issues")


class FMEAItemResponse(BaseModel):
    """Single FMEA item."""
    failure_mode: str
    effect: str
    cause: str
    severity: int = Field(..., ge=1, le=10)
    occurrence: int = Field(..., ge=1, le=10)
    detection: int = Field(..., ge=1, le=10)
    rpn: int
    recommended_action: str
    priority: str


class FMEAResponse(BaseModel):
    """Response containing FMEA analysis."""
    component: str
    items: List[FMEAItemResponse]
    high_risk_count: int
    total_rpn: int


class FiveWhyRequest(BaseModel):
    """Request for 5-Why analysis."""
    symptom: str = Field(..., description="Initial problem/observation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_depth: int = Field(5, description="Maximum depth")


class WhyStepResponse(BaseModel):
    """Single step in 5-Why chain."""
    level: int
    why: str
    because: str
    is_measurable: bool
    evidence: Optional[str] = None


class FiveWhyResponse(BaseModel):
    """Response containing 5-Why analysis."""
    symptom: str
    chain: List[WhyStepResponse]
    root_cause: Optional[str]
    depth: int


class HybridScoreRequest(BaseModel):
    """Request for hybrid scoring."""
    event: Dict[str, Any] = Field(..., description="Event details")
    causal_chain: List[Dict[str, Any]] = Field(default=[], description="Causal relationships")
    physics_data: Optional[Dict[str, float]] = Field(None, description="Physics measurements")
    historical_matches: int = Field(0, description="Number of historical matches")


class HybridScoreResponse(BaseModel):
    """Response with hybrid scores."""
    hybrid_score: float
    physics_score: float
    chain_score: float
    experience_score: float
    interpretation: str
    physics_detail: Dict[str, float] = {}


class SatelliteStatusResponse(BaseModel):
    """Response with satellite constellation status."""
    satellites: Dict[str, Dict[str, Any]]
    available_count: int
    total_count: int


class DynamicIndicesRequest(BaseModel):
    """Request for dynamic index calculation."""
    data: Dict[str, List[float]] = Field(..., description="Data arrays (sst, ssh, ice, wind)")


class DynamicIndicesResponse(BaseModel):
    """Response with calculated indices."""
    thermodynamics: Dict[str, Any]
    oceanography: Dict[str, Any]
    cryosphere: Dict[str, Any]
    anemometry: Dict[str, Any]
    precipitation: Dict[str, Any]




class DiscoveryRequest(BaseModel):
    """Request for causal discovery."""
    dataset_name: str
    variables: Optional[List[str]] = None
    time_column: Optional[str] = None
    max_lag: int = 7
    alpha_level: float = 0.05
    domain: str = "flood"
    use_llm: bool = True


class CausalLinkResponse(BaseModel):
    """A single causal link in the discovery graph."""
    source: str
    target: str
    lag: int
    strength: float
    p_value: float
    explanation: Optional[str] = None
    physics_valid: Optional[bool] = None
    physics_score: Optional[float] = None


class DiscoveryResponse(BaseModel):
    """Response containing discovered causal graph."""
    variables: List[str]
    links: List[CausalLinkResponse]
    max_lag: int
    alpha: float
    method: str
# ============== ISHIKAWA ENDPOINTS ==============

@router.post("/ishikawa", response_model=IshikawaResponse)
async def generate_ishikawa(request: IshikawaRequest):
    """
    Generate Ishikawa (fishbone) diagram for root cause analysis.
    
    Categories adapted for oceanography/flood:
    - ATMOSPHERE: Wind, pressure, precipitation
    - OCEAN: Tides, currents, stratification
    - CRYOSPHERE: Ice, freshwater flux
    - MEASUREMENT: Sensor issues, calibration
    - MODEL: Forecast errors, resolution
    - EXTERNAL: Rivers, anthropogenic, seismic
    """
    try:
        llm = get_llm_service()
        available = await llm.check_availability()
        
        if available:
            extension = RootCauseLLMExtension(llm)
            diagram = await extension.generate_ishikawa_diagram(
                event_description=request.event_description,
                event_location=request.event_location,
                event_time=request.event_time,
                observed_data=request.observed_data,
                domain=request.domain,
            )
        else:
            # Use template if LLM not available
            diagram = create_flood_ishikawa_template()
            diagram.effect = request.event_description
        
        return IshikawaResponse(
            effect=diagram.effect,
            causes=[
                IshikawaCauseResponse(
                    category=cause.category.value,
                    description=cause.description,
                    evidence_level=cause.evidence_level,
                    contributing_factors=cause.contributing_factors,
                )
                for cause in diagram.get_all_causes()
            ],
            summary=diagram.summarize(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ishikawa generation failed: {str(e)}")


@router.get("/ishikawa/template")
async def get_ishikawa_template():
    """Get a template Ishikawa diagram for flood events."""
    template = create_flood_ishikawa_template()
    
    return {
        "effect": template.effect,
        "categories": {
            cat.value: [
                {
                    "description": c.description,
                    "evidence_level": c.evidence_level,
                    "factors": c.contributing_factors,
                }
                for c in template.get_causes_by_category(cat)
            ]
            for cat in IshikawaCategory
        },
    }


# ============== FMEA ENDPOINTS ==============

@router.post("/fmea", response_model=FMEAResponse)
async def generate_fmea(request: FMEARequest):
    """
    Generate FMEA (Failure Mode and Effects Analysis) for a component.
    
    Calculates Risk Priority Number (RPN) = Severity × Occurrence × Detection
    High RPN (>100) indicates high priority for action.
    """
    try:
        llm = get_llm_service()
        available = await llm.check_availability()
        
        if available:
            extension = RootCauseLLMExtension(llm)
            analysis = await extension.generate_fmea_analysis(
                component=request.component,
                function=request.function,
                data_source=request.data_source,
                known_issues=request.known_issues,
            )
        else:
            # Use template
            analysis = create_satellite_fmea_template()
            analysis.component = request.component
        
        return FMEAResponse(
            component=analysis.component,
            items=[
                FMEAItemResponse(
                    failure_mode=item.failure_mode,
                    effect=item.effect,
                    cause=item.cause,
                    severity=item.severity,
                    occurrence=item.occurrence,
                    detection=item.detection,
                    rpn=item.rpn,
                    recommended_action=item.recommended_action,
                    priority=item.priority,
                )
                for item in analysis.get_items_sorted_by_rpn()
            ],
            high_risk_count=len(analysis.get_high_risk_items()),
            total_rpn=sum(item.rpn for item in analysis.items),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FMEA generation failed: {str(e)}")


@router.get("/fmea/template")
async def get_fmea_template():
    """Get a template FMEA for satellite data quality."""
    template = create_satellite_fmea_template()
    
    return {
        "component": template.component,
        "items": [
            {
                "failure_mode": item.failure_mode,
                "effect": item.effect,
                "cause": item.cause,
                "severity": item.severity,
                "occurrence": item.occurrence,
                "detection": item.detection,
                "rpn": item.rpn,
                "priority": item.priority,
            }
            for item in template.items
        ],
    }


# ============== 5-WHY ENDPOINTS ==============

@router.post("/5why", response_model=FiveWhyResponse)
async def run_five_why(request: FiveWhyRequest):
    """
    Run 5-Why analysis to drill down to root cause.
    
    Uses LLM to iteratively ask "Why?" and provide physics-grounded answers
    until reaching a fundamental root cause.
    """
    try:
        llm = get_llm_service()
        available = await llm.check_availability()
        
        if available:
            extension = RootCauseLLMExtension(llm)
            analysis = await extension.run_five_why_analysis(
                symptom=request.symptom,
                context=request.context,
                max_depth=request.max_depth,
            )
        else:
            # Return empty analysis if LLM not available
            from src.analysis.root_cause import FiveWhyAnalysis
            analysis = FiveWhyAnalysis(symptom=request.symptom)
            analysis.root_cause = "LLM unavailable - manual analysis required"
        
        return FiveWhyResponse(
            symptom=analysis.symptom,
            chain=[
                WhyStepResponse(
                    level=step.level,
                    why=step.why,
                    because=step.because,
                    is_measurable=step.is_measurable,
                    evidence=step.evidence,
                )
                for step in analysis.steps
            ],
            root_cause=analysis.root_cause,
            depth=analysis.depth,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"5-Why analysis failed: {str(e)}")


# ============== HYBRID SCORING ENDPOINTS ==============

@router.post("/score/hybrid", response_model=HybridScoreResponse)
async def calculate_hybrid_score(request: HybridScoreRequest):
    """
    Calculate hybrid knowledge score combining:
    - Physics-based validation (equations)
    - Chain-based scoring (causal path strength)
    - Experience-based scoring (historical patterns)
    
    Like a hybrid car: physics engine + experience engine working together.
    """
    try:
        extension = RootCauseLLMExtension(None)  # Doesn't need LLM for scoring
        
        result = await extension.calculate_hybrid_score(
            event=request.event,
            causal_chain=request.causal_chain,
            physics_data=request.physics_data,
            historical_matches=request.historical_matches,
        )
        
        return HybridScoreResponse(
            hybrid_score=result["hybrid_score"],
            physics_score=result["physics_score"],
            chain_score=result["chain_score"],
            experience_score=result["experience_score"],
            interpretation=result["interpretation"],
            physics_detail=result.get("physics_detail", {}),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid scoring failed: {str(e)}")


@router.post("/score/physics")
async def calculate_physics_score(
    wind_speed_ms: float = 10.0,
    pressure_hpa: float = 1000.0,
    reference_pressure_hpa: float = 1013.25,
    fetch_km: float = 100.0,
    depth_m: float = 20.0,
    observed_surge_m: float = 0.0,
):
    """
    Calculate physics-based score using storm surge equations.
    
    Equations:
    - Wind setup: η = (CD × ρ_air × U² × L) / (ρ_water × g × h)
    - Inverse barometer: η = -ΔP / (ρ_water × g)
    
    Returns expected surge and validation score vs observed.
    """
    scorer = FloodPhysicsScore(
        wind_speed_ms=wind_speed_ms,
        pressure_hpa=pressure_hpa,
        ref_pressure_hpa=reference_pressure_hpa,
        fetch_km=fetch_km,
        depth_m=depth_m,
        observed_surge_m=observed_surge_m,
    )
    
    return {
        "wind_setup_m": scorer.wind_setup(),
        "inverse_barometer_m": scorer.inverse_barometer(),
        "total_expected_surge_m": scorer.total_surge(),
        "observed_surge_m": observed_surge_m,
        "validation_score": scorer.validation_score(),
        "physics_constants": {
            "rho_water": 1025.0,
            "rho_air": 1.225,
            "gravity": 9.81,
            "drag_coefficient": 0.0013,
        },
    }


# ============== SATELLITE FUSION ENDPOINTS ==============

@router.get("/satellites/status", response_model=SatelliteStatusResponse)
async def get_satellite_status():
    """
    Get status of all satellites in the constellation.
    
    Shows which satellites are operational, degraded, or offline.
    """
    engine = SatelliteFusionEngine()
    status = engine.get_satellite_status()
    
    available = [name for name, info in status.items() if info["status"] == "operational"]
    
    return SatelliteStatusResponse(
        satellites=status,
        available_count=len(available),
        total_count=len(status),
    )


@router.get("/satellites/{name}")
async def get_satellite_info(name: str):
    """Get detailed information about a specific satellite."""
    if name not in SATELLITE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Satellite {name} not found")
    
    config = SATELLITE_CONFIGS[name]
    
    return {
        "name": config.name,
        "short_name": config.short_name,
        "orbital": {
            "inclination_deg": config.inclination_deg,
            "repeat_days": config.repeat_days,
            "altitude_km": config.altitude_km,
            "ground_track_spacing_km": config.ground_track_spacing_km,
        },
        "instruments": [i.value for i in config.instruments],
        "coverage": {
            "lat_min": config.lat_min,
            "lat_max": config.lat_max,
        },
        "performance": {
            "along_track_resolution_km": config.along_track_resolution_km,
            "measurement_precision_cm": config.measurement_precision_cm,
            "frequency_ghz": config.frequency_ghz,
        },
        "status": config.status.value,
        "quality_weight": config.quality_weight,
        "launch_date": config.launch_date,
    }


@router.put("/satellites/{name}/status")
async def set_satellite_status(name: str, status: str):
    """
    Update satellite status (e.g., mark as offline).
    
    Valid statuses: operational, degraded, offline
    """
    if name not in SATELLITE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Satellite {name} not found")
    
    try:
        new_status = SatelliteStatus(status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Use: {[s.value for s in SatelliteStatus]}"
        )
    
    engine = SatelliteFusionEngine()
    engine.set_satellite_status(name, new_status)
    
    return {"satellite": name, "status": new_status.value}


# ============== DYNAMIC INDICES ENDPOINTS ==============

@router.post("/indices/calculate", response_model=DynamicIndicesResponse)
async def calculate_dynamic_indices(request: DynamicIndicesRequest):
    """
    Calculate dynamic indices from satellite data.
    
    Replaces static indices with real-time values:
    - thermodynamics: From SST anomalies
    - oceanography: From SSH/SLA patterns
    - cryosphere: From sea ice concentration
    - anemometry: From altimeter wind
    """
    import numpy as np
    
    calculator = DynamicIndexCalculator()
    
    # Convert lists to numpy arrays
    data = {}
    for key, values in request.data.items():
        if values:
            data[key] = np.array(values)
    
    indices = calculator.calculate_all_indices(data)
    
    return DynamicIndicesResponse(
        thermodynamics=indices.get("thermodynamics", {"index": 0.0}),
        oceanography=indices.get("oceanography", {"index": 0.0}),
        cryosphere=indices.get("cryosphere", {"index": 0.0}),
        anemometry=indices.get("anemometry", {"index": 0.0}),
        precipitation=indices.get("precipitation", {"index": 5.0}),
    )


@router.get("/indices/weights")
async def get_index_weights():
    """
    Get default weights for hybrid scoring.
    
    Weights determine how physics, chain, and experience scores combine.
    """
    config = RootCauseConfig()
    
    return {
        "physics_weight": config.physics_weight,
        "chain_weight": config.chain_weight,
        "experience_weight": config.experience_weight,
        "description": {
            "physics": "Weight for physics-based validation (equations)",
            "chain": "Weight for causal chain strength",
            "experience": "Weight for historical pattern matching",
        },
    }


# ============== COMPREHENSIVE ANALYSIS ==============

@router.post("/comprehensive")
async def run_comprehensive_analysis(
    event_description: str,
    event_location: str,
    event_time: str,
    observed_data: Dict[str, Any] = None,
    causal_chain: List[Dict[str, Any]] = None,
    historical_matches: int = 0,
):
    """
    Run comprehensive root cause analysis combining all methods:
    1. Ishikawa diagram
    2. 5-Why analysis
    3. Hybrid scoring
    
    Returns integrated results for decision support.
    """
    try:
        llm = get_llm_service()
        available = await llm.check_availability()
        
        results = {
            "event": {
                "description": event_description,
                "location": event_location,
                "time": event_time,
            },
            "llm_available": available,
        }
        
        # Run analyses in parallel if LLM available
        if available:
            extension = RootCauseLLMExtension(llm)
            
            # Ishikawa
            ishikawa_task = extension.generate_ishikawa_diagram(
                event_description=event_description,
                event_location=event_location,
                event_time=event_time,
                observed_data=observed_data,
            )
            
            # 5-Why
            five_why_task = extension.run_five_why_analysis(
                symptom=event_description,
                context=observed_data,
            )
            
            ishikawa_result, five_why_result = await asyncio.gather(
                ishikawa_task, five_why_task
            )
            
            results["ishikawa"] = {
                "effect": ishikawa_result.effect,
                "causes_count": len(ishikawa_result.get_all_causes()),
                "summary": ishikawa_result.summarize(),
            }
            
            results["five_why"] = {
                "symptom": five_why_result.symptom,
                "root_cause": five_why_result.root_cause,
                "depth": five_why_result.depth,
            }
        else:
            # Use templates
            template = create_flood_ishikawa_template()
            results["ishikawa"] = {
                "effect": event_description,
                "causes_count": len(template.get_all_causes()),
                "summary": "Using template (LLM unavailable)",
            }
            results["five_why"] = {
                "symptom": event_description,
                "root_cause": "Manual analysis required (LLM unavailable)",
                "depth": 0,
            }
        
        # Hybrid scoring (doesn't need LLM)
        extension = RootCauseLLMExtension(None)
        
        # Extract physics data if available
        physics_data = None
        if observed_data:
            physics_data = {
                k: v for k, v in observed_data.items()
                if k in ["wind_speed", "pressure", "reference_pressure", "fetch", "depth", "observed_surge"]
            }
        
        score_result = await extension.calculate_hybrid_score(
            event={"description": event_description, "location": event_location},
            causal_chain=causal_chain or [],
            physics_data=physics_data,
            historical_matches=historical_matches,
        )
        
        results["hybrid_score"] = score_result
        
        # Summary
        results["summary"] = {
            "primary_root_cause": results.get("five_why", {}).get("root_cause", "Unknown"),
            "confidence": score_result["hybrid_score"],
            "confidence_interpretation": score_result["interpretation"],
            "recommended_actions": [
                "Verify physics calculations with observed data",
                "Cross-reference with historical events",
                "Validate causal chain with domain experts",
            ],
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


# Export router for inclusion in main app
__all__ = ["router"]


# ============== CAUSAL DISCOVERY ==============

@router.post("/discover", response_model=DiscoveryResponse)
async def discover_causality(request: DiscoveryRequest):
    """Run causal discovery on a dataset."""
    data_service = get_data_service()
    
    df = data_service.get_dataset(request.dataset_name)
    if df is None:
        raise HTTPException(404, f"Dataset '{request.dataset_name}' not found")
    
    # Configure discovery
    config = DiscoveryConfig(
        max_lag=request.max_lag,
        alpha_level=request.alpha_level,
        use_llm_explanations=request.use_llm,
    )
    
    service = CausalDiscoveryService(config)
    
    try:
        graph = await service.discover(
            df=df,
            variables=request.variables,
            time_column=request.time_column,
            domain=request.domain,
        )
        
        return DiscoveryResponse(
            variables=graph.variables,
            links=[CausalLinkResponse(
                source=l.source,
                target=l.target,
                lag=l.lag,
                strength=l.strength,
                p_value=l.p_value,
                explanation=l.explanation,
                physics_valid=l.physics_valid,
                physics_score=l.physics_score,
            ) for l in graph.links],
            max_lag=graph.max_lag,
            alpha=graph.alpha,
            method=graph.discovery_method,
        )
    
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/discover/correlations")
async def find_correlations(
    dataset_name: str,
    source_var: str,
    target_var: str,
    max_lag: int = 30,
):
    """Find cross-correlations between two variables."""
    data_service = get_data_service()
    
    df = data_service.get_dataset(dataset_name)
    if df is None:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    
    service = get_discovery_service()
    
    try:
        results = await service.find_cross_correlations(
            df=df,
            source_var=source_var,
            target_var=target_var,
            max_lag=max_lag,
        )
        return {"correlations": results[:20]}  # Top 20
    
    except Exception as e:
        raise HTTPException(500, str(e))

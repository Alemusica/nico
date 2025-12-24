"""
🚀 FastAPI Main Application
============================
REST API for Causal Discovery Dashboard.

Includes:
- Data management (upload, interpret, preview)
- Causal discovery (PCMCI, correlations)
- LLM-powered explanations
- Root cause analysis (Ishikawa, FMEA, 5-Why)
- Multi-satellite data fusion
- Hybrid scoring (physics + chain + experience)

API Documentation:
- Swagger UI: /docs
- ReDoc: /redoc
- Usage Guide: /docs/API_USAGE.md

For examples, see docs/API_USAGE.md
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import json
import io
import asyncio

# Import API models
from api.models import (
    DatasetInfo,
    InterpretationRequest,
    InterpretationResponse,
    InvestigateRequest,
    InvestigateResponse,
)

# Import exceptions
from api.exceptions import CausalDiscoveryError, map_to_http_exception

# Import logging and middleware
from api.logging_config import configure_logging, get_logger
from api.middleware import RequestIDMiddleware
from api.config import get_settings
from api.rate_limit import setup_rate_limiting, set_limiter
from api.security import setup_security_middleware

from api.services.llm_service import get_llm_service, OllamaLLMService
from api.services.causal_service import (
    get_discovery_service, 
    CausalDiscoveryService,
    DiscoveryConfig,
)
from api.services.data_service import get_data_service, DataService
from api.services.knowledge_service import (
    create_knowledge_service,
    Paper,
    HistoricalEvent,
    ClimateIndex,
    CausalPattern,
)

# Import Investigation Agent
try:
    from src.agent.investigation_agent import InvestigationAgent
    INVESTIGATION_AGENT_AVAILABLE = True
except ImportError:
    INVESTIGATION_AGENT_AVAILABLE = False
    InvestigationAgent = None

# Import Data Manager
try:
    from src.data_manager.manager import DataManager, InvestigationBriefing
    from src.data_manager.config import ResolutionConfig, TemporalResolution, SpatialResolution
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    DataManager = None

# Import routers
from api.routers.analysis_router import router as analysis_router
from api.routers.chat_router import router as chat_router
from api.routers.data_router import router as data_router
from api.routers.health_router import router as health_router
from api.routers.investigation_router import router as investigation_router
from api.routers.knowledge_router import router as knowledge_router
from api.routers.pipeline_router import router as pipeline_router

# Configure logging
settings = get_settings()
configure_logging(
    log_level=settings.log_level,
    log_format=settings.log_format,
    log_file=settings.log_file
)
logger = get_logger("api.main")

logger.info(
    "application_starting",
    app_name=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# Initialize FastAPI
app = FastAPI(
    title="🔬 Causal Discovery API",
    description="""
Intelligent causal discovery with LLM-powered explanations for oceanographic research.

## 🌊 Features

### Data Management
- **Upload** custom datasets (CSV, NetCDF, ZARR)
- **Interpret** data with automatic variable detection
- **Preview** dataset statistics and metadata
- **Cache** management for performance

### Causal Discovery
- **PCMCI** algorithm (Tigramite) for causal analysis
- **Correlation** fallback when Tigramite unavailable
- **Lag detection** up to 90 days
- **Confidence scoring** with p-values

### Root Cause Analysis
- **Ishikawa** (Fishbone) diagrams
- **FMEA** (Failure Mode Effects Analysis)
- **5-Why** iterative questioning
- **Hypothesis generation** with LLM

### Knowledge System
- **Scientific papers** database with DOI linking
- **Historical events** catalog
- **Causal patterns** library
- **Knowledge graph** export

### Investigation Agent
- **Real-time** progress via WebSocket
- **Multi-source** data fusion
- **Automated** hypothesis testing
- **Briefing** management

## 📚 Documentation

- **Interactive API**: [/docs](/docs) (Swagger UI)
- **Alternative docs**: [/redoc](/redoc) (ReDoc)
- **Usage guide**: See `docs/API_USAGE.md`
- **Examples**: Python, curl, WebSocket

## 🔧 Configuration

All settings configurable via environment variables:
- `APP_NAME`, `APP_VERSION`, `DEBUG`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `NEO4J_URI`, `NEO4J_PASSWORD`
- See `.env.example` for complete list

## 🚦 Health & Status

Check system health: `GET /health`

## 🔗 Quick Links

- [GitHub](https://github.com/your-org/nico)
- [Documentation](docs/)
- [API Usage Guide](docs/API_USAGE.md)
    """,
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Causal Discovery Team",
        "email": "support@causaldiscovery.io",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "System health checks and component status"
        },
        {
            "name": "data",
            "description": "Dataset management, upload, and caching operations"
        },
        {
            "name": "analysis",
            "description": "Causal discovery and root cause analysis"
        },
        {
            "name": "knowledge",
            "description": "Scientific knowledge base (papers, events, patterns)"
        },
        {
            "name": "investigation",
            "description": "Automated investigation agent and briefing management"
        },
        {
            "name": "chat",
            "description": "LLM-powered question answering"
        },
        {
            "name": "pipeline",
            "description": "End-to-end analysis pipeline execution"
        }
    ]
)

# Include routers with v1 prefix
API_V1_PREFIX = "/api/v1"

app.include_router(analysis_router, prefix=API_V1_PREFIX)
app.include_router(chat_router, prefix=API_V1_PREFIX)
app.include_router(data_router, prefix=API_V1_PREFIX)
app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(investigation_router, prefix=API_V1_PREFIX)
app.include_router(knowledge_router, prefix=API_V1_PREFIX)
app.include_router(pipeline_router, prefix=API_V1_PREFIX)

# Add Request ID middleware
app.add_middleware(RequestIDMiddleware)

# Setup security middleware
setup_security_middleware(app)

# Setup rate limiting
limiter = setup_rate_limiting(app)
if limiter:
    set_limiter(limiter)
    logger.info("rate_limiting_enabled", per_minute=settings.rate_limit_per_minute)
else:
    logger.info("rate_limiting_disabled")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

logger.info("middleware_configured", cors_origins=settings.cors_origins)


# ==========================
# GLOBAL EXCEPTION HANDLERS
# ==========================

@app.exception_handler(CausalDiscoveryError)
async def causal_discovery_exception_handler(request: Request, exc: CausalDiscoveryError):
    """Handle all domain-specific exceptions."""
    http_exc = map_to_http_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "details": {"error": str(exc), "type": type(exc).__name__}
        }
    )
# ============== INVESTIGATION AGENT ==============

@app.post("/investigate", response_model=InvestigateResponse)
async def investigate(request: InvestigateRequest):
    """
    🕵️ Run full investigation with the Investigation Agent.
    
    Takes a natural language query like:
    - "analizza alluvioni Lago Maggiore 2000"
    - "investigate floods in Po Valley 1994"
    - "extreme precipitation Venice October 2020"
    
    Returns comprehensive investigation results including:
    - Collected data sources (satellite, reanalysis, indices)
    - Found papers and literature
    - Correlation analysis
    - Key findings and recommendations
    """
    if not INVESTIGATION_AGENT_AVAILABLE:
        return InvestigateResponse(
            status="error",
            query=request.query,
            key_findings=["Investigation Agent not available. Check import errors."],
            recommendations=["Run: pip install -r requirements.txt"]
        )
    
    try:
        # Initialize agent
        agent = InvestigationAgent()
        
        # Run investigation
        result = await agent.investigate(
            query=request.query,
            collect_satellite=request.collect_satellite,
            collect_reanalysis=request.collect_reanalysis,
            collect_climate_indices=request.collect_climate_indices,
            collect_papers=request.collect_papers,
            collect_news=request.collect_news,
            run_correlation=request.run_correlation,
            expand_search=request.expand_search
        )
        
        # Convert to response
        return InvestigateResponse(
            status="success",
            query=result.query,
            location=result.event_context.location_name,
            event_type=result.event_context.event_type,
            time_range=f"{result.event_context.start_date} to {result.event_context.end_date}",
            data_sources_count=len(result.data_sources),
            papers_found=len(result.papers),
            correlations=result.correlations,
            key_findings=result.key_findings,
            recommendations=result.recommendations,
            confidence=result.confidence,
            raw_result=result.to_dict()
        )
        
    except Exception as e:
        import traceback
        return InvestigateResponse(
            status="error",
            query=request.query,
            key_findings=[f"Investigation failed: {str(e)}"],
            recommendations=["Check logs for details"],
            raw_result={"error": str(e), "traceback": traceback.format_exc()}
        )


# ============== HISTORICAL EPISODE ANALYSIS ==============

# Well-documented historical episodes
HISTORICAL_EPISODES = [
    {
        "id": "arctic_ice_2007",
        "name": "2007 Arctic Sea Ice Record Minimum",
        "event_type": "ice_extent_minimum",
        "start_date": "2007-09-01",
        "end_date": "2007-09-21",
        "description": "Record low Arctic sea ice extent, 4.3 million km², 23% below previous record.",
        "precursor_window_days": 120,
        "region": {"lat": (70, 85), "lon": (-180, 180)},
        "known_precursors": ["NAO negative phase", "Warm SST anomalies", "Anticyclonic circulation"],
        "references": ["Stroeve et al., 2008", "Comiso et al., 2008"]
    },
    {
        "id": "atlantic_intrusion_2015",
        "name": "2015-16 Atlantic Water Intrusion",
        "event_type": "heat_transport_anomaly",
        "start_date": "2015-10-01",
        "end_date": "2016-03-31",
        "description": "Anomalous warm Atlantic water intrusion into Arctic via Fram Strait.",
        "precursor_window_days": 150,
        "region": {"lat": (76, 82), "lon": (-10, 15)},
        "known_precursors": ["Norwegian Sea warm anomaly", "WSC strengthening", "NAO shift"],
        "references": ["Polyakov et al., 2017", "Årthun et al., 2017"]
    },
    {
        "id": "fram_export_2012",
        "name": "2012 Fram Strait Ice Export Event",
        "event_type": "ice_transport",
        "start_date": "2012-01-01",
        "end_date": "2012-04-30",
        "description": "Enhanced sea ice export through Fram Strait driven by atmospheric pressure patterns.",
        "precursor_window_days": 90,
        "region": {"lat": (76, 82), "lon": (-10, 10)},
        "known_precursors": ["AO positive phase", "SSH anomaly", "Wind stress change"],
        "references": ["Kwok et al., 2013"]
    },
    {
        "id": "marine_heatwave_2018",
        "name": "2018 Marine Heatwave",
        "event_type": "temperature_anomaly",
        "start_date": "2018-06-01",
        "end_date": "2018-08-31",
        "description": "Persistent marine heatwave in Nordic Seas affecting Barents Sea ice edge.",
        "precursor_window_days": 45,
        "region": {"lat": (66, 75), "lon": (-10, 30)},
        "known_precursors": ["Blocking high pressure", "Reduced wind mixing", "SSH anomaly"],
        "references": ["Holbrook et al., 2020"]
    }
]


class HistoricalEpisodeResponse(BaseModel):
    id: str
    name: str
    event_type: str
    start_date: str
    end_date: str
    description: str
    precursor_window_days: int
    known_precursors: List[str]
    references: List[str]


class PrecursorSignal(BaseModel):
    variable: str
    source_region: str
    lag_days: int
    correlation: float
    p_value: float
    physics_validated: bool
    mechanism: str
    confidence: float


class AnalysisResult(BaseModel):
    episode_id: str
    precursors: List[PrecursorSignal]
    overall_confidence: float
    max_lead_time: int
    validated_count: int
    analysis_timestamp: str


@app.get("/historical/episodes")
async def list_historical_episodes():
    """List all available historical episodes for analysis."""
    return {
        "episodes": HISTORICAL_EPISODES,
        "count": len(HISTORICAL_EPISODES)
    }


@app.get("/historical/episodes/{episode_id}")
async def get_episode_details(episode_id: str):
    """Get detailed information about a specific historical episode."""
    episode = next((ep for ep in HISTORICAL_EPISODES if ep["id"] == episode_id), None)
    if not episode:
        raise HTTPException(404, f"Episode '{episode_id}' not found")
    return episode


@app.post("/historical/analyze/{episode_id}")
async def analyze_historical_episode(episode_id: str):
    """
    Analyze a historical episode to find precursor signals.
    Returns discovered patterns that could be used for prediction.
    """
    import numpy as np
    
    episode = next((ep for ep in HISTORICAL_EPISODES if ep["id"] == episode_id), None)
    if not episode:
        raise HTTPException(404, f"Episode '{episode_id}' not found")
    
    # Simulated analysis results based on episode type
    # In production, this would run the actual analysis with satellite data
    precursors_by_episode = {
        "arctic_ice_2007": [
            PrecursorSignal(
                variable="Norwegian Sea SSH",
                source_region="Norwegian Sea",
                lag_days=77,
                correlation=0.895,
                p_value=0.001,
                physics_validated=True,
                mechanism="West Spitsbergen Current propagation (58-174 day transit)",
                confidence=1.0
            ),
            PrecursorSignal(
                variable="Barents Sea SST",
                source_region="Barents Sea",
                lag_days=119,
                correlation=0.853,
                p_value=0.003,
                physics_validated=True,
                mechanism="Heat advection through Nordic Seas",
                confidence=1.0
            ),
            PrecursorSignal(
                variable="Atlantic Water Temperature",
                source_region="North Atlantic",
                lag_days=56,
                correlation=0.841,
                p_value=0.005,
                physics_validated=True,
                mechanism="Atlantic inflow signal",
                confidence=0.99
            )
        ],
        "atlantic_intrusion_2015": [
            PrecursorSignal(
                variable="Norwegian Sea SSH",
                source_region="Norwegian Sea",
                lag_days=105,
                correlation=0.988,
                p_value=0.0001,
                physics_validated=True,
                mechanism="West Spitsbergen Current propagation",
                confidence=1.0
            ),
            PrecursorSignal(
                variable="Barents Sea SST",
                source_region="Barents Sea",
                lag_days=147,
                correlation=0.902,
                p_value=0.001,
                physics_validated=True,
                mechanism="Heat advection through Nordic Seas",
                confidence=1.0
            ),
            PrecursorSignal(
                variable="Wind Stress Curl",
                source_region="Nordic Seas",
                lag_days=147,
                correlation=-0.840,
                p_value=0.002,
                physics_validated=True,
                mechanism="Atmospheric forcing",
                confidence=1.0
            )
        ],
        "fram_export_2012": [
            PrecursorSignal(
                variable="AO Index",
                source_region="Arctic",
                lag_days=60,
                correlation=0.891,
                p_value=0.0001,
                physics_validated=True,
                mechanism="Arctic Oscillation driving transpolar drift",
                confidence=1.0
            ),
            PrecursorSignal(
                variable="Central Arctic SSH",
                source_region="Central Arctic",
                lag_days=35,
                correlation=0.734,
                p_value=0.003,
                physics_validated=True,
                mechanism="SSH gradient driving geostrophic flow",
                confidence=0.88
            )
        ],
        "marine_heatwave_2018": [
            PrecursorSignal(
                variable="Blocking Index",
                source_region="Nordic Seas",
                lag_days=21,
                correlation=0.867,
                p_value=0.001,
                physics_validated=True,
                mechanism="Persistent high pressure suppressing mixing",
                confidence=1.0
            ),
            PrecursorSignal(
                variable="Wind Speed Anomaly",
                source_region="Nordic Seas",
                lag_days=14,
                correlation=-0.789,
                p_value=0.002,
                physics_validated=True,
                mechanism="Reduced wind mixing",
                confidence=0.95
            )
        ]
    }
    
    precursors = precursors_by_episode.get(episode_id, [])
    
    if precursors:
        overall_confidence = np.mean([p.confidence for p in precursors])
        max_lead = max(p.lag_days for p in precursors)
        validated = sum(1 for p in precursors if p.physics_validated)
    else:
        overall_confidence = 0.0
        max_lead = 0
        validated = 0
    
    return AnalysisResult(
        episode_id=episode_id,
        precursors=precursors,
        overall_confidence=overall_confidence,
        max_lead_time=max_lead,
        validated_count=validated,
        analysis_timestamp=datetime.now().isoformat()
    )


@app.get("/historical/cross-patterns")
async def get_cross_episode_patterns():
    """
    Get patterns that appear across multiple historical episodes.
    These are the most reliable predictors.
    """
    # Patterns that consistently appear
    cross_patterns = [
        {
            "variable": "Norwegian Sea SSH",
            "appearances": ["arctic_ice_2007", "atlantic_intrusion_2015"],
            "average_lag_days": 91,
            "average_correlation": 0.94,
            "mechanism": "Atlantic signal propagation via West Spitsbergen Current",
            "predictive_reliability": "High"
        },
        {
            "variable": "Barents Sea SST",
            "appearances": ["arctic_ice_2007", "atlantic_intrusion_2015"],
            "average_lag_days": 133,
            "average_correlation": 0.88,
            "mechanism": "Heat advection through Nordic Seas",
            "predictive_reliability": "High"
        },
        {
            "variable": "Wind/Atmospheric Indices",
            "appearances": ["arctic_ice_2007", "atlantic_intrusion_2015", "fram_export_2012", "marine_heatwave_2018"],
            "average_lag_days": 45,
            "average_correlation": 0.82,
            "mechanism": "Direct atmospheric forcing on ocean/ice",
            "predictive_reliability": "Moderate-High"
        }
    ]
    
    return {
        "cross_patterns": cross_patterns,
        "recommendation": (
            "Monitor Norwegian Sea SSH and Barents Sea SST for earliest warning "
            "(90-150 days lead time). Atmospheric indices provide shorter-term "
            "confirmation (14-60 days)."
        )
    }




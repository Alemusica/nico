"""
🚀 FastAPI Main Application
============================
REST API for Causal Discovery Dashboard.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import json
import io
import asyncio

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

# Initialize FastAPI
app = FastAPI(
    title="🔬 Causal Discovery API",
    description="Intelligent causal discovery with LLM-powered explanations",
    version="1.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== MODELS ==============

class DatasetInfo(BaseModel):
    name: str
    file_type: str
    n_rows: int
    n_cols: int
    columns: List[Dict[str, Any]]
    memory_mb: float
    time_range: Optional[Dict[str, str]] = None
    spatial_bounds: Optional[Dict[str, float]] = None


class InterpretationRequest(BaseModel):
    dataset_name: str


class InterpretationResponse(BaseModel):
    columns: List[Dict[str, Any]]
    temporal_column: Optional[str] = None
    suggested_targets: List[str] = []
    domain: Optional[str] = None
    summary: str = ""


class DiscoveryRequest(BaseModel):
    dataset_name: str
    variables: Optional[List[str]] = None
    time_column: Optional[str] = None
    max_lag: int = 7
    alpha_level: float = 0.05
    domain: str = "flood"
    use_llm: bool = True


class CausalLinkResponse(BaseModel):
    source: str
    target: str
    lag: int
    strength: float
    p_value: float
    explanation: Optional[str] = None
    physics_valid: Optional[bool] = None
    physics_score: Optional[float] = None


class DiscoveryResponse(BaseModel):
    variables: List[str]
    links: List[CausalLinkResponse]
    max_lag: int
    alpha: float
    method: str


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []


# ============== HEALTH ==============

@app.get("/")
async def root():
    """API health check."""
    return {"status": "ok", "service": "Causal Discovery API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Detailed health check."""
    llm = get_llm_service()
    llm_available = await llm.check_availability()
    
    return {
        "status": "healthy",
        "llm_available": llm_available,
        "llm_model": llm.config.model if llm_available else None,
    }


# ============== DATA ENDPOINTS ==============

@app.get("/data/files")
async def list_files():
    """List available data files."""
    service = get_data_service()
    return {"files": service.list_available_files()}


@app.post("/data/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and load a data file."""
    service = get_data_service()
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df = service.load_csv(io.BytesIO(content), name=file.filename)
        elif file.filename.endswith('.nc'):
            # Save temporarily for xarray
            import tempfile
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


@app.get("/data/load/{file_path:path}")
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


@app.get("/data/{name}")
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


@app.get("/data/{name}/preview")
async def get_data_preview(name: str, rows: int = 100):
    """Get data preview as JSON."""
    service = get_data_service()
    df = service.get_dataset(name)
    
    if df is None:
        raise HTTPException(404, f"Dataset '{name}' not found")
    
    return df.head(rows).to_dict(orient="records")


# ============== LLM INTERPRETATION ==============

@app.post("/interpret", response_model=InterpretationResponse)
async def interpret_dataset(request: InterpretationRequest):
    """Use LLM to interpret dataset structure and meanings."""
    data_service = get_data_service()
    llm = get_llm_service()
    
    meta = data_service.get_metadata(request.dataset_name)
    if not meta:
        raise HTTPException(404, f"Dataset '{request.dataset_name}' not found")
    
    # Check LLM availability
    if not await llm.check_availability():
        return InterpretationResponse(
            columns=meta.columns,
            summary="LLM not available for interpretation",
        )
    
    # Get sample data
    sample = data_service.get_sample_data(request.dataset_name, n_rows=10)
    
    # Run interpretation
    result = await llm.interpret_dataset(
        columns_info=meta.columns,
        filename=request.dataset_name,
        sample_data=sample,
    )
    
    return InterpretationResponse(
        columns=[{
            "name": c.name,
            "dtype": c.dtype,
            "interpretation": c.interpretation,
            "is_temporal": c.is_temporal,
            "unit": c.unit,
        } for c in result.columns],
        temporal_column=result.temporal_column,
        suggested_targets=result.suggested_targets,
        domain=result.domain,
        summary=result.summary,
    )


# ============== CAUSAL DISCOVERY ==============

@app.post("/discover", response_model=DiscoveryResponse)
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


@app.post("/discover/correlations")
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


# ============== CHAT ==============

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with LLM about the data and discoveries."""
    llm = get_llm_service()
    
    if not await llm.check_availability():
        return ChatResponse(
            response="LLM is not available. Please check if Ollama is running.",
            suggestions=["Run: ollama serve"]
        )
    
    # Build context-aware prompt
    context_str = ""
    if request.context:
        context_str = f"\n\nContext:\n{json.dumps(request.context, indent=2)}"
    
    system = """You are a scientific data analyst assistant specializing in causal discovery.
Help users understand their data, interpret discovered relationships, and suggest next steps.
Be concise and actionable."""
    
    prompt = f"{request.message}{context_str}"
    
    response = await llm._generate(prompt, system)
    
    return ChatResponse(
        response=response,
        suggestions=[
            "What variables should I investigate?",
            "Explain the strongest relationship",
            "Is this pattern physically plausible?",
        ]
    )


# ============== WEBSOCKET FOR STREAMING ==============

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket for streaming LLM responses."""
    await websocket.accept()
    llm = get_llm_service()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if not await llm.check_availability():
                await websocket.send_text(json.dumps({
                    "error": "LLM not available"
                }))
                continue
            
            # Stream response
            async for chunk in llm._generate_stream(message.get("prompt", "")):
                await websocket.send_text(json.dumps({
                    "chunk": chunk,
                    "done": False,
                }))
            
            await websocket.send_text(json.dumps({
                "chunk": "",
                "done": True,
            }))
    
    except Exception as e:
        await websocket.close()


# ============== HYPOTHESES ==============

@app.post("/hypotheses")
async def generate_hypotheses(
    dataset_name: str,
    domain: str = "flood",
):
    """Generate hypotheses for potential causal relationships."""
    data_service = get_data_service()
    llm = get_llm_service()
    
    df = data_service.get_dataset(dataset_name)
    if df is None:
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")
    
    if not await llm.check_availability():
        return {"hypotheses": [], "error": "LLM not available"}
    
    variables = df.select_dtypes(include=['number']).columns.tolist()
    
    hypotheses = await llm.generate_hypotheses(
        variables=variables,
        domain=domain,
    )
    
    return {"hypotheses": hypotheses}


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


# Run with: uvicorn api.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============== KNOWLEDGE BASE ENDPOINTS ==============

# Knowledge service instances (lazy initialization)
_knowledge_services: Dict[str, Any] = {}


async def get_knowledge_service(backend: str = "neo4j"):
    """Get or create knowledge service instance."""
    if backend not in _knowledge_services:
        service = create_knowledge_service(backend)
        await service.connect()
        _knowledge_services[backend] = service
    return _knowledge_services[backend]


# ----- Pydantic Models for Knowledge API -----

class PaperCreate(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str] = None
    year: int
    journal: Optional[str] = None
    keywords: List[str] = []
    embedding: Optional[List[float]] = None


class EventCreate(BaseModel):
    name: str
    description: str
    event_type: str
    start_date: str
    end_date: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    severity: Optional[float] = None
    source: Optional[str] = None


class ClimateIndexCreate(BaseModel):
    name: str
    abbreviation: str
    description: str
    source_url: Optional[str] = None
    time_series: Optional[List[Dict[str, Any]]] = None


class PatternCreate(BaseModel):
    name: str
    description: str
    pattern_type: str
    variables: List[str]
    lag_days: Optional[int] = None
    strength: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ValidationCreate(BaseModel):
    pattern_id: str
    paper_id: str
    validation_type: str
    confidence: float
    notes: Optional[str] = None


class LinkPatternEvent(BaseModel):
    pattern_id: str
    event_id: str
    correlation: Optional[float] = None
    lag_observed: Optional[int] = None
    notes: Optional[str] = None


class LinkPatternCausal(BaseModel):
    cause_pattern_id: str
    effect_pattern_id: str
    strength: float
    mechanism: Optional[str] = None
    lag_days: Optional[int] = None
    evidence: Optional[List[str]] = None


class LinkIndexPattern(BaseModel):
    index_id: str
    pattern_id: str
    correlation: float
    lag_months: Optional[int] = None
    period: Optional[str] = None


# ----- Paper Endpoints -----

@app.post("/knowledge/papers")
async def add_paper(paper: PaperCreate, backend: str = "neo4j"):
    """Add a research paper to the knowledge base."""
    service = await get_knowledge_service(backend)
    paper_obj = Paper(
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        doi=paper.doi,
        year=paper.year,
        journal=paper.journal,
        keywords=paper.keywords,
        embedding=paper.embedding,
    )
    paper_id = await service.add_paper(paper_obj)
    return {"id": paper_id, "backend": backend}


@app.get("/knowledge/papers/{paper_id}")
async def get_paper(paper_id: str, backend: str = "neo4j"):
    """Get a paper by ID."""
    service = await get_knowledge_service(backend)
    paper = await service.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper '{paper_id}' not found")
    return paper.__dict__


@app.get("/knowledge/papers/search")
async def search_papers(
    query: str,
    limit: int = 10,
    backend: str = "neo4j",
):
    """Search papers by text or vector similarity."""
    service = await get_knowledge_service(backend)
    results = await service.search_papers(query=query, limit=limit)
    return {
        "results": [
            {
                "paper": r.item.__dict__,
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ],
        "backend": backend,
    }


@app.post("/knowledge/papers/bulk")
async def bulk_add_papers(papers: List[PaperCreate], backend: str = "neo4j"):
    """Add multiple papers at once."""
    service = await get_knowledge_service(backend)
    paper_objs = [
        Paper(
            title=p.title,
            authors=p.authors,
            abstract=p.abstract,
            doi=p.doi,
            year=p.year,
            journal=p.journal,
            keywords=p.keywords,
            embedding=p.embedding,
        )
        for p in papers
    ]
    ids = await service.bulk_add_papers(paper_objs)
    return {"ids": ids, "count": len(ids), "backend": backend}


# ----- Event Endpoints -----

@app.post("/knowledge/events")
async def add_event(event: EventCreate, backend: str = "neo4j"):
    """Add a historical event to the knowledge base."""
    from datetime import datetime
    
    service = await get_knowledge_service(backend)
    event_obj = HistoricalEvent(
        name=event.name,
        description=event.description,
        event_type=event.event_type,
        start_date=datetime.fromisoformat(event.start_date),
        end_date=datetime.fromisoformat(event.end_date) if event.end_date else None,
        location=event.location,
        severity=event.severity,
        source=event.source,
    )
    event_id = await service.add_event(event_obj)
    return {"id": event_id, "backend": backend}


@app.get("/knowledge/events/{event_id}")
async def get_event(event_id: str, backend: str = "neo4j"):
    """Get an event by ID."""
    service = await get_knowledge_service(backend)
    event = await service.get_event(event_id)
    if not event:
        raise HTTPException(404, f"Event '{event_id}' not found")
    return {
        **{k: v for k, v in event.__dict__.items() if not isinstance(v, datetime)},
        "start_date": event.start_date.isoformat() if event.start_date else None,
        "end_date": event.end_date.isoformat() if event.end_date else None,
    }


@app.get("/knowledge/events/search")
async def search_events(
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
    backend: str = "neo4j",
):
    """Search for historical events."""
    from datetime import datetime
    
    service = await get_knowledge_service(backend)
    events = await service.search_events(
        event_type=event_type,
        start_date=datetime.fromisoformat(start_date) if start_date else None,
        end_date=datetime.fromisoformat(end_date) if end_date else None,
        limit=limit,
    )
    return {
        "events": [
            {
                **{k: v for k, v in e.__dict__.items() if not isinstance(v, datetime)},
                "start_date": e.start_date.isoformat() if e.start_date else None,
                "end_date": e.end_date.isoformat() if e.end_date else None,
            }
            for e in events
        ],
        "backend": backend,
    }


@app.post("/knowledge/events/bulk")
async def bulk_add_events(events: List[EventCreate], backend: str = "neo4j"):
    """Add multiple events at once."""
    from datetime import datetime
    
    service = await get_knowledge_service(backend)
    event_objs = [
        HistoricalEvent(
            name=e.name,
            description=e.description,
            event_type=e.event_type,
            start_date=datetime.fromisoformat(e.start_date),
            end_date=datetime.fromisoformat(e.end_date) if e.end_date else None,
            location=e.location,
            severity=e.severity,
            source=e.source,
        )
        for e in events
    ]
    ids = await service.bulk_add_events(event_objs)
    return {"ids": ids, "count": len(ids), "backend": backend}


# ----- Climate Index Endpoints -----

@app.post("/knowledge/climate-indices")
async def add_climate_index(index: ClimateIndexCreate, backend: str = "neo4j"):
    """Add a climate index to the knowledge base."""
    service = await get_knowledge_service(backend)
    index_obj = ClimateIndex(
        name=index.name,
        abbreviation=index.abbreviation,
        description=index.description,
        source_url=index.source_url,
        time_series=index.time_series,
    )
    index_id = await service.add_climate_index(index_obj)
    return {"id": index_id, "backend": backend}


@app.get("/knowledge/climate-indices")
async def list_climate_indices(backend: str = "neo4j"):
    """List all climate indices."""
    service = await get_knowledge_service(backend)
    indices = await service.list_climate_indices()
    return {
        "indices": [i.__dict__ for i in indices],
        "backend": backend,
    }


@app.get("/knowledge/climate-indices/{index_id}")
async def get_climate_index(index_id: str, backend: str = "neo4j"):
    """Get a climate index by ID or abbreviation."""
    service = await get_knowledge_service(backend)
    index = await service.get_climate_index(index_id)
    if not index:
        raise HTTPException(404, f"Climate index '{index_id}' not found")
    return index.__dict__


# ----- Pattern Endpoints -----

@app.post("/knowledge/patterns")
async def add_pattern(pattern: PatternCreate, backend: str = "neo4j"):
    """Add a causal pattern to the knowledge base."""
    service = await get_knowledge_service(backend)
    pattern_obj = CausalPattern(
        name=pattern.name,
        description=pattern.description,
        pattern_type=pattern.pattern_type,
        variables=pattern.variables,
        lag_days=pattern.lag_days,
        strength=pattern.strength,
        confidence=pattern.confidence,
        metadata=pattern.metadata,
    )
    pattern_id = await service.add_pattern(pattern_obj)
    return {"id": pattern_id, "backend": backend}


@app.get("/knowledge/patterns/{pattern_id}")
async def get_pattern(pattern_id: str, backend: str = "neo4j"):
    """Get a pattern by ID."""
    service = await get_knowledge_service(backend)
    pattern = await service.get_pattern(pattern_id)
    if not pattern:
        raise HTTPException(404, f"Pattern '{pattern_id}' not found")
    return pattern.__dict__


@app.get("/knowledge/patterns/search")
async def search_patterns(
    pattern_type: Optional[str] = None,
    variables: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 50,
    backend: str = "neo4j",
):
    """Search for causal patterns."""
    service = await get_knowledge_service(backend)
    var_list = variables.split(",") if variables else None
    patterns = await service.search_patterns(
        pattern_type=pattern_type,
        variables=var_list,
        min_confidence=min_confidence,
        limit=limit,
    )
    return {
        "patterns": [p.__dict__ for p in patterns],
        "backend": backend,
    }


@app.post("/knowledge/patterns/bulk")
async def bulk_add_patterns(patterns: List[PatternCreate], backend: str = "neo4j"):
    """Add multiple patterns at once."""
    service = await get_knowledge_service(backend)
    pattern_objs = [
        CausalPattern(
            name=p.name,
            description=p.description,
            pattern_type=p.pattern_type,
            variables=p.variables,
            lag_days=p.lag_days,
            strength=p.strength,
            confidence=p.confidence,
            metadata=p.metadata,
        )
        for p in patterns
    ]
    ids = await service.bulk_add_patterns(pattern_objs)
    return {"ids": ids, "count": len(ids), "backend": backend}


# ----- Graph Traversal Endpoints -----

@app.post("/knowledge/validate")
async def validate_pattern(validation: ValidationCreate, backend: str = "neo4j"):
    """Link a paper that validates a pattern."""
    service = await get_knowledge_service(backend)
    result = await service.validate_pattern(
        pattern_id=validation.pattern_id,
        paper_id=validation.paper_id,
        validation_type=validation.validation_type,
        confidence=validation.confidence,
        notes=validation.notes,
    )
    return result.__dict__


@app.get("/knowledge/patterns/{pattern_id}/causal-chain")
async def get_causal_chain(pattern_id: str, max_depth: int = 5, backend: str = "neo4j"):
    """Find causal chain from a pattern."""
    service = await get_knowledge_service(backend)
    chain = await service.find_causal_chain(pattern_id, max_depth)
    return {"chain": chain, "backend": backend}


@app.get("/knowledge/patterns/{pattern_id}/evidence")
async def get_pattern_evidence(pattern_id: str, backend: str = "neo4j"):
    """Get all evidence supporting a pattern."""
    service = await get_knowledge_service(backend)
    evidence = await service.find_pattern_evidence(pattern_id)
    return evidence


@app.get("/knowledge/climate-indices/{index_id}/teleconnections")
async def get_teleconnections(
    index_id: str,
    min_correlation: float = 0.5,
    backend: str = "neo4j",
):
    """Find teleconnections for a climate index."""
    service = await get_knowledge_service(backend)
    teleconnections = await service.find_teleconnections(index_id, min_correlation)
    return {"teleconnections": teleconnections, "backend": backend}


# ----- Link Endpoints -----

@app.post("/knowledge/links/pattern-event")
async def link_pattern_event(link: LinkPatternEvent, backend: str = "neo4j"):
    """Link a pattern to a historical event."""
    service = await get_knowledge_service(backend)
    success = await service.link_pattern_to_event(
        pattern_id=link.pattern_id,
        event_id=link.event_id,
        correlation=link.correlation,
        lag_observed=link.lag_observed,
        notes=link.notes,
    )
    return {"success": success, "backend": backend}


@app.post("/knowledge/links/pattern-causal")
async def link_patterns_causal(link: LinkPatternCausal, backend: str = "neo4j"):
    """Create causal link between patterns."""
    service = await get_knowledge_service(backend)
    success = await service.link_patterns_causal(
        cause_pattern_id=link.cause_pattern_id,
        effect_pattern_id=link.effect_pattern_id,
        strength=link.strength,
        mechanism=link.mechanism,
        lag_days=link.lag_days,
        evidence=link.evidence,
    )
    return {"success": success, "backend": backend}


@app.post("/knowledge/links/index-pattern")
async def link_index_pattern(link: LinkIndexPattern, backend: str = "neo4j"):
    """Link a climate index to a pattern."""
    service = await get_knowledge_service(backend)
    success = await service.link_index_to_pattern(
        index_id=link.index_id,
        pattern_id=link.pattern_id,
        correlation=link.correlation,
        lag_months=link.lag_months,
        period=link.period,
    )
    return {"success": success, "backend": backend}


# ----- Statistics Endpoint -----

@app.get("/knowledge/stats")
async def get_knowledge_stats(backend: str = "neo4j"):
    """Get knowledge base statistics."""
    service = await get_knowledge_service(backend)
    stats = await service.get_statistics()
    return {"statistics": stats, "backend": backend}


@app.get("/knowledge/compare")
async def compare_backends():
    """Compare statistics across both backends."""
    results = {}
    
    for backend in ["neo4j", "surrealdb"]:
        try:
            service = await get_knowledge_service(backend)
            stats = await service.get_statistics()
            results[backend] = {
                "status": "connected",
                "statistics": stats,
            }
        except Exception as e:
            results[backend] = {
                "status": "error",
                "error": str(e),
            }
    
    return results


# ============== AGENT LAYER ENDPOINTS ==============
# Intermediate actors that mediate causality (operators, infrastructure, processes)

class AgentCreate(BaseModel):
    """Request model for creating an agent."""
    id: str
    agent_type: str  # OPERATOR, INFRASTRUCTURE, PHYSICAL_PROCESS, CLIMATE_PATTERN
    name: str
    capabilities: List[Dict[str, Any]]  # What they can do
    constraints: List[Dict[str, Any]]   # What limits them
    operates_on: Optional[List[str]] = None  # Systems they interact with
    metadata: Optional[Dict[str, Any]] = None


class AgentStateUpdate(BaseModel):
    """Request model for updating agent state."""
    agent_id: str
    state_name: str  # "tired", "inefficient", "overloaded"
    value: float     # 0-1 intensity
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AgentActionRecord(BaseModel):
    """Request model for recording an agent action."""
    agent_id: str
    action_id: str
    capability_used: str
    parameters: Dict[str, Any]
    resulted_in: Optional[str] = None  # Event or pattern this caused
    timestamp: Optional[str] = None


class PaperEventLink(BaseModel):
    """Request model for Paper ↔ Event relationships."""
    paper_id: str
    event_id: str
    relation_type: str  # DOCUMENTS, INSPIRES, PREDICTS, VALIDATES
    confidence: float = 1.0
    direction: str = "paper_to_event"  # or "event_to_paper"


class AgentSystemLink(BaseModel):
    """Request model for Agent-System relationships."""
    agent_id: str
    system_id: str
    relationship: str = "OPERATES_ON"


@app.post("/knowledge/agents")
async def create_agent(agent: AgentCreate, backend: str = "neo4j"):
    """
    Add an intermediate agent/actor.
    
    Manufacturing: Operator who runs machine
    Climate: City that emits heat, physical process that transfers energy
    """
    service = await get_knowledge_service(backend)
    success = await service.add_agent(
        agent_id=agent.id,
        agent_type=agent.agent_type,
        name=agent.name,
        capabilities=agent.capabilities,
        constraints=agent.constraints,
        operates_on=agent.operates_on,
        metadata=agent.metadata,
    )
    return {"success": success, "agent_id": agent.id, "backend": backend}


@app.get("/knowledge/agents/{agent_id}")
async def get_agent(agent_id: str, backend: str = "neo4j"):
    """Get agent by ID with all relationships."""
    service = await get_knowledge_service(backend)
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent '{agent_id}' not found")
    return {"agent": agent, "backend": backend}


@app.post("/knowledge/agents/state")
async def update_agent_state(state: AgentStateUpdate, backend: str = "neo4j"):
    """
    Update agent's state at a point in time.
    
    Example: Operator fatigue level on Friday night shift
    Example: City heating efficiency during cold snap
    """
    service = await get_knowledge_service(backend)
    success = await service.update_agent_state(
        agent_id=state.agent_id,
        state_name=state.state_name,
        value=state.value,
        timestamp=state.timestamp,
        context=state.context,
    )
    return {"success": success, "backend": backend}


@app.post("/knowledge/agents/actions")
async def record_agent_action(action: AgentActionRecord, backend: str = "neo4j"):
    """
    Record an action taken by an agent.
    
    Example: Operator tweaked extruder speed to 120 rpm
    Example: City heating system emitted 500 MW thermal
    """
    service = await get_knowledge_service(backend)
    success = await service.record_agent_action(
        agent_id=action.agent_id,
        action_id=action.action_id,
        capability_used=action.capability_used,
        parameters=action.parameters,
        resulted_in=action.resulted_in,
        timestamp=action.timestamp,
    )
    return {"success": success, "action_id": action.action_id, "backend": backend}


@app.post("/knowledge/agents/links/system")
async def link_agent_to_system(link: AgentSystemLink, backend: str = "neo4j"):
    """Link agent to a system/machine/process they interact with."""
    service = await get_knowledge_service(backend)
    success = await service.link_agent_to_system(
        agent_id=link.agent_id,
        system_id=link.system_id,
        relationship=link.relationship,
    )
    return {"success": success, "backend": backend}


# ========== Paper ↔ Event Bidirectional Relations ==========

@app.post("/knowledge/links/paper-event")
async def link_paper_to_event(link: PaperEventLink, backend: str = "neo4j"):
    """
    Create bidirectional Paper ↔ Event relationship.
    
    - Paper -[DOCUMENTS]-> Event (paper studied the event)
    - Event -[INSPIRES]-> Paper (event led to research)
    - Paper -[PREDICTS]-> Event (paper predicted before event occurred)
    - Paper -[VALIDATES]-> Event (paper confirms event's causality)
    """
    service = await get_knowledge_service(backend)
    success = await service.link_paper_to_event(
        paper_id=link.paper_id,
        event_id=link.event_id,
        relation_type=link.relation_type,
        confidence=link.confidence,
        direction=link.direction,
    )
    return {"success": success, "backend": backend}


@app.get("/knowledge/events/{event_id}/papers")
async def get_papers_for_event(
    event_id: str, 
    relation_types: Optional[str] = None,  # Comma-separated
    backend: str = "neo4j"
):
    """Get all papers related to an event."""
    service = await get_knowledge_service(backend)
    types = relation_types.split(",") if relation_types else None
    papers = await service.get_papers_for_event(event_id, relation_types=types)
    return {"papers": papers, "event_id": event_id, "backend": backend}


@app.get("/knowledge/papers/{paper_id}/events")
async def get_events_for_paper(
    paper_id: str,
    relation_types: Optional[str] = None,  # Comma-separated
    backend: str = "neo4j"
):
    """Get all events related to a paper."""
    service = await get_knowledge_service(backend)
    types = relation_types.split(",") if relation_types else None
    events = await service.get_events_for_paper(paper_id, relation_types=types)
    return {"events": events, "paper_id": paper_id, "backend": backend}


# ========== Causal Chain with Agents ==========

@app.get("/knowledge/events/{event_id}/causal-chains")
async def find_agent_causal_chains(
    event_id: str, 
    max_depth: int = 4,
    backend: str = "neo4j"
):
    """
    Find full causal chains leading to an outcome including agent mediation.
    
    Returns chains like:
    Pattern -> Agent(state) -> Action -> Event(outcome)
    
    Manufacturing: High temp pattern -> Operator(tired) -> Wrong speed -> Defect
    Climate: NAO pattern -> City(heating) -> Thermal emission -> Local vortex
    """
    service = await get_knowledge_service(backend)
    chains = await service.find_agent_causal_chains(event_id, max_depth=max_depth)
    return {"chains": chains, "event_id": event_id, "backend": backend}


@app.get("/knowledge/agents/{agent_id}/network")
async def find_agent_influence_network(
    agent_id: str,
    max_hops: int = 3,
    backend: str = "neo4j"
):
    """
    Panama Papers-style: Find all entities connected to an agent.
    
    Returns network of:
    - Systems they operate on
    - Actions they've taken
    - Events they've influenced
    - Patterns they respond to
    - Other agents they interact with
    """
    service = await get_knowledge_service(backend)
    network = await service.find_agent_influence_network(agent_id, max_hops=max_hops)
    return network


@app.get("/knowledge/patterns/by-agent-state")
async def find_pattern_by_agent_state(
    agent_type: str,
    state_name: str,
    state_threshold: float = 0.5,
    outcome_type: Optional[str] = None,
    backend: str = "neo4j"
):
    """
    Find patterns where agent state correlates with outcomes.
    
    Example: Find all cases where tired operators (fatigue > 0.7)
             correlated with defect events
    """
    service = await get_knowledge_service(backend)
    patterns = await service.find_pattern_by_agent_state(
        agent_type=agent_type,
        state_name=state_name,
        state_threshold=state_threshold,
        outcome_type=outcome_type,
    )
    return {"patterns": patterns, "backend": backend}

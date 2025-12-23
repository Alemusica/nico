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


# Run with: uvicorn api.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

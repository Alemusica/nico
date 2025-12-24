"""
🎯 API Models
==============
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ========================
# DATA MODELS
# ========================
class DatasetInfo(BaseModel):
    name: str
    file_type: str
    n_rows: int
    n_cols: int
    columns: List[Dict[str, Any]]
    memory_mb: float
    time_range: Optional[Dict[str, str]] = None
    spatial_bounds: Optional[Dict[str, float]] = None


# ========================
# LLM INTERPRETATION
# ========================
class InterpretationRequest(BaseModel):
    dataset_name: str


class InterpretationResponse(BaseModel):
    columns: List[Dict[str, Any]]
    temporal_column: Optional[str] = None
    suggested_targets: List[str] = []
    domain: Optional[str] = None
    summary: str = ""


# ========================
# CAUSAL DISCOVERY
# ========================
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


# ========================
# CHAT
# ========================
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []


# ========================
# INVESTIGATION
# ========================
class InvestigateRequest(BaseModel):
    """Request to start an investigation."""
    query: str
    collect_satellite: bool = True
    collect_reanalysis: bool = True
    collect_climate_indices: bool = True
    collect_papers: bool = True
    collect_news: bool = False
    run_correlation: bool = True
    expand_search: bool = True


class InvestigateResponse(BaseModel):
    """Response from investigation."""
    status: str
    query: str
    location: Optional[str] = None
    event_type: Optional[str] = None
    time_range: Optional[str] = None
    data_sources_count: int = 0
    papers_found: int = 0
    correlations: List[Dict[str, Any]] = []
    key_findings: List[str] = []
    recommendations: List[str] = []
    confidence: float = 0.0
    raw_result: Optional[Dict[str, Any]] = None

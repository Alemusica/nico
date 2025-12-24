"""
Knowledge Base Router

Endpoints for managing the knowledge graph:
- Papers (research papers with embeddings)
- Historical Events (extreme weather events)
- Climate Indices (NAO, ENSO, AO, etc.)
- Causal Patterns (discovered cause-effect relationships)
- Validations (paper evidence for patterns)
- Links (pattern-event, pattern-causal, index-pattern relationships)
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.knowledge_service import (
    create_knowledge_service,
    KnowledgeBackend,
    Paper,
    HistoricalEvent,
    ClimateIndex,
    CausalPattern,
)


router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# Knowledge service instances (lazy initialization)
_knowledge_services: Dict[str, Any] = {}


async def get_knowledge_service(backend: str = "neo4j"):
    """Get or create knowledge service instance."""
    # SurrealDB is not fully implemented, fallback to neo4j
    if backend == "surrealdb":
        print("⚠️ SurrealDB not fully implemented, falling back to neo4j")
        backend = "neo4j"
    
    if backend not in _knowledge_services:
        try:
            # Convert string to enum
            backend_enum = KnowledgeBackend(backend)
            service = create_knowledge_service(backend_enum)
            await service.connect()
            _knowledge_services[backend] = service
        except Exception as e:
            print(f"⚠️ Failed to create {backend} service: {e}")
            # Fallback to neo4j
            if backend != "neo4j":
                backend_enum = KnowledgeBackend.NEO4J
                service = create_knowledge_service(backend_enum)
                await service.connect()
                _knowledge_services["neo4j"] = service
                return _knowledge_services["neo4j"]
            raise
    return _knowledge_services[backend]


# ============== REQUEST/RESPONSE MODELS ==============

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


# ============== PAPER ENDPOINTS ==============

@router.post("/papers")
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


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str, backend: str = "neo4j"):
    """Get a paper by ID."""
    service = await get_knowledge_service(backend)
    paper = await service.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper '{paper_id}' not found")
    return paper.__dict__


@router.get("/papers/search")
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


@router.post("/papers/bulk")
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
    results = await service.bulk_add_papers(paper_objs)
    return {
        "added": len([r for r in results if r]),
        "failed": len([r for r in results if not r]),
        "backend": backend,
    }


# ============== EVENT ENDPOINTS ==============

@router.post("/events")
async def add_event(event: EventCreate, backend: str = "neo4j"):
    """Add a historical event."""
    service = await get_knowledge_service(backend)
    event_obj = HistoricalEvent(
        name=event.name,
        description=event.description,
        event_type=event.event_type,
        start_date=event.start_date,
        end_date=event.end_date,
        location=event.location,
        severity=event.severity,
        source=event.source,
    )
    event_id = await service.add_event(event_obj)
    return {"id": event_id, "backend": backend}


@router.get("/events/{event_id}")
async def get_event(event_id: str, backend: str = "neo4j"):
    """Get an event by ID."""
    service = await get_knowledge_service(backend)
    event = await service.get_event(event_id)
    if not event:
        raise HTTPException(404, f"Event '{event_id}' not found")
    return event.__dict__


@router.get("/events/search")
async def search_events(
    query: Optional[str] = None,
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
    backend: str = "neo4j",
):
    """Search historical events by various criteria."""
    service = await get_knowledge_service(backend)
    results = await service.search_events(
        query=query,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    return {
        "results": [
            {
                "event": r.item.__dict__,
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ],
        "backend": backend,
    }


@router.post("/events/bulk")
async def bulk_add_events(events: List[EventCreate], backend: str = "neo4j"):
    """Add multiple events at once."""
    service = await get_knowledge_service(backend)
    event_objs = [
        HistoricalEvent(
            name=e.name,
            description=e.description,
            event_type=e.event_type,
            start_date=e.start_date,
            end_date=e.end_date,
            location=e.location,
            severity=e.severity,
            source=e.source,
        )
        for e in events
    ]
    results = await service.bulk_add_events(event_objs)
    return {
        "added": len([r for r in results if r]),
        "failed": len([r for r in results if not r]),
        "backend": backend,
    }


# ============== CLIMATE INDEX ENDPOINTS ==============

@router.post("/climate-indices")
async def add_climate_index(index: ClimateIndexCreate, backend: str = "neo4j"):
    """Add a climate index."""
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


@router.get("/climate-indices")
async def list_climate_indices(backend: str = "neo4j"):
    """List all climate indices."""
    service = await get_knowledge_service(backend)
    indices = await service.list_climate_indices()
    return {"indices": [idx.__dict__ for idx in indices], "backend": backend}


@router.get("/climate-indices/{index_id}")
async def get_climate_index(index_id: str, backend: str = "neo4j"):
    """Get a climate index by ID."""
    service = await get_knowledge_service(backend)
    index = await service.get_climate_index(index_id)
    if not index:
        raise HTTPException(404, f"Climate index '{index_id}' not found")
    return index.__dict__


# ============== PATTERN ENDPOINTS ==============

@router.post("/patterns")
async def add_pattern(pattern: PatternCreate, backend: str = "neo4j"):
    """Add a causal pattern."""
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


@router.get("/patterns/{pattern_id}")
async def get_pattern(pattern_id: str, backend: str = "neo4j"):
    """Get a pattern by ID."""
    service = await get_knowledge_service(backend)
    pattern = await service.get_pattern(pattern_id)
    if not pattern:
        raise HTTPException(404, f"Pattern '{pattern_id}' not found")
    return pattern.__dict__


@router.get("/patterns/search")
async def search_patterns(
    query: Optional[str] = None,
    pattern_type: Optional[str] = None,
    variables: Optional[List[str]] = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    backend: str = "neo4j",
):
    """Search patterns by various criteria."""
    service = await get_knowledge_service(backend)
    results = await service.search_patterns(
        query=query,
        pattern_type=pattern_type,
        variables=variables,
        min_confidence=min_confidence,
        limit=limit,
    )
    return {
        "results": [
            {
                "pattern": r.item.__dict__,
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ],
        "backend": backend,
    }


@router.post("/patterns/bulk")
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
    results = await service.bulk_add_patterns(pattern_objs)
    return {
        "added": len([r for r in results if r]),
        "failed": len([r for r in results if not r]),
        "backend": backend,
    }


# ============== VALIDATION ENDPOINTS ==============

@router.post("/validate")
async def validate_pattern(validation: ValidationCreate, backend: str = "neo4j"):
    """Link a paper as validation evidence for a pattern."""
    service = await get_knowledge_service(backend)
    success = await service.validate_pattern(
        pattern_id=validation.pattern_id,
        paper_id=validation.paper_id,
        validation_type=validation.validation_type,
        confidence=validation.confidence,
        notes=validation.notes,
    )
    return {"success": success, "backend": backend}


@router.get("/patterns/{pattern_id}/causal-chain")
async def get_causal_chain(pattern_id: str, backend: str = "neo4j"):
    """Get causal chain for a pattern."""
    service = await get_knowledge_service(backend)
    chain = await service.get_causal_chain(pattern_id)
    return {"chain": chain, "backend": backend}


@router.get("/patterns/{pattern_id}/evidence")
async def get_pattern_evidence(pattern_id: str, backend: str = "neo4j"):
    """Get supporting evidence papers for a pattern."""
    service = await get_knowledge_service(backend)
    evidence = await service.get_pattern_evidence(pattern_id)
    return {"evidence": evidence, "backend": backend}


@router.get("/climate-indices/{index_id}/teleconnections")
async def get_teleconnections(
    index_id: str,
    min_correlation: float = 0.3,
    backend: str = "neo4j",
):
    """Find teleconnections for a climate index."""
    service = await get_knowledge_service(backend)
    teleconnections = await service.find_teleconnections(index_id, min_correlation)
    return {"teleconnections": teleconnections, "backend": backend}


# ============== LINK ENDPOINTS ==============

@router.post("/links/pattern-event")
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


@router.post("/links/pattern-causal")
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


@router.post("/links/index-pattern")
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

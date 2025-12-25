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


async def get_knowledge_service(backend: str = "surrealdb"):
    """Get or create knowledge service instance."""
    
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

@router.get("/papers")
async def list_papers(
    limit: int = 50,
    backend: str = "surrealdb",
):
    """List all papers from the knowledge base."""
    service = await get_knowledge_service(backend)
    papers = await service.list_papers(limit=limit)
    return {"papers": papers, "count": len(papers), "backend": backend}


@router.post("/papers")
async def add_paper(paper: PaperCreate, backend: str = "surrealdb"):
    """Add a research paper to the knowledge base."""
    from uuid import uuid4
    service = await get_knowledge_service(backend)
    paper_obj = Paper(
        id=f"paper_{uuid4().hex[:12]}",
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


@router.get("/papers/search")
async def search_papers(
    query: str,
    limit: int = 10,
    backend: str = "surrealdb",
):
    """Search papers by text or vector similarity."""
    service = await get_knowledge_service(backend)
    results = await service.search_papers(query=query, limit=limit)
    return {
        "results": [
            {
                "paper": r.item.__dict__,
                "score": r.score,
                "type": r.item_type,
            }
            for r in results
        ],
        "total": len(results),
    }


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str, backend: str = "surrealdb"):
    """Get a paper by ID."""
    service = await get_knowledge_service(backend)
    paper = await service.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper '{paper_id}' not found")
    return paper.__dict__


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

@router.get("/events")
async def list_events(
    limit: int = 50,
    backend: str = "surrealdb",
):
    """List all events from the knowledge base."""
    service = await get_knowledge_service(backend)
    events = await service.list_events(limit=limit)
    return {"events": events, "count": len(events), "backend": backend}


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

@router.get("/patterns")
async def list_patterns(
    limit: int = 50,
    backend: str = "surrealdb",
):
    """List all patterns from the knowledge base."""
    service = await get_knowledge_service(backend)
    patterns = await service.list_patterns(limit=limit)
    return {"patterns": patterns, "count": len(patterns), "backend": backend}


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


@router.get("/stats")
async def get_knowledge_stats(backend: str = "surrealdb"):
    """Get knowledge base statistics."""
    try:
        service = await get_knowledge_service(backend)
        stats = await service.get_stats()
        return {
            "backend": backend,
            "stats": stats,
            "status": "healthy"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


# ============== GRAPH EXPLORER ENDPOINTS ==============

class GraphNode(BaseModel):
    """Node for Cosmograph visualization."""
    id: str
    type: str  # paper, event, pattern, data_source, climate_index
    label: str
    date: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    confidence: Optional[float] = None
    cluster: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GraphLink(BaseModel):
    """Link for Cosmograph visualization."""
    source: str
    target: str
    type: str  # cited_by, caused_by, related_to, validates, uses_data
    strength: float = 0.5
    label: Optional[str] = None


class GraphData(BaseModel):
    """Full graph data for Cosmograph."""
    nodes: List[GraphNode]
    links: List[GraphLink]
    stats: Dict[str, int]


@router.get("/graph", response_model=GraphData)
async def get_knowledge_graph(
    backend: str = "surrealdb",
    include_papers: bool = True,
    include_events: bool = True,
    include_patterns: bool = True,
    limit_papers: int = 100,
    limit_events: int = 50,
):
    """
    Get knowledge graph data for Cosmograph visualization.
    
    Returns nodes (papers, events, patterns) and links between them.
    Event-centric: events are the main focus, papers and patterns connect to them.
    """
    service = await get_knowledge_service(backend)
    
    nodes: List[GraphNode] = []
    links: List[GraphLink] = []
    node_ids = set()
    
    # Helper to access dict or object attributes
    def get_attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    
    # 1. Load Events (central nodes)
    if include_events:
        events = await service.list_events(limit=limit_events)
        for event in events:
            event_id = f"event:{get_attr(event, 'id')}"
            node_ids.add(event_id)
            
            # Extract location if available
            lat, lon = None, None
            location = get_attr(event, 'location')
            if location and isinstance(location, dict):
                lat = location.get('lat') or location.get('latitude')
                lon = location.get('lon') or location.get('longitude')
            
            # Handle severity (can be string or float)
            severity = get_attr(event, 'severity')
            if isinstance(severity, str):
                severity_map = {'extreme': 1.0, 'severe': 0.8, 'moderate': 0.5, 'minor': 0.3}
                severity = severity_map.get(severity.lower(), 0.5)
            
            nodes.append(GraphNode(
                id=event_id,
                type="event",
                label=get_attr(event, 'name', 'Unknown Event'),
                date=get_attr(event, 'start_date'),
                lat=lat,
                lon=lon,
                confidence=severity,
                cluster=get_attr(event, 'event_type'),
                metadata={
                    "description": get_attr(event, 'description'),
                    "end_date": get_attr(event, 'end_date'),
                    "source": get_attr(event, 'source'),
                }
            ))
    
    # 2. Load Papers
    if include_papers:
        papers = await service.list_papers(limit=limit_papers)
        for paper in papers:
            paper_id = f"paper:{get_attr(paper, 'id')}"
            node_ids.add(paper_id)
            
            title = get_attr(paper, 'title', 'Unknown Paper')
            year = get_attr(paper, 'year')
            abstract = get_attr(paper, 'abstract', '')
            keywords = get_attr(paper, 'keywords', [])
            
            nodes.append(GraphNode(
                id=paper_id,
                type="paper",
                label=title[:60] + "..." if len(title) > 60 else title,
                date=f"{year}-01-01" if year else None,
                confidence=0.8,  # Papers are generally reliable
                cluster=keywords[0] if keywords else "research",
                metadata={
                    "authors": get_attr(paper, 'authors', []),
                    "journal": get_attr(paper, 'journal'),
                    "doi": get_attr(paper, 'doi'),
                    "abstract": abstract[:200] if abstract else None,
                    "keywords": keywords,
                }
            ))
            
            # Link papers to events by keyword matching
            for event_node in [n for n in nodes if n.type == "event"]:
                event_name_lower = event_node.label.lower()
                paper_text = f"{title} {abstract or ''}".lower()
                
                # Check for keyword overlap
                if any(kw.lower() in paper_text for kw in [event_name_lower.split()[0]]):
                    links.append(GraphLink(
                        source=paper_id,
                        target=event_node.id,
                        type="related_to",
                        strength=0.6,
                        label="mentions"
                    ))
    
    # 3. Load Patterns
    if include_patterns:
        patterns = await service.list_patterns(limit=50)
        for pattern in patterns:
            pattern_id = f"pattern:{get_attr(pattern, 'id')}"
            node_ids.add(pattern_id)
            
            pattern_name = get_attr(pattern, 'name', 'Unknown Pattern')
            pattern_type = get_attr(pattern, 'pattern_type', '')
            strength = get_attr(pattern, 'strength')
            confidence = get_attr(pattern, 'confidence')
            lag_days = get_attr(pattern, 'lag_days')
            
            nodes.append(GraphNode(
                id=pattern_id,
                type="pattern",
                label=pattern_name,
                confidence=confidence or strength,
                cluster=pattern_type,
                metadata={
                    "description": get_attr(pattern, 'description'),
                    "variables": get_attr(pattern, 'variables', []),
                    "lag_days": lag_days,
                }
            ))
            
            # Link patterns to events by type matching
            for event_node in [n for n in nodes if n.type == "event"]:
                if event_node.cluster and pattern_type:
                    if event_node.cluster.lower() in pattern_type.lower():
                        links.append(GraphLink(
                            source=pattern_id,
                            target=event_node.id,
                            type="caused_by",
                            strength=strength or 0.5,
                            label=f"τ={lag_days}d" if lag_days else None
                        ))
    
    # Stats
    stats = {
        "total_nodes": len(nodes),
        "papers": len([n for n in nodes if n.type == "paper"]),
        "events": len([n for n in nodes if n.type == "event"]),
        "patterns": len([n for n in nodes if n.type == "pattern"]),
        "total_links": len(links),
    }
    
    return GraphData(nodes=nodes, links=links, stats=stats)


@router.get("/graph/expand/{node_id}")
async def expand_graph_node(
    node_id: str,
    backend: str = "surrealdb",
    depth: int = 1,
):
    """
    Expand a node to show its connections.
    
    Used for progressive exploration:
    - Click event → show related papers, patterns
    - Click paper → show cited papers, related events
    - Click pattern → show causing/caused patterns
    """
    service = await get_knowledge_service(backend)
    
    # Parse node type and id
    parts = node_id.split(":")
    if len(parts) != 2:
        raise HTTPException(400, "Invalid node_id format. Expected 'type:id'")
    
    node_type, entity_id = parts
    
    expanded_nodes: List[GraphNode] = []
    expanded_links: List[GraphLink] = []
    
    if node_type == "event":
        # Find related papers and patterns
        event = await service.get_event(entity_id)
        if event:
            # Search papers mentioning this event
            results = await service.search_papers(
                query=f"{event.name} {event.event_type}",
                limit=10
            )
            for r in results:
                paper = r.item
                expanded_nodes.append(GraphNode(
                    id=f"paper:{paper.id}",
                    type="paper",
                    label=paper.title[:50] + "...",
                    date=f"{paper.year}-01-01" if paper.year else None,
                    confidence=r.score,
                    cluster=paper.keywords[0] if paper.keywords else "research",
                ))
                expanded_links.append(GraphLink(
                    source=f"paper:{paper.id}",
                    target=node_id,
                    type="related_to",
                    strength=r.score,
                ))
    
    elif node_type == "paper":
        # Find events mentioned in paper
        paper = await service.get_paper(entity_id)
        if paper:
            events = await service.list_events(limit=20)
            for event in events:
                # Check if event is mentioned in paper
                paper_text = f"{paper.title} {paper.abstract or ''}".lower()
                if event.name.lower() in paper_text:
                    expanded_nodes.append(GraphNode(
                        id=f"event:{event.id}",
                        type="event",
                        label=event.name,
                        date=event.start_date,
                        confidence=event.severity,
                        cluster=event.event_type,
                    ))
                    expanded_links.append(GraphLink(
                        source=node_id,
                        target=f"event:{event.id}",
                        type="related_to",
                        strength=0.7,
                    ))
    
    return {
        "center_node": node_id,
        "expanded_nodes": [n.dict() for n in expanded_nodes],
        "expanded_links": [l.dict() for l in expanded_links],
        "total_expanded": len(expanded_nodes),
    }



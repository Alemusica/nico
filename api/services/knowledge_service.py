"""
Knowledge Service - Abstract Base and Common Interfaces

Provides unified interface for knowledge graph operations across
different backends (Neo4j, SurrealDB).

Supports:
- Scientific paper embeddings and semantic search
- Historical event storage and correlation
- Causal pattern validation against literature
- Climate index (NAO, ENSO) data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class KnowledgeBackend(Enum):
    """Supported knowledge graph backends."""
    NEO4J = "neo4j"
    SURREALDB = "surrealdb"


@dataclass
class Paper:
    """Scientific paper with embedding for semantic search."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    embedding: Optional[List[float]] = None
    keywords: List[str] = field(default_factory=list)
    domain: str = "oceanography"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "doi": self.doi,
            "year": self.year,
            "journal": self.journal,
            "embedding": self.embedding,
            "keywords": self.keywords,
            "domain": self.domain
        }


@dataclass
class HistoricalEvent:
    """Historical event (flood, storm, etc.) for pattern validation."""
    id: str
    event_type: str  # "flood", "storm_surge", "drought"
    date: datetime
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    magnitude: Optional[float] = None
    description: Optional[str] = None
    source: Optional[str] = None  # "newspaper", "scientific", "official"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "date": self.date.isoformat(),
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "magnitude": self.magnitude,
            "description": self.description,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class ClimateIndex:
    """Climate index data (NAO, ENSO, etc.)."""
    id: str
    name: str  # "NAO", "ENSO", "AMO", "PDO"
    date: datetime
    value: float
    source: str = "NOAA"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "date": self.date.isoformat(),
            "value": self.value,
            "source": self.source
        }


@dataclass
class CausalPattern:
    """Validated causal pattern from literature or discovery."""
    id: str
    source_variable: str
    target_variable: str
    lag_days: int
    strength: float  # correlation coefficient
    confidence: float  # 0-1, based on validation
    physics_valid: bool
    discovery_method: str  # "pcmci", "literature", "correlation"
    validated_by: List[str] = field(default_factory=list)  # paper IDs
    observed_in: List[str] = field(default_factory=list)  # event IDs
    domain: str = "oceanography"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_variable": self.source_variable,
            "target_variable": self.target_variable,
            "lag_days": self.lag_days,
            "strength": self.strength,
            "confidence": self.confidence,
            "physics_valid": self.physics_valid,
            "discovery_method": self.discovery_method,
            "validated_by": self.validated_by,
            "observed_in": self.observed_in,
            "domain": self.domain
        }


@dataclass
class ValidationResult:
    """Result of validating a pattern against knowledge base."""
    pattern_id: str
    is_known: bool
    confidence: float
    supporting_papers: List[Paper]
    similar_events: List[HistoricalEvent]
    climate_correlations: List[Tuple[str, float]]  # (index_name, correlation)
    explanation: str


@dataclass 
class SearchResult:
    """Result from semantic search."""
    item: Any  # Paper, Event, or Pattern
    score: float
    item_type: str


class KnowledgeService(ABC):
    """
    Abstract base class for knowledge graph services.
    
    Implementations must support:
    - Vector similarity search for papers
    - Graph traversal for relationships
    - Time-based queries for events
    - Pattern validation
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the knowledge graph."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the knowledge graph."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check connection health and return status."""
        pass
    
    # ========== Paper Operations ==========
    
    @abstractmethod
    async def add_paper(self, paper: Paper) -> str:
        """Add a paper to the knowledge graph. Returns paper ID."""
        pass
    
    @abstractmethod
    async def search_papers(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        min_score: float = 0.7
    ) -> List[SearchResult]:
        """Semantic search for papers using embedding similarity."""
        pass
    
    @abstractmethod
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by ID."""
        pass
    
    @abstractmethod
    async def search_papers_by_keywords(
        self, 
        keywords: List[str], 
        limit: int = 10
    ) -> List[Paper]:
        """Search papers by keywords."""
        pass
    
    # ========== Event Operations ==========
    
    @abstractmethod
    async def add_event(self, event: HistoricalEvent) -> str:
        """Add a historical event. Returns event ID."""
        pass
    
    @abstractmethod
    async def search_events(
        self,
        event_type: Optional[str] = None,
        location: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[HistoricalEvent]:
        """Search events with filters."""
        pass
    
    @abstractmethod
    async def find_correlated_events(
        self,
        event_id: str,
        lag_range: Tuple[int, int] = (-30, 30),
        limit: int = 20
    ) -> List[Tuple[HistoricalEvent, int]]:
        """Find events correlated with a given event within lag range."""
        pass
    
    # ========== Climate Index Operations ==========
    
    @abstractmethod
    async def add_climate_index(self, index: ClimateIndex) -> str:
        """Add climate index data point."""
        pass
    
    @abstractmethod
    async def get_climate_index(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ClimateIndex]:
        """Get climate index values for a date range."""
        pass
    
    @abstractmethod
    async def correlate_with_climate(
        self,
        event_id: str,
        index_names: List[str] = ["NAO", "ENSO"],
        lag_range: Tuple[int, int] = (-60, 0)
    ) -> List[Tuple[str, float, int]]:
        """Correlate event with climate indices. Returns (index, correlation, lag)."""
        pass
    
    # ========== Pattern Operations ==========
    
    @abstractmethod
    async def add_pattern(self, pattern: CausalPattern) -> str:
        """Add a causal pattern. Returns pattern ID."""
        pass
    
    @abstractmethod
    async def validate_pattern(
        self,
        source: str,
        target: str,
        lag: int,
        domain: str = "oceanography"
    ) -> ValidationResult:
        """Validate a discovered pattern against knowledge base."""
        pass
    
    @abstractmethod
    async def find_similar_patterns(
        self,
        source: str,
        target: str,
        limit: int = 10
    ) -> List[CausalPattern]:
        """Find patterns similar to the given source-target pair."""
        pass
    
    @abstractmethod
    async def link_paper_to_pattern(
        self, 
        paper_id: str, 
        pattern_id: str,
        confidence: float = 1.0
    ) -> bool:
        """Create relationship: Paper -[VALIDATES]-> Pattern."""
        pass
    
    @abstractmethod
    async def link_event_to_pattern(
        self, 
        event_id: str, 
        pattern_id: str
    ) -> bool:
        """Create relationship: Pattern -[OBSERVED_IN]-> Event."""
        pass
    
    # ========== Graph Traversal ==========
    
    @abstractmethod
    async def get_pattern_evidence(
        self, 
        pattern_id: str
    ) -> Dict[str, Any]:
        """
        Get all evidence for a pattern:
        - Validating papers
        - Historical events where observed
        - Climate correlations
        """
        pass
    
    @abstractmethod
    async def find_teleconnections(
        self,
        region: str,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find teleconnection patterns affecting a region.
        Traverses graph to find indirect causal chains.
        """
        pass
    
    # ========== Bulk Operations ==========
    
    @abstractmethod
    async def bulk_add_papers(self, papers: List[Paper]) -> int:
        """Bulk add papers. Returns count added."""
        pass
    
    @abstractmethod
    async def bulk_add_events(self, events: List[HistoricalEvent]) -> int:
        """Bulk add events. Returns count added."""
        pass
    
    # ========== Agent Layer Operations ==========
    # Intermediate actors that mediate causality (operators, infrastructure, processes)
    
    @abstractmethod
    async def add_agent(
        self,
        agent_id: str,
        agent_type: str,  # OPERATOR, INFRASTRUCTURE, PHYSICAL_PROCESS, CLIMATE_PATTERN
        name: str,
        capabilities: List[Dict[str, Any]],  # What they CAN do
        constraints: List[Dict[str, Any]],   # What limits them
        operates_on: Optional[List[str]] = None,  # Systems they interact with
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an intermediate agent/actor.
        
        Manufacturing: Operator who runs machine
        Climate: City that emits heat, physical process that transfers energy
        """
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        pass
    
    @abstractmethod
    async def update_agent_state(
        self,
        agent_id: str,
        state_name: str,  # "tired", "inefficient", "overloaded"
        value: float,     # 0-1 intensity
        timestamp: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update agent's state.
        
        Example: Operator fatigue level on Friday night shift
        Example: City heating efficiency during cold snap
        """
        pass
    
    @abstractmethod
    async def record_agent_action(
        self,
        agent_id: str,
        action_id: str,
        capability_used: str,
        parameters: Dict[str, Any],
        resulted_in: Optional[str] = None,  # Event or pattern this caused
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Record an action taken by an agent.
        
        Example: Operator tweaked extruder speed to 120 rpm
        Example: City heating system emitted 500 MW thermal
        """
        pass
    
    @abstractmethod
    async def link_agent_to_system(
        self,
        agent_id: str,
        system_id: str,
        relationship: str = "OPERATES_ON"  # or INFLUENCES, MONITORS
    ) -> bool:
        """Link agent to a system/machine/process they interact with."""
        pass
    
    # ========== Paper ↔ Event Bidirectional Relations ==========
    # Papers document events, events inspire papers
    
    @abstractmethod
    async def link_paper_to_event(
        self,
        paper_id: str,
        event_id: str,
        relation_type: str,  # DOCUMENTS, INSPIRES, PREDICTS, VALIDATES
        confidence: float = 1.0,
        direction: str = "paper_to_event"  # or "event_to_paper"
    ) -> bool:
        """
        Create bidirectional Paper ↔ Event relationship.
        
        - Paper -[DOCUMENTS]-> Event (paper studied the event)
        - Event -[INSPIRES]-> Paper (event led to research)
        - Paper -[PREDICTS]-> Event (paper predicted before event occurred)
        - Paper -[VALIDATES]-> Event (paper confirms event's causality)
        """
        pass
    
    @abstractmethod
    async def get_papers_for_event(
        self, 
        event_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all papers related to an event."""
        pass
    
    @abstractmethod
    async def get_events_for_paper(
        self,
        paper_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all events related to a paper."""
        pass
    
    # ========== Causal Chain with Agents ==========
    
    @abstractmethod
    async def find_agent_causal_chains(
        self,
        outcome_event_id: str,
        max_depth: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Find full causal chains leading to an outcome, including agent mediation.
        
        Returns chains like:
        Pattern -> Agent(state) -> Action -> Event(outcome)
        
        Manufacturing: High temp pattern -> Operator(tired) -> Wrong speed -> Defect
        Climate: NAO pattern -> City(heating) -> Thermal emission -> Local vortex
        """
        pass
    
    @abstractmethod
    async def find_agent_influence_network(
        self,
        agent_id: str,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """
        Panama Papers-style: Find all entities connected to an agent.
        
        Returns network of:
        - Systems they operate on
        - Actions they've taken
        - Events they've influenced
        - Patterns they respond to
        - Other agents they interact with
        """
        pass
    
    @abstractmethod
    async def find_pattern_by_agent_state(
        self,
        agent_type: str,
        state_name: str,
        state_threshold: float = 0.5,
        outcome_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find patterns where agent state correlates with outcomes.
        
        Example: Find all cases where tired operators (fatigue > 0.7)
                 correlated with defect events
        """
        pass


def create_knowledge_service(
    backend: KnowledgeBackend,
    **config
) -> KnowledgeService:
    """
    Factory function to create appropriate knowledge service.
    
    Args:
        backend: Which backend to use (neo4j or surrealdb)
        **config: Backend-specific configuration
        
    Returns:
        Configured KnowledgeService instance
    """
    if backend == KnowledgeBackend.NEO4J:
        from .neo4j_knowledge import Neo4jKnowledgeService
        return Neo4jKnowledgeService(**config)
    elif backend == KnowledgeBackend.SURREALDB:
        from .surrealdb_knowledge import SurrealDBKnowledgeService
        return SurrealDBKnowledgeService(**config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

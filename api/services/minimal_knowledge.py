"""
Minimal In-Memory Knowledge Service
====================================
Lightweight implementation for development and fallback scenarios.
Implements only core paper storage functionality.

Design:
- Simple in-memory storage with dict
- No external dependencies
- Thread-safe operations
- Easy to extend incrementally
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from .knowledge_service import (
    KnowledgeService,
    Paper,
    HistoricalEvent,
    CausalPattern,
    ClimateIndex,
    ValidationResult,
    SearchResult
)

logger = logging.getLogger(__name__)


class MinimalKnowledgeService(KnowledgeService):
    """
    Minimal in-memory implementation.
    Focuses on core paper storage functionality.
    """
    
    def __init__(self):
        self._papers: Dict[str, Paper] = {}
        self._events: Dict[str, HistoricalEvent] = {}
        self._patterns: Dict[str, CausalPattern] = {}
        self._indices: Dict[str, ClimateIndex] = {}
        self._connected = False
        self._lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Connect (no-op for in-memory)."""
        self._connected = True
        logger.info("✅ Minimal knowledge service initialized (in-memory)")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect (no-op for in-memory)."""
        self._connected = False
        
    async def health_check(self) -> Dict[str, Any]:
        """Health check."""
        return {
            "status": "healthy",
            "backend": "in-memory",
            "papers_count": len(self._papers),
            "events_count": len(self._events)
        }
    
    # ========== Core Paper Operations ==========
    
    async def add_paper(self, paper: Paper) -> str:
        """Add single paper."""
        async with self._lock:
            self._papers[paper.id] = paper
            return paper.id
    
    async def bulk_add_papers(self, papers: List[Paper]) -> int:
        """Add multiple papers efficiently."""
        async with self._lock:
            added = 0
            for paper in papers:
                if paper.id not in self._papers:
                    self._papers[paper.id] = paper
                    added += 1
            logger.info(f"💾 Saved {added} papers to in-memory store (total: {len(self._papers)})")
            return added
    
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID."""
        return self._papers.get(paper_id)
    
    async def search_papers(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        min_score: float = 0.7
    ) -> List[SearchResult]:
        """Semantic search (returns all papers for now)."""
        results = []
        for paper in list(self._papers.values())[:limit]:
            results.append(SearchResult(
                item=paper,
                score=0.8,  # Mock score
                source="in-memory"
            ))
        return results
    
    async def search_papers_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10
    ) -> List[Paper]:
        """Keyword search."""
        results = []
        for paper in self._papers.values():
            if any(kw.lower() in paper.title.lower() for kw in keywords):
                results.append(paper)
                if len(results) >= limit:
                    break
        return results
    
    # ========== Event Operations (Stub) ==========
    
    async def add_event(self, event: HistoricalEvent) -> str:
        """Add event (minimal implementation)."""
        async with self._lock:
            self._events[event.id] = event
            return event.id
    
    async def get_event(self, event_id: str) -> Optional[HistoricalEvent]:
        """Get event by ID."""
        return self._events.get(event_id)
    
    async def find_correlated_events(
        self,
        event_id: str,
        min_correlation: float = 0.7
    ) -> List[Tuple[HistoricalEvent, float]]:
        """Find correlated events (returns empty for now)."""
        return []
    
    # ========== Pattern Operations (Stub) ==========
    
    async def add_pattern(self, pattern: CausalPattern) -> str:
        """Add pattern."""
        async with self._lock:
            self._patterns[pattern.id] = pattern
            return pattern.id
    
    async def get_pattern(self, pattern_id: str) -> Optional[CausalPattern]:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)
    
    async def validate_pattern(
        self,
        pattern: CausalPattern
    ) -> ValidationResult:
        """Validate pattern (minimal implementation)."""
        return ValidationResult(
            pattern_id=pattern.id,
            is_known=False,
            confidence=0.5,
            supporting_papers=[],
            similar_events=[],
            climate_correlations=[],
            explanation="Minimal validation - in-memory mode"
        )
    
    async def find_similar_patterns(
        self,
        pattern: CausalPattern,
        threshold: float = 0.8
    ) -> List[Tuple[CausalPattern, float]]:
        """Find similar patterns."""
        return []
    
    async def get_pattern_evidence(
        self,
        pattern_id: str
    ) -> Dict[str, Any]:
        """Get pattern evidence."""
        return {
            "papers": [],
            "events": [],
            "confidence": 0.5
        }
    
    # ========== Climate Index Operations (Stub) ==========
    
    async def add_climate_index(self, index: ClimateIndex) -> str:
        """Add climate index."""
        async with self._lock:
            self._indices[index.name] = index
            return index.name
    
    async def get_climate_index(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[ClimateIndex]:
        """Get climate index data."""
        return self._indices.get(name)
    
    async def correlate_with_climate(
        self,
        event_id: str
    ) -> List[Tuple[str, float]]:
        """Correlate event with climate indices."""
        return []
    
    # ========== Link Operations (Stub) ==========
    
    async def link_paper_to_event(
        self,
        paper_id: str,
        event_id: str,
        relevance: float = 1.0
    ) -> bool:
        """Link paper to event."""
        return True  # No-op success
    
    async def link_paper_to_pattern(
        self,
        paper_id: str,
        pattern_id: str,
        validates: bool = True
    ) -> bool:
        """Link paper to pattern."""
        return True
    
    async def link_event_to_pattern(
        self,
        event_id: str,
        pattern_id: str,
        exhibits: bool = True
    ) -> bool:
        """Link event to pattern."""
        return True
    
    async def get_papers_for_event(
        self,
        event_id: str
    ) -> List[Paper]:
        """Get papers linked to event."""
        return []
    
    async def get_events_for_paper(
        self,
        paper_id: str
    ) -> List[HistoricalEvent]:
        """Get events linked to paper."""
        return []
    
    # ========== Agent Operations (Not Implemented) ==========
    
    async def add_agent(self, agent_id: str, agent_type: str, properties: Dict[str, Any]) -> str:
        """Add agent (not implemented)."""
        logger.warning("Agent operations not supported in minimal mode")
        return agent_id
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent (not implemented)."""
        return None
    
    async def update_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Update agent state (not implemented)."""
        return False
    
    async def link_agent_to_system(self, agent_id: str, system_id: str, role: str) -> bool:
        """Link agent to system (not implemented)."""
        return False
    
    async def record_agent_action(
        self,
        agent_id: str,
        action_type: str,
        target_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Record agent action (not implemented)."""
        return ""
    
    async def find_agent_causal_chains(
        self,
        agent_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """Find agent causal chains (not implemented)."""
        return []
    
    async def find_agent_influence_network(
        self,
        agent_id: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Find agent influence network (not implemented)."""
        return {}
    
    async def find_pattern_by_agent_state(
        self,
        agent_states: Dict[str, Any]
    ) -> List[CausalPattern]:
        """Find patterns by agent state (not implemented)."""
        return []
    
    async def bulk_add_events(self, events: List[HistoricalEvent]) -> int:
        """Bulk add events to storage."""
        async with self._lock:
            added = 0
            for event in events:
                if event.event_id not in self._events:
                    self._events[event.event_id] = event
                    added += 1
            logger.info(f"💾 Saved {added} events to in-memory store (total: {len(self._events)})")
            return added
    
    async def search_events(
        self,
        event_type: Optional[str] = None,
        location: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[HistoricalEvent]:
        """Search events with filters."""
        results = []
        for event in self._events.values():
            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if location and event.location != location:
                continue
            if start_date and event.date < start_date:
                continue
            if end_date and event.date > end_date:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results
    
    async def find_teleconnections(
        self,
        region: str,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """Find teleconnection patterns (not implemented)."""
        return []
    
    # ========== Stats ==========
    
    async def get_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        return {
            "papers": len(self._papers),
            "events": len(self._events),
            "patterns": len(self._patterns),
            "climate_indices": len(self._indices),
            "relationships": 0
        }

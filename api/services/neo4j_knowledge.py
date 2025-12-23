"""
Neo4j Knowledge Service Implementation

Uses Neo4j's native vector indexes (HNSW) for semantic search
and Cypher for graph traversal queries.

Requirements:
    pip install neo4j

Setup:
    1. Install Neo4j Community (free) or use Neo4j Aura (cloud)
    2. Enable vector indexes (Neo4j 5.11+)
    3. Configure connection in environment variables
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from .knowledge_service import (
    KnowledgeService,
    Paper,
    HistoricalEvent,
    ClimateIndex,
    CausalPattern,
    ValidationResult,
    SearchResult
)

logger = logging.getLogger(__name__)


class Neo4jKnowledgeService(KnowledgeService):
    """
    Neo4j implementation of Knowledge Service.
    
    Features:
    - HNSW vector indexes for paper embeddings
    - Native graph traversal for causal chains
    - Cypher queries for pattern matching
    """
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = "neo4j",
        embedding_dimension: int = 1536  # OpenAI ada-002 default
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self.embedding_dimension = embedding_dimension
        self._driver = None
        
    async def connect(self) -> bool:
        """Connect to Neo4j and initialize schema."""
        try:
            from neo4j import AsyncGraphDatabase
            
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Test connection
            async with self._driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 AS test")
                await result.consume()
            
            # Initialize schema
            await self._init_schema()
            
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j health."""
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run("""
                    CALL dbms.components() YIELD name, versions, edition
                    RETURN name, versions, edition
                """)
                record = await result.single()
                
                return {
                    "status": "healthy",
                    "backend": "neo4j",
                    "name": record["name"] if record else "unknown",
                    "version": record["versions"][0] if record and record["versions"] else "unknown",
                    "edition": record["edition"] if record else "unknown"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "neo4j",
                "error": str(e)
            }
    
    async def _init_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        async with self._driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT climate_id IF NOT EXISTS FOR (c:ClimateIndex) REQUIRE c.id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")
            
            # Create vector index for paper embeddings
            try:
                await session.run("""
                    CREATE VECTOR INDEX paper_embeddings IF NOT EXISTS
                    FOR (p:Paper) ON p.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: $dimension,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """, dimension=self.embedding_dimension)
                logger.info("Created vector index for papers")
            except Exception as e:
                logger.debug(f"Vector index may already exist: {e}")
            
            # Create full-text index for paper search
            try:
                await session.run("""
                    CREATE FULLTEXT INDEX paper_search IF NOT EXISTS
                    FOR (p:Paper) ON EACH [p.title, p.abstract]
                """)
            except Exception as e:
                logger.debug(f"Fulltext index may already exist: {e}")
            
            # Create indexes for common queries
            indexes = [
                "CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
                "CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.date)",
                "CREATE INDEX climate_name IF NOT EXISTS FOR (c:ClimateIndex) ON (c.name)",
                "CREATE INDEX pattern_domain IF NOT EXISTS FOR (p:Pattern) ON (p.domain)",
            ]
            
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index may already exist: {e}")
    
    # ========== Paper Operations ==========
    
    async def add_paper(self, paper: Paper) -> str:
        """Add paper with embedding to Neo4j."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MERGE (p:Paper {id: $id})
                SET p.title = $title,
                    p.authors = $authors,
                    p.abstract = $abstract,
                    p.doi = $doi,
                    p.year = $year,
                    p.journal = $journal,
                    p.embedding = $embedding,
                    p.keywords = $keywords,
                    p.domain = $domain,
                    p.created_at = datetime()
                RETURN p.id AS id
            """, **paper.to_dict())
            
            record = await result.single()
            return record["id"]
    
    async def search_papers(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.7
    ) -> List[SearchResult]:
        """Semantic search using Neo4j vector index."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                CALL db.index.vector.queryNodes('paper_embeddings', $limit, $embedding)
                YIELD node AS paper, score
                WHERE score >= $min_score
                RETURN paper, score
                ORDER BY score DESC
            """, embedding=query_embedding, limit=limit, min_score=min_score)
            
            results = []
            async for record in result:
                paper_data = dict(record["paper"])
                paper = Paper(
                    id=paper_data["id"],
                    title=paper_data.get("title", ""),
                    authors=paper_data.get("authors", []),
                    abstract=paper_data.get("abstract", ""),
                    doi=paper_data.get("doi"),
                    year=paper_data.get("year"),
                    journal=paper_data.get("journal"),
                    keywords=paper_data.get("keywords", []),
                    domain=paper_data.get("domain", "oceanography")
                )
                results.append(SearchResult(
                    item=paper,
                    score=record["score"],
                    item_type="paper"
                ))
            
            return results
    
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (p:Paper {id: $id})
                RETURN p
            """, id=paper_id)
            
            record = await result.single()
            if not record:
                return None
            
            paper_data = dict(record["p"])
            return Paper(
                id=paper_data["id"],
                title=paper_data.get("title", ""),
                authors=paper_data.get("authors", []),
                abstract=paper_data.get("abstract", ""),
                doi=paper_data.get("doi"),
                year=paper_data.get("year"),
                journal=paper_data.get("journal"),
                keywords=paper_data.get("keywords", []),
                domain=paper_data.get("domain", "oceanography")
            )
    
    async def search_papers_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10
    ) -> List[Paper]:
        """Search papers using full-text index."""
        query_string = " OR ".join(keywords)
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                CALL db.index.fulltext.queryNodes('paper_search', $query)
                YIELD node AS paper, score
                RETURN paper, score
                ORDER BY score DESC
                LIMIT $limit
            """, query=query_string, limit=limit)
            
            papers = []
            async for record in result:
                paper_data = dict(record["paper"])
                papers.append(Paper(
                    id=paper_data["id"],
                    title=paper_data.get("title", ""),
                    authors=paper_data.get("authors", []),
                    abstract=paper_data.get("abstract", ""),
                    doi=paper_data.get("doi"),
                    year=paper_data.get("year"),
                    journal=paper_data.get("journal"),
                    keywords=paper_data.get("keywords", []),
                    domain=paper_data.get("domain", "oceanography")
                ))
            
            return papers
    
    # ========== Event Operations ==========
    
    async def add_event(self, event: HistoricalEvent) -> str:
        """Add historical event."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MERGE (e:Event {id: $id})
                SET e.event_type = $event_type,
                    e.date = datetime($date),
                    e.location = $location,
                    e.latitude = $latitude,
                    e.longitude = $longitude,
                    e.magnitude = $magnitude,
                    e.description = $description,
                    e.source = $source,
                    e.metadata = $metadata
                RETURN e.id AS id
            """, **event.to_dict())
            
            record = await result.single()
            return record["id"]
    
    async def search_events(
        self,
        event_type: Optional[str] = None,
        location: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[HistoricalEvent]:
        """Search events with filters."""
        conditions = []
        params = {"limit": limit}
        
        if event_type:
            conditions.append("e.event_type = $event_type")
            params["event_type"] = event_type
        if location:
            conditions.append("e.location CONTAINS $location")
            params["location"] = location
        if start_date:
            conditions.append("e.date >= datetime($start_date)")
            params["start_date"] = start_date.isoformat()
        if end_date:
            conditions.append("e.date <= datetime($end_date)")
            params["end_date"] = end_date.isoformat()
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(f"""
                MATCH (e:Event)
                WHERE {where_clause}
                RETURN e
                ORDER BY e.date DESC
                LIMIT $limit
            """, **params)
            
            events = []
            async for record in result:
                event_data = dict(record["e"])
                events.append(HistoricalEvent(
                    id=event_data["id"],
                    event_type=event_data["event_type"],
                    date=datetime.fromisoformat(str(event_data["date"]).replace("Z", "+00:00")),
                    location=event_data["location"],
                    latitude=event_data.get("latitude"),
                    longitude=event_data.get("longitude"),
                    magnitude=event_data.get("magnitude"),
                    description=event_data.get("description"),
                    source=event_data.get("source"),
                    metadata=event_data.get("metadata", {})
                ))
            
            return events
    
    async def find_correlated_events(
        self,
        event_id: str,
        lag_range: Tuple[int, int] = (-30, 30),
        limit: int = 20
    ) -> List[Tuple[HistoricalEvent, int]]:
        """Find events within lag range of given event."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (e1:Event {id: $event_id})
                MATCH (e2:Event)
                WHERE e2.id <> e1.id
                  AND e2.date >= e1.date - duration({days: $max_lag})
                  AND e2.date <= e1.date + duration({days: $max_lag})
                WITH e2, duration.between(e1.date, e2.date).days AS lag_days
                RETURN e2, lag_days
                ORDER BY abs(lag_days)
                LIMIT $limit
            """, event_id=event_id, max_lag=max(abs(lag_range[0]), abs(lag_range[1])), limit=limit)
            
            results = []
            async for record in result:
                event_data = dict(record["e2"])
                event = HistoricalEvent(
                    id=event_data["id"],
                    event_type=event_data["event_type"],
                    date=datetime.fromisoformat(str(event_data["date"]).replace("Z", "+00:00")),
                    location=event_data["location"],
                    latitude=event_data.get("latitude"),
                    longitude=event_data.get("longitude"),
                    magnitude=event_data.get("magnitude"),
                    description=event_data.get("description"),
                    source=event_data.get("source")
                )
                results.append((event, record["lag_days"]))
            
            return results
    
    # ========== Climate Index Operations ==========
    
    async def add_climate_index(self, index: ClimateIndex) -> str:
        """Add climate index data point."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MERGE (c:ClimateIndex {id: $id})
                SET c.name = $name,
                    c.date = datetime($date),
                    c.value = $value,
                    c.source = $source
                RETURN c.id AS id
            """, **index.to_dict())
            
            record = await result.single()
            return record["id"]
    
    async def get_climate_index(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ClimateIndex]:
        """Get climate index values for date range."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (c:ClimateIndex)
                WHERE c.name = $name
                  AND c.date >= datetime($start_date)
                  AND c.date <= datetime($end_date)
                RETURN c
                ORDER BY c.date
            """, name=name, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            
            indices = []
            async for record in result:
                data = dict(record["c"])
                indices.append(ClimateIndex(
                    id=data["id"],
                    name=data["name"],
                    date=datetime.fromisoformat(str(data["date"]).replace("Z", "+00:00")),
                    value=data["value"],
                    source=data.get("source", "NOAA")
                ))
            
            return indices
    
    async def correlate_with_climate(
        self,
        event_id: str,
        index_names: List[str] = ["NAO", "ENSO"],
        lag_range: Tuple[int, int] = (-60, 0)
    ) -> List[Tuple[str, float, int]]:
        """
        Correlate event with climate indices.
        Returns list of (index_name, value_at_lag, lag_days).
        """
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (e:Event {id: $event_id})
                MATCH (c:ClimateIndex)
                WHERE c.name IN $index_names
                  AND c.date >= e.date - duration({days: $abs_lag})
                  AND c.date <= e.date
                WITH c.name AS index_name, c.value AS value, 
                     duration.between(c.date, e.date).days AS lag_days
                ORDER BY index_name, lag_days
                RETURN index_name, value, lag_days
            """, event_id=event_id, index_names=index_names, 
                abs_lag=abs(lag_range[0]))
            
            results = []
            async for record in result:
                results.append((
                    record["index_name"],
                    record["value"],
                    record["lag_days"]
                ))
            
            return results
    
    # ========== Pattern Operations ==========
    
    async def add_pattern(self, pattern: CausalPattern) -> str:
        """Add causal pattern."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MERGE (p:Pattern {id: $id})
                SET p.source_variable = $source_variable,
                    p.target_variable = $target_variable,
                    p.lag_days = $lag_days,
                    p.strength = $strength,
                    p.confidence = $confidence,
                    p.physics_valid = $physics_valid,
                    p.discovery_method = $discovery_method,
                    p.domain = $domain,
                    p.created_at = datetime()
                RETURN p.id AS id
            """, **pattern.to_dict())
            
            record = await result.single()
            return record["id"]
    
    async def validate_pattern(
        self,
        source: str,
        target: str,
        lag: int,
        domain: str = "oceanography"
    ) -> ValidationResult:
        """Validate pattern against knowledge base."""
        async with self._driver.session(database=self.database) as session:
            # Find matching or similar patterns
            result = await session.run("""
                MATCH (p:Pattern)
                WHERE p.domain = $domain
                  AND (p.source_variable = $source OR p.source_variable CONTAINS $source)
                  AND (p.target_variable = $target OR p.target_variable CONTAINS $target)
                OPTIONAL MATCH (paper:Paper)-[:VALIDATES]->(p)
                OPTIONAL MATCH (p)-[:OBSERVED_IN]->(event:Event)
                RETURN p, collect(DISTINCT paper) AS papers, collect(DISTINCT event) AS events
                LIMIT 10
            """, source=source, target=target, domain=domain)
            
            supporting_papers = []
            similar_events = []
            is_known = False
            confidence = 0.0
            
            async for record in result:
                pattern_data = dict(record["p"])
                
                # Check if exact match (within lag tolerance)
                if abs(pattern_data.get("lag_days", 0) - lag) <= 3:
                    is_known = True
                    confidence = max(confidence, pattern_data.get("confidence", 0.5))
                
                # Collect papers
                for paper_data in record["papers"]:
                    if paper_data:
                        paper_dict = dict(paper_data)
                        supporting_papers.append(Paper(
                            id=paper_dict["id"],
                            title=paper_dict.get("title", ""),
                            authors=paper_dict.get("authors", []),
                            abstract=paper_dict.get("abstract", ""),
                            doi=paper_dict.get("doi"),
                            year=paper_dict.get("year")
                        ))
                
                # Collect events
                for event_data in record["events"]:
                    if event_data:
                        event_dict = dict(event_data)
                        similar_events.append(HistoricalEvent(
                            id=event_dict["id"],
                            event_type=event_dict["event_type"],
                            date=datetime.fromisoformat(str(event_dict["date"]).replace("Z", "+00:00")),
                            location=event_dict["location"]
                        ))
            
            explanation = self._generate_validation_explanation(
                source, target, lag, is_known, len(supporting_papers), len(similar_events)
            )
            
            return ValidationResult(
                pattern_id=f"{source}->{target}@{lag}",
                is_known=is_known,
                confidence=confidence,
                supporting_papers=supporting_papers,
                similar_events=similar_events,
                climate_correlations=[],
                explanation=explanation
            )
    
    def _generate_validation_explanation(
        self,
        source: str,
        target: str,
        lag: int,
        is_known: bool,
        paper_count: int,
        event_count: int
    ) -> str:
        """Generate explanation for validation result."""
        if is_known:
            return (
                f"Pattern '{source} → {target}' (lag={lag}d) is a KNOWN relationship. "
                f"Found {paper_count} supporting papers and {event_count} historical events."
            )
        elif paper_count > 0 or event_count > 0:
            return (
                f"Pattern '{source} → {target}' (lag={lag}d) has PARTIAL support. "
                f"Similar patterns found in {paper_count} papers and {event_count} events."
            )
        else:
            return (
                f"Pattern '{source} → {target}' (lag={lag}d) is NOVEL. "
                f"No matching patterns in knowledge base. Consider literature review."
            )
    
    async def find_similar_patterns(
        self,
        source: str,
        target: str,
        limit: int = 10
    ) -> List[CausalPattern]:
        """Find patterns similar to source-target pair."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (p:Pattern)
                WHERE p.source_variable CONTAINS $source
                   OR p.target_variable CONTAINS $target
                   OR p.source_variable CONTAINS $target
                   OR p.target_variable CONTAINS $source
                RETURN p
                ORDER BY p.confidence DESC
                LIMIT $limit
            """, source=source, target=target, limit=limit)
            
            patterns = []
            async for record in result:
                data = dict(record["p"])
                patterns.append(CausalPattern(
                    id=data["id"],
                    source_variable=data["source_variable"],
                    target_variable=data["target_variable"],
                    lag_days=data["lag_days"],
                    strength=data["strength"],
                    confidence=data["confidence"],
                    physics_valid=data["physics_valid"],
                    discovery_method=data["discovery_method"],
                    domain=data.get("domain", "oceanography")
                ))
            
            return patterns
    
    async def link_paper_to_pattern(
        self,
        paper_id: str,
        pattern_id: str,
        confidence: float = 1.0
    ) -> bool:
        """Create VALIDATES relationship."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (paper:Paper {id: $paper_id})
                MATCH (pattern:Pattern {id: $pattern_id})
                MERGE (paper)-[r:VALIDATES]->(pattern)
                SET r.confidence = $confidence,
                    r.created_at = datetime()
                RETURN r
            """, paper_id=paper_id, pattern_id=pattern_id, confidence=confidence)
            
            record = await result.single()
            return record is not None
    
    async def link_event_to_pattern(
        self,
        event_id: str,
        pattern_id: str
    ) -> bool:
        """Create OBSERVED_IN relationship."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (pattern:Pattern {id: $pattern_id})
                MATCH (event:Event {id: $event_id})
                MERGE (pattern)-[r:OBSERVED_IN]->(event)
                SET r.created_at = datetime()
                RETURN r
            """, event_id=event_id, pattern_id=pattern_id)
            
            record = await result.single()
            return record is not None
    
    # ========== Graph Traversal ==========
    
    async def get_pattern_evidence(
        self,
        pattern_id: str
    ) -> Dict[str, Any]:
        """Get all evidence for a pattern."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (p:Pattern {id: $pattern_id})
                OPTIONAL MATCH (paper:Paper)-[v:VALIDATES]->(p)
                OPTIONAL MATCH (p)-[o:OBSERVED_IN]->(event:Event)
                RETURN p,
                       collect(DISTINCT {paper: paper, confidence: v.confidence}) AS papers,
                       collect(DISTINCT event) AS events
            """, pattern_id=pattern_id)
            
            record = await result.single()
            if not record:
                return {"error": "Pattern not found"}
            
            pattern_data = dict(record["p"])
            
            return {
                "pattern": CausalPattern(
                    id=pattern_data["id"],
                    source_variable=pattern_data["source_variable"],
                    target_variable=pattern_data["target_variable"],
                    lag_days=pattern_data["lag_days"],
                    strength=pattern_data["strength"],
                    confidence=pattern_data["confidence"],
                    physics_valid=pattern_data["physics_valid"],
                    discovery_method=pattern_data["discovery_method"]
                ).to_dict(),
                "supporting_papers": [
                    {
                        "paper": dict(p["paper"]) if p["paper"] else None,
                        "confidence": p["confidence"]
                    }
                    for p in record["papers"] if p["paper"]
                ],
                "observed_events": [
                    dict(e) for e in record["events"] if e
                ]
            }
    
    async def find_teleconnections(
        self,
        region: str,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """Find teleconnection patterns affecting a region."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH path = (start:Pattern)-[:CAUSES*1..$max_hops]->(end:Pattern)
                WHERE end.target_variable CONTAINS $region
                   OR start.source_variable CONTAINS $region
                RETURN path,
                       [n IN nodes(path) | n.source_variable + ' → ' + n.target_variable] AS chain,
                       length(path) AS hops
                ORDER BY hops
                LIMIT 20
            """, region=region, max_hops=max_hops)
            
            teleconnections = []
            async for record in result:
                teleconnections.append({
                    "chain": record["chain"],
                    "hops": record["hops"]
                })
            
            return teleconnections
    
    # ========== Bulk Operations ==========
    
    async def bulk_add_papers(self, papers: List[Paper]) -> int:
        """Bulk add papers using UNWIND."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                UNWIND $papers AS paper
                MERGE (p:Paper {id: paper.id})
                SET p.title = paper.title,
                    p.authors = paper.authors,
                    p.abstract = paper.abstract,
                    p.doi = paper.doi,
                    p.year = paper.year,
                    p.journal = paper.journal,
                    p.embedding = paper.embedding,
                    p.keywords = paper.keywords,
                    p.domain = paper.domain
                RETURN count(p) AS count
            """, papers=[p.to_dict() for p in papers])
            
            record = await result.single()
            return record["count"] if record else 0
    
    async def bulk_add_events(self, events: List[HistoricalEvent]) -> int:
        """Bulk add events using UNWIND."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                UNWIND $events AS event
                MERGE (e:Event {id: event.id})
                SET e.event_type = event.event_type,
                    e.date = datetime(event.date),
                    e.location = event.location,
                    e.latitude = event.latitude,
                    e.longitude = event.longitude,
                    e.magnitude = event.magnitude,
                    e.description = event.description,
                    e.source = event.source
                RETURN count(e) AS count
            """, events=[e.to_dict() for e in events])
            
            record = await result.single()
            return record["count"] if record else 0
    
    # ========== Agent Layer Operations ==========
    
    async def add_agent(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        capabilities: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        operates_on: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add agent node with capabilities and constraints."""
        async with self._driver.session(database=self.database) as session:
            await session.run("""
                MERGE (a:Agent {id: $id})
                SET a.agent_type = $agent_type,
                    a.name = $name,
                    a.capabilities = $capabilities,
                    a.constraints = $constraints,
                    a.operates_on = $operates_on,
                    a.metadata = $metadata,
                    a.created_at = datetime()
                
                // Create capability nodes for detailed tracking
                WITH a
                UNWIND $capabilities AS cap
                MERGE (c:Capability {name: cap.name, agent_id: $id})
                SET c.capability_type = cap.type,
                    c.parameters = cap.parameters,
                    c.constraints = cap.constraints
                MERGE (a)-[:HAS_CAPABILITY]->(c)
            """, 
                id=agent_id, 
                agent_type=agent_type, 
                name=name,
                capabilities=[c if isinstance(c, dict) else {"name": c} for c in capabilities],
                constraints=[c if isinstance(c, dict) else {"name": c} for c in constraints],
                operates_on=operates_on or [],
                metadata=metadata or {}
            )
            return True
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent with all relationships."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (a:Agent {id: $id})
                OPTIONAL MATCH (a)-[:HAS_CAPABILITY]->(cap:Capability)
                OPTIONAL MATCH (a)-[:OPERATES_ON]->(sys)
                OPTIONAL MATCH (a)-[:HAS_STATE]->(state:AgentState)
                RETURN a,
                       collect(DISTINCT cap) AS capabilities,
                       collect(DISTINCT sys) AS systems,
                       collect(DISTINCT state) AS states
            """, id=agent_id)
            
            record = await result.single()
            if not record:
                return None
            
            agent_data = dict(record["a"])
            return {
                **agent_data,
                "capabilities": [dict(c) for c in record["capabilities"]],
                "systems": [dict(s) for s in record["systems"]],
                "current_states": [dict(s) for s in record["states"]]
            }
    
    async def update_agent_state(
        self,
        agent_id: str,
        state_name: str,
        value: float,
        timestamp: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record agent state at a point in time."""
        async with self._driver.session(database=self.database) as session:
            await session.run("""
                MATCH (a:Agent {id: $agent_id})
                MERGE (s:AgentState {
                    agent_id: $agent_id, 
                    state_name: $state_name,
                    timestamp: datetime($timestamp)
                })
                SET s.value = $value,
                    s.context = $context
                MERGE (a)-[:HAS_STATE]->(s)
            """, 
                agent_id=agent_id, 
                state_name=state_name, 
                value=value,
                timestamp=timestamp or datetime.now().isoformat(),
                context=context or {}
            )
            return True
    
    async def record_agent_action(
        self,
        agent_id: str,
        action_id: str,
        capability_used: str,
        parameters: Dict[str, Any],
        resulted_in: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """Record an action taken by agent and its outcome."""
        async with self._driver.session(database=self.database) as session:
            query = """
                MATCH (a:Agent {id: $agent_id})
                CREATE (act:Action {
                    id: $action_id,
                    capability_used: $capability_used,
                    parameters: $parameters,
                    timestamp: datetime($timestamp)
                })
                MERGE (a)-[:TOOK_ACTION]->(act)
            """
            
            if resulted_in:
                query += """
                    WITH act
                    MATCH (outcome) WHERE outcome.id = $resulted_in
                    MERGE (act)-[:RESULTED_IN]->(outcome)
                """
            
            await session.run(query,
                agent_id=agent_id,
                action_id=action_id,
                capability_used=capability_used,
                parameters=parameters,
                timestamp=timestamp or datetime.now().isoformat(),
                resulted_in=resulted_in
            )
            return True
    
    async def link_agent_to_system(
        self,
        agent_id: str,
        system_id: str,
        relationship: str = "OPERATES_ON"
    ) -> bool:
        """Create agent-system relationship."""
        async with self._driver.session(database=self.database) as session:
            await session.run(f"""
                MATCH (a:Agent {{id: $agent_id}})
                MATCH (s) WHERE s.id = $system_id
                MERGE (a)-[:{relationship}]->(s)
            """, agent_id=agent_id, system_id=system_id)
            return True
    
    # ========== Paper ↔ Event Bidirectional Relations ==========
    
    async def link_paper_to_event(
        self,
        paper_id: str,
        event_id: str,
        relation_type: str,
        confidence: float = 1.0,
        direction: str = "paper_to_event"
    ) -> bool:
        """Create bidirectional Paper ↔ Event relationship."""
        async with self._driver.session(database=self.database) as session:
            if direction == "paper_to_event":
                await session.run(f"""
                    MATCH (p:Paper {{id: $paper_id}})
                    MATCH (e:Event {{id: $event_id}})
                    MERGE (p)-[r:{relation_type}]->(e)
                    SET r.confidence = $confidence,
                        r.created_at = datetime()
                """, paper_id=paper_id, event_id=event_id, confidence=confidence)
            else:
                await session.run(f"""
                    MATCH (e:Event {{id: $event_id}})
                    MATCH (p:Paper {{id: $paper_id}})
                    MERGE (e)-[r:{relation_type}]->(p)
                    SET r.confidence = $confidence,
                        r.created_at = datetime()
                """, paper_id=paper_id, event_id=event_id, confidence=confidence)
            return True
    
    async def get_papers_for_event(
        self, 
        event_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all papers related to an event."""
        async with self._driver.session(database=self.database) as session:
            rel_filter = ""
            if relation_types:
                rel_filter = "WHERE type(r) IN $relation_types"
            
            result = await session.run(f"""
                MATCH (e:Event {{id: $event_id}})<-[r]-(p:Paper)
                {rel_filter}
                RETURN p, type(r) AS relation, r.confidence AS confidence
                UNION
                MATCH (e:Event {{id: $event_id}})-[r]->(p:Paper)
                {rel_filter}
                RETURN p, type(r) AS relation, r.confidence AS confidence
            """, event_id=event_id, relation_types=relation_types)
            
            papers = []
            async for record in result:
                papers.append({
                    "paper": dict(record["p"]),
                    "relation": record["relation"],
                    "confidence": record["confidence"]
                })
            return papers
    
    async def get_events_for_paper(
        self,
        paper_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all events related to a paper."""
        async with self._driver.session(database=self.database) as session:
            rel_filter = ""
            if relation_types:
                rel_filter = "WHERE type(r) IN $relation_types"
            
            result = await session.run(f"""
                MATCH (p:Paper {{id: $paper_id}})-[r]->(e:Event)
                {rel_filter}
                RETURN e, type(r) AS relation, r.confidence AS confidence
                UNION
                MATCH (p:Paper {{id: $paper_id}})<-[r]-(e:Event)
                {rel_filter}
                RETURN e, type(r) AS relation, r.confidence AS confidence
            """, paper_id=paper_id, relation_types=relation_types)
            
            events = []
            async for record in result:
                events.append({
                    "event": dict(record["e"]),
                    "relation": record["relation"],
                    "confidence": record["confidence"]
                })
            return events
    
    # ========== Causal Chain with Agents ==========
    
    async def find_agent_causal_chains(
        self,
        outcome_event_id: str,
        max_depth: int = 4
    ) -> List[Dict[str, Any]]:
        """Find full causal chains leading to an outcome including agent mediation."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (outcome:Event {id: $event_id})
                MATCH (action:Action)-[:RESULTED_IN]->(outcome)
                MATCH (agent:Agent)-[:TOOK_ACTION]->(action)
                OPTIONAL MATCH (agent)-[:HAS_STATE]->(state:AgentState)
                WHERE state.timestamp <= action.timestamp
                OPTIONAL MATCH (pattern:Pattern)-[:INFLUENCES]->(agent)
                OPTIONAL MATCH (agent)-[:OPERATES_ON]->(system)
                RETURN DISTINCT
                    outcome, action, agent,
                    collect(DISTINCT state) AS agent_states,
                    collect(DISTINCT pattern) AS triggering_patterns,
                    collect(DISTINCT system) AS operated_systems
                ORDER BY action.timestamp DESC
            """, event_id=outcome_event_id, max_depth=max_depth)
            
            chains = []
            async for record in result:
                chains.append({
                    "outcome": dict(record["outcome"]),
                    "action": dict(record["action"]),
                    "agent": dict(record["agent"]),
                    "agent_states": [dict(s) for s in record["agent_states"]],
                    "triggering_patterns": [dict(p) for p in record["triggering_patterns"]],
                    "operated_systems": [dict(s) for s in record["operated_systems"]],
                    "causal_narrative": self._build_narrative(record)
                })
            return chains
    
    def _build_narrative(self, record) -> str:
        """Build human-readable narrative from causal chain."""
        agent = record["agent"]
        action = record["action"]
        outcome = record["outcome"]
        states = record["agent_states"]
        
        narrative = f"{agent.get('name', 'Unknown agent')} ({agent.get('agent_type', '')})"
        if states:
            state_desc = ", ".join([
                f"{s.get('state_name', '')}={s.get('value', 0):.1f}" 
                for s in states if s
            ])
            narrative += f" [state: {state_desc}]"
        narrative += f" → {action.get('capability_used', 'unknown action')}"
        narrative += f" → {outcome.get('event_type', 'outcome')}"
        return narrative
    
    async def find_agent_influence_network(
        self,
        agent_id: str,
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """Panama Papers-style network analysis for an agent."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (a:Agent {id: $agent_id})
                OPTIONAL MATCH (a)-[:OPERATES_ON]->(systems)
                OPTIONAL MATCH (a)-[:TOOK_ACTION]->(actions)
                OPTIONAL MATCH (actions)-[:RESULTED_IN]->(outcomes)
                OPTIONAL MATCH (patterns:Pattern)-[:INFLUENCES]->(a)
                OPTIONAL MATCH (patterns)-[:INFLUENCES]->(other_agents:Agent)
                WHERE other_agents.id <> $agent_id
                OPTIONAL MATCH (papers:Paper)-[:DOCUMENTS]->(outcomes)
                RETURN a AS agent,
                       collect(DISTINCT systems) AS systems,
                       collect(DISTINCT actions) AS actions,
                       collect(DISTINCT outcomes) AS outcomes,
                       collect(DISTINCT patterns) AS influencing_patterns,
                       collect(DISTINCT other_agents) AS related_agents,
                       collect(DISTINCT papers) AS documenting_papers
            """, agent_id=agent_id, max_hops=max_hops)
            
            record = await result.single()
            if not record:
                return {"error": "Agent not found"}
            
            return {
                "agent": dict(record["agent"]),
                "network": {
                    "systems_operated": [dict(s) for s in record["systems"] if s],
                    "actions_taken": [dict(a) for a in record["actions"] if a],
                    "outcomes_caused": [dict(o) for o in record["outcomes"] if o],
                    "influencing_patterns": [dict(p) for p in record["influencing_patterns"] if p],
                    "related_agents": [dict(a) for a in record["related_agents"] if a],
                    "documenting_papers": [dict(p) for p in record["documenting_papers"] if p]
                }
            }
    
    async def find_pattern_by_agent_state(
        self,
        agent_type: str,
        state_name: str,
        state_threshold: float = 0.5,
        outcome_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find patterns where agent state correlates with outcomes."""
        async with self._driver.session(database=self.database) as session:
            outcome_filter = ""
            if outcome_type:
                outcome_filter = "AND outcome.event_type = $outcome_type"
            
            result = await session.run(f"""
                MATCH (agent:Agent {{agent_type: $agent_type}})
                MATCH (agent)-[:HAS_STATE]->(state:AgentState)
                WHERE state.state_name = $state_name 
                  AND state.value >= $state_threshold
                MATCH (agent)-[:TOOK_ACTION]->(action:Action)
                MATCH (action)-[:RESULTED_IN]->(outcome:Event)
                WHERE action.timestamp >= state.timestamp
                  {outcome_filter}
                RETURN agent.agent_type AS agent_type,
                       state.state_name AS state_factor,
                       avg(state.value) AS avg_state_value,
                       outcome.event_type AS outcome_type,
                       count(DISTINCT outcome) AS occurrence_count,
                       collect(DISTINCT {{
                           agent: agent.name,
                           state_value: state.value,
                           outcome_id: outcome.id
                       }})[0..10] AS sample_cases
                ORDER BY occurrence_count DESC
            """, 
                agent_type=agent_type, 
                state_name=state_name, 
                state_threshold=state_threshold,
                outcome_type=outcome_type
            )
            
            patterns = []
            async for record in result:
                patterns.append({
                    "pattern": {
                        "agent_type": record["agent_type"],
                        "state_factor": record["state_factor"],
                        "avg_state_value": record["avg_state_value"],
                        "outcome_type": record["outcome_type"]
                    },
                    "statistics": {
                        "occurrence_count": record["occurrence_count"],
                        "confidence": min(record["occurrence_count"] / 10, 1.0)
                    },
                    "sample_cases": record["sample_cases"]
                })
            return patterns

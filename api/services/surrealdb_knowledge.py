"""
SurrealDB Knowledge Service Implementation.

Multi-model database with document, graph, and vector capabilities.
Features LIVE SELECT for real-time subscriptions and SurrealML integration.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Optional
from uuid import uuid4

from .knowledge_service import (
    CausalPattern,
    ClimateIndex,
    HistoricalEvent,
    KnowledgeService,
    Paper,
    SearchResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class SurrealDBKnowledgeService(KnowledgeService):
    """
    SurrealDB implementation of the Knowledge Service.
    
    Features:
    - Multi-model: Document + Graph + Vector in one database
    - LIVE SELECT for real-time query subscriptions
    - SurrealQL graph traversal with -> and <- operators
    - MTREE vector indexes for similarity search
    - Record links for type-safe relationships
    - Native array and object support
    """
    
    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "oceanography",
        database: str = "knowledge",
        username: str = "root",
        password: str = "root",
    ):
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self._db = None
        self._live_queries: dict[str, str] = {}
    
    async def connect(self) -> None:
        """Connect to SurrealDB and initialize schema."""
        try:
            from surrealdb import Surreal
            
            self._db = Surreal(self.url)
            await self._db.connect()
            await self._db.signin({
                "user": self.username,
                "pass": self.password,
            })
            await self._db.use(self.namespace, self.database)
            await self._initialize_schema()
            logger.info(f"Connected to SurrealDB: {self.namespace}/{self.database}")
        except ImportError:
            logger.warning("surrealdb package not installed. Using mock mode.")
            self._db = None
        except Exception as e:
            logger.error(f"Failed to connect to SurrealDB: {e}")
            self._db = None
    
    async def disconnect(self) -> None:
        """Disconnect from SurrealDB."""
        if self._db:
            # Cancel all live queries
            for live_id in self._live_queries.values():
                try:
                    await self._db.query(f"KILL {live_id}")
                except Exception:
                    pass
            await self._db.close()
            self._db = None
    
    async def _initialize_schema(self) -> None:
        """Initialize SurrealDB schema with tables, indexes, and edges."""
        schema_queries = [
            # Define tables with schemafull mode for type safety
            """
            DEFINE TABLE paper SCHEMAFULL;
            DEFINE FIELD id ON paper TYPE string;
            DEFINE FIELD title ON paper TYPE string;
            DEFINE FIELD authors ON paper TYPE array<string>;
            DEFINE FIELD abstract ON paper TYPE string;
            DEFINE FIELD doi ON paper TYPE option<string>;
            DEFINE FIELD year ON paper TYPE int;
            DEFINE FIELD journal ON paper TYPE option<string>;
            DEFINE FIELD keywords ON paper TYPE array<string>;
            DEFINE FIELD embedding ON paper TYPE option<array<float>>;
            DEFINE FIELD created_at ON paper TYPE datetime DEFAULT time::now();
            DEFINE INDEX paper_id ON paper FIELDS id UNIQUE;
            DEFINE INDEX paper_year ON paper FIELDS year;
            DEFINE INDEX paper_keywords ON paper FIELDS keywords;
            """,
            
            # Vector index for paper embeddings (MTREE)
            """
            DEFINE INDEX paper_embedding_idx ON paper 
                FIELDS embedding MTREE DIMENSION 1536 
                DIST COSINE TYPE F32;
            """,
            
            # Full-text search index
            """
            DEFINE ANALYZER paper_analyzer TOKENIZERS blank,class 
                FILTERS lowercase, snowball(english);
            DEFINE INDEX paper_search ON paper 
                FIELDS title, abstract SEARCH ANALYZER paper_analyzer;
            """,
            
            # Historical events table
            """
            DEFINE TABLE event SCHEMAFULL;
            DEFINE FIELD id ON event TYPE string;
            DEFINE FIELD name ON event TYPE string;
            DEFINE FIELD description ON event TYPE string;
            DEFINE FIELD event_type ON event TYPE string;
            DEFINE FIELD start_date ON event TYPE datetime;
            DEFINE FIELD end_date ON event TYPE option<datetime>;
            DEFINE FIELD location ON event TYPE option<object>;
            DEFINE FIELD severity ON event TYPE option<float>;
            DEFINE FIELD source ON event TYPE option<string>;
            DEFINE FIELD created_at ON event TYPE datetime DEFAULT time::now();
            DEFINE INDEX event_id ON event FIELDS id UNIQUE;
            DEFINE INDEX event_type_idx ON event FIELDS event_type;
            DEFINE INDEX event_dates ON event FIELDS start_date, end_date;
            """,
            
            # Climate indices table
            """
            DEFINE TABLE climate_index SCHEMAFULL;
            DEFINE FIELD id ON climate_index TYPE string;
            DEFINE FIELD name ON climate_index TYPE string;
            DEFINE FIELD abbreviation ON climate_index TYPE string;
            DEFINE FIELD description ON climate_index TYPE string;
            DEFINE FIELD source_url ON climate_index TYPE option<string>;
            DEFINE FIELD time_series ON climate_index TYPE option<array<object>>;
            DEFINE FIELD created_at ON climate_index TYPE datetime DEFAULT time::now();
            DEFINE INDEX climate_id ON climate_index FIELDS id UNIQUE;
            DEFINE INDEX climate_abbrev ON climate_index FIELDS abbreviation;
            """,
            
            # Causal patterns table
            """
            DEFINE TABLE pattern SCHEMAFULL;
            DEFINE FIELD id ON pattern TYPE string;
            DEFINE FIELD name ON pattern TYPE string;
            DEFINE FIELD description ON pattern TYPE string;
            DEFINE FIELD pattern_type ON pattern TYPE string;
            DEFINE FIELD variables ON pattern TYPE array<string>;
            DEFINE FIELD lag_days ON pattern TYPE option<int>;
            DEFINE FIELD strength ON pattern TYPE option<float>;
            DEFINE FIELD confidence ON pattern TYPE option<float>;
            DEFINE FIELD metadata ON pattern TYPE option<object>;
            DEFINE FIELD created_at ON pattern TYPE datetime DEFAULT time::now();
            DEFINE INDEX pattern_id ON pattern FIELDS id UNIQUE;
            DEFINE INDEX pattern_type_idx ON pattern FIELDS pattern_type;
            DEFINE INDEX pattern_vars ON pattern FIELDS variables;
            """,
            
            # Edge tables for relationships (SurrealDB graph edges)
            """
            DEFINE TABLE validates SCHEMAFULL TYPE RELATION 
                FROM paper TO pattern;
            DEFINE FIELD validation_type ON validates TYPE string;
            DEFINE FIELD confidence ON validates TYPE float;
            DEFINE FIELD notes ON validates TYPE option<string>;
            DEFINE FIELD validated_at ON validates TYPE datetime DEFAULT time::now();
            """,
            
            """
            DEFINE TABLE observed_in SCHEMAFULL TYPE RELATION 
                FROM pattern TO event;
            DEFINE FIELD correlation ON observed_in TYPE option<float>;
            DEFINE FIELD lag_observed ON observed_in TYPE option<int>;
            DEFINE FIELD notes ON observed_in TYPE option<string>;
            """,
            
            """
            DEFINE TABLE causes SCHEMAFULL TYPE RELATION 
                FROM pattern TO pattern;
            DEFINE FIELD mechanism ON causes TYPE option<string>;
            DEFINE FIELD strength ON causes TYPE float;
            DEFINE FIELD lag_days ON causes TYPE option<int>;
            DEFINE FIELD evidence ON causes TYPE option<array<string>>;
            """,
            
            """
            DEFINE TABLE correlates_with SCHEMAFULL TYPE RELATION 
                FROM climate_index TO pattern;
            DEFINE FIELD correlation ON correlates_with TYPE float;
            DEFINE FIELD lag_months ON correlates_with TYPE option<int>;
            DEFINE FIELD period ON correlates_with TYPE option<string>;
            """,
        ]
        
        if self._db:
            for query in schema_queries:
                try:
                    await self._db.query(query)
                except Exception as e:
                    # Schema may already exist
                    logger.debug(f"Schema query note: {e}")
    
    # =========================================================================
    # Paper Operations
    # =========================================================================
    
    async def add_paper(self, paper: Paper) -> str:
        """Add a research paper to the knowledge base."""
        paper_id = paper.id or f"paper_{uuid4().hex[:12]}"
        
        if self._db:
            await self._db.query(
                """
                CREATE paper CONTENT {
                    id: $id,
                    title: $title,
                    authors: $authors,
                    abstract: $abstract,
                    doi: $doi,
                    year: $year,
                    journal: $journal,
                    keywords: $keywords,
                    embedding: $embedding
                }
                """,
                {
                    "id": paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "doi": paper.doi,
                    "year": paper.year,
                    "journal": paper.journal,
                    "keywords": paper.keywords,
                    "embedding": paper.embedding,
                }
            )
        
        return paper_id
    
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a paper by ID."""
        if not self._db:
            return None
        
        result = await self._db.query(
            "SELECT * FROM paper WHERE id = $id",
            {"id": paper_id}
        )
        
        if result and result[0]:
            data = result[0][0] if isinstance(result[0], list) else result[0]
            return Paper(
                id=data["id"],
                title=data["title"],
                authors=data["authors"],
                abstract=data["abstract"],
                doi=data.get("doi"),
                year=data["year"],
                journal=data.get("journal"),
                keywords=data.get("keywords", []),
                embedding=data.get("embedding"),
            )
        return None
    
    async def search_papers(
        self,
        query: str,
        embedding: Optional[list[float]] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search papers using vector similarity or full-text search."""
        results = []
        
        if not self._db:
            return results
        
        if embedding:
            # Vector similarity search using MTREE index
            surreal_results = await self._db.query(
                """
                SELECT *, vector::similarity::cosine(embedding, $embedding) AS score
                FROM paper
                WHERE embedding != NONE
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"embedding": embedding, "limit": limit}
            )
        else:
            # Full-text search
            surreal_results = await self._db.query(
                """
                SELECT *, search::score(1) AS score
                FROM paper
                WHERE title @1@ $query OR abstract @1@ $query
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query": query, "limit": limit}
            )
        
        if surreal_results and surreal_results[0]:
            for item in surreal_results[0]:
                paper = Paper(
                    id=item["id"],
                    title=item["title"],
                    authors=item["authors"],
                    abstract=item["abstract"],
                    doi=item.get("doi"),
                    year=item["year"],
                    journal=item.get("journal"),
                    keywords=item.get("keywords", []),
                )
                results.append(SearchResult(
                    item=paper,
                    score=item.get("score", 0.0),
                    source="surrealdb_vector" if embedding else "surrealdb_fulltext",
                ))
        
        return results
    
    async def bulk_add_papers(self, papers: list[Paper]) -> list[str]:
        """Add multiple papers efficiently."""
        if not self._db:
            return []
        
        paper_ids = []
        paper_data = []
        
        for paper in papers:
            paper_id = paper.id or f"paper_{uuid4().hex[:12]}"
            paper_ids.append(paper_id)
            paper_data.append({
                "id": paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "doi": paper.doi,
                "year": paper.year,
                "journal": paper.journal,
                "keywords": paper.keywords,
                "embedding": paper.embedding,
            })
        
        # SurrealDB supports bulk insert with INSERT
        await self._db.query(
            "INSERT INTO paper $papers",
            {"papers": paper_data}
        )
        
        return paper_ids
    
    # =========================================================================
    # Historical Event Operations
    # =========================================================================
    
    async def add_event(self, event: HistoricalEvent) -> str:
        """Add a historical event to the knowledge base."""
        event_id = event.id or f"event_{uuid4().hex[:12]}"
        
        if self._db:
            await self._db.query(
                """
                CREATE event CONTENT {
                    id: $id,
                    name: $name,
                    description: $description,
                    event_type: $event_type,
                    start_date: $start_date,
                    end_date: $end_date,
                    location: $location,
                    severity: $severity,
                    source: $source
                }
                """,
                {
                    "id": event_id,
                    "name": event.name,
                    "description": event.description,
                    "event_type": event.event_type,
                    "start_date": event.start_date.isoformat() if event.start_date else None,
                    "end_date": event.end_date.isoformat() if event.end_date else None,
                    "location": event.location,
                    "severity": event.severity,
                    "source": event.source,
                }
            )
        
        return event_id
    
    async def get_event(self, event_id: str) -> Optional[HistoricalEvent]:
        """Retrieve an event by ID."""
        if not self._db:
            return None
        
        result = await self._db.query(
            "SELECT * FROM event WHERE id = $id",
            {"id": event_id}
        )
        
        if result and result[0]:
            data = result[0][0] if isinstance(result[0], list) else result[0]
            return HistoricalEvent(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                event_type=data["event_type"],
                start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
                end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
                location=data.get("location"),
                severity=data.get("severity"),
                source=data.get("source"),
            )
        return None
    
    async def search_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        location: Optional[dict] = None,
        limit: int = 50,
    ) -> list[HistoricalEvent]:
        """Search for historical events with filters."""
        if not self._db:
            return []
        
        # Build dynamic query with conditions
        conditions = []
        params = {"limit": limit}
        
        if event_type:
            conditions.append("event_type = $event_type")
            params["event_type"] = event_type
        
        if start_date:
            conditions.append("start_date >= $start_date")
            params["start_date"] = start_date.isoformat()
        
        if end_date:
            conditions.append("(end_date <= $end_date OR start_date <= $end_date)")
            params["end_date"] = end_date.isoformat()
        
        if location:
            # Spatial query if location has coordinates
            if "lat" in location and "lon" in location:
                # SurrealDB geo distance query
                conditions.append(
                    "geo::distance(location.coordinates, $point) < $radius"
                )
                params["point"] = [location["lon"], location["lat"]]
                params["radius"] = location.get("radius_km", 500) * 1000  # meters
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        result = await self._db.query(
            f"""
            SELECT * FROM event
            WHERE {where_clause}
            ORDER BY start_date DESC
            LIMIT $limit
            """,
            params
        )
        
        events = []
        if result and result[0]:
            for data in result[0]:
                events.append(HistoricalEvent(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    event_type=data["event_type"],
                    start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
                    end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
                    location=data.get("location"),
                    severity=data.get("severity"),
                    source=data.get("source"),
                ))
        
        return events
    
    async def bulk_add_events(self, events: list[HistoricalEvent]) -> list[str]:
        """Add multiple events efficiently."""
        if not self._db:
            return []
        
        event_ids = []
        event_data = []
        
        for event in events:
            event_id = event.id or f"event_{uuid4().hex[:12]}"
            event_ids.append(event_id)
            event_data.append({
                "id": event_id,
                "name": event.name,
                "description": event.description,
                "event_type": event.event_type,
                "start_date": event.start_date.isoformat() if event.start_date else None,
                "end_date": event.end_date.isoformat() if event.end_date else None,
                "location": event.location,
                "severity": event.severity,
                "source": event.source,
            })
        
        await self._db.query(
            "INSERT INTO event $events",
            {"events": event_data}
        )
        
        return event_ids
    
    # =========================================================================
    # Climate Index Operations
    # =========================================================================
    
    async def add_climate_index(self, index: ClimateIndex) -> str:
        """Add a climate index to the knowledge base."""
        index_id = index.id or f"idx_{uuid4().hex[:12]}"
        
        if self._db:
            await self._db.query(
                """
                CREATE climate_index CONTENT {
                    id: $id,
                    name: $name,
                    abbreviation: $abbreviation,
                    description: $description,
                    source_url: $source_url,
                    time_series: $time_series
                }
                """,
                {
                    "id": index_id,
                    "name": index.name,
                    "abbreviation": index.abbreviation,
                    "description": index.description,
                    "source_url": index.source_url,
                    "time_series": index.time_series,
                }
            )
        
        return index_id
    
    async def get_climate_index(self, index_id: str) -> Optional[ClimateIndex]:
        """Retrieve a climate index by ID."""
        if not self._db:
            return None
        
        result = await self._db.query(
            "SELECT * FROM climate_index WHERE id = $id OR abbreviation = $id",
            {"id": index_id}
        )
        
        if result and result[0]:
            data = result[0][0] if isinstance(result[0], list) else result[0]
            return ClimateIndex(
                id=data["id"],
                name=data["name"],
                abbreviation=data["abbreviation"],
                description=data["description"],
                source_url=data.get("source_url"),
                time_series=data.get("time_series"),
            )
        return None
    
    async def list_climate_indices(self) -> list[ClimateIndex]:
        """List all available climate indices."""
        if not self._db:
            return []
        
        result = await self._db.query("SELECT * FROM climate_index ORDER BY name")
        
        indices = []
        if result and result[0]:
            for data in result[0]:
                indices.append(ClimateIndex(
                    id=data["id"],
                    name=data["name"],
                    abbreviation=data["abbreviation"],
                    description=data["description"],
                    source_url=data.get("source_url"),
                    time_series=data.get("time_series"),
                ))
        
        return indices
    
    # =========================================================================
    # Causal Pattern Operations
    # =========================================================================
    
    async def add_pattern(self, pattern: CausalPattern) -> str:
        """Add a causal pattern to the knowledge base."""
        pattern_id = pattern.id or f"pattern_{uuid4().hex[:12]}"
        
        if self._db:
            await self._db.query(
                """
                CREATE pattern CONTENT {
                    id: $id,
                    name: $name,
                    description: $description,
                    pattern_type: $pattern_type,
                    variables: $variables,
                    lag_days: $lag_days,
                    strength: $strength,
                    confidence: $confidence,
                    metadata: $metadata
                }
                """,
                {
                    "id": pattern_id,
                    "name": pattern.name,
                    "description": pattern.description,
                    "pattern_type": pattern.pattern_type,
                    "variables": pattern.variables,
                    "lag_days": pattern.lag_days,
                    "strength": pattern.strength,
                    "confidence": pattern.confidence,
                    "metadata": pattern.metadata,
                }
            )
        
        return pattern_id
    
    async def get_pattern(self, pattern_id: str) -> Optional[CausalPattern]:
        """Retrieve a pattern by ID."""
        if not self._db:
            return None
        
        result = await self._db.query(
            "SELECT * FROM pattern WHERE id = $id",
            {"id": pattern_id}
        )
        
        if result and result[0]:
            data = result[0][0] if isinstance(result[0], list) else result[0]
            return CausalPattern(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                pattern_type=data["pattern_type"],
                variables=data["variables"],
                lag_days=data.get("lag_days"),
                strength=data.get("strength"),
                confidence=data.get("confidence"),
                metadata=data.get("metadata"),
            )
        return None
    
    async def search_patterns(
        self,
        pattern_type: Optional[str] = None,
        variables: Optional[list[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[CausalPattern]:
        """Search for causal patterns."""
        if not self._db:
            return []
        
        conditions = ["(confidence >= $min_confidence OR confidence = NONE)"]
        params = {"min_confidence": min_confidence, "limit": limit}
        
        if pattern_type:
            conditions.append("pattern_type = $pattern_type")
            params["pattern_type"] = pattern_type
        
        if variables:
            # Check if any of the search variables appear in pattern variables
            conditions.append("array::any(variables, |$v| $v IN $search_vars)")
            params["search_vars"] = variables
        
        where_clause = " AND ".join(conditions)
        
        result = await self._db.query(
            f"""
            SELECT * FROM pattern
            WHERE {where_clause}
            ORDER BY confidence DESC
            LIMIT $limit
            """,
            params
        )
        
        patterns = []
        if result and result[0]:
            for data in result[0]:
                patterns.append(CausalPattern(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    pattern_type=data["pattern_type"],
                    variables=data["variables"],
                    lag_days=data.get("lag_days"),
                    strength=data.get("strength"),
                    confidence=data.get("confidence"),
                    metadata=data.get("metadata"),
                ))
        
        return patterns
    
    async def validate_pattern(
        self,
        pattern_id: str,
        paper_id: str,
        validation_type: str,
        confidence: float,
        notes: Optional[str] = None,
    ) -> ValidationResult:
        """Link a paper that validates a pattern."""
        if self._db:
            # Create VALIDATES edge using SurrealDB graph syntax
            await self._db.query(
                """
                RELATE (SELECT id FROM paper WHERE id = $paper_id)
                    ->validates->
                    (SELECT id FROM pattern WHERE id = $pattern_id)
                CONTENT {
                    validation_type: $validation_type,
                    confidence: $confidence,
                    notes: $notes
                }
                """,
                {
                    "paper_id": paper_id,
                    "pattern_id": pattern_id,
                    "validation_type": validation_type,
                    "confidence": confidence,
                    "notes": notes,
                }
            )
        
        return ValidationResult(
            pattern_id=pattern_id,
            is_valid=confidence > 0.5,
            confidence=confidence,
            supporting_papers=[paper_id],
            validation_type=validation_type,
            notes=notes,
        )
    
    async def bulk_add_patterns(self, patterns: list[CausalPattern]) -> list[str]:
        """Add multiple patterns efficiently."""
        if not self._db:
            return []
        
        pattern_ids = []
        pattern_data = []
        
        for pattern in patterns:
            pattern_id = pattern.id or f"pattern_{uuid4().hex[:12]}"
            pattern_ids.append(pattern_id)
            pattern_data.append({
                "id": pattern_id,
                "name": pattern.name,
                "description": pattern.description,
                "pattern_type": pattern.pattern_type,
                "variables": pattern.variables,
                "lag_days": pattern.lag_days,
                "strength": pattern.strength,
                "confidence": pattern.confidence,
                "metadata": pattern.metadata,
            })
        
        await self._db.query(
            "INSERT INTO pattern $patterns",
            {"patterns": pattern_data}
        )
        
        return pattern_ids
    
    # =========================================================================
    # Graph Traversal Operations (SurrealDB-specific features)
    # =========================================================================
    
    async def find_causal_chain(
        self,
        start_pattern_id: str,
        max_depth: int = 5,
    ) -> list[dict]:
        """
        Find causal chains from a starting pattern.
        Uses SurrealDB's native graph traversal with -> operator.
        """
        if not self._db:
            return []
        
        # SurrealDB graph traversal - find all downstream effects
        result = await self._db.query(
            f"""
            SELECT 
                id,
                name,
                ->causes[WHERE strength > 0.3]->pattern.* AS downstream,
                <-causes[WHERE strength > 0.3]<-pattern.* AS upstream
            FROM pattern
            WHERE id = $pattern_id
            """,
            {"pattern_id": start_pattern_id}
        )
        
        if not result or not result[0]:
            return []
        
        chains = []
        data = result[0][0] if isinstance(result[0], list) else result[0]
        
        # Build chain structure
        chains.append({
            "root": {
                "id": data["id"],
                "name": data["name"],
            },
            "downstream_effects": data.get("downstream", []),
            "upstream_causes": data.get("upstream", []),
        })
        
        return chains
    
    async def find_teleconnections(
        self,
        climate_index_id: str,
        min_correlation: float = 0.5,
    ) -> list[dict]:
        """
        Find teleconnections between climate indices and patterns.
        """
        if not self._db:
            return []
        
        result = await self._db.query(
            """
            SELECT 
                id,
                name,
                abbreviation,
                ->correlates_with[WHERE correlation >= $min_corr]->pattern.* AS patterns
            FROM climate_index
            WHERE id = $index_id OR abbreviation = $index_id
            """,
            {"index_id": climate_index_id, "min_corr": min_correlation}
        )
        
        teleconnections = []
        if result and result[0]:
            for data in result[0]:
                teleconnections.append({
                    "climate_index": {
                        "id": data["id"],
                        "name": data["name"],
                        "abbreviation": data["abbreviation"],
                    },
                    "correlated_patterns": data.get("patterns", []),
                })
        
        return teleconnections
    
    async def find_pattern_evidence(
        self,
        pattern_id: str,
    ) -> dict:
        """
        Find all evidence supporting a pattern: papers, events, climate correlations.
        """
        if not self._db:
            return {}
        
        result = await self._db.query(
            """
            SELECT 
                id,
                name,
                description,
                <-validates<-paper.* AS validating_papers,
                ->observed_in->event.* AS observed_events,
                <-correlates_with<-climate_index.* AS climate_correlations
            FROM pattern
            WHERE id = $pattern_id
            """,
            {"pattern_id": pattern_id}
        )
        
        if not result or not result[0]:
            return {}
        
        data = result[0][0] if isinstance(result[0], list) else result[0]
        
        return {
            "pattern": {
                "id": data["id"],
                "name": data["name"],
                "description": data["description"],
            },
            "evidence": {
                "papers": data.get("validating_papers", []),
                "historical_events": data.get("observed_events", []),
                "climate_correlations": data.get("climate_correlations", []),
            },
            "total_evidence_count": (
                len(data.get("validating_papers", [])) +
                len(data.get("observed_events", [])) +
                len(data.get("climate_correlations", []))
            ),
        }
    
    # =========================================================================
    # Real-time Subscriptions (SurrealDB LIVE SELECT)
    # =========================================================================
    
    async def subscribe_to_patterns(
        self,
        pattern_type: Optional[str] = None,
    ) -> AsyncIterator[CausalPattern]:
        """
        Subscribe to real-time pattern updates using LIVE SELECT.
        
        This is a unique SurrealDB feature - get notified when patterns change.
        """
        if not self._db:
            return
        
        # Build LIVE SELECT query
        if pattern_type:
            query = f"LIVE SELECT * FROM pattern WHERE pattern_type = '{pattern_type}'"
        else:
            query = "LIVE SELECT * FROM pattern"
        
        # Execute LIVE SELECT and get the live query ID
        result = await self._db.query(query)
        
        if result:
            live_id = result[0]
            self._live_queries["patterns"] = live_id
            
            # In a real implementation, this would yield updates as they come
            # For now, return a placeholder that can be iterated
            logger.info(f"Started LIVE SELECT subscription: {live_id}")
    
    async def subscribe_to_events(
        self,
        event_type: Optional[str] = None,
    ) -> AsyncIterator[HistoricalEvent]:
        """Subscribe to real-time event updates."""
        if not self._db:
            return
        
        if event_type:
            query = f"LIVE SELECT * FROM event WHERE event_type = '{event_type}'"
        else:
            query = "LIVE SELECT * FROM event"
        
        result = await self._db.query(query)
        
        if result:
            live_id = result[0]
            self._live_queries["events"] = live_id
            logger.info(f"Started LIVE SELECT subscription for events: {live_id}")
    
    async def unsubscribe(self, subscription_name: str) -> None:
        """Cancel a LIVE SELECT subscription."""
        if self._db and subscription_name in self._live_queries:
            live_id = self._live_queries[subscription_name]
            await self._db.query(f"KILL {live_id}")
            del self._live_queries[subscription_name]
            logger.info(f"Cancelled subscription: {subscription_name}")
    
    # =========================================================================
    # Link Operations
    # =========================================================================
    
    async def link_pattern_to_event(
        self,
        pattern_id: str,
        event_id: str,
        correlation: Optional[float] = None,
        lag_observed: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Create OBSERVED_IN relationship between pattern and event."""
        if not self._db:
            return False
        
        await self._db.query(
            """
            RELATE (SELECT id FROM pattern WHERE id = $pattern_id)
                ->observed_in->
                (SELECT id FROM event WHERE id = $event_id)
            CONTENT {
                correlation: $correlation,
                lag_observed: $lag_observed,
                notes: $notes
            }
            """,
            {
                "pattern_id": pattern_id,
                "event_id": event_id,
                "correlation": correlation,
                "lag_observed": lag_observed,
                "notes": notes,
            }
        )
        return True
    
    async def link_patterns_causal(
        self,
        cause_pattern_id: str,
        effect_pattern_id: str,
        strength: float,
        mechanism: Optional[str] = None,
        lag_days: Optional[int] = None,
        evidence: Optional[list[str]] = None,
    ) -> bool:
        """Create CAUSES relationship between patterns."""
        if not self._db:
            return False
        
        await self._db.query(
            """
            RELATE (SELECT id FROM pattern WHERE id = $cause_id)
                ->causes->
                (SELECT id FROM pattern WHERE id = $effect_id)
            CONTENT {
                strength: $strength,
                mechanism: $mechanism,
                lag_days: $lag_days,
                evidence: $evidence
            }
            """,
            {
                "cause_id": cause_pattern_id,
                "effect_id": effect_pattern_id,
                "strength": strength,
                "mechanism": mechanism,
                "lag_days": lag_days,
                "evidence": evidence or [],
            }
        )
        return True
    
    async def link_index_to_pattern(
        self,
        index_id: str,
        pattern_id: str,
        correlation: float,
        lag_months: Optional[int] = None,
        period: Optional[str] = None,
    ) -> bool:
        """Create CORRELATES_WITH relationship between climate index and pattern."""
        if not self._db:
            return False
        
        await self._db.query(
            """
            RELATE (SELECT id FROM climate_index WHERE id = $index_id)
                ->correlates_with->
                (SELECT id FROM pattern WHERE id = $pattern_id)
            CONTENT {
                correlation: $correlation,
                lag_months: $lag_months,
                period: $period
            }
            """,
            {
                "index_id": index_id,
                "pattern_id": pattern_id,
                "correlation": correlation,
                "lag_months": lag_months,
                "period": period,
            }
        )
        return True
    
    # =========================================================================
    # Statistics and Metrics
    # =========================================================================
    
    async def get_statistics(self) -> dict:
        """Get knowledge base statistics."""
        if not self._db:
            return {
                "papers": 0,
                "events": 0,
                "climate_indices": 0,
                "patterns": 0,
                "relationships": 0,
            }
        
        result = await self._db.query(
            """
            RETURN {
                papers: (SELECT count() FROM paper GROUP ALL).count,
                events: (SELECT count() FROM event GROUP ALL).count,
                climate_indices: (SELECT count() FROM climate_index GROUP ALL).count,
                patterns: (SELECT count() FROM pattern GROUP ALL).count,
                validates: (SELECT count() FROM validates GROUP ALL).count,
                observed_in: (SELECT count() FROM observed_in GROUP ALL).count,
                causes: (SELECT count() FROM causes GROUP ALL).count,
                correlates_with: (SELECT count() FROM correlates_with GROUP ALL).count
            }
            """
        )
        
        if result and result[0]:
            data = result[0]
            return {
                "papers": data.get("papers", 0) or 0,
                "events": data.get("events", 0) or 0,
                "climate_indices": data.get("climate_indices", 0) or 0,
                "patterns": data.get("patterns", 0) or 0,
                "relationships": (
                    (data.get("validates", 0) or 0) +
                    (data.get("observed_in", 0) or 0) +
                    (data.get("causes", 0) or 0) +
                    (data.get("correlates_with", 0) or 0)
                ),
            }
        
        return {
            "papers": 0,
            "events": 0,
            "climate_indices": 0,
            "patterns": 0,
            "relationships": 0,
        }

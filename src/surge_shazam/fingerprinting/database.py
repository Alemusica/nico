"""
🗄️ Fingerprint Database - SurrealDB Storage
============================================

Store and retrieve fingerprints using SurrealDB with HNSW vector index
for fast similarity search.

Integrates with:
- CTW schema (docs/SURREALDB_SCHEMA.md)
- Existing surrealdb_knowledge.py service
- FingerprintEngine (engine.py)

Features:
- HNSW vector index for 100-dim MiniRocket embeddings
- Link fingerprints to events, observations, alerts
- Similarity search with configurable threshold
- Batch operations for efficiency

Usage:
    from src.surge_shazam.fingerprinting.database import FingerprintDB
    
    db = FingerprintDB()
    await db.connect()
    
    # Store fingerprint
    fp_id = await db.store(fingerprint)
    
    # Search similar
    results = await db.search_similar(query_fp, top_k=10)
    
    # Link to event
    await db.link_to_event(fp_id, event_id)

Requirements:
    pip install surrealdb numpy
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from uuid import uuid4
import json

import numpy as np

from .engine import Fingerprint, SimilarityResult

logger = logging.getLogger(__name__)

# Check for surrealdb
try:
    from surrealdb import Surreal
    HAS_SURREALDB = True
except ImportError:
    HAS_SURREALDB = False
    Surreal = None
    logger.warning("⚠️ surrealdb not installed. Install with: pip install surrealdb")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "url": "ws://localhost:8000/rpc",
    "namespace": "causal",
    "database": "knowledge",
    "username": "root",
    "password": "root",
}


# =============================================================================
# FINGERPRINT DATABASE
# =============================================================================

class FingerprintDB:
    """
    SurrealDB storage for fingerprints with vector similarity search.
    
    Uses HNSW index for fast approximate nearest neighbor search on
    MiniRocket embeddings (100 dimensions by default).
    """
    
    def __init__(
        self,
        url: str = None,
        namespace: str = None,
        database: str = None,
        username: str = None,
        password: str = None,
        embedding_dim: int = 100,
    ):
        """
        Initialize fingerprint database.
        
        Args:
            url: SurrealDB WebSocket URL
            namespace: SurrealDB namespace
            database: SurrealDB database name
            username: Auth username
            password: Auth password
            embedding_dim: Expected embedding dimension (for index)
        """
        self.url = url or DEFAULT_CONFIG["url"]
        self.namespace = namespace or DEFAULT_CONFIG["namespace"]
        self.database = database or DEFAULT_CONFIG["database"]
        self.username = username or DEFAULT_CONFIG["username"]
        self.password = password or DEFAULT_CONFIG["password"]
        self.embedding_dim = embedding_dim
        
        self._db: Optional[Any] = None
        self._connected = False
        self._schema_initialized = False
        
        # In-memory fallback
        self._memory_store: Dict[str, Fingerprint] = {}
        self._memory_mode = False
    
    async def connect(self) -> bool:
        """
        Connect to SurrealDB.
        
        Returns:
            True if connected, False if using memory fallback
        """
        if self._connected:
            return not self._memory_mode
        
        if not HAS_SURREALDB:
            logger.warning("SurrealDB not available, using in-memory storage")
            self._memory_mode = True
            self._connected = True
            return False
        
        try:
            # SurrealDB sync client
            def connect_sync():
                db = Surreal(self.url)
                db.signin({"user": self.username, "pass": self.password})
                db.use(self.namespace, self.database)
                return db
            
            self._db = await asyncio.to_thread(connect_sync)
            self._connected = True
            self._memory_mode = False
            
            logger.info(f"✅ Connected to SurrealDB: {self.namespace}/{self.database}")
            
            # Initialize schema if needed
            if not self._schema_initialized:
                await self._init_schema()
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to SurrealDB: {e}. Using memory fallback.")
            self._memory_mode = True
            self._connected = True
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from SurrealDB."""
        if self._db:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
        self._connected = False
    
    async def _query(self, query: str, params: Optional[dict] = None) -> Any:
        """Execute a SurrealDB query."""
        if not self._db:
            return None
        return await asyncio.to_thread(self._db.query, query, params or {})
    
    async def _init_schema(self) -> None:
        """Initialize fingerprint table and HNSW index."""
        if self._memory_mode:
            return
        
        schema = f"""
        -- Fingerprint table (if not exists from seed_ctw_schema.py)
        DEFINE TABLE IF NOT EXISTS fingerprint SCHEMAFULL;
        DEFINE FIELD IF NOT EXISTS id ON fingerprint TYPE string;
        DEFINE FIELD IF NOT EXISTS event_id ON fingerprint TYPE string;
        DEFINE FIELD IF NOT EXISTS embedding ON fingerprint TYPE array<float>;
        DEFINE FIELD IF NOT EXISTS timestamp_start ON fingerprint TYPE datetime;
        DEFINE FIELD IF NOT EXISTS timestamp_end ON fingerprint TYPE datetime;
        DEFINE FIELD IF NOT EXISTS variables_used ON fingerprint TYPE array<string>;
        DEFINE FIELD IF NOT EXISTS sources_used ON fingerprint TYPE array<string>;
        DEFINE FIELD IF NOT EXISTS bbox ON fingerprint TYPE option<array<float>>;
        DEFINE FIELD IF NOT EXISTS center ON fingerprint TYPE option<array<float>>;
        DEFINE FIELD IF NOT EXISTS n_samples ON fingerprint TYPE int;
        DEFINE FIELD IF NOT EXISTS n_variables ON fingerprint TYPE int;
        DEFINE FIELD IF NOT EXISTS missing_ratio ON fingerprint TYPE float;
        DEFINE FIELD IF NOT EXISTS uncertainty ON fingerprint TYPE float;
        DEFINE FIELD IF NOT EXISTS extraction_method ON fingerprint TYPE string;
        DEFINE FIELD IF NOT EXISTS variable_stats ON fingerprint TYPE option<object>;
        DEFINE FIELD IF NOT EXISTS metadata ON fingerprint TYPE option<object>;
        DEFINE FIELD IF NOT EXISTS created_at ON fingerprint TYPE datetime DEFAULT time::now();
        
        -- Indexes
        DEFINE INDEX IF NOT EXISTS fp_id ON fingerprint FIELDS id UNIQUE;
        DEFINE INDEX IF NOT EXISTS fp_event ON fingerprint FIELDS event_id;
        DEFINE INDEX IF NOT EXISTS fp_method ON fingerprint FIELDS extraction_method;
        DEFINE INDEX IF NOT EXISTS fp_time ON fingerprint FIELDS timestamp_start, timestamp_end;
        
        -- HNSW vector index for similarity search
        DEFINE INDEX IF NOT EXISTS fp_embedding_hnsw ON fingerprint 
            FIELDS embedding HNSW DIMENSION {self.embedding_dim} 
            DIST COSINE TYPE F32;
        """
        
        try:
            await self._query(schema)
            self._schema_initialized = True
            logger.info("✅ Fingerprint schema initialized")
        except Exception as e:
            logger.warning(f"Schema init warning: {e}")
            # May already exist, continue anyway
            self._schema_initialized = True
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    async def store(self, fingerprint: Fingerprint) -> str:
        """
        Store a fingerprint.
        
        Args:
            fingerprint: Fingerprint object to store
            
        Returns:
            Fingerprint ID
        """
        await self.connect()
        
        fp_id = fingerprint.event_id or f"fp_{uuid4().hex[:12]}"
        
        if self._memory_mode:
            self._memory_store[fp_id] = fingerprint
            return fp_id
        
        # Prepare data
        data = {
            "id": fp_id,
            "event_id": fingerprint.event_id,
            "embedding": fingerprint.embedding.tolist(),
            "timestamp_start": fingerprint.timestamp_start.isoformat(),
            "timestamp_end": fingerprint.timestamp_end.isoformat(),
            "variables_used": fingerprint.variables_used,
            "sources_used": fingerprint.sources_used,
            "bbox": list(fingerprint.bbox) if fingerprint.bbox else None,
            "center": list(fingerprint.center) if fingerprint.center else None,
            "n_samples": fingerprint.n_samples,
            "n_variables": fingerprint.n_variables,
            "missing_ratio": fingerprint.missing_ratio,
            "uncertainty": fingerprint.uncertainty,
            "extraction_method": fingerprint.extraction_method,
            "variable_stats": fingerprint.variable_stats,
            "metadata": fingerprint.metadata,
        }
        
        try:
            await self._query(
                """
                CREATE fingerprint CONTENT $data
                """,
                {"data": data}
            )
            logger.debug(f"Stored fingerprint: {fp_id}")
            return fp_id
        except Exception as e:
            logger.error(f"Failed to store fingerprint: {e}")
            # Fallback to memory
            self._memory_store[fp_id] = fingerprint
            return fp_id
    
    async def get(self, fp_id: str) -> Optional[Fingerprint]:
        """
        Retrieve a fingerprint by ID.
        
        Args:
            fp_id: Fingerprint ID
            
        Returns:
            Fingerprint object or None
        """
        await self.connect()
        
        if self._memory_mode:
            return self._memory_store.get(fp_id)
        
        try:
            result = await self._query(
                "SELECT * FROM fingerprint WHERE id = $id",
                {"id": fp_id}
            )
            
            if result and len(result) > 0:
                data = result[0] if isinstance(result[0], dict) else result[0][0]
                return self._parse_fingerprint(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get fingerprint: {e}")
            return self._memory_store.get(fp_id)
    
    async def delete(self, fp_id: str) -> bool:
        """
        Delete a fingerprint.
        
        Args:
            fp_id: Fingerprint ID
            
        Returns:
            True if deleted
        """
        await self.connect()
        
        if self._memory_mode:
            if fp_id in self._memory_store:
                del self._memory_store[fp_id]
                return True
            return False
        
        try:
            await self._query(
                "DELETE FROM fingerprint WHERE id = $id",
                {"id": fp_id}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete fingerprint: {e}")
            return False
    
    async def list_all(self, limit: int = 100) -> List[Fingerprint]:
        """
        List all fingerprints.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of Fingerprint objects
        """
        await self.connect()
        
        if self._memory_mode:
            return list(self._memory_store.values())[:limit]
        
        try:
            result = await self._query(
                "SELECT * FROM fingerprint ORDER BY created_at DESC LIMIT $limit",
                {"limit": limit}
            )
            
            fingerprints = []
            if result:
                for row in result:
                    if isinstance(row, dict):
                        fingerprints.append(self._parse_fingerprint(row))
                    elif isinstance(row, list):
                        for item in row:
                            fingerprints.append(self._parse_fingerprint(item))
            
            return fingerprints
            
        except Exception as e:
            logger.error(f"Failed to list fingerprints: {e}")
            return list(self._memory_store.values())[:limit]
    
    # =========================================================================
    # SIMILARITY SEARCH
    # =========================================================================
    
    async def search_similar(
        self,
        query: Fingerprint,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[Fingerprint, float]]:
        """
        Search for similar fingerprints using vector similarity.
        
        Args:
            query: Query fingerprint
            top_k: Number of results
            min_similarity: Minimum similarity threshold [0, 1]
            
        Returns:
            List of (Fingerprint, similarity_score) tuples
        """
        await self.connect()
        
        if self._memory_mode:
            return self._memory_search(query, top_k, min_similarity)
        
        try:
            # Use SurrealDB vector search
            result = await self._query(
                """
                SELECT *, 
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM fingerprint
                WHERE id != $query_id
                ORDER BY similarity DESC
                LIMIT $limit
                """,
                {
                    "embedding": query.embedding.tolist(),
                    "query_id": query.event_id,
                    "limit": top_k,
                }
            )
            
            results = []
            if result:
                for row in result:
                    if isinstance(row, dict):
                        sim = row.get("similarity", 0.0)
                        # Convert cosine similarity from [-1, 1] to [0, 1]
                        sim_normalized = (sim + 1) / 2
                        
                        if sim_normalized >= min_similarity:
                            fp = self._parse_fingerprint(row)
                            results.append((fp, sim_normalized))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}. Using memory fallback.")
            return self._memory_search(query, top_k, min_similarity)
    
    def _memory_search(
        self,
        query: Fingerprint,
        top_k: int,
        min_similarity: float,
    ) -> List[Tuple[Fingerprint, float]]:
        """In-memory similarity search fallback."""
        results = []
        
        for fp_id, fp in self._memory_store.items():
            if fp_id == query.event_id:
                continue
            
            # Cosine similarity
            sim = self._cosine_similarity(query.embedding, fp.embedding)
            
            if sim >= min_similarity:
                results.append((fp, sim))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
        
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        sim = dot / (norm_a * norm_b)
        # Convert from [-1, 1] to [0, 1]
        return float((sim + 1) / 2)
    
    # =========================================================================
    # LINKING OPERATIONS
    # =========================================================================
    
    async def link_to_event(
        self,
        fp_id: str,
        event_id: str,
        match_type: str = "precursor",
        confidence: float = 1.0,
    ) -> bool:
        """
        Link a fingerprint to a historical event.
        
        Args:
            fp_id: Fingerprint ID
            event_id: Event ID
            match_type: Type of match (precursor, during, after)
            confidence: Match confidence
            
        Returns:
            True if linked
        """
        if self._memory_mode:
            return True  # No-op for memory mode
        
        try:
            await self._query(
                """
                RELATE fingerprint:$fp_id->matched_event->event:$event_id
                CONTENT {
                    match_type: $match_type,
                    confidence: $confidence,
                    linked_at: time::now()
                }
                """,
                {
                    "fp_id": fp_id,
                    "event_id": event_id,
                    "match_type": match_type,
                    "confidence": confidence,
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to link fingerprint to event: {e}")
            return False
    
    async def link_to_observation(
        self,
        fp_id: str,
        observation_id: str,
    ) -> bool:
        """
        Link a fingerprint to source observations.
        
        Args:
            fp_id: Fingerprint ID
            observation_id: Observation ID
            
        Returns:
            True if linked
        """
        if self._memory_mode:
            return True
        
        try:
            await self._query(
                """
                RELATE observation:$obs_id->has_fingerprint->fingerprint:$fp_id
                CONTENT { extracted_at: time::now() }
                """,
                {"obs_id": observation_id, "fp_id": fp_id}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to link observation to fingerprint: {e}")
            return False
    
    async def link_to_alert(
        self,
        fp_id: str,
        alert_id: str,
        similarity: float,
    ) -> bool:
        """
        Link a fingerprint match to an alert.
        
        Args:
            fp_id: Fingerprint ID that triggered the alert
            alert_id: Alert ID
            similarity: Match similarity score
            
        Returns:
            True if linked
        """
        if self._memory_mode:
            return True
        
        try:
            await self._query(
                """
                RELATE alert:$alert_id->triggered_by->fingerprint:$fp_id
                CONTENT {
                    similarity: $similarity,
                    triggered_at: time::now()
                }
                """,
                {"alert_id": alert_id, "fp_id": fp_id, "similarity": similarity}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to link alert to fingerprint: {e}")
            return False
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    async def store_batch(self, fingerprints: List[Fingerprint]) -> List[str]:
        """
        Store multiple fingerprints efficiently.
        
        Args:
            fingerprints: List of Fingerprint objects
            
        Returns:
            List of fingerprint IDs
        """
        await self.connect()
        
        fp_ids = []
        batch_data = []
        
        for fp in fingerprints:
            fp_id = fp.event_id or f"fp_{uuid4().hex[:12]}"
            fp_ids.append(fp_id)
            
            if self._memory_mode:
                self._memory_store[fp_id] = fp
            else:
                batch_data.append({
                    "id": fp_id,
                    "event_id": fp.event_id,
                    "embedding": fp.embedding.tolist(),
                    "timestamp_start": fp.timestamp_start.isoformat(),
                    "timestamp_end": fp.timestamp_end.isoformat(),
                    "variables_used": fp.variables_used,
                    "sources_used": fp.sources_used,
                    "bbox": list(fp.bbox) if fp.bbox else None,
                    "center": list(fp.center) if fp.center else None,
                    "n_samples": fp.n_samples,
                    "n_variables": fp.n_variables,
                    "missing_ratio": fp.missing_ratio,
                    "uncertainty": fp.uncertainty,
                    "extraction_method": fp.extraction_method,
                    "variable_stats": fp.variable_stats,
                    "metadata": fp.metadata,
                })
        
        if not self._memory_mode and batch_data:
            try:
                await self._query(
                    "INSERT INTO fingerprint $data",
                    {"data": batch_data}
                )
                logger.info(f"Stored {len(batch_data)} fingerprints")
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                # Fallback to individual inserts
                for fp, fp_id in zip(fingerprints, fp_ids):
                    self._memory_store[fp_id] = fp
        
        return fp_ids
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with stats
        """
        await self.connect()
        
        if self._memory_mode:
            return {
                "total_fingerprints": len(self._memory_store),
                "storage": "memory",
                "connected": False,
            }
        
        try:
            result = await self._query(
                """
                SELECT 
                    count() AS total,
                    math::mean(n_samples) AS avg_samples,
                    math::mean(uncertainty) AS avg_uncertainty
                FROM fingerprint
                GROUP ALL
                """
            )
            
            if result and len(result) > 0:
                data = result[0] if isinstance(result[0], dict) else {}
                return {
                    "total_fingerprints": data.get("total", 0),
                    "avg_samples": data.get("avg_samples", 0),
                    "avg_uncertainty": data.get("avg_uncertainty", 0),
                    "storage": "surrealdb",
                    "connected": True,
                }
            
            return {
                "total_fingerprints": 0,
                "storage": "surrealdb",
                "connected": True,
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_fingerprints": len(self._memory_store),
                "storage": "memory_fallback",
                "error": str(e),
            }
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _parse_fingerprint(self, data: Dict) -> Fingerprint:
        """Parse SurrealDB row to Fingerprint object."""
        # Handle SurrealDB id format (fingerprint:xxx -> xxx)
        fp_id = str(data.get("id", ""))
        if ":" in fp_id:
            fp_id = fp_id.split(":")[-1]
        
        # Parse timestamps
        ts_start = data.get("timestamp_start")
        if isinstance(ts_start, str):
            ts_start = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
        elif not isinstance(ts_start, datetime):
            ts_start = datetime.now()
        
        ts_end = data.get("timestamp_end")
        if isinstance(ts_end, str):
            ts_end = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))
        elif not isinstance(ts_end, datetime):
            ts_end = datetime.now()
        
        # Parse embedding
        embedding = data.get("embedding", [])
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Parse bbox and center
        bbox = data.get("bbox")
        if bbox and isinstance(bbox, list):
            bbox = tuple(bbox)
        
        center = data.get("center")
        if center and isinstance(center, list):
            center = tuple(center)
        
        return Fingerprint(
            event_id=data.get("event_id", fp_id),
            embedding=embedding,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            variables_used=data.get("variables_used", []),
            sources_used=data.get("sources_used", []),
            bbox=bbox,
            center=center,
            n_samples=data.get("n_samples", 0),
            n_variables=data.get("n_variables", 0),
            missing_ratio=data.get("missing_ratio", 0.0),
            uncertainty=data.get("uncertainty", 0.0),
            variable_stats=data.get("variable_stats", {}),
            extraction_method=data.get("extraction_method", "unknown"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_db: Optional[FingerprintDB] = None


async def get_fingerprint_db() -> FingerprintDB:
    """Get default fingerprint database instance."""
    global _default_db
    if _default_db is None:
        _default_db = FingerprintDB()
        await _default_db.connect()
    return _default_db


async def store_fingerprint(fp: Fingerprint) -> str:
    """Quick store fingerprint."""
    db = await get_fingerprint_db()
    return await db.store(fp)


async def search_fingerprints(
    query: Fingerprint,
    top_k: int = 10,
) -> List[Tuple[Fingerprint, float]]:
    """Quick similarity search."""
    db = await get_fingerprint_db()
    return await db.search_similar(query, top_k)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    from .engine import FingerprintEngine
    
    async def test():
        print("=== 🗄️ Fingerprint Database Test ===\n")
        
        # Initialize
        db = FingerprintDB()
        connected = await db.connect()
        print(f"Connected to SurrealDB: {connected}")
        print(f"Using memory mode: {db._memory_mode}")
        
        # Create test fingerprints
        engine = FingerprintEngine(n_kernels=1000, embedding_dim=50)
        
        np.random.seed(42)
        dates = pd.date_range("2000-10-01", periods=100, freq="D")
        
        df1 = pd.DataFrame({
            "precipitation": np.random.exponential(5, 100),
            "pressure": 101325 - np.random.exponential(1000, 100),
        }, index=dates)
        
        df2 = pd.DataFrame({
            "precipitation": np.random.exponential(1, 100),
            "pressure": 101325 + np.random.exponential(500, 100),
        }, index=dates)
        
        fp1 = engine.extract(df1, "flood_event", sources=["era5"])
        fp2 = engine.extract(df2, "dry_event", sources=["era5"])
        
        # Store
        print("\n📥 Storing fingerprints...")
        id1 = await db.store(fp1)
        id2 = await db.store(fp2)
        print(f"   Stored: {id1}, {id2}")
        
        # Retrieve
        print("\n📤 Retrieving...")
        fp_retrieved = await db.get(id1)
        print(f"   Retrieved: {fp_retrieved.event_id if fp_retrieved else 'None'}")
        
        # Search
        print("\n🔍 Searching similar to flood_event...")
        results = await db.search_similar(fp1, top_k=5)
        for fp, sim in results:
            print(f"   {fp.event_id}: similarity={sim:.4f}")
        
        # Stats
        print("\n📊 Stats:")
        stats = await db.get_stats()
        for k, v in stats.items():
            print(f"   {k}: {v}")
        
        await db.disconnect()
        print("\n✅ Test complete!")
    
    asyncio.run(test())

"""
Causal Graph Storage - SurrealDB

Store and query physics-validated causal chains discovered by PCMCI.
Integrates with existing pcmci_engine.py results.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

try:
    from surrealdb import Surreal
    HAS_SURREALDB = True
except ImportError:
    HAS_SURREALDB = False
    Surreal = None


@dataclass
class CausalEdge:
    """A causal relationship between dataset variables."""
    source_dataset: str
    source_variable: str
    target_dataset: str
    target_variable: str
    lag_days: int
    correlation: float
    physics_mechanism: str  # "teleconnection", "hydrology", "air_sea_interaction"
    physics_score: float    # 0-1, from PhysicsValidator
    discovered_by: str      # "pcmci", "cross_region", "manual", "literature"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def source_key(self) -> str:
        return f"{self.source_dataset}.{self.source_variable}"
    
    @property
    def target_key(self) -> str:
        return f"{self.target_dataset}.{self.target_variable}"


class CausalGraphDB:
    """
    Store and query causal chains in SurrealDB.
    
    Schema:
    - dataset: nodes representing data sources
    - causal_edge: directed edges with lag, correlation, physics validation
    """
    
    def __init__(self, url: str = "ws://localhost:8001/rpc"):
        if not HAS_SURREALDB:
            raise ImportError("surrealdb package required: pip install surrealdb")
        
        self.url = url
        self.db = Surreal(url)
        self._connected = False
        self._initialized = False
    
    async def connect(self):
        """Connect to SurrealDB and initialize schema."""
        if self._connected:
            return
        
        await self.db.connect()
        await self.db.use("causal", "knowledge")
        self._connected = True
        
        if not self._initialized:
            await self._init_schema()
            self._initialized = True
    
    async def _init_schema(self):
        """Create tables if not exist."""
        # Dataset nodes
        await self.db.query("""
            DEFINE TABLE IF NOT EXISTS dataset SCHEMAFULL;
            DEFINE FIELD id ON dataset TYPE string;
            DEFINE FIELD name ON dataset TYPE string;
            DEFINE FIELD variables ON dataset TYPE array;
            DEFINE FIELD latency ON dataset TYPE string;
            DEFINE FIELD latency_badge ON dataset TYPE string;
            DEFINE FIELD provider ON dataset TYPE string;
            DEFINE FIELD created_at ON dataset TYPE datetime DEFAULT time::now();
        """)
        
        # Causal edges
        await self.db.query("""
            DEFINE TABLE IF NOT EXISTS causal_edge SCHEMAFULL;
            DEFINE FIELD source ON causal_edge TYPE string;
            DEFINE FIELD target ON causal_edge TYPE string;
            DEFINE FIELD source_var ON causal_edge TYPE string;
            DEFINE FIELD target_var ON causal_edge TYPE string;
            DEFINE FIELD lag_days ON causal_edge TYPE int;
            DEFINE FIELD correlation ON causal_edge TYPE float;
            DEFINE FIELD physics_mechanism ON causal_edge TYPE string;
            DEFINE FIELD physics_score ON causal_edge TYPE float;
            DEFINE FIELD discovered_by ON causal_edge TYPE string;
            DEFINE FIELD created_at ON causal_edge TYPE datetime DEFAULT time::now();
            DEFINE FIELD metadata ON causal_edge TYPE object;
        """)
        
        # Indexes for fast queries
        await self.db.query("""
            DEFINE INDEX idx_edge_source ON causal_edge FIELDS source;
            DEFINE INDEX idx_edge_target ON causal_edge FIELDS target;
            DEFINE INDEX idx_edge_score ON causal_edge FIELDS physics_score;
        """)
    
    async def add_dataset(self, dataset_id: str, metadata: Dict[str, Any]):
        """Add or update a dataset node."""
        await self.connect()
        
        await self.db.query("""
            UPSERT dataset:$id SET
                id = $id,
                name = $name,
                variables = $vars,
                latency = $latency,
                latency_badge = $badge,
                provider = $provider
        """, {
            "id": dataset_id,
            "name": metadata.get("name", dataset_id),
            "vars": metadata.get("variables", []),
            "latency": metadata.get("latency"),
            "badge": metadata.get("latency_badge"),
            "provider": metadata.get("provider"),
        })
    
    async def add_edge(self, edge: CausalEdge, metadata: Dict[str, Any] = None):
        """Add a causal edge between dataset variables."""
        await self.connect()
        
        result = await self.db.query("""
            CREATE causal_edge SET
                source = $src,
                target = $tgt,
                source_var = $src_var,
                target_var = $tgt_var,
                lag_days = $lag,
                correlation = $corr,
                physics_mechanism = $mechanism,
                physics_score = $score,
                discovered_by = $by,
                metadata = $meta
        """, {
            "src": edge.source_dataset,
            "tgt": edge.target_dataset,
            "src_var": edge.source_variable,
            "tgt_var": edge.target_variable,
            "lag": edge.lag_days,
            "corr": edge.correlation,
            "mechanism": edge.physics_mechanism,
            "score": edge.physics_score,
            "by": edge.discovered_by,
            "meta": metadata or {},
        })
        return result
    
    async def get_precursors(
        self, 
        target_dataset: str, 
        target_variable: str = None,
        min_score: float = 0.5,
    ) -> List[Dict]:
        """
        Find what causes a target (precursors/drivers).
        
        Args:
            target_dataset: Target dataset ID
            target_variable: Optional specific variable
            min_score: Minimum physics validation score
        
        Returns:
            List of causal edges pointing to target
        """
        await self.connect()
        
        if target_variable:
            result = await self.db.query("""
                SELECT * FROM causal_edge 
                WHERE target = $t AND target_var = $v AND physics_score >= $min
                ORDER BY physics_score DESC, lag_days ASC
            """, {"t": target_dataset, "v": target_variable, "min": min_score})
        else:
            result = await self.db.query("""
                SELECT * FROM causal_edge 
                WHERE target = $t AND physics_score >= $min
                ORDER BY physics_score DESC, lag_days ASC
            """, {"t": target_dataset, "min": min_score})
        
        return result[0]["result"] if result else []
    
    async def get_effects(
        self,
        source_dataset: str,
        source_variable: str = None,
    ) -> List[Dict]:
        """
        Find what a source variable affects (effects/impacts).
        
        Args:
            source_dataset: Source dataset ID
            source_variable: Optional specific variable
        
        Returns:
            List of causal edges from source
        """
        await self.connect()
        
        if source_variable:
            result = await self.db.query("""
                SELECT * FROM causal_edge 
                WHERE source = $s AND source_var = $v
                ORDER BY lag_days ASC
            """, {"s": source_dataset, "v": source_variable})
        else:
            result = await self.db.query("""
                SELECT * FROM causal_edge 
                WHERE source = $s
                ORDER BY lag_days ASC
            """, {"s": source_dataset})
        
        return result[0]["result"] if result else []
    
    async def get_causal_chain(
        self,
        start_dataset: str,
        start_variable: str,
        max_depth: int = 3,
    ) -> List[List[Dict]]:
        """
        Trace causal chain forward from a starting point.
        
        Returns list of paths, each path is a list of edges.
        """
        await self.connect()
        
        # BFS to find all paths
        paths = []
        queue = [[(start_dataset, start_variable, None)]]  # (dataset, var, edge)
        
        while queue:
            path = queue.pop(0)
            if len(path) > max_depth:
                continue
            
            current_ds, current_var, _ = path[-1]
            effects = await self.get_effects(current_ds, current_var)
            
            if not effects:
                if len(path) > 1:
                    paths.append([p[2] for p in path[1:]])  # Extract edges
            else:
                for effect in effects:
                    new_path = path + [(effect["target"], effect["target_var"], effect)]
                    queue.append(new_path)
        
        return paths
    
    async def get_all_edges(self) -> List[Dict]:
        """Get all causal edges in the graph."""
        await self.connect()
        result = await self.db.query("SELECT * FROM causal_edge ORDER BY physics_score DESC")
        return result[0]["result"] if result else []
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the causal graph."""
        await self.connect()
        
        edges = await self.get_all_edges()
        
        sources = set()
        targets = set()
        mechanisms = {}
        
        for edge in edges:
            sources.add(edge.get("source"))
            targets.add(edge.get("target"))
            mech = edge.get("physics_mechanism", "unknown")
            mechanisms[mech] = mechanisms.get(mech, 0) + 1
        
        return {
            "total_edges": len(edges),
            "unique_sources": len(sources),
            "unique_targets": len(targets),
            "mechanisms": mechanisms,
            "avg_physics_score": sum(e.get("physics_score", 0) for e in edges) / len(edges) if edges else 0,
        }


# Pre-defined physics chains (from domain knowledge)
KNOWN_CAUSAL_CHAINS = [
    CausalEdge(
        source_dataset="noaa_climate_indices",
        source_variable="NAO",
        target_dataset="era5_reanalysis",
        target_variable="precipitation",
        lag_days=10,
        correlation=0.6,
        physics_mechanism="teleconnection",
        physics_score=0.8,
        discovered_by="literature",
    ),
    CausalEdge(
        source_dataset="era5_reanalysis",
        source_variable="precipitation",
        target_dataset="era5_reanalysis",
        target_variable="runoff",
        lag_days=2,
        correlation=0.75,
        physics_mechanism="hydrology",
        physics_score=0.9,
        discovered_by="physics",
    ),
    CausalEdge(
        source_dataset="cygnss_wind",
        source_variable="wind_speed",
        target_dataset="cmems_sst",
        target_variable="analysed_sst",
        lag_days=0,
        correlation=0.5,
        physics_mechanism="air_sea_interaction",
        physics_score=0.7,
        discovered_by="physics",
    ),
    CausalEdge(
        source_dataset="slcci_altimetry",
        source_variable="corssh",
        target_dataset="cmems_sealevel",
        target_variable="sla",
        lag_days=63,
        correlation=0.866,
        physics_mechanism="arctic_signal_propagation",
        physics_score=1.0,
        discovered_by="cross_region",
    ),
]


async def seed_known_chains(db: CausalGraphDB):
    """Seed database with known physics-validated causal chains."""
    for edge in KNOWN_CAUSAL_CHAINS:
        await db.add_edge(edge, metadata={"seeded": True})
    print(f"✅ Seeded {len(KNOWN_CAUSAL_CHAINS)} known causal chains")


# Singleton instance
_db = None


def get_causal_db() -> CausalGraphDB:
    """Get singleton causal graph database."""
    global _db
    if _db is None:
        _db = CausalGraphDB()
    return _db


# CLI test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=== Causal Graph DB Test ===\n")
        
        db = CausalGraphDB()
        
        try:
            await db.connect()
            print("✅ Connected to SurrealDB")
            
            # Seed known chains
            await seed_known_chains(db)
            
            # Query
            print("\nPrecursors of ERA5 precipitation:")
            precursors = await db.get_precursors("era5_reanalysis", "precipitation")
            for p in precursors:
                print(f"  - {p['source']}.{p['source_var']} (lag={p['lag_days']}d, score={p['physics_score']})")
            
            # Stats
            stats = await db.get_graph_stats()
            print(f"\nGraph stats: {stats}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("   Make sure SurrealDB is running: docker start surrealdb")
    
    asyncio.run(test())

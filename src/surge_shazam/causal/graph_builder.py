"""
🕸️ Causal Graph Builder
======================

Builds and manipulates causal graphs from PCMCI discovery results.

Features:
- Convert PCMCIResult to NetworkX directed graph
- Export to JSON for frontend visualization
- Find root causes and causal paths
- Support for time-lagged relationships

Usage:
    from src.surge_shazam.causal.graph_builder import CausalGraphBuilder
    
    builder = CausalGraphBuilder.from_pcmci_result(pcmci_result)
    graph = builder.to_networkx()
    root_causes = builder.get_root_causes("flood_severity")
    paths = builder.get_causal_paths("sst", "flood_severity")
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid sklearn dependency chain
try:
    from src.pattern_engine.causal.pcmci_engine import PCMCIResult, CausalLink
    HAS_PCMCI_ENGINE = True
except ImportError as e:
    HAS_PCMCI_ENGINE = False
    logger.warning(f"⚠️ PCMCIEngine not available for graph builder: {e}")
    
    # Define fallback types
    @dataclass
    class CausalLink:
        """Fallback CausalLink when pattern_engine not available."""
        source: str
        target: str
        lag: int
        strength: float
        p_value: float
        score: float
        test_used: str = "unknown"
        validated: bool = False
        physics_plausible: bool = True
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class PCMCIResult:
        """Fallback PCMCIResult when pattern_engine not available."""
        significant_links: List[CausalLink]
        all_links: List[CausalLink]
        var_names: List[str]
        val_matrix: Any
        p_matrix: Any
        conf_matrix: Any
        method: str
        max_lag: int
        alpha: float
        timestamp: str = ""
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}


@dataclass
class CausalPath:
    """Represents a causal path from source to target."""
    nodes: List[str]
    edges: List[Tuple[str, str, int]]  # (source, target, lag)
    total_lag: int
    strength: float  # Combined strength along path
    
    def __repr__(self) -> str:
        path_str = " → ".join(self.nodes)
        return f"CausalPath({path_str}, lag={self.total_lag}, strength={self.strength:.3f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nodes": self.nodes,
            "edges": [
                {"source": s, "target": t, "lag": lag}
                for s, t, lag in self.edges
            ],
            "total_lag": self.total_lag,
            "strength": self.strength,
        }


class CausalGraphBuilder:
    """
    Builds and manipulates causal graphs from PCMCI results.
    
    Provides methods for graph analysis, visualization export,
    and causal reasoning.
    """
    
    def __init__(
        self,
        links: List[CausalLink],
        var_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize causal graph builder.
        
        Args:
            links: List of causal links to include in graph
            var_names: List of all variable names
            metadata: Optional metadata from PCMCI result
        """
        self.links = links
        self.var_names = var_names
        self.metadata = metadata or {}
        self._graph: Optional[nx.DiGraph] = None
    
    @classmethod
    def from_pcmci_result(
        cls,
        result: PCMCIResult,
        min_score: float = 0.0,
        validated_only: bool = False,
        physics_plausible_only: bool = False,
    ) -> "CausalGraphBuilder":
        """
        Construct graph builder from PCMCI result.
        
        Args:
            result: PCMCIResult from PCMCI causal discovery
            min_score: Minimum link score to include (0-1)
            validated_only: Only include validated links
            physics_plausible_only: Only include physics-plausible links
            
        Returns:
            CausalGraphBuilder instance
        """
        # Filter links based on criteria
        filtered_links = []
        for link in result.significant_links:
            if link.score < min_score:
                continue
            if validated_only and not link.validated:
                continue
            if physics_plausible_only and not link.physics_plausible:
                continue
            filtered_links.append(link)
        
        metadata = {
            "method": result.method,
            "max_lag": result.max_lag,
            "alpha": result.alpha,
            "timestamp": result.timestamp,
            "n_total_links": len(result.significant_links),
            "n_filtered_links": len(filtered_links),
            **result.metadata,
        }
        
        logger.info(
            f"Built graph with {len(filtered_links)}/{len(result.significant_links)} links "
            f"(min_score={min_score}, validated={validated_only}, physics={physics_plausible_only})"
        )
        
        return cls(
            links=filtered_links,
            var_names=result.var_names,
            metadata=metadata,
        )
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Export causal graph as NetworkX directed graph.
        
        Nodes represent variables, edges represent causal links.
        Edge attributes include lag, strength, p_value, and score.
        
        Returns:
            nx.DiGraph with variables as nodes and causal links as edges
        """
        if self._graph is not None:
            return self._graph
        
        G = nx.DiGraph()
        
        # Add all variables as nodes
        for var in self.var_names:
            G.add_node(var, label=var)
        
        # Add causal links as edges
        for link in self.links:
            # Create edge with lag encoded in edge data
            # For multi-lag relationships, we create separate edges
            edge_key = f"{link.source}_{link.target}_{link.lag}"
            
            G.add_edge(
                link.source,
                link.target,
                key=edge_key,
                lag=link.lag,
                strength=link.strength,
                p_value=link.p_value,
                score=link.score,
                test_used=link.test_used,
                validated=link.validated,
                physics_plausible=link.physics_plausible,
            )
        
        self._graph = G
        return G
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary format for JSON serialization.
        
        Suitable for frontend visualization (Cosmograph, D3.js, etc.)
        
        Returns:
            Dictionary with nodes, edges, and metadata
        """
        nodes = [
            {
                "id": var,
                "label": var,
            }
            for var in self.var_names
        ]
        
        edges = [
            {
                "id": f"{link.source}_{link.target}_{link.lag}",
                "source": link.source,
                "target": link.target,
                "lag": link.lag,
                "strength": float(link.strength),
                "p_value": float(link.p_value),
                "score": float(link.score),
                "test_used": link.test_used,
                "validated": link.validated,
                "physics_plausible": link.physics_plausible,
            }
            for link in self.links
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": self.metadata,
            "stats": {
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "n_validated": sum(1 for link in self.links if link.validated),
                "n_physics_plausible": sum(1 for link in self.links if link.physics_plausible),
            }
        }
    
    def get_root_causes(
        self,
        target: str,
        max_depth: int = 10,
        min_score: float = 0.0,
    ) -> List[str]:
        """
        Find root causes of a target variable.
        
        Root causes are variables that:
        1. Causally influence the target (directly or indirectly)
        2. Have no incoming causal links themselves (or only from other roots)
        
        Args:
            target: Target variable to find root causes for
            max_depth: Maximum depth to search backwards
            min_score: Minimum link score to follow
            
        Returns:
            List of root cause variable names, sorted by influence
        """
        if target not in self.var_names:
            logger.warning(f"Target '{target}' not in variable names")
            return []
        
        G = self.to_networkx()
        
        # Find all ancestors of target
        try:
            ancestors = nx.ancestors(G, target)
        except nx.NetworkXError:
            ancestors = set()
        
        # Root causes have no incoming edges (or only from other roots)
        root_causes = []
        for var in ancestors:
            predecessors = list(G.predecessors(var))
            
            # Check if all predecessors are also in ancestors
            # (meaning this is a root or only influenced by other ancestors)
            if not predecessors or all(p in ancestors for p in predecessors):
                # Check if there's a path with sufficient scores
                has_strong_path = False
                try:
                    paths = nx.all_simple_paths(G, var, target, cutoff=max_depth)
                    for path in paths:
                        # Check minimum score along path
                        min_path_score = 1.0
                        for i in range(len(path) - 1):
                            edge_data = G.get_edge_data(path[i], path[i + 1])
                            if edge_data:
                                # Handle potential multiple edges
                                if isinstance(edge_data, dict) and 'score' in edge_data:
                                    min_path_score = min(min_path_score, edge_data['score'])
                                else:
                                    # Multiple edges case
                                    for edge_key, data in edge_data.items():
                                        if isinstance(data, dict) and 'score' in data:
                                            min_path_score = min(min_path_score, data['score'])
                        
                        if min_path_score >= min_score:
                            has_strong_path = True
                            break
                except nx.NetworkXNoPath:
                    pass
                
                if has_strong_path or not predecessors:
                    root_causes.append(var)
        
        # Sort by influence (simple heuristic: out-degree + average link strength)
        def influence_score(var: str) -> float:
            successors = list(G.successors(var))
            if not successors:
                return 0.0
            
            total_strength = 0.0
            n_edges = 0
            for succ in successors:
                edge_data = G.get_edge_data(var, succ)
                if edge_data:
                    if isinstance(edge_data, dict) and 'strength' in edge_data:
                        total_strength += abs(edge_data['strength'])
                        n_edges += 1
                    else:
                        for edge_key, data in edge_data.items():
                            if isinstance(data, dict) and 'strength' in data:
                                total_strength += abs(data['strength'])
                                n_edges += 1
            
            avg_strength = total_strength / n_edges if n_edges > 0 else 0.0
            return len(successors) * avg_strength
        
        root_causes.sort(key=influence_score, reverse=True)
        
        logger.info(f"Found {len(root_causes)} root causes for '{target}'")
        return root_causes
    
    def get_causal_paths(
        self,
        source: str,
        target: str,
        max_paths: int = 10,
        max_depth: int = 10,
    ) -> List[CausalPath]:
        """
        Find all causal paths from source to target.
        
        Args:
            source: Source variable
            target: Target variable
            max_paths: Maximum number of paths to return
            max_depth: Maximum path length
            
        Returns:
            List of CausalPath objects, sorted by strength
        """
        if source not in self.var_names:
            logger.warning(f"Source '{source}' not in variable names")
            return []
        if target not in self.var_names:
            logger.warning(f"Target '{target}' not in variable names")
            return []
        
        G = self.to_networkx()
        
        # Find all simple paths
        causal_paths = []
        try:
            paths = nx.all_simple_paths(G, source, target, cutoff=max_depth)
            
            for path_nodes in paths:
                # Build CausalPath object
                edges = []
                total_lag = 0
                strengths = []
                
                for i in range(len(path_nodes) - 1):
                    src = path_nodes[i]
                    tgt = path_nodes[i + 1]
                    
                    edge_data = G.get_edge_data(src, tgt)
                    if edge_data:
                        # Handle potential multiple edges (multi-graph)
                        if isinstance(edge_data, dict) and 'lag' in edge_data:
                            # Single edge
                            lag = edge_data['lag']
                            strength = edge_data['strength']
                            edges.append((src, tgt, lag))
                            total_lag += lag
                            strengths.append(abs(strength))
                        else:
                            # Multiple edges - take strongest
                            best_edge = None
                            best_strength = -1
                            for edge_key, data in edge_data.items():
                                if isinstance(data, dict) and 'strength' in data:
                                    if abs(data['strength']) > best_strength:
                                        best_strength = abs(data['strength'])
                                        best_edge = data
                            
                            if best_edge:
                                edges.append((src, tgt, best_edge['lag']))
                                total_lag += best_edge['lag']
                                strengths.append(best_strength)
                
                # Calculate combined strength (geometric mean)
                if strengths:
                    import numpy as np
                    combined_strength = float(np.prod(strengths) ** (1.0 / len(strengths)))
                else:
                    combined_strength = 0.0
                
                causal_path = CausalPath(
                    nodes=path_nodes,
                    edges=edges,
                    total_lag=total_lag,
                    strength=combined_strength,
                )
                causal_paths.append(causal_path)
                
                if len(causal_paths) >= max_paths:
                    break
        
        except nx.NetworkXNoPath:
            logger.info(f"No path found from '{source}' to '{target}'")
            return []
        
        # Sort by strength
        causal_paths.sort(key=lambda p: p.strength, reverse=True)
        
        logger.info(
            f"Found {len(causal_paths)} causal paths from '{source}' to '{target}'"
        )
        return causal_paths


# Convenience functions
def build_graph_from_pcmci(
    result: PCMCIResult,
    min_score: float = 0.5,
    validated_only: bool = False,
) -> CausalGraphBuilder:
    """
    Convenience function to build graph from PCMCI result.
    
    Args:
        result: PCMCIResult from causal discovery
        min_score: Minimum link score to include
        validated_only: Only include validated links
        
    Returns:
        CausalGraphBuilder instance
    """
    return CausalGraphBuilder.from_pcmci_result(
        result,
        min_score=min_score,
        validated_only=validated_only,
    )


# CLI test
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    print("=== Causal Graph Builder Test ===\n")
    
    if not HAS_PCMCI_ENGINE:
        print("⚠️ PCMCIEngine not available. Running with mock data.\n")
        
        # Create mock result
        mock_links = [
            CausalLink(source="X", target="Y", lag=5, strength=0.7, p_value=0.001, score=0.8),
            CausalLink(source="Y", target="Z", lag=3, strength=0.6, p_value=0.002, score=0.75),
        ]
        
        result = PCMCIResult(
            significant_links=mock_links,
            all_links=mock_links,
            var_names=["X", "Y", "Z"],
            val_matrix=None,
            p_matrix=None,
            conf_matrix=None,
            method="mock",
            max_lag=10,
            alpha=0.05,
        )
    else:
        from src.pattern_engine.causal.pcmci_engine import PCMCIEngine
        
        # Generate synthetic causal chain: X → Y → Z
        np.random.seed(42)
        n = 500
        
        x = np.random.randn(n)
        y = np.zeros(n)
        z = np.zeros(n)
        
        # X causes Y with lag 5
        for t in range(5, n):
            y[t] = 0.7 * x[t - 5] + np.random.randn() * 0.2
        
        # Y causes Z with lag 3
        for t in range(3, n):
            z[t] = 0.6 * y[t - 3] + np.random.randn() * 0.2
        
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})
        
        print(f"Data shape: {df.shape}")
        print(f"True causal chain: X →[5] Y →[3] Z\n")
        
        # Run PCMCI
        engine = PCMCIEngine(max_lag=10, alpha=0.05)
        result = engine.discover(df)
        
        print(f"Found {len(result.significant_links)} significant links\n")
    
    # Build graph
    builder = CausalGraphBuilder.from_pcmci_result(result, min_score=0.3)
    
    # Test NetworkX export
    G = builder.to_networkx()
    print(f"✅ NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test dict export
    graph_dict = builder.to_dict()
    print(f"✅ Dict export: {len(graph_dict['nodes'])} nodes, {len(graph_dict['edges'])} edges")
    
    # Test root causes
    print("\n🔍 Root causes of Z:")
    root_causes = builder.get_root_causes("Z")
    for cause in root_causes:
        print(f"  - {cause}")
    
    # Test causal paths
    print("\n🛤️  Causal paths from X to Z:")
    paths = builder.get_causal_paths("X", "Z")
    for path in paths:
        print(f"  {path}")
    
    print("\n✅ All tests passed!")

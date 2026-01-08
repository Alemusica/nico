"""
Tests for CausalGraphBuilder.

Tests graph construction, export, and analysis methods.
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from typing import List

from src.pattern_engine.causal.pcmci_engine import (
    PCMCIEngine,
    PCMCIResult,
    CausalLink,
)
from src.surge_shazam.causal.graph_builder import (
    CausalGraphBuilder,
    CausalPath,
    build_graph_from_pcmci,
)


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Generate synthetic causal chain: X → Y → Z."""
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
    
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


@pytest.fixture
def pcmci_result(synthetic_data: pd.DataFrame) -> PCMCIResult:
    """Run PCMCI on synthetic data."""
    engine = PCMCIEngine(max_lag=10, alpha=0.05, verbose=False)
    result = engine.discover(synthetic_data)
    return result


@pytest.fixture
def simple_links() -> List[CausalLink]:
    """Create simple causal links for testing."""
    return [
        CausalLink(
            source="A",
            target="B",
            lag=1,
            strength=0.8,
            p_value=0.001,
            score=0.9,
            test_used="parcorr",
        ),
        CausalLink(
            source="B",
            target="C",
            lag=2,
            strength=0.7,
            p_value=0.002,
            score=0.85,
            test_used="parcorr",
        ),
        CausalLink(
            source="A",
            target="C",
            lag=3,
            strength=0.5,
            p_value=0.01,
            score=0.7,
            test_used="parcorr",
        ),
    ]


class TestCausalGraphBuilder:
    """Tests for CausalGraphBuilder class."""
    
    def test_initialization(self, simple_links: List[CausalLink]):
        """Test basic initialization."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(
            links=simple_links,
            var_names=var_names,
        )
        
        assert builder.links == simple_links
        assert builder.var_names == var_names
        assert builder.metadata == {}
        assert builder._graph is None
    
    def test_from_pcmci_result(self, pcmci_result: PCMCIResult):
        """Test construction from PCMCI result."""
        builder = CausalGraphBuilder.from_pcmci_result(pcmci_result)
        
        assert len(builder.links) > 0
        assert len(builder.var_names) == 3
        assert "X" in builder.var_names
        assert "Y" in builder.var_names
        assert "Z" in builder.var_names
        assert "method" in builder.metadata
        assert "max_lag" in builder.metadata
    
    def test_from_pcmci_result_with_filters(self, pcmci_result: PCMCIResult):
        """Test filtering links when constructing from PCMCI result."""
        # No filter
        builder_all = CausalGraphBuilder.from_pcmci_result(
            pcmci_result,
            min_score=0.0,
        )
        n_all = len(builder_all.links)
        
        # High score filter
        builder_filtered = CausalGraphBuilder.from_pcmci_result(
            pcmci_result,
            min_score=0.8,
        )
        n_filtered = len(builder_filtered.links)
        
        assert n_filtered <= n_all
    
    def test_to_networkx(self, simple_links: List[CausalLink]):
        """Test export to NetworkX graph."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        G = builder.to_networkx()
        
        # Check graph properties
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3
        
        # Check nodes
        assert "A" in G.nodes
        assert "B" in G.nodes
        assert "C" in G.nodes
        
        # Check edges and attributes
        assert G.has_edge("A", "B")
        edge_data = G.get_edge_data("A", "B")
        assert edge_data is not None
        
        # Handle multi-graph case
        if isinstance(edge_data, dict) and 'lag' in edge_data:
            assert edge_data['lag'] == 1
            assert edge_data['strength'] == 0.8
            assert edge_data['score'] == 0.9
    
    def test_to_networkx_caching(self, simple_links: List[CausalLink]):
        """Test that to_networkx caches the graph."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        G1 = builder.to_networkx()
        G2 = builder.to_networkx()
        
        # Should return same object
        assert G1 is G2
    
    def test_to_dict(self, simple_links: List[CausalLink]):
        """Test export to dictionary."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        graph_dict = builder.to_dict()
        
        # Check structure
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert "metadata" in graph_dict
        assert "stats" in graph_dict
        
        # Check nodes
        assert len(graph_dict["nodes"]) == 3
        node_ids = [n["id"] for n in graph_dict["nodes"]]
        assert "A" in node_ids
        assert "B" in node_ids
        assert "C" in node_ids
        
        # Check edges
        assert len(graph_dict["edges"]) == 3
        edge = graph_dict["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "lag" in edge
        assert "strength" in edge
        assert "p_value" in edge
        assert "score" in edge
        
        # Check stats
        assert graph_dict["stats"]["n_nodes"] == 3
        assert graph_dict["stats"]["n_edges"] == 3
    
    def test_get_root_causes_simple(self, simple_links: List[CausalLink]):
        """Test finding root causes in simple chain."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        # Root causes of C should be A (and possibly B)
        root_causes = builder.get_root_causes("C")
        
        assert len(root_causes) > 0
        assert "A" in root_causes
    
    def test_get_root_causes_pcmci(self, pcmci_result: PCMCIResult):
        """Test finding root causes on PCMCI result."""
        builder = CausalGraphBuilder.from_pcmci_result(
            pcmci_result,
            min_score=0.3,
        )
        
        # Root causes of Z
        root_causes = builder.get_root_causes("Z")
        
        # Should find X as root cause (if causal chain discovered)
        assert isinstance(root_causes, list)
        # Note: Actual content depends on PCMCI discovery
    
    def test_get_root_causes_nonexistent_target(self, simple_links: List[CausalLink]):
        """Test root causes with nonexistent target."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        root_causes = builder.get_root_causes("NONEXISTENT")
        assert root_causes == []
    
    def test_get_causal_paths_simple(self, simple_links: List[CausalLink]):
        """Test finding causal paths."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        # Find paths from A to C
        paths = builder.get_causal_paths("A", "C")
        
        assert len(paths) > 0
        assert all(isinstance(p, CausalPath) for p in paths)
        
        # Should find both direct and indirect paths
        # Direct: A → C
        # Indirect: A → B → C
        path_lengths = [len(p.nodes) for p in paths]
        assert 2 in path_lengths  # Direct path
        assert 3 in path_lengths  # Indirect path
    
    def test_get_causal_paths_pcmci(self, pcmci_result: PCMCIResult):
        """Test finding causal paths on PCMCI result."""
        builder = CausalGraphBuilder.from_pcmci_result(
            pcmci_result,
            min_score=0.3,
        )
        
        # Try to find paths (depends on discovery)
        # This is more of a smoke test
        for source in builder.var_names:
            for target in builder.var_names:
                if source != target:
                    paths = builder.get_causal_paths(source, target, max_paths=5)
                    assert isinstance(paths, list)
    
    def test_get_causal_paths_no_path(self, simple_links: List[CausalLink]):
        """Test when no path exists."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        # No path from C to A (reverse direction)
        paths = builder.get_causal_paths("C", "A")
        assert paths == []
    
    def test_get_causal_paths_nonexistent_nodes(self, simple_links: List[CausalLink]):
        """Test causal paths with nonexistent nodes."""
        var_names = ["A", "B", "C"]
        builder = CausalGraphBuilder(links=simple_links, var_names=var_names)
        
        paths = builder.get_causal_paths("A", "NONEXISTENT")
        assert paths == []
        
        paths = builder.get_causal_paths("NONEXISTENT", "C")
        assert paths == []
    
    def test_causal_path_to_dict(self):
        """Test CausalPath to_dict method."""
        path = CausalPath(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 1), ("B", "C", 2)],
            total_lag=3,
            strength=0.8,
        )
        
        path_dict = path.to_dict()
        
        assert path_dict["nodes"] == ["A", "B", "C"]
        assert len(path_dict["edges"]) == 2
        assert path_dict["total_lag"] == 3
        assert path_dict["strength"] == 0.8
    
    def test_build_graph_from_pcmci_convenience(self, pcmci_result: PCMCIResult):
        """Test convenience function."""
        builder = build_graph_from_pcmci(
            pcmci_result,
            min_score=0.5,
            validated_only=False,
        )
        
        assert isinstance(builder, CausalGraphBuilder)
        assert len(builder.var_names) == 3


class TestCausalGraphBuilderIntegration:
    """Integration tests with real PCMCI workflow."""
    
    def test_full_workflow(self, synthetic_data: pd.DataFrame):
        """Test complete workflow from data to graph analysis."""
        # 1. Run PCMCI
        engine = PCMCIEngine(max_lag=10, alpha=0.05, verbose=False)
        result = engine.discover(synthetic_data)
        
        assert len(result.significant_links) > 0
        
        # 2. Build graph
        builder = CausalGraphBuilder.from_pcmci_result(
            result,
            min_score=0.3,
        )
        
        assert len(builder.links) > 0
        
        # 3. Export to NetworkX
        G = builder.to_networkx()
        assert G.number_of_nodes() == 3
        
        # 4. Export to dict
        graph_dict = builder.to_dict()
        assert len(graph_dict["nodes"]) == 3
        
        # 5. Find root causes
        for var in ["X", "Y", "Z"]:
            root_causes = builder.get_root_causes(var)
            assert isinstance(root_causes, list)
        
        # 6. Find paths
        paths_xz = builder.get_causal_paths("X", "Z")
        # May or may not find paths depending on discovery
        assert isinstance(paths_xz, list)
    
    def test_empty_result(self):
        """Test handling of empty PCMCI result."""
        empty_result = PCMCIResult(
            significant_links=[],
            all_links=[],
            var_names=["A", "B", "C"],
            val_matrix=np.zeros((3, 3, 5)),
            p_matrix=np.ones((3, 3, 5)),
            conf_matrix=None,
            method="pcmci",
            max_lag=5,
            alpha=0.05,
        )
        
        builder = CausalGraphBuilder.from_pcmci_result(empty_result)
        
        assert len(builder.links) == 0
        assert builder.var_names == ["A", "B", "C"]
        
        G = builder.to_networkx()
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 0
        
        root_causes = builder.get_root_causes("C")
        assert root_causes == []


class TestCausalPath:
    """Tests for CausalPath dataclass."""
    
    def test_creation(self):
        """Test CausalPath creation."""
        path = CausalPath(
            nodes=["A", "B", "C"],
            edges=[("A", "B", 1), ("B", "C", 2)],
            total_lag=3,
            strength=0.75,
        )
        
        assert path.nodes == ["A", "B", "C"]
        assert len(path.edges) == 2
        assert path.total_lag == 3
        assert path.strength == 0.75
    
    def test_repr(self):
        """Test string representation."""
        path = CausalPath(
            nodes=["X", "Y", "Z"],
            edges=[("X", "Y", 5), ("Y", "Z", 3)],
            total_lag=8,
            strength=0.85,
        )
        
        repr_str = repr(path)
        assert "X" in repr_str
        assert "Y" in repr_str
        assert "Z" in repr_str
        assert "8" in repr_str  # lag
        assert "0.85" in repr_str  # strength


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Causal discovery for pattern detection.

Uses PCMCI and other methods to discover causal relationships
between features and outcomes, including time-lagged effects.

Example:
    "Temperature at time T causes failure at time T+24" (24-hour lag)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

import pandas as pd
import numpy as np

from ..core.config import CausalConfig
from ..core.pattern import (
    Pattern, PatternType, PatternDatabase,
    Condition, ConditionOperator, Outcome
)

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """A causal edge in the graph."""
    source: str  # Cause variable
    target: str  # Effect variable
    lag: int  # Time lag (0 = contemporaneous)
    strength: float  # Causal strength (correlation/coefficient)
    p_value: float  # Statistical significance
    confidence: float  # Confidence in this edge
    
    def __str__(self) -> str:
        lag_str = f"[lag={self.lag}]" if self.lag > 0 else ""
        return f"{self.source} → {self.target}{lag_str} (strength={self.strength:.3f})"


@dataclass
class CausalGraph:
    """
    Causal graph representation.
    
    Nodes are variables, edges represent causal relationships.
    """
    nodes: List[str]
    edges: List[CausalEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_causes(self, target: str) -> List[CausalEdge]:
        """Get all causes of a target variable."""
        return [e for e in self.edges if e.target == target]
    
    def get_effects(self, source: str) -> List[CausalEdge]:
        """Get all effects of a source variable."""
        return [e for e in self.edges if e.source == source]
    
    def get_parents(self, node: str, max_lag: Optional[int] = None) -> List[str]:
        """Get parent nodes (causes) of a node."""
        parents = []
        for edge in self.edges:
            if edge.target == node:
                if max_lag is None or edge.lag <= max_lag:
                    parents.append(edge.source)
        return list(set(parents))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": self.nodes,
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "lag": e.lag,
                    "strength": e.strength,
                    "p_value": e.p_value,
                    "confidence": e.confidence,
                }
                for e in self.edges
            ],
            "metadata": self.metadata,
        }
    
    def to_networkx(self) -> Any:
        """Convert to NetworkX graph."""
        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_nodes_from(self.nodes)
            for edge in self.edges:
                G.add_edge(
                    edge.source, edge.target,
                    lag=edge.lag,
                    strength=edge.strength,
                    p_value=edge.p_value,
                )
            return G
        except ImportError:
            raise ImportError("NetworkX required for graph conversion")


class CausalDiscovery:
    """
    Discover causal relationships in data.
    
    Supports:
    - PCMCI (Peter-Clark Momentary Conditional Independence)
    - Granger causality
    - Correlation-based discovery
    
    Example:
        discovery = CausalDiscovery(CausalConfig(
            max_lag=24,
            alpha_level=0.05
        ))
        
        graph = discovery.discover(df, target="failure")
        
        # Get what causes failures
        causes = graph.get_causes("failure")
        for edge in causes:
            print(f"{edge.source} causes failure with lag {edge.lag}")
    """
    
    def __init__(
        self,
        config: Optional[CausalConfig] = None,
        random_seed: int = 42,
    ):
        self.config = config or CausalConfig()
        self.random_seed = random_seed
        self._graph: Optional[CausalGraph] = None
    
    def discover(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        method: str = "auto",
    ) -> CausalGraph:
        """
        Discover causal relationships.
        
        Args:
            df: Time-series DataFrame (rows = time, columns = variables)
            target: Target variable to focus on (optional)
            method: Discovery method ("pcmci", "granger", "correlation", "auto")
            
        Returns:
            CausalGraph with discovered relationships
        """
        if method == "auto":
            method = self._select_method(df)
        
        if method == "pcmci":
            return self._discover_pcmci(df, target)
        elif method == "granger":
            return self._discover_granger(df, target)
        elif method == "correlation":
            return self._discover_correlation(df, target)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def discover_patterns(
        self,
        df: pd.DataFrame,
        target: str,
    ) -> List[Pattern]:
        """
        Discover causal patterns leading to target.
        
        Returns Pattern objects that can be stored in PatternDatabase.
        """
        graph = self.discover(df, target)
        patterns = []
        
        # Get causes of target
        causes = graph.get_causes(target)
        
        for edge in causes:
            if edge.p_value > self.config.alpha_level:
                continue
            
            if abs(edge.strength) < self.config.min_effect_size:
                continue
            
            # Determine condition direction
            if edge.strength > 0:
                # Positive effect: high source → high target
                operator = ConditionOperator.GREATER
                threshold = df[edge.source].median()
            else:
                # Negative effect: high source → low target
                operator = ConditionOperator.LESS
                threshold = df[edge.source].median()
            
            condition = Condition(
                feature=edge.source,
                operator=operator,
                value=round(threshold, 3),
                metadata={"lag": edge.lag},
            )
            
            pattern = Pattern(
                pattern_type=PatternType.CAUSATION,
                conditions=[condition],
                outcome=Outcome(
                    name=target,
                    value="increase" if edge.strength > 0 else "decrease",
                    probability=abs(edge.strength),
                    lag=edge.lag,
                ),
                confidence=1 - edge.p_value,
                causal_strength=edge.strength,
                causal_lag=edge.lag,
                description=f"{edge.source} causes {target} with lag {edge.lag}",
                metadata={
                    "p_value": edge.p_value,
                    "method": "causal_discovery",
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    def _select_method(self, df: pd.DataFrame) -> str:
        """Auto-select best causal discovery method."""
        n_samples, n_features = df.shape
        
        # PCMCI is best for time series with many samples
        if n_samples >= 100 and n_features <= 20:
            try:
                import tigramite
                return "pcmci"
            except ImportError:
                pass
        
        # Granger for moderate data
        if n_samples >= 50:
            return "granger"
        
        # Correlation for small data
        return "correlation"
    
    def _discover_pcmci(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> CausalGraph:
        """Discover causal relationships using PCMCI."""
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError:
            logger.warning("Tigramite not installed, falling back to correlation")
            return self._discover_correlation(df, target)
        
        # Prepare data for Tigramite
        var_names = list(df.columns)
        data = df.values
        
        dataframe = pp.DataFrame(
            data,
            var_names=var_names,
        )
        
        # Select independence test
        if self.config.ci_test == "parcorr":
            cond_ind_test = ParCorr()
        else:
            cond_ind_test = ParCorr()  # Default
        
        # Run PCMCI
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
        )
        
        results = pcmci.run_pcmci(
            tau_max=self.config.max_lag,
            alpha_level=self.config.alpha_level,
        )
        
        # Extract edges
        edges = []
        val_matrix = results["val_matrix"]
        p_matrix = results["p_matrix"]
        
        n_vars = len(var_names)
        for i in range(n_vars):
            for j in range(n_vars):
                for lag in range(self.config.max_lag + 1):
                    # Skip contemporaneous self-links
                    if i == j and lag == 0:
                        continue
                    
                    # Skip if not contemporaneous but configured
                    if lag == 0 and not self.config.include_contemporaneous:
                        continue
                    
                    strength = val_matrix[i, j, lag]
                    p_value = p_matrix[i, j, lag]
                    
                    if p_value < self.config.alpha_level:
                        edges.append(CausalEdge(
                            source=var_names[i],
                            target=var_names[j],
                            lag=lag,
                            strength=float(strength),
                            p_value=float(p_value),
                            confidence=1 - p_value,
                        ))
        
        self._graph = CausalGraph(
            nodes=var_names,
            edges=edges,
            metadata={"method": "pcmci", "max_lag": self.config.max_lag},
        )
        
        return self._graph
    
    def _discover_granger(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> CausalGraph:
        """Discover causal relationships using Granger causality."""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        var_names = list(df.columns)
        edges = []
        
        # Test all pairs
        variables = [target] if target else var_names
        
        for effect_var in variables:
            for cause_var in var_names:
                if cause_var == effect_var:
                    continue
                
                try:
                    # Prepare data
                    test_data = df[[effect_var, cause_var]].dropna()
                    
                    if len(test_data) < self.config.max_lag + 10:
                        continue
                    
                    # Run Granger test
                    results = grangercausalitytests(
                        test_data,
                        maxlag=min(self.config.max_lag, len(test_data) // 4),
                        verbose=False,
                    )
                    
                    # Find best lag
                    best_lag = None
                    best_p = 1.0
                    
                    for lag, result in results.items():
                        p_value = result[0]["ssr_ftest"][1]
                        if p_value < best_p:
                            best_p = p_value
                            best_lag = lag
                    
                    if best_lag and best_p < self.config.alpha_level:
                        # Estimate strength using correlation at lag
                        lagged_cause = df[cause_var].shift(best_lag)
                        strength = df[effect_var].corr(lagged_cause)
                        
                        edges.append(CausalEdge(
                            source=cause_var,
                            target=effect_var,
                            lag=best_lag,
                            strength=float(strength) if not np.isnan(strength) else 0,
                            p_value=float(best_p),
                            confidence=1 - best_p,
                        ))
                        
                except Exception as e:
                    logger.debug(f"Granger test failed for {cause_var} → {effect_var}: {e}")
        
        self._graph = CausalGraph(
            nodes=var_names,
            edges=edges,
            metadata={"method": "granger", "max_lag": self.config.max_lag},
        )
        
        return self._graph
    
    def _discover_correlation(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> CausalGraph:
        """Discover relationships using lagged correlations."""
        var_names = list(df.columns)
        edges = []
        
        variables = [target] if target else var_names
        
        for effect_var in variables:
            for cause_var in var_names:
                if cause_var == effect_var:
                    continue
                
                # Test different lags
                best_lag = 0
                best_corr = 0
                best_p = 1.0
                
                for lag in range(self.config.max_lag + 1):
                    if lag == 0 and not self.config.include_contemporaneous:
                        continue
                    
                    # Calculate lagged correlation
                    if lag > 0:
                        lagged = df[cause_var].shift(lag)
                    else:
                        lagged = df[cause_var]
                    
                    valid_mask = ~(lagged.isna() | df[effect_var].isna())
                    if valid_mask.sum() < 10:
                        continue
                    
                    corr = df[effect_var][valid_mask].corr(lagged[valid_mask])
                    
                    if np.isnan(corr):
                        continue
                    
                    # Simple significance test (Fisher's z)
                    n = valid_mask.sum()
                    z = 0.5 * np.log((1 + corr) / (1 - corr + 1e-10))
                    se = 1 / np.sqrt(n - 3)
                    p_value = 2 * (1 - self._normal_cdf(abs(z) / se))
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                        best_p = p_value
                
                if best_p < self.config.alpha_level and abs(best_corr) >= self.config.min_effect_size:
                    edges.append(CausalEdge(
                        source=cause_var,
                        target=effect_var,
                        lag=best_lag,
                        strength=float(best_corr),
                        p_value=float(best_p),
                        confidence=1 - best_p,
                    ))
        
        self._graph = CausalGraph(
            nodes=var_names,
            edges=edges,
            metadata={"method": "correlation", "max_lag": self.config.max_lag},
        )
        
        return self._graph
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        from math import erf, sqrt
        return 0.5 * (1 + erf(x / sqrt(2)))
    
    def plot_graph(
        self,
        graph: Optional[CausalGraph] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Any:
        """Plot the causal graph."""
        import matplotlib.pyplot as plt
        
        graph = graph or self._graph
        if graph is None:
            raise ValueError("No graph to plot")
        
        try:
            import networkx as nx
            
            G = graph.to_networkx()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Layout
            pos = nx.spring_layout(G, seed=self.random_seed)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color="lightblue")
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
            
            # Draw edges with strength as width
            edge_widths = [abs(G[u][v]["strength"]) * 3 for u, v in G.edges()]
            edge_colors = ["green" if G[u][v]["strength"] > 0 else "red" for u, v in G.edges()]
            
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                edge_color=edge_colors,
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.1",
            )
            
            # Add edge labels (lag)
            edge_labels = {(u, v): f"lag={G[u][v]['lag']}" for u, v in G.edges() if G[u][v]["lag"] > 0}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
            
            ax.set_title("Causal Graph")
            ax.axis("off")
            
            return fig
            
        except ImportError:
            logger.warning("NetworkX/Matplotlib required for plotting")
            return None

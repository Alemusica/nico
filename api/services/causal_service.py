"""
🔬 Causal Discovery Service
============================
Integrates PCMCI causal discovery with LLM explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio

# Tigramite for PCMCI
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.independence_tests.gpdc import GPDC
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    # Silent import - warning only on first use
    pp = None
    PCMCI = None
    ParCorr = None
    GPDC = None

# NetworkX for graph representation
import networkx as nx

# Scipy for correlation analysis
from scipy import stats
from scipy.signal import correlate

from .llm_service import get_llm_service, OllamaLLMService


@dataclass
class CausalLink:
    """A discovered causal relationship."""
    source: str
    target: str
    lag: int  # Time steps
    strength: float  # Effect strength (-1 to 1)
    p_value: float
    ci_test: str = "parcorr"  # Test used
    explanation: Optional[str] = None  # LLM explanation
    physics_valid: Optional[bool] = None
    physics_score: Optional[float] = None


@dataclass
class CausalGraph:
    """Causal graph from discovery."""
    links: List[CausalLink] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    max_lag: int = 5
    alpha: float = 0.05
    discovery_method: str = "pcmci"
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        G.add_nodes_from(self.variables)
        for link in self.links:
            G.add_edge(
                link.source, 
                link.target,
                lag=link.lag,
                strength=link.strength,
                p_value=link.p_value,
            )
        return G
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "variables": self.variables,
            "links": [
                {
                    "source": l.source,
                    "target": l.target,
                    "lag": l.lag,
                    "strength": l.strength,
                    "p_value": l.p_value,
                    "explanation": l.explanation,
                    "physics_valid": l.physics_valid,
                    "physics_score": l.physics_score,
                }
                for l in self.links
            ],
            "max_lag": self.max_lag,
            "alpha": self.alpha,
            "method": self.discovery_method,
        }


@dataclass  
class DiscoveryConfig:
    """Configuration for causal discovery."""
    max_lag: int = 7  # Max time lag to consider
    alpha_level: float = 0.05  # Significance threshold
    ci_test: str = "parcorr"  # "parcorr", "gpdc", "cmi_knn"
    min_effect_size: float = 0.1  # Minimum correlation to report
    include_contemporaneous: bool = True  # Include lag=0
    use_llm_explanations: bool = True


class CausalDiscoveryService:
    """
    Service for causal discovery combining statistical methods with LLM.
    
    Pipeline:
    1. Load and preprocess time series data
    2. Run PCMCI for causal discovery
    3. Filter by significance and effect size
    4. LLM explains each relationship
    5. Physics validation
    6. Generate summary and hypotheses
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.llm: Optional[OllamaLLMService] = None
        
    async def _get_llm(self) -> OllamaLLMService:
        """Lazy load LLM service."""
        if self.llm is None:
            self.llm = get_llm_service()
            await self.llm.check_availability()
        return self.llm
    
    def _preprocess_data(
        self, 
        df: pd.DataFrame,
        variables: Optional[List[str]] = None,
        time_column: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess DataFrame for causal discovery.
        
        Returns:
            data: np.ndarray of shape (T, N) - time x variables
            var_names: List of variable names
        """
        # Select numeric columns
        if variables:
            cols = [c for c in variables if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove time column if specified
        if time_column and time_column in cols:
            cols.remove(time_column)
        
        # Extract data
        data = df[cols].values.astype(np.float64)
        
        # Handle missing values - interpolate
        for i in range(data.shape[1]):
            mask = np.isnan(data[:, i])
            if mask.any():
                x = np.arange(len(data))
                data[mask, i] = np.interp(
                    x[mask], 
                    x[~mask], 
                    data[~mask, i]
                )
        
        # Standardize
        data = (data - np.nanmean(data, axis=0)) / (np.nanstd(data, axis=0) + 1e-8)
        
        return data, cols
    
    def _run_pcmci(
        self,
        data: np.ndarray,
        var_names: List[str],
    ) -> Dict[str, Any]:
        """
        Run PCMCI causal discovery.
        
        Returns dict with results including significant links.
        """
        if not TIGRAMITE_AVAILABLE:
            return self._run_correlation_fallback(data, var_names)
        
        # Create tigramite dataframe
        dataframe = pp.DataFrame(
            data,
            var_names=var_names,
        )
        
        # Select independence test
        if self.config.ci_test == "gpdc":
            ci_test = GPDC(significance='analytic')
        else:
            ci_test = ParCorr(significance='analytic')
        
        # Run PCMCI
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ci_test,
        )
        
        results = pcmci.run_pcmci(
            tau_max=self.config.max_lag,
            pc_alpha=self.config.alpha_level,
        )
        
        return {
            "p_matrix": results['p_matrix'],
            "val_matrix": results['val_matrix'],
            "var_names": var_names,
            "method": "pcmci",
        }
    
    def _run_correlation_fallback(
        self,
        data: np.ndarray,
        var_names: List[str],
    ) -> Dict[str, Any]:
        """
        Fallback: cross-correlation analysis when tigramite unavailable.
        """
        n_vars = len(var_names)
        max_lag = self.config.max_lag
        
        # Initialize matrices
        p_matrix = np.ones((n_vars, n_vars, max_lag + 1))
        val_matrix = np.zeros((n_vars, n_vars, max_lag + 1))
        
        for i in range(n_vars):
            for j in range(n_vars):
                for lag in range(max_lag + 1):
                    if lag == 0 and i == j:
                        continue
                    
                    # Compute lagged correlation
                    x = data[lag:, i]  # Effect at time t
                    y = data[:len(x), j]  # Cause at time t-lag
                    
                    if len(x) > 10:
                        corr, p_val = stats.pearsonr(x, y)
                        val_matrix[i, j, lag] = corr
                        p_matrix[i, j, lag] = p_val
        
        return {
            "p_matrix": p_matrix,
            "val_matrix": val_matrix,
            "var_names": var_names,
            "method": "correlation",
        }
    
    def _extract_links(
        self,
        results: Dict[str, Any],
    ) -> List[CausalLink]:
        """Extract significant causal links from PCMCI results."""
        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]
        var_names = results["var_names"]
        method = results["method"]
        
        links = []
        n_vars = len(var_names)
        
        for i in range(n_vars):  # Target
            for j in range(n_vars):  # Source
                for lag in range(p_matrix.shape[2]):
                    if lag == 0 and i == j:
                        continue  # Skip self at lag 0
                    
                    p_val = p_matrix[i, j, lag]
                    strength = val_matrix[i, j, lag]
                    
                    # Filter by significance and effect size
                    if (p_val < self.config.alpha_level and 
                        abs(strength) >= self.config.min_effect_size):
                        
                        # Skip lag 0 if not wanted
                        if lag == 0 and not self.config.include_contemporaneous:
                            continue
                        
                        links.append(CausalLink(
                            source=var_names[j],
                            target=var_names[i],
                            lag=lag,
                            strength=float(strength),
                            p_value=float(p_val),
                            ci_test=method,
                        ))
        
        # Sort by absolute strength
        links.sort(key=lambda x: abs(x.strength), reverse=True)
        
        return links
    
    async def discover(
        self,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        domain: str = "flood",
    ) -> CausalGraph:
        """
        Run full causal discovery pipeline.
        
        Args:
            df: Input DataFrame with time series data
            variables: Columns to analyze (default: all numeric)
            time_column: Name of time column (will be excluded from analysis)
            domain: Domain for physics validation
            
        Returns:
            CausalGraph with discovered links and explanations
        """
        # Preprocess
        data, var_names = self._preprocess_data(df, variables, time_column)
        
        if len(var_names) < 2:
            raise ValueError("Need at least 2 variables for causal discovery")
        
        if data.shape[0] < self.config.max_lag * 2:
            raise ValueError(f"Need at least {self.config.max_lag * 2} time points")
        
        # Run PCMCI
        results = self._run_pcmci(data, var_names)
        
        # Extract significant links
        links = self._extract_links(results)
        
        # Add LLM explanations if enabled
        if self.config.use_llm_explanations and links:
            llm = await self._get_llm()
            if llm._available:
                for link in links[:10]:  # Limit to top 10 for speed
                    explanation = await llm.explain_causal_relationship(
                        source=link.source,
                        target=link.target,
                        lag=link.lag,
                        strength=link.strength,
                        p_value=link.p_value,
                        domain=domain,
                    )
                    link.explanation = explanation
                    
                    # Physics validation
                    validation = await llm.validate_pattern_physics(
                        pattern_description=f"{link.source} → {link.target} (lag={link.lag})",
                        variables=[link.source, link.target],
                        domain=domain,
                        statistical_confidence=1 - link.p_value,
                    )
                    link.physics_valid = validation.get("is_valid")
                    link.physics_score = validation.get("physics_score")
        
        return CausalGraph(
            links=links,
            variables=var_names,
            max_lag=self.config.max_lag,
            alpha=self.config.alpha_level,
            discovery_method=results["method"],
        )
    
    async def find_cross_correlations(
        self,
        df: pd.DataFrame,
        source_var: str,
        target_var: str,
        max_lag: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Find optimal lag between two variables using cross-correlation.
        
        Returns list of (lag, correlation, p_value) sorted by absolute correlation.
        """
        x = df[source_var].values
        y = df[target_var].values
        
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        # Standardize
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        
        results = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x_lagged = x[:lag]
                y_curr = y[-lag:]
            elif lag > 0:
                x_lagged = x[lag:]
                y_curr = y[:-lag]
            else:
                x_lagged = x
                y_curr = y
            
            if len(x_lagged) > 10:
                corr, p_val = stats.pearsonr(x_lagged, y_curr)
                results.append({
                    "lag": lag,
                    "correlation": float(corr),
                    "p_value": float(p_val),
                    "n_samples": len(x_lagged),
                })
        
        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return results


# Singleton
_discovery_service: Optional[CausalDiscoveryService] = None

def get_discovery_service() -> CausalDiscoveryService:
    """Get or create discovery service singleton."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = CausalDiscoveryService()
    return _discovery_service


# Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Create synthetic data with known causal structure
        np.random.seed(42)
        n = 500
        
        # X causes Y with lag 2, Y causes Z with lag 1
        x = np.random.randn(n)
        y = np.zeros(n)
        z = np.zeros(n)
        
        for t in range(2, n):
            y[t] = 0.7 * x[t-2] + 0.3 * np.random.randn()
        
        for t in range(1, n):
            z[t] = 0.6 * y[t-1] + 0.4 * np.random.randn()
        
        df = pd.DataFrame({
            "precipitation": x,
            "river_level": y,
            "flood_risk": z,
        })
        
        print("Testing Causal Discovery...")
        service = CausalDiscoveryService(DiscoveryConfig(
            max_lag=5,
            use_llm_explanations=False,  # Skip LLM for quick test
        ))
        
        graph = await service.discover(df, domain="flood")
        
        print(f"\nDiscovered {len(graph.links)} causal links:")
        for link in graph.links:
            print(f"  {link.source} → {link.target} "
                  f"(lag={link.lag}, r={link.strength:.3f}, p={link.p_value:.4f})")
        
        # Should find: precipitation → river_level (lag ~2)
        #              river_level → flood_risk (lag ~1)
        
    asyncio.run(test())

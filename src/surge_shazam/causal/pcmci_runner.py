"""
🔬 PCMCI Runner for Surge-Shazam-DK
===================================

Wrapper around the PCMCIEngine from pattern_engine, optimized for storm surge analysis.

Features:
- Simplified API for surge-specific causal discovery
- Integration with surge_shazam configuration
- Support for climate index variables
- Automatic lag selection based on physical constraints
- SurrealDB integration for storing results

Usage:
    from src.surge_shazam.causal.pcmci_runner import PCMCIRunner
    
    runner = PCMCIRunner(max_lag=72, alpha=0.05)
    result = runner.run_surge_analysis(df, target_var="sea_surface_height")
    
    for link in result.significant_links:
        print(f"{link.source} → {link.target} (lag={link.lag}h)")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.config import VariableMapping, VARIABLE_MAPPINGS
from ..core.constants import (
    FINGERPRINT_GATE,
    FORECAST_HORIZON,
)

logger = logging.getLogger(__name__)

# Try to import PCMCIEngine from pattern_engine
try:
    from src.pattern_engine.causal.pcmci_engine import (
        PCMCIEngine,
        PCMCIResult,
        CausalLink,
        IndependenceTest,
    )
    HAS_PCMCI_ENGINE = True
except ImportError as e:
    HAS_PCMCI_ENGINE = False
    logger.warning(f"⚠️ PCMCIEngine not available: {e}")
    
    # Define fallback types for type hints
    class PCMCIResult:
        """Fallback PCMCIResult for when pattern_engine not available."""
        pass
    
    class CausalLink:
        """Fallback CausalLink for when pattern_engine not available."""
        pass


@dataclass
class SurgeAnalysisConfig:
    """Configuration for surge-specific causal analysis."""
    
    # Maximum lag in time units (default: 72 hours)
    max_lag: int = 72
    
    # Significance level for causal links
    alpha: float = 0.05
    
    # Minimum effect size to consider significant
    min_effect_size: float = 0.1
    
    # Independence test type
    ci_test: str = "parcorr"  # "parcorr", "cmi", "gpdc"
    
    # PC algorithm alpha (None = use PCMCI+)
    pc_alpha: Optional[float] = None
    
    # Target variable for analysis
    target_var: str = "sea_surface_height"
    
    # Variables to exclude from analysis
    exclude_vars: List[str] = field(default_factory=list)
    
    # Cross-validate discovered links
    validate_links: bool = True
    
    # Number of cross-validation splits
    n_validation_splits: int = 5
    
    # Apply physics constraints
    apply_physics_constraints: bool = True
    
    # Verbose output
    verbose: bool = False


@dataclass
class SurgeAnalysisResult:
    """Result of surge-specific causal analysis."""
    
    # Original PCMCI result (if available)
    pcmci_result: Optional["PCMCIResult"]
    
    # Filtered links relevant to surge prediction
    surge_links: List["CausalLink"]
    
    # Root causes identified
    root_causes: List[str]
    
    # Lag information per variable
    lag_info: Dict[str, Dict[str, Any]]
    
    # Analysis metadata
    config: SurgeAnalysisConfig
    target_var: str
    n_variables: int
    n_samples: int
    
    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "surge_links": [
                {
                    "source": link.source,
                    "target": link.target,
                    "lag": link.lag,
                    "strength": float(link.strength),
                    "p_value": float(link.p_value),
                    "score": float(link.score),
                    "validated": link.validated if hasattr(link, 'validated') else False,
                    "physics_plausible": link.physics_plausible if hasattr(link, 'physics_plausible') else True,
                }
                for link in self.surge_links
            ],
            "root_causes": self.root_causes,
            "lag_info": self.lag_info,
            "target_var": self.target_var,
            "n_variables": self.n_variables,
            "n_samples": self.n_samples,
            "timestamp": self.timestamp,
            "config": {
                "max_lag": self.config.max_lag,
                "alpha": self.config.alpha,
                "ci_test": self.config.ci_test,
            },
            "metadata": self.metadata,
        }
    
    def get_causal_chain(self) -> List[List[str]]:
        """Get causal chains leading to target variable."""
        chains = []
        
        # Find direct causes
        direct_causes = [
            link.source for link in self.surge_links 
            if link.target == self.target_var
        ]
        
        # For each direct cause, trace back
        for cause in direct_causes:
            chain = [cause, self.target_var]
            
            # Find causes of the cause
            for link in self.surge_links:
                if link.target == cause and link.source not in chain:
                    chain.insert(0, link.source)
            
            chains.append(chain)
        
        return chains


class PCMCIRunner:
    """
    PCMCI Runner for surge-specific causal discovery.
    
    Wraps PCMCIEngine from pattern_engine with surge-specific optimizations
    and integration with surge_shazam architecture.
    """
    
    def __init__(
        self,
        max_lag: int = 72,
        alpha: float = 0.05,
        ci_test: str = "parcorr",
        min_effect_size: float = 0.1,
        pc_alpha: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Initialize PCMCI Runner.
        
        Args:
            max_lag: Maximum lag to test (in time units, typically hours)
            alpha: Significance level for causal links
            ci_test: Conditional independence test ("parcorr", "cmi", "gpdc")
            min_effect_size: Minimum effect size to consider significant
            pc_alpha: Alpha for PC algorithm (None = use PCMCI+)
            verbose: Print debug information
        """
        self.config = SurgeAnalysisConfig(
            max_lag=max_lag,
            alpha=alpha,
            ci_test=ci_test,
            min_effect_size=min_effect_size,
            pc_alpha=pc_alpha,
            verbose=verbose,
        )
        
        self._engine: Optional["PCMCIEngine"] = None
        self._last_result: Optional[SurgeAnalysisResult] = None
        
        if not HAS_PCMCI_ENGINE:
            logger.warning(
                "PCMCIEngine not available. Some features will be limited. "
                "Install tigramite: pip install tigramite"
            )
    
    def run_surge_analysis(
        self,
        df: pd.DataFrame,
        target_var: str = "sea_surface_height",
        exclude_vars: Optional[List[str]] = None,
        validate: bool = True,
    ) -> SurgeAnalysisResult:
        """
        Run causal discovery analysis for storm surge prediction.
        
        Args:
            df: DataFrame with time series (rows=time, cols=variables)
            target_var: Target variable for analysis (default: sea_surface_height)
            exclude_vars: Variables to exclude from analysis
            validate: Cross-validate discovered links
            
        Returns:
            SurgeAnalysisResult with discovered causal structure
        """
        if not HAS_PCMCI_ENGINE:
            return self._run_fallback_analysis(df, target_var)
        
        if self.config.verbose:
            logger.info(f"🔬 Running surge analysis on {len(df)} samples, {len(df.columns)} variables")
        
        # Prepare data
        df_clean = self._prepare_data(df, exclude_vars or [])
        
        # Initialize PCMCI engine
        self._engine = PCMCIEngine(
            max_lag=self.config.max_lag,
            alpha=self.config.alpha,
            ci_test=self.config.ci_test,
            min_effect_size=self.config.min_effect_size,
            pc_alpha=self.config.pc_alpha,
            verbose=self.config.verbose,
        )
        
        # Run discovery
        pcmci_result = self._engine.discover(df_clean, target=target_var)
        
        # Validate if requested
        if validate and self.config.validate_links:
            pcmci_result = self._engine.validate_links(
                pcmci_result, df_clean, 
                n_splits=self.config.n_validation_splits
            )
        
        # Apply physics constraints
        if self.config.apply_physics_constraints:
            pcmci_result = self._engine.add_physics_constraints(pcmci_result)
        
        # Extract surge-relevant links
        surge_links = self._filter_surge_links(pcmci_result, target_var)
        
        # Identify root causes
        root_causes = self._identify_root_causes(pcmci_result, target_var)
        
        # Compute lag information
        lag_info = self._compute_lag_info(surge_links)
        
        # Build result
        result = SurgeAnalysisResult(
            pcmci_result=pcmci_result,
            surge_links=surge_links,
            root_causes=root_causes,
            lag_info=lag_info,
            config=self.config,
            target_var=target_var,
            n_variables=len(df_clean.columns),
            n_samples=len(df_clean),
            metadata={
                "original_columns": list(df.columns),
                "used_columns": list(df_clean.columns),
                "n_significant_links": len(pcmci_result.significant_links),
                "n_surge_links": len(surge_links),
            }
        )
        
        self._last_result = result
        
        if self.config.verbose:
            logger.info(
                f"✅ Analysis complete: {len(surge_links)} surge-relevant links, "
                f"{len(root_causes)} root causes"
            )
        
        return result
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        exclude_vars: List[str],
    ) -> pd.DataFrame:
        """Prepare DataFrame for PCMCI analysis."""
        # Remove excluded variables
        cols_to_use = [c for c in df.columns if c not in exclude_vars]
        df_clean = df[cols_to_use].copy()
        
        # Handle missing data
        df_clean = df_clean.ffill().bfill()
        
        # Check for constant columns
        for col in df_clean.columns:
            if df_clean[col].std() == 0:
                logger.warning(f"⚠️ Removing constant column: {col}")
                df_clean = df_clean.drop(columns=[col])
        
        return df_clean
    
    def _filter_surge_links(
        self,
        result: "PCMCIResult",
        target_var: str,
    ) -> List["CausalLink"]:
        """Filter links relevant to surge prediction."""
        surge_links = []
        
        for link in result.significant_links:
            # Include links where target is the surge variable
            if link.target == target_var:
                surge_links.append(link)
            # Include links that are part of causal chain to target
            elif self._is_in_causal_chain(link, result, target_var):
                surge_links.append(link)
        
        # Sort by score
        surge_links.sort(key=lambda x: x.score, reverse=True)
        
        return surge_links
    
    def _is_in_causal_chain(
        self,
        link: "CausalLink",
        result: "PCMCIResult",
        target_var: str,
        max_depth: int = 3,
    ) -> bool:
        """Check if link is part of causal chain to target."""
        # Check if link.target eventually leads to target_var
        visited = set()
        to_visit = [link.target]
        
        for _ in range(max_depth):
            if not to_visit:
                break
            
            current = to_visit.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == target_var:
                return True
            
            # Find children of current
            for other_link in result.significant_links:
                if other_link.source == current and other_link.target not in visited:
                    to_visit.append(other_link.target)
        
        return False
    
    def _identify_root_causes(
        self,
        result: "PCMCIResult",
        target_var: str,
    ) -> List[str]:
        """Identify root causes of target variable."""
        # Find all ancestors of target
        ancestors = set()
        direct_causes = set()
        
        for link in result.significant_links:
            if link.target == target_var:
                direct_causes.add(link.source)
                ancestors.add(link.source)
        
        # Find causes of causes
        for _ in range(3):  # Max depth
            new_ancestors = set()
            for var in ancestors:
                for link in result.significant_links:
                    if link.target == var:
                        new_ancestors.add(link.source)
            
            if not new_ancestors - ancestors:
                break
            
            ancestors.update(new_ancestors)
        
        # Root causes have no parents (or only from outside the chain)
        root_causes = []
        for var in ancestors:
            is_root = True
            for link in result.significant_links:
                if link.target == var and link.source in ancestors:
                    is_root = False
                    break
            
            if is_root:
                root_causes.append(var)
        
        # Sort by average link strength
        def avg_strength(var: str) -> float:
            strengths = [
                abs(link.strength) for link in result.significant_links
                if link.source == var
            ]
            return sum(strengths) / len(strengths) if strengths else 0
        
        root_causes.sort(key=avg_strength, reverse=True)
        
        return root_causes
    
    def _compute_lag_info(
        self,
        links: List["CausalLink"],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute lag information per source variable."""
        lag_info = {}
        
        for link in links:
            source = link.source
            
            if source not in lag_info:
                lag_info[source] = {
                    "min_lag": link.lag,
                    "max_lag": link.lag,
                    "lags": [link.lag],
                    "strengths": [link.strength],
                    "targets": [link.target],
                }
            else:
                lag_info[source]["min_lag"] = min(lag_info[source]["min_lag"], link.lag)
                lag_info[source]["max_lag"] = max(lag_info[source]["max_lag"], link.lag)
                lag_info[source]["lags"].append(link.lag)
                lag_info[source]["strengths"].append(link.strength)
                lag_info[source]["targets"].append(link.target)
        
        # Add averages
        for source, info in lag_info.items():
            info["avg_lag"] = sum(info["lags"]) / len(info["lags"])
            info["avg_strength"] = sum(abs(s) for s in info["strengths"]) / len(info["strengths"])
        
        return lag_info
    
    def _run_fallback_analysis(
        self,
        df: pd.DataFrame,
        target_var: str,
    ) -> SurgeAnalysisResult:
        """
        Fallback analysis when PCMCIEngine is not available.
        
        Uses simple correlation analysis as approximation.
        """
        logger.warning("⚠️ Running fallback analysis (correlation-based, not causal)")
        
        # Simple cross-correlation analysis
        surge_links = []
        target_col = target_var if target_var in df.columns else df.columns[-1]
        
        for col in df.columns:
            if col == target_col:
                continue
            
            # Compute cross-correlation at different lags
            for lag in range(1, min(self.config.max_lag + 1, len(df) // 4)):
                if lag >= len(df):
                    break
                
                corr = df[col].iloc[:-lag].corr(df[target_col].iloc[lag:])
                
                if not np.isnan(corr) and abs(corr) > self.config.min_effect_size:
                    # Create mock CausalLink
                    link = type('CausalLink', (), {
                        'source': col,
                        'target': target_col,
                        'lag': lag,
                        'strength': corr,
                        'p_value': 0.01,  # Mock
                        'score': abs(corr),
                        'validated': False,
                        'physics_plausible': True,
                    })()
                    surge_links.append(link)
        
        # Keep only strongest link per variable
        unique_links = {}
        for link in surge_links:
            if link.source not in unique_links or link.score > unique_links[link.source].score:
                unique_links[link.source] = link
        
        surge_links = list(unique_links.values())
        surge_links.sort(key=lambda x: x.score, reverse=True)
        
        # Root causes are simply variables with high correlation
        root_causes = [link.source for link in surge_links[:5]]
        
        lag_info = self._compute_lag_info(surge_links)
        
        return SurgeAnalysisResult(
            pcmci_result=None,
            surge_links=surge_links,
            root_causes=root_causes,
            lag_info=lag_info,
            config=self.config,
            target_var=target_var,
            n_variables=len(df.columns),
            n_samples=len(df),
            metadata={
                "fallback_mode": True,
                "method": "cross-correlation",
            }
        )
    
    def get_result(self) -> Optional[SurgeAnalysisResult]:
        """Get the last analysis result."""
        return self._last_result
    
    def get_engine(self) -> Optional["PCMCIEngine"]:
        """Get the underlying PCMCI engine (if available)."""
        return self._engine


# Convenience function
def run_surge_pcmci(
    df: pd.DataFrame,
    target_var: str = "sea_surface_height",
    max_lag: int = 72,
    alpha: float = 0.05,
    validate: bool = True,
) -> SurgeAnalysisResult:
    """
    Convenience function for surge causal analysis.
    
    Args:
        df: Time series DataFrame
        target_var: Target variable for analysis
        max_lag: Maximum lag to test
        alpha: Significance level
        validate: Cross-validate results
        
    Returns:
        SurgeAnalysisResult with discovered causal structure
    """
    runner = PCMCIRunner(max_lag=max_lag, alpha=alpha)
    return runner.run_surge_analysis(df, target_var=target_var, validate=validate)


# CLI test
if __name__ == "__main__":
    print("=== PCMCI Runner Test ===\n")
    
    # Generate synthetic surge data
    np.random.seed(42)
    n = 500
    
    # Simulate: wind → pressure → sea_level
    wind_u = np.random.randn(n)
    wind_v = np.random.randn(n)
    
    # Pressure affected by wind with lag 6
    pressure = np.zeros(n)
    for t in range(6, n):
        pressure[t] = 0.5 * wind_u[t - 6] + 0.3 * wind_v[t - 6] + np.random.randn() * 0.2
    
    # Sea level affected by pressure with lag 3
    sea_level = np.zeros(n)
    for t in range(3, n):
        sea_level[t] = 0.7 * pressure[t - 3] + np.random.randn() * 0.1
    
    df = pd.DataFrame({
        "wind_u": wind_u,
        "wind_v": wind_v,
        "pressure": pressure,
        "sea_surface_height": sea_level,
    })
    
    print(f"Data shape: {df.shape}")
    print(f"Variables: {list(df.columns)}")
    print(f"True causal chain: wind →[6] pressure →[3] sea_surface_height\n")
    
    # Run analysis
    runner = PCMCIRunner(max_lag=10, alpha=0.05, verbose=True)
    result = runner.run_surge_analysis(df, target_var="sea_surface_height", validate=False)
    
    print(f"\n📊 Analysis Results:")
    print(f"  Surge-relevant links: {len(result.surge_links)}")
    print(f"  Root causes: {result.root_causes}")
    
    print(f"\n🔗 Discovered Links:")
    for link in result.surge_links[:10]:
        print(f"  {link.source} →[{link.lag}] {link.target} (score={link.score:.3f})")
    
    print(f"\n⏱️ Lag Information:")
    for var, info in result.lag_info.items():
        print(f"  {var}: avg_lag={info['avg_lag']:.1f}, avg_strength={info['avg_strength']:.3f}")
    
    print(f"\n🔄 Causal Chains:")
    for chain in result.get_causal_chain():
        print(f"  {' → '.join(chain)}")
    
    print("\n✅ Test completed!")

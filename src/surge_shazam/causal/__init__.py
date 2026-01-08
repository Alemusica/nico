"""
Surge Shazam Causal Analysis Module
====================================

Tools for causal discovery and graph analysis.

Components:
- PCMCIRunner: PCMCI wrapper for surge-specific causal analysis
- CausalGraphBuilder: Build and analyze causal graphs
"""

from src.surge_shazam.causal.pcmci_runner import (
    PCMCIRunner,
    SurgeAnalysisConfig,
    SurgeAnalysisResult,
    run_surge_pcmci,
)

from src.surge_shazam.causal.graph_builder import (
    CausalGraphBuilder,
    CausalPath,
    build_graph_from_pcmci,
)

__all__ = [
    # PCMCI Runner
    "PCMCIRunner",
    "SurgeAnalysisConfig",
    "SurgeAnalysisResult",
    "run_surge_pcmci",
    # Graph Builder
    "CausalGraphBuilder",
    "CausalPath",
    "build_graph_from_pcmci",
]

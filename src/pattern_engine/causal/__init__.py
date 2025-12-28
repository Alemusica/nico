"""
Causal inference module for Pattern Detection Engine.

Discovers causal relationships between conditions and outcomes.
Includes PCMCI engine for rigorous time-lagged causal discovery.
"""

from .discovery import CausalDiscovery, CausalGraph, CausalEdge

# PCMCI Engine (requires tigramite)
try:
    from .pcmci_engine import (
        PCMCIEngine,
        PCMCIResult,
        CausalLink,
        IndependenceTest,
        HAS_TIGRAMITE,
        discover_causal_links,
    )
except ImportError:
    HAS_TIGRAMITE = False
    PCMCIEngine = None
    PCMCIResult = None
    CausalLink = None
    IndependenceTest = None
    discover_causal_links = None

# Ishikawa Diagram Generator
from .ishikawa import (
    IshikawaDiagram,
    Cause,
    CauseCategory,
    create_ishikawa_from_pcmci,
)

__all__ = [
    # Base discovery
    "CausalDiscovery",
    "CausalGraph",
    "CausalEdge",
    # PCMCI
    "PCMCIEngine",
    "PCMCIResult",
    "CausalLink",
    "IndependenceTest",
    "HAS_TIGRAMITE",
    "discover_causal_links",
    # Ishikawa
    "IshikawaDiagram",
    "Cause",
    "CauseCategory",
    "create_ishikawa_from_pcmci",
]

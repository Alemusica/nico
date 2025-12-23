"""
Causal inference module for Pattern Detection Engine.

Discovers causal relationships between conditions and outcomes.
"""

from .discovery import CausalDiscovery, CausalGraph

__all__ = [
    "CausalDiscovery",
    "CausalGraph",
]

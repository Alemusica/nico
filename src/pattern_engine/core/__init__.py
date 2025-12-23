"""
Core module for Pattern Detection Engine.

Contains fundamental data structures, configuration, and base classes.
"""

from .config import PatternEngineConfig
from .pattern import Pattern, PatternMatch, PatternDatabase

__all__ = [
    "PatternEngineConfig",
    "Pattern",
    "PatternMatch",
    "PatternDatabase",
]

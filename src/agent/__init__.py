"""
Investigation Agent Module
==========================

LLM-powered agent for natural language investigations.
"""

from .investigation_agent import (
    InvestigationAgent,
    InvestigationResult,
    EventContext,
    QueryParser,
)

__all__ = [
    "InvestigationAgent",
    "InvestigationResult",
    "EventContext",
    "QueryParser",
]

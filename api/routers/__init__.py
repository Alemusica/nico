"""
🚀 API Routers
===============
FastAPI routers for different API domains.
"""

from .analysis_router import router as analysis_router
from .chat_router import router as chat_router
from .data_router import router as data_router
from .health_router import router as health_router
from .investigation_router import router as investigation_router
from .knowledge_router import router as knowledge_router
from .pipeline_router import router as pipeline_router

__all__ = ["analysis_router", "chat_router", "data_router", "health_router", "investigation_router", "knowledge_router", "pipeline_router"]


"""
🚀 API Routers
===============
FastAPI routers for different API domains.
"""

from .analysis_router import router as analysis_router
from .data_router import router as data_router
from .investigation_router import router as investigation_router

__all__ = ["analysis_router", "data_router", "investigation_router"]


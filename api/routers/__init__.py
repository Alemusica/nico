"""
🚀 API Routers
===============
FastAPI routers for different API domains.
"""

from .analysis_router import router as analysis_router
from .investigation_router import router as investigation_router

__all__ = ["analysis_router", "investigation_router"]


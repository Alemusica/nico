"""
Services Layer
==============
Business logic services shared by API and Streamlit.

This layer provides:
- GateService: Gate selection and spatial operations
- DataService: Data loading and filtering
- AnalysisService: Analysis operations

Usage:
    from src.services import GateService, DataService
    
    gate_service = GateService()
    bbox = gate_service.get_bbox("fram_strait")
"""

from src.services.gate_service import GateService
from src.services.data_service import DataService
from src.services.analysis_service import AnalysisService

__all__ = [
    "GateService",
    "DataService",
    "AnalysisService",
]

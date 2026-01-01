"""
Services Layer
==============
Business logic services shared by API and Streamlit.

This layer provides:
- GateService: Gate selection and spatial operations
- DataService: Data loading and filtering
- AnalysisService: Analysis operations
- SLCCIService: ESA Sea Level CCI data loading and processing

Usage:
    from src.services import GateService, DataService, SLCCIService
    
    gate_service = GateService()
    bbox = gate_service.get_bbox("fram_strait")
    
    slcci_service = SLCCIService()
    pass_data = slcci_service.load_pass_data(gate_path, pass_number)
"""

from src.services.gate_service import GateService
from src.services.data_service import DataService
from src.services.analysis_service import AnalysisService
from src.services.slcci_service import SLCCIService, SLCCIConfig, PassData

__all__ = [
    "GateService",
    "DataService",
    "AnalysisService",
    "SLCCIService",
    "SLCCIConfig",
    "PassData",
]

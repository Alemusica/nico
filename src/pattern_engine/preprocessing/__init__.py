"""
Preprocessing module for Pattern Detection Engine.

Handles heterogeneous data ingestion, cleaning, and transformation.
Includes tsfresh integration for automatic feature extraction.
"""

from .ingestor import DataIngestor
from .tidier import DataTidier
from .schema import SchemaInferrer, DataSchema

# TSFresh extractor (with graceful fallback)
try:
    from .tsfresh_extractor import (
        TSFreshExtractor,
        FeatureExtractionConfig,
        quick_extract,
        manual_extract_basic_features,
        TSFRESH_AVAILABLE,
    )
except ImportError:
    TSFRESH_AVAILABLE = False
    TSFreshExtractor = None
    FeatureExtractionConfig = None
    quick_extract = None
    manual_extract_basic_features = None

__all__ = [
    "DataIngestor",
    "DataTidier", 
    "SchemaInferrer",
    "DataSchema",
    # TSFresh
    "TSFreshExtractor",
    "FeatureExtractionConfig",
    "quick_extract",
    "manual_extract_basic_features",
    "TSFRESH_AVAILABLE",
]

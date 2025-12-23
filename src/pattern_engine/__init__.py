"""
Pattern Detection Engine
========================

A domain-agnostic pattern detection framework for identifying correlations
between heterogeneous input conditions and outcomes.

Key Features:
- Heterogeneous data ingestion and preprocessing
- Supervised pattern detection (with labeled outcomes)
- Unsupervised pattern detection (anomaly/cluster discovery)
- Causal inference for condition→outcome relationships
- Flexible output formats for various downstream applications

Example Use Cases:
- Manufacturing: batch conditions → failure prediction
- Energy: weather/load patterns → demand forecasting
- Climate: environmental conditions → extreme events

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    RAW HETEROGENEOUS DATA                    │
    │  (CSV, JSON, Parquet, Sensors, Logs, Time-series, etc.)     │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    PREPROCESSING LAYER                       │
    │  - Schema inference & validation                            │
    │  - Missing value handling                                   │
    │  - Type normalization                                       │
    │  - Feature engineering                                      │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    PATTERN DETECTION                         │
    │  ┌──────────────────┐      ┌──────────────────┐            │
    │  │    SUPERVISED    │      │   UNSUPERVISED   │            │
    │  │  - Decision Tree │      │  - Clustering    │            │
    │  │  - Random Forest │      │  - Anomaly Det.  │            │
    │  │  - XGBoost       │      │  - Association   │            │
    │  │  - Neural Net    │      │  - PCA/UMAP      │            │
    │  └──────────────────┘      └──────────────────┘            │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    CAUSAL INFERENCE                          │
    │  - Condition → Outcome graph discovery (PCMCI)              │
    │  - Lag identification                                       │
    │  - Confounding variable detection                           │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    OUTPUT LAYER                              │
    │  - Pattern reports (JSON, HTML, PDF)                        │
    │  - Alert triggers                                           │
    │  - API endpoints                                            │
    │  - Dashboard integration                                    │
    └─────────────────────────────────────────────────────────────┘
"""

__version__ = "0.1.0"
__author__ = "Pattern Engine Team"

from .core.config import (
    PatternEngineConfig,
    DataSourceConfig,
    PreprocessingConfig,
    SupervisedConfig,
    UnsupervisedConfig,
    CausalConfig,
    OutputConfig,
    DetectionMethod,
    OutputFormat,
)
from .core.pattern import (
    Pattern,
    PatternMatch, 
    PatternDatabase,
    PatternType,
    Condition,
    ConditionOperator,
    Outcome,
)
from .preprocessing.ingestor import DataIngestor, IngestedData
from .preprocessing.tidier import DataTidier, TidyingReport
from .preprocessing.schema import SchemaInferrer, DataSchema, ColumnSchema
from .detection.base import BaseDetector, DetectionResult
from .detection.supervised import SupervisedDetector
from .detection.unsupervised import UnsupervisedDetector
from .causal.discovery import CausalDiscovery, CausalGraph, CausalEdge
from .output.reporter import PatternReporter

# Physics validation (new)
from .physics.rules import (
    PhysicsValidator,
    PhysicsRule,
    ValidationResult,
    Domain,
    create_flood_validator,
    create_manufacturing_validator,
    create_energy_validator,
)

# Gray zone detection (new)
from .output.gray_zone import (
    GrayZoneDetector,
    GrayZoneConfig,
    GrayZonePattern,
    ReviewPriority,
)

# Optional: TSFresh and mlxtend (graceful fallback)
try:
    from .preprocessing.tsfresh_extractor import (
        TSFreshExtractor,
        FeatureExtractionConfig,
        quick_extract,
        TSFRESH_AVAILABLE,
    )
except ImportError:
    TSFRESH_AVAILABLE = False
    TSFreshExtractor = None
    FeatureExtractionConfig = None
    quick_extract = None

try:
    from .detection.association_rules import (
        AssociationRuleDetector,
        AssociationRuleConfig,
        AssociationRule,
        quick_association_rules,
        MLXTEND_AVAILABLE,
    )
except ImportError:
    MLXTEND_AVAILABLE = False
    AssociationRuleDetector = None
    AssociationRuleConfig = None
    AssociationRule = None
    quick_association_rules = None

__all__ = [
    # Configuration
    "PatternEngineConfig",
    "DataSourceConfig",
    "PreprocessingConfig",
    "SupervisedConfig",
    "UnsupervisedConfig",
    "CausalConfig",
    "OutputConfig",
    "DetectionMethod",
    "OutputFormat",
    # Core pattern types
    "Pattern",
    "PatternMatch", 
    "PatternDatabase",
    "PatternType",
    "Condition",
    "ConditionOperator",
    "Outcome",
    # Preprocessing
    "DataIngestor",
    "IngestedData",
    "DataTidier",
    "TidyingReport",
    "SchemaInferrer",
    "DataSchema",
    "ColumnSchema",
    # Detection
    "BaseDetector",
    "DetectionResult",
    "SupervisedDetector",
    "UnsupervisedDetector",
    # Causal
    "CausalDiscovery",
    "CausalGraph",
    "CausalEdge",
    # Output
    "PatternReporter",
    # Physics validation
    "PhysicsValidator",
    "PhysicsRule",
    "ValidationResult",
    "Domain",
    "create_flood_validator",
    "create_manufacturing_validator",
    "create_energy_validator",
    # Gray zone
    "GrayZoneDetector",
    "GrayZoneConfig",
    "GrayZonePattern",
    "ReviewPriority",
    # TSFresh (optional)
    "TSFreshExtractor",
    "FeatureExtractionConfig",
    "quick_extract",
    "TSFRESH_AVAILABLE",
    # Association rules (optional)
    "AssociationRuleDetector",
    "AssociationRuleConfig",
    "AssociationRule",
    "quick_association_rules",
    "MLXTEND_AVAILABLE",
]

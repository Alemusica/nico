"""
Configuration for the Pattern Detection Engine.

Defines settings for data ingestion, preprocessing, detection methods,
and output formats.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pathlib import Path


class DetectionMethod(Enum):
    """Available pattern detection methods."""
    # Supervised methods
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    
    # Unsupervised methods
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ASSOCIATION_RULES = "association_rules"
    PCA = "pca"
    UMAP = "umap"


class OutputFormat(Enum):
    """Available output formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    HTML = "html"
    PDF = "pdf"
    DASHBOARD = "dashboard"


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    name: str
    source_type: str  # "csv", "json", "parquet", "database", "api", "sensor"
    path_or_uri: str
    schema: Optional[Dict[str, str]] = None  # column_name: dtype
    timestamp_column: Optional[str] = None
    id_column: Optional[str] = None  # e.g., batch_number, process_id
    target_column: Optional[str] = None  # outcome column for supervised learning
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Missing value handling
    missing_strategy: str = "infer"  # "drop", "fill_mean", "fill_median", "fill_mode", "interpolate", "infer"
    
    # Type handling
    auto_convert_types: bool = True
    datetime_formats: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ])
    
    # Feature engineering
    create_time_features: bool = True  # hour, day_of_week, month, etc.
    create_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    
    # Normalization
    normalize_numeric: bool = True
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    
    # Categorical encoding
    encode_categorical: bool = True
    encoding_method: str = "auto"  # "onehot", "label", "target", "auto"
    
    # Outlier handling
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_action: str = "flag"  # "drop", "cap", "flag"


@dataclass
class SupervisedConfig:
    """Configuration for supervised pattern detection."""
    methods: List[DetectionMethod] = field(default_factory=lambda: [
        DetectionMethod.RANDOM_FOREST,
        DetectionMethod.XGBOOST,
    ])
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Feature selection
    feature_selection: bool = True
    max_features: Optional[int] = None
    min_feature_importance: float = 0.01
    
    # Model hyperparameters (method -> params dict)
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Class imbalance handling
    handle_imbalance: bool = True
    imbalance_method: str = "smote"  # "smote", "undersample", "class_weight"


@dataclass
class UnsupervisedConfig:
    """Configuration for unsupervised pattern detection."""
    methods: List[DetectionMethod] = field(default_factory=lambda: [
        DetectionMethod.ISOLATION_FOREST,
        DetectionMethod.KMEANS,
    ])
    
    # Clustering parameters
    n_clusters: Optional[int] = None  # None = auto-determine
    min_cluster_size: int = 10
    
    # Anomaly detection
    contamination: float = 0.1  # expected proportion of anomalies
    
    # Dimensionality reduction
    reduce_dimensions: bool = True
    n_components: int = 2
    
    # Association rules
    min_support: float = 0.1
    min_confidence: float = 0.5
    min_lift: float = 1.0


@dataclass
class CausalConfig:
    """Configuration for causal discovery."""
    enabled: bool = True
    
    # PCMCI parameters
    max_lag: int = 10
    alpha_level: float = 0.05
    
    # Conditional independence test
    ci_test: str = "parcorr"  # "parcorr", "gpdc", "cmi_knn"
    
    # Graph construction
    include_contemporaneous: bool = True
    min_effect_size: float = 0.1


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.JSON])
    output_dir: Path = field(default_factory=lambda: Path("./pattern_output"))
    
    # Report settings
    include_visualizations: bool = True
    include_feature_importance: bool = True
    include_causal_graph: bool = True
    
    # Alert settings
    alert_on_anomaly: bool = False
    alert_threshold: float = 0.9
    alert_webhook: Optional[str] = None


@dataclass
class PatternEngineConfig:
    """
    Main configuration for the Pattern Detection Engine.
    
    Example usage:
        config = PatternEngineConfig(
            name="manufacturing_quality",
            data_sources=[
                DataSourceConfig(
                    name="batch_data",
                    source_type="csv",
                    path_or_uri="data/batches.csv",
                    id_column="batch_number",
                    target_column="failure_status"
                ),
                DataSourceConfig(
                    name="environment",
                    source_type="csv", 
                    path_or_uri="data/environment.csv",
                    timestamp_column="timestamp"
                )
            ],
            preprocessing=PreprocessingConfig(
                create_lag_features=True,
                lag_periods=[1, 6, 12, 24]
            )
        )
    """
    name: str
    description: str = ""
    
    # Data sources (heterogeneous)
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    
    # Processing configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    unsupervised: UnsupervisedConfig = field(default_factory=UnsupervisedConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # General settings
    random_seed: int = 42
    n_jobs: int = -1  # -1 = use all cores
    verbose: bool = True
    
    def add_data_source(self, source: DataSourceConfig) -> "PatternEngineConfig":
        """Add a data source to the configuration."""
        self.data_sources.append(source)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternEngineConfig":
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if "data_sources" in data:
            data["data_sources"] = [
                DataSourceConfig(**ds) if isinstance(ds, dict) else ds
                for ds in data["data_sources"]
            ]
        if "preprocessing" in data and isinstance(data["preprocessing"], dict):
            data["preprocessing"] = PreprocessingConfig(**data["preprocessing"])
        if "supervised" in data and isinstance(data["supervised"], dict):
            data["supervised"] = SupervisedConfig(**data["supervised"])
        if "unsupervised" in data and isinstance(data["unsupervised"], dict):
            data["unsupervised"] = UnsupervisedConfig(**data["unsupervised"])
        if "causal" in data and isinstance(data["causal"], dict):
            data["causal"] = CausalConfig(**data["causal"])
        if "output" in data and isinstance(data["output"], dict):
            data["output"] = OutputConfig(**data["output"])
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PatternEngineConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

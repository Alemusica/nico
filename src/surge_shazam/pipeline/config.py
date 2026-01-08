"""
Pipeline configuration for Surge-Shazam-DK.

Centralizes all pipeline parameters with type safety and YAML loading support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core.constants import (
    # Thresholds
    FINGERPRINT_GATE,
    ENSEMBLE_GATE,
    PHYSICS_RESIDUAL_THRESHOLD,
    ALERT_THRESHOLD,
    GRAY_ZONE_HISTORICAL_MIN,
    
    # Physics parameters
    LAMBDA_PHYSICS_INIT,
    LAMBDA_DECAY,
    LAMBDA_PHYSICS_MIN,
    
    # Data resolution
    ERA5_RESOLUTION,
    FINGERPRINT_TIME_RESOLUTION,
    FORECAST_HORIZON,
    
    # Spatial bounds
    DENMARK_BOUNDS,
    NORTH_SEA_BOUNDS,
    ATLANTIC_BOUNDS,
)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing stage."""
    
    # Temporal parameters
    time_resolution_hours: float = FINGERPRINT_TIME_RESOLUTION
    forecast_horizon_hours: int = FORECAST_HORIZON
    
    # Spatial parameters
    spatial_resolution_degrees: float = ERA5_RESOLUTION
    
    # Spatial bounds (default: Denmark focus)
    lat_min: float = DENMARK_BOUNDS["lat_min"]
    lat_max: float = DENMARK_BOUNDS["lat_max"]
    lon_min: float = DENMARK_BOUNDS["lon_min"]
    lon_max: float = DENMARK_BOUNDS["lon_max"]
    
    # Interpolation
    interpolation_method: str = "linear"
    fill_missing: bool = True
    
    # Normalization
    normalize_data: bool = True
    normalization_method: str = "zscore"  # "zscore" or "minmax"
    
    # Quality control
    remove_outliers: bool = True
    outlier_std_threshold: float = 5.0


@dataclass
class CausalDiscoveryConfig:
    """Configuration for causal discovery stage (PCMCI)."""
    
    # PCMCI parameters
    tau_max: int = 24  # Maximum lag in hours
    pc_alpha: float = 0.01  # Significance level for PC algorithm
    
    # Conditional independence test
    cond_ind_test: str = "parcorr"  # "parcorr" or "gpdc"
    
    # Variable selection
    selected_vars: list[str] = field(default_factory=lambda: [
        "wind_u",
        "wind_v", 
        "pressure",
        "sea_surface_height",
        "sst"
    ])
    
    # Parallelization
    n_jobs: int = -1  # -1 = use all CPUs
    
    # Teleconnection detection
    detect_teleconnections: bool = True
    teleconnection_min_distance_km: float = 100.0


@dataclass
class GraphBuildingConfig:
    """Configuration for causal graph building stage."""
    
    # Node creation
    min_node_strength: float = 0.1
    max_nodes: int = 50
    
    # Edge pruning
    min_edge_weight: float = 0.05
    prune_weak_edges: bool = True
    
    # Graph structure
    allow_cycles: bool = False
    max_parents_per_node: int = 5
    
    # Physics constraints
    enforce_physics_constraints: bool = True
    physics_residual_threshold: float = PHYSICS_RESIDUAL_THRESHOLD


@dataclass
class PhysicsConfig:
    """Configuration for physics-guided learning."""
    
    # Loss function weights
    lambda_physics_init: float = LAMBDA_PHYSICS_INIT
    lambda_decay: float = LAMBDA_DECAY
    lambda_physics_min: float = LAMBDA_PHYSICS_MIN
    
    # Physics constraints
    use_swe_constraint: bool = True  # Shallow Water Equations
    use_inverse_barometer: bool = True
    use_wind_stress: bool = True
    
    # Training
    max_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class GatesConfig:
    """Configuration for pipeline gates (thresholds)."""
    
    # Stage thresholds
    fingerprint_threshold: float = FINGERPRINT_GATE
    ensemble_threshold: float = ENSEMBLE_GATE
    physics_residual_threshold: float = PHYSICS_RESIDUAL_THRESHOLD
    alert_threshold: float = ALERT_THRESHOLD
    
    # Gray zone
    gray_zone_historical_min: float = GRAY_ZONE_HISTORICAL_MIN
    enable_gray_zone: bool = True


@dataclass
class OutputConfig:
    """Configuration for pipeline output."""
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Saving options
    save_graphs: bool = True
    save_predictions: bool = True
    save_diagnostics: bool = True
    
    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"
    
    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Aggregates all stage-specific configs with sensible defaults.
    """
    
    # Sub-configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    causal_discovery: CausalDiscoveryConfig = field(default_factory=CausalDiscoveryConfig)
    graph_building: GraphBuildingConfig = field(default_factory=GraphBuildingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Global settings
    random_seed: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance with loaded parameters
            
        Example YAML structure:
            preprocessing:
              time_resolution_hours: 1.0
              normalize_data: true
            causal_discovery:
              tau_max: 24
              pc_alpha: 0.01
            gates:
              fingerprint_threshold: 0.6
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PipelineConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            PipelineConfig instance
        """
        # Extract sub-configs
        preprocessing_dict = config_dict.get("preprocessing", {})
        causal_discovery_dict = config_dict.get("causal_discovery", {})
        graph_building_dict = config_dict.get("graph_building", {})
        physics_dict = config_dict.get("physics", {})
        gates_dict = config_dict.get("gates", {})
        output_dict = config_dict.get("output", {})
        
        # Convert output_dir strings to Path
        if "output_dir" in output_dict and isinstance(output_dict["output_dir"], str):
            output_dict["output_dir"] = Path(output_dict["output_dir"])
        if "log_file" in output_dict and isinstance(output_dict["log_file"], str):
            output_dict["log_file"] = Path(output_dict["log_file"])
        
        return cls(
            preprocessing=PreprocessingConfig(**preprocessing_dict),
            causal_discovery=CausalDiscoveryConfig(**causal_discovery_dict),
            graph_building=GraphBuildingConfig(**graph_building_dict),
            physics=PhysicsConfig(**physics_dict),
            gates=GatesConfig(**gates_dict),
            output=OutputConfig(**output_dict),
            random_seed=config_dict.get("random_seed", 42),
            n_jobs=config_dict.get("n_jobs", -1),
            verbose=config_dict.get("verbose", True),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "preprocessing": {
                k: v for k, v in self.preprocessing.__dict__.items()
            },
            "causal_discovery": {
                k: v for k, v in self.causal_discovery.__dict__.items()
            },
            "graph_building": {
                k: v for k, v in self.graph_building.__dict__.items()
            },
            "physics": {
                k: v for k, v in self.physics.__dict__.items()
            },
            "gates": {
                k: v for k, v in self.gates.__dict__.items()
            },
            "output": {
                k: str(v) if isinstance(v, Path) else v 
                for k, v in self.output.__dict__.items()
            },
            "random_seed": self.random_seed,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }
    
    def save_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> PipelineConfig:
    """
    Get default pipeline configuration.
    
    Returns:
        PipelineConfig with all default values
    """
    return PipelineConfig()

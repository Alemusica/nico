"""
Pipeline stages for Surge-Shazam-DK.

Implements the staged prediction workflow:
1. Preprocessing: Data harmonization and quality control
2. Causal Discovery: PCMCI for lag detection
3. Graph Building: Construct causal graph
4. Ensemble Prediction: GNN + physics-guided learning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .config import PipelineConfig
from .gates import PipelineState, PipelineGates, Stage, GateResult


# =============================================================================
# Data structures for stage inputs/outputs
# =============================================================================

@dataclass
class PreprocessingResult:
    """Output from preprocessing stage."""
    
    # Harmonized data
    data: xr.Dataset
    
    # Metadata
    variables: list[str]
    time_range: tuple[pd.Timestamp, pd.Timestamp]
    spatial_extent: dict[str, float]
    
    # Quality metrics
    missing_data_pct: float
    outliers_removed: int
    
    # Additional info
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalDiscoveryResult:
    """Output from causal discovery stage."""
    
    # Causal links (source, target, lag, coefficient)
    causal_links: list[tuple[str, str, int, float]]
    
    # P-values for each link
    p_values: dict[tuple[str, str, int], float]
    
    # Lag matrix (max lag for each variable pair)
    lag_matrix: np.ndarray
    
    # Variable names
    variables: list[str]
    
    # Teleconnections detected
    teleconnections: list[dict[str, Any]] = field(default_factory=list)
    
    # Additional info
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphBuildingResult:
    """Output from graph building stage."""
    
    # Graph structure (adjacency matrix)
    adjacency_matrix: np.ndarray
    
    # Node features
    node_features: dict[str, np.ndarray]
    
    # Edge weights
    edge_weights: dict[tuple[int, int], float]
    
    # Variable to node mapping
    variable_names: list[str]
    
    # Physics validation
    physics_residual: float
    physics_constraints_satisfied: bool
    
    # Additional info
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    
    # Final prediction
    predicted_surge_m: float
    predicted_surge_uncertainty: float
    predicted_location: str
    predicted_time_hours: float
    
    # Confidence scores
    fingerprint_confidence: float
    gnn_confidence: float
    ensemble_confidence: float
    physics_residual: float
    
    # Pipeline state
    final_state: PipelineState
    gate_results: list[GateResult]
    
    # Intermediate results
    preprocessing_result: PreprocessingResult | None = None
    causal_discovery_result: CausalDiscoveryResult | None = None
    graph_building_result: GraphBuildingResult | None = None
    
    # Additional info
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Abstract base class for pipeline stages
# =============================================================================

class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage implements:
    - run(): Execute the stage logic
    - validate_input(): Check input data
    - get_metadata(): Return stage information
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline stage.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stage_name = self.__class__.__name__
    
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """
        Execute the stage logic.
        
        Args:
            input_data: Input for this stage
            
        Returns:
            Stage-specific output
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for this stage.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get stage metadata.
        
        Returns:
            Dictionary with stage information
        """
        return {
            "stage_name": self.stage_name,
            "config": self.config.to_dict(),
        }


# =============================================================================
# Concrete stage implementations
# =============================================================================

class PreprocessingStage(PipelineStage):
    """
    Preprocessing stage: Data harmonization and quality control.
    
    Uses DataHarmonizer (to be implemented in data/preprocessors/tensor_builder.py)
    for:
    - Temporal alignment
    - Spatial interpolation
    - Missing data handling
    - Outlier removal
    - Normalization
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize preprocessing stage.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.preproc_config = config.preprocessing
    
    def run(self, input_data: xr.Dataset | dict[str, xr.Dataset]) -> PreprocessingResult:
        """
        Run preprocessing on input data.
        
        Args:
            input_data: Raw data from multiple sources (xarray Dataset or dict of Datasets)
            
        Returns:
            PreprocessingResult with harmonized data
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data for preprocessing stage")
        
        # Convert single dataset to dict
        if isinstance(input_data, xr.Dataset):
            datasets = {"main": input_data}
        else:
            datasets = input_data
        
        # TODO: Use actual DataHarmonizer when implemented
        # For now, create a simple harmonized output
        harmonized_data = self._harmonize_data(datasets)
        
        # Quality control
        missing_pct, outliers_removed = self._quality_control(harmonized_data)
        
        # Get metadata
        time_range = (
            pd.Timestamp(harmonized_data.time.values[0]),
            pd.Timestamp(harmonized_data.time.values[-1])
        )
        
        spatial_extent = {
            "lat_min": float(harmonized_data.lat.min()),
            "lat_max": float(harmonized_data.lat.max()),
            "lon_min": float(harmonized_data.lon.min()),
            "lon_max": float(harmonized_data.lon.max()),
        }
        
        return PreprocessingResult(
            data=harmonized_data,
            variables=list(harmonized_data.data_vars),
            time_range=time_range,
            spatial_extent=spatial_extent,
            missing_data_pct=missing_pct,
            outliers_removed=outliers_removed,
            metadata={"stage": "preprocessing"}
        )
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid
        """
        if isinstance(input_data, xr.Dataset):
            return True
        
        if isinstance(input_data, dict):
            return all(isinstance(v, xr.Dataset) for v in input_data.values())
        
        return False
    
    def _harmonize_data(self, datasets: dict[str, xr.Dataset]) -> xr.Dataset:
        """
        Harmonize data from multiple sources.
        
        TODO: Replace with actual DataHarmonizer implementation
        """
        # Simple implementation: take first dataset
        # In reality, this should merge all datasets with proper interpolation
        first_dataset = list(datasets.values())[0]
        
        # Apply normalization if configured
        if self.preproc_config.normalize_data:
            first_dataset = self._normalize(first_dataset)
        
        return first_dataset
    
    def _normalize(self, data: xr.Dataset) -> xr.Dataset:
        """Apply normalization to data."""
        if self.preproc_config.normalization_method == "zscore":
            # Z-score normalization
            data = (data - data.mean()) / data.std()
        elif self.preproc_config.normalization_method == "minmax":
            # Min-max normalization
            data = (data - data.min()) / (data.max() - data.min())
        
        return data
    
    def _quality_control(self, data: xr.Dataset) -> tuple[float, int]:
        """
        Perform quality control.
        
        Returns:
            (missing_data_pct, outliers_removed)
        """
        # Calculate missing data percentage
        total_values = data.sizes.get("time", 0) * data.sizes.get("lat", 0) * data.sizes.get("lon", 0)
        if total_values == 0:
            return 0.0, 0
        
        # Count missing values across all variables
        missing_count = 0
        for var in data.data_vars:
            missing_count += int(data[var].isnull().sum())
        
        missing_pct = (missing_count / (total_values * len(data.data_vars))) * 100
        
        # Remove outliers if configured
        outliers_removed = 0
        if self.preproc_config.remove_outliers:
            for var in data.data_vars:
                mean = float(data[var].mean())
                std = float(data[var].std())
                threshold = self.preproc_config.outlier_std_threshold
                
                # Mark outliers as NaN
                outlier_mask = np.abs(data[var] - mean) > threshold * std
                outliers_removed += int(outlier_mask.sum())
        
        return missing_pct, outliers_removed


class CausalDiscoveryStage(PipelineStage):
    """
    Causal discovery stage: PCMCI for lag detection.
    
    Uses PCMCIRunner (to be implemented in causal/pcmci_runner.py) for:
    - Conditional independence testing
    - Lag detection
    - Teleconnection identification
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize causal discovery stage.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.causal_config = config.causal_discovery
    
    def run(self, preprocessing_result: PreprocessingResult) -> CausalDiscoveryResult:
        """
        Run causal discovery on preprocessed data.
        
        Args:
            preprocessing_result: Output from preprocessing stage
            
        Returns:
            CausalDiscoveryResult with discovered causal links
        """
        if not self.validate_input(preprocessing_result):
            raise ValueError("Invalid input for causal discovery stage")
        
        # TODO: Use actual PCMCIRunner when implemented
        # For now, create mock causal links
        causal_links, p_values = self._discover_causal_links(preprocessing_result.data)
        
        # Create lag matrix
        lag_matrix = self._create_lag_matrix(causal_links, preprocessing_result.variables)
        
        # Detect teleconnections if configured
        teleconnections = []
        if self.causal_config.detect_teleconnections:
            teleconnections = self._detect_teleconnections(causal_links)
        
        return CausalDiscoveryResult(
            causal_links=causal_links,
            p_values=p_values,
            lag_matrix=lag_matrix,
            variables=preprocessing_result.variables,
            teleconnections=teleconnections,
            metadata={"stage": "causal_discovery"}
        )
    
    def validate_input(self, preprocessing_result: Any) -> bool:
        """
        Validate input data.
        
        Args:
            preprocessing_result: Input to validate
            
        Returns:
            True if valid
        """
        if not isinstance(preprocessing_result, PreprocessingResult):
            return False
        
        return preprocessing_result.data is not None
    
    def _discover_causal_links(
        self, 
        data: xr.Dataset
    ) -> tuple[list[tuple[str, str, int, float]], dict[tuple[str, str, int], float]]:
        """
        Discover causal links using PCMCI.
        
        TODO: Replace with actual PCMCIRunner implementation
        
        Returns:
            (causal_links, p_values)
        """
        variables = list(data.data_vars)
        causal_links = []
        p_values = {}
        
        # Mock implementation: create some example links
        # In reality, this would run PCMCI algorithm
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Create a mock link with random strength
                    lag = np.random.randint(1, self.causal_config.tau_max)
                    coef = np.random.uniform(0.1, 0.5)
                    p_val = np.random.uniform(0, self.causal_config.pc_alpha * 2)
                    
                    # Only add link if p-value is significant
                    if p_val < self.causal_config.pc_alpha:
                        causal_links.append((var1, var2, lag, coef))
                        p_values[(var1, var2, lag)] = p_val
        
        return causal_links, p_values
    
    def _create_lag_matrix(
        self, 
        causal_links: list[tuple[str, str, int, float]], 
        variables: list[str]
    ) -> np.ndarray:
        """
        Create lag matrix from causal links.
        
        Args:
            causal_links: List of (source, target, lag, coef)
            variables: List of variable names
            
        Returns:
            Lag matrix (n_vars x n_vars)
        """
        n_vars = len(variables)
        lag_matrix = np.zeros((n_vars, n_vars))
        
        var_to_idx = {var: i for i, var in enumerate(variables)}
        
        for source, target, lag, _ in causal_links:
            if source in var_to_idx and target in var_to_idx:
                i = var_to_idx[source]
                j = var_to_idx[target]
                lag_matrix[i, j] = lag
        
        return lag_matrix
    
    def _detect_teleconnections(
        self, 
        causal_links: list[tuple[str, str, int, float]]
    ) -> list[dict[str, Any]]:
        """
        Detect teleconnections (long-distance causal links).
        
        TODO: Implement proper spatial distance calculation
        """
        teleconnections = []
        
        # Mock implementation
        # In reality, would calculate spatial distance and filter by threshold
        for source, target, lag, coef in causal_links:
            if lag > 12:  # Long lag suggests potential teleconnection
                teleconnections.append({
                    "source": source,
                    "target": target,
                    "lag_hours": lag,
                    "coefficient": coef,
                    "distance_km": 500.0,  # Mock distance
                })
        
        return teleconnections


class GraphBuildingStage(PipelineStage):
    """
    Graph building stage: Construct causal graph from discovered links.
    
    Uses CausalGraphBuilder (to be implemented in causal/graph_builder.py) for:
    - Node and edge creation
    - Graph pruning
    - Physics constraint validation
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize graph building stage.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        self.graph_config = config.graph_building
    
    def run(self, causal_discovery_result: CausalDiscoveryResult) -> GraphBuildingResult:
        """
        Build causal graph from discovered links.
        
        Args:
            causal_discovery_result: Output from causal discovery stage
            
        Returns:
            GraphBuildingResult with graph structure
        """
        if not self.validate_input(causal_discovery_result):
            raise ValueError("Invalid input for graph building stage")
        
        # Build graph structure
        adjacency_matrix = self._build_adjacency_matrix(
            causal_discovery_result.causal_links,
            causal_discovery_result.variables
        )
        
        # Extract edge weights
        edge_weights = self._extract_edge_weights(
            causal_discovery_result.causal_links,
            causal_discovery_result.variables
        )
        
        # Create node features (mock)
        node_features = self._create_node_features(causal_discovery_result.variables)
        
        # Validate physics constraints
        physics_residual, constraints_satisfied = self._validate_physics()
        
        return GraphBuildingResult(
            adjacency_matrix=adjacency_matrix,
            node_features=node_features,
            edge_weights=edge_weights,
            variable_names=causal_discovery_result.variables,
            physics_residual=physics_residual,
            physics_constraints_satisfied=constraints_satisfied,
            metadata={"stage": "graph_building"}
        )
    
    def validate_input(self, causal_discovery_result: Any) -> bool:
        """
        Validate input data.
        
        Args:
            causal_discovery_result: Input to validate
            
        Returns:
            True if valid
        """
        if not isinstance(causal_discovery_result, CausalDiscoveryResult):
            return False
        
        return len(causal_discovery_result.causal_links) > 0
    
    def _build_adjacency_matrix(
        self,
        causal_links: list[tuple[str, str, int, float]],
        variables: list[str]
    ) -> np.ndarray:
        """
        Build adjacency matrix from causal links.
        
        Args:
            causal_links: List of (source, target, lag, coef)
            variables: List of variable names
            
        Returns:
            Adjacency matrix (n_vars x n_vars)
        """
        n_vars = len(variables)
        adjacency = np.zeros((n_vars, n_vars))
        
        var_to_idx = {var: i for i, var in enumerate(variables)}
        
        for source, target, _, coef in causal_links:
            if source in var_to_idx and target in var_to_idx:
                i = var_to_idx[source]
                j = var_to_idx[target]
                
                # Apply pruning threshold
                if abs(coef) >= self.graph_config.min_edge_weight:
                    adjacency[i, j] = 1
        
        return adjacency
    
    def _extract_edge_weights(
        self,
        causal_links: list[tuple[str, str, int, float]],
        variables: list[str]
    ) -> dict[tuple[int, int], float]:
        """
        Extract edge weights from causal links.
        
        Args:
            causal_links: List of (source, target, lag, coef)
            variables: List of variable names
            
        Returns:
            Dictionary mapping (source_idx, target_idx) to weight
        """
        var_to_idx = {var: i for i, var in enumerate(variables)}
        edge_weights = {}
        
        for source, target, _, coef in causal_links:
            if source in var_to_idx and target in var_to_idx:
                i = var_to_idx[source]
                j = var_to_idx[target]
                
                if abs(coef) >= self.graph_config.min_edge_weight:
                    edge_weights[(i, j)] = coef
        
        return edge_weights
    
    def _create_node_features(self, variables: list[str]) -> dict[str, np.ndarray]:
        """
        Create node features for each variable.
        
        TODO: Implement proper feature extraction
        """
        node_features = {}
        
        for var in variables:
            # Mock features (in reality, would extract from data)
            node_features[var] = np.random.randn(10)
        
        return node_features
    
    def _validate_physics(self) -> tuple[float, bool]:
        """
        Validate graph against physics constraints.
        
        TODO: Implement actual physics validation
        
        Returns:
            (physics_residual, constraints_satisfied)
        """
        # Mock validation
        if self.graph_config.enforce_physics_constraints:
            physics_residual = np.random.uniform(0, 0.1)
            constraints_satisfied = physics_residual < self.graph_config.physics_residual_threshold
        else:
            physics_residual = 0.0
            constraints_satisfied = True
        
        return physics_residual, constraints_satisfied


# =============================================================================
# Main pipeline orchestrator
# =============================================================================

class SurgeShazamPipeline:
    """
    Main pipeline orchestrator for Surge-Shazam-DK.
    
    Coordinates all stages:
    1. Preprocessing
    2. Causal Discovery
    3. Graph Building
    4. Gate-based prediction
    """
    
    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Initialize stages
        self.preprocessing_stage = PreprocessingStage(self.config)
        self.causal_discovery_stage = CausalDiscoveryStage(self.config)
        self.graph_building_stage = GraphBuildingStage(self.config)
        
        # Initialize gates
        self.gates = PipelineGates()
    
    def run(
        self,
        input_data: xr.Dataset | dict[str, xr.Dataset],
        initial_fingerprint_confidence: float = 0.0
    ) -> PipelineResult:
        """
        Run full pipeline.
        
        Args:
            input_data: Raw input data
            initial_fingerprint_confidence: Initial fingerprint match confidence
            
        Returns:
            PipelineResult with predictions and metadata
        """
        # Stage 1: Preprocessing
        if self.config.verbose:
            print("Stage 1: Preprocessing...")
        
        preprocessing_result = self.preprocessing_stage.run(input_data)
        
        # Stage 2: Causal Discovery
        if self.config.verbose:
            print("Stage 2: Causal Discovery...")
        
        causal_discovery_result = self.causal_discovery_stage.run(preprocessing_result)
        
        # Stage 3: Graph Building
        if self.config.verbose:
            print("Stage 3: Graph Building...")
        
        graph_building_result = self.graph_building_stage.run(causal_discovery_result)
        
        # Stage 4: Gate-based prediction
        if self.config.verbose:
            print("Stage 4: Gate evaluation...")
        
        final_state, gate_results = self._run_gates(
            initial_fingerprint_confidence,
            graph_building_result
        )
        
        # Generate final prediction
        return self._generate_prediction(
            final_state,
            gate_results,
            preprocessing_result,
            causal_discovery_result,
            graph_building_result
        )
    
    def _run_gates(
        self,
        fingerprint_confidence: float,
        graph_result: GraphBuildingResult
    ) -> tuple[PipelineState, list[GateResult]]:
        """
        Run gate-based evaluation.
        
        Args:
            fingerprint_confidence: Fingerprint match confidence
            graph_result: Graph building result
            
        Returns:
            (final_state, gate_results)
        """
        # Create initial state
        initial_state = PipelineState(
            current_stage=Stage.IDLE,
            fingerprint_confidence=fingerprint_confidence,
            gnn_confidence=0.75,  # Mock GNN confidence
            physics_residual=graph_result.physics_residual,
            ensemble_confidence=0.0,
        )
        
        # Run through gates
        final_state, gate_results = self.gates.run_full_pipeline(initial_state)
        
        return final_state, gate_results
    
    def _generate_prediction(
        self,
        final_state: PipelineState,
        gate_results: list[GateResult],
        preprocessing_result: PreprocessingResult,
        causal_discovery_result: CausalDiscoveryResult,
        graph_building_result: GraphBuildingResult
    ) -> PipelineResult:
        """
        Generate final prediction from pipeline state.
        
        Args:
            final_state: Final pipeline state
            gate_results: Results from gate evaluation
            preprocessing_result: Preprocessing output
            causal_discovery_result: Causal discovery output
            graph_building_result: Graph building output
            
        Returns:
            Complete pipeline result
        """
        # Mock prediction values (in reality, would come from GNN)
        predicted_surge = 1.5  # meters
        uncertainty = 0.3  # meters
        location = "Copenhagen Harbor"
        time_hours = 24.0  # hours ahead
        
        return PipelineResult(
            predicted_surge_m=predicted_surge,
            predicted_surge_uncertainty=uncertainty,
            predicted_location=location,
            predicted_time_hours=time_hours,
            fingerprint_confidence=final_state.fingerprint_confidence,
            gnn_confidence=final_state.gnn_confidence,
            ensemble_confidence=final_state.ensemble_confidence,
            physics_residual=final_state.physics_residual,
            final_state=final_state,
            gate_results=gate_results,
            preprocessing_result=preprocessing_result,
            causal_discovery_result=causal_discovery_result,
            graph_building_result=graph_building_result,
            metadata={
                "pipeline_version": "1.0",
                "config": self.config.to_dict(),
            }
        )
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "SurgeShazamPipeline":
        """
        Create pipeline from YAML configuration.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Configured SurgeShazamPipeline instance
        """
        config = PipelineConfig.from_yaml(config_path)
        return cls(config)
    
    def save_config(self, output_path: str | Path) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config
        """
        self.config.save_yaml(output_path)

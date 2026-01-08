"""
Tests for pipeline stages and configuration.

Tests all pipeline components:
- Configuration loading and saving
- Individual stage execution
- Full pipeline orchestration
- Gate integration
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from src.surge_shazam.pipeline.config import (
    PipelineConfig,
    PreprocessingConfig,
    CausalDiscoveryConfig,
    GraphBuildingConfig,
    PhysicsConfig,
    GatesConfig,
    OutputConfig,
    get_default_config,
)
from src.surge_shazam.pipeline.stages import (
    PreprocessingStage,
    CausalDiscoveryStage,
    GraphBuildingStage,
    SurgeShazamPipeline,
    PreprocessingResult,
    CausalDiscoveryResult,
    GraphBuildingResult,
    PipelineResult,
)
from src.surge_shazam.pipeline.gates import Stage, AlertLevel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    # Create sample data (10 time steps, 5x5 spatial grid)
    time = pd.date_range("2024-01-01", periods=10, freq="1h")
    lat = np.linspace(54.5, 58.0, 5)
    lon = np.linspace(7.5, 15.5, 5)
    
    data = xr.Dataset(
        {
            "wind_u": (["time", "lat", "lon"], np.random.randn(10, 5, 5)),
            "wind_v": (["time", "lat", "lon"], np.random.randn(10, 5, 5)),
            "pressure": (["time", "lat", "lon"], np.random.randn(10, 5, 5) + 1013.0),
            "sea_surface_height": (["time", "lat", "lon"], np.random.randn(10, 5, 5) * 0.5),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )
    
    return data


@pytest.fixture
def default_config():
    """Create default pipeline configuration."""
    return get_default_config()


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Create a sample YAML configuration file."""
    config_dict = {
        "preprocessing": {
            "time_resolution_hours": 2.0,
            "normalize_data": True,
            "normalization_method": "minmax",
        },
        "causal_discovery": {
            "tau_max": 12,
            "pc_alpha": 0.05,
            "detect_teleconnections": False,
        },
        "gates": {
            "fingerprint_threshold": 0.65,
            "alert_threshold": 0.85,
        },
        "random_seed": 123,
        "verbose": False,
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
    
    return config_path


# =============================================================================
# Configuration Tests
# =============================================================================

def test_default_config_creation():
    """Test creating default configuration."""
    config = get_default_config()
    
    assert isinstance(config, PipelineConfig)
    assert isinstance(config.preprocessing, PreprocessingConfig)
    assert isinstance(config.causal_discovery, CausalDiscoveryConfig)
    assert isinstance(config.graph_building, GraphBuildingConfig)
    assert isinstance(config.physics, PhysicsConfig)
    assert isinstance(config.gates, GatesConfig)
    assert isinstance(config.output, OutputConfig)


def test_config_from_yaml(sample_yaml_config):
    """Test loading configuration from YAML."""
    config = PipelineConfig.from_yaml(sample_yaml_config)
    
    assert config.preprocessing.time_resolution_hours == 2.0
    assert config.preprocessing.normalization_method == "minmax"
    assert config.causal_discovery.tau_max == 12
    assert config.causal_discovery.pc_alpha == 0.05
    assert config.causal_discovery.detect_teleconnections is False
    assert config.gates.fingerprint_threshold == 0.65
    assert config.gates.alert_threshold == 0.85
    assert config.random_seed == 123
    assert config.verbose is False


def test_config_to_dict(default_config):
    """Test converting configuration to dictionary."""
    config_dict = default_config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "preprocessing" in config_dict
    assert "causal_discovery" in config_dict
    assert "graph_building" in config_dict
    assert "physics" in config_dict
    assert "gates" in config_dict
    assert "output" in config_dict
    assert "random_seed" in config_dict


def test_config_save_yaml(default_config, tmp_path):
    """Test saving configuration to YAML."""
    output_path = tmp_path / "saved_config.yaml"
    default_config.save_yaml(output_path)
    
    assert output_path.exists()
    
    # Verify it can be loaded back
    loaded_config = PipelineConfig.from_yaml(output_path)
    assert loaded_config.random_seed == default_config.random_seed


def test_config_from_dict():
    """Test creating configuration from dictionary."""
    config_dict = {
        "preprocessing": {
            "normalize_data": False,
        },
        "random_seed": 999,
    }
    
    config = PipelineConfig.from_dict(config_dict)
    
    assert config.preprocessing.normalize_data is False
    assert config.random_seed == 999


def test_config_invalid_yaml():
    """Test loading from non-existent YAML file."""
    with pytest.raises(FileNotFoundError):
        PipelineConfig.from_yaml("nonexistent.yaml")


# =============================================================================
# Preprocessing Stage Tests
# =============================================================================

def test_preprocessing_stage_initialization(default_config):
    """Test preprocessing stage initialization."""
    stage = PreprocessingStage(default_config)
    
    assert stage.stage_name == "PreprocessingStage"
    assert stage.config == default_config
    assert stage.preproc_config == default_config.preprocessing


def test_preprocessing_stage_run(default_config, sample_dataset):
    """Test preprocessing stage execution."""
    stage = PreprocessingStage(default_config)
    result = stage.run(sample_dataset)
    
    assert isinstance(result, PreprocessingResult)
    assert isinstance(result.data, xr.Dataset)
    assert len(result.variables) > 0
    assert isinstance(result.time_range, tuple)
    assert isinstance(result.spatial_extent, dict)
    assert result.missing_data_pct >= 0


def test_preprocessing_validate_input(default_config, sample_dataset):
    """Test preprocessing input validation."""
    stage = PreprocessingStage(default_config)
    
    # Valid inputs
    assert stage.validate_input(sample_dataset) is True
    assert stage.validate_input({"source1": sample_dataset}) is True
    
    # Invalid inputs
    assert stage.validate_input("not a dataset") is False
    assert stage.validate_input(None) is False


def test_preprocessing_quality_control(default_config, sample_dataset):
    """Test quality control in preprocessing."""
    # Add some NaN values
    sample_dataset["wind_u"][0, 0, 0] = np.nan
    
    stage = PreprocessingStage(default_config)
    result = stage.run(sample_dataset)
    
    assert result.missing_data_pct > 0


# =============================================================================
# Causal Discovery Stage Tests
# =============================================================================

def test_causal_discovery_stage_initialization(default_config):
    """Test causal discovery stage initialization."""
    stage = CausalDiscoveryStage(default_config)
    
    assert stage.stage_name == "CausalDiscoveryStage"
    assert stage.causal_config == default_config.causal_discovery


def test_causal_discovery_stage_run(default_config, sample_dataset):
    """Test causal discovery stage execution."""
    # First run preprocessing
    preproc_stage = PreprocessingStage(default_config)
    preproc_result = preproc_stage.run(sample_dataset)
    
    # Run causal discovery
    causal_stage = CausalDiscoveryStage(default_config)
    result = causal_stage.run(preproc_result)
    
    assert isinstance(result, CausalDiscoveryResult)
    assert isinstance(result.causal_links, list)
    assert isinstance(result.p_values, dict)
    assert isinstance(result.lag_matrix, np.ndarray)
    assert len(result.variables) > 0


def test_causal_discovery_validate_input(default_config):
    """Test causal discovery input validation."""
    stage = CausalDiscoveryStage(default_config)
    
    # Invalid inputs
    assert stage.validate_input("not a preprocessing result") is False
    assert stage.validate_input(None) is False


def test_causal_discovery_lag_matrix(default_config, sample_dataset):
    """Test lag matrix creation."""
    preproc_stage = PreprocessingStage(default_config)
    preproc_result = preproc_stage.run(sample_dataset)
    
    causal_stage = CausalDiscoveryStage(default_config)
    result = causal_stage.run(preproc_result)
    
    # Check lag matrix shape
    n_vars = len(result.variables)
    assert result.lag_matrix.shape == (n_vars, n_vars)


def test_causal_discovery_teleconnections(default_config, sample_dataset):
    """Test teleconnection detection."""
    # Enable teleconnection detection
    default_config.causal_discovery.detect_teleconnections = True
    
    preproc_stage = PreprocessingStage(default_config)
    preproc_result = preproc_stage.run(sample_dataset)
    
    causal_stage = CausalDiscoveryStage(default_config)
    result = causal_stage.run(preproc_result)
    
    assert isinstance(result.teleconnections, list)


# =============================================================================
# Graph Building Stage Tests
# =============================================================================

def test_graph_building_stage_initialization(default_config):
    """Test graph building stage initialization."""
    stage = GraphBuildingStage(default_config)
    
    assert stage.stage_name == "GraphBuildingStage"
    assert stage.graph_config == default_config.graph_building


def test_graph_building_stage_run(default_config, sample_dataset):
    """Test graph building stage execution."""
    # Run preprocessing and causal discovery first
    preproc_stage = PreprocessingStage(default_config)
    preproc_result = preproc_stage.run(sample_dataset)
    
    causal_stage = CausalDiscoveryStage(default_config)
    causal_result = causal_stage.run(preproc_result)
    
    # Run graph building
    graph_stage = GraphBuildingStage(default_config)
    result = graph_stage.run(causal_result)
    
    assert isinstance(result, GraphBuildingResult)
    assert isinstance(result.adjacency_matrix, np.ndarray)
    assert isinstance(result.node_features, dict)
    assert isinstance(result.edge_weights, dict)
    assert len(result.variable_names) > 0
    assert isinstance(result.physics_residual, float)
    assert isinstance(result.physics_constraints_satisfied, bool)


def test_graph_building_validate_input(default_config):
    """Test graph building input validation."""
    stage = GraphBuildingStage(default_config)
    
    # Invalid inputs
    assert stage.validate_input("not a causal result") is False
    assert stage.validate_input(None) is False


def test_graph_building_adjacency_matrix(default_config, sample_dataset):
    """Test adjacency matrix creation."""
    preproc_stage = PreprocessingStage(default_config)
    preproc_result = preproc_stage.run(sample_dataset)
    
    causal_stage = CausalDiscoveryStage(default_config)
    causal_result = causal_stage.run(preproc_result)
    
    graph_stage = GraphBuildingStage(default_config)
    result = graph_stage.run(causal_result)
    
    # Check adjacency matrix properties
    n_vars = len(result.variable_names)
    assert result.adjacency_matrix.shape == (n_vars, n_vars)
    assert np.all((result.adjacency_matrix == 0) | (result.adjacency_matrix == 1))


# =============================================================================
# Full Pipeline Tests
# =============================================================================

def test_pipeline_initialization(default_config):
    """Test pipeline initialization."""
    pipeline = SurgeShazamPipeline(default_config)
    
    assert pipeline.config == default_config
    assert isinstance(pipeline.preprocessing_stage, PreprocessingStage)
    assert isinstance(pipeline.causal_discovery_stage, CausalDiscoveryStage)
    assert isinstance(pipeline.graph_building_stage, GraphBuildingStage)


def test_pipeline_run(default_config, sample_dataset):
    """Test full pipeline execution."""
    pipeline = SurgeShazamPipeline(default_config)
    result = pipeline.run(sample_dataset, initial_fingerprint_confidence=0.7)
    
    assert isinstance(result, PipelineResult)
    assert isinstance(result.predicted_surge_m, float)
    assert isinstance(result.predicted_surge_uncertainty, float)
    assert isinstance(result.predicted_location, str)
    assert isinstance(result.predicted_time_hours, float)
    assert result.fingerprint_confidence >= 0
    assert result.gnn_confidence >= 0
    assert result.ensemble_confidence >= 0
    assert result.physics_residual >= 0
    assert isinstance(result.final_state.alert_level, AlertLevel)


def test_pipeline_from_yaml(sample_yaml_config, sample_dataset):
    """Test creating pipeline from YAML config."""
    pipeline = SurgeShazamPipeline.from_yaml(sample_yaml_config)
    
    assert pipeline.config.random_seed == 123
    assert pipeline.config.verbose is False
    
    # Test running pipeline
    result = pipeline.run(sample_dataset, initial_fingerprint_confidence=0.7)
    assert isinstance(result, PipelineResult)


def test_pipeline_save_config(default_config, tmp_path):
    """Test saving pipeline configuration."""
    pipeline = SurgeShazamPipeline(default_config)
    output_path = tmp_path / "pipeline_config.yaml"
    
    pipeline.save_config(output_path)
    
    assert output_path.exists()


def test_pipeline_with_different_configs(sample_dataset):
    """Test pipeline with different configurations."""
    # Configuration 1: Verbose mode
    config1 = PipelineConfig()
    config1.verbose = True
    pipeline1 = SurgeShazamPipeline(config1)
    result1 = pipeline1.run(sample_dataset, initial_fingerprint_confidence=0.8)
    
    assert isinstance(result1, PipelineResult)
    
    # Configuration 2: Non-verbose mode
    config2 = PipelineConfig()
    config2.verbose = False
    pipeline2 = SurgeShazamPipeline(config2)
    result2 = pipeline2.run(sample_dataset, initial_fingerprint_confidence=0.8)
    
    assert isinstance(result2, PipelineResult)


def test_pipeline_gate_integration(default_config, sample_dataset):
    """Test integration with pipeline gates."""
    pipeline = SurgeShazamPipeline(default_config)
    
    # High fingerprint confidence
    result = pipeline.run(sample_dataset, initial_fingerprint_confidence=0.85)
    
    assert len(result.gate_results) > 0
    assert result.final_state.current_stage in [Stage.IDLE, Stage.FINAL]


def test_pipeline_low_confidence(default_config, sample_dataset):
    """Test pipeline with low initial confidence."""
    pipeline = SurgeShazamPipeline(default_config)
    
    # Low fingerprint confidence (should fail early)
    result = pipeline.run(sample_dataset, initial_fingerprint_confidence=0.3)
    
    assert isinstance(result, PipelineResult)
    # Should still produce a result, even if confidence is low


def test_pipeline_metadata(default_config, sample_dataset):
    """Test pipeline result metadata."""
    pipeline = SurgeShazamPipeline(default_config)
    result = pipeline.run(sample_dataset, initial_fingerprint_confidence=0.7)
    
    assert "pipeline_version" in result.metadata
    assert "config" in result.metadata
    
    # Check intermediate results
    assert isinstance(result.preprocessing_result, PreprocessingResult)
    assert isinstance(result.causal_discovery_result, CausalDiscoveryResult)
    assert isinstance(result.graph_building_result, GraphBuildingResult)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

def test_preprocessing_with_empty_dataset(default_config):
    """Test preprocessing with empty dataset."""
    empty_dataset = xr.Dataset()
    stage = PreprocessingStage(default_config)
    
    # Should handle gracefully (or raise informative error)
    # This depends on implementation details


def test_pipeline_with_all_nan_data(default_config):
    """Test pipeline with all NaN data."""
    time = pd.date_range("2024-01-01", periods=10, freq="1h")
    lat = np.linspace(54.5, 58.0, 5)
    lon = np.linspace(7.5, 15.5, 5)
    
    data = xr.Dataset(
        {
            "wind_u": (["time", "lat", "lon"], np.full((10, 5, 5), np.nan)),
        },
        coords={"time": time, "lat": lat, "lon": lon}
    )
    
    pipeline = SurgeShazamPipeline(default_config)
    
    # Should raise an error or handle gracefully
    # With all NaN data, causal discovery will find no links
    # This should fail at graph building stage (no links = invalid input)
    with pytest.raises(ValueError, match="Invalid input for graph building stage"):
        result = pipeline.run(data, initial_fingerprint_confidence=0.7)


def test_config_roundtrip(default_config, tmp_path):
    """Test config save/load roundtrip."""
    # Save config
    save_path = tmp_path / "roundtrip_config.yaml"
    default_config.save_yaml(save_path)
    
    # Load config
    loaded_config = PipelineConfig.from_yaml(save_path)
    
    # Compare (should be equivalent)
    assert loaded_config.random_seed == default_config.random_seed
    assert loaded_config.n_jobs == default_config.n_jobs
    assert loaded_config.verbose == default_config.verbose


# =============================================================================
# Performance and Integration Tests
# =============================================================================

def test_pipeline_performance(default_config, sample_dataset):
    """Test pipeline runs within reasonable time."""
    import time
    
    pipeline = SurgeShazamPipeline(default_config)
    
    start = time.time()
    result = pipeline.run(sample_dataset, initial_fingerprint_confidence=0.7)
    elapsed = time.time() - start
    
    # Should complete within 10 seconds for small dataset
    assert elapsed < 10.0
    assert isinstance(result, PipelineResult)


def test_multiple_pipeline_runs(default_config, sample_dataset):
    """Test running pipeline multiple times."""
    pipeline = SurgeShazamPipeline(default_config)
    
    results = []
    for confidence in [0.5, 0.7, 0.9]:
        result = pipeline.run(sample_dataset, initial_fingerprint_confidence=confidence)
        results.append(result)
    
    assert len(results) == 3
    assert all(isinstance(r, PipelineResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

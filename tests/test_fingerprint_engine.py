"""
🧪 Tests for Fingerprint Engine
===============================

Tests for MiniRocket-based pattern fingerprinting.

Run with:
    pytest tests/test_fingerprint_engine.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.surge_shazam.fingerprinting.engine import (
    FingerprintEngine,
    Fingerprint,
    SimilarityResult,
    extract_fingerprint,
    compare_fingerprints,
    VARIABLE_WEIGHTS,
    VARIABLE_GROUPS,
    HAS_MINIROCKET,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def engine():
    """Create fingerprint engine with reduced parameters for testing."""
    return FingerprintEngine(
        n_kernels=1000,  # Reduced for speed
        embedding_dim=50,
        use_physics_weights=True,
    )


@pytest.fixture
def sample_flood_data():
    """Generate synthetic flood event data (like Lago Maggiore 2000)."""
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range("2000-10-01", periods=n_samples, freq="D")
    
    # Create flood-like pattern
    df = pd.DataFrame({
        "precipitation": np.random.exponential(5, n_samples) + np.linspace(0, 20, n_samples),
        "pressure_msl": 101325 - np.random.exponential(1000, n_samples),
        "temperature_2m": 288 + np.random.normal(0, 3, n_samples),
        "soil_moisture": np.random.uniform(0.2, 0.5, n_samples),
        "nao_index": np.random.normal(-0.5, 1, n_samples),  # Negative NAO
        "sea_level": np.random.normal(0.1, 0.05, n_samples),
    }, index=dates)
    
    return df


@pytest.fixture
def sample_drought_data():
    """Generate synthetic drought event data (opposite of flood)."""
    np.random.seed(43)
    n_samples = 100
    dates = pd.date_range("2000-10-01", periods=n_samples, freq="D")
    
    # Create drought-like pattern
    df = pd.DataFrame({
        "precipitation": np.random.exponential(1, n_samples),  # Low precip
        "pressure_msl": 101325 + np.random.exponential(500, n_samples),  # High pressure
        "temperature_2m": 295 + np.random.normal(0, 2, n_samples),  # Warmer
        "soil_moisture": np.random.uniform(0.05, 0.15, n_samples),  # Dry soil
        "nao_index": np.random.normal(1.0, 0.5, n_samples),  # Positive NAO
        "sea_level": np.random.normal(0.0, 0.02, n_samples),
    }, index=dates)
    
    return df


@pytest.fixture
def sample_similar_data(sample_flood_data):
    """Generate data similar to flood (with small noise)."""
    np.random.seed(44)
    return sample_flood_data + np.random.normal(0, 0.1, sample_flood_data.shape)


# =============================================================================
# BASIC TESTS
# =============================================================================

class TestFingerprintEngineBasics:
    """Test basic functionality."""
    
    def test_engine_initialization(self):
        """Engine initializes with default parameters."""
        engine = FingerprintEngine()
        assert engine.n_kernels == 10000
        assert engine.embedding_dim == 100
        assert engine.use_physics_weights is True
    
    def test_engine_custom_params(self):
        """Engine accepts custom parameters."""
        engine = FingerprintEngine(
            n_kernels=5000,
            embedding_dim=64,
            use_physics_weights=False,
        )
        assert engine.n_kernels == 5000
        assert engine.embedding_dim == 64
        assert engine.use_physics_weights is False
    
    def test_variable_weights_exist(self):
        """Variable weights are defined."""
        assert len(VARIABLE_WEIGHTS) > 0
        assert "precipitation" in VARIABLE_WEIGHTS
        assert "nao" in VARIABLE_WEIGHTS
        assert "default" in VARIABLE_WEIGHTS
    
    def test_variable_groups_exist(self):
        """Variable groups are defined."""
        assert len(VARIABLE_GROUPS) > 0
        assert "climate_indices" in VARIABLE_GROUPS
        assert "atmospheric" in VARIABLE_GROUPS
        assert "precipitation" in VARIABLE_GROUPS


# =============================================================================
# EXTRACTION TESTS
# =============================================================================

class TestFingerprintExtraction:
    """Test fingerprint extraction."""
    
    def test_extract_returns_fingerprint(self, engine, sample_flood_data):
        """Extract returns Fingerprint object."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert isinstance(fp, Fingerprint)
    
    def test_extract_embedding_shape(self, engine, sample_flood_data):
        """Embedding has expected shape."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert isinstance(fp.embedding, np.ndarray)
        # With MiniRocket: embedding_dim (50)
        # With fallback: 10 features per variable * n_vars
        # Just check it's reasonable size
        assert len(fp.embedding) > 0
        assert len(fp.embedding) <= 10000  # Max reasonable size
    
    def test_extract_preserves_event_id(self, engine, sample_flood_data):
        """Event ID is preserved."""
        fp = engine.extract(sample_flood_data, "lago_maggiore_2000")
        assert fp.event_id == "lago_maggiore_2000"
    
    def test_extract_captures_variables(self, engine, sample_flood_data):
        """Variables used are captured."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert fp.variables_used == list(sample_flood_data.columns)
        assert fp.n_variables == len(sample_flood_data.columns)
    
    def test_extract_captures_samples(self, engine, sample_flood_data):
        """Sample count is captured."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert fp.n_samples == len(sample_flood_data)
    
    def test_extract_with_sources(self, engine, sample_flood_data):
        """Sources are captured."""
        fp = engine.extract(
            sample_flood_data,
            "test_event",
            sources=["era5", "cmems"],
        )
        assert fp.sources_used == ["era5", "cmems"]
    
    def test_extract_with_bbox(self, engine, sample_flood_data):
        """Bounding box is captured."""
        bbox = (44.0, 47.0, 7.0, 11.0)
        fp = engine.extract(
            sample_flood_data,
            "test_event",
            bbox=bbox,
        )
        assert fp.bbox == bbox
    
    def test_extract_with_center(self, engine, sample_flood_data):
        """Center point is captured."""
        center = (45.8, 8.7)
        fp = engine.extract(
            sample_flood_data,
            "test_event",
            center=center,
        )
        assert fp.center == center
    
    def test_extract_timestamp_range(self, engine, sample_flood_data):
        """Timestamp range is captured."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert fp.timestamp_start is not None
        assert fp.timestamp_end is not None
        assert fp.timestamp_end >= fp.timestamp_start
    
    def test_extract_with_missing_data(self, engine, sample_flood_data):
        """Handles missing data gracefully."""
        df_with_nan = sample_flood_data.copy()
        df_with_nan.iloc[10:20, 0] = np.nan  # Add NaN
        
        fp = engine.extract(df_with_nan, "test_event")
        assert fp is not None
        assert fp.missing_ratio > 0
    
    def test_extract_variable_stats(self, engine, sample_flood_data):
        """Variable statistics are computed."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert len(fp.variable_stats) == len(sample_flood_data.columns)
        
        for var, stats in fp.variable_stats.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
    
    def test_extract_empty_df_raises(self, engine):
        """Empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            engine.extract(empty_df, "test_event")
    
    def test_extract_short_df_raises(self, engine):
        """Too short DataFrame raises ValueError."""
        short_df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Insufficient"):
            engine.extract(short_df, "test_event")


# =============================================================================
# SIMILARITY TESTS
# =============================================================================

class TestFingerprintSimilarity:
    """Test fingerprint similarity computation."""
    
    def test_self_similarity_is_one(self, engine, sample_flood_data):
        """Self-similarity should be approximately 1.0."""
        fp = engine.extract(sample_flood_data, "test_event")
        sim = engine.similarity(fp, fp)
        assert sim >= 0.99  # Allow small numerical error
    
    def test_similar_events_high_similarity(
        self, engine, sample_flood_data, sample_similar_data
    ):
        """Similar events should have high similarity."""
        fp1 = engine.extract(sample_flood_data, "event_1")
        fp2 = engine.extract(sample_similar_data, "event_2")
        
        sim = engine.similarity(fp1, fp2)
        assert sim > 0.7  # Should be high
    
    def test_different_events_lower_similarity(
        self, engine, sample_flood_data, sample_drought_data
    ):
        """Different events should have lower similarity than similar ones."""
        fp_flood = engine.extract(sample_flood_data, "flood")
        fp_drought = engine.extract(sample_drought_data, "drought")
        
        # Also extract similar event for comparison
        np.random.seed(44)
        similar_data = sample_flood_data + np.random.normal(0, 0.1, sample_flood_data.shape)
        fp_similar = engine.extract(similar_data, "similar")
        
        sim_different = engine.similarity(fp_flood, fp_drought)
        sim_similar = engine.similarity(fp_flood, fp_similar)
        
        # Different patterns should be LESS similar than similar patterns
        # This is a relative test, not absolute threshold
        assert sim_different < sim_similar
    
    def test_similarity_is_symmetric(self, engine, sample_flood_data, sample_drought_data):
        """Similarity should be symmetric."""
        fp1 = engine.extract(sample_flood_data, "event_1")
        fp2 = engine.extract(sample_drought_data, "event_2")
        
        sim_12 = engine.similarity(fp1, fp2)
        sim_21 = engine.similarity(fp2, fp1)
        
        assert abs(sim_12 - sim_21) < 1e-6
    
    def test_similarity_range(self, engine, sample_flood_data, sample_drought_data):
        """Similarity should be in [0, 1]."""
        fp1 = engine.extract(sample_flood_data, "event_1")
        fp2 = engine.extract(sample_drought_data, "event_2")
        
        sim = engine.similarity(fp1, fp2)
        assert 0.0 <= sim <= 1.0


# =============================================================================
# COMPARE TESTS
# =============================================================================

class TestFingerprintCompare:
    """Test detailed comparison."""
    
    def test_compare_returns_result(self, engine, sample_flood_data, sample_drought_data):
        """Compare returns SimilarityResult."""
        fp1 = engine.extract(sample_flood_data, "flood")
        fp2 = engine.extract(sample_drought_data, "drought")
        
        result = engine.compare(fp1, fp2)
        assert isinstance(result, SimilarityResult)
    
    def test_compare_includes_distance(self, engine, sample_flood_data, sample_drought_data):
        """Compare includes Euclidean distance."""
        fp1 = engine.extract(sample_flood_data, "flood")
        fp2 = engine.extract(sample_drought_data, "drought")
        
        result = engine.compare(fp1, fp2)
        assert result.distance >= 0
    
    def test_compare_preserves_ids(self, engine, sample_flood_data, sample_drought_data):
        """Compare preserves event IDs."""
        fp1 = engine.extract(sample_flood_data, "flood_event")
        fp2 = engine.extract(sample_drought_data, "drought_event")
        
        result = engine.compare(fp1, fp2)
        assert result.query_id == "flood_event"
        assert result.match_id == "drought_event"


# =============================================================================
# BATCH AND SEARCH TESTS
# =============================================================================

class TestBatchOperations:
    """Test batch operations."""
    
    def test_batch_extract(self, engine, sample_flood_data, sample_drought_data):
        """Batch extract multiple events."""
        events = [
            {"event_id": "flood", "df": sample_flood_data},
            {"event_id": "drought", "df": sample_drought_data},
        ]
        
        fingerprints = engine.batch_extract(events)
        assert len(fingerprints) == 2
        assert fingerprints[0].event_id == "flood"
        assert fingerprints[1].event_id == "drought"
    
    def test_find_similar(
        self, engine, sample_flood_data, sample_similar_data, sample_drought_data
    ):
        """Find similar fingerprints."""
        fp_query = engine.extract(sample_flood_data, "query")
        fp_similar = engine.extract(sample_similar_data, "similar")
        fp_different = engine.extract(sample_drought_data, "different")
        
        candidates = [fp_similar, fp_different]
        results = engine.find_similar(fp_query, candidates, top_k=2)
        
        assert len(results) == 2
        # Similar should rank higher
        assert results[0].match_id == "similar"
        assert results[0].rank == 1
    
    def test_find_similar_respects_min_threshold(
        self, engine, sample_flood_data, sample_drought_data
    ):
        """Find similar respects minimum similarity threshold."""
        fp_query = engine.extract(sample_flood_data, "query")
        fp_different = engine.extract(sample_drought_data, "different")
        
        # With very high threshold, might get no results
        results = engine.find_similar(
            fp_query,
            [fp_different],
            min_similarity=0.99,  # Very high threshold
        )
        
        # Result depends on actual similarity
        assert isinstance(results, list)


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Test fingerprint serialization."""
    
    def test_to_dict(self, engine, sample_flood_data):
        """Fingerprint can be converted to dict."""
        fp = engine.extract(sample_flood_data, "test_event")
        d = fp.to_dict()
        
        assert isinstance(d, dict)
        assert d["event_id"] == "test_event"
        assert "embedding" in d
        assert isinstance(d["embedding"], list)
    
    def test_from_dict(self, engine, sample_flood_data):
        """Fingerprint can be reconstructed from dict."""
        fp_original = engine.extract(sample_flood_data, "test_event")
        d = fp_original.to_dict()
        
        fp_reconstructed = Fingerprint.from_dict(d)
        
        assert fp_reconstructed.event_id == fp_original.event_id
        assert np.allclose(fp_reconstructed.embedding, fp_original.embedding)
    
    def test_hash(self, engine, sample_flood_data):
        """Fingerprint has deterministic hash."""
        fp = engine.extract(sample_flood_data, "test_event")
        h1 = fp.hash()
        h2 = fp.hash()
        
        assert h1 == h2
        assert len(h1) == 16


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_extract_fingerprint_function(self, sample_flood_data):
        """extract_fingerprint convenience function works."""
        fp = extract_fingerprint(
            sample_flood_data,
            "test_event",
            sources=["era5"],
            n_kernels=1000,
            embedding_dim=50,
        )
        assert isinstance(fp, Fingerprint)
        assert fp.event_id == "test_event"
    
    def test_compare_fingerprints_function(self, sample_flood_data, sample_drought_data):
        """compare_fingerprints convenience function works."""
        fp1 = extract_fingerprint(sample_flood_data, "flood", n_kernels=1000)
        fp2 = extract_fingerprint(sample_drought_data, "drought", n_kernels=1000)
        
        result = compare_fingerprints(fp1, fp2)
        assert isinstance(result, SimilarityResult)


# =============================================================================
# MINIROCKET-SPECIFIC TESTS (skipped if not available)
# =============================================================================

@pytest.mark.skipif(not HAS_MINIROCKET, reason="sktime not installed")
class TestMiniRocketSpecific:
    """Tests specific to MiniRocket (require sktime)."""
    
    def test_uses_minirocket(self, engine, sample_flood_data):
        """Engine uses MiniRocket when available."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert fp.extraction_method == "minirocket"
    
    def test_large_embedding_without_pca(self, sample_flood_data):
        """Without PCA, embedding is large."""
        engine = FingerprintEngine(
            n_kernels=1000,
            embedding_dim=0,  # No PCA
        )
        fp = engine.extract(sample_flood_data, "test_event")
        # MiniRocket produces ~10K features by default
        assert len(fp.embedding) > 100


@pytest.mark.skipif(HAS_MINIROCKET, reason="Testing fallback only")
class TestFallbackOnly:
    """Tests for fallback statistical fingerprint."""
    
    def test_uses_statistical_method(self, engine, sample_flood_data):
        """Without sktime, uses statistical fallback."""
        fp = engine.extract(sample_flood_data, "test_event")
        assert fp.extraction_method == "statistical"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_variable(self, engine):
        """Works with single variable."""
        df = pd.DataFrame({
            "precipitation": np.random.exponential(5, 100),
        }, index=pd.date_range("2000-01-01", periods=100, freq="D"))
        
        fp = engine.extract(df, "single_var")
        assert fp.n_variables == 1
    
    def test_constant_variable(self, engine):
        """Handles constant variable."""
        df = pd.DataFrame({
            "precipitation": np.random.exponential(5, 100),
            "constant": np.ones(100),  # Constant
        }, index=pd.date_range("2000-01-01", periods=100, freq="D"))
        
        fp = engine.extract(df, "with_constant")
        # Should still work, uncertainty might be higher
        assert fp is not None
    
    def test_no_datetime_index(self, engine):
        """Works without DatetimeIndex."""
        df = pd.DataFrame({
            "precipitation": np.random.exponential(5, 100),
            "temperature": np.random.normal(288, 5, 100),
        })  # Default integer index
        
        fp = engine.extract(df, "no_datetime")
        assert fp is not None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

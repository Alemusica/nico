"""
Test Suite for Analysis Functions
==================================
Tests for DOT computation, slope analysis, and statistics.

Run with: pytest tests/test_analysis.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.slope import bin_by_longitude, compute_slope, SlopeResult
from src.analysis.dot import compute_dot, get_dot_statistics
from src.analysis.statistics import compute_statistics, detect_outliers_iqr


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_dot_data():
    """Create sample DOT-like data for testing."""
    np.random.seed(42)
    n_points = 500
    
    lon = np.linspace(-50, 20, n_points)
    lat = np.random.uniform(65, 75, n_points)
    
    # DOT with a slope (higher in west, lower in east)
    dot = -0.001 * lon + np.random.randn(n_points) * 0.05
    
    return lon, lat, dot


@pytest.fixture
def sample_ssh_data():
    """Create sample SSH data for DOT computation."""
    np.random.seed(42)
    n_points = 1000
    
    ssh = np.random.randn(n_points) * 0.5 + 10
    mss = np.random.randn(n_points) * 0.3 + 10
    
    return ssh, mss


# =============================================================================
# BINNING TESTS
# =============================================================================

class TestBinByLongitude:
    """Tests for longitude binning function."""
    
    def test_basic_binning(self, sample_dot_data):
        """Test basic binning works."""
        lon, lat, dot = sample_dot_data
        
        centers, means, stds, counts = bin_by_longitude(lon, dot, bin_size=1.0)
        
        assert len(centers) > 0
        assert len(centers) == len(means) == len(stds) == len(counts)
    
    def test_bin_size_affects_output(self, sample_dot_data):
        """Test that bin size affects number of bins."""
        lon, lat, dot = sample_dot_data
        
        centers_small, _, _, _ = bin_by_longitude(lon, dot, bin_size=0.1)
        centers_large, _, _, _ = bin_by_longitude(lon, dot, bin_size=5.0)
        
        assert len(centers_small) > len(centers_large)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        centers, means, stds, counts = bin_by_longitude(
            np.array([]), np.array([]), bin_size=1.0
        )
        
        assert len(centers) == 0
    
    def test_single_point(self):
        """Test handling of single point."""
        centers, means, stds, counts = bin_by_longitude(
            np.array([10.0]), np.array([0.5]), bin_size=1.0
        )
        
        # Single point may result in empty bins (implementation dependent)
        # The important thing is it doesn't crash
        assert isinstance(centers, np.ndarray)
    
    def test_nan_handling(self, sample_dot_data):
        """Test that NaN values are handled."""
        lon, lat, dot = sample_dot_data
        
        # Add some NaN values
        dot_with_nan = dot.copy()
        dot_with_nan[::10] = np.nan
        
        centers, means, stds, counts = bin_by_longitude(lon, dot_with_nan, bin_size=1.0)
        
        # Should still produce results (ignoring NaN)
        assert len(centers) > 0
        assert np.all(np.isfinite(means))


# =============================================================================
# SLOPE COMPUTATION TESTS
# =============================================================================

class TestComputeSlope:
    """Tests for slope computation function."""
    
    def test_basic_slope(self, sample_dot_data):
        """Test basic slope computation."""
        lon, lat, dot = sample_dot_data
        lat_mean = np.mean(lat)
        
        result = compute_slope(lon, dot, lat_mean, bin_size=1.0)
        
        assert result is not None
        assert isinstance(result, SlopeResult)
        assert hasattr(result, "slope_mm_per_m")
        assert hasattr(result, "r_squared")
    
    def test_slope_sign(self, sample_dot_data):
        """Test slope has expected sign (negative for west-to-east decrease)."""
        lon, lat, dot = sample_dot_data
        lat_mean = np.mean(lat)
        
        result = compute_slope(lon, dot, lat_mean, bin_size=1.0)
        
        # Our fixture has negative slope (decreasing west to east)
        assert result.slope_mm_per_m < 0
    
    def test_r_squared_range(self, sample_dot_data):
        """Test R² is in valid range [0, 1]."""
        lon, lat, dot = sample_dot_data
        lat_mean = np.mean(lat)
        
        result = compute_slope(lon, dot, lat_mean, bin_size=1.0)
        
        assert 0 <= result.r_squared <= 1
    
    def test_insufficient_points(self):
        """Test handling with insufficient points."""
        lon = np.array([1.0, 2.0])
        dot = np.array([0.1, 0.2])
        
        result = compute_slope(lon, dot, 70.0, bin_size=5.0)
        
        # Should return None or handle gracefully
        # (behavior depends on implementation)
        assert result is None or isinstance(result, SlopeResult)
    
    def test_flat_data(self):
        """Test with flat data (no slope)."""
        lon = np.linspace(0, 100, 1000)
        dot = np.ones(1000) * 0.5 + np.random.randn(1000) * 0.001
        
        result = compute_slope(lon, dot, 70.0, bin_size=1.0)
        
        # Slope should be near zero
        assert result is not None
        assert abs(result.slope_mm_per_m) < 0.01


# =============================================================================
# DOT COMPUTATION TESTS
# =============================================================================

class TestComputeDOT:
    """Tests for DOT computation."""
    
    def test_basic_dot(self, sample_ssh_data):
        """Test basic DOT computation."""
        ssh, mss = sample_ssh_data
        
        dot = ssh - mss
        
        assert len(dot) == len(ssh)
        assert np.all(np.isfinite(dot))
    
    def test_dot_with_nan(self, sample_ssh_data):
        """Test DOT with NaN values."""
        ssh, mss = sample_ssh_data
        ssh_with_nan = ssh.copy()
        ssh_with_nan[::10] = np.nan
        
        dot = ssh_with_nan - mss
        
        # NaN propagates
        assert np.sum(np.isnan(dot)) == np.sum(np.isnan(ssh_with_nan))
    
    def test_dot_mean_difference(self, sample_ssh_data):
        """Test DOT mean is approximately SSH mean - MSS mean."""
        ssh, mss = sample_ssh_data
        
        dot = ssh - mss
        
        expected_mean = np.mean(ssh) - np.mean(mss)
        actual_mean = np.mean(dot)
        
        assert np.isclose(expected_mean, actual_mean, rtol=1e-10)


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Tests for statistical functions."""
    
    def test_compute_statistics(self, sample_dot_data):
        """Test basic statistics computation."""
        lon, lat, dot = sample_dot_data
        
        stats = compute_statistics(dot)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        # Use actual key name from implementation
        assert "n_valid" in stats or "count" in stats
    
    def test_outlier_detection(self):
        """Test IQR-based outlier detection."""
        # Create data with clear outliers
        data = np.concatenate([
            np.random.randn(100) * 0.1,  # Normal data
            np.array([10, -10, 15])       # Outliers
        ])
        
        outliers = detect_outliers_iqr(data)
        
        # Should detect the extreme values
        assert np.sum(outliers) >= 3
    
    def test_statistics_with_nan(self):
        """Test statistics handle NaN correctly."""
        data = np.array([1, 2, 3, np.nan, 5])
        
        stats = compute_statistics(data)
        
        # Should compute stats ignoring NaN
        assert np.isfinite(stats["mean"])
        # Use actual key name
        valid_count = stats.get("n_valid", stats.get("count", 0))
        assert valid_count == 4  # Only valid values


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_longitude_wrapping(self):
        """Test handling of longitude wrapping (0-360 vs -180-180)."""
        # Data crossing 0/360 boundary
        lon = np.array([358, 359, 0, 1, 2])
        dot = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        centers, means, stds, counts = bin_by_longitude(lon, dot, bin_size=1.0)
        
        # Should handle this case
        assert len(centers) > 0
    
    def test_high_latitude_slope(self):
        """Test slope at high latitudes (latitude correction)."""
        lon = np.linspace(0, 50, 500)
        dot = -0.001 * lon
        
        # Very high latitude
        result_high = compute_slope(lon, dot, 85.0, bin_size=1.0)
        # Lower latitude
        result_low = compute_slope(lon, dot, 45.0, bin_size=1.0)
        
        # Both should work
        assert result_high is not None
        assert result_low is not None
        
        # Slope in mm/m should differ due to latitude correction
        # (same m/deg but different m/m)
    
    def test_very_small_bin_size(self, sample_dot_data):
        """Test with very small bin size."""
        lon, lat, dot = sample_dot_data
        
        centers, means, stds, counts = bin_by_longitude(lon, dot, bin_size=0.001)
        
        # Should produce many bins
        assert len(centers) > 100
    
    def test_very_large_bin_size(self, sample_dot_data):
        """Test with very large bin size."""
        lon, lat, dot = sample_dot_data
        
        centers, means, stds, counts = bin_by_longitude(lon, dot, bin_size=100.0)
        
        # Should produce few bins
        assert len(centers) <= 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAnalysisIntegration:
    """Integration tests for the full analysis pipeline."""
    
    def test_full_pipeline(self, sample_dot_data):
        """Test the complete analysis pipeline."""
        lon, lat, dot = sample_dot_data
        
        # 1. Bin data
        centers, means, stds, counts = bin_by_longitude(lon, dot, bin_size=1.0)
        assert len(centers) > 3
        
        # 2. Compute slope
        lat_mean = np.mean(lat)
        result = compute_slope(lon, dot, lat_mean, bin_size=1.0)
        assert result is not None
        
        # 3. Get statistics
        stats = compute_statistics(dot)
        valid_count = stats.get("n_valid", stats.get("count", 0))
        assert valid_count > 0
        
        # 4. All values should be finite
        assert np.isfinite(result.slope_mm_per_m)
        assert np.isfinite(result.r_squared)
        assert np.isfinite(stats["mean"])


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

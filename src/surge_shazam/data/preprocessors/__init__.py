"""
🔄 Data Preprocessors Module
============================

Unified data preprocessing pipeline combining interpolation and normalization.

This module provides the DataHarmonizer class that orchestrates interpolation
and normalization for complete data preprocessing workflows.

Components:
- TimeSeriesInterpolator: Fill gaps in time series data
- DataNormalizer: Scale and standardize data
- DataHarmonizer: Combined preprocessing pipeline

Requirements:
    pip install pandas numpy scipy scikit-learn

Usage:
    from src.surge_shazam.data.preprocessors import DataHarmonizer
    
    harmonizer = DataHarmonizer(
        interpolation_method="cubic",
        normalization_method="zscore"
    )
    
    df_processed = harmonizer.fit_transform(df)
    
    # Get quality metrics
    metrics = harmonizer.get_metrics()
    print(f"Quality: {metrics['quality_score']:.2f}")
"""

import pandas as pd
from typing import Optional, Dict, Any, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .interpolation import (
    TimeSeriesInterpolator,
    InterpolationMethod,
    InterpolationResult,
    interpolate_linear,
    interpolate_spline,
)

from .normalization import (
    DataNormalizer,
    NormalizationMethod,
    NormalizationResult,
    zscore_normalize,
    minmax_normalize,
    robust_normalize,
)

logger = logging.getLogger(__name__)


__all__ = [
    # Main classes
    "DataHarmonizer",
    "TimeSeriesInterpolator",
    "DataNormalizer",
    # Enums
    "InterpolationMethod",
    "NormalizationMethod",
    # Results
    "InterpolationResult",
    "NormalizationResult",
    "HarmonizationResult",
    # Convenience functions
    "interpolate_linear",
    "interpolate_spline",
    "zscore_normalize",
    "minmax_normalize",
    "robust_normalize",
]


@dataclass
class HarmonizationResult:
    """Result of complete harmonization pipeline."""
    data: pd.DataFrame
    interpolation_result: Optional[InterpolationResult]
    normalization_result: Optional[NormalizationResult]
    pipeline_steps: list
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pipeline_steps": self.pipeline_steps,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "interpolation": self.interpolation_result.to_dict() if self.interpolation_result else None,
            "normalization": self.normalization_result.to_dict() if self.normalization_result else None,
            "metadata": self.metadata,
        }


class DataHarmonizer:
    """
    Unified data harmonization pipeline.
    
    Combines interpolation and normalization into a single preprocessing workflow.
    Tracks quality metrics and provides reversible transformations.
    
    Pipeline order:
    1. Interpolation (fill gaps)
    2. Normalization (scale values)
    """
    
    def __init__(
        self,
        interpolation_method: str = InterpolationMethod.LINEAR,
        normalization_method: str = NormalizationMethod.ZSCORE,
        max_gap_size: Optional[int] = None,
        feature_range: Tuple[float, float] = (0, 1),
        clip_outliers: bool = False,
        outlier_std: float = 3.0,
        skip_interpolation: bool = False,
        skip_normalization: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize data harmonizer.
        
        Args:
            interpolation_method: Method for filling gaps
            normalization_method: Method for scaling data
            max_gap_size: Maximum gap size to interpolate
            feature_range: Range for min-max scaling
            clip_outliers: Clip outliers before processing
            outlier_std: Standard deviations for outlier clipping
            skip_interpolation: Skip interpolation step
            skip_normalization: Skip normalization step
            verbose: Print debug info
        """
        self.interpolation_method = interpolation_method
        self.normalization_method = normalization_method
        self.max_gap_size = max_gap_size
        self.feature_range = feature_range
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.skip_interpolation = skip_interpolation
        self.skip_normalization = skip_normalization
        self.verbose = verbose
        
        # Initialize components
        self._interpolator: Optional[TimeSeriesInterpolator] = None
        self._normalizer: Optional[DataNormalizer] = None
        self._is_fitted: bool = False
        self._last_result: Optional[HarmonizationResult] = None
        
        if not skip_interpolation:
            self._interpolator = TimeSeriesInterpolator(
                method=interpolation_method,
                limit=max_gap_size,
                verbose=verbose,
            )
        
        if not skip_normalization:
            self._normalizer = DataNormalizer(
                method=normalization_method,
                feature_range=feature_range,
                clip_outliers=clip_outliers,
                outlier_std=outlier_std,
                verbose=verbose,
            )
    
    def fit(self, data: pd.DataFrame) -> "DataHarmonizer":
        """
        Fit the harmonizer to data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for chaining
        """
        if self.verbose:
            logger.info("🔄 Fitting data harmonizer...")
        
        # Step 1: Interpolate
        processed = data.copy()
        if not self.skip_interpolation and self._interpolator is not None:
            processed = self._interpolator.fill_gaps(processed, self.max_gap_size)
        
        # Step 2: Fit normalizer on interpolated data
        if not self.skip_normalization and self._normalizer is not None:
            self._normalizer.fit(processed)
        
        self._is_fitted = True
        
        if self.verbose:
            logger.info("✅ Data harmonizer fitted")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Harmonized DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Harmonizer not fitted. Call fit() first.")
        
        if self.verbose:
            logger.info("🔄 Transforming data through harmonization pipeline...")
        
        pipeline_steps = []
        processed = data.copy()
        interpolation_result = None
        normalization_result = None
        
        # Step 1: Interpolate
        if not self.skip_interpolation and self._interpolator is not None:
            processed = self._interpolator.fill_gaps(processed, self.max_gap_size)
            interpolation_result = self._interpolator.get_result()
            pipeline_steps.append("interpolation")
        
        # Step 2: Normalize
        if not self.skip_normalization and self._normalizer is not None:
            processed = self._normalizer.transform(processed)
            normalization_result = self._normalizer.get_result()
            pipeline_steps.append("normalization")
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            interpolation_result,
            normalization_result,
        )
        
        # Store result
        self._last_result = HarmonizationResult(
            data=processed,
            interpolation_result=interpolation_result,
            normalization_result=normalization_result,
            pipeline_steps=pipeline_steps,
            quality_score=quality_score,
            metadata={
                "n_samples": len(data),
                "n_features": len(data.columns),
                "interpolation_method": self.interpolation_method,
                "normalization_method": self.normalization_method,
            }
        )
        
        if self.verbose:
            logger.info(f"✅ Harmonization complete (quality: {quality_score:.2f})")
        
        return processed
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to data and transform in one step.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Harmonized DataFrame
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the harmonization transformations.
        
        Note: Only normalization can be reversed. Interpolation is irreversible.
        
        Args:
            data: Harmonized DataFrame
            
        Returns:
            DataFrame with reversed normalization
        """
        if not self._is_fitted:
            raise RuntimeError("Harmonizer not fitted. Call fit() first.")
        
        if self.verbose:
            logger.info("🔄 Reversing harmonization (normalization only)...")
        
        result = data.copy()
        
        # Reverse normalization
        if not self.skip_normalization and self._normalizer is not None:
            result = self._normalizer.inverse_transform(result)
        
        # Note: Interpolation cannot be reversed
        if not self.skip_interpolation:
            logger.warning(
                "⚠️ Interpolation cannot be reversed. "
                "Returning denormalized data with interpolated values."
            )
        
        return result
    
    def _calculate_quality_score(
        self,
        interp_result: Optional[InterpolationResult],
        norm_result: Optional[NormalizationResult],
    ) -> float:
        """
        Calculate overall pipeline quality score.
        
        Combines quality metrics from interpolation and normalization.
        
        Returns score in [0, 1] where 1 is perfect.
        """
        scores = []
        
        # Interpolation quality
        if interp_result is not None:
            scores.append(interp_result.quality_score)
        
        # Normalization is always high quality (reversible)
        if norm_result is not None:
            scores.append(0.95)
        
        # If no steps, return perfect score
        if not scores:
            return 1.0
        
        # Average of component scores
        return sum(scores) / len(scores)
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get metrics from last harmonization.
        
        Returns:
            Dictionary with quality metrics or None if no harmonization done yet
        """
        if self._last_result is None:
            return None
        
        return self._last_result.to_dict()
    
    def get_result(self) -> Optional[HarmonizationResult]:
        """Get the last harmonization result."""
        return self._last_result
    
    def get_interpolator(self) -> Optional[TimeSeriesInterpolator]:
        """Get the interpolator component."""
        return self._interpolator
    
    def get_normalizer(self) -> Optional[DataNormalizer]:
        """Get the normalizer component."""
        return self._normalizer


# CLI test
if __name__ == "__main__":
    import numpy as np
    
    print("=== Data Harmonizer Test ===\n")
    
    # Generate test data with gaps and different scales
    np.random.seed(42)
    n = 200
    
    # Create time series with gaps and different scales
    x = np.linspace(0, 10, n)
    y1 = np.sin(x) * 50 + 100 + np.random.randn(n) * 5
    y2 = np.cos(x) * 20 + 50 + np.random.randn(n) * 2
    y3 = np.random.exponential(10, n)
    
    # Introduce gaps
    gap_indices = np.random.choice(n, size=30, replace=False)
    y1[gap_indices[:10]] = np.nan
    y2[gap_indices[10:20]] = np.nan
    y3[gap_indices[20:]] = np.nan
    
    df = pd.DataFrame({
        "feature1": y1,
        "feature2": y2,
        "feature3": y3,
    })
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isna().sum())
    print(f"\nOriginal statistics:")
    print(df.describe())
    
    # Test full harmonization pipeline
    print("\n--- Full Harmonization Pipeline ---")
    harmonizer = DataHarmonizer(
        interpolation_method="cubic",
        normalization_method="zscore",
        verbose=True,
    )
    
    df_harmonized = harmonizer.fit_transform(df)
    
    print(f"\nHarmonized statistics:")
    print(df_harmonized.describe())
    
    # Get metrics
    metrics = harmonizer.get_metrics()
    print(f"\n📊 Harmonization Metrics:")
    print(f"  Pipeline steps: {metrics['pipeline_steps']}")
    print(f"  Quality score: {metrics['quality_score']:.2f}")
    if metrics['interpolation']:
        print(f"  Gaps filled: {metrics['interpolation']['gaps_filled']}")
    
    # Test inverse transform
    print("\n--- Inverse Transform ---")
    df_restored = harmonizer.inverse_transform(df_harmonized)
    print(f"\nRestored statistics:")
    print(df_restored.describe())
    
    # Test with different configurations
    print("\n--- Custom Configuration ---")
    harmonizer2 = DataHarmonizer(
        interpolation_method="linear",
        normalization_method="minmax",
        feature_range=(-1, 1),
        max_gap_size=5,
        clip_outliers=True,
        verbose=True,
    )
    
    df_custom = harmonizer2.fit_transform(df)
    metrics2 = harmonizer2.get_metrics()
    print(f"\nQuality score: {metrics2['quality_score']:.2f}")
    print(f"Value range: [{df_custom.min().min():.2f}, {df_custom.max().max():.2f}]")
    
    print("\n✅ Test completed!")

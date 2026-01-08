"""
🔧 Time Series Interpolation Module
===================================

Advanced interpolation methods for filling gaps in time series data.

Features:
- Linear interpolation
- Spline interpolation (cubic, quadratic)
- Intelligent gap filling with configurable strategies
- Support for pandas DataFrame and xarray Dataset
- Quality metrics and validation

Requirements:
    pip install pandas scipy xarray

Usage:
    from src.surge_shazam.data.preprocessors.interpolation import TimeSeriesInterpolator
    
    interpolator = TimeSeriesInterpolator(method="cubic")
    df_filled = interpolator.fill_gaps(df, max_gap_size=10)
    
    # Check quality
    metrics = interpolator.get_quality_metrics()
    print(f"Filled {metrics['gaps_filled']} gaps, quality: {metrics['quality_score']:.2f}")
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Union, Optional, Literal, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Check for xarray
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    logger.warning("⚠️ xarray not installed. Install with: pip install xarray")

# Check for scipy
try:
    from scipy.interpolate import interp1d, UnivariateSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("⚠️ scipy not installed. Install with: pip install scipy")


class InterpolationMethod:
    """Enumeration of available interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    QUADRATIC = "quadratic"
    NEAREST = "nearest"
    SPLINE = "spline"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"


@dataclass
class InterpolationResult:
    """Result of interpolation operation."""
    data: Union[pd.DataFrame, "xr.Dataset"]
    method_used: str
    gaps_filled: int
    total_points: int
    fill_ratio: float
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "method_used": self.method_used,
            "gaps_filled": self.gaps_filled,
            "total_points": self.total_points,
            "fill_ratio": self.fill_ratio,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class TimeSeriesInterpolator:
    """
    Advanced time series interpolator with multiple methods.
    
    Handles missing data in time series using various interpolation strategies.
    Tracks quality metrics and provides validation.
    """
    
    def __init__(
        self,
        method: str = InterpolationMethod.LINEAR,
        limit: Optional[int] = None,
        limit_direction: Literal["forward", "backward", "both"] = "both",
        limit_area: Optional[Literal["inside", "outside"]] = None,
        verbose: bool = False,
    ):
        """
        Initialize time series interpolator.
        
        Args:
            method: Interpolation method to use
            limit: Maximum number of consecutive NaNs to fill
            limit_direction: Direction to fill gaps
            limit_area: Where to fill (inside valid values, outside, or None for all)
            verbose: Print debug info
        """
        self.method = method
        self.limit = limit
        self.limit_direction = limit_direction
        self.limit_area = limit_area
        self.verbose = verbose
        self._last_result: Optional[InterpolationResult] = None
        
        # Validate method
        valid_methods = [
            InterpolationMethod.LINEAR,
            InterpolationMethod.CUBIC,
            InterpolationMethod.QUADRATIC,
            InterpolationMethod.NEAREST,
            InterpolationMethod.SPLINE,
            InterpolationMethod.FORWARD_FILL,
            InterpolationMethod.BACKWARD_FILL,
        ]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from: {valid_methods}")
        
        if method in [InterpolationMethod.CUBIC, InterpolationMethod.SPLINE] and not HAS_SCIPY:
            raise ImportError(
                f"Method '{method}' requires scipy. "
                "Install with: pip install scipy"
            )
    
    def interpolate_linear(
        self,
        data: Union[pd.DataFrame, pd.Series],
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Perform linear interpolation.
        
        Args:
            data: Input data with missing values
            **kwargs: Additional arguments for pandas interpolate()
            
        Returns:
            Interpolated data
        """
        if self.verbose:
            logger.info("🔧 Running linear interpolation...")
        
        return data.interpolate(
            method="linear",
            limit=self.limit,
            limit_direction=self.limit_direction,
            limit_area=self.limit_area,
            **kwargs,
        )
    
    def interpolate_spline(
        self,
        data: Union[pd.DataFrame, pd.Series],
        order: int = 3,
        smooth: float = 0.0,
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Perform spline interpolation using scipy.
        
        Args:
            data: Input data with missing values
            order: Spline order (1=linear, 2=quadratic, 3=cubic)
            smooth: Smoothing factor (0=exact interpolation)
            **kwargs: Additional arguments
            
        Returns:
            Interpolated data
        """
        if not HAS_SCIPY:
            raise ImportError("Spline interpolation requires scipy")
        
        if self.verbose:
            logger.info(f"🔧 Running spline interpolation (order={order})...")
        
        if isinstance(data, pd.Series):
            return self._interpolate_series_spline(data, order, smooth)
        elif isinstance(data, pd.DataFrame):
            result = data.copy()
            for col in data.columns:
                result[col] = self._interpolate_series_spline(data[col], order, smooth)
            return result
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _interpolate_series_spline(
        self,
        series: pd.Series,
        order: int,
        smooth: float,
    ) -> pd.Series:
        """Interpolate a single series using spline."""
        # Get valid (non-NaN) points
        valid_mask = ~series.isna()
        valid_indices = np.where(valid_mask)[0]
        valid_values = series.values[valid_mask]
        
        if len(valid_indices) < order + 1:
            logger.warning(
                f"Not enough valid points ({len(valid_indices)}) for order {order} spline. "
                "Falling back to linear."
            )
            return series.interpolate(method="linear")
        
        # Create spline
        try:
            spline = UnivariateSpline(
                valid_indices,
                valid_values,
                k=min(order, len(valid_indices) - 1),
                s=smooth,
            )
            
            # Interpolate missing values
            result = series.copy()
            missing_mask = series.isna()
            missing_indices = np.where(missing_mask)[0]
            
            if len(missing_indices) > 0:
                # Only interpolate within the range of valid data
                min_valid_idx = valid_indices.min()
                max_valid_idx = valid_indices.max()
                
                for idx in missing_indices:
                    if min_valid_idx <= idx <= max_valid_idx:
                        result.iloc[idx] = spline(idx)
            
            return result
            
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}. Falling back to linear.")
            return series.interpolate(method="linear")
    
    def fill_gaps(
        self,
        data: Union[pd.DataFrame, pd.Series, "xr.Dataset"],
        max_gap_size: Optional[int] = None,
        quality_check: bool = True,
    ) -> Union[pd.DataFrame, pd.Series, "xr.Dataset"]:
        """
        Fill gaps in time series data using configured method.
        
        Args:
            data: Input data with missing values
            max_gap_size: Maximum gap size to fill (overrides self.limit)
            quality_check: Compute quality metrics
            
        Returns:
            Data with filled gaps
        """
        if self.verbose:
            logger.info(f"🔧 Filling gaps using method: {self.method}")
        
        # Handle xarray Dataset
        if HAS_XARRAY and isinstance(data, xr.Dataset):
            return self._fill_gaps_xarray(data, max_gap_size, quality_check)
        
        # Count gaps before
        gaps_before = self._count_missing(data)
        total_points = self._count_total(data)
        
        # Apply interpolation based on method
        limit = max_gap_size if max_gap_size is not None else self.limit
        
        if self.method == InterpolationMethod.LINEAR:
            filled = data.interpolate(
                method="linear",
                limit=limit,
                limit_direction=self.limit_direction,
                limit_area=self.limit_area,
            )
        
        elif self.method in [InterpolationMethod.CUBIC, InterpolationMethod.QUADRATIC]:
            filled = data.interpolate(
                method=self.method,
                limit=limit,
                limit_direction=self.limit_direction,
                limit_area=self.limit_area,
            )
        
        elif self.method == InterpolationMethod.SPLINE:
            order = 3  # Cubic spline
            filled = self.interpolate_spline(data, order=order)
        
        elif self.method == InterpolationMethod.NEAREST:
            filled = data.interpolate(
                method="nearest",
                limit=limit,
                limit_direction=self.limit_direction,
                limit_area=self.limit_area,
            )
        
        elif self.method == InterpolationMethod.FORWARD_FILL:
            filled = data.ffill(limit=limit)
        
        elif self.method == InterpolationMethod.BACKWARD_FILL:
            filled = data.bfill(limit=limit)
        
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Count gaps after
        gaps_after = self._count_missing(filled)
        gaps_filled = gaps_before - gaps_after
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(data, filled) if quality_check else 1.0
        
        # Store result
        self._last_result = InterpolationResult(
            data=filled,
            method_used=self.method,
            gaps_filled=gaps_filled,
            total_points=total_points,
            fill_ratio=gaps_filled / gaps_before if gaps_before > 0 else 0.0,
            quality_score=quality_score,
            metadata={
                "gaps_before": gaps_before,
                "gaps_after": gaps_after,
                "max_gap_size": limit,
            }
        )
        
        if self.verbose:
            logger.info(
                f"✅ Filled {gaps_filled}/{gaps_before} gaps "
                f"(quality: {quality_score:.2f})"
            )
        
        return filled
    
    def _fill_gaps_xarray(
        self,
        data: "xr.Dataset",
        max_gap_size: Optional[int],
        quality_check: bool,
    ) -> "xr.Dataset":
        """Fill gaps in xarray Dataset."""
        filled = data.copy()
        
        # Interpolate each data variable
        for var in data.data_vars:
            if self.verbose:
                logger.info(f"  Interpolating variable: {var}")
            
            # Convert to DataFrame for interpolation
            df = data[var].to_dataframe()
            df_filled = self.fill_gaps(df, max_gap_size, quality_check=False)
            
            # Update dataset
            filled[var] = df_filled.to_xarray()[var]
        
        return filled
    
    def _count_missing(self, data: Union[pd.DataFrame, pd.Series]) -> int:
        """Count total missing values."""
        if isinstance(data, pd.Series):
            return data.isna().sum()
        elif isinstance(data, pd.DataFrame):
            return data.isna().sum().sum()
        else:
            return 0
    
    def _count_total(self, data: Union[pd.DataFrame, pd.Series]) -> int:
        """Count total data points."""
        if isinstance(data, pd.Series):
            return len(data)
        elif isinstance(data, pd.DataFrame):
            return data.size
        else:
            return 0
    
    def _calculate_quality_score(
        self,
        original: Union[pd.DataFrame, pd.Series],
        filled: Union[pd.DataFrame, pd.Series],
    ) -> float:
        """
        Calculate interpolation quality score.
        
        Quality is based on:
        - Smoothness of interpolated sections
        - Consistency with surrounding values
        - No extreme outliers introduced
        
        Returns score in [0, 1] where 1 is perfect.
        """
        # For now, simple heuristic based on variance change
        try:
            if isinstance(original, pd.Series):
                original_std = original.std()
                filled_std = filled.std()
            else:
                original_std = original.std().mean()
                filled_std = filled.std().mean()
            
            # Penalize if variance changed significantly
            if original_std == 0 or np.isnan(original_std):
                return 1.0
            
            variance_ratio = filled_std / original_std
            
            # Ideal ratio is close to 1.0
            if 0.8 <= variance_ratio <= 1.2:
                quality = 1.0
            elif 0.5 <= variance_ratio <= 1.5:
                quality = 0.8
            elif 0.3 <= variance_ratio <= 2.0:
                quality = 0.6
            else:
                quality = 0.4
            
            return quality
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.5
    
    def get_quality_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get quality metrics from last interpolation.
        
        Returns:
            Dictionary with quality metrics or None if no interpolation done yet
        """
        if self._last_result is None:
            return None
        
        return self._last_result.to_dict()
    
    def get_result(self) -> Optional[InterpolationResult]:
        """Get the last interpolation result."""
        return self._last_result


# Convenience functions
def interpolate_linear(
    data: Union[pd.DataFrame, pd.Series],
    limit: Optional[int] = None,
) -> Union[pd.DataFrame, pd.Series]:
    """Quick linear interpolation."""
    interpolator = TimeSeriesInterpolator(method=InterpolationMethod.LINEAR, limit=limit)
    return interpolator.interpolate_linear(data)


def interpolate_spline(
    data: Union[pd.DataFrame, pd.Series],
    order: int = 3,
    limit: Optional[int] = None,
) -> Union[pd.DataFrame, pd.Series]:
    """Quick spline interpolation."""
    interpolator = TimeSeriesInterpolator(method=InterpolationMethod.SPLINE, limit=limit)
    return interpolator.interpolate_spline(data, order=order)


# CLI test
if __name__ == "__main__":
    print("=== Time Series Interpolator Test ===\n")
    
    # Generate test data with gaps
    np.random.seed(42)
    n = 100
    
    # Create time series with random gaps
    x = np.linspace(0, 10, n)
    y = np.sin(x) + np.random.randn(n) * 0.1
    
    # Introduce gaps
    gap_indices = np.random.choice(n, size=20, replace=False)
    y[gap_indices] = np.nan
    
    df = pd.DataFrame({
        "time": x,
        "value": y,
    }).set_index("time")
    
    print(f"Data shape: {df.shape}")
    print(f"Missing values: {df['value'].isna().sum()}/{len(df)}")
    
    # Test linear interpolation
    print("\n--- Linear Interpolation ---")
    interpolator = TimeSeriesInterpolator(method="linear", verbose=True)
    df_linear = interpolator.fill_gaps(df)
    metrics = interpolator.get_quality_metrics()
    print(f"Gaps filled: {metrics['gaps_filled']}")
    print(f"Quality score: {metrics['quality_score']:.2f}")
    
    # Test spline interpolation
    if HAS_SCIPY:
        print("\n--- Spline Interpolation ---")
        interpolator = TimeSeriesInterpolator(method="spline", verbose=True)
        df_spline = interpolator.fill_gaps(df)
        metrics = interpolator.get_quality_metrics()
        print(f"Gaps filled: {metrics['gaps_filled']}")
        print(f"Quality score: {metrics['quality_score']:.2f}")
    
    print("\n✅ Test completed!")

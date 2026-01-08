"""
📊 Data Normalization Module
============================

Advanced normalization methods for scaling and standardizing data.

Features:
- Z-score standardization
- Min-max scaling
- Robust scaling (IQR-based)
- Support for pandas DataFrame
- Reversible transformations
- Outlier handling

Requirements:
    pip install pandas numpy scikit-learn

Usage:
    from src.surge_shazam.data.preprocessors.normalization import DataNormalizer
    
    normalizer = DataNormalizer(method="zscore")
    df_normalized = normalizer.fit_transform(df)
    
    # Reverse transformation
    df_original = normalizer.inverse_transform(df_normalized)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Union, Optional, Literal, Dict, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Check for sklearn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")


class NormalizationMethod:
    """Enumeration of available normalization methods."""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NONE = "none"


@dataclass
class NormalizationResult:
    """Result of normalization operation."""
    data: pd.DataFrame
    method_used: str
    columns_normalized: list
    parameters: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "method_used": self.method_used,
            "columns_normalized": self.columns_normalized,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class DataNormalizer:
    """
    Advanced data normalizer with multiple scaling methods.
    
    Supports z-score, min-max, and robust scaling with reversible transformations.
    Handles outliers and maintains statistics for inverse transformation.
    """
    
    def __init__(
        self,
        method: str = NormalizationMethod.ZSCORE,
        feature_range: Tuple[float, float] = (0, 1),
        with_centering: bool = True,
        with_scaling: bool = True,
        clip_outliers: bool = False,
        outlier_std: float = 3.0,
        verbose: bool = False,
    ):
        """
        Initialize data normalizer.
        
        Args:
            method: Normalization method to use
            feature_range: Range for min-max scaling
            with_centering: Whether to center data (subtract mean/median)
            with_scaling: Whether to scale data (divide by std/IQR)
            clip_outliers: Clip outliers before normalization
            outlier_std: Number of standard deviations for outlier clipping
            verbose: Print debug info
        """
        self.method = method
        self.feature_range = feature_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.verbose = verbose
        
        # Storage for fit parameters
        self._scaler: Optional[Any] = None
        self._statistics: Dict[str, Dict[str, float]] = {}
        self._columns: Optional[list] = None
        self._is_fitted: bool = False
        self._last_result: Optional[NormalizationResult] = None
        
        # Validate method
        valid_methods = [
            NormalizationMethod.ZSCORE,
            NormalizationMethod.MINMAX,
            NormalizationMethod.ROBUST,
            NormalizationMethod.MAXABS,
            NormalizationMethod.NONE,
        ]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Choose from: {valid_methods}")
        
        if method != NormalizationMethod.NONE and not HAS_SKLEARN:
            logger.warning(
                "scikit-learn not installed. Only basic normalization available. "
                "Install with: pip install scikit-learn"
            )
    
    def fit(self, data: pd.DataFrame) -> "DataNormalizer":
        """
        Fit the normalizer to data (compute statistics).
        
        Args:
            data: Input DataFrame
            
        Returns:
            Self for chaining
        """
        if self.verbose:
            logger.info(f"🔧 Fitting normalizer with method: {self.method}")
        
        self._columns = list(data.columns)
        
        # Clip outliers if requested
        if self.clip_outliers:
            data = self._clip_outliers(data)
        
        # Fit based on method
        if self.method == NormalizationMethod.ZSCORE:
            if HAS_SKLEARN:
                self._scaler = StandardScaler(
                    with_mean=self.with_centering,
                    with_std=self.with_scaling,
                )
                self._scaler.fit(data.values)
            else:
                self._fit_zscore_manual(data)
        
        elif self.method == NormalizationMethod.MINMAX:
            if HAS_SKLEARN:
                self._scaler = MinMaxScaler(feature_range=self.feature_range)
                self._scaler.fit(data.values)
            else:
                self._fit_minmax_manual(data)
        
        elif self.method == NormalizationMethod.ROBUST:
            if HAS_SKLEARN:
                self._scaler = RobustScaler(
                    with_centering=self.with_centering,
                    with_scaling=self.with_scaling,
                )
                self._scaler.fit(data.values)
            else:
                self._fit_robust_manual(data)
        
        elif self.method == NormalizationMethod.MAXABS:
            if HAS_SKLEARN:
                from sklearn.preprocessing import MaxAbsScaler
                self._scaler = MaxAbsScaler()
                self._scaler.fit(data.values)
            else:
                self._fit_maxabs_manual(data)
        
        self._is_fitted = True
        
        # Populate _statistics from sklearn scaler if used
        if self._scaler is not None and HAS_SKLEARN:
            self._populate_statistics_from_sklearn(data)
        
        if self.verbose:
            logger.info(f"✅ Normalizer fitted on {len(self._columns)} columns")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if self.verbose:
            logger.info(f"🔧 Transforming data with method: {self.method}")
        
        # Handle different columns
        if list(data.columns) != self._columns:
            logger.warning("Input columns differ from fitted columns")
            # Try to match columns
            common_cols = [col for col in self._columns if col in data.columns]
            if not common_cols:
                raise ValueError("No matching columns found")
            data = data[common_cols]
        
        # Clip outliers if requested
        if self.clip_outliers:
            data = self._clip_outliers(data)
        
        # Transform based on method
        if self.method == NormalizationMethod.NONE:
            result = data.copy()
        
        elif self._scaler is not None and HAS_SKLEARN:
            transformed = self._scaler.transform(data.values)
            result = pd.DataFrame(transformed, columns=data.columns, index=data.index)
        
        else:
            # Manual transformation
            result = self._transform_manual(data)
        
        # Store result
        self._last_result = NormalizationResult(
            data=result,
            method_used=self.method,
            columns_normalized=list(data.columns),
            parameters=self._get_parameters(),
            metadata={
                "n_samples": len(data),
                "n_features": len(data.columns),
            }
        )
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to data and transform in one step.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the normalization transformation.
        
        Args:
            data: Normalized DataFrame
            
        Returns:
            Original scale DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if self.verbose:
            logger.info("🔧 Reversing normalization...")
        
        if self.method == NormalizationMethod.NONE:
            return data.copy()
        
        if self._scaler is not None and HAS_SKLEARN:
            original = self._scaler.inverse_transform(data.values)
            result = pd.DataFrame(original, columns=data.columns, index=data.index)
        else:
            result = self._inverse_transform_manual(data)
        
        return result
    
    def zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score normalization (mean=0, std=1).
        
        Args:
            data: Input DataFrame
            
        Returns:
            Z-score normalized DataFrame
        """
        if self.verbose:
            logger.info("🔧 Applying z-score normalization...")
        
        result = data.copy()
        
        for col in data.columns:
            mean = data[col].mean()
            std = data[col].std()
            
            if std > 0:
                result[col] = (data[col] - mean) / std
            else:
                result[col] = data[col] - mean
        
        return result
    
    def minmax(
        self,
        data: pd.DataFrame,
        feature_range: Tuple[float, float] = (0, 1),
    ) -> pd.DataFrame:
        """
        Apply min-max scaling to specified range.
        
        Args:
            data: Input DataFrame
            feature_range: Target range (min, max)
            
        Returns:
            Min-max scaled DataFrame
        """
        if self.verbose:
            logger.info(f"🔧 Applying min-max scaling to {feature_range}...")
        
        result = data.copy()
        min_val, max_val = feature_range
        
        for col in data.columns:
            col_min = data[col].min()
            col_max = data[col].max()
            
            if col_max > col_min:
                # Scale to [0, 1] then to desired range
                normalized = (data[col] - col_min) / (col_max - col_min)
                result[col] = normalized * (max_val - min_val) + min_val
            else:
                result[col] = min_val
        
        return result
    
    def robust_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust scaling using median and IQR.
        
        More resistant to outliers than z-score.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Robust scaled DataFrame
        """
        if self.verbose:
            logger.info("🔧 Applying robust scaling...")
        
        result = data.copy()
        
        for col in data.columns:
            median = data[col].median()
            q25 = data[col].quantile(0.25)
            q75 = data[col].quantile(0.75)
            iqr = q75 - q25
            
            if iqr > 0:
                result[col] = (data[col] - median) / iqr
            else:
                result[col] = data[col] - median
        
        return result
    
    def _fit_zscore_manual(self, data: pd.DataFrame) -> None:
        """Fit z-score manually without sklearn."""
        for col in data.columns:
            self._statistics[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
            }
    
    def _fit_minmax_manual(self, data: pd.DataFrame) -> None:
        """Fit min-max manually without sklearn."""
        for col in data.columns:
            self._statistics[col] = {
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "range": float(data[col].max() - data[col].min()),
            }
    
    def _fit_robust_manual(self, data: pd.DataFrame) -> None:
        """Fit robust scaler manually without sklearn."""
        for col in data.columns:
            self._statistics[col] = {
                "median": float(data[col].median()),
                "q25": float(data[col].quantile(0.25)),
                "q75": float(data[col].quantile(0.75)),
                "iqr": float(data[col].quantile(0.75) - data[col].quantile(0.25)),
            }
    
    def _fit_maxabs_manual(self, data: pd.DataFrame) -> None:
        """Fit max-abs scaler manually without sklearn."""
        for col in data.columns:
            self._statistics[col] = {
                "max_abs": float(data[col].abs().max()),
            }
    
    def _populate_statistics_from_sklearn(self, data: pd.DataFrame) -> None:
        """Extract statistics from sklearn scaler and populate _statistics."""
        for i, col in enumerate(data.columns):
            if self.method == NormalizationMethod.ZSCORE:
                self._statistics[col] = {
                    "mean": float(self._scaler.mean_[i]),
                    "std": float(self._scaler.scale_[i]) if self._scaler.scale_ is not None else 1.0,
                }
            elif self.method == NormalizationMethod.MINMAX:
                self._statistics[col] = {
                    "min": float(self._scaler.data_min_[i]),
                    "max": float(self._scaler.data_max_[i]),
                    "range": float(self._scaler.data_range_[i]),
                }
            elif self.method == NormalizationMethod.ROBUST:
                self._statistics[col] = {
                    "median": float(self._scaler.center_[i]) if self._scaler.center_ is not None else 0.0,
                    "iqr": float(self._scaler.scale_[i]) if self._scaler.scale_ is not None else 1.0,
                }
            elif self.method == NormalizationMethod.MAXABS:
                self._statistics[col] = {
                    "max_abs": float(self._scaler.max_abs_[i]),
                }
    
    def _transform_manual(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform manually using stored statistics."""
        result = data.copy()
        
        if self.method == NormalizationMethod.ZSCORE:
            for col in data.columns:
                stats = self._statistics[col]
                if stats["std"] > 0:
                    result[col] = (data[col] - stats["mean"]) / stats["std"]
                else:
                    result[col] = data[col] - stats["mean"]
        
        elif self.method == NormalizationMethod.MINMAX:
            min_val, max_val = self.feature_range
            for col in data.columns:
                stats = self._statistics[col]
                if stats["range"] > 0:
                    normalized = (data[col] - stats["min"]) / stats["range"]
                    result[col] = normalized * (max_val - min_val) + min_val
                else:
                    result[col] = min_val
        
        elif self.method == NormalizationMethod.ROBUST:
            for col in data.columns:
                stats = self._statistics[col]
                if stats["iqr"] > 0:
                    result[col] = (data[col] - stats["median"]) / stats["iqr"]
                else:
                    result[col] = data[col] - stats["median"]
        
        elif self.method == NormalizationMethod.MAXABS:
            for col in data.columns:
                stats = self._statistics[col]
                if stats["max_abs"] > 0:
                    result[col] = data[col] / stats["max_abs"]
        
        return result
    
    def _inverse_transform_manual(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform manually using stored statistics."""
        result = data.copy()
        
        if self.method == NormalizationMethod.ZSCORE:
            for col in data.columns:
                stats = self._statistics[col]
                result[col] = data[col] * stats["std"] + stats["mean"]
        
        elif self.method == NormalizationMethod.MINMAX:
            min_val, max_val = self.feature_range
            for col in data.columns:
                stats = self._statistics[col]
                normalized = (data[col] - min_val) / (max_val - min_val)
                result[col] = normalized * stats["range"] + stats["min"]
        
        elif self.method == NormalizationMethod.ROBUST:
            for col in data.columns:
                stats = self._statistics[col]
                result[col] = data[col] * stats["iqr"] + stats["median"]
        
        elif self.method == NormalizationMethod.MAXABS:
            for col in data.columns:
                stats = self._statistics[col]
                result[col] = data[col] * stats["max_abs"]
        
        return result
    
    def _clip_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers based on standard deviations."""
        result = data.copy()
        
        for col in data.columns:
            mean = data[col].mean()
            std = data[col].std()
            
            lower = mean - self.outlier_std * std
            upper = mean + self.outlier_std * std
            
            result[col] = data[col].clip(lower, upper)
        
        return result
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get normalization parameters."""
        params = {
            "method": self.method,
            "statistics": self._statistics,
        }
        
        if self.method == NormalizationMethod.MINMAX:
            params["feature_range"] = self.feature_range
        
        if self.method in [NormalizationMethod.ZSCORE, NormalizationMethod.ROBUST]:
            params["with_centering"] = self.with_centering
            params["with_scaling"] = self.with_scaling
        
        return params
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get computed statistics for each column."""
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        return self._statistics.copy()
    
    def get_result(self) -> Optional[NormalizationResult]:
        """Get the last normalization result."""
        return self._last_result


# Convenience functions
def zscore_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Quick z-score normalization."""
    normalizer = DataNormalizer(method=NormalizationMethod.ZSCORE)
    return normalizer.fit_transform(data)


def minmax_normalize(
    data: pd.DataFrame,
    feature_range: Tuple[float, float] = (0, 1),
) -> pd.DataFrame:
    """Quick min-max normalization."""
    normalizer = DataNormalizer(method=NormalizationMethod.MINMAX, feature_range=feature_range)
    return normalizer.fit_transform(data)


def robust_normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Quick robust normalization."""
    normalizer = DataNormalizer(method=NormalizationMethod.ROBUST)
    return normalizer.fit_transform(data)


# CLI test
if __name__ == "__main__":
    print("=== Data Normalizer Test ===\n")
    
    # Generate test data
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        "feature1": np.random.randn(n) * 10 + 50,
        "feature2": np.random.randn(n) * 5 + 100,
        "feature3": np.random.exponential(2, n),
    })
    
    print(f"Original data shape: {df.shape}")
    print(f"\nOriginal statistics:")
    print(df.describe())
    
    # Test z-score normalization
    print("\n--- Z-Score Normalization ---")
    normalizer = DataNormalizer(method="zscore", verbose=True)
    df_zscore = normalizer.fit_transform(df)
    print(f"\nNormalized statistics:")
    print(df_zscore.describe())
    
    # Test inverse transform
    df_restored = normalizer.inverse_transform(df_zscore)
    print(f"\nRestored statistics:")
    print(df_restored.describe())
    
    # Verify restoration
    diff = np.abs(df.values - df_restored.values).max()
    print(f"\nMax difference after inverse: {diff:.2e}")
    
    # Test min-max normalization
    print("\n--- Min-Max Normalization ---")
    normalizer = DataNormalizer(method="minmax", feature_range=(0, 1), verbose=True)
    df_minmax = normalizer.fit_transform(df)
    print(f"\nNormalized statistics:")
    print(df_minmax.describe())
    
    # Test robust scaling
    print("\n--- Robust Scaling ---")
    normalizer = DataNormalizer(method="robust", verbose=True)
    df_robust = normalizer.fit_transform(df)
    print(f"\nNormalized statistics:")
    print(df_robust.describe())
    
    print("\n✅ Test completed!")

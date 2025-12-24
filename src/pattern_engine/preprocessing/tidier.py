"""
Data tidier for cleaning and transforming heterogeneous data.

Handles:
- Missing values
- Type conversions
- Feature engineering
- Normalization
- Outlier handling
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats

from ..core.config import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class TidyingReport:
    """Report of tidying operations performed."""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    columns_added: List[str]
    columns_removed: List[str]
    columns_converted: Dict[str, Tuple[str, str]]  # col -> (old_type, new_type)
    missing_values_handled: Dict[str, int]
    outliers_detected: Dict[str, int]
    operations_applied: List[str]
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Data Tidying Report ===",
            f"Shape: {self.original_shape} → {self.final_shape}",
            f"Columns added: {len(self.columns_added)}",
            f"Columns removed: {len(self.columns_removed)}",
            f"Type conversions: {len(self.columns_converted)}",
            f"Missing values handled: {sum(self.missing_values_handled.values())}",
            f"Outliers detected: {sum(self.outliers_detected.values())}",
            f"Operations: {len(self.operations_applied)}",
        ]
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)


class DataTidier:
    """
    Clean and transform heterogeneous data for pattern detection.
    
    Example:
        tidier = DataTidier(PreprocessingConfig(
            missing_strategy="interpolate",
            create_lag_features=True,
            normalize_numeric=True
        ))
        
        clean_df, report = tidier.tidy(raw_df)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._fitted_params: Dict[str, Any] = {}
    
    def tidy(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        id_column: Optional[str] = None,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, TidyingReport]:
        """
        Clean and transform the DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Target/outcome column (excluded from some transformations)
            timestamp_column: Timestamp column for time-based features
            id_column: ID column (e.g., batch_number)
            fit: Whether to fit parameters (True for training, False for inference)
            
        Returns:
            Tuple of (cleaned DataFrame, tidying report)
        """
        report = TidyingReport(
            original_shape=df.shape,
            final_shape=df.shape,
            columns_added=[],
            columns_removed=[],
            columns_converted={},
            missing_values_handled={},
            outliers_detected={},
            operations_applied=[],
        )
        
        result = df.copy()
        
        # 1. Auto-convert types
        if self.config.auto_convert_types:
            result, conversions = self._convert_types(result, timestamp_column)
            report.columns_converted = conversions
            report.operations_applied.append("type_conversion")
        
        # 2. Handle missing values
        result, missing_handled = self._handle_missing(result, target_column)
        report.missing_values_handled = missing_handled
        report.operations_applied.append("missing_value_handling")
        
        # 3. Create time features
        if self.config.create_time_features and timestamp_column:
            result, time_cols = self._create_time_features(result, timestamp_column)
            report.columns_added.extend(time_cols)
            report.operations_applied.append("time_feature_creation")
        
        # 4. Create lag features
        if self.config.create_lag_features and timestamp_column:
            result, lag_cols = self._create_lag_features(result, target_column, timestamp_column)
            report.columns_added.extend(lag_cols)
            report.operations_applied.append("lag_feature_creation")
        
        # 5. Handle outliers
        if self.config.handle_outliers:
            result, outliers = self._handle_outliers(result, target_column, fit)
            report.outliers_detected = outliers
            report.operations_applied.append("outlier_handling")
        
        # 6. Normalize numeric columns
        if self.config.normalize_numeric:
            result = self._normalize(result, target_column, fit)
            report.operations_applied.append("normalization")
        
        # 7. Encode categorical columns
        if self.config.encode_categorical:
            result, encoded_cols = self._encode_categorical(result, target_column, fit)
            report.columns_added.extend(encoded_cols)
            report.operations_applied.append("categorical_encoding")
        
        report.final_shape = result.shape
        
        if self.config:
            logger.info(report.summary())
        
        return result, report
    
    def _convert_types(
        self,
        df: pd.DataFrame,
        timestamp_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
        """Auto-convert column types."""
        conversions = {}
        result = df.copy()
        
        for col in result.columns:
            old_type = str(result[col].dtype)
            
            # Try datetime conversion
            if col == timestamp_column or "date" in col.lower() or "time" in col.lower():
                try:
                    result[col] = pd.to_datetime(result[col])
                    new_type = str(result[col].dtype)
                    if new_type != old_type:
                        conversions[col] = (old_type, new_type)
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Try numeric conversion for object columns
            if result[col].dtype == "object":
                try:
                    numeric = pd.to_numeric(result[col], errors="coerce")
                    # Only convert if most values are numeric
                    if numeric.notna().sum() / len(numeric) > 0.5:
                        result[col] = numeric
                        conversions[col] = (old_type, str(result[col].dtype))
                except (ValueError, TypeError):
                    pass
        
        return result, conversions
    
    def _handle_missing(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Handle missing values."""
        result = df.copy()
        missing_handled = {}
        
        for col in result.columns:
            missing_count = result[col].isna().sum()
            if missing_count == 0:
                continue
            
            missing_handled[col] = missing_count
            
            # Don't modify target column
            if col == target_column:
                continue
            
            strategy = self.config.missing_strategy
            
            # Infer strategy based on column type
            if strategy == "infer":
                if result[col].dtype in ["float64", "int64"]:
                    strategy = "interpolate"
                elif result[col].dtype == "object":
                    strategy = "fill_mode"
                else:
                    strategy = "fill_median"
            
            if strategy == "drop":
                result = result.dropna(subset=[col])
            elif strategy == "fill_mean":
                if result[col].dtype in ["float64", "int64"]:
                    result[col] = result[col].fillna(result[col].mean())
            elif strategy == "fill_median":
                if result[col].dtype in ["float64", "int64"]:
                    result[col] = result[col].fillna(result[col].median())
            elif strategy == "fill_mode":
                mode = result[col].mode()
                if len(mode) > 0:
                    result[col] = result[col].fillna(mode[0])
            elif strategy == "interpolate":
                if result[col].dtype in ["float64", "int64"]:
                    result[col] = result[col].interpolate(method="linear")
                    # Fill remaining NaNs at edges
                    result[col] = result[col].fillna(method="bfill").fillna(method="ffill")
            elif strategy == "fill_zero":
                result[col] = result[col].fillna(0)
        
        return result, missing_handled
    
    def _create_time_features(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create time-based features from timestamp."""
        result = df.copy()
        new_cols = []
        
        if timestamp_column not in result.columns:
            return result, new_cols
        
        ts = result[timestamp_column]
        
        if not pd.api.types.is_datetime64_any_dtype(ts):
            try:
                ts = pd.to_datetime(ts)
            except (ValueError, TypeError):
                return result, new_cols
        
        # Create features
        features = {
            f"{timestamp_column}_hour": ts.dt.hour,
            f"{timestamp_column}_dayofweek": ts.dt.dayofweek,
            f"{timestamp_column}_month": ts.dt.month,
            f"{timestamp_column}_quarter": ts.dt.quarter,
            f"{timestamp_column}_year": ts.dt.year,
            f"{timestamp_column}_is_weekend": ts.dt.dayofweek >= 5,
        }
        
        for name, values in features.items():
            result[name] = values
            new_cols.append(name)
        
        return result, new_cols
    
    def _create_lag_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        timestamp_column: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create lagged features for numeric columns."""
        result = df.copy()
        new_cols = []
        
        # Sort by timestamp
        if timestamp_column in result.columns:
            result = result.sort_values(timestamp_column)
        
        # Find numeric columns to lag
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target from lagging
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Limit to most important columns (avoid explosion)
        if len(numeric_cols) > 10:
            numeric_cols = numeric_cols[:10]
        
        for col in numeric_cols:
            for lag in self.config.lag_periods:
                lag_name = f"{col}_lag_{lag}"
                result[lag_name] = result[col].shift(lag)
                new_cols.append(lag_name)
        
        return result, new_cols
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Detect and handle outliers."""
        result = df.copy()
        outliers_detected = {}
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target from outlier handling
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        for col in numeric_cols:
            method = self.config.outlier_method
            
            if method == "iqr":
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
            elif method == "zscore":
                mean = result[col].mean()
                std = result[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std
            else:
                continue
            
            # Store bounds for inference
            if fit:
                self._fitted_params[f"outlier_bounds_{col}"] = (lower, upper)
            elif f"outlier_bounds_{col}" in self._fitted_params:
                lower, upper = self._fitted_params[f"outlier_bounds_{col}"]
            
            # Detect outliers
            is_outlier = (result[col] < lower) | (result[col] > upper)
            outlier_count = is_outlier.sum()
            
            if outlier_count > 0:
                outliers_detected[col] = outlier_count
                
                action = self.config.outlier_action
                if action == "drop":
                    result = result[~is_outlier]
                elif action == "cap":
                    result.loc[result[col] < lower, col] = lower
                    result.loc[result[col] > upper, col] = upper
                elif action == "flag":
                    result[f"{col}_is_outlier"] = is_outlier
        
        return result, outliers_detected
    
    def _normalize(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """Normalize numeric columns."""
        result = df.copy()
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target from normalization
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        method = self.config.normalization_method
        
        for col in numeric_cols:
            if fit:
                if method == "standard":
                    mean = result[col].mean()
                    std = result[col].std()
                    self._fitted_params[f"norm_{col}"] = {"mean": mean, "std": std}
                elif method == "minmax":
                    min_val = result[col].min()
                    max_val = result[col].max()
                    self._fitted_params[f"norm_{col}"] = {"min": min_val, "max": max_val}
                elif method == "robust":
                    median = result[col].median()
                    iqr = result[col].quantile(0.75) - result[col].quantile(0.25)
                    self._fitted_params[f"norm_{col}"] = {"median": median, "iqr": iqr}
            
            params = self._fitted_params.get(f"norm_{col}", {})
            
            if method == "standard" and params:
                std = params["std"]
                if std > 0:
                    result[col] = (result[col] - params["mean"]) / std
            elif method == "minmax" and params:
                range_val = params["max"] - params["min"]
                if range_val > 0:
                    result[col] = (result[col] - params["min"]) / range_val
            elif method == "robust" and params:
                iqr = params["iqr"]
                if iqr > 0:
                    result[col] = (result[col] - params["median"]) / iqr
        
        return result
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical columns."""
        result = df.copy()
        new_cols = []
        
        cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Exclude target from encoding
        if target_column and target_column in cat_cols:
            cat_cols.remove(target_column)
        
        method = self.config.encoding_method
        
        for col in cat_cols:
            n_unique = result[col].nunique()
            
            # Auto-select method
            if method == "auto":
                if n_unique <= 10:
                    actual_method = "onehot"
                else:
                    actual_method = "label"
            else:
                actual_method = method
            
            if actual_method == "onehot":
                dummies = pd.get_dummies(result[col], prefix=col, drop_first=True)
                result = pd.concat([result, dummies], axis=1)
                new_cols.extend(dummies.columns.tolist())
                result = result.drop(columns=[col])
            
            elif actual_method == "label":
                if fit:
                    mapping = {v: i for i, v in enumerate(result[col].unique())}
                    self._fitted_params[f"label_encode_{col}"] = mapping
                else:
                    mapping = self._fitted_params.get(f"label_encode_{col}", {})
                
                result[col] = result[col].map(mapping)
        
        return result, new_cols
    
    def fit(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> "DataTidier":
        """Fit the tidier to training data without transforming."""
        self.tidy(df, target_column, timestamp_column, fit=True)
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, TidyingReport]:
        """Transform new data using fitted parameters."""
        return self.tidy(df, target_column, timestamp_column, fit=False)

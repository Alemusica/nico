"""
TSFresh Feature Extraction Integration
======================================

Automatic extraction of 800+ time-series features using tsfresh.
https://tsfresh.readthedocs.io/

Features include:
- Statistical: mean, variance, skewness, kurtosis
- Temporal: autocorrelation, partial_autocorrelation
- Entropy: sample_entropy, approximate_entropy
- Peaks: number_peaks, number_cwt_peaks
- Frequency: fft_coefficient, spectral_welch_density
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging
import warnings

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import tsfresh (may not be installed)
try:
    from tsfresh import extract_features, select_features
    from tsfresh.feature_extraction import (
        MinimalFCParameters,
        EfficientFCParameters,
        ComprehensiveFCParameters,
    )
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    logger.warning("tsfresh not installed. Install with: pip install tsfresh")


@dataclass
class FeatureExtractionConfig:
    """
    Configuration for tsfresh feature extraction.
    
    Attributes:
        mode: 'minimal' (~10 features), 'efficient' (~~750), 'comprehensive' (~800+)
        n_jobs: Parallel jobs for extraction (-1 = all cores)
        disable_progressbar: Disable tsfresh progress bar
        impute: Whether to impute NaN values in extracted features
        select_relevant: Use target to select statistically relevant features only
        fdr_level: FDR level for feature selection (0.05 = 5% false discovery)
    """
    mode: str = "efficient"  # minimal, efficient, comprehensive
    n_jobs: int = 1  # Be careful with parallelism
    disable_progressbar: bool = True
    impute: bool = True
    select_relevant: bool = False
    fdr_level: float = 0.05
    
    # Custom feature set (overrides mode if provided)
    custom_features: Optional[Dict[str, Any]] = None


class TSFreshExtractor:
    """
    Feature extractor using tsfresh library.
    
    Example:
        extractor = TSFreshExtractor(FeatureExtractionConfig(mode='efficient'))
        
        # Extract features from time series
        features_df = extractor.extract(
            df=raw_df,
            column_id='batch_id',
            column_sort='timestamp',
            column_value='sensor_reading'
        )
    """
    
    # Feature set presets
    FEATURE_SETS = {
        "minimal": "MinimalFCParameters",
        "efficient": "EfficientFCParameters", 
        "comprehensive": "ComprehensiveFCParameters",
    }
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        self.config = config or FeatureExtractionConfig()
        self._feature_params = None
        self._extracted_features: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        
        if not TSFRESH_AVAILABLE:
            raise ImportError(
                "tsfresh is not installed. Install with: pip install tsfresh>=0.20.2"
            )
        
        self._setup_feature_params()
    
    def _setup_feature_params(self):
        """Setup feature extraction parameters."""
        if self.config.custom_features:
            self._feature_params = self.config.custom_features
        else:
            mode = self.config.mode
            if mode == "minimal":
                self._feature_params = MinimalFCParameters()
            elif mode == "efficient":
                self._feature_params = EfficientFCParameters()
            elif mode == "comprehensive":
                self._feature_params = ComprehensiveFCParameters()
            else:
                logger.warning(f"Unknown mode {mode}, using efficient")
                self._feature_params = EfficientFCParameters()
    
    def extract(
        self,
        df: pd.DataFrame,
        column_id: str,
        column_sort: Optional[str] = None,
        column_value: Optional[Union[str, List[str]]] = None,
        column_kind: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract tsfresh features from time series data.
        
        tsfresh expects data in "long" format:
            - column_id: identifies each time series (e.g., batch_id, sensor_id)
            - column_sort: time ordering column (e.g., timestamp)
            - column_value: value column(s) to extract features from
            - column_kind: if multiple value columns, this groups them
        
        Args:
            df: DataFrame in long format
            column_id: ID column identifying each series
            column_sort: Time/order column (optional but recommended)
            column_value: Value column(s) to process
            column_kind: Column specifying the kind of measurement
            
        Returns:
            DataFrame with extracted features, one row per ID
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()
        
        logger.info(f"Extracting features with mode={self.config.mode}")
        logger.info(f"Input shape: {df.shape}")
        
        # Suppress tsfresh warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                extracted = extract_features(
                    df,
                    column_id=column_id,
                    column_sort=column_sort,
                    column_value=column_value,
                    column_kind=column_kind,
                    default_fc_parameters=self._feature_params,
                    n_jobs=self.config.n_jobs,
                    disable_progressbar=self.config.disable_progressbar,
                )
                
                # Impute NaN values
                if self.config.impute:
                    extracted = impute(extracted)
                
                self._extracted_features = extracted
                logger.info(f"Extracted {extracted.shape[1]} features")
                
                return extracted
                
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                raise
    
    def extract_from_wide(
        self,
        df: pd.DataFrame,
        id_column: str,
        value_columns: List[str],
        timestamp_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract features from "wide" format data.
        
        Converts wide format to long format, then extracts.
        
        Wide format example:
            batch_id, timestamp, temp, pressure, humidity
            1, 2024-01-01, 25.0, 1013, 60
            1, 2024-01-02, 26.0, 1012, 62
        
        Args:
            df: Wide format DataFrame
            id_column: Column identifying each series (batch_id)
            value_columns: List of value columns to melt
            timestamp_column: Time column
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Converting wide format to long format")
        
        # Melt to long format
        id_vars = [id_column]
        if timestamp_column:
            id_vars.append(timestamp_column)
        
        long_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_columns,
            var_name="_feature_kind",
            value_name="_feature_value",
        )
        
        # Extract features
        return self.extract(
            df=long_df,
            column_id=id_column,
            column_sort=timestamp_column,
            column_value="_feature_value",
            column_kind="_feature_kind",
        )
    
    def select_relevant(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        fdr_level: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Select statistically relevant features using tsfresh.
        
        Uses the Benjamini-Hochberg procedure to control false discovery rate.
        
        Args:
            features_df: Extracted features
            target: Target variable (binary or continuous)
            fdr_level: False discovery rate level (default from config)
            
        Returns:
            DataFrame with only relevant features
        """
        if features_df.empty:
            return features_df
        
        fdr = fdr_level or self.config.fdr_level
        
        logger.info(f"Selecting relevant features at FDR={fdr}")
        
        try:
            selected = select_features(features_df, target, fdr_level=fdr)
            self._selected_features = selected.columns.tolist()
            
            logger.info(f"Selected {len(self._selected_features)} out of {features_df.shape[1]} features")
            
            return selected
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return features_df
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        if self._extracted_features is not None:
            return self._extracted_features.columns.tolist()
        return []
    
    def get_selected_features(self) -> List[str]:
        """Get names of selected relevant features."""
        return self._selected_features or []


def quick_extract(
    df: pd.DataFrame,
    id_column: str,
    value_columns: List[str],
    timestamp_column: Optional[str] = None,
    mode: str = "efficient",
) -> pd.DataFrame:
    """
    Quick feature extraction helper function.
    
    Example:
        features = quick_extract(
            batch_data,
            id_column='batch_id',
            value_columns=['temp', 'pressure', 'humidity'],
            timestamp_column='timestamp',
            mode='efficient'
        )
    """
    if not TSFRESH_AVAILABLE:
        raise ImportError("tsfresh not installed")
    
    extractor = TSFreshExtractor(FeatureExtractionConfig(mode=mode))
    return extractor.extract_from_wide(
        df, id_column, value_columns, timestamp_column
    )


# Feature extraction without tsfresh (fallback)
def manual_extract_basic_features(
    df: pd.DataFrame,
    id_column: str,
    value_columns: List[str],
) -> pd.DataFrame:
    """
    Manual basic feature extraction (fallback when tsfresh unavailable).
    
    Extracts: mean, std, min, max, median, skew, kurtosis
    """
    features_list = []
    
    for id_val in df[id_column].unique():
        subset = df[df[id_column] == id_val]
        features = {id_column: id_val}
        
        for col in value_columns:
            if col not in subset.columns:
                continue
            
            values = subset[col].dropna()
            if len(values) == 0:
                continue
            
            features[f"{col}__mean"] = values.mean()
            features[f"{col}__std"] = values.std()
            features[f"{col}__min"] = values.min()
            features[f"{col}__max"] = values.max()
            features[f"{col}__median"] = values.median()
            
            if len(values) > 2:
                from scipy.stats import skew, kurtosis
                features[f"{col}__skewness"] = skew(values)
                features[f"{col}__kurtosis"] = kurtosis(values)
        
        features_list.append(features)
    
    result = pd.DataFrame(features_list)
    if id_column in result.columns:
        result = result.set_index(id_column)
    
    return result

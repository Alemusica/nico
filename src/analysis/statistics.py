"""
Statistical Analysis
====================
Functions for computing statistics on satellite altimetry data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Any


def compute_statistics(data: np.ndarray | xr.DataArray) -> dict[str, Any]:
    """
    Compute comprehensive statistics for a data array.
    
    Parameters
    ----------
    data : array-like
        Input data
        
    Returns
    -------
    dict
        Statistics including count, mean, std, min, max, percentiles
    """
    values = np.asarray(data).flatten()
    valid = values[np.isfinite(values)]
    
    if len(valid) == 0:
        return {"n_valid": 0, "n_total": len(values)}
    
    return {
        "n_valid": len(valid),
        "n_total": len(values),
        "pct_valid": 100 * len(valid) / len(values),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "p25": float(np.percentile(valid, 25)),
        "p75": float(np.percentile(valid, 75)),
        "iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
    }


def compute_monthly_statistics(
    df: pd.DataFrame,
    value_col: str = "dot",
) -> pd.DataFrame:
    """
    Compute monthly statistics for a value column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'month' column and value column
    value_col : str
        Name of column to analyze
        
    Returns
    -------
    pd.DataFrame
        Monthly statistics
    """
    monthly_stats = []
    
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    
    for month in range(1, 13):
        month_data = df[df["month"] == month][value_col]
        
        if len(month_data) == 0:
            monthly_stats.append({
                "month": month,
                "month_name": month_names[month],
                "n_obs": 0,
            })
            continue
        
        stats = compute_statistics(month_data.values)
        stats["month"] = month
        stats["month_name"] = month_names[month]
        stats["n_obs"] = len(month_data)
        
        if "cycle" in df.columns:
            stats["n_cycles"] = df[df["month"] == month]["cycle"].nunique()
        
        monthly_stats.append(stats)
    
    return pd.DataFrame(monthly_stats)


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Compute correlation matrix for selected columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list[str]
        Columns to include in correlation
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    return df[columns].corr()


def detect_outliers_iqr(
    data: np.ndarray,
    k: float = 1.5,
) -> np.ndarray:
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    k : float
        IQR multiplier (default 1.5)
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = outlier)
    """
    q1, q3 = np.nanpercentile(data, [25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def summarize_timeline(
    timeline_df: pd.DataFrame,
    slope_col: str = "slope_mm_per_m",
) -> dict:
    """
    Summarize slope timeline statistics.
    
    Parameters
    ----------
    timeline_df : pd.DataFrame
        Timeline DataFrame from compute_slope_timeline
    slope_col : str
        Name of slope column
        
    Returns
    -------
    dict
        Summary statistics
    """
    if timeline_df.empty:
        return {"n_periods": 0}
    
    slopes = timeline_df[slope_col].values
    
    return {
        "n_periods": len(timeline_df),
        "mean_slope": float(np.mean(slopes)),
        "std_slope": float(np.std(slopes)),
        "min_slope": float(np.min(slopes)),
        "max_slope": float(np.max(slopes)),
        "median_slope": float(np.median(slopes)),
        "time_start": timeline_df["date"].min(),
        "time_end": timeline_df["date"].max(),
    }

"""
Analysis Module
===============
DOT computation, slope analysis, and statistical functions.
"""
from .dot import compute_dot, compute_dot_dataframe
from .slope import compute_slope, compute_slope_timeline, bin_by_longitude
from .statistics import compute_statistics, compute_monthly_statistics

__all__ = [
    "compute_dot",
    "compute_dot_dataframe",
    "compute_slope",
    "compute_slope_timeline",
    "bin_by_longitude",
    "compute_statistics",
    "compute_monthly_statistics",
]

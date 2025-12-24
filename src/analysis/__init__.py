"""
Analysis Module
===============
DOT computation, slope analysis, statistical functions, and root cause analysis.
"""
from .dot import compute_dot, compute_dot_dataframe
from .slope import compute_slope, compute_slope_timeline, bin_by_longitude
from .statistics import compute_statistics, compute_monthly_statistics
from .root_cause import (
    IshikawaCategory,
    IshikawaCause,
    IshikawaDiagram,
    FMEAItem,
    FMEAAnalysis,
    WhyStep,
    FiveWhyAnalysis,
    FloodPhysicsScore,
    RootCauseAnalyzer,
    create_flood_ishikawa_template,
    create_satellite_fmea_template,
)

__all__ = [
    # DOT and slope
    "compute_dot",
    "compute_dot_dataframe",
    "compute_slope",
    "compute_slope_timeline",
    "bin_by_longitude",
    "compute_statistics",
    "compute_monthly_statistics",
    # Root cause analysis
    "IshikawaCategory",
    "IshikawaCause",
    "IshikawaDiagram",
    "FMEAItem",
    "FMEAAnalysis",
    "WhyStep",
    "FiveWhyAnalysis",
    "FloodPhysicsScore",
    "RootCauseAnalyzer",
    "create_flood_ishikawa_template",
    "create_satellite_fmea_template",
]

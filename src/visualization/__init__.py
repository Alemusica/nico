"""
Visualization Module
====================
Plotting functions for satellite altimetry data visualization.
"""
from .plotly_charts import (
    create_slope_timeline_plot,
    create_dot_profile_plot,
    create_spatial_scatter_plot,
    create_monthly_subplots,
)
from .matplotlib_charts import (
    create_three_panel_plot,
    create_monthly_analysis_figure,
    create_dot_map,
)

__all__ = [
    # Plotly (interactive)
    "create_slope_timeline_plot",
    "create_dot_profile_plot",
    "create_spatial_scatter_plot",
    "create_monthly_subplots",
    # Matplotlib (static)
    "create_three_panel_plot",
    "create_monthly_analysis_figure",
    "create_dot_map",
]

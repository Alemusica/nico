"""
Example modules for Pattern Detection Engine.

Available examples:
- manufacturing_example: Batch failure prediction in hose manufacturing
- load_forecasting_example: Electricity demand pattern detection
"""

from .manufacturing_example import run_manufacturing_example, generate_synthetic_manufacturing_data
from .load_forecasting_example import run_load_forecasting_example, generate_synthetic_load_data

__all__ = [
    "run_manufacturing_example",
    "generate_synthetic_manufacturing_data",
    "run_load_forecasting_example",
    "generate_synthetic_load_data",
]

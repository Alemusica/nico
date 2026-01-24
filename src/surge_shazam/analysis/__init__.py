"""
Analysis tools for Surge-Shazam.

Modules:
- sonification: Convert atmospheric data to audible sound
"""

from .sonification import (
    sonify_pressure_series,
    compute_spectral_slope,
    classify_noise_color,
)

__all__ = [
    "sonify_pressure_series",
    "compute_spectral_slope",
    "classify_noise_color",
]

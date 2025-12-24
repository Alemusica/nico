"""
Output module for Pattern Detection Engine.

Handles report generation, gray zone analysis, and various output formats.
"""

from .reporter import PatternReporter
from .gray_zone import (
    GrayZoneDetector,
    GrayZoneConfig,
    GrayZonePattern,
    ReviewPriority,
)

__all__ = [
    "PatternReporter",
    "GrayZoneDetector",
    "GrayZoneConfig",
    "GrayZonePattern",
    "ReviewPriority",
]

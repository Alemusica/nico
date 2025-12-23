"""
Detection module for Pattern Detection Engine.

Contains supervised and unsupervised pattern detection algorithms.
"""

from .supervised import SupervisedDetector
from .unsupervised import UnsupervisedDetector
from .base import BaseDetector

__all__ = [
    "BaseDetector",
    "SupervisedDetector",
    "UnsupervisedDetector",
]

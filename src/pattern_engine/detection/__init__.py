"""
Detection module for Pattern Detection Engine.

Contains supervised and unsupervised pattern detection algorithms.
Includes association rules mining with mlxtend.
"""

from .supervised import SupervisedDetector
from .unsupervised import UnsupervisedDetector
from .base import BaseDetector

# Association rules (with graceful fallback)
try:
    from .association_rules import (
        AssociationRuleDetector,
        AssociationRuleConfig,
        AssociationRule,
        quick_association_rules,
        MLXTEND_AVAILABLE,
    )
except ImportError:
    MLXTEND_AVAILABLE = False
    AssociationRuleDetector = None
    AssociationRuleConfig = None
    AssociationRule = None
    quick_association_rules = None

__all__ = [
    "BaseDetector",
    "SupervisedDetector",
    "UnsupervisedDetector",
    # Association rules
    "AssociationRuleDetector",
    "AssociationRuleConfig",
    "AssociationRule",
    "quick_association_rules",
    "MLXTEND_AVAILABLE",
]

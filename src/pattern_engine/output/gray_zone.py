"""
Gray Zone Detector
==================

Identifies patterns that have high statistical confidence but weak physics support.
These patterns need human review before being deployed.

Gray Zone = {patterns where P(statistical) is high BUT P(physics) is low}

Example:
- A pattern "high humidity → failure" might have 90% confidence statistically
- But physically, humidity shouldn't affect the process at measured levels
- This goes to gray zone for human review
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import pandas as pd
import numpy as np

from ..physics.rules import PhysicsValidator, ValidationResult
from ..core.pattern import Pattern

logger = logging.getLogger(__name__)


class ReviewPriority(Enum):
    """Priority levels for human review."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GrayZonePattern:
    """
    A pattern flagged for human review.
    
    Attributes:
        pattern: The original pattern
        statistical_score: Confidence/support from detection
        physics_score: Score from physics validation
        gray_score: Combined "grayness" score (higher = more suspicious)
        priority: Review priority
        reason: Why this pattern is flagged
        suggestions: Possible explanations or actions
    """
    pattern: Pattern
    statistical_score: float
    physics_score: float
    gray_score: float
    priority: ReviewPriority
    reason: str
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": str(self.pattern.pattern_id) if self.pattern else None,
            "pattern_conditions": [str(c) for c in self.pattern.conditions] if self.pattern else [],
            "statistical_score": self.statistical_score,
            "physics_score": self.physics_score,
            "gray_score": self.gray_score,
            "priority": self.priority.value,
            "reason": self.reason,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


@dataclass 
class GrayZoneConfig:
    """
    Configuration for gray zone detection.
    
    Attributes:
        statistical_threshold: Min statistical score to consider (0-1)
        physics_threshold: Physics score below this = weak support
        gray_score_threshold: Gray score above this = flagged
        auto_reject_physics_below: Auto-reject patterns with physics below this
        auto_approve_physics_above: Auto-approve patterns with physics above this
    """
    statistical_threshold: float = 0.5  # Only consider patterns with conf > 50%
    physics_threshold: float = 0.4  # Physics < 40% = weak
    gray_score_threshold: float = 0.5  # Gray score > 50% = needs review
    auto_reject_physics_below: float = 0.1  # Physics < 10% = auto reject
    auto_approve_physics_above: float = 0.8  # Physics > 80% = auto approve


class GrayZoneDetector:
    """
    Detects patterns that need human review.
    
    Logic:
    1. High stats + Low physics = Gray Zone (needs review)
    2. High stats + High physics = Auto-approved
    3. Low stats = Filtered out earlier
    4. Very low physics = Auto-rejected
    
    Example:
        detector = GrayZoneDetector(physics_validator=validator)
        
        gray_patterns = detector.analyze(
            patterns=detected_patterns,
            data=original_data
        )
        
        for gp in gray_patterns:
            print(f"{gp.pattern}: gray={gp.gray_score:.2f}, reason={gp.reason}")
    """
    
    def __init__(
        self,
        physics_validator: PhysicsValidator,
        config: Optional[GrayZoneConfig] = None,
    ):
        self.physics_validator = physics_validator
        self.config = config or GrayZoneConfig()
        
        self._gray_patterns: List[GrayZonePattern] = []
        self._approved_patterns: List[Pattern] = []
        self._rejected_patterns: List[Pattern] = []
    
    def analyze(
        self,
        patterns: List[Pattern],
        data: pd.DataFrame,
    ) -> List[GrayZonePattern]:
        """
        Analyze patterns for gray zone classification.
        
        Args:
            patterns: List of detected patterns
            data: Original data used for detection
            
        Returns:
            List of GrayZonePattern objects needing review
        """
        self._gray_patterns = []
        self._approved_patterns = []
        self._rejected_patterns = []
        
        logger.info(f"Analyzing {len(patterns)} patterns for gray zone")
        
        # Apply physics validation to data
        validated_data = self.physics_validator.apply(data)
        
        for pattern in patterns:
            # Get statistical score (confidence)
            stat_score = pattern.confidence if hasattr(pattern, 'confidence') else 0.5
            
            # Skip low statistical confidence
            if stat_score < self.config.statistical_threshold:
                continue
            
            # Get matching samples for this pattern
            matching_mask = self._get_matching_samples(pattern, data)
            matching_data = data[matching_mask] if matching_mask.sum() > 0 else data.head(1)
            
            # Compute physics score
            validation_result = self.physics_validator.validate_pattern(
                pattern_conditions={c.feature: c.value for c in pattern.conditions},
                sample_data=matching_data,
                statistical_confidence=stat_score,
            )
            
            physics_score = validation_result.physics_score
            
            # Classify pattern
            if physics_score < self.config.auto_reject_physics_below:
                # Auto reject - physics makes no sense
                self._rejected_patterns.append(pattern)
                logger.debug(f"Pattern rejected (physics={physics_score:.2f})")
                
            elif physics_score > self.config.auto_approve_physics_above:
                # Auto approve - strong physics support
                self._approved_patterns.append(pattern)
                logger.debug(f"Pattern approved (physics={physics_score:.2f})")
                
            elif physics_score < self.config.physics_threshold:
                # Gray zone - high stats but weak physics
                gray_score = self._compute_gray_score(stat_score, physics_score)
                
                if gray_score > self.config.gray_score_threshold:
                    gray_pattern = self._create_gray_pattern(
                        pattern=pattern,
                        stat_score=stat_score,
                        physics_score=physics_score,
                        gray_score=gray_score,
                        matching_data=matching_data,
                    )
                    self._gray_patterns.append(gray_pattern)
                else:
                    # Borderline - approve with caution
                    self._approved_patterns.append(pattern)
            else:
                # Moderate physics support - approve
                self._approved_patterns.append(pattern)
        
        # Sort by gray score (most suspicious first)
        self._gray_patterns.sort(key=lambda x: x.gray_score, reverse=True)
        
        logger.info(
            f"Analysis complete: {len(self._approved_patterns)} approved, "
            f"{len(self._gray_patterns)} gray zone, {len(self._rejected_patterns)} rejected"
        )
        
        return self._gray_patterns
    
    def _get_matching_samples(
        self,
        pattern: Pattern,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Get boolean mask of samples matching pattern conditions."""
        mask = pd.Series(True, index=data.index)
        
        for condition in pattern.conditions:
            feature = condition.feature
            if feature not in data.columns:
                continue
            
            op = condition.operator.value if hasattr(condition.operator, 'value') else condition.operator
            value = condition.value
            
            if op == ">=":
                mask &= data[feature] >= value
            elif op == "<=":
                mask &= data[feature] <= value
            elif op == ">":
                mask &= data[feature] > value
            elif op == "<":
                mask &= data[feature] < value
            elif op == "==":
                mask &= data[feature] == value
            elif op == "!=":
                mask &= data[feature] != value
        
        return mask
    
    def _compute_gray_score(
        self,
        stat_score: float,
        physics_score: float,
    ) -> float:
        """
        Compute gray zone score.
        
        Higher = more suspicious (high stats, low physics)
        Formula: gray = stat_score * (1 - physics_score)
        """
        return stat_score * (1 - physics_score)
    
    def _create_gray_pattern(
        self,
        pattern: Pattern,
        stat_score: float,
        physics_score: float,
        gray_score: float,
        matching_data: pd.DataFrame,
    ) -> GrayZonePattern:
        """Create a GrayZonePattern with explanations."""
        
        # Determine priority
        if gray_score > 0.8:
            priority = ReviewPriority.CRITICAL
        elif gray_score > 0.6:
            priority = ReviewPriority.HIGH
        elif gray_score > 0.4:
            priority = ReviewPriority.MEDIUM
        else:
            priority = ReviewPriority.LOW
        
        # Generate reason
        conditions_str = ", ".join(str(c) for c in pattern.conditions)
        reason = (
            f"High statistical confidence ({stat_score:.1%}) but weak physics support ({physics_score:.1%}). "
            f"Conditions: {conditions_str}"
        )
        
        # Generate suggestions
        suggestions = []
        
        if physics_score < 0.2:
            suggestions.append("Consider if this is a spurious correlation due to confounding variables")
            suggestions.append("Check if there's a hidden causal mechanism not captured by physics rules")
        
        if stat_score > 0.9 and physics_score < 0.3:
            suggestions.append("High confidence with very weak physics - likely spurious or data artifact")
            suggestions.append("Verify data quality and collection methodology")
        
        suggestions.append("Review domain expert knowledge for possible physical explanation")
        suggestions.append("Consider adding new physics rules if pattern is validated")
        
        return GrayZonePattern(
            pattern=pattern,
            statistical_score=stat_score,
            physics_score=physics_score,
            gray_score=gray_score,
            priority=priority,
            reason=reason,
            suggestions=suggestions,
            metadata={
                "n_matching_samples": len(matching_data),
            }
        )
    
    def get_approved(self) -> List[Pattern]:
        """Get auto-approved patterns."""
        return self._approved_patterns
    
    def get_rejected(self) -> List[Pattern]:
        """Get auto-rejected patterns."""
        return self._rejected_patterns
    
    def get_gray_zone(self) -> List[GrayZonePattern]:
        """Get patterns needing human review."""
        return self._gray_patterns
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_analyzed": (
                len(self._approved_patterns) + 
                len(self._gray_patterns) + 
                len(self._rejected_patterns)
            ),
            "approved": len(self._approved_patterns),
            "gray_zone": len(self._gray_patterns),
            "rejected": len(self._rejected_patterns),
            "gray_by_priority": {
                priority.value: sum(1 for p in self._gray_patterns if p.priority == priority)
                for priority in ReviewPriority
            },
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert gray zone patterns to DataFrame."""
        if not self._gray_patterns:
            return pd.DataFrame()
        
        return pd.DataFrame([gp.to_dict() for gp in self._gray_patterns])
    
    def print_report(self, top_k: int = 10):
        """Print human-readable report."""
        summary = self.summary()
        
        print("\n" + "=" * 60)
        print("GRAY ZONE ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nTotal patterns analyzed: {summary['total_analyzed']}")
        print(f"  ✓ Auto-approved: {summary['approved']}")
        print(f"  ⚠ Gray zone (needs review): {summary['gray_zone']}")
        print(f"  ✗ Auto-rejected: {summary['rejected']}")
        
        if self._gray_patterns:
            print("\n" + "-" * 60)
            print("PATTERNS NEEDING HUMAN REVIEW")
            print("-" * 60)
            
            for i, gp in enumerate(self._gray_patterns[:top_k], 1):
                print(f"\n{i}. Priority: {gp.priority.value.upper()}")
                print(f"   Gray Score: {gp.gray_score:.2f} (Stats: {gp.statistical_score:.2f}, Physics: {gp.physics_score:.2f})")
                print(f"   Reason: {gp.reason}")
                if gp.suggestions:
                    print(f"   Suggestions:")
                    for s in gp.suggestions[:2]:
                        print(f"     - {s}")
        
        print("\n" + "=" * 60)

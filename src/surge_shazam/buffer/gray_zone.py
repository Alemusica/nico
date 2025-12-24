"""
Gray Zone Pattern Buffer for unvalidated historical correlations.

Key concept: Strong historical correlations that physics can't yet explain
(due to missing intermediate data, computational limits, etc.) should NOT
be discarded. They go into "gray zone" with warning flags.

The cockpit shows BOTH views to the human decision-maker:
- Historical/experience confidence
- Physics validation status

When science catches up (more data, better models), patterns graduate
from gray zone to validated.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json


class PatternStatus(Enum):
    """Status of a discovered pattern."""
    VALIDATED = "validated"           # Both historical + physics OK
    GRAY_ZONE = "gray_zone"           # Historical strong, physics weak
    REJECTED = "rejected"             # Failed both checks
    PENDING = "pending"               # Awaiting validation


@dataclass
class PatternRecord:
    """
    Record of a discovered correlation pattern.
    
    Example: "Low pressure near Portugal + NW wind duration > 12h"
             → "Surge +1.5m in Esbjerg after 24-48h"
    """
    
    # Unique identifier
    pattern_id: str
    
    # Description (human-readable)
    description: str
    
    # Fingerprint hash (for matching)
    fingerprint_hash: bytes | None = None
    
    # Confidence scores
    historical_confidence: float = 0.0  # From pattern matching (0-1)
    physics_confidence: float = 0.0     # From SWE residual (0-1, higher = better)
    
    # Validation status
    status: PatternStatus = PatternStatus.PENDING
    
    # Why physics validation failed (if applicable)
    physics_failure_reason: str = ""
    
    # Event statistics
    times_matched: int = 0
    times_correct: int = 0  # Led to actual surge
    
    # Temporal info
    discovered_at: datetime = field(default_factory=datetime.now)
    last_matched_at: datetime | None = None
    validated_at: datetime | None = None
    
    # Associated data
    example_events: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def combined_confidence(self) -> float:
        """Weighted combination of historical and physics confidence."""
        # If gray zone: trust historical more
        if self.status == PatternStatus.GRAY_ZONE:
            return 0.7 * self.historical_confidence + 0.3 * self.physics_confidence
        # If validated: equal weight
        return 0.5 * self.historical_confidence + 0.5 * self.physics_confidence
    
    @property
    def accuracy(self) -> float | None:
        """Historical accuracy of this pattern."""
        if self.times_matched == 0:
            return None
        return self.times_correct / self.times_matched
    
    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "historical_confidence": self.historical_confidence,
            "physics_confidence": self.physics_confidence,
            "status": self.status.value,
            "physics_failure_reason": self.physics_failure_reason,
            "times_matched": self.times_matched,
            "times_correct": self.times_correct,
            "discovered_at": self.discovered_at.isoformat(),
            "last_matched_at": self.last_matched_at.isoformat() if self.last_matched_at else None,
        }


class GrayZoneBuffer:
    """
    Buffer for gray zone patterns.
    
    Maintains patterns that have strong historical evidence but lack
    full physics validation. These are NOT discarded - they're shown
    to decision-makers with appropriate warnings.
    """
    
    def __init__(
        self,
        historical_threshold: float = 0.75,
        physics_threshold: float = 0.05,  # Residual < this = physics OK
        max_patterns: int = 1000
    ):
        self.historical_threshold = historical_threshold
        self.physics_threshold = physics_threshold
        self.max_patterns = max_patterns
        
        self.patterns: dict[str, PatternRecord] = {}
        self.gray_zone: dict[str, PatternRecord] = {}
        self.validated: dict[str, PatternRecord] = {}
    
    def add_pattern(
        self,
        pattern_id: str,
        description: str,
        historical_confidence: float,
        physics_residual: float,
        fingerprint_hash: bytes | None = None,
        example_event: dict | None = None
    ) -> PatternRecord:
        """
        Add or update a pattern with validation.
        
        Args:
            pattern_id: Unique identifier
            description: Human-readable description
            historical_confidence: Pattern match confidence (0-1)
            physics_residual: SWE residual (lower = better physics match)
            fingerprint_hash: Hash for future matching
            example_event: Example event data
            
        Returns:
            PatternRecord with assigned status
        """
        # Convert physics residual to confidence (inverse relationship)
        physics_confidence = max(0.0, 1.0 - physics_residual / 0.1)
        
        # Determine status
        if historical_confidence >= self.historical_threshold:
            if physics_residual < self.physics_threshold:
                status = PatternStatus.VALIDATED
            else:
                status = PatternStatus.GRAY_ZONE
        else:
            if physics_residual < self.physics_threshold:
                status = PatternStatus.PENDING
            else:
                status = PatternStatus.REJECTED
        
        # Create or update record
        if pattern_id in self.patterns:
            record = self.patterns[pattern_id]
            record.historical_confidence = max(record.historical_confidence, historical_confidence)
            record.physics_confidence = physics_confidence
            record.times_matched += 1
            record.last_matched_at = datetime.now()
            record.status = status
        else:
            record = PatternRecord(
                pattern_id=pattern_id,
                description=description,
                fingerprint_hash=fingerprint_hash,
                historical_confidence=historical_confidence,
                physics_confidence=physics_confidence,
                status=status,
                times_matched=1,
            )
            self.patterns[pattern_id] = record
        
        # Set physics failure reason
        if status == PatternStatus.GRAY_ZONE:
            if physics_residual > 0.1:
                record.physics_failure_reason = "High SWE residual - physics equations not satisfied"
            else:
                record.physics_failure_reason = "Intermediate causation unclear"
        
        # Add example event
        if example_event:
            record.example_events.append(example_event)
            if len(record.example_events) > 10:
                record.example_events = record.example_events[-10:]
        
        # Sort into appropriate bucket
        self._sort_pattern(record)
        
        return record
    
    def _sort_pattern(self, record: PatternRecord) -> None:
        """Sort pattern into validated/gray_zone based on status."""
        pattern_id = record.pattern_id
        
        # Remove from other buckets first
        self.gray_zone.pop(pattern_id, None)
        self.validated.pop(pattern_id, None)
        
        if record.status == PatternStatus.VALIDATED:
            self.validated[pattern_id] = record
        elif record.status == PatternStatus.GRAY_ZONE:
            self.gray_zone[pattern_id] = record
    
    def get_gray_zone_patterns(self) -> list[PatternRecord]:
        """Get all gray zone patterns, sorted by historical confidence."""
        return sorted(
            self.gray_zone.values(),
            key=lambda p: p.historical_confidence,
            reverse=True
        )
    
    def get_validated_patterns(self) -> list[PatternRecord]:
        """Get all validated patterns."""
        return list(self.validated.values())
    
    def promote_to_validated(self, pattern_id: str, reason: str = "") -> bool:
        """
        Promote a gray zone pattern to validated.
        
        Called when physics validation succeeds (e.g., new data/models available).
        """
        if pattern_id not in self.gray_zone:
            return False
        
        record = self.gray_zone.pop(pattern_id)
        record.status = PatternStatus.VALIDATED
        record.validated_at = datetime.now()
        record.physics_failure_reason = ""
        self.validated[pattern_id] = record
        
        return True
    
    def get_cockpit_summary(self) -> dict:
        """
        Get summary for cockpit display.
        
        Returns dict with counts and top patterns for each category.
        """
        return {
            "validated_count": len(self.validated),
            "gray_zone_count": len(self.gray_zone),
            "total_patterns": len(self.patterns),
            "top_gray_zone": [
                {
                    "id": p.pattern_id,
                    "description": p.description,
                    "historical_conf": p.historical_confidence,
                    "physics_conf": p.physics_confidence,
                    "reason": p.physics_failure_reason,
                }
                for p in self.get_gray_zone_patterns()[:5]
            ],
            "recent_validated": [
                {
                    "id": p.pattern_id,
                    "description": p.description,
                    "combined_conf": p.combined_confidence,
                }
                for p in sorted(
                    self.validated.values(),
                    key=lambda x: x.validated_at or x.discovered_at,
                    reverse=True
                )[:5]
            ],
        }
    
    def save(self, filepath: str) -> None:
        """Save buffer to JSON file."""
        data = {
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
            "config": {
                "historical_threshold": self.historical_threshold,
                "physics_threshold": self.physics_threshold,
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load buffer from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.historical_threshold = data["config"]["historical_threshold"]
        self.physics_threshold = data["config"]["physics_threshold"]
        
        # Reconstruct patterns (simplified - full implementation would deserialize all fields)
        # This is a skeleton - expand as needed
        pass

"""
Core pattern data structures.

Defines Pattern, PatternMatch, and PatternDatabase classes for
representing and storing detected patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path


class PatternType(Enum):
    """Types of detected patterns."""
    CORRELATION = "correlation"       # A correlates with B
    CAUSATION = "causation"          # A causes B (with lag)
    ANOMALY = "anomaly"              # Unusual data point
    CLUSTER = "cluster"              # Group of similar items
    SEQUENCE = "sequence"            # Temporal sequence pattern
    ASSOCIATION = "association"      # Items frequently occur together
    THRESHOLD = "threshold"          # Value exceeds threshold
    TREND = "trend"                  # Increasing/decreasing trend


class ConditionOperator(Enum):
    """Operators for pattern conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    IN_RANGE = "in_range"
    NOT_IN_RANGE = "not_in_range"
    CONTAINS = "contains"
    MATCHES = "matches"  # regex


@dataclass
class Condition:
    """
    A single condition in a pattern.
    
    Example:
        Condition(
            feature="temperature",
            operator=ConditionOperator.GREATER_EQUAL,
            value=25.0,
            metadata={"unit": "celsius", "location": "extrusion_dept"}
        )
    """
    feature: str
    operator: ConditionOperator
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, data_value: Any) -> bool:
        """Evaluate if data_value satisfies this condition."""
        try:
            if self.operator == ConditionOperator.EQUALS:
                return data_value == self.value
            elif self.operator == ConditionOperator.NOT_EQUALS:
                return data_value != self.value
            elif self.operator == ConditionOperator.GREATER:
                return data_value > self.value
            elif self.operator == ConditionOperator.GREATER_EQUAL:
                return data_value >= self.value
            elif self.operator == ConditionOperator.LESS:
                return data_value < self.value
            elif self.operator == ConditionOperator.LESS_EQUAL:
                return data_value <= self.value
            elif self.operator == ConditionOperator.IN_RANGE:
                low, high = self.value
                return low <= data_value <= high
            elif self.operator == ConditionOperator.NOT_IN_RANGE:
                low, high = self.value
                return not (low <= data_value <= high)
            elif self.operator == ConditionOperator.CONTAINS:
                return self.value in str(data_value)
            elif self.operator == ConditionOperator.MATCHES:
                import re
                return bool(re.match(self.value, str(data_value)))
        except (TypeError, ValueError):
            return False
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "operator": self.operator.value,
            "value": self.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create from dictionary."""
        return cls(
            feature=data["feature"],
            operator=ConditionOperator(data["operator"]),
            value=data["value"],
            metadata=data.get("metadata", {}),
        )
    
    def __str__(self) -> str:
        return f"{self.feature} {self.operator.value} {self.value}"


@dataclass
class Outcome:
    """
    The outcome/result associated with a pattern.
    
    Example:
        Outcome(
            name="burst_failure",
            value=True,
            probability=0.85,
            metadata={"failure_type": "structural", "test_stage": "burst"}
        )
    """
    name: str
    value: Any
    probability: Optional[float] = None  # Confidence/probability of this outcome
    lag: Optional[int] = None  # Time lag between condition and outcome
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "probability": self.probability,
            "lag": self.lag,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Outcome":
        return cls(**data)


@dataclass
class Pattern:
    """
    A detected pattern representing conditions that lead to outcomes.
    
    Example (Manufacturing):
        Pattern(
            pattern_type=PatternType.CAUSATION,
            conditions=[
                Condition("temperature", ConditionOperator.GREATER_EQUAL, 25.0),
                Condition("extrusion_speed", ConditionOperator.GREATER, 150),
            ],
            outcome=Outcome("burst_failure", True, probability=0.78, lag=24),
            support=0.15,  # 15% of cases
            confidence=0.78,  # 78% of matching conditions led to failure
            description="High temperature (≥25°C) combined with high extrusion speed (>150) leads to burst failure"
        )
    """
    pattern_id: str = ""
    pattern_type: PatternType = PatternType.CORRELATION
    conditions: List[Condition] = field(default_factory=list)
    outcome: Optional[Outcome] = None
    
    # Pattern quality metrics
    support: float = 0.0  # Proportion of data where this pattern appears
    confidence: float = 0.0  # P(outcome | conditions)
    lift: float = 1.0  # confidence / P(outcome)
    
    # Feature importance for this pattern
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Causal information
    causal_strength: Optional[float] = None  # Strength of causal link
    causal_lag: Optional[int] = None  # Time lag for causal effect
    
    # Metadata
    description: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
    data_source: str = ""
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate pattern ID if not provided."""
        if not self.pattern_id:
            self.pattern_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on pattern content."""
        content = json.dumps({
            "type": self.pattern_type.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """Check if data matches all conditions of this pattern."""
        return all(
            cond.evaluate(data.get(cond.feature))
            for cond in self.conditions
        )
    
    def add_condition(self, condition: Condition) -> "Pattern":
        """Add a condition to the pattern."""
        self.conditions.append(condition)
        self.pattern_id = self._generate_id()  # Regenerate ID
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "feature_importance": self.feature_importance,
            "causal_strength": self.causal_strength,
            "causal_lag": self.causal_lag,
            "description": self.description,
            "discovered_at": self.discovered_at.isoformat(),
            "data_source": self.data_source,
            "sample_size": self.sample_size,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data.get("pattern_id", ""),
            pattern_type=PatternType(data["pattern_type"]),
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])],
            outcome=Outcome.from_dict(data["outcome"]) if data.get("outcome") else None,
            support=data.get("support", 0.0),
            confidence=data.get("confidence", 0.0),
            lift=data.get("lift", 1.0),
            feature_importance=data.get("feature_importance", {}),
            causal_strength=data.get("causal_strength"),
            causal_lag=data.get("causal_lag"),
            description=data.get("description", ""),
            discovered_at=datetime.fromisoformat(data["discovered_at"]) if data.get("discovered_at") else datetime.now(),
            data_source=data.get("data_source", ""),
            sample_size=data.get("sample_size", 0),
            metadata=data.get("metadata", {}),
        )
    
    def __str__(self) -> str:
        conditions_str = " AND ".join(str(c) for c in self.conditions)
        outcome_str = f" → {self.outcome.name}={self.outcome.value}" if self.outcome else ""
        return f"[{self.pattern_type.value}] {conditions_str}{outcome_str} (conf={self.confidence:.2f})"


@dataclass
class PatternMatch:
    """
    A match of a pattern against actual data.
    
    Represents an instance where data matched a known pattern.
    """
    pattern: Pattern
    matched_data: Dict[str, Any]
    match_score: float  # 0-1, how well it matches
    predicted_outcome: Optional[Any] = None
    actual_outcome: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_correct(self) -> Optional[bool]:
        """Check if prediction was correct (if actual outcome is known)."""
        if self.actual_outcome is None or self.predicted_outcome is None:
            return None
        return self.predicted_outcome == self.actual_outcome
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern.pattern_id,
            "matched_data": self.matched_data,
            "match_score": self.match_score,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class PatternDatabase:
    """
    Storage and retrieval of patterns.
    
    Supports:
    - Adding/removing patterns
    - Querying patterns by type, features, outcome
    - Persistence to JSON/Parquet
    - Pattern versioning
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.patterns: Dict[str, Pattern] = {}
        self.matches: List[PatternMatch] = []
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_pattern(self, pattern: Pattern) -> str:
        """Add a pattern to the database. Returns pattern ID."""
        self.patterns[pattern.pattern_id] = pattern
        return pattern.pattern_id
    
    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern. Returns True if removed."""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            return True
        return False
    
    def find_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.0,
        min_support: float = 0.0,
        features: Optional[Set[str]] = None,
        outcome_name: Optional[str] = None,
    ) -> List[Pattern]:
        """Find patterns matching criteria."""
        results = []
        for pattern in self.patterns.values():
            # Filter by type
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            
            # Filter by confidence
            if pattern.confidence < min_confidence:
                continue
            
            # Filter by support
            if pattern.support < min_support:
                continue
            
            # Filter by features
            if features:
                pattern_features = {c.feature for c in pattern.conditions}
                if not features.intersection(pattern_features):
                    continue
            
            # Filter by outcome
            if outcome_name and (not pattern.outcome or pattern.outcome.name != outcome_name):
                continue
            
            results.append(pattern)
        
        return results
    
    def match_data(self, data: Dict[str, Any]) -> List[PatternMatch]:
        """Find all patterns that match the given data."""
        matches = []
        for pattern in self.patterns.values():
            if pattern.matches(data):
                match = PatternMatch(
                    pattern=pattern,
                    matched_data=data,
                    match_score=1.0,  # Exact match
                    predicted_outcome=pattern.outcome.value if pattern.outcome else None,
                )
                matches.append(match)
                self.matches.append(match)
        return matches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        type_counts = {}
        for pattern in self.patterns.values():
            ptype = pattern.pattern_type.value
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        return {
            "name": self.name,
            "total_patterns": len(self.patterns),
            "total_matches": len(self.matches),
            "patterns_by_type": type_counts,
            "avg_confidence": sum(p.confidence for p in self.patterns.values()) / max(len(self.patterns), 1),
            "avg_support": sum(p.support for p in self.patterns.values()) / max(len(self.patterns), 1),
            "created_at": self.created_at.isoformat(),
        }
    
    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """Save database to file."""
        path = Path(path)
        data = {
            "name": self.name,
            "patterns": [p.to_dict() for p in self.patterns.values()],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
        
        if format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "parquet":
            import pandas as pd
            df = pd.DataFrame([p.to_dict() for p in self.patterns.values()])
            df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path], format: str = "json") -> "PatternDatabase":
        """Load database from file."""
        path = Path(path)
        
        if format == "json":
            with open(path, "r") as f:
                data = json.load(f)
        elif format == "parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            data = {
                "name": path.stem,
                "patterns": df.to_dict("records"),
            }
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        db = cls(name=data.get("name", "loaded"))
        for pattern_data in data.get("patterns", []):
            pattern = Pattern.from_dict(pattern_data)
            db.add_pattern(pattern)
        db.metadata = data.get("metadata", {})
        
        return db
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __iter__(self):
        return iter(self.patterns.values())

"""
Base detector class for pattern detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import pandas as pd

from ..core.pattern import Pattern, PatternDatabase


@dataclass
class DetectionResult:
    """Result from pattern detection."""
    patterns: List[Pattern]
    database: PatternDatabase
    model: Any  # Fitted model
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    @property
    def top_patterns(self) -> List[Pattern]:
        """Get top patterns by confidence."""
        return sorted(self.patterns, key=lambda p: p.confidence, reverse=True)


class BaseDetector(ABC):
    """
    Abstract base class for pattern detectors.
    
    All detectors must implement:
    - fit(): Train on data
    - detect(): Find patterns
    - predict(): Apply patterns to new data
    """
    
    def __init__(self, config: Optional[Any] = None, random_seed: int = 42):
        self.config = config
        self.random_seed = random_seed
        self._fitted = False
        self._model = None
        self._feature_names: List[str] = []
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "BaseDetector":
        """
        Fit the detector to training data.
        
        Args:
            X: Feature matrix
            y: Target variable (for supervised methods)
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def detect(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> DetectionResult:
        """
        Detect patterns in the data.
        
        Args:
            X: Feature matrix
            y: Target variable (for supervised methods)
            
        Returns:
            DetectionResult containing discovered patterns
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply detected patterns to new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions or pattern matches
        """
        pass
    
    def fit_detect(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> DetectionResult:
        """Fit and detect in one step."""
        self.fit(X, y)
        return self.detect(X, y)
    
    @property
    def is_fitted(self) -> bool:
        """Check if detector has been fitted."""
        return self._fitted
    
    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before use")

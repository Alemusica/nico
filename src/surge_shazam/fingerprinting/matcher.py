"""
🔍 Fingerprint Matcher - High-Level Pattern Matching
====================================================

Provides high-level API for matching fingerprints against a database
of historical patterns, with support for:
- Real-time monitoring (continuous matching)
- Batch processing
- Alert generation
- Physics-based filtering

Usage:
    from src.surge_shazam.fingerprinting import FingerprintMatcher
    
    matcher = FingerprintMatcher()
    await matcher.initialize()
    
    # Single match
    matches = await matcher.match(current_fingerprint)
    
    # Monitor real-time
    async for alert in matcher.monitor(data_stream):
        print(f"Alert: {alert}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, AsyncIterator, Callable
from enum import Enum

import numpy as np
import pandas as pd

from .engine import Fingerprint, FingerprintEngine, SimilarityResult
from .database import FingerprintDB

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"           # Informational match
    LOW = "low"             # Low confidence match
    MEDIUM = "medium"       # Medium confidence
    HIGH = "high"           # High confidence - attention needed
    CRITICAL = "critical"   # Very high confidence - immediate action


class MatchType(Enum):
    """Type of pattern match."""
    EXACT = "exact"                 # Very high similarity (>0.95)
    STRONG = "strong"               # High similarity (>0.85)
    MODERATE = "moderate"           # Moderate similarity (>0.70)
    WEAK = "weak"                   # Weak similarity (>0.50)
    PARTIAL = "partial"             # Partial match on subset of variables


@dataclass
class MatchResult:
    """Result of a fingerprint match."""
    query_id: str
    matched_id: str
    similarity: float
    match_type: MatchType
    matched_event_name: str = ""
    lag_to_event_days: Optional[int] = None
    
    # Breakdown
    common_variables: List[str] = field(default_factory=list)
    variable_similarities: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"MatchResult({self.query_id} ↔ {self.matched_id}, sim={self.similarity:.3f}, type={self.match_type.value})"


@dataclass
class Alert:
    """Alert generated from pattern match."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    
    # Match info
    match_result: MatchResult
    confidence: float
    
    # Location/time
    region: Optional[str] = None
    bbox: Optional[tuple] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    similar_historical_events: List[str] = field(default_factory=list)
    
    # Status
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "confidence": self.confidence,
            "region": self.region,
            "timestamp": self.timestamp.isoformat(),
            "match": {
                "query_id": self.match_result.query_id,
                "matched_id": self.match_result.matched_id,
                "similarity": self.match_result.similarity,
                "match_type": self.match_result.match_type.value,
            },
            "actions": self.recommended_actions,
            "historical_events": self.similar_historical_events,
            "acknowledged": self.acknowledged,
        }


# =============================================================================
# MATCHER CONFIGURATION
# =============================================================================

@dataclass
class MatcherConfig:
    """Configuration for fingerprint matcher."""
    
    # Similarity thresholds
    exact_threshold: float = 0.95
    strong_threshold: float = 0.85
    moderate_threshold: float = 0.70
    weak_threshold: float = 0.50
    
    # Alert thresholds
    alert_threshold: float = 0.75  # Generate alert above this
    critical_threshold: float = 0.90  # Critical alert above this
    
    # Search parameters
    top_k: int = 10
    min_common_variables: int = 3
    
    # Physics filtering
    require_physics_plausible: bool = True
    max_lag_days: int = 30  # Max reasonable precursor lag
    
    # Monitoring
    monitoring_interval_seconds: float = 60.0


DEFAULT_CONFIG = MatcherConfig()


# =============================================================================
# FINGERPRINT MATCHER
# =============================================================================

class FingerprintMatcher:
    """
    High-level fingerprint matching engine.
    
    Combines FingerprintEngine and FingerprintDB with alert logic.
    """
    
    def __init__(
        self,
        config: MatcherConfig = None,
        engine: FingerprintEngine = None,
        database: FingerprintDB = None,
    ):
        """
        Initialize matcher.
        
        Args:
            config: Matcher configuration
            engine: Fingerprint extraction engine (optional)
            database: Fingerprint database (optional)
        """
        self.config = config or DEFAULT_CONFIG
        self._engine = engine
        self._db = database
        self._initialized = False
        
        # Alert callbacks
        self._alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize engine and database connections."""
        if self._initialized:
            return
        
        if self._engine is None:
            self._engine = FingerprintEngine(
                n_kernels=10000,
                embedding_dim=100,
                use_physics_weights=True,
            )
        
        if self._db is None:
            self._db = FingerprintDB()
            await self._db.connect()
        
        self._initialized = True
        logger.info("✅ FingerprintMatcher initialized")
    
    async def close(self) -> None:
        """Close connections and stop monitoring."""
        await self.stop_monitoring()
        if self._db:
            await self._db.disconnect()
        self._initialized = False
    
    # =========================================================================
    # MATCHING OPERATIONS
    # =========================================================================
    
    async def match(
        self,
        fingerprint: Fingerprint,
        top_k: int = None,
    ) -> List[MatchResult]:
        """
        Match a fingerprint against the database.
        
        Args:
            fingerprint: Query fingerprint
            top_k: Number of top matches (default from config)
            
        Returns:
            List of MatchResult objects, sorted by similarity
        """
        await self.initialize()
        
        top_k = top_k or self.config.top_k
        
        # Search database
        db_results = await self._db.search_similar(
            fingerprint,
            top_k=top_k,
            min_similarity=self.config.weak_threshold,
        )
        
        # Convert to MatchResult
        results = []
        for matched_fp, similarity in db_results:
            match_type = self._classify_match(similarity)
            
            # Find common variables
            common_vars = list(
                set(fingerprint.variables_used) & 
                set(matched_fp.variables_used)
            )
            
            # Check minimum common variables
            if len(common_vars) < self.config.min_common_variables:
                match_type = MatchType.PARTIAL
            
            result = MatchResult(
                query_id=fingerprint.event_id,
                matched_id=matched_fp.event_id,
                similarity=similarity,
                match_type=match_type,
                common_variables=common_vars,
                metadata={
                    "matched_sources": matched_fp.sources_used,
                    "matched_uncertainty": matched_fp.uncertainty,
                },
            )
            results.append(result)
        
        return results
    
    async def match_and_alert(
        self,
        fingerprint: Fingerprint,
        region: str = None,
    ) -> Optional[Alert]:
        """
        Match fingerprint and generate alert if threshold exceeded.
        
        Args:
            fingerprint: Query fingerprint
            region: Region name for alert context
            
        Returns:
            Alert if generated, None otherwise
        """
        matches = await self.match(fingerprint)
        
        if not matches:
            return None
        
        best_match = matches[0]
        
        # Check if alert threshold exceeded
        if best_match.similarity < self.config.alert_threshold:
            return None
        
        # Determine severity
        if best_match.similarity >= self.config.critical_threshold:
            severity = AlertSeverity.CRITICAL
        elif best_match.similarity >= self.config.strong_threshold:
            severity = AlertSeverity.HIGH
        elif best_match.similarity >= self.config.moderate_threshold:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        # Generate alert
        alert = Alert(
            id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fingerprint.event_id[:8]}",
            severity=severity,
            title=f"Pattern Match: {best_match.match_type.value.upper()}",
            message=self._generate_alert_message(best_match, region),
            match_result=best_match,
            confidence=best_match.similarity,
            region=region,
            bbox=fingerprint.bbox,
            recommended_actions=self._get_recommended_actions(severity),
            similar_historical_events=[m.matched_id for m in matches[:5]],
        )
        
        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        return alert
    
    def _classify_match(self, similarity: float) -> MatchType:
        """Classify match type based on similarity."""
        if similarity >= self.config.exact_threshold:
            return MatchType.EXACT
        elif similarity >= self.config.strong_threshold:
            return MatchType.STRONG
        elif similarity >= self.config.moderate_threshold:
            return MatchType.MODERATE
        elif similarity >= self.config.weak_threshold:
            return MatchType.WEAK
        else:
            return MatchType.PARTIAL
    
    def _generate_alert_message(
        self,
        match: MatchResult,
        region: str = None,
    ) -> str:
        """Generate human-readable alert message."""
        region_str = f" in {region}" if region else ""
        
        if match.match_type == MatchType.EXACT:
            return (
                f"🚨 EXACT pattern match detected{region_str}! "
                f"Current conditions match historical event '{match.matched_id}' "
                f"with {match.similarity*100:.1f}% similarity. "
                f"Immediate attention recommended."
            )
        elif match.match_type == MatchType.STRONG:
            return (
                f"⚠️ Strong pattern match{region_str}. "
                f"Similarity to '{match.matched_id}': {match.similarity*100:.1f}%. "
                f"Monitor situation closely."
            )
        elif match.match_type == MatchType.MODERATE:
            return (
                f"📊 Moderate pattern match{region_str}. "
                f"Some similarity to '{match.matched_id}' ({match.similarity*100:.1f}%). "
                f"Continue monitoring."
            )
        else:
            return (
                f"ℹ️ Weak pattern correlation{region_str} "
                f"with '{match.matched_id}' ({match.similarity*100:.1f}%)."
            )
    
    def _get_recommended_actions(self, severity: AlertSeverity) -> List[str]:
        """Get recommended actions based on severity."""
        if severity == AlertSeverity.CRITICAL:
            return [
                "Review current meteorological data immediately",
                "Check satellite imagery for the region",
                "Alert emergency response teams",
                "Prepare public warning communications",
            ]
        elif severity == AlertSeverity.HIGH:
            return [
                "Monitor situation with increased frequency",
                "Cross-reference with other data sources",
                "Prepare contingency plans",
                "Brief relevant stakeholders",
            ]
        elif severity == AlertSeverity.MEDIUM:
            return [
                "Continue standard monitoring",
                "Document current conditions",
                "Review historical event progression",
            ]
        else:
            return [
                "Log for reference",
                "Continue routine monitoring",
            ]
    
    # =========================================================================
    # MONITORING
    # =========================================================================
    
    async def monitor(
        self,
        data_source: AsyncIterator[pd.DataFrame],
        event_id_prefix: str = "realtime",
        region: str = None,
    ) -> AsyncIterator[Alert]:
        """
        Monitor a data stream for pattern matches.
        
        Args:
            data_source: Async iterator yielding DataFrames
            event_id_prefix: Prefix for generated event IDs
            region: Region name for alerts
            
        Yields:
            Alert objects when matches are found
        """
        await self.initialize()
        
        self._monitoring = True
        counter = 0
        
        try:
            async for df in data_source:
                if not self._monitoring:
                    break
                
                counter += 1
                event_id = f"{event_id_prefix}_{counter:06d}"
                
                try:
                    # Extract fingerprint
                    fp = self._engine.extract(df, event_id)
                    
                    # Match and alert
                    alert = await self.match_and_alert(fp, region=region)
                    
                    if alert:
                        yield alert
                        
                except Exception as e:
                    logger.warning(f"Monitor iteration error: {e}")
                
                # Rate limiting
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
        finally:
            self._monitoring = False
    
    async def start_monitoring(
        self,
        data_source: AsyncIterator[pd.DataFrame],
        **kwargs,
    ) -> None:
        """Start background monitoring task."""
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Monitoring already running")
            return
        
        async def _monitor_loop():
            async for alert in self.monitor(data_source, **kwargs):
                logger.info(f"Alert generated: {alert.id} ({alert.severity.value})")
        
        self._monitor_task = asyncio.create_task(_monitor_loop())
        logger.info("🔄 Background monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("⏹️ Monitoring stopped")
    
    # =========================================================================
    # ALERT HANDLERS
    # =========================================================================
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add callback for alerts."""
        self._alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Remove alert callback."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    async def batch_match(
        self,
        fingerprints: List[Fingerprint],
    ) -> Dict[str, List[MatchResult]]:
        """
        Match multiple fingerprints.
        
        Args:
            fingerprints: List of fingerprints to match
            
        Returns:
            Dict mapping event_id to list of matches
        """
        results = {}
        
        for fp in fingerprints:
            matches = await self.match(fp)
            results[fp.event_id] = matches
        
        return results
    
    async def find_precursors(
        self,
        target_event_id: str,
        time_window_days: int = 30,
    ) -> List[MatchResult]:
        """
        Find potential precursor patterns for a target event.
        
        Args:
            target_event_id: Event to find precursors for
            time_window_days: Look back window
            
        Returns:
            List of potential precursor matches
        """
        await self.initialize()
        
        # Get target fingerprint
        target_fp = await self._db.get(target_event_id)
        if not target_fp:
            return []
        
        # Find similar patterns
        matches = await self.match(target_fp)
        
        # Filter by lag (patterns that occurred before similar events)
        precursors = []
        for match in matches:
            if match.lag_to_event_days is not None:
                if 0 < match.lag_to_event_days <= time_window_days:
                    precursors.append(match)
        
        return precursors


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_matcher: Optional[FingerprintMatcher] = None


async def get_matcher() -> FingerprintMatcher:
    """Get default matcher instance."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = FingerprintMatcher()
        await _default_matcher.initialize()
    return _default_matcher


async def quick_match(fingerprint: Fingerprint) -> List[MatchResult]:
    """Quick fingerprint match."""
    matcher = await get_matcher()
    return await matcher.match(fingerprint)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("=== 🔍 Fingerprint Matcher Test ===\n")
        
        # Create matcher
        matcher = FingerprintMatcher()
        await matcher.initialize()
        
        # Create test fingerprints
        engine = FingerprintEngine(n_kernels=1000, embedding_dim=50)
        
        np.random.seed(42)
        dates = pd.date_range("2000-10-01", periods=100, freq="D")
        
        # Historical flood pattern
        df_flood = pd.DataFrame({
            "precipitation": np.random.exponential(5, 100) + np.linspace(0, 20, 100),
            "pressure": 101325 - np.random.exponential(1000, 100),
            "nao": np.random.normal(-0.5, 1, 100),
        }, index=dates)
        
        fp_flood = engine.extract(df_flood, "lago_maggiore_2000", sources=["era5"])
        
        # Store in database
        await matcher._db.store(fp_flood)
        
        # Current observation (similar to flood)
        df_current = df_flood + np.random.normal(0, 0.5, df_flood.shape)
        fp_current = engine.extract(df_current, "current_observation", sources=["era5"])
        
        # Match
        print("🔍 Matching current observation...")
        matches = await matcher.match(fp_current)
        
        for m in matches:
            print(f"   {m}")
        
        # Alert
        print("\n🚨 Checking for alert...")
        alert = await matcher.match_and_alert(fp_current, region="Northern Italy")
        
        if alert:
            print(f"   Alert: {alert.title}")
            print(f"   Severity: {alert.severity.value}")
            print(f"   Message: {alert.message}")
        else:
            print("   No alert generated")
        
        await matcher.close()
        print("\n✅ Test complete!")
    
    asyncio.run(test())

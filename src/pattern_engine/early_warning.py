"""
🚨 Early Warning Alert System
=============================

Generates early warning alerts based on learned causal patterns.
When precursor conditions are detected, issues alerts for potential events.

Alert Levels:
- 🟢 GREEN: Normal conditions, no precursors detected
- 🟡 YELLOW: Some precursors active, elevated risk
- 🟠 ORANGE: Multiple precursors active, high probability
- 🔴 RED: Strong precursor match, event likely imminent

Features:
- Real-time precursor monitoring
- Multi-region alert aggregation
- Confidence-weighted scoring
- Alert history and validation
- Integration with PCMCI patterns

Usage:
    alert_system = EarlyWarningSystem()
    
    # Configure pattern to monitor
    alert_system.add_pattern(
        pattern_id="nao_flood_pattern",
        precursors={"NAO": {"threshold": -1.5, "lag": 7}},
        effect="flood",
        region="Lake Maggiore"
    )
    
    # Check current conditions
    alert = await alert_system.check_conditions({
        "NAO": -2.1,
        "SST_anomaly": 1.5,
    })
    
    print(f"Alert Level: {alert.level}")
    print(f"Probability: {alert.probability}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import math


class AlertLevel(Enum):
    """Alert severity levels."""
    GREEN = "green"      # Normal
    YELLOW = "yellow"    # Watch
    ORANGE = "orange"    # Warning
    RED = "red"          # Critical
    
    @property
    def severity(self) -> int:
        """Numeric severity for comparison."""
        return {
            AlertLevel.GREEN: 0,
            AlertLevel.YELLOW: 1,
            AlertLevel.ORANGE: 2,
            AlertLevel.RED: 3,
        }[self]
    
    @property
    def emoji(self) -> str:
        return {
            AlertLevel.GREEN: "🟢",
            AlertLevel.YELLOW: "🟡",
            AlertLevel.ORANGE: "🟠",
            AlertLevel.RED: "🔴",
        }[self]
    
    @property
    def description(self) -> str:
        return {
            AlertLevel.GREEN: "Normal conditions",
            AlertLevel.YELLOW: "Elevated risk - monitor closely",
            AlertLevel.ORANGE: "High risk - prepare response",
            AlertLevel.RED: "Critical - event imminent",
        }[self]


class ThresholdType(Enum):
    """Type of threshold comparison."""
    ABOVE = "above"       # Trigger when value > threshold
    BELOW = "below"       # Trigger when value < threshold
    DEVIATION = "deviation"  # Trigger when |value - mean| > threshold


@dataclass
class Precursor:
    """A precursor condition to monitor."""
    variable: str
    threshold: float
    threshold_type: ThresholdType
    lag_days: int  # How many days before effect
    weight: float = 1.0  # Importance weight
    description: str = ""
    
    def is_triggered(self, value: float, baseline: float = 0) -> bool:
        """Check if precursor condition is triggered."""
        if self.threshold_type == ThresholdType.ABOVE:
            return value > self.threshold
        elif self.threshold_type == ThresholdType.BELOW:
            return value < self.threshold
        elif self.threshold_type == ThresholdType.DEVIATION:
            return abs(value - baseline) > self.threshold
        return False
    
    def trigger_strength(self, value: float, baseline: float = 0) -> float:
        """Calculate how strongly the precursor is triggered (0-1)."""
        if self.threshold_type == ThresholdType.ABOVE:
            if value <= self.threshold:
                return 0
            # How far above threshold
            excess = value - self.threshold
            return min(1.0, excess / abs(self.threshold) if self.threshold != 0 else excess)
        
        elif self.threshold_type == ThresholdType.BELOW:
            if value >= self.threshold:
                return 0
            excess = self.threshold - value
            return min(1.0, excess / abs(self.threshold) if self.threshold != 0 else excess)
        
        elif self.threshold_type == ThresholdType.DEVIATION:
            deviation = abs(value - baseline)
            if deviation <= self.threshold:
                return 0
            excess = deviation - self.threshold
            return min(1.0, excess / self.threshold if self.threshold != 0 else excess)
        
        return 0
    
    def to_dict(self) -> Dict:
        return {
            "variable": self.variable,
            "threshold": self.threshold,
            "threshold_type": self.threshold_type.value,
            "lag_days": self.lag_days,
            "weight": self.weight,
            "description": self.description,
        }


@dataclass
class AlertPattern:
    """A pattern to monitor for early warning."""
    id: str
    name: str
    precursors: List[Precursor]
    effect: str
    region: str
    base_probability: float = 0.1  # Background probability
    confidence: float = 0.5  # How confident we are in pattern
    seasonality: Optional[str] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_probability(
        self,
        current_values: Dict[str, float],
        baselines: Dict[str, float] = None,
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate event probability based on current precursor values.
        
        Returns (probability, list of triggered precursors)
        """
        baselines = baselines or {}
        triggered = []
        total_weight = sum(p.weight for p in self.precursors)
        weighted_strength = 0
        
        for precursor in self.precursors:
            if precursor.variable not in current_values:
                continue
            
            value = current_values[precursor.variable]
            baseline = baselines.get(precursor.variable, 0)
            
            if precursor.is_triggered(value, baseline):
                strength = precursor.trigger_strength(value, baseline)
                weighted_strength += precursor.weight * strength
                
                triggered.append({
                    "variable": precursor.variable,
                    "value": value,
                    "threshold": precursor.threshold,
                    "strength": strength,
                    "lag_days": precursor.lag_days,
                })
        
        # Convert weighted strength to probability
        # Using logistic function
        if total_weight > 0:
            normalized_strength = weighted_strength / total_weight
        else:
            normalized_strength = 0
        
        # Probability increases from base to ~1 as strength increases
        # P = base + (1 - base) * (1 - e^(-k*strength))
        k = 3  # Steepness
        probability = self.base_probability + (1 - self.base_probability) * (
            1 - math.exp(-k * normalized_strength)
        )
        
        # Adjust by pattern confidence
        probability = self.base_probability + (probability - self.base_probability) * self.confidence
        
        return float(probability), triggered
    
    def get_alert_level(self, probability: float) -> AlertLevel:
        """Determine alert level from probability."""
        if probability >= 0.75:
            return AlertLevel.RED
        elif probability >= 0.50:
            return AlertLevel.ORANGE
        elif probability >= 0.25:
            return AlertLevel.YELLOW
        else:
            return AlertLevel.GREEN
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "precursors": [p.to_dict() for p in self.precursors],
            "effect": self.effect,
            "region": self.region,
            "base_probability": self.base_probability,
            "confidence": self.confidence,
            "seasonality": self.seasonality,
            "active": self.active,
        }


@dataclass
class Alert:
    """An early warning alert."""
    id: str
    pattern_id: str
    region: str
    effect: str
    level: AlertLevel
    probability: float
    triggered_precursors: List[Dict]
    estimated_time_to_event: Optional[int]  # Days
    message: str
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    expires: Optional[str] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "pattern_id": self.pattern_id,
            "region": self.region,
            "effect": self.effect,
            "level": self.level.value,
            "level_emoji": self.level.emoji,
            "probability": self.probability,
            "triggered_precursors": self.triggered_precursors,
            "estimated_time_to_event": self.estimated_time_to_event,
            "message": self.message,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
            "expires": self.expires,
            "acknowledged": self.acknowledged,
        }
    
    def to_notification(self) -> Dict:
        """Format as notification for UI/webhook."""
        return {
            "title": f"{self.level.emoji} {self.effect.title()} Alert - {self.region}",
            "body": self.message,
            "level": self.level.value,
            "region": self.region,
            "probability": f"{self.probability*100:.0f}%",
            "time_to_event": f"{self.estimated_time_to_event} days" if self.estimated_time_to_event else "Unknown",
            "timestamp": self.timestamp,
        }


@dataclass
class AlertHistory:
    """Historical record of alerts."""
    alerts: List[Alert] = field(default_factory=list)
    validated: Dict[str, bool] = field(default_factory=dict)  # alert_id -> did_event_occur
    
    def add(self, alert: Alert) -> None:
        self.alerts.append(alert)
    
    def validate(self, alert_id: str, event_occurred: bool) -> None:
        """Record whether the predicted event actually occurred."""
        self.validated[alert_id] = event_occurred
    
    def get_accuracy(self, level: AlertLevel = None) -> Dict[str, float]:
        """Calculate alert accuracy statistics."""
        if not self.validated:
            return {"accuracy": None, "total": 0}
        
        filtered = [
            (a, self.validated[a.id])
            for a in self.alerts
            if a.id in self.validated and (level is None or a.level == level)
        ]
        
        if not filtered:
            return {"accuracy": None, "total": 0}
        
        true_positives = sum(1 for a, v in filtered if v and a.probability >= 0.5)
        true_negatives = sum(1 for a, v in filtered if not v and a.probability < 0.5)
        total = len(filtered)
        
        return {
            "accuracy": (true_positives + true_negatives) / total,
            "true_positives": true_positives,
            "total_validated": total,
            "events_occurred": sum(1 for _, v in filtered if v),
        }


class EarlyWarningSystem:
    """
    Complete early warning system for monitoring and alerting.
    """
    
    def __init__(self):
        self.patterns: Dict[str, AlertPattern] = {}
        self.history = AlertHistory()
        self._alert_counter = 0
        self._callbacks: List[Callable[[Alert], None]] = []
    
    def add_pattern(
        self,
        pattern_id: str,
        name: str,
        precursors: List[Dict],
        effect: str,
        region: str,
        confidence: float = 0.5,
        base_probability: float = 0.1,
    ) -> AlertPattern:
        """
        Add a pattern to monitor.
        
        Args:
            pattern_id: Unique identifier
            name: Human-readable name
            precursors: List of precursor configs:
                [{variable, threshold, threshold_type, lag_days, weight}, ...]
            effect: The effect to predict (e.g., "flood")
            region: Geographic region
            confidence: How confident we are (from PCMCI validation)
            base_probability: Background probability of event
        """
        prec_list = []
        for p in precursors:
            prec = Precursor(
                variable=p["variable"],
                threshold=p["threshold"],
                threshold_type=ThresholdType(p.get("threshold_type", "above")),
                lag_days=p.get("lag_days", 7),
                weight=p.get("weight", 1.0),
                description=p.get("description", ""),
            )
            prec_list.append(prec)
        
        pattern = AlertPattern(
            id=pattern_id,
            name=name,
            precursors=prec_list,
            effect=effect,
            region=region,
            confidence=confidence,
            base_probability=base_probability,
        )
        
        self.patterns[pattern_id] = pattern
        return pattern
    
    def add_pattern_from_pcmci(
        self,
        pattern_id: str,
        name: str,
        pcmci_links: List[Dict],
        effect: str,
        region: str,
        historical_data: Any = None,  # DataFrame for threshold estimation
    ) -> AlertPattern:
        """
        Create alert pattern from PCMCI causal links.
        
        Args:
            pcmci_links: Links from PCMCIResult.significant_links
            historical_data: Optional DataFrame to estimate thresholds
        """
        precursors = []
        
        for link in pcmci_links:
            if link.get("target", link.get("effect")) != effect:
                continue
            
            source = link.get("source", link.get("cause"))
            lag = link.get("lag", 0)
            score = link.get("score", link.get("confidence", 0.5))
            strength = link.get("strength", 0)
            
            # Determine threshold type and value from strength sign
            if historical_data is not None and source in historical_data.columns:
                # Use percentile-based threshold
                values = historical_data[source].dropna()
                if strength > 0:
                    # High values are precursors
                    threshold = float(values.quantile(0.9))
                    threshold_type = "above"
                else:
                    # Low values are precursors
                    threshold = float(values.quantile(0.1))
                    threshold_type = "below"
            else:
                # Default thresholds
                if strength > 0:
                    threshold = 0.5  # Above 0.5 standard deviations
                    threshold_type = "above"
                else:
                    threshold = -0.5
                    threshold_type = "below"
            
            precursors.append({
                "variable": source,
                "threshold": threshold,
                "threshold_type": threshold_type,
                "lag_days": lag,
                "weight": score,
            })
        
        return self.add_pattern(
            pattern_id=pattern_id,
            name=name,
            precursors=precursors,
            effect=effect,
            region=region,
            confidence=sum(l.get("score", 0.5) for l in pcmci_links) / len(pcmci_links) if pcmci_links else 0.5,
        )
    
    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register callback for new alerts (e.g., for webhooks)."""
        self._callbacks.append(callback)
    
    def check_conditions(
        self,
        current_values: Dict[str, float],
        baselines: Dict[str, float] = None,
        pattern_ids: List[str] = None,
    ) -> List[Alert]:
        """
        Check current conditions against all (or specified) patterns.
        
        Args:
            current_values: Current values of monitored variables
            baselines: Baseline/climatological values for deviation checks
            pattern_ids: Specific patterns to check (all if None)
            
        Returns:
            List of alerts (including GREEN level)
        """
        alerts = []
        patterns_to_check = pattern_ids or list(self.patterns.keys())
        
        for pattern_id in patterns_to_check:
            if pattern_id not in self.patterns:
                continue
            
            pattern = self.patterns[pattern_id]
            if not pattern.active:
                continue
            
            # Calculate probability
            probability, triggered = pattern.calculate_probability(
                current_values, baselines
            )
            
            # Determine alert level
            level = pattern.get_alert_level(probability)
            
            # Estimate time to event from lag
            if triggered:
                min_lag = min(t["lag_days"] for t in triggered)
            else:
                min_lag = None
            
            # Generate message
            message = self._generate_message(pattern, level, probability, triggered)
            recommendations = self._generate_recommendations(pattern, level, triggered)
            
            # Create alert
            self._alert_counter += 1
            alert = Alert(
                id=f"ALT_{self._alert_counter:06d}",
                pattern_id=pattern_id,
                region=pattern.region,
                effect=pattern.effect,
                level=level,
                probability=probability,
                triggered_precursors=triggered,
                estimated_time_to_event=min_lag,
                message=message,
                recommendations=recommendations,
                expires=(datetime.now() + timedelta(days=min_lag if min_lag else 7)).isoformat(),
            )
            
            alerts.append(alert)
            
            # Trigger callbacks for non-green alerts
            if level != AlertLevel.GREEN:
                self.history.add(alert)
                for callback in self._callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Callback error: {e}")
        
        return alerts
    
    def _generate_message(
        self,
        pattern: AlertPattern,
        level: AlertLevel,
        probability: float,
        triggered: List[Dict],
    ) -> str:
        """Generate human-readable alert message."""
        if level == AlertLevel.GREEN:
            return f"No {pattern.effect} precursors detected for {pattern.region}. Conditions normal."
        
        precursor_names = [t["variable"] for t in triggered]
        n_triggered = len(triggered)
        
        if level == AlertLevel.RED:
            msg = f"CRITICAL: High probability ({probability*100:.0f}%) of {pattern.effect} event in {pattern.region}."
        elif level == AlertLevel.ORANGE:
            msg = f"WARNING: Elevated probability ({probability*100:.0f}%) of {pattern.effect} event in {pattern.region}."
        else:
            msg = f"WATCH: Some {pattern.effect} precursors active in {pattern.region} ({probability*100:.0f}% probability)."
        
        msg += f" {n_triggered} precursor(s) triggered: {', '.join(precursor_names[:3])}"
        
        if triggered:
            min_lag = min(t["lag_days"] for t in triggered)
            if min_lag > 0:
                msg += f". Event possible in ~{min_lag} days."
        
        return msg
    
    def _generate_recommendations(
        self,
        pattern: AlertPattern,
        level: AlertLevel,
        triggered: List[Dict],
    ) -> List[str]:
        """Generate action recommendations based on alert level."""
        recs = []
        
        if level == AlertLevel.GREEN:
            recs.append("Continue normal monitoring")
            return recs
        
        if level == AlertLevel.YELLOW:
            recs.append("Increase monitoring frequency")
            recs.append("Review emergency response plans")
        
        if level == AlertLevel.ORANGE:
            recs.append("Alert relevant authorities")
            recs.append("Prepare emergency response teams")
            recs.append("Issue public advisory")
        
        if level == AlertLevel.RED:
            recs.append("ACTIVATE emergency response")
            recs.append("Issue public warnings")
            recs.append("Begin evacuation preparations if needed")
            recs.append("Coordinate with emergency services")
        
        # Pattern-specific recommendations
        effect = pattern.effect.lower()
        if "flood" in effect:
            recs.append("Check flood barriers and drainage")
            if level.severity >= AlertLevel.ORANGE.severity:
                recs.append("Prepare sandbags and pumping equipment")
        
        if "storm" in effect or "surge" in effect:
            recs.append("Secure loose objects")
            recs.append("Check coastal defenses")
        
        return recs
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of all monitored patterns."""
        return {
            "patterns_monitored": len(self.patterns),
            "patterns_active": sum(1 for p in self.patterns.values() if p.active),
            "patterns": [p.to_dict() for p in self.patterns.values()],
            "recent_alerts": [a.to_dict() for a in self.history.alerts[-10:]],
            "accuracy": self.history.get_accuracy(),
        }
    
    def export_config(self) -> str:
        """Export configuration as JSON."""
        return json.dumps({
            "patterns": [p.to_dict() for p in self.patterns.values()],
        }, indent=2)
    
    def import_config(self, config_json: str) -> int:
        """Import configuration from JSON. Returns number of patterns added."""
        config = json.loads(config_json)
        count = 0
        
        for p in config.get("patterns", []):
            self.add_pattern(
                pattern_id=p["id"],
                name=p["name"],
                precursors=p["precursors"],
                effect=p["effect"],
                region=p["region"],
                confidence=p.get("confidence", 0.5),
                base_probability=p.get("base_probability", 0.1),
            )
            count += 1
        
        return count


# Pre-configured patterns for common events
def create_default_patterns() -> EarlyWarningSystem:
    """Create system with default patterns for common climate events."""
    system = EarlyWarningSystem()
    
    # NAO-driven flooding in Alpine lakes
    system.add_pattern(
        pattern_id="alpine_nao_flood",
        name="NAO-driven Alpine Flooding",
        precursors=[
            {"variable": "NAO_index", "threshold": -1.5, "threshold_type": "below", "lag_days": 7, "weight": 0.9},
            {"variable": "SST_med", "threshold": 1.0, "threshold_type": "above", "lag_days": 5, "weight": 0.6},
            {"variable": "precipitation_anomaly", "threshold": 50, "threshold_type": "above", "lag_days": 2, "weight": 0.8},
        ],
        effect="flood",
        region="Alpine Lakes",
        confidence=0.75,
    )
    
    # Adriatic storm surge (Venice)
    system.add_pattern(
        pattern_id="adriatic_surge",
        name="Adriatic Storm Surge",
        precursors=[
            {"variable": "pressure_adriatic", "threshold": 1000, "threshold_type": "below", "lag_days": 2, "weight": 0.85},
            {"variable": "wind_sirocco", "threshold": 15, "threshold_type": "above", "lag_days": 1, "weight": 0.9},
            {"variable": "sea_level_anomaly", "threshold": 0.5, "threshold_type": "above", "lag_days": 1, "weight": 0.7},
        ],
        effect="storm_surge",
        region="Venice",
        confidence=0.8,
    )
    
    # North Sea storm surge
    system.add_pattern(
        pattern_id="north_sea_surge",
        name="North Sea Storm Surge",
        precursors=[
            {"variable": "NAO_index", "threshold": 2.0, "threshold_type": "above", "lag_days": 3, "weight": 0.7},
            {"variable": "wind_speed_north_sea", "threshold": 25, "threshold_type": "above", "lag_days": 1, "weight": 0.9},
            {"variable": "pressure_atlantic", "threshold": 980, "threshold_type": "below", "lag_days": 2, "weight": 0.8},
        ],
        effect="storm_surge",
        region="Netherlands/Germany Coast",
        confidence=0.7,
    )
    
    return system


# CLI test
if __name__ == "__main__":
    print("=== Early Warning Alert System Test ===\n")
    
    # Create system with default patterns
    system = create_default_patterns()
    
    print(f"Loaded {len(system.patterns)} patterns:\n")
    for p in system.patterns.values():
        print(f"  📋 {p.name} ({p.region})")
        print(f"     Effect: {p.effect}")
        print(f"     Precursors: {len(p.precursors)}")
        print(f"     Confidence: {p.confidence:.0%}")
        print()
    
    # Test with current conditions
    print("=== Testing with sample conditions ===\n")
    
    current = {
        "NAO_index": -2.3,  # Strong negative NAO
        "SST_med": 1.2,     # Warm Mediterranean
        "precipitation_anomaly": 80,  # High precipitation
        "pressure_adriatic": 998,
        "wind_sirocco": 12,
        "sea_level_anomaly": 0.3,
    }
    
    print("Current conditions:")
    for var, val in current.items():
        print(f"  {var}: {val}")
    print()
    
    alerts = system.check_conditions(current)
    
    print("Alerts generated:\n")
    for alert in alerts:
        print(f"{alert.level.emoji} [{alert.region}] {alert.effect.upper()}")
        print(f"   Level: {alert.level.value}")
        print(f"   Probability: {alert.probability*100:.0f}%")
        print(f"   Message: {alert.message}")
        if alert.triggered_precursors:
            print(f"   Triggered: {[t['variable'] for t in alert.triggered_precursors]}")
        if alert.recommendations:
            print(f"   Recommendations:")
            for rec in alert.recommendations[:3]:
                print(f"     • {rec}")
        print()

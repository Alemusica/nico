"""
Pipeline gates for staged prediction with confidence thresholds.

"Quarti di finale" concept: Confidence grows through stages.
When we reach the "finals" (high combined confidence), event is imminent.

Stage 1: Fingerprint match > 60% → Activate GNN
Stage 2: Ensemble confidence > 70% → Request physics check  
Stage 3: Physics residual < 0.05 → Pattern validated
Stage 3b: Physics weak but historical > 80% → Gray zone (show both views)
Final: Combined > 80% → ALERT

The system PROPOSES, the human DECIDES.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

from ..core.constants import (
    FINGERPRINT_GATE,
    ENSEMBLE_GATE,
    PHYSICS_RESIDUAL_THRESHOLD,
    ALERT_THRESHOLD,
    GRAY_ZONE_HISTORICAL_MIN,
)


class Stage(Enum):
    """Pipeline stages."""
    IDLE = "idle"
    FINGERPRINT = "fingerprint"
    GNN_PREDICTION = "gnn_prediction"
    PHYSICS_CHECK = "physics_check"
    ENSEMBLE = "ensemble"
    FINAL = "final"


class AlertLevel(Enum):
    """Alert levels for output."""
    NONE = "none"
    GREEN = "green"       # Low probability
    YELLOW = "yellow"     # Monitor
    ORANGE = "orange"     # Prepare
    RED = "red"           # Imminent


@dataclass
class GateResult:
    """Result of passing through a gate."""
    passed: bool
    confidence: float
    next_stage: Stage
    message: str
    metadata: dict[str, Any] | None = None


@dataclass
class PipelineState:
    """Current state of the prediction pipeline."""
    current_stage: Stage = Stage.IDLE
    
    # Confidence scores from each stage
    fingerprint_confidence: float = 0.0
    gnn_confidence: float = 0.0
    physics_residual: float = 1.0  # High = bad
    ensemble_confidence: float = 0.0
    
    # Predictions
    predicted_surge_m: float | None = None
    predicted_surge_uncertainty: float | None = None
    predicted_location: str | None = None
    predicted_time_hours: float | None = None
    
    # Gray zone flag
    is_gray_zone: bool = False
    gray_zone_reason: str = ""
    
    @property
    def historical_confidence(self) -> float:
        """Combined historical confidence (fingerprint + pattern stats)."""
        return self.fingerprint_confidence
    
    @property
    def physics_confidence(self) -> float:
        """Convert residual to confidence (0-1, higher = better)."""
        return max(0.0, 1.0 - self.physics_residual / 0.1)
    
    @property
    def combined_confidence(self) -> float:
        """
        Final combined confidence.
        
        If gray zone: weight historical more (0.7/0.3)
        If validated: equal weight (0.5/0.5)
        """
        if self.is_gray_zone:
            return 0.7 * self.historical_confidence + 0.3 * self.physics_confidence
        return 0.5 * self.historical_confidence + 0.5 * self.physics_confidence
    
    @property
    def alert_level(self) -> AlertLevel:
        """Determine alert level from combined confidence."""
        conf = self.combined_confidence
        if conf >= ALERT_THRESHOLD:
            return AlertLevel.RED
        elif conf >= ENSEMBLE_GATE:
            return AlertLevel.ORANGE
        elif conf >= FINGERPRINT_GATE:
            return AlertLevel.YELLOW
        elif conf > 0.3:
            return AlertLevel.GREEN
        return AlertLevel.NONE


class Gate:
    """
    A single gate in the pipeline.
    
    Gates check conditions and decide whether to proceed to next stage.
    """
    
    def __init__(
        self,
        name: str,
        threshold: float,
        next_stage_if_pass: Stage,
        next_stage_if_fail: Stage,
        check_function: Callable[[PipelineState], float] | None = None
    ):
        self.name = name
        self.threshold = threshold
        self.next_stage_if_pass = next_stage_if_pass
        self.next_stage_if_fail = next_stage_if_fail
        self.check_function = check_function
    
    def evaluate(self, state: PipelineState) -> GateResult:
        """Evaluate whether state passes this gate."""
        if self.check_function:
            value = self.check_function(state)
        else:
            value = 0.0
        
        passed = value >= self.threshold
        
        return GateResult(
            passed=passed,
            confidence=value,
            next_stage=self.next_stage_if_pass if passed else self.next_stage_if_fail,
            message=f"{self.name}: {'PASS' if passed else 'FAIL'} ({value:.2f} vs {self.threshold:.2f})"
        )


class PipelineGates:
    """
    Collection of all pipeline gates.
    
    Orchestrates the staged prediction flow.
    """
    
    def __init__(self):
        self.gates = {
            Stage.FINGERPRINT: Gate(
                name="Fingerprint Match",
                threshold=FINGERPRINT_GATE,
                next_stage_if_pass=Stage.GNN_PREDICTION,
                next_stage_if_fail=Stage.IDLE,
                check_function=lambda s: s.fingerprint_confidence
            ),
            Stage.GNN_PREDICTION: Gate(
                name="GNN Ensemble",
                threshold=ENSEMBLE_GATE,
                next_stage_if_pass=Stage.PHYSICS_CHECK,
                next_stage_if_fail=Stage.IDLE,
                check_function=lambda s: s.gnn_confidence
            ),
            Stage.PHYSICS_CHECK: Gate(
                name="Physics Validation",
                threshold=1.0 - PHYSICS_RESIDUAL_THRESHOLD,  # Inverted (low residual = pass)
                next_stage_if_pass=Stage.ENSEMBLE,
                next_stage_if_fail=Stage.ENSEMBLE,  # Still proceed, but mark gray zone
                check_function=lambda s: s.physics_confidence
            ),
            Stage.ENSEMBLE: Gate(
                name="Final Confidence",
                threshold=ALERT_THRESHOLD,
                next_stage_if_pass=Stage.FINAL,
                next_stage_if_fail=Stage.IDLE,
                check_function=lambda s: s.combined_confidence
            ),
        }
    
    def process_stage(self, state: PipelineState) -> tuple[PipelineState, GateResult]:
        """
        Process current stage and update state.
        
        Returns updated state and gate result.
        """
        current = state.current_stage
        
        if current not in self.gates:
            return state, GateResult(
                passed=False,
                confidence=0.0,
                next_stage=Stage.IDLE,
                message=f"No gate for stage {current}"
            )
        
        gate = self.gates[current]
        result = gate.evaluate(state)
        
        # Special handling for physics check → gray zone
        if current == Stage.PHYSICS_CHECK and not result.passed:
            # Check if historical is strong enough for gray zone
            if state.historical_confidence >= GRAY_ZONE_HISTORICAL_MIN:
                state.is_gray_zone = True
                state.gray_zone_reason = "Strong historical pattern, physics not fully validated"
                result.message += " → GRAY ZONE (historical strong)"
        
        state.current_stage = result.next_stage
        return state, result
    
    def run_full_pipeline(
        self,
        initial_state: PipelineState,
        stage_callbacks: dict[Stage, Callable[[PipelineState], PipelineState]] | None = None
    ) -> tuple[PipelineState, list[GateResult]]:
        """
        Run full pipeline from initial state.
        
        Args:
            initial_state: Starting state with fingerprint confidence set
            stage_callbacks: Optional callbacks to run at each stage (e.g., GNN inference)
            
        Returns:
            Final state and list of gate results
        """
        state = initial_state
        results = []
        
        # Start at fingerprint stage
        state.current_stage = Stage.FINGERPRINT
        
        max_iterations = 10  # Safety limit
        for _ in range(max_iterations):
            if state.current_stage in (Stage.IDLE, Stage.FINAL):
                break
            
            # Run stage callback if provided (e.g., run GNN)
            if stage_callbacks and state.current_stage in stage_callbacks:
                state = stage_callbacks[state.current_stage](state)
            
            # Evaluate gate
            state, result = self.process_stage(state)
            results.append(result)
        
        return state, results


def get_cockpit_output(state: PipelineState) -> dict[str, Any]:
    """
    Generate output for cockpit display.
    
    Shows BOTH experience (historical) and science (physics) views.
    Human makes final decision.
    """
    return {
        # Experience view (left panel)
        "experience": {
            "pattern_match_pct": state.fingerprint_confidence * 100,
            "historical_probability_pct": state.historical_confidence * 100,
            "confidence_bar": "=" * int(state.historical_confidence * 20),
        },
        
        # Science view (right panel)
        "science": {
            "physics_residual": state.physics_residual,
            "physics_status": "OK" if state.physics_residual < PHYSICS_RESIDUAL_THRESHOLD else "WEAK",
            "swe_constraint": "OK" if state.physics_residual < 0.1 else "VIOLATED",
            "physics_probability_pct": state.physics_confidence * 100,
            "confidence_bar": "=" * int(state.physics_confidence * 20),
        },
        
        # Combined output
        "combined": {
            "predicted_surge_m": state.predicted_surge_m,
            "uncertainty_m": state.predicted_surge_uncertainty,
            "location": state.predicted_location,
            "time_hours": state.predicted_time_hours,
            "combined_confidence_pct": state.combined_confidence * 100,
            "alert_level": state.alert_level.value,
            "is_gray_zone": state.is_gray_zone,
            "gray_zone_reason": state.gray_zone_reason,
        },
        
        # Recommendation (system proposes, human decides)
        "recommendation": _get_recommendation(state),
    }


def _get_recommendation(state: PipelineState) -> str:
    """Generate human-readable recommendation."""
    alert = state.alert_level
    
    if alert == AlertLevel.RED:
        return "⚠️ ALLERTA ROSSA: Surge imminente. Attivare procedure emergenza."
    elif alert == AlertLevel.ORANGE:
        if state.is_gray_zone:
            return "🟠 ALLERTA ARANCIONE (GRAY ZONE): Pattern storico forte, fisica non completamente validata. Monitorare attentamente."
        return "🟠 ALLERTA ARANCIONE: Surge probabile. Preparare risorse."
    elif alert == AlertLevel.YELLOW:
        return "🟡 ATTENZIONE: Pattern rilevato. Continuare monitoraggio."
    elif alert == AlertLevel.GREEN:
        return "🟢 BASSO RISCHIO: Situazione normale."
    return "⚪ NESSUN PATTERN RILEVATO"

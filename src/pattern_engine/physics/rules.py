"""
Physics Rules Module
====================

Configurable physics validation for domain-specific pattern detection.
Filters spurious correlations by checking if patterns make physical sense.

Domains supported:
- Flood/Storm Surge: wind_setup, pressure_effect, tide_interaction
- Manufacturing: temperature_effect, speed_effect, material_properties
- Energy/Load: thermal_demand, solar_effect

Usage:
    validator = PhysicsValidator(domain='flood')
    validated_df = validator.apply(features_df, rules=['wind_setup', 'pressure_effect'])
    gray_patterns = validator.get_gray_zone()
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Supported domains for physics validation."""
    FLOOD = "flood"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    GENERIC = "generic"


@dataclass
class PhysicsRule:
    """
    A single physics rule for validation.
    
    Attributes:
        name: Rule identifier
        domain: Which domain this rule applies to
        description: Human-readable explanation
        formula: Function that computes expected physical value
        required_columns: Columns needed for computation
        threshold: Minimum expected vs observed ratio for validation
    """
    name: str
    domain: Domain
    description: str
    formula: Callable[[pd.DataFrame], pd.Series]
    required_columns: List[str]
    threshold: float = 0.3  # Expected/Observed ratio threshold
    weight: float = 1.0  # Importance weight for gray zone scoring


@dataclass
class ValidationResult:
    """Result of physics validation."""
    pattern_id: str
    statistical_confidence: float
    physics_score: float  # 0-1, how well it matches physics
    is_valid: bool
    is_gray_zone: bool
    rules_applied: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def combined_score(self) -> float:
        """Combined score: statistical * physics."""
        return self.statistical_confidence * self.physics_score


# ============================================================================
# FLOOD / STORM SURGE PHYSICS RULES
# ============================================================================

def wind_setup_formula(df: pd.DataFrame) -> pd.Series:
    """
    Wind setup estimate: water level rise due to wind stress.
    
    Formula: setup ≈ (wind_speed² × fetch × duration) / (g × depth)
    Simplified: setup ≈ (wind_speed² × duration_h) / 9.81
    
    Physical basis: Wind stress τ ∝ ρ_air × C_d × U²
    """
    g = 9.81  # gravity
    
    # Handle column name variations
    wind_col = next((c for c in df.columns if 'wind' in c.lower() and 'speed' in c.lower()), None)
    duration_col = next((c for c in df.columns if 'duration' in c.lower() or 'time' in c.lower()), None)
    
    if wind_col is None:
        wind_col = 'wind_speed'
    if duration_col is None:
        duration_col = 'duration_h'
    
    if wind_col not in df.columns:
        logger.warning(f"Column {wind_col} not found, using zeros")
        return pd.Series(0, index=df.index)
    
    wind_speed = df[wind_col].fillna(0)
    duration = df.get(duration_col, pd.Series(1, index=df.index)).fillna(1)
    
    # Simplified wind setup estimate (meters)
    setup = (wind_speed ** 2 * duration) / (g * 100)  # scaled
    return setup


def pressure_effect_formula(df: pd.DataFrame) -> pd.Series:
    """
    Inverse barometer effect: 1 hPa drop ≈ 1 cm water rise.
    
    Formula: Δh ≈ (P_ref - P_actual) / 100  [in meters]
    Reference pressure: 1013.25 hPa
    """
    P_ref = 1013.25
    
    pressure_col = next((c for c in df.columns if 'pressure' in c.lower() or 'slp' in c.lower()), None)
    if pressure_col is None:
        pressure_col = 'pressure_hPa'
    
    if pressure_col not in df.columns:
        return pd.Series(0, index=df.index)
    
    pressure = df[pressure_col].fillna(P_ref)
    effect = (P_ref - pressure) / 100  # Convert to meters
    return effect.clip(lower=0)


def wind_direction_factor(df: pd.DataFrame) -> pd.Series:
    """
    Wind direction effectiveness for surge (Denmark context).
    
    NW-N winds push water into Danish Straits (high effect).
    S-SE winds push water out (low/negative effect).
    """
    dir_col = next((c for c in df.columns if 'dir' in c.lower()), None)
    
    if dir_col is None or dir_col not in df.columns:
        return pd.Series(1, index=df.index)
    
    # Direction in degrees (0=N, 90=E, 180=S, 270=W)
    direction = df[dir_col].fillna(0)
    
    # NW (315) to N (0/360) = high effectiveness
    # S (180) = low effectiveness
    # Cosine-based factor: max at 330° (NNW), min at 150° (SSE)
    optimal_dir = 330  # NNW most effective for Denmark
    angle_diff = np.abs(((direction - optimal_dir + 180) % 360) - 180)
    factor = np.cos(np.radians(angle_diff))
    
    return pd.Series(factor, index=df.index).clip(lower=0)


FLOOD_RULES = [
    PhysicsRule(
        name="wind_setup",
        domain=Domain.FLOOD,
        description="Wind stress causing water setup (τ ∝ U²)",
        formula=wind_setup_formula,
        required_columns=["wind_speed"],
        threshold=0.2,
        weight=1.0,
    ),
    PhysicsRule(
        name="pressure_effect",
        domain=Domain.FLOOD,
        description="Inverse barometer effect (~1cm per hPa)",
        formula=pressure_effect_formula,
        required_columns=["pressure_hPa"],
        threshold=0.1,
        weight=0.5,
    ),
    PhysicsRule(
        name="wind_direction",
        domain=Domain.FLOOD,
        description="Wind direction effectiveness for surge",
        formula=wind_direction_factor,
        required_columns=["wind_dir"],
        threshold=0.3,
        weight=0.8,
    ),
]


# ============================================================================
# MANUFACTURING PHYSICS RULES
# ============================================================================

def temperature_effect_formula(df: pd.DataFrame) -> pd.Series:
    """
    Temperature effect on material properties.
    
    For polymers/rubber: Viscosity drops, flow increases with temp.
    Critical range: ~20-30°C for many extrusion processes.
    Effect: T > 25°C → increased defect risk.
    """
    temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
    if temp_col is None:
        temp_col = 'temperature'
    
    if temp_col not in df.columns:
        return pd.Series(0, index=df.index)
    
    temp = df[temp_col].fillna(20)
    
    # Risk increases above 23°C, exponential above 25°C
    T_optimal = 22
    T_critical = 25
    
    effect = np.where(
        temp <= T_optimal,
        0,
        np.where(
            temp <= T_critical,
            (temp - T_optimal) / (T_critical - T_optimal) * 0.5,
            0.5 + 0.5 * (1 - np.exp(-(temp - T_critical) / 3))
        )
    )
    return pd.Series(effect, index=df.index)


def extrusion_speed_effect(df: pd.DataFrame) -> pd.Series:
    """
    Extrusion speed effect on quality.
    
    Too fast: insufficient material consolidation, air bubbles.
    Too slow: overcooling, poor bonding.
    Optimal range depends on material.
    """
    speed_col = next((c for c in df.columns if 'speed' in c.lower()), None)
    if speed_col is None:
        speed_col = 'extrusion_speed'
    
    if speed_col not in df.columns:
        return pd.Series(0, index=df.index)
    
    speed = df[speed_col].fillna(100)
    
    # Quadratic risk: optimal around 100-120, risk at extremes
    optimal_speed = 110
    sigma = 30
    
    deviation = np.abs(speed - optimal_speed)
    effect = 1 - np.exp(-(deviation / sigma) ** 2)
    
    return pd.Series(effect, index=df.index)


def humidity_effect_formula(df: pd.DataFrame) -> pd.Series:
    """
    Humidity effect on material curing/bonding.
    
    High humidity can affect adhesion, curing time.
    Critical above 65% for many polymer processes.
    """
    humid_col = next((c for c in df.columns if 'humid' in c.lower()), None)
    if humid_col is None:
        humid_col = 'humidity'
    
    if humid_col not in df.columns:
        return pd.Series(0, index=df.index)
    
    humidity = df[humid_col].fillna(50)
    
    # Effect increases above 60%
    effect = np.where(
        humidity <= 60,
        0,
        (humidity - 60) / 40  # Linear increase to max at 100%
    )
    return pd.Series(effect, index=df.index).clip(0, 1)


MANUFACTURING_RULES = [
    PhysicsRule(
        name="temperature_effect",
        domain=Domain.MANUFACTURING,
        description="Temperature effect on material viscosity/flow",
        formula=temperature_effect_formula,
        required_columns=["temperature"],
        threshold=0.25,
        weight=1.0,
    ),
    PhysicsRule(
        name="speed_effect",
        domain=Domain.MANUFACTURING,
        description="Extrusion speed effect on consolidation",
        formula=extrusion_speed_effect,
        required_columns=["extrusion_speed"],
        threshold=0.3,
        weight=0.8,
    ),
    PhysicsRule(
        name="humidity_effect",
        domain=Domain.MANUFACTURING,
        description="Humidity effect on curing/bonding",
        formula=humidity_effect_formula,
        required_columns=["humidity"],
        threshold=0.2,
        weight=0.5,
    ),
]


# ============================================================================
# ENERGY / LOAD FORECASTING PHYSICS RULES
# ============================================================================

def thermal_demand_formula(df: pd.DataFrame) -> pd.Series:
    """
    Thermal demand from temperature deviation.
    
    Heating degree days (HDD): T < 18°C → heating demand
    Cooling degree days (CDD): T > 24°C → cooling demand
    """
    temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
    if temp_col not in df.columns:
        return pd.Series(0, index=df.index)
    
    temp = df[temp_col].fillna(20)
    
    # Combined thermal demand
    hdd = np.maximum(18 - temp, 0)  # Heating
    cdd = np.maximum(temp - 24, 0)  # Cooling (stronger effect due to AC)
    
    demand = hdd * 1.0 + cdd * 1.5  # Cooling has higher load impact
    return pd.Series(demand, index=df.index)


ENERGY_RULES = [
    PhysicsRule(
        name="thermal_demand",
        domain=Domain.ENERGY,
        description="Temperature-driven heating/cooling demand",
        formula=thermal_demand_formula,
        required_columns=["temperature"],
        threshold=0.3,
        weight=1.0,
    ),
]


# ============================================================================
# PHYSICS VALIDATOR CLASS
# ============================================================================

class PhysicsValidator:
    """
    Validates patterns against physics rules.
    
    Example:
        validator = PhysicsValidator(domain='manufacturing')
        
        # Apply validation
        df_validated = validator.apply(
            features_df,
            rules=['temperature_effect', 'speed_effect'],
            outcome_column='failure'
        )
        
        # Get gray zone patterns
        gray_patterns = validator.get_gray_zone(
            patterns_df,
            statistical_confidence_col='confidence',
            threshold=0.7  # Patterns with conf>0.7 but physics<0.3
        )
    """
    
    # Rule registry by domain
    RULES_REGISTRY = {
        Domain.FLOOD: {r.name: r for r in FLOOD_RULES},
        Domain.MANUFACTURING: {r.name: r for r in MANUFACTURING_RULES},
        Domain.ENERGY: {r.name: r for r in ENERGY_RULES},
    }
    
    def __init__(
        self,
        domain: str = "generic",
        custom_rules: Optional[List[PhysicsRule]] = None,
    ):
        """
        Args:
            domain: One of 'flood', 'manufacturing', 'energy', 'generic'
            custom_rules: Additional custom physics rules
        """
        self.domain = Domain(domain) if isinstance(domain, str) else domain
        self.rules: Dict[str, PhysicsRule] = {}
        
        # Load domain rules
        if self.domain in self.RULES_REGISTRY:
            self.rules.update(self.RULES_REGISTRY[self.domain])
        
        # Add custom rules
        if custom_rules:
            for rule in custom_rules:
                self.rules[rule.name] = rule
        
        self._validation_results: List[ValidationResult] = []
        self._gray_zone_df: Optional[pd.DataFrame] = None
    
    def available_rules(self) -> List[str]:
        """List available rules for current domain."""
        return list(self.rules.keys())
    
    def apply(
        self,
        df: pd.DataFrame,
        rules: Optional[List[str]] = None,
        outcome_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply physics rules to compute expected physical values.
        
        Args:
            df: Features DataFrame
            rules: List of rule names to apply (None = all)
            outcome_column: Column with observed outcome for comparison
            
        Returns:
            DataFrame with added physics columns
        """
        result = df.copy()
        rules_to_apply = rules or list(self.rules.keys())
        
        physics_scores = []
        
        for rule_name in rules_to_apply:
            if rule_name not in self.rules:
                logger.warning(f"Rule {rule_name} not found, skipping")
                continue
            
            rule = self.rules[rule_name]
            
            # Check required columns
            missing = [c for c in rule.required_columns 
                      if not any(c.lower() in col.lower() for col in df.columns)]
            if missing:
                logger.debug(f"Rule {rule_name}: missing columns {missing}")
                continue
            
            # Apply formula
            try:
                physics_value = rule.formula(df)
                col_name = f"physics_{rule_name}"
                result[col_name] = physics_value
                physics_scores.append((col_name, rule.weight))
                logger.debug(f"Applied rule {rule_name}")
            except Exception as e:
                logger.warning(f"Rule {rule_name} failed: {e}")
        
        # Compute combined physics score
        if physics_scores:
            total_weight = sum(w for _, w in physics_scores)
            combined = sum(
                result[col] * weight for col, weight in physics_scores
            ) / total_weight
            result['physics_score_combined'] = combined.clip(0, 1)
        
        return result
    
    def validate_pattern(
        self,
        pattern_conditions: Dict[str, Any],
        sample_data: pd.DataFrame,
        statistical_confidence: float,
    ) -> ValidationResult:
        """
        Validate a single pattern against physics rules.
        
        Args:
            pattern_conditions: Dict of feature -> value conditions
            sample_data: Data samples matching the pattern
            statistical_confidence: Confidence from statistical detection
            
        Returns:
            ValidationResult with physics score and gray zone flag
        """
        # Apply physics rules to matching samples
        validated_df = self.apply(sample_data)
        
        if 'physics_score_combined' not in validated_df.columns:
            physics_score = 0.5  # Neutral if no rules apply
        else:
            physics_score = validated_df['physics_score_combined'].mean()
        
        # Determine validity and gray zone
        is_valid = physics_score >= 0.3
        is_gray = (
            statistical_confidence >= 0.6 and 
            physics_score < 0.4
        )
        
        result = ValidationResult(
            pattern_id=str(hash(frozenset(pattern_conditions.items()))),
            statistical_confidence=statistical_confidence,
            physics_score=physics_score,
            is_valid=is_valid,
            is_gray_zone=is_gray,
            rules_applied=list(self.rules.keys()),
            details={
                "conditions": pattern_conditions,
                "n_samples": len(sample_data),
            }
        )
        
        self._validation_results.append(result)
        return result
    
    def compute_gray_zone(
        self,
        patterns_df: pd.DataFrame,
        confidence_col: str = 'confidence',
        support_col: str = 'support',
        physics_threshold: float = 0.4,
        confidence_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        Identify patterns in the gray zone.
        
        Gray zone = high statistical confidence + low physics score
        These need human review.
        
        Args:
            patterns_df: DataFrame with patterns and their metrics
            confidence_col: Column with statistical confidence
            physics_threshold: Physics score below this = low
            confidence_threshold: Confidence above this = high
            
        Returns:
            DataFrame of gray zone patterns
        """
        df = patterns_df.copy()
        
        # Need physics scores
        if 'physics_score_combined' not in df.columns:
            df['physics_score_combined'] = 0.5  # Default neutral
        
        # Flag gray zone
        df['is_gray_zone'] = (
            (df[confidence_col] >= confidence_threshold) &
            (df['physics_score_combined'] < physics_threshold)
        )
        
        # Gray zone score (higher = more concerning)
        df['gray_zone_score'] = np.where(
            df['is_gray_zone'],
            df[confidence_col] * (1 - df['physics_score_combined']),
            0
        )
        
        self._gray_zone_df = df[df['is_gray_zone']].sort_values(
            'gray_zone_score', ascending=False
        )
        
        return self._gray_zone_df
    
    def get_gray_zone(self) -> Optional[pd.DataFrame]:
        """Get last computed gray zone patterns."""
        return self._gray_zone_df
    
    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self._validation_results:
            return {"status": "no validations performed"}
        
        valid_count = sum(1 for r in self._validation_results if r.is_valid)
        gray_count = sum(1 for r in self._validation_results if r.is_gray_zone)
        
        return {
            "total_validated": len(self._validation_results),
            "valid_patterns": valid_count,
            "invalid_patterns": len(self._validation_results) - valid_count,
            "gray_zone_patterns": gray_count,
            "avg_physics_score": np.mean([r.physics_score for r in self._validation_results]),
            "avg_combined_score": np.mean([r.combined_score for r in self._validation_results]),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_flood_validator() -> PhysicsValidator:
    """Create validator preconfigured for flood/storm surge."""
    return PhysicsValidator(domain='flood')


def create_manufacturing_validator() -> PhysicsValidator:
    """Create validator preconfigured for manufacturing."""
    return PhysicsValidator(domain='manufacturing')


def create_energy_validator() -> PhysicsValidator:
    """Create validator preconfigured for energy/load forecasting."""
    return PhysicsValidator(domain='energy')

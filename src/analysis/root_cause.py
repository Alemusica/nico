"""
🔍 Root Cause Analysis Methods
===============================

Orthodox quality assurance methods adapted for scientific/flood analysis:

1. ISHIKAWA (Fishbone) - 6M categories for oceanography
2. FMEA - Failure Mode Effects Analysis for sensor/data quality
3. 5-WHY - Iterative root cause drilling
4. PARETO - 80/20 rule for prioritizing causes
5. FAULT TREE - Boolean logic for complex events

These integrate with:
- PCMCI statistical causality
- Physics engine validation
- LLM explanations
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json


# =============================================================================
# ISHIKAWA (FISHBONE) DIAGRAM - Adapted for Oceanography/Flood
# =============================================================================

class IshikawaCategory(Enum):
    """
    6M categories adapted for oceanographic/flood analysis.
    
    Original manufacturing 6M:
        Man, Machine, Method, Material, Measurement, Mother Nature
    
    Adapted for oceanography:
        Atmosphere, Ocean, Cryosphere, Measurement, Model, External
    """
    # Atmospheric factors
    ATMOSPHERE = "atmosphere"  # Wind, pressure, NAO, storms
    
    # Ocean factors  
    OCEAN = "ocean"  # Currents, SST, salinity, SSH
    
    # Cryosphere factors
    CRYOSPHERE = "cryosphere"  # Sea ice, glaciers, freshwater flux
    
    # Measurement factors (satellites, instruments)
    MEASUREMENT = "measurement"  # Sensor drift, orbit errors, calibration
    
    # Model/Method factors
    MODEL = "model"  # Algorithm errors, interpolation, grid resolution
    
    # External/Human factors
    EXTERNAL = "external"  # Policy, infrastructure, coastal changes


@dataclass
class IshikawaCause:
    """A single cause in the fishbone diagram."""
    name: str
    category: IshikawaCategory
    description: str
    sub_causes: List[str] = field(default_factory=list)
    
    # Evidence
    data_support: float = 0.0  # 0-1, correlation with effect
    physics_support: float = 0.0  # 0-1, physical plausibility
    literature_support: float = 0.0  # 0-1, documented in papers
    
    # Metrics
    lag_days: Optional[int] = None
    effect_magnitude: Optional[float] = None
    
    @property
    def total_score(self) -> float:
        """Combined evidence score."""
        return (self.data_support * 0.4 + 
                self.physics_support * 0.3 + 
                self.literature_support * 0.3)


@dataclass
class IshikawaDiagram:
    """Complete fishbone diagram for root cause analysis."""
    effect: str  # The problem/event being analyzed
    effect_description: str
    causes: List[IshikawaCause] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_cause(self, cause: IshikawaCause):
        self.causes.append(cause)
    
    def get_by_category(self, category: IshikawaCategory) -> List[IshikawaCause]:
        return [c for c in self.causes if c.category == category]
    
    def get_top_causes(self, n: int = 5) -> List[IshikawaCause]:
        """Get top N causes by total evidence score."""
        return sorted(self.causes, key=lambda c: c.total_score, reverse=True)[:n]
    
    def to_dict(self) -> Dict:
        return {
            "effect": self.effect,
            "effect_description": self.effect_description,
            "created_at": self.created_at,
            "categories": {
                cat.value: [
                    {
                        "name": c.name,
                        "description": c.description,
                        "sub_causes": c.sub_causes,
                        "scores": {
                            "data": c.data_support,
                            "physics": c.physics_support,
                            "literature": c.literature_support,
                            "total": c.total_score
                        },
                        "lag_days": c.lag_days,
                        "magnitude": c.effect_magnitude
                    }
                    for c in self.get_by_category(cat)
                ]
                for cat in IshikawaCategory
            }
        }


# =============================================================================
# FMEA - Failure Mode Effects Analysis (for data quality)
# =============================================================================

@dataclass
class FMEAItem:
    """
    Single FMEA analysis item.
    
    RPN = Severity × Occurrence × Detection
    
    For satellite/sensor data:
    - Severity: How bad is bad data? (1-10)
    - Occurrence: How often does this failure happen? (1-10)
    - Detection: Can we detect/filter it? (1-10, high=hard to detect)
    """
    failure_mode: str
    potential_effect: str
    potential_cause: str
    
    # Traditional FMEA scores (1-10)
    severity: int = 5
    occurrence: int = 5
    detection: int = 5
    
    # Controls
    current_controls: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    @property
    def rpn(self) -> int:
        """Risk Priority Number = S × O × D"""
        return self.severity * self.occurrence * self.detection


@dataclass
class FMEAAnalysis:
    """Complete FMEA for a system/process."""
    system_name: str
    process_step: str
    items: List[FMEAItem] = field(default_factory=list)
    
    def add_item(self, item: FMEAItem):
        self.items.append(item)
    
    def get_critical_items(self, rpn_threshold: int = 100) -> List[FMEAItem]:
        """Items needing immediate attention."""
        return [i for i in self.items if i.rpn >= rpn_threshold]
    
    def prioritized_list(self) -> List[FMEAItem]:
        """All items sorted by RPN descending."""
        return sorted(self.items, key=lambda x: x.rpn, reverse=True)


# =============================================================================
# 5-WHY Analysis
# =============================================================================

@dataclass
class WhyStep:
    """Single step in 5-Why chain."""
    level: int  # 1-5
    question: str  # "Why did X happen?"
    answer: str
    evidence: str = ""
    confidence: float = 0.5  # 0-1


@dataclass
class FiveWhyAnalysis:
    """
    5-Why root cause drilling.
    
    Keeps asking "why" until root cause is found.
    Integrated with LLM for answer generation.
    """
    problem_statement: str
    steps: List[WhyStep] = field(default_factory=list)
    root_cause: Optional[str] = None
    
    def add_why(self, question: str, answer: str, evidence: str = "", confidence: float = 0.5):
        level = len(self.steps) + 1
        self.steps.append(WhyStep(
            level=level,
            question=question,
            answer=answer,
            evidence=evidence,
            confidence=confidence
        ))
        if level >= 5:
            self.root_cause = answer
    
    def get_chain(self) -> List[str]:
        """Get the why-chain as list of answers."""
        return [s.answer for s in self.steps]
    
    def average_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)


# =============================================================================
# FLOOD-SPECIFIC PHYSICS SCORING
# =============================================================================

@dataclass
class FloodPhysicsScore:
    """
    Physics-based scoring for flood/surge causes.
    
    Uses known equations:
    - Wind setup: η = (τ·L)/(ρ·g·h) where τ ∝ U²
    - Inverse barometer: Δη = -ΔP/(ρ·g) ≈ 1 cm per hPa
    - Coriolis deflection: affects current direction
    """
    
    # Wind contribution
    wind_speed_ms: float = 0.0
    wind_fetch_km: float = 0.0
    wind_direction_deg: float = 0.0
    wind_setup_m: float = 0.0  # Calculated
    
    # Pressure contribution
    pressure_anomaly_hpa: float = 0.0
    ib_effect_m: float = 0.0  # Inverse barometer, calculated
    
    # Current/advection contribution
    current_speed_ms: float = 0.0
    current_contribution_m: float = 0.0
    
    # Total and validation
    total_surge_m: float = 0.0
    observed_surge_m: float = 0.0
    
    # Physics constants
    RHO_WATER: float = 1025.0  # kg/m³
    RHO_AIR: float = 1.225  # kg/m³
    G: float = 9.81  # m/s²
    CD: float = 0.0013  # Drag coefficient
    
    def calculate_wind_setup(self, depth_m: float = 10.0):
        """
        Calculate wind setup using simplified formula.
        η = (CD · ρair · U² · L) / (ρwater · g · h)
        """
        tau = self.CD * self.RHO_AIR * self.wind_speed_ms**2
        self.wind_setup_m = (tau * self.wind_fetch_km * 1000) / (
            self.RHO_WATER * self.G * depth_m
        )
        return self.wind_setup_m
    
    def calculate_ib_effect(self):
        """
        Calculate inverse barometer effect.
        Δη = -ΔP / (ρ·g) ≈ 1 cm per hPa
        """
        self.ib_effect_m = -self.pressure_anomaly_hpa * 100 / (
            self.RHO_WATER * self.G
        )
        return self.ib_effect_m
    
    def calculate_total(self):
        """Sum all contributions."""
        self.total_surge_m = (
            self.wind_setup_m + 
            self.ib_effect_m + 
            self.current_contribution_m
        )
        return self.total_surge_m
    
    def validation_score(self) -> float:
        """
        How well does physics explain the observation?
        Returns 0-1 where 1 = perfect match.
        """
        if self.observed_surge_m == 0:
            return 0.0
        
        error = abs(self.total_surge_m - self.observed_surge_m)
        relative_error = error / abs(self.observed_surge_m)
        
        # Score decreases with error
        return max(0.0, 1.0 - relative_error)
    
    def to_dict(self) -> Dict:
        return {
            "contributions": {
                "wind_setup_m": round(self.wind_setup_m, 4),
                "inverse_barometer_m": round(self.ib_effect_m, 4),
                "current_m": round(self.current_contribution_m, 4),
                "total_calculated_m": round(self.total_surge_m, 4),
            },
            "inputs": {
                "wind_speed_ms": self.wind_speed_ms,
                "wind_fetch_km": self.wind_fetch_km,
                "pressure_anomaly_hpa": self.pressure_anomaly_hpa,
            },
            "validation": {
                "observed_m": self.observed_surge_m,
                "physics_score": round(self.validation_score(), 3),
            }
        }


# =============================================================================
# ROOT CAUSE ANALYZER - Integrates All Methods
# =============================================================================

class RootCauseAnalyzer:
    """
    Unified root cause analysis combining:
    - Ishikawa categorical analysis
    - FMEA for measurement/model quality
    - 5-Why drilling
    - Physics-based scoring
    - PCMCI statistical causality (from causal_service)
    """
    
    def __init__(self):
        self.ishikawa: Optional[IshikawaDiagram] = None
        self.fmea: Optional[FMEAAnalysis] = None
        self.five_why: Optional[FiveWhyAnalysis] = None
        self.physics_score: Optional[FloodPhysicsScore] = None
    
    def create_ishikawa(self, effect: str, description: str) -> IshikawaDiagram:
        """Initialize fishbone diagram for an event."""
        self.ishikawa = IshikawaDiagram(effect=effect, effect_description=description)
        return self.ishikawa
    
    def create_fmea(self, system: str, process: str) -> FMEAAnalysis:
        """Initialize FMEA for data quality analysis."""
        self.fmea = FMEAAnalysis(system_name=system, process_step=process)
        return self.fmea
    
    def create_five_why(self, problem: str) -> FiveWhyAnalysis:
        """Initialize 5-Why analysis."""
        self.five_why = FiveWhyAnalysis(problem_statement=problem)
        return self.five_why
    
    def create_physics_score(self) -> FloodPhysicsScore:
        """Initialize physics-based scoring."""
        self.physics_score = FloodPhysicsScore()
        return self.physics_score
    
    def integrate_pcmci_results(
        self, 
        pcmci_links: List[Dict],
        ishikawa: IshikawaDiagram
    ) -> IshikawaDiagram:
        """
        Add PCMCI discovered links as causes in Ishikawa diagram.
        
        Maps variables to categories:
        - wind*, pressure* → ATMOSPHERE
        - ssh, sst, current*, salinity → OCEAN
        - ice*, snow* → CRYOSPHERE
        - Others → MODEL
        """
        category_map = {
            "wind": IshikawaCategory.ATMOSPHERE,
            "pressure": IshikawaCategory.ATMOSPHERE,
            "nao": IshikawaCategory.ATMOSPHERE,
            "ssh": IshikawaCategory.OCEAN,
            "sst": IshikawaCategory.OCEAN,
            "sla": IshikawaCategory.OCEAN,
            "current": IshikawaCategory.OCEAN,
            "salinity": IshikawaCategory.OCEAN,
            "ice": IshikawaCategory.CRYOSPHERE,
            "snow": IshikawaCategory.CRYOSPHERE,
        }
        
        for link in pcmci_links:
            source = link.get("source", "")
            
            # Find category
            category = IshikawaCategory.MODEL  # default
            for key, cat in category_map.items():
                if key in source.lower():
                    category = cat
                    break
            
            cause = IshikawaCause(
                name=source,
                category=category,
                description=f"Statistical cause of {link.get('target')}",
                data_support=abs(link.get("strength", 0)),
                physics_support=link.get("physics_score", 0.5),
                lag_days=link.get("lag", 0),
                effect_magnitude=link.get("strength", 0)
            )
            ishikawa.add_cause(cause)
        
        return ishikawa
    
    def get_comprehensive_report(self) -> Dict:
        """Generate combined report from all analyses."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "analyses": {}
        }
        
        if self.ishikawa:
            report["analyses"]["ishikawa"] = self.ishikawa.to_dict()
            report["analyses"]["top_causes"] = [
                {"name": c.name, "score": c.total_score}
                for c in self.ishikawa.get_top_causes(5)
            ]
        
        if self.fmea:
            critical = self.fmea.get_critical_items()
            report["analyses"]["fmea"] = {
                "total_items": len(self.fmea.items),
                "critical_items": len(critical),
                "highest_rpn": max((i.rpn for i in self.fmea.items), default=0),
                "critical_failures": [i.failure_mode for i in critical]
            }
        
        if self.five_why:
            report["analyses"]["five_why"] = {
                "problem": self.five_why.problem_statement,
                "root_cause": self.five_why.root_cause,
                "chain": self.five_why.get_chain(),
                "confidence": self.five_why.average_confidence()
            }
        
        if self.physics_score:
            report["analyses"]["physics"] = self.physics_score.to_dict()
        
        return report


# =============================================================================
# HELPER: Create Standard Flood Ishikawa Template
# =============================================================================

def create_flood_ishikawa_template(event_name: str) -> IshikawaDiagram:
    """
    Create pre-populated Ishikawa diagram for flood/surge analysis.
    
    Based on standard flood causation factors from literature.
    """
    diagram = IshikawaDiagram(
        effect=event_name,
        effect_description="Storm surge or flood event requiring root cause analysis"
    )
    
    # Atmospheric causes
    diagram.add_cause(IshikawaCause(
        name="Wind Forcing",
        category=IshikawaCategory.ATMOSPHERE,
        description="Wind stress on water surface causing setup",
        sub_causes=["Wind speed", "Wind direction", "Fetch length", "Duration"]
    ))
    diagram.add_cause(IshikawaCause(
        name="Atmospheric Pressure",
        category=IshikawaCategory.ATMOSPHERE,
        description="Inverse barometer effect from low pressure",
        sub_causes=["Central pressure", "Pressure gradient", "Storm speed"]
    ))
    diagram.add_cause(IshikawaCause(
        name="Large-scale Patterns",
        category=IshikawaCategory.ATMOSPHERE,
        description="Climate indices affecting storm tracks",
        sub_causes=["NAO", "AO", "AMO", "Jet stream position"]
    ))
    
    # Ocean causes
    diagram.add_cause(IshikawaCause(
        name="Background SSH",
        category=IshikawaCategory.OCEAN,
        description="Pre-existing sea level conditions",
        sub_causes=["Seasonal cycle", "Steric height", "Mass distribution"]
    ))
    diagram.add_cause(IshikawaCause(
        name="Ocean Currents",
        category=IshikawaCategory.OCEAN,
        description="Advection of water mass",
        sub_causes=["Gulf Stream", "Atlantic inflow", "Coastal currents"]
    ))
    diagram.add_cause(IshikawaCause(
        name="Water Temperature",
        category=IshikawaCategory.OCEAN,
        description="SST affecting storm intensification",
        sub_causes=["SST anomaly", "Heat content", "Mixed layer depth"]
    ))
    
    # Measurement factors
    diagram.add_cause(IshikawaCause(
        name="Satellite Coverage",
        category=IshikawaCategory.MEASUREMENT,
        description="Gaps in altimetry observations",
        sub_causes=["Orbit gaps", "Offline periods", "Quality flags"]
    ))
    diagram.add_cause(IshikawaCause(
        name="Instrument Drift",
        category=IshikawaCategory.MEASUREMENT,
        description="Sensor calibration changes over time",
        sub_causes=["Radiometer drift", "Altimeter bias", "Cross-calibration"]
    ))
    
    return diagram


# =============================================================================
# HELPER: Create Standard Satellite Data FMEA
# =============================================================================

def create_satellite_fmea_template() -> FMEAAnalysis:
    """
    Create FMEA template for satellite altimetry data quality.
    """
    fmea = FMEAAnalysis(
        system_name="Multi-Satellite Altimetry System",
        process_step="Data Fusion and Analysis"
    )
    
    # Common failure modes for satellite data
    fmea.add_item(FMEAItem(
        failure_mode="Satellite offline during event",
        potential_effect="Missing data in critical period",
        potential_cause="Orbital maneuver, instrument anomaly",
        severity=8,
        occurrence=3,
        detection=2,
        current_controls=["Multi-satellite constellation"],
        recommended_actions=["Interpolation from nearby tracks"]
    ))
    
    fmea.add_item(FMEAItem(
        failure_mode="Land contamination in coastal zone",
        potential_effect="Spurious high SSH values",
        potential_cause="Radar footprint includes land",
        severity=7,
        occurrence=6,
        detection=4,
        current_controls=["Quality flags", "Land mask"],
        recommended_actions=["SAR mode for coastal", "Retracking algorithms"]
    ))
    
    fmea.add_item(FMEAItem(
        failure_mode="Rain cell contamination",
        potential_effect="Attenuated/noisy SSH",
        potential_cause="Heavy precipitation in altimeter path",
        severity=5,
        occurrence=4,
        detection=5,
        current_controls=["Rain flag filtering"],
        recommended_actions=["Dual-frequency correction", "Co-located radiometer"]
    ))
    
    fmea.add_item(FMEAItem(
        failure_mode="Sea state bias error",
        potential_effect="Systematic SSH underestimate in high waves",
        potential_cause="Empirical SSB model limitations",
        severity=6,
        occurrence=5,
        detection=6,
        current_controls=["SSB correction applied"],
        recommended_actions=["3D SSB models", "Cross-validation with buoys"]
    ))
    
    fmea.add_item(FMEAItem(
        failure_mode="Geoid model error",
        potential_effect="Systematic regional bias in DOT",
        potential_cause="Gravity field model uncertainty",
        severity=7,
        occurrence=3,
        detection=7,
        current_controls=["Latest GRACE/GOCE geoid"],
        recommended_actions=["Multi-geoid ensemble", "Regional validation"]
    ))
    
    return fmea

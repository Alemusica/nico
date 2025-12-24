"""
🧠 Enhanced Knowledge Scorer with Dynamic Satellite Indices
============================================================

Extends KnowledgeScorer to support:
1. Dynamic indices from satellite data (instead of static)
2. Physics-based validation scoring
3. Causal chain scoring
4. Experience-based scoring
5. Hybrid scoring combining all three

This module bridges:
- KnowledgeScorer (pipeline scoring)
- SatelliteFusionEngine (dynamic indices)
- RootCauseAnalyzer (physics validation)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

# Import base components
from src.pipeline.knowledge_scorer import KnowledgeScorer, ScoredKnowledge
from src.data.satellite_fusion import (
    SatelliteFusionEngine,
    DynamicIndexCalculator,
    SatelliteObservation,
    SATELLITE_CONFIGS,
)
from src.analysis.root_cause import (
    FloodPhysicsScore,
    RootCauseAnalyzer,
    IshikawaDiagram,
    FMEAAnalysis,
)


@dataclass
class DynamicIndices:
    """Dynamic indices calculated from satellite data."""
    thermodynamics: float = 0.0  # From SST
    oceanography: float = 0.0    # From SSH/SLA
    cryosphere: float = 0.0      # From sea ice
    anemometry: float = 0.0      # From altimeter wind
    precipitation: float = 5.0    # Placeholder (reanalysis)
    
    # Metadata
    satellites_used: List[str] = field(default_factory=list)
    data_coverage: float = 0.0
    calculation_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class HybridScore:
    """Combined physics + chain + experience score."""
    hybrid_score: float
    physics_score: float
    chain_score: float
    experience_score: float
    
    weights: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    
    # Details
    physics_detail: Dict[str, float] = field(default_factory=dict)
    chain_links: int = 0
    historical_matches: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnhancedKnowledgeScorer(KnowledgeScorer):
    """
    Enhanced Knowledge Scorer with dynamic satellite indices.
    
    Key Features:
    1. DYNAMIC INDICES
       - Calculate from real satellite data
       - Handle offline/unavailable satellites
       - Weight by quality and proximity
    
    2. PHYSICS VALIDATION
       - Wind setup equation
       - Inverse barometer effect
       - Compare expected vs observed
    
    3. CHAIN SCORING
       - Strength of causal path
       - Confidence from p-values
       - Number of links
    
    4. EXPERIENCE SCORING
       - Historical pattern matching
       - Logarithmic scaling
       - Cross-event similarity
    
    5. HYBRID COMBINATION
       - Configurable weights
       - Physics: 40%, Chain: 30%, Experience: 30% (default)
    """
    
    def __init__(
        self,
        correlations_dir: Path = None,
        refined_dir: Path = None,
        events_file: Path = None,
        output_dir: Path = None,
        physics_weight: float = 0.4,
        chain_weight: float = 0.3,
        experience_weight: float = 0.3,
    ):
        super().__init__(correlations_dir, refined_dir, events_file, output_dir)
        
        # Hybrid scoring weights
        self.physics_weight = physics_weight
        self.chain_weight = chain_weight
        self.experience_weight = experience_weight
        
        # Satellite fusion engine
        self.fusion_engine = SatelliteFusionEngine()
        self.index_calculator = DynamicIndexCalculator(self.fusion_engine)
        
        # Root cause analyzer
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        print(f"   🛰️ Satellite fusion enabled")
        print(f"   ⚖️ Weights: physics={physics_weight}, chain={chain_weight}, experience={experience_weight}")
    
    # =========================================================================
    # Dynamic Satellite Indices
    # =========================================================================
    
    def calculate_dynamic_indices(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[datetime, datetime],
        satellite_data: Dict[str, np.ndarray] = None,
    ) -> DynamicIndices:
        """
        Calculate dynamic indices from satellite data.
        
        If satellite_data is provided, uses it directly.
        Otherwise, queries the fusion engine.
        """
        indices = DynamicIndices()
        
        if satellite_data:
            # Use provided data
            calc_indices = self.index_calculator.calculate_all_indices(satellite_data)
            
            indices.thermodynamics = calc_indices.get("thermodynamics", {}).get("index", 0.0)
            indices.oceanography = calc_indices.get("oceanography", {}).get("index", 0.0)
            indices.cryosphere = calc_indices.get("cryosphere", {}).get("index", 0.0)
            indices.anemometry = calc_indices.get("anemometry", {}).get("index", 0.0)
            indices.precipitation = calc_indices.get("precipitation", {}).get("index", 5.0)
            
            # Get satellites that contributed
            indices.satellites_used = self.fusion_engine.get_available_satellites(time_range[0])
            indices.data_coverage = np.mean([
                calc_indices.get(f, {}).get("data_coverage", 0)
                for f in ["thermodynamics", "oceanography", "cryosphere", "anemometry"]
            ])
        else:
            # Query fusion engine (when data files are available)
            observations = self.fusion_engine.query_observations(
                lat_range, lon_range, time_range
            )
            
            if observations:
                # Group by variable and calculate
                ssh_data = np.array([o.ssh for o in observations if o.ssh])
                
                if len(ssh_data) > 0:
                    indices.oceanography = min(10, np.nanstd(ssh_data) / 0.1 * 5)
                
                indices.satellites_used = list(set(o.satellite for o in observations))
                indices.data_coverage = len(observations) / max(1, len(self.fusion_engine.satellites))
        
        return indices
    
    def get_static_fallback_indices(self) -> DynamicIndices:
        """
        Return static fallback indices when satellite data unavailable.
        """
        return DynamicIndices(
            thermodynamics=5.0,
            oceanography=5.0,
            cryosphere=5.0,
            anemometry=5.0,
            precipitation=5.0,
            satellites_used=[],
            data_coverage=0.0,
        )
    
    # =========================================================================
    # Physics-Based Scoring
    # =========================================================================
    
    def calculate_physics_score(
        self,
        wind_speed_ms: float = 0.0,
        pressure_hpa: float = 1013.25,
        ref_pressure_hpa: float = 1013.25,
        fetch_km: float = 100.0,
        depth_m: float = 50.0,
        observed_surge_m: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate physics-based validation score.
        
        Returns (score, detail_dict) where score is 0-1.
        """
        scorer = FloodPhysicsScore(
            wind_speed_ms=wind_speed_ms,
            pressure_hpa=pressure_hpa,
            ref_pressure_hpa=ref_pressure_hpa,
            fetch_km=fetch_km,
            depth_m=depth_m,
            observed_surge_m=observed_surge_m,
        )
        
        score = scorer.validation_score()
        detail = {
            "wind_setup_m": scorer.wind_setup(),
            "inverse_barometer_m": scorer.inverse_barometer(),
            "total_expected_m": scorer.total_surge(),
            "observed_m": observed_surge_m,
        }
        
        return score, detail
    
    # =========================================================================
    # Chain-Based Scoring
    # =========================================================================
    
    def calculate_chain_score(
        self,
        causal_chain: List[Dict[str, Any]],
    ) -> Tuple[float, int]:
        """
        Calculate score based on causal chain strength.
        
        High strength + low p-value = high score.
        
        Returns (score, num_links) where score is 0-1.
        """
        if not causal_chain:
            return 0.0, 0
        
        total_strength = sum(abs(link.get("strength", 0)) for link in causal_chain)
        avg_strength = total_strength / len(causal_chain)
        
        # Get minimum p-value (best significance)
        p_values = [link.get("p_value", 1.0) for link in causal_chain]
        min_pvalue = min(p_values) if p_values else 1.0
        
        # Combined score: high strength + low p-value
        # strength is typically 0-1, p-value is 0-1
        # Good: strength=0.7, p=0.01 → score = 0.7 * 0.99 = 0.69
        score = avg_strength * (1 - min_pvalue)
        
        return min(1.0, score), len(causal_chain)
    
    # =========================================================================
    # Experience-Based Scoring
    # =========================================================================
    
    def calculate_experience_score(
        self,
        historical_matches: int,
    ) -> float:
        """
        Calculate score based on historical pattern matching.
        
        Logarithmic scaling:
        - 1 match = ~0.30
        - 10 matches = ~0.70
        - 100 matches = ~1.00
        
        Returns score 0-1.
        """
        import math
        
        if historical_matches <= 0:
            return 0.0
        
        # Logarithmic scaling
        score = 0.3 + 0.35 * math.log10(1 + historical_matches)
        return min(1.0, score)
    
    # =========================================================================
    # Hybrid Scoring
    # =========================================================================
    
    def calculate_hybrid_score(
        self,
        physics_data: Dict[str, float] = None,
        causal_chain: List[Dict[str, Any]] = None,
        historical_matches: int = 0,
    ) -> HybridScore:
        """
        Calculate hybrid score combining physics + chain + experience.
        
        Like a hybrid car: physics engine + experience engine working together.
        """
        # Physics score
        if physics_data:
            physics_score, physics_detail = self.calculate_physics_score(
                wind_speed_ms=physics_data.get("wind_speed", 0),
                pressure_hpa=physics_data.get("pressure", 1013.25),
                ref_pressure_hpa=physics_data.get("reference_pressure", 1013.25),
                fetch_km=physics_data.get("fetch", 100),
                depth_m=physics_data.get("depth", 50),
                observed_surge_m=physics_data.get("observed_surge", 0),
            )
        else:
            physics_score = 0.5  # Neutral if no data
            physics_detail = {}
        
        # Chain score
        chain_score, chain_links = self.calculate_chain_score(causal_chain or [])
        
        # Experience score
        experience_score = self.calculate_experience_score(historical_matches)
        
        # Hybrid combination
        hybrid_score = (
            self.physics_weight * physics_score +
            self.chain_weight * chain_score +
            self.experience_weight * experience_score
        )
        
        # Interpretation
        if hybrid_score >= 0.8:
            interpretation = "High confidence - strongly supported by physics, chain, and experience"
        elif hybrid_score >= 0.6:
            interpretation = "Moderate confidence - good support, some uncertainty"
        elif hybrid_score >= 0.4:
            interpretation = "Low confidence - partial support, needs verification"
        else:
            interpretation = "Very low confidence - limited evidence, speculative"
        
        return HybridScore(
            hybrid_score=round(hybrid_score, 3),
            physics_score=round(physics_score, 3),
            chain_score=round(chain_score, 3),
            experience_score=round(experience_score, 3),
            weights={
                "physics": self.physics_weight,
                "chain": self.chain_weight,
                "experience": self.experience_weight,
            },
            interpretation=interpretation,
            physics_detail=physics_detail,
            chain_links=chain_links,
            historical_matches=historical_matches,
        )
    
    # =========================================================================
    # Enhanced Scoring with Dynamic Indices
    # =========================================================================
    
    def score_event_enhanced(
        self,
        event,
        correlations,
        items,
        satellite_data: Dict[str, np.ndarray] = None,
        physics_data: Dict[str, float] = None,
        historical_matches: int = 0,
    ) -> Tuple[ScoredKnowledge, DynamicIndices, HybridScore]:
        """
        Enhanced scoring with dynamic indices and hybrid scoring.
        
        Returns:
            - ScoredKnowledge: Base scored knowledge
            - DynamicIndices: Satellite-derived indices
            - HybridScore: Physics+chain+experience score
        """
        # Get base knowledge score
        knowledge = self.score_event(event, correlations, items)
        
        # Calculate dynamic indices
        try:
            dynamic_indices = self.calculate_dynamic_indices(
                lat_range=(60, 85),  # Arctic
                lon_range=(-30, 30),  # North Atlantic / Arctic
                time_range=(
                    datetime.fromisoformat(event.date) - timedelta(days=30),
                    datetime.fromisoformat(event.date) + timedelta(days=30),
                ) if hasattr(event, 'date') else (
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                ),
                satellite_data=satellite_data,
            )
        except Exception:
            dynamic_indices = self.get_static_fallback_indices()
        
        # Override static indices with dynamic if available
        if dynamic_indices.data_coverage > 0:
            knowledge.thermodynamics = dynamic_indices.thermodynamics
            knowledge.oceanography = dynamic_indices.oceanography
            knowledge.cryosphere = dynamic_indices.cryosphere
            knowledge.anemometry = dynamic_indices.anemometry
        
        # Calculate hybrid score
        # Build causal chain from top correlations
        causal_chain = []
        event_corrs = [c for c in correlations if c.event_id == event.id]
        for corr in sorted(event_corrs, key=lambda c: -c.overall_score)[:10]:
            causal_chain.append({
                "source": corr.topic if hasattr(corr, 'topic') else "unknown",
                "target": "event",
                "strength": corr.overall_score,
                "p_value": 0.05,  # Placeholder
            })
        
        hybrid = self.calculate_hybrid_score(
            physics_data=physics_data,
            causal_chain=causal_chain,
            historical_matches=historical_matches,
        )
        
        # Update knowledge summary with hybrid info
        knowledge.summary += f" | Hybrid: {hybrid.hybrid_score:.2f} ({hybrid.interpretation})"
        
        return knowledge, dynamic_indices, hybrid


# Factory function
def create_enhanced_scorer(
    physics_weight: float = 0.4,
    chain_weight: float = 0.3,
    experience_weight: float = 0.3,
) -> EnhancedKnowledgeScorer:
    """Create an enhanced knowledge scorer with custom weights."""
    return EnhancedKnowledgeScorer(
        physics_weight=physics_weight,
        chain_weight=chain_weight,
        experience_weight=experience_weight,
    )


# Quick test
if __name__ == "__main__":
    print("Testing Enhanced Knowledge Scorer...")
    
    scorer = create_enhanced_scorer()
    
    # Test hybrid scoring
    hybrid = scorer.calculate_hybrid_score(
        physics_data={
            "wind_speed": 15.0,
            "pressure": 980.0,
            "reference_pressure": 1013.25,
            "fetch": 200.0,
            "depth": 50.0,
            "observed_surge": 0.4,
        },
        causal_chain=[
            {"source": "wind", "target": "ssh", "strength": 0.7, "p_value": 0.01},
            {"source": "pressure", "target": "ssh", "strength": 0.5, "p_value": 0.02},
        ],
        historical_matches=5,
    )
    
    print(f"\nHybrid Score Results:")
    print(f"  Physics: {hybrid.physics_score:.3f}")
    print(f"  Chain: {hybrid.chain_score:.3f}")
    print(f"  Experience: {hybrid.experience_score:.3f}")
    print(f"  HYBRID: {hybrid.hybrid_score:.3f}")
    print(f"  {hybrid.interpretation}")
    
    # Test dynamic indices with mock data
    import numpy as np
    
    mock_data = {
        "sst": np.random.normal(15, 2, (100,)),
        "ssh": np.random.normal(0, 0.1, (100,)),
        "ice": np.random.uniform(0, 50, (100,)),
        "wind": np.random.normal(10, 3, (100,)),
    }
    
    indices = scorer.calculate_dynamic_indices(
        lat_range=(70, 85),
        lon_range=(-30, 30),
        time_range=(datetime.now() - timedelta(days=30), datetime.now()),
        satellite_data=mock_data,
    )
    
    print(f"\nDynamic Indices:")
    print(f"  Thermodynamics: {indices.thermodynamics:.1f}")
    print(f"  Oceanography: {indices.oceanography:.1f}")
    print(f"  Cryosphere: {indices.cryosphere:.1f}")
    print(f"  Anemometry: {indices.anemometry:.1f}")
    print(f"  Data coverage: {indices.data_coverage:.1%}")

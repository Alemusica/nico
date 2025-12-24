"""
🧠 Knowledge Scorer Module
===========================

Transforms correlations into scored knowledge with multi-factor indices:

1. MULTI-FACTOR INDICES
   - thermodynamics: Temperature/heat-related confidence
   - anemometry: Wind/atmospheric data confidence  
   - precipitation: Rain/freshwater data confidence
   - cryosphere: Ice/snow data confidence
   - oceanography: Current/transport data confidence

2. DATA DENSITY SCORING
   - High density = more corroborating evidence
   - Low density = high uncertainty

3. TEMPORAL PROXIMITY WEIGHTING
   - Recent data = higher weight
   - >90 days from event = exponential decay

4. CONSENSUS SCORING
   - Multiple sources agreeing = higher confidence
   - Conflicting data = lower confidence

Output: Scored knowledge ready for knowledge base storage.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict

from .raffinatore import RefinedItem
from .correlatore import Correlation, OceanEvent


@dataclass
class FactorScore:
    """Score for a single factor/index."""
    factor: str
    score: float  # 0-10
    confidence: float  # 0-1
    data_points: int
    reasoning: str


@dataclass
class ScoredKnowledge:
    """
    A scored piece of knowledge derived from correlations.
    
    This is the final output that goes into the knowledge base.
    """
    id: str
    event_id: str
    event_name: str
    
    # Multi-factor indices (0-10 scale)
    thermodynamics: float = 0.0
    anemometry: float = 0.0
    precipitation: float = 0.0
    cryosphere: float = 0.0
    oceanography: float = 0.0
    
    # Factor details
    factor_scores: List[Dict[str, Any]] = field(default_factory=list)
    
    # Overall scores
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # Data quality
    data_density: float = 0.0  # Items per time window
    source_diversity: float = 0.0  # Different source types
    temporal_coverage: float = 0.0  # Days covered
    
    # Supporting data
    precursor_count: int = 0
    concurrent_count: int = 0
    consequence_count: int = 0
    
    # Top correlations
    top_correlations: List[str] = field(default_factory=list)
    
    # Reasoning/explanation
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoredKnowledge":
        return cls(**data)


class KnowledgeScorer:
    """
    Knowledge Scorer - Transforms correlations into scored knowledge.
    
    Scoring Philosophy:
    - Scores are 0-10 scale
    - Confidence is 0-1 scale
    - High scores require multiple corroborating sources
    - Temporal proximity to event increases weight
    - Missing data results in low scores with low confidence
    """
    
    # Topic to factor mapping
    TOPIC_FACTOR_MAP = {
        "temperature": "thermodynamics",
        "sea_ice": "cryosphere",
        "atmosphere": "anemometry",
        "precipitation": "precipitation",
        "salinity": "oceanography",
        "currents": "oceanography",
        "sea_level": "oceanography",
    }
    
    # Minimum thresholds
    MIN_DATA_POINTS = 3  # Need at least 3 items for confidence
    MIN_CONFIDENCE = 0.3  # Below this = essentially unknown
    
    def __init__(
        self,
        correlations_dir: Path = None,
        refined_dir: Path = None,
        events_file: Path = None,
        output_dir: Path = None,
    ):
        base = Path(__file__).parent.parent.parent.parent / "data" / "pipeline"
        self.correlations_dir = correlations_dir or base / "correlations"
        self.refined_dir = refined_dir or base / "refined"
        self.events_file = events_file or base / "events.json"
        self.output_dir = output_dir or base / "knowledge"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 KnowledgeScorer initialized")
        print(f"   📂 Correlations: {self.correlations_dir}")
        print(f"   📂 Output: {self.output_dir}")
    
    # =========================================================================
    # Factor Scoring
    # =========================================================================
    
    def score_factor(
        self,
        factor: str,
        items: List[RefinedItem],
        correlations: List[Correlation],
    ) -> FactorScore:
        """
        Score a single factor based on relevant items and correlations.
        """
        # Filter items by topic
        relevant_topics = [
            topic for topic, fac in self.TOPIC_FACTOR_MAP.items()
            if fac == factor
        ]
        
        relevant_items = [
            item for item in items
            if any(t in item.topics for t in relevant_topics)
        ]
        
        data_points = len(relevant_items)
        
        if data_points == 0:
            return FactorScore(
                factor=factor,
                score=0.0,
                confidence=0.0,
                data_points=0,
                reasoning="No data available for this factor"
            )
        
        # Calculate base score from relevance scores
        avg_relevance = sum(i.relevance_score for i in relevant_items) / data_points
        
        # Weight by correlation quality
        relevant_item_ids = {i.id for i in relevant_items}
        relevant_corrs = [
            c for c in correlations 
            if c.item_id in relevant_item_ids
        ]
        
        if relevant_corrs:
            avg_corr_score = sum(c.overall_score for c in relevant_corrs) / len(relevant_corrs)
            avg_decay = sum(c.decay_weight for c in relevant_corrs) / len(relevant_corrs)
        else:
            avg_corr_score = 0.5
            avg_decay = 0.5
        
        # Combined score (0-10)
        base_score = (avg_relevance * 0.4 + avg_corr_score * 10 * 0.3 + avg_decay * 10 * 0.3)
        
        # Confidence based on data density
        if data_points >= 10:
            confidence = 0.9
        elif data_points >= 5:
            confidence = 0.7
        elif data_points >= 3:
            confidence = 0.5
        else:
            confidence = 0.3
            base_score *= 0.5  # Penalize low-data scores
        
        # Build reasoning
        precursors = sum(1 for c in relevant_corrs if c.hypothesis_type == "precursor")
        reasoning = f"{data_points} data points, {precursors} precursors"
        
        if confidence < 0.5:
            reasoning += " (LOW CONFIDENCE: sparse data)"
        
        return FactorScore(
            factor=factor,
            score=round(min(10, base_score), 1),
            confidence=round(confidence, 2),
            data_points=data_points,
            reasoning=reasoning
        )
    
    # =========================================================================
    # Data Quality Scoring
    # =========================================================================
    
    def calculate_data_density(
        self, 
        correlations: List[Correlation]
    ) -> float:
        """Calculate data density (items per 30-day window)."""
        if not correlations:
            return 0.0
        
        # Get unique time windows
        days = sorted(set(abs(c.days_from_event) for c in correlations))
        if not days:
            return 0.0
        
        time_span = max(days) - min(days) + 1
        windows = max(1, time_span / 30)
        
        return len(correlations) / windows
    
    def calculate_source_diversity(
        self, 
        items: List[RefinedItem]
    ) -> float:
        """Calculate diversity of source types (0-1)."""
        if not items:
            return 0.0
        
        source_types = set(i.source_type for i in items)
        source_names = set(i.source_name for i in items if i.source_name)
        
        # More diversity = better
        type_score = min(1.0, len(source_types) / 3)  # Max 3 types
        name_score = min(1.0, len(source_names) / 10)  # Max 10 sources
        
        return (type_score + name_score) / 2
    
    def calculate_temporal_coverage(
        self, 
        correlations: List[Correlation]
    ) -> float:
        """Calculate temporal coverage (0-1 based on days covered)."""
        if not correlations:
            return 0.0
        
        days = set(c.days_from_event for c in correlations)
        
        # Good coverage = items before, during, and after
        has_before = any(d > 0 for c in correlations if c.temporal_relation == "before" for d in [c.days_from_event])
        has_during = any(c.temporal_relation == "during" for c in correlations)
        has_after = any(c.temporal_relation == "after" for c in correlations)
        
        coverage = (has_before + has_during + has_after) / 3
        
        # Also consider span
        day_span = max(days) - min(days) if len(days) > 1 else 0
        span_score = min(1.0, day_span / 90)  # 90 days = full coverage
        
        return (coverage + span_score) / 2
    
    # =========================================================================
    # Main Scoring Process
    # =========================================================================
    
    def score_event(
        self,
        event: OceanEvent,
        correlations: List[Correlation],
        items: List[RefinedItem],
    ) -> ScoredKnowledge:
        """Score knowledge for a single event."""
        
        # Get correlations for this event
        event_corrs = [c for c in correlations if c.event_id == event.id]
        
        # Get items for these correlations
        item_ids = {c.item_id for c in event_corrs}
        event_items = [i for i in items if i.id in item_ids]
        
        # Score each factor
        factors = ["thermodynamics", "anemometry", "precipitation", 
                   "cryosphere", "oceanography"]
        
        factor_scores = {}
        for factor in factors:
            fs = self.score_factor(factor, event_items, event_corrs)
            factor_scores[factor] = fs
        
        # Calculate overall score (weighted average)
        weights = {
            "thermodynamics": 1.5,  # More important for ocean events
            "oceanography": 1.5,
            "cryosphere": 1.2,
            "anemometry": 0.8,
            "precipitation": 1.0,
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(
            factor_scores[f].score * weights[f] * factor_scores[f].confidence
            for f in factors
        )
        confidence_sum = sum(
            factor_scores[f].confidence * weights[f]
            for f in factors
        )
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        avg_confidence = confidence_sum / total_weight if total_weight > 0 else 0
        
        # Count by hypothesis type
        precursors = sum(1 for c in event_corrs if c.hypothesis_type == "precursor")
        concurrent = sum(1 for c in event_corrs if c.hypothesis_type == "concurrent")
        consequences = sum(1 for c in event_corrs if c.hypothesis_type == "consequence")
        
        # Data quality metrics
        data_density = self.calculate_data_density(event_corrs)
        source_diversity = self.calculate_source_diversity(event_items)
        temporal_coverage = self.calculate_temporal_coverage(event_corrs)
        
        # Top correlations
        top_corrs = sorted(event_corrs, key=lambda c: c.overall_score, reverse=True)[:5]
        
        # Key findings
        findings = []
        
        # Check for precursor signals
        if precursors > 0:
            findings.append(f"Found {precursors} potential precursor signal(s)")
        
        # Strong factors
        strong_factors = [f for f, fs in factor_scores.items() 
                        if fs.score >= 7 and fs.confidence >= 0.5]
        if strong_factors:
            findings.append(f"Strong evidence in: {', '.join(strong_factors)}")
        
        # Weak factors
        weak_factors = [f for f, fs in factor_scores.items() 
                       if fs.data_points == 0]
        if weak_factors:
            findings.append(f"No data for: {', '.join(weak_factors)}")
        
        # Summary
        summary = f"Event '{event.name}' has {len(event_corrs)} correlations "
        summary += f"with overall score {overall_score:.1f}/10 "
        summary += f"(confidence: {avg_confidence:.0%})"
        
        # Create scored knowledge
        knowledge = ScoredKnowledge(
            id=f"knowledge_{event.id}",
            event_id=event.id,
            event_name=event.name,
            thermodynamics=factor_scores["thermodynamics"].score,
            anemometry=factor_scores["anemometry"].score,
            precipitation=factor_scores["precipitation"].score,
            cryosphere=factor_scores["cryosphere"].score,
            oceanography=factor_scores["oceanography"].score,
            factor_scores=[asdict(fs) for fs in factor_scores.values()],
            overall_score=round(overall_score, 1),
            confidence=round(avg_confidence, 2),
            data_density=round(data_density, 2),
            source_diversity=round(source_diversity, 2),
            temporal_coverage=round(temporal_coverage, 2),
            precursor_count=precursors,
            concurrent_count=concurrent,
            consequence_count=consequences,
            top_correlations=[c.id for c in top_corrs],
            summary=summary,
            key_findings=findings,
        )
        
        return knowledge
    
    def score_all(self) -> List[ScoredKnowledge]:
        """Score knowledge for all events."""
        
        # Load data
        correlations = []
        for file in self.correlations_dir.glob("*.json"):
            try:
                with open(file) as f:
                    correlations.append(Correlation.from_dict(json.load(f)))
            except:
                pass
        
        items = []
        for file in self.refined_dir.glob("*.json"):
            try:
                with open(file) as f:
                    items.append(RefinedItem.from_dict(json.load(f)))
            except:
                pass
        
        events = []
        if self.events_file.exists():
            with open(self.events_file) as f:
                events_data = json.load(f)
            events = [OceanEvent.from_dict(e) for e in events_data]
        
        print("\n" + "="*60)
        print("🧠 KNOWLEDGE SCORER - MULTI-FACTOR SCORING")
        print("="*60)
        print(f"📥 Correlations: {len(correlations)}")
        print(f"📥 Items: {len(items)}")
        print(f"📅 Events: {len(events)}")
        
        # Score each event
        knowledge_items = []
        
        for event in events:
            knowledge = self.score_event(event, correlations, items)
            knowledge_items.append(knowledge)
            
            # Save
            output_file = self.output_dir / f"{knowledge.id}.json"
            with open(output_file, "w") as f:
                json.dump(knowledge.to_dict(), f, indent=2)
        
        # Print summary
        print(f"\n📊 KNOWLEDGE SCORES:")
        print("-" * 70)
        print(f"{'Event':<35} | {'Score':>6} | {'Conf':>5} | {'Precursors':>10}")
        print("-" * 70)
        
        for k in sorted(knowledge_items, key=lambda x: -x.overall_score):
            print(f"{k.event_name[:35]:<35} | {k.overall_score:>6.1f} | {k.confidence:>5.0%} | {k.precursor_count:>10}")
        
        # Factor summary
        print(f"\n📈 FACTOR SCORES (average across events):")
        factors = ["thermodynamics", "anemometry", "precipitation", 
                   "cryosphere", "oceanography"]
        
        for factor in factors:
            avg = sum(getattr(k, factor) for k in knowledge_items) / len(knowledge_items) if knowledge_items else 0
            print(f"   {factor:15s}: {avg:.1f}/10")
        
        return knowledge_items
    
    def get_knowledge(self, event_id: str = None) -> List[ScoredKnowledge]:
        """Load knowledge from disk."""
        knowledge = []
        for file in self.output_dir.glob("*.json"):
            try:
                with open(file) as f:
                    k = ScoredKnowledge.from_dict(json.load(f))
                    if event_id is None or k.event_id == event_id:
                        knowledge.append(k)
            except:
                pass
        return knowledge
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge statistics."""
        knowledge = self.get_knowledge()
        
        if not knowledge:
            return {"total": 0}
        
        avg_score = sum(k.overall_score for k in knowledge) / len(knowledge)
        avg_conf = sum(k.confidence for k in knowledge) / len(knowledge)
        total_precursors = sum(k.precursor_count for k in knowledge)
        
        factor_avgs = {}
        for factor in ["thermodynamics", "anemometry", "precipitation", 
                       "cryosphere", "oceanography"]:
            factor_avgs[factor] = round(
                sum(getattr(k, factor) for k in knowledge) / len(knowledge), 1
            )
        
        return {
            "total_events": len(knowledge),
            "avg_overall_score": round(avg_score, 1),
            "avg_confidence": round(avg_conf, 2),
            "total_precursors": total_precursors,
            "factor_averages": factor_avgs,
        }

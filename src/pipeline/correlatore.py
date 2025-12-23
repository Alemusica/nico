"""
🔗 Correlatore Module (Event-Data Correlation)
===============================================

Links refined data to oceanographic events with:

1. TEMPORAL CORRELATION - Time proximity to events
2. SPATIAL CORRELATION - Geographic co-location
3. CAUSAL HYPOTHESIS - Potential cause-effect relationships
4. DECAY WEIGHTING - Time-decayed relevance (90 days = 37% weight)

Output: Correlations ready for knowledge scoring.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict

from .raffinatore import RefinedItem


@dataclass
class OceanEvent:
    """A known oceanographic event or phenomenon."""
    id: str
    name: str
    event_type: str  # storm, anomaly, ice_event, transport_event
    
    # Temporal
    start_date: str
    end_date: Optional[str] = None
    duration_days: Optional[int] = None
    
    # Spatial
    regions: List[str] = field(default_factory=list)
    lat_center: Optional[float] = None
    lon_center: Optional[float] = None
    
    # Characteristics
    magnitude: Optional[float] = None  # Normalized 0-10
    description: str = ""
    
    # Related measurements
    measurements: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OceanEvent":
        return cls(**data)


@dataclass
class Correlation:
    """A correlation between data and an event."""
    id: str
    item_id: str  # RefinedItem ID
    event_id: str  # OceanEvent ID
    
    # Correlation scores (0-1)
    temporal_score: float = 0.0
    spatial_score: float = 0.0
    topical_score: float = 0.0
    overall_score: float = 0.0
    
    # Temporal details
    days_from_event: int = 0
    temporal_relation: str = "unknown"  # before, during, after
    decay_weight: float = 1.0  # exp(-days/90)
    
    # Causal hypothesis
    hypothesis_type: str = "unknown"  # precursor, concurrent, consequence
    causal_confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correlation":
        return cls(**data)


class Correlatore:
    """
    Event-Data Correlator - Links data to events with temporal decay.
    
    Key Formula: 
        decay_weight = exp(-days / DECAY_CONSTANT)
        
    With DECAY_CONSTANT=90:
        - 0 days: 100%
        - 30 days: 72%
        - 60 days: 51%
        - 90 days: 37%
        - 180 days: 14%
    """
    
    # Temporal decay constant (days)
    DECAY_CONSTANT = 90
    
    # Event type to topic mapping
    EVENT_TOPIC_MAP = {
        "ice_event": ["sea_ice", "temperature"],
        "storm": ["atmosphere", "sea_level", "currents"],
        "anomaly": ["temperature", "salinity", "currents"],
        "transport_event": ["currents", "temperature", "salinity"],
        "precipitation_event": ["precipitation", "salinity"],
    }
    
    # Region adjacency for spatial scoring
    REGION_ADJACENCY = {
        "fram_strait": ["arctic", "greenland", "norwegian_sea", "barents_sea"],
        "barents_sea": ["arctic", "norwegian_sea", "fram_strait", "siberian"],
        "norwegian_sea": ["fram_strait", "barents_sea", "north_atlantic"],
        "north_atlantic": ["norwegian_sea", "greenland"],
        "bering_strait": ["arctic", "beaufort_sea"],
        "beaufort_sea": ["arctic", "bering_strait", "siberian"],
        "siberian": ["arctic", "beaufort_sea", "barents_sea"],
        "greenland": ["arctic", "fram_strait", "north_atlantic"],
        "arctic": ["fram_strait", "barents_sea", "siberian", "beaufort_sea", 
                   "bering_strait", "greenland"],
    }
    
    def __init__(
        self,
        refined_dir: Path = None,
        events_file: Path = None,
        output_dir: Path = None,
    ):
        base = Path(__file__).parent.parent.parent.parent / "data" / "pipeline"
        self.refined_dir = refined_dir or base / "refined"
        self.events_file = events_file or base / "events.json"
        self.output_dir = output_dir or base / "correlations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize events (create sample if needed)
        self._init_events()
        
        print(f"🔗 Correlatore initialized")
        print(f"   📂 Refined data: {self.refined_dir}")
        print(f"   📂 Events: {self.events_file}")
        print(f"   📂 Output: {self.output_dir}")
        print(f"   ⏱️ Decay constant: {self.DECAY_CONSTANT} days")
    
    def _init_events(self):
        """Initialize sample events if file doesn't exist."""
        if not self.events_file.exists():
            self.events_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Sample events for Arctic research
            sample_events = [
                OceanEvent(
                    id="evt_001",
                    name="2023 Arctic Sea Ice Minimum",
                    event_type="ice_event",
                    start_date="2023-09-10",
                    end_date="2023-09-20",
                    regions=["arctic", "beaufort_sea", "siberian"],
                    magnitude=7.5,
                    description="Annual Arctic sea ice minimum, 6th lowest on record",
                    measurements={"extent_km2": 4.23e6, "anomaly_km2": -1.2e6},
                ),
                OceanEvent(
                    id="evt_002",
                    name="Atlantic Water Pulse Fram Strait",
                    event_type="transport_event",
                    start_date="2023-06-01",
                    end_date="2023-08-15",
                    regions=["fram_strait", "barents_sea"],
                    magnitude=6.0,
                    description="Enhanced Atlantic water transport into Arctic",
                    measurements={"transport_sv": 8.5, "temperature_anomaly": 1.2},
                ),
                OceanEvent(
                    id="evt_003",
                    name="Barents Sea Marine Heatwave",
                    event_type="anomaly",
                    start_date="2023-07-15",
                    end_date="2023-08-30",
                    regions=["barents_sea", "norwegian_sea"],
                    magnitude=5.5,
                    description="Persistent temperature anomaly in Barents Sea",
                    measurements={"sst_anomaly": 3.2, "duration_days": 46},
                ),
                OceanEvent(
                    id="evt_004",
                    name="Arctic Storm December 2022",
                    event_type="storm",
                    start_date="2022-12-18",
                    end_date="2022-12-25",
                    regions=["arctic", "fram_strait", "greenland"],
                    magnitude=8.0,
                    description="Major polar low affecting Arctic basin",
                    measurements={"min_pressure_hpa": 968, "max_wind_ms": 35},
                ),
            ]
            
            events_data = [e.to_dict() for e in sample_events]
            with open(self.events_file, "w") as f:
                json.dump(events_data, f, indent=2)
    
    # =========================================================================
    # Temporal Correlation
    # =========================================================================
    
    def calculate_temporal_score(
        self, 
        item: RefinedItem, 
        event: OceanEvent
    ) -> Tuple[float, int, str]:
        """
        Calculate temporal correlation between item and event.
        
        Returns:
            (score, days_from_event, relation)
        """
        # Parse dates
        item_date = None
        if item.published_date:
            try:
                item_date = datetime.fromisoformat(
                    item.published_date.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except:
                pass
        
        event_start = datetime.fromisoformat(event.start_date)
        event_end = None
        if event.end_date:
            event_end = datetime.fromisoformat(event.end_date)
        
        if not item_date:
            return (0.0, 999, "unknown")
        
        # Calculate days from event
        if event_end and event_start <= item_date <= event_end:
            days_from = 0
            relation = "during"
        elif item_date < event_start:
            days_from = (event_start - item_date).days
            relation = "before"
        else:
            ref_date = event_end or event_start
            days_from = (item_date - ref_date).days
            relation = "after"
        
        # Score with decay
        score = math.exp(-days_from / self.DECAY_CONSTANT)
        
        return (score, days_from, relation)
    
    def calculate_decay_weight(self, days: int) -> float:
        """Calculate temporal decay weight."""
        return math.exp(-abs(days) / self.DECAY_CONSTANT)
    
    # =========================================================================
    # Spatial Correlation
    # =========================================================================
    
    def calculate_spatial_score(
        self, 
        item: RefinedItem, 
        event: OceanEvent
    ) -> float:
        """
        Calculate spatial correlation based on region overlap.
        
        Scoring:
        - Same region: 1.0
        - Adjacent region: 0.7
        - Remote region: 0.3
        - No match: 0.0
        """
        if not item.locations or not event.regions:
            return 0.0
        
        max_score = 0.0
        
        for item_loc in item.locations:
            for event_region in event.regions:
                if item_loc == event_region:
                    max_score = max(max_score, 1.0)
                elif event_region in self.REGION_ADJACENCY.get(item_loc, []):
                    max_score = max(max_score, 0.7)
                elif item_loc in self.REGION_ADJACENCY.get(event_region, []):
                    max_score = max(max_score, 0.7)
                elif item_loc == "arctic" or event_region == "arctic":
                    max_score = max(max_score, 0.5)
                else:
                    max_score = max(max_score, 0.3)
        
        return max_score
    
    # =========================================================================
    # Topical Correlation
    # =========================================================================
    
    def calculate_topical_score(
        self, 
        item: RefinedItem, 
        event: OceanEvent
    ) -> float:
        """
        Calculate topical correlation based on topic overlap.
        """
        if not item.topics:
            return 0.0
        
        relevant_topics = self.EVENT_TOPIC_MAP.get(event.event_type, [])
        
        overlap = len(set(item.topics) & set(relevant_topics))
        max_possible = max(len(relevant_topics), 1)
        
        return min(1.0, overlap / max_possible)
    
    # =========================================================================
    # Causal Hypothesis Generation
    # =========================================================================
    
    def generate_hypothesis(
        self, 
        item: RefinedItem, 
        event: OceanEvent,
        temporal_relation: str,
        days_from_event: int,
    ) -> Tuple[str, float, List[str]]:
        """
        Generate causal hypothesis for the correlation.
        
        Returns:
            (hypothesis_type, confidence, supporting_evidence)
        """
        evidence = []
        confidence = 0.0
        
        # Determine hypothesis type based on temporal relation
        if temporal_relation == "before" and days_from_event <= 30:
            hypothesis_type = "precursor"
            confidence = 0.6
            evidence.append(f"Published {days_from_event} days before event")
            
            # Boost if topics align with event drivers
            if event.event_type == "ice_event" and "temperature" in item.topics:
                confidence += 0.2
                evidence.append("Temperature topic may indicate thermal precursor")
            
            if event.event_type == "storm" and "atmosphere" in item.topics:
                confidence += 0.2
                evidence.append("Atmospheric topic may indicate storm precursor")
        
        elif temporal_relation == "during":
            hypothesis_type = "concurrent"
            confidence = 0.8
            evidence.append("Published during event period")
        
        elif temporal_relation == "after" and days_from_event <= 90:
            hypothesis_type = "consequence"
            confidence = 0.5
            evidence.append(f"Published {days_from_event} days after event")
            
            # Reports/analyses tend to follow events
            if item.source_type == "paper":
                confidence += 0.2
                evidence.append("Scientific paper may analyze event consequences")
        
        else:
            hypothesis_type = "weak"
            confidence = 0.2
            evidence.append(f"Temporally distant ({days_from_event} days)")
        
        # Measurement evidence
        if item.measurements:
            for m in item.measurements:
                if m["type"] == "temperature" and event.event_type in ["ice_event", "anomaly"]:
                    evidence.append(f"Contains temperature measurement: {m['value']}{m['unit']}")
                    confidence = min(1.0, confidence + 0.1)
        
        return (hypothesis_type, min(1.0, confidence), evidence)
    
    # =========================================================================
    # Main Correlation Process
    # =========================================================================
    
    def correlate_item(
        self, 
        item: RefinedItem, 
        events: List[OceanEvent]
    ) -> List[Correlation]:
        """Correlate a single item with all events."""
        correlations = []
        
        for event in events:
            # Calculate scores
            temp_score, days_from, relation = self.calculate_temporal_score(item, event)
            spatial_score = self.calculate_spatial_score(item, event)
            topical_score = self.calculate_topical_score(item, event)
            
            # Only create correlation if there's some relevance
            overall = (temp_score * 0.4 + spatial_score * 0.3 + topical_score * 0.3)
            
            if overall < 0.1:
                continue
            
            # Generate hypothesis
            hyp_type, hyp_conf, evidence = self.generate_hypothesis(
                item, event, relation, days_from
            )
            
            # Create correlation
            corr = Correlation(
                id=f"corr_{item.id}_{event.id}",
                item_id=item.id,
                event_id=event.id,
                temporal_score=round(temp_score, 3),
                spatial_score=round(spatial_score, 3),
                topical_score=round(topical_score, 3),
                overall_score=round(overall, 3),
                days_from_event=days_from,
                temporal_relation=relation,
                decay_weight=round(self.calculate_decay_weight(days_from), 3),
                hypothesis_type=hyp_type,
                causal_confidence=round(hyp_conf, 3),
                supporting_evidence=evidence,
            )
            
            correlations.append(corr)
        
        return correlations
    
    def correlate_all(
        self, 
        refined_items: List[RefinedItem] = None
    ) -> List[Correlation]:
        """Correlate all refined items with all events."""
        
        # Load refined items if not provided
        if refined_items is None:
            refined_items = []
            for file in self.refined_dir.glob("*.json"):
                try:
                    with open(file) as f:
                        refined_items.append(RefinedItem.from_dict(json.load(f)))
                except:
                    pass
        
        # Load events
        with open(self.events_file) as f:
            events_data = json.load(f)
        events = [OceanEvent.from_dict(e) for e in events_data]
        
        print("\n" + "="*60)
        print("🔗 CORRELATORE - EVENT CORRELATION")
        print("="*60)
        print(f"📥 Refined items: {len(refined_items)}")
        print(f"📅 Events: {len(events)}")
        
        # Correlate all
        all_correlations = []
        
        for item in refined_items:
            correlations = self.correlate_item(item, events)
            all_correlations.extend(correlations)
        
        print(f"   🔗 Total correlations: {len(all_correlations)}")
        
        # Save correlations
        for corr in all_correlations:
            output_file = self.output_dir / f"{corr.id}.json"
            with open(output_file, "w") as f:
                json.dump(corr.to_dict(), f, indent=2)
        
        # Summary by event
        print(f"\n📊 CORRELATIONS BY EVENT:")
        event_corrs = defaultdict(list)
        for c in all_correlations:
            event_corrs[c.event_id].append(c)
        
        for event_id, corrs in event_corrs.items():
            event = next((e for e in events if e.id == event_id), None)
            name = event.name if event else event_id
            precursors = sum(1 for c in corrs if c.hypothesis_type == "precursor")
            print(f"   {name[:40]:40s} | {len(corrs):3d} corrs | {precursors:2d} precursors")
        
        # Summary by hypothesis type
        print(f"\n📈 CORRELATIONS BY HYPOTHESIS TYPE:")
        hyp_counts = defaultdict(int)
        for c in all_correlations:
            hyp_counts[c.hypothesis_type] += 1
        
        for hyp_type, count in sorted(hyp_counts.items(), key=lambda x: -x[1]):
            print(f"   {hyp_type:15s}: {count}")
        
        return all_correlations
    
    def add_event(self, event: OceanEvent):
        """Add a new event to the events file."""
        # Load existing
        if self.events_file.exists():
            with open(self.events_file) as f:
                events = json.load(f)
        else:
            events = []
        
        # Add new
        events.append(event.to_dict())
        
        # Save
        with open(self.events_file, "w") as f:
            json.dump(events, f, indent=2)
        
        print(f"   ✅ Added event: {event.name}")
    
    def get_correlations(self) -> List[Correlation]:
        """Load all correlations from disk."""
        correlations = []
        for file in self.output_dir.glob("*.json"):
            try:
                with open(file) as f:
                    correlations.append(Correlation.from_dict(json.load(f)))
            except:
                pass
        return correlations
    
    def get_precursors(self, event_id: str, max_days: int = 30) -> List[Correlation]:
        """Get precursor correlations for an event."""
        correlations = self.get_correlations()
        return [
            c for c in correlations 
            if c.event_id == event_id 
            and c.hypothesis_type == "precursor"
            and c.days_from_event <= max_days
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get correlation statistics."""
        correlations = self.get_correlations()
        
        hyp_counts = defaultdict(int)
        relation_counts = defaultdict(int)
        
        for c in correlations:
            hyp_counts[c.hypothesis_type] += 1
            relation_counts[c.temporal_relation] += 1
        
        avg_score = sum(c.overall_score for c in correlations) / len(correlations) if correlations else 0
        
        return {
            "total_correlations": len(correlations),
            "avg_overall_score": round(avg_score, 3),
            "by_hypothesis": dict(hyp_counts),
            "by_relation": dict(relation_counts),
            "precursors": hyp_counts.get("precursor", 0),
        }

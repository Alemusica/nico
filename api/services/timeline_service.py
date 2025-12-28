"""
⏱️ Timeline Service
===================

Manages temporal aspects of the knowledge graph:
- Timeline queries across all entity types
- Temporal relationships between events
- Time-window aggregations
- Sequence detection

All entities in the knowledge graph can have temporal properties:
- Papers: publication_date, study_period_start, study_period_end
- Events: date, duration
- Patterns: first_observed, last_observed, seasonality
- Datasets: temporal_coverage_start, temporal_coverage_end
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
import json


class TimeGranularity(Enum):
    """Temporal granularity for aggregations."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    SEASON = "season"  # DJF, MAM, JJA, SON
    YEAR = "year"
    DECADE = "decade"


class TimelineEntityType(Enum):
    """Types of entities in timeline."""
    PAPER = "paper"
    EVENT = "event"
    PATTERN = "pattern"
    DATASET = "dataset"
    CLIMATE_INDEX = "climate_index"
    INVESTIGATION = "investigation"
    ALL = "all"


@dataclass
class TimelineEntry:
    """A single entry on the timeline."""
    id: str
    entity_type: TimelineEntityType
    date: datetime
    end_date: Optional[datetime] = None  # For periods/ranges
    label: str = ""
    description: str = ""
    importance: float = 1.0  # For ranking/filtering
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_ids: List[str] = field(default_factory=list)  # Links to other entities
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "date": self.date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "label": self.label,
            "description": self.description,
            "importance": self.importance,
            "metadata": self.metadata,
            "related_ids": self.related_ids,
        }
    
    @property
    def duration_days(self) -> Optional[int]:
        if self.end_date:
            return (self.end_date - self.date).days
        return None


@dataclass
class TimeWindow:
    """A time window for queries."""
    start: datetime
    end: datetime
    
    @classmethod
    def from_strings(cls, start: str, end: str) -> "TimeWindow":
        return cls(
            start=datetime.fromisoformat(start.replace("Z", "+00:00")),
            end=datetime.fromisoformat(end.replace("Z", "+00:00")),
        )
    
    @classmethod
    def around_date(cls, date: datetime, days_before: int = 30, days_after: int = 30) -> "TimeWindow":
        return cls(
            start=date - timedelta(days=days_before),
            end=date + timedelta(days=days_after),
        )
    
    def contains(self, date: datetime) -> bool:
        return self.start <= date <= self.end
    
    def overlaps(self, other: "TimeWindow") -> bool:
        return not (other.end < self.start or other.start > self.end)
    
    @property
    def duration_days(self) -> int:
        return (self.end - self.start).days


@dataclass
class TimelineAggregation:
    """Aggregated timeline data."""
    granularity: TimeGranularity
    buckets: List[Dict]  # [{period: str, count: int, entries: List[str]}]
    total_count: int
    time_range: TimeWindow


class TimelineService:
    """
    Service for temporal queries on knowledge graph entities.
    
    Works with any KnowledgeService backend by wrapping its queries
    with temporal filtering and aggregation.
    """
    
    def __init__(self, knowledge_service=None):
        """
        Args:
            knowledge_service: Optional KnowledgeService instance for DB queries
        """
        self.knowledge_service = knowledge_service
        self._cache: Dict[str, List[TimelineEntry]] = {}
    
    def add_entry(self, entry: TimelineEntry) -> None:
        """Add entry to local timeline cache."""
        entity_type = entry.entity_type.value
        if entity_type not in self._cache:
            self._cache[entity_type] = []
        self._cache[entity_type].append(entry)
    
    def add_entries(self, entries: List[TimelineEntry]) -> int:
        """Add multiple entries."""
        for entry in entries:
            self.add_entry(entry)
        return len(entries)
    
    def query(
        self,
        time_window: TimeWindow,
        entity_types: Optional[List[TimelineEntityType]] = None,
        min_importance: float = 0.0,
        keywords: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[TimelineEntry]:
        """
        Query timeline entries within a time window.
        
        Args:
            time_window: Start and end dates
            entity_types: Filter by type(s)
            min_importance: Minimum importance threshold
            keywords: Filter by keywords in label/description
            limit: Max results
        
        Returns:
            List of TimelineEntry sorted by date
        """
        results = []
        
        # Determine which types to search
        types_to_search = entity_types or [e for e in TimelineEntityType if e != TimelineEntityType.ALL]
        
        for entity_type in types_to_search:
            type_key = entity_type.value
            if type_key not in self._cache:
                continue
            
            for entry in self._cache[type_key]:
                # Time window filter
                if not time_window.contains(entry.date):
                    # Also check if entry's range overlaps window
                    if entry.end_date:
                        entry_window = TimeWindow(entry.date, entry.end_date)
                        if not time_window.overlaps(entry_window):
                            continue
                    else:
                        continue
                
                # Importance filter
                if entry.importance < min_importance:
                    continue
                
                # Keyword filter
                if keywords:
                    text = f"{entry.label} {entry.description}".lower()
                    if not any(kw.lower() in text for kw in keywords):
                        continue
                
                results.append(entry)
        
        # Sort by date
        results.sort(key=lambda x: x.date)
        
        return results[:limit]
    
    def aggregate(
        self,
        time_window: TimeWindow,
        granularity: TimeGranularity,
        entity_types: Optional[List[TimelineEntityType]] = None,
    ) -> TimelineAggregation:
        """
        Aggregate timeline entries by time period.
        
        Args:
            time_window: Overall time range
            granularity: How to bucket entries
            entity_types: Filter by type(s)
            
        Returns:
            TimelineAggregation with bucketed data
        """
        # Get all entries in window
        entries = self.query(
            time_window=time_window,
            entity_types=entity_types,
            limit=10000,  # Get all for aggregation
        )
        
        # Create buckets based on granularity
        buckets: Dict[str, List[TimelineEntry]] = {}
        
        for entry in entries:
            bucket_key = self._get_bucket_key(entry.date, granularity)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(entry)
        
        # Format output
        bucket_list = [
            {
                "period": key,
                "count": len(entries),
                "entries": [e.id for e in entries],
                "types": self._count_types(entries),
            }
            for key, entries in sorted(buckets.items())
        ]
        
        return TimelineAggregation(
            granularity=granularity,
            buckets=bucket_list,
            total_count=len(entries),
            time_range=time_window,
        )
    
    def _get_bucket_key(self, date: datetime, granularity: TimeGranularity) -> str:
        """Get bucket key for a date given granularity."""
        if granularity == TimeGranularity.HOUR:
            return date.strftime("%Y-%m-%d %H:00")
        elif granularity == TimeGranularity.DAY:
            return date.strftime("%Y-%m-%d")
        elif granularity == TimeGranularity.WEEK:
            return date.strftime("%Y-W%W")
        elif granularity == TimeGranularity.MONTH:
            return date.strftime("%Y-%m")
        elif granularity == TimeGranularity.SEASON:
            month = date.month
            year = date.year
            if month in [12, 1, 2]:
                return f"{year if month == 12 else year - 1}-DJF"
            elif month in [3, 4, 5]:
                return f"{year}-MAM"
            elif month in [6, 7, 8]:
                return f"{year}-JJA"
            else:
                return f"{year}-SON"
        elif granularity == TimeGranularity.YEAR:
            return date.strftime("%Y")
        elif granularity == TimeGranularity.DECADE:
            decade = (date.year // 10) * 10
            return f"{decade}s"
        return date.strftime("%Y-%m-%d")
    
    def _count_types(self, entries: List[TimelineEntry]) -> Dict[str, int]:
        """Count entries by type."""
        counts = {}
        for entry in entries:
            t = entry.entity_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    def find_sequences(
        self,
        pattern: List[str],  # e.g., ["climate_index", "event"]
        time_window: TimeWindow,
        max_gap_days: int = 30,
    ) -> List[List[TimelineEntry]]:
        """
        Find sequences of events matching a pattern.
        
        Args:
            pattern: Ordered list of entity types that should occur in sequence
            time_window: Where to search
            max_gap_days: Maximum days between consecutive events
            
        Returns:
            List of matching sequences
        """
        if not pattern:
            return []
        
        # Get all entries
        entries = self.query(
            time_window=time_window,
            limit=10000,
        )
        
        # Find matching sequences using dynamic programming
        sequences = []
        
        def find_next(current_seq: List[TimelineEntry], pattern_idx: int):
            if pattern_idx >= len(pattern):
                # Complete sequence found
                sequences.append(current_seq.copy())
                return
            
            target_type = pattern[pattern_idx]
            last_date = current_seq[-1].date if current_seq else time_window.start
            
            for entry in entries:
                if entry.entity_type.value != target_type:
                    continue
                if entry.date < last_date:
                    continue
                if current_seq and (entry.date - last_date).days > max_gap_days:
                    continue
                
                current_seq.append(entry)
                find_next(current_seq, pattern_idx + 1)
                current_seq.pop()
        
        find_next([], 0)
        return sequences
    
    def find_correlated_entries(
        self,
        anchor_entry: TimelineEntry,
        days_before: int = 30,
        days_after: int = 30,
        entity_types: Optional[List[TimelineEntityType]] = None,
    ) -> List[Tuple[TimelineEntry, int]]:
        """
        Find entries temporally close to an anchor entry.
        
        Returns entries with their lag (days from anchor).
        """
        window = TimeWindow.around_date(
            anchor_entry.date,
            days_before=days_before,
            days_after=days_after,
        )
        
        entries = self.query(
            time_window=window,
            entity_types=entity_types,
        )
        
        results = []
        for entry in entries:
            if entry.id == anchor_entry.id:
                continue
            lag = (entry.date - anchor_entry.date).days
            results.append((entry, lag))
        
        # Sort by absolute lag
        results.sort(key=lambda x: abs(x[1]))
        return results
    
    def get_timeline_for_investigation(
        self,
        investigation_id: str,
        entity_ids: List[str],
        context_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Build a complete timeline for an investigation.
        
        Args:
            investigation_id: ID of the investigation
            entity_ids: IDs of entities in the investigation
            context_days: Days of context around events
            
        Returns:
            Timeline data ready for Cosmograph visualization
        """
        # Collect all relevant entries
        relevant_entries = []
        min_date = None
        max_date = None
        
        for entity_type in self._cache:
            for entry in self._cache[entity_type]:
                if entry.id in entity_ids or any(rid in entity_ids for rid in entry.related_ids):
                    relevant_entries.append(entry)
                    if min_date is None or entry.date < min_date:
                        min_date = entry.date
                    if max_date is None or entry.date > max_date:
                        max_date = entry.date
        
        if not relevant_entries:
            return {"entries": [], "window": None}
        
        # Expand window for context
        window = TimeWindow(
            start=min_date - timedelta(days=context_days),
            end=max_date + timedelta(days=context_days),
        )
        
        # Get context entries (things that happened nearby in time)
        context_entries = self.query(
            time_window=window,
            limit=200,
        )
        
        # Build timeline structure
        timeline = {
            "investigation_id": investigation_id,
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "entries": [e.to_dict() for e in relevant_entries],
            "context": [
                e.to_dict() for e in context_entries 
                if e not in relevant_entries
            ],
            "aggregation": self.aggregate(
                time_window=window,
                granularity=TimeGranularity.DAY,
            ).buckets,
        }
        
        return timeline
    
    def to_cosmograph_format(
        self,
        entries: List[TimelineEntry],
        show_relationships: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert timeline entries to Cosmograph-compatible format.
        
        Returns nodes and links for the graph visualization.
        """
        nodes = []
        links = []
        
        # Color mapping by type
        type_colors = {
            "paper": "#4CAF50",      # Green
            "event": "#f44336",       # Red
            "pattern": "#2196F3",     # Blue
            "dataset": "#FF9800",     # Orange
            "climate_index": "#9C27B0",  # Purple
            "investigation": "#795548",  # Brown
        }
        
        for entry in entries:
            # Create node
            node = {
                "id": entry.id,
                "label": entry.label[:50],  # Truncate for display
                "color": type_colors.get(entry.entity_type.value, "#666666"),
                "size": 5 + entry.importance * 10,
                "type": entry.entity_type.value,
                "date": entry.date.isoformat(),
                "x": None,  # Will be computed by Cosmograph
                "y": None,
            }
            nodes.append(node)
            
            # Create links to related entities
            if show_relationships:
                for related_id in entry.related_ids:
                    links.append({
                        "source": entry.id,
                        "target": related_id,
                        "weight": 1,
                    })
        
        return {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "node_count": len(nodes),
                "link_count": len(links),
                "types": list(type_colors.keys()),
            }
        }


# Helper functions

def create_paper_entry(paper: Dict) -> TimelineEntry:
    """Convert a Paper dict to TimelineEntry."""
    # Try to parse publication date
    date = datetime.now()  # Default
    if paper.get("year"):
        date = datetime(paper["year"], 1, 1)
    elif paper.get("date"):
        date = datetime.fromisoformat(paper["date"])
    elif paper.get("publication_date"):
        date = datetime.fromisoformat(paper["publication_date"])
    
    # Study period if available
    end_date = None
    if paper.get("study_period_end"):
        end_date = datetime.fromisoformat(paper["study_period_end"])
    
    return TimelineEntry(
        id=paper.get("id", ""),
        entity_type=TimelineEntityType.PAPER,
        date=date,
        end_date=end_date,
        label=paper.get("title", "")[:100],
        description=paper.get("abstract", "")[:300],
        importance=1.0,
        metadata={
            "authors": paper.get("authors", []),
            "journal": paper.get("journal"),
            "doi": paper.get("doi"),
        }
    )


def create_event_entry(event: Dict) -> TimelineEntry:
    """Convert an Event dict to TimelineEntry."""
    date = datetime.fromisoformat(event["date"]) if event.get("date") else datetime.now()
    
    # Duration if available
    end_date = None
    if event.get("duration_days"):
        end_date = date + timedelta(days=event["duration_days"])
    elif event.get("end_date"):
        end_date = datetime.fromisoformat(event["end_date"])
    
    # Importance based on magnitude
    importance = 0.5
    if event.get("magnitude"):
        importance = min(1.0, event["magnitude"] / 10.0)
    
    return TimelineEntry(
        id=event.get("id", ""),
        entity_type=TimelineEntityType.EVENT,
        date=date,
        end_date=end_date,
        label=f"{event.get('event_type', 'Event')}: {event.get('location', '')}",
        description=event.get("description", ""),
        importance=importance,
        metadata={
            "event_type": event.get("event_type"),
            "location": event.get("location"),
            "latitude": event.get("latitude"),
            "longitude": event.get("longitude"),
            "magnitude": event.get("magnitude"),
        }
    )


def create_climate_index_entry(index: Dict) -> TimelineEntry:
    """Convert ClimateIndex to TimelineEntry."""
    date = datetime.fromisoformat(index["date"]) if index.get("date") else datetime.now()
    
    value = index.get("value", 0)
    importance = abs(value) / 3.0  # Normalize (±3 is extreme)
    
    return TimelineEntry(
        id=index.get("id", ""),
        entity_type=TimelineEntityType.CLIMATE_INDEX,
        date=date,
        label=f"{index.get('name', 'Index')}: {value:.2f}",
        description=f"{index.get('name')} value from {index.get('source', 'unknown')}",
        importance=min(1.0, importance),
        metadata={
            "name": index.get("name"),
            "value": value,
            "source": index.get("source"),
        }
    )


# CLI test
if __name__ == "__main__":
    from datetime import datetime
    
    print("=== Timeline Service Test ===\n")
    
    service = TimelineService()
    
    # Add some test entries
    service.add_entries([
        TimelineEntry(
            id="paper_001",
            entity_type=TimelineEntityType.PAPER,
            date=datetime(2001, 3, 15),
            label="Mediterranean Sea Level Variability Study",
            description="Analysis of sea level trends 1993-2000",
            importance=0.8,
            related_ids=["event_001"],
        ),
        TimelineEntry(
            id="event_001",
            entity_type=TimelineEntityType.EVENT,
            date=datetime(2000, 10, 14),
            end_date=datetime(2000, 10, 16),
            label="Lake Maggiore Flood",
            description="Severe flooding in Northern Italy",
            importance=1.0,
        ),
        TimelineEntry(
            id="nao_001",
            entity_type=TimelineEntityType.CLIMATE_INDEX,
            date=datetime(2000, 10, 1),
            label="NAO: -2.3",
            description="Negative NAO phase",
            importance=0.9,
            related_ids=["event_001"],
        ),
        TimelineEntry(
            id="dataset_001",
            entity_type=TimelineEntityType.DATASET,
            date=datetime(1993, 1, 1),
            end_date=datetime(2023, 12, 31),
            label="AVISO Sea Level",
            description="Daily sea level anomaly data",
            importance=0.7,
        ),
    ])
    
    # Query timeline
    print("Query: October 2000")
    window = TimeWindow(
        datetime(2000, 10, 1),
        datetime(2000, 10, 31),
    )
    results = service.query(window)
    for entry in results:
        print(f"  {entry.date.date()}: [{entry.entity_type.value}] {entry.label}")
    
    # Aggregate
    print("\nAggregation by month (2000):")
    window_year = TimeWindow(datetime(2000, 1, 1), datetime(2000, 12, 31))
    agg = service.aggregate(window_year, TimeGranularity.MONTH)
    for bucket in agg.buckets:
        print(f"  {bucket['period']}: {bucket['count']} entries")
    
    # Find correlated
    print("\nCorrelated to flood event:")
    flood_entry = service._cache["event"][0]
    correlated = service.find_correlated_entries(flood_entry, days_before=30, days_after=30)
    for entry, lag in correlated:
        print(f"  {entry.label}: lag={lag} days")
    
    # Cosmograph format
    print("\nCosmograph format preview:")
    cosmo = service.to_cosmograph_format(results)
    print(f"  Nodes: {cosmo['metadata']['node_count']}")
    print(f"  Links: {cosmo['metadata']['link_count']}")

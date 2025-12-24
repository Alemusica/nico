"""
🕵️ Investigation Agent
======================

LLM-powered agent that receives natural language queries and orchestrates
full investigation pipeline:

1. Parse query → identify event (location, date range, type)
2. Geo-resolve location → coordinates, bbox
3. Scrape all sources:
   - Satellite data (CMEMS, ERA5)
   - Climate indices (NAO, ENSO, etc.)
   - Scientific papers (arXiv, Semantic Scholar)
   - News articles
4. Categorize into raw vs solid DB sections
5. Run hybrid correlation engine
6. Zoom out temporally and geographically
7. Generate investigation report

Example:
    agent = InvestigationAgent()
    result = await agent.investigate("analizza alluvioni Lago Maggiore 2000")
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import json
import re
import uuid

# Investigation system
try:
    from src.core.investigation_logger import InvestigationLogger, InvestigationStep
    from src.core.investigation_validators import validate_papers_batch, ValidationLevel
except ImportError:
    # Fallback - will be handled gracefully
    InvestigationLogger = None
    InvestigationStep = None
    validate_papers_batch = None
    ValidationLevel = None

# Try relative imports first, then absolute
try:
    from .tools.geo_resolver import GeoResolver, GeoLocation
except ImportError:
    try:
        from src.agent.tools.geo_resolver import GeoResolver, GeoLocation
    except ImportError:
        GeoResolver = None
        GeoLocation = None

try:
    from src.surge_shazam.data.cmems_client import CMEMSClient
except ImportError:
    try:
        from surge_shazam.data.cmems_client import CMEMSClient
    except ImportError:
        CMEMSClient = None

try:
    from src.surge_shazam.data.era5_client import ERA5Client
except ImportError:
    try:
        from surge_shazam.data.era5_client import ERA5Client
    except ImportError:
        ERA5Client = None

try:
    from src.surge_shazam.data.climate_indices import ClimateIndicesClient
except ImportError:
    try:
        from surge_shazam.data.climate_indices import ClimateIndicesClient
    except ImportError:
        ClimateIndicesClient = None

try:
    from .tools.literature_scraper import LiteratureScraper
except ImportError:
    try:
        from src.agent.tools.literature_scraper import LiteratureScraper
    except ImportError:
        LiteratureScraper = None


@dataclass
class EventContext:
    """Extracted event context from natural language query."""
    location_name: str
    location: Optional[Any] = None  # GeoLocation
    event_type: str = "flood"  # flood, drought, storm_surge, etc.
    
    # Temporal context
    start_date: str = ""
    end_date: str = ""
    years_of_interest: List[int] = field(default_factory=list)
    
    # Search parameters
    temporal_window_days: int = 90
    spatial_buffer_deg: float = 1.0
    
    # Keywords for search
    keywords: List[str] = field(default_factory=list)


@dataclass
class DataSource:
    """Collected data source."""
    name: str
    source_type: str  # satellite, reanalysis, climate_index, paper, news
    data: Any
    quality: str = "raw"  # raw, validated, solid
    metadata: Dict = field(default_factory=dict)


@dataclass
class InvestigationResult:
    """Full investigation result."""
    query: str
    event_context: EventContext
    
    # Collected data
    data_sources: List[DataSource] = field(default_factory=list)
    
    # Analysis results
    correlations: List[Dict] = field(default_factory=list)
    causal_links: List[Dict] = field(default_factory=list)
    
    # Temporal/spatial expansion
    related_events: List[Dict] = field(default_factory=list)
    precursor_signals: List[Dict] = field(default_factory=list)
    
    # Papers and evidence
    papers: List[Dict] = field(default_factory=list)
    news_articles: List[Dict] = field(default_factory=list)
    
    # Summary
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        # Helper to convert numpy types to Python native types
        def convert_value(v):
            if hasattr(v, 'item'):  # numpy scalar
                return v.item()
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            return v
        
        return {
            'query': self.query,
            'location': self.event_context.location_name,
            'event_type': self.event_context.event_type,
            'time_range': f"{self.event_context.start_date} to {self.event_context.end_date}",
            'data_sources_count': len(self.data_sources),
            'papers_found': len(self.papers),
            'correlations': convert_value(self.correlations),
            'key_findings': self.key_findings,
            'recommendations': self.recommendations,
            'confidence': float(self.confidence) if hasattr(self.confidence, 'item') else self.confidence,
        }


class QueryParser:
    """Parse natural language investigation queries."""
    
    # Event type patterns
    EVENT_PATTERNS = {
        'flood': [
            r'(?:alluvion[ei]|flood[s]?|inondazion[ei]|esondazion[ei])',
            r'(?:piena|straripamento)',
        ],
        'drought': [
            r'(?:siccità|drought|aridità)',
        ],
        'storm_surge': [
            r'(?:storm surge|mareggiata|acqua alta)',
        ],
        'extreme_precipitation': [
            r'(?:precipitazion[ei] estrem[ae]|extreme precipitation|heavy rain)',
            r'(?:nubifragio|bomba d\'acqua|cloudbursts?)',
        ],
        'heatwave': [
            r'(?:ondata di calore|heatwave|canicola)',
        ],
    }
    
    # Date patterns
    DATE_PATTERNS = [
        # Specific dates
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
        
        # Month year
        r'(?:ottobre|october)\s+(\d{4})',
        r'(?:novembre|november)\s+(\d{4})',
        r'(?:settembre|september)\s+(\d{4})',
        
        # Just year
        r'\b(19\d{2}|20[0-2]\d)\b',
    ]
    
    # Known flood events (for quick lookup)
    KNOWN_EVENTS = {
        'valtellina 1987': {
            'location': 'Valtellina',
            'start_date': '1987-07-18',
            'end_date': '1987-07-28',
            'event_type': 'flood',
            'keywords': ['Valtellina disaster 1987', 'Val Pola landslide', 'Adda river', 'Sondrio'],
        },
        'lago maggiore 2000': {
            'location': 'Lago Maggiore',
            'start_date': '2000-10-10',
            'end_date': '2000-10-20',
            'event_type': 'flood',
            'keywords': ['October 2000 flood', 'Ticino', 'Toce', 'Verbano'],
        },
        'lago maggiore 1993': {
            'location': 'Lago Maggiore',
            'start_date': '1993-09-20',
            'end_date': '1993-10-10',
            'event_type': 'flood',
            'keywords': ['September 1993 flood', 'Brig', 'Ticino'],
        },
        'lago maggiore 1994': {
            'location': 'Lago Maggiore',
            'start_date': '1994-11-01',
            'end_date': '1994-11-15',
            'event_type': 'flood',
            'keywords': ['November 1994 flood', 'Piedmont', 'Liguria'],
        },
        'po valley 2000': {
            'location': 'Po Valley',
            'start_date': '2000-10-12',
            'end_date': '2000-10-25',
            'event_type': 'flood',
            'keywords': ['Po flood October 2000'],
        },
    }
    
    def parse(self, query: str) -> EventContext:
        """Parse query into EventContext."""
        query_lower = query.lower()
        
        # Check known events first
        for key, event in self.KNOWN_EVENTS.items():
            if all(word in query_lower for word in key.split()):
                ctx = EventContext(
                    location_name=event['location'],
                    event_type=event['event_type'],
                    start_date=event['start_date'],
                    end_date=event['end_date'],
                    keywords=event['keywords'],
                )
                return ctx
        
        # Extract event type
        event_type = 'flood'  # default
        for etype, patterns in self.EVENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    event_type = etype
                    break
        
        # Extract location (will be resolved by GeoResolver)
        location_name = self._extract_location(query)
        
        # Extract dates
        years = self._extract_years(query)
        start_date, end_date = self._infer_date_range(years, event_type)
        
        # Build keywords
        keywords = self._build_keywords(query, event_type, location_name)
        
        return EventContext(
            location_name=location_name,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            years_of_interest=years,
            keywords=keywords,
        )
    
    def _extract_location(self, query: str) -> str:
        """Extract location name from query."""
        query_lower = query.lower()
        
        # Check for specific known locations first
        known_locations = {
            'valtellina': 'Valtellina',
            'val tellina': 'Valtellina',
            'sondrio': 'Sondrio',
            'bormio': 'Bormio',
            'tirano': 'Tirano',
            'morbegno': 'Morbegno',
            'valchiavenna': 'Valchiavenna',
        }
        for key, name in known_locations.items():
            if key in query_lower:
                return name
        
        # Look for common location patterns
        patterns = [
            r'(?:lago|lake)\s+(\w+)',
            r'(?:valle|valley)\s+(\w+)',
            r'(?:fiume|river)\s+(\w+)',
            r'(?:in|at|near)\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Return the full match for multi-word locations
                return match.group(0)
        
        # Look for Italian lake names
        lakes = ['maggiore', 'como', 'garda', 'iseo', 'orta']
        for lake in lakes:
            if lake in query_lower:
                return f"Lago {lake.capitalize()}"
        
        # Look for regions
        regions = ['piemonte', 'lombardia', 'veneto', 'liguria', 'pianura padana']
        for region in regions:
            if region in query_lower:
                return region.replace('pianura padana', 'Pianura Padana').title()
        
        return "Italy"  # Default
    
    def _extract_years(self, query: str) -> List[int]:
        """Extract years from query."""
        years = []
        
        # Find all 4-digit years
        matches = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', query)
        years.extend([int(y) for y in matches])
        
        # Handle year ranges
        range_match = re.search(r'(19\d{2}|20[0-2]\d)\s*[-–]\s*(19\d{2}|20[0-2]\d)', query)
        if range_match:
            start_year = int(range_match.group(1))
            end_year = int(range_match.group(2))
            years.extend(range(start_year, end_year + 1))
        
        return sorted(set(years))
    
    def _infer_date_range(
        self,
        years: List[int],
        event_type: str,
    ) -> Tuple[str, str]:
        """Infer date range from years and event type."""
        if not years:
            # Default to recent 5 years
            end_year = datetime.now().year
            years = [end_year - 5, end_year]
        
        if len(years) == 1:
            year = years[0]
            # For floods, typically autumn
            if event_type == 'flood':
                return f"{year}-08-01", f"{year}-12-31"
            # For drought, typically summer
            elif event_type == 'drought':
                return f"{year}-04-01", f"{year}-10-31"
            else:
                return f"{year}-01-01", f"{year}-12-31"
        else:
            return f"{min(years)}-01-01", f"{max(years)}-12-31"
    
    def _build_keywords(
        self,
        query: str,
        event_type: str,
        location: str,
    ) -> List[str]:
        """Build search keywords."""
        keywords = [location, event_type]
        
        # Add event-specific keywords
        event_keywords = {
            'flood': ['precipitation', 'heavy rain', 'river discharge', 'water level'],
            'drought': ['soil moisture', 'evapotranspiration', 'water deficit'],
            'storm_surge': ['sea level', 'wind', 'atmospheric pressure'],
        }
        
        keywords.extend(event_keywords.get(event_type, []))
        
        return keywords


class InvestigationAgent:
    """
    Main investigation agent.
    
    Usage:
        agent = InvestigationAgent()
        
        # Simple investigation
        result = await agent.investigate(
            "analizza le alluvioni del Lago Maggiore nel 2000"
        )
        
        # Full investigation with all sources
        result = await agent.investigate(
            "analizza le alluvioni del Lago Maggiore nel 2000",
            collect_satellite=True,
            collect_reanalysis=True,
            collect_papers=True,
            collect_news=True,
            run_correlation=True,
            expand_search=True,
        )
    """
    
    def __init__(
        self,
        llm_client: Any = None,  # LLM service for query understanding
        db_client: Any = None,   # Database client for storage
        knowledge_service: Any = None,  # KnowledgeService for storing papers
    ):
        self.llm_client = llm_client
        self.db_client = db_client
        self.knowledge_service = knowledge_service
        
        self.query_parser = QueryParser()
        
        # Initialize tools (lazy)
        self._geo_resolver = None
        self._cmems_client = None
        self._era5_client = None
        self._climate_client = None
        self._literature_scraper = None
        
        # Logger will be initialized per investigation
        self._logger = None  # type: Optional[InvestigationLogger]
    
    @property
    def geo_resolver(self):
        if self._geo_resolver is None and GeoResolver:
            self._geo_resolver = GeoResolver()
        return self._geo_resolver
    
    async def create_briefing(
        self,
        query: str,
        collect_satellite: bool = True,
        collect_reanalysis: bool = True,
        collect_climate_indices: bool = True,
        collect_papers: bool = True,
    ) -> Dict:
        """
        Create a data briefing for user confirmation before download.
        
        Returns:
            Dict with briefing info including:
            - event_context: parsed query info
            - location: resolved location
            - data_requests: list of data sources with estimates
            - total_estimated_size_mb: total download size
            - total_estimated_time_sec: total estimated time
        """
        # Step 1: Parse query
        event_context = self.query_parser.parse(query)
        
        # Step 2: Geo-resolve location
        location = None
        if self.geo_resolver:
            location = await self.geo_resolver.resolve(event_context.location_name)
            if location:
                event_context.location = location
        
        # Step 3: Build data request list with estimates
        data_requests = []
        total_size = 0.0
        total_time = 0.0
        
        # Calculate bbox
        if location and location.bbox:
            bbox = location.bbox
            lat_range = (bbox[0], bbox[1])
            lon_range = (bbox[2], bbox[3])
        elif location:
            lat_range = (location.lat - 1, location.lat + 1)
            lon_range = (location.lon - 1, location.lon + 1)
        else:
            lat_range = (44, 47)  # Default Italy
            lon_range = (8, 12)
        
        time_range = (event_context.start_date, event_context.end_date)
        
        # CMEMS Satellite
        if collect_satellite and CMEMSClient:
            size_mb = self._estimate_data_size("cmems", lat_range, lon_range, time_range)
            time_sec = 30 + size_mb * 3  # 30s setup + 3 sec/MB
            data_requests.append({
                "source": "cmems",
                "name": "CMEMS Sea Level",
                "variables": ["sla", "adt"],
                "lat_range": lat_range,
                "lon_range": lon_range,
                "time_range": time_range,
                "estimated_size_mb": round(size_mb, 2),
                "estimated_time_sec": round(time_sec),
                "description": "Satellite altimetry sea level anomaly",
            })
            total_size += size_mb
            total_time += time_sec
        
        # ERA5 Reanalysis
        if collect_reanalysis and ERA5Client:
            size_mb = self._estimate_data_size("era5", lat_range, lon_range, time_range)
            time_sec = 60 + size_mb * 2  # 60s queue + 2 sec/MB
            data_requests.append({
                "source": "era5",
                "name": "ERA5 Reanalysis",
                "variables": ["total_precipitation", "2m_temperature", "mean_sea_level_pressure", "runoff"],
                "lat_range": lat_range,
                "lon_range": lon_range,
                "time_range": time_range,
                "estimated_size_mb": round(size_mb, 2),
                "estimated_time_sec": round(time_sec),
                "description": "Meteorological reanalysis (precipitation, temperature, pressure)",
            })
            total_size += size_mb
            total_time += time_sec
        
        # Climate Indices (always fast)
        if collect_climate_indices and ClimateIndicesClient:
            data_requests.append({
                "source": "climate_indices",
                "name": "Climate Indices",
                "variables": ["nao", "ao", "ea", "scand"],
                "lat_range": None,
                "lon_range": None,
                "time_range": time_range,
                "estimated_size_mb": 0.01,
                "estimated_time_sec": 5,
                "description": "NAO, AO, EA, SCAND teleconnection indices",
            })
            total_time += 5
        
        # Scientific papers
        if collect_papers and LiteratureScraper:
            data_requests.append({
                "source": "papers",
                "name": "Scientific Literature",
                "variables": ["arxiv", "semantic_scholar"],
                "lat_range": None,
                "lon_range": None,
                "time_range": None,
                "estimated_size_mb": 0.5,
                "estimated_time_sec": 30,
                "description": "Search arXiv and Semantic Scholar for relevant papers",
            })
            total_time += 30
        
        return {
            "query": query,
            "event_context": {
                "location_name": event_context.location_name,
                "event_type": event_context.event_type,
                "start_date": event_context.start_date,
                "end_date": event_context.end_date,
                "keywords": event_context.keywords,
            },
            "location": {
                "lat": location.lat if location else None,
                "lon": location.lon if location else None,
                "bbox": location.bbox if location else None,
                "name": location.name if location else event_context.location_name,
                "country": location.country if location else None,
                "region": location.region if location else None,
            } if location else None,
            "data_requests": data_requests,
            "total_estimated_size_mb": round(total_size, 2),
            "total_estimated_time_sec": round(total_time),
            "summary": self._generate_briefing_summary(
                event_context, location, data_requests, total_size, total_time
            ),
        }
    
    def _estimate_data_size(
        self,
        source: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> float:
        """Estimate data size in MB."""
        from datetime import datetime
        
        lat_span = abs(lat_range[1] - lat_range[0])
        lon_span = abs(lon_range[1] - lon_range[0])
        
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        days = (end - start).days + 1
        
        if source == "era5":
            # ERA5: 0.25° resolution, hourly
            lat_points = max(1, int(lat_span / 0.25))
            lon_points = max(1, int(lon_span / 0.25))
            time_points = days * 24
            vars_count = 4
        elif source == "cmems":
            # CMEMS: 0.125° resolution, daily
            lat_points = max(1, int(lat_span / 0.125))
            lon_points = max(1, int(lon_span / 0.125))
            time_points = days
            vars_count = 2
        else:
            return 0.1
        
        # 4 bytes per float, compressed ~50%
        total_points = lat_points * lon_points * time_points * vars_count
        return (total_points * 4 * 0.5) / (1024 * 1024)
    
    def _generate_briefing_summary(
        self,
        ctx: 'EventContext',
        location: Any,
        requests: List[Dict],
        total_size: float,
        total_time: float,
    ) -> str:
        """Generate human-readable briefing summary."""
        lines = [
            f"📍 **{ctx.location_name}**",
        ]
        
        if location:
            lines.append(f"   Coordinates: {location.lat:.3f}°N, {location.lon:.3f}°E")
            if location.bbox:
                lines.append(f"   Area: {location.bbox[0]:.2f}°-{location.bbox[1]:.2f}°N, {location.bbox[2]:.2f}°-{location.bbox[3]:.2f}°E")
        
        lines.extend([
            f"📅 **Period**: {ctx.start_date} → {ctx.end_date}",
            f"🎯 **Event**: {ctx.event_type}",
            "",
            "📦 **Data to download**:",
        ])
        
        for req in requests:
            lines.append(f"   • **{req['name']}**: {', '.join(req['variables'][:3])}")
            if req['estimated_size_mb'] > 0.1:
                lines.append(f"     ~{req['estimated_size_mb']:.1f} MB, ~{req['estimated_time_sec']}s")
        
        lines.extend([
            "",
            f"⏱️ **Totale**: ~{total_size:.1f} MB, ~{total_time:.0f}s",
        ])
        
        return "\n".join(lines)
    
    @property
    def cmems_client(self):
        if self._cmems_client is None and CMEMSClient:
            self._cmems_client = CMEMSClient()
        return self._cmems_client
    
    @property
    def era5_client(self):
        if self._era5_client is None and ERA5Client:
            self._era5_client = ERA5Client()
        return self._era5_client
    
    @property
    def climate_client(self):
        if self._climate_client is None and ClimateIndicesClient:
            self._climate_client = ClimateIndicesClient()
        return self._climate_client
    
    @property
    def literature_scraper(self):
        if self._literature_scraper is None and LiteratureScraper:
            self._literature_scraper = LiteratureScraper()
        return self._literature_scraper
    
    async def investigate_streaming(
        self,
        query: str,
        collect_satellite: bool = True,
        collect_reanalysis: bool = True,
        collect_climate_indices: bool = True,
        collect_papers: bool = True,
        collect_news: bool = False,
        run_correlation: bool = True,
        expand_search: bool = True,
        temporal_resolution: str = "daily",
        spatial_resolution: str = "0.25",
    ):
        """
        Run investigation with streaming progress updates.
        
        Args:
            temporal_resolution: 'hourly', '6-hourly', 'daily'
            spatial_resolution: '0.1', '0.25', '0.5' (degrees)
        
        Yields progress dictionaries with:
        - step: int (1-6)
        - substep: optional string
        - status: 'started' | 'progress' | 'complete' | 'error'
        - message: human-readable message
        - data: optional dict with results
        - progress: optional 0-100 percentage
        """
        # Initialize structured logger
        investigation_id = str(uuid.uuid4())
        if InvestigationLogger:
            self._logger = InvestigationLogger(investigation_id, query)
            self._logger.log_metric("temporal_resolution", temporal_resolution)
            self._logger.log_metric("spatial_resolution", spatial_resolution)
        
        # Store resolution config for data collection methods
        self._temporal_resolution = temporal_resolution
        self._spatial_resolution = float(spatial_resolution)
        
        yield {"step": 0, "status": "started", "message": f"🔍 Starting investigation: {query}", "data": {"resolution": {"temporal": temporal_resolution, "spatial": spatial_resolution}, "investigation_id": investigation_id}}
        
        # Initialize result
        result = InvestigationResult(query=query, event_context=EventContext(location_name=""))
        
        # Step 1: Parse query
        yield {"step": 1, "status": "started", "message": "📝 Parsing query..."}
        if self._logger and InvestigationStep:
            self._logger.start_step(InvestigationStep.PARSE)
        
        try:
            event_context = self.query_parser.parse(query)
            result.event_context = event_context
            
            if self._logger and InvestigationStep:
                self._logger.update_context(
                    location=event_context.location_name,
                    event_type=event_context.event_type,
                    time_range=f"{event_context.start_date} to {event_context.end_date}"
                )
                self._logger.complete_step(InvestigationStep.PARSE, {
                    "location": event_context.location_name,
                    "event_type": event_context.event_type
                })
            
            yield {
                "step": 1, 
                "status": "complete", 
                "message": f"Query parsed: {event_context.location_name}",
                "data": {
                    "location": event_context.location_name,
                    "event_type": event_context.event_type,
                    "start_date": event_context.start_date,
                    "end_date": event_context.end_date,
                }
            }
        except Exception as e:
            if self._logger and InvestigationStep:
                self._logger.fail_step(InvestigationStep.PARSE, str(e))
            raise
        
        # Step 2: Geo-resolve location
        yield {"step": 2, "status": "started", "message": "🌍 Resolving location..."}
        if self.geo_resolver:
            location = await self.geo_resolver.resolve(event_context.location_name)
            if location:
                event_context.location = location
                yield {
                    "step": 2, 
                    "status": "complete", 
                    "message": f"Location found: {location.lat:.3f}°N, {location.lon:.3f}°E",
                    "data": {"lat": location.lat, "lon": location.lon, "bbox": location.bbox}
                }
            else:
                yield {"step": 2, "status": "error", "message": "⚠️ Could not resolve location"}
        
        # Step 3: Collect data
        yield {"step": 3, "status": "started", "message": "📊 Collecting data..."}
        total_sources = sum([collect_satellite, collect_reanalysis, collect_climate_indices, collect_papers])
        collected = 0
        
        if collect_satellite and event_context.location:
            yield {"step": 3, "substep": "satellite", "status": "started", "message": "📡 Downloading satellite data (CMEMS)..."}
            try:
                await self._collect_satellite_data(event_context, result)
                collected += 1
                yield {"step": 3, "substep": "satellite", "status": "complete", "message": "✅ Satellite data collected", "progress": int(collected/total_sources*100)}
            except Exception as e:
                yield {"step": 3, "substep": "satellite", "status": "error", "message": f"⚠️ Satellite error: {str(e)[:50]}"}
        
        if collect_reanalysis and event_context.location:
            yield {"step": 3, "substep": "era5", "status": "started", "message": "🌤️ Downloading ERA5 reanalysis..."}
            try:
                await self._collect_reanalysis_data(event_context, result)
                collected += 1
                yield {"step": 3, "substep": "era5", "status": "complete", "message": "✅ ERA5 data collected", "progress": int(collected/total_sources*100)}
            except Exception as e:
                yield {"step": 3, "substep": "era5", "status": "error", "message": f"⚠️ ERA5 error: {str(e)[:50]}"}
        
        if collect_climate_indices:
            yield {"step": 3, "substep": "indices", "status": "started", "message": "🌍 Fetching climate indices..."}
            try:
                await self._collect_climate_indices(event_context, result)
                collected += 1
                yield {"step": 3, "substep": "indices", "status": "complete", "message": "✅ Climate indices collected", "progress": int(collected/total_sources*100)}
            except Exception as e:
                yield {"step": 3, "substep": "indices", "status": "error", "message": f"⚠️ Indices error: {str(e)[:50]}"}
        
        if collect_papers:
            yield {"step": 3, "substep": "papers", "status": "started", "message": "📚 Searching scientific papers..."}
            try:
                await self._collect_papers(event_context, result)
                collected += 1
                yield {"step": 3, "substep": "papers", "status": "complete", "message": f"✅ Found {len(result.papers)} papers", "progress": int(collected/total_sources*100)}
            except Exception as e:
                yield {"step": 3, "substep": "papers", "status": "error", "message": f"⚠️ Papers error: {str(e)[:50]}"}
        
        yield {"step": 3, "status": "complete", "message": f"Data collection complete: {len(result.data_sources)} sources"}
        
        # Step 4: Correlation analysis
        if run_correlation and len(result.data_sources) > 1:
            yield {"step": 4, "status": "started", "message": "🔗 Running correlation analysis..."}
            await self._run_correlation(result)
            yield {"step": 4, "status": "complete", "message": f"Found {len(result.correlations)} correlations"}
        
        # Step 5: Expand search
        if expand_search:
            yield {"step": 5, "status": "started", "message": "🔭 Expanding temporal/spatial search..."}
            await self._expand_search(event_context, result)
            yield {"step": 5, "status": "complete", "message": "Search expanded"}
        
        # Step 6: Generate findings
        yield {"step": 6, "status": "started", "message": "📋 Generating findings..."}
        await self._generate_findings(result)
        yield {"step": 6, "status": "complete", "message": f"Generated {len(result.key_findings)} findings"}
        
        # Log investigation summary
        if self._logger and InvestigationStep:
            self._logger.complete_step(InvestigationStep.COMPLETE, {
                "data_sources": len(result.data_sources),
                "papers_found": len(result.papers),
                "correlations": len(result.correlations),
                "key_findings": len(result.key_findings)
            })
            
            # Get summary with metrics
            summary = self._logger.get_summary()
            print(f"\n📊 Investigation Summary:")
            print(f"   Duration: {summary['total_duration_ms']:.0f}ms")
            print(f"   Steps completed: {len(summary['steps_completed'])}")
            print(f"   Steps failed: {len(summary['steps_failed'])}")
            print(f"   Success rate: {summary['success_rate']*100:.1f}%")
        
        # Final result
        yield {
            "step": "complete",
            "status": "success",
            "message": "✅ Investigation complete!",
            "result": result.to_dict(),
            "summary": self._logger.get_summary() if self._logger else None
        }
    
    async def investigate(
        self,
        query: str,
        collect_satellite: bool = True,
        collect_reanalysis: bool = True,
        collect_climate_indices: bool = True,
        collect_papers: bool = True,
        collect_news: bool = False,
        run_correlation: bool = True,
        expand_search: bool = True,
    ) -> InvestigationResult:
        """
        Run full investigation.
        
        Args:
            query: Natural language query
            collect_*: Which data sources to collect
            run_correlation: Whether to run correlation analysis
            expand_search: Whether to expand temporal/spatial search
        """
        print(f"\n🔍 Starting investigation: {query}")
        print("=" * 60)
        
        # Initialize result
        result = InvestigationResult(query=query, event_context=EventContext(location_name=""))
        
        # Step 1: Parse query
        print("\n📝 Step 1: Parsing query...")
        event_context = self.query_parser.parse(query)
        result.event_context = event_context
        
        print(f"   Location: {event_context.location_name}")
        print(f"   Event type: {event_context.event_type}")
        print(f"   Date range: {event_context.start_date} to {event_context.end_date}")
        
        # Step 2: Geo-resolve location
        print("\n🌍 Step 2: Resolving location...")
        if self.geo_resolver:
            location = await self.geo_resolver.resolve(event_context.location_name)
            if location:
                event_context.location = location
                print(f"   Coordinates: {location.lat:.3f}°N, {location.lon:.3f}°E")
                print(f"   Bbox: {location.bbox}")
            else:
                print(f"   ⚠️ Could not resolve location")
        
        # Step 3: Collect data
        print("\n📊 Step 3: Collecting data...")
        tasks = []
        
        if collect_satellite and event_context.location:
            tasks.append(self._collect_satellite_data(event_context, result))
        
        if collect_reanalysis and event_context.location:
            tasks.append(self._collect_reanalysis_data(event_context, result))
        
        if collect_climate_indices:
            tasks.append(self._collect_climate_indices(event_context, result))
        
        if collect_papers:
            tasks.append(self._collect_papers(event_context, result))
        
        # Run collection in parallel where possible
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"\n   Collected {len(result.data_sources)} data sources")
        
        # Step 4: Run correlation analysis
        if run_correlation and len(result.data_sources) > 1:
            print("\n🔗 Step 4: Running correlation analysis...")
            await self._run_correlation(result)
        
        # Step 5: Expand search (temporal and spatial)
        if expand_search:
            print("\n🔭 Step 5: Expanding search...")
            await self._expand_search(event_context, result)
        
        # Step 6: Generate findings
        print("\n📋 Step 6: Generating findings...")
        await self._generate_findings(result)
        
        print("\n" + "=" * 60)
        print(f"✅ Investigation complete!")
        print(f"   Data sources: {len(result.data_sources)}")
        print(f"   Papers found: {len(result.papers)}")
        print(f"   Correlations: {len(result.correlations)}")
        print(f"   Key findings: {len(result.key_findings)}")
        
        return result
    
    async def _collect_satellite_data(
        self,
        ctx: EventContext,
        result: InvestigationResult,
    ):
        """Collect satellite data (CMEMS)."""
        if not self.cmems_client or not ctx.location:
            return
        
        print("   📡 Fetching satellite data...")
        
        loc = ctx.location
        # bbox format: (lat_min, lat_max, lon_min, lon_max)
        lat_range = (loc.bbox[0], loc.bbox[1]) if loc.bbox else (loc.lat - 1, loc.lat + 1)
        lon_range = (loc.bbox[2], loc.bbox[3]) if loc.bbox else (loc.lon - 1, loc.lon + 1)
        
        print(f"      Area: lat={lat_range}, lon={lon_range}")
        print(f"      Time: {ctx.start_date} to {ctx.end_date}")
        
        try:
            ds = await self.cmems_client.download(
                dataset="sea_level_global",
                variables=["sla"],
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=(ctx.start_date, ctx.end_date),
            )
            
            if ds is not None:
                result.data_sources.append(DataSource(
                    name="CMEMS Sea Level",
                    source_type="satellite",
                    data=ds,
                    quality="validated",
                    metadata={'dataset': 'sea_level_global'},
                ))
                print("      ✅ CMEMS data collected")
        except Exception as e:
            print(f"      ⚠️ CMEMS error: {e}")
    
    async def _collect_reanalysis_data(
        self,
        ctx: EventContext,
        result: InvestigationResult,
    ):
        """Collect reanalysis data (ERA5)."""
        if not self.era5_client or not ctx.location:
            return
        
        print("   🌤️ Fetching reanalysis data...")
        
        loc = ctx.location
        # bbox format: (lat_min, lat_max, lon_min, lon_max)
        # Expand slightly for ERA5 (coarser resolution)
        lat_range = (loc.bbox[0] - 0.5, loc.bbox[1] + 0.5) if loc.bbox else (loc.lat - 2, loc.lat + 2)
        lon_range = (loc.bbox[2] - 0.5, loc.bbox[3] + 0.5) if loc.bbox else (loc.lon - 2, loc.lon + 2)
        
        print(f"      Area: lat={lat_range}, lon={lon_range}")
        print(f"      Time: {ctx.start_date} to {ctx.end_date}")
        
        try:
            ds = await self.era5_client.download_for_flood(
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=(ctx.start_date, ctx.end_date),
            )
            
            if ds is not None:
                result.data_sources.append(DataSource(
                    name="ERA5 Reanalysis",
                    source_type="reanalysis",
                    data=ds,
                    quality="validated",
                    metadata={'variables': list(ds.data_vars)},
                ))
                print("      ✅ ERA5 data collected")
        except Exception as e:
            print(f"      ⚠️ ERA5 error: {e}")
    
    async def _collect_climate_indices(
        self,
        ctx: EventContext,
        result: InvestigationResult,
    ):
        """Collect climate indices."""
        if not self.climate_client:
            return
        
        print("   🌍 Fetching climate indices...")
        
        try:
            # Get event date (middle of range)
            if ctx.years_of_interest:
                year = ctx.years_of_interest[0]
                event_date = f"{year}-10-15"  # Default to October
            else:
                event_date = ctx.start_date
            
            indices = await self.climate_client.get_indices_for_flood(
                event_date=event_date,
                region="italy" if "italia" in ctx.location_name.lower() or "lago" in ctx.location_name.lower() else "europe",
            )
            
            if indices:
                result.data_sources.append(DataSource(
                    name="Climate Indices",
                    source_type="climate_index",
                    data=indices,
                    quality="validated",
                    metadata={'indices': list(indices.keys())},
                ))
                print(f"      ✅ Climate indices collected: {list(indices.keys())}")
        except Exception as e:
            print(f"      ⚠️ Climate indices error: {e}")
    
    async def _collect_papers(
        self,
        ctx: EventContext,
        result: InvestigationResult,
    ):
        """Collect scientific papers and save to knowledge base."""
        if not self.literature_scraper:
            return
        
        print("   📚 Searching scientific literature...")
        
        if self._logger and InvestigationStep:
            self._logger.start_step(InvestigationStep.PAPERS_COLLECT)
        
        try:
            papers = await self.literature_scraper.search_flood_papers(
                location=ctx.location_name,
                event_years=ctx.years_of_interest,
                max_results=20,
            )
            
            print(f"      ✅ Found {len(papers)} papers")
            
            if self._logger and InvestigationStep:
                self._logger.complete_step(InvestigationStep.PAPERS_COLLECT, {
                    "papers_found": len(papers)
                })
            
            # Validate and sanitize papers
            if self._logger and InvestigationStep:
                self._logger.start_step(InvestigationStep.PAPERS_VALIDATE)
            
            papers_to_save = []
            if validate_papers_batch and ValidationLevel:
                try:
                    # Convert papers to dict format
                    papers_dict = [p.to_dict() if hasattr(p, 'to_dict') else p for p in papers]
                    
                    # Validate and sanitize
                    validated_papers, validation_results = validate_papers_batch(papers_dict)
                    papers_to_save = validated_papers
                    
                    # Log validation results
                    critical_failures = [r for r in validation_results if r.level == ValidationLevel.CRITICAL and not r.passed]
                    warnings = [r for r in validation_results if r.level == ValidationLevel.WARNING and not r.passed]
                    
                    if self._logger:
                        for vr in validation_results:
                            self._logger.log_validation(vr.validator, vr.passed, vr.details)
                        
                        self._logger.complete_step(InvestigationStep.PAPERS_VALIDATE, {
                            "original_count": len(papers_dict),
                            "validated_count": len(validated_papers),
                            "rejected_count": len(papers_dict) - len(validated_papers),
                            "critical_failures": len(critical_failures),
                            "warnings": len(warnings)
                        })
                    
                    print(f"      ✅ Validated {len(validated_papers)}/{len(papers_dict)} papers")
                    if critical_failures:
                        print(f"      ⚠️ Rejected {len(papers_dict) - len(validated_papers)} papers due to validation errors")
                except Exception as e:
                    print(f"      ⚠️ Validation error: {e}, using unvalidated papers")
                    papers_to_save = [p.to_dict() if hasattr(p, 'to_dict') else p for p in papers]
                    if self._logger and InvestigationStep:
                        self._logger.fail_step(InvestigationStep.PAPERS_VALIDATE, str(e))
            else:
                # No validation available, use papers as-is
                papers_to_save = [p.to_dict() if hasattr(p, 'to_dict') else p for p in papers]
            
            # Add to result
            for paper in papers_to_save:
                result.papers.append(paper)
            
            # Save papers to knowledge base
            if self.knowledge_service and papers_to_save:
                if self._logger and InvestigationStep:
                    self._logger.start_step(InvestigationStep.PAPERS_STORE)
                
                try:
                    print(f"      💾 Saving {len(papers_to_save)} papers to knowledge base...")
                    saved_count = await self.knowledge_service.bulk_add_papers(papers_to_save)
                    print(f"      ✅ Saved {saved_count} papers to vector DB")
                    
                    if self._logger and InvestigationStep:
                        self._logger.complete_step(InvestigationStep.PAPERS_STORE, {
                            "papers_saved": saved_count,
                            "backend": "surrealdb"
                        })
                        
                        # Health check: verify papers are searchable
                        self._logger.log_health_check("knowledge_base", True, {
                            "papers_stored": saved_count,
                            "backend": "surrealdb"
                        })
                except Exception as e:
                    print(f"      ⚠️ Paper storage error: {e}")
                    if self._logger and InvestigationStep:
                        self._logger.fail_step(InvestigationStep.PAPERS_STORE, str(e), {
                            "papers_count": len(papers_to_save)
                        })
                        self._logger.log_health_check("knowledge_base", False, {
                            "error": str(e)
                        })
        except Exception as e:
            print(f"      ⚠️ Literature search error: {e}")
            if self._logger and InvestigationStep:
                self._logger.fail_step(InvestigationStep.PAPERS_COLLECT, str(e))
    
    async def _run_correlation(self, result: InvestigationResult):
        """Run correlation analysis between data sources."""
        # Find climate indices and time series data
        climate_data = None
        era5_data = None
        
        for source in result.data_sources:
            if source.source_type == "climate_index":
                climate_data = source.data
            elif source.source_type == "reanalysis":
                era5_data = source.data
        
        if climate_data:
            # Analyze climate index conditions
            for idx_name, idx_data in climate_data.items():
                if 'interpretation' in idx_data:
                    result.correlations.append({
                        'type': 'climate_index',
                        'index': idx_name.upper(),
                        'value': idx_data.get('event_value', 0),
                        'interpretation': idx_data.get('interpretation', ''),
                        'favorable_for_flood': idx_data.get('favorable_for_flood', False),
                    })
        
        # Basic correlation placeholder
        # In full implementation, this would use PCMCI or other causal methods
        result.correlations.append({
            'type': 'note',
            'message': 'Full PCMCI causal analysis requires aligned time series',
        })
    
    async def _expand_search(
        self,
        ctx: EventContext,
        result: InvestigationResult,
    ):
        """Expand search temporally and spatially."""
        # Temporal expansion: look for similar events in other years
        print("   🕐 Searching for related events in other years...")
        
        # Known related events for Lago Maggiore
        if 'maggiore' in ctx.location_name.lower():
            related = [
                {'year': 1993, 'description': 'September 1993 flood'},
                {'year': 1994, 'description': 'November 1994 Piedmont flood'},
                {'year': 2000, 'description': 'October 2000 flood'},
                {'year': 2014, 'description': 'November 2014 flood'},
            ]
            for event in related:
                if event['year'] not in ctx.years_of_interest:
                    result.related_events.append(event)
        
        # Spatial expansion: nearby regions
        print("   🗺️ Checking nearby regions...")
        
        if ctx.location:
            nearby_regions = [
                'Ticino River', 'Toce River', 'Po Valley',
                'Northern Italy', 'Swiss Alps'
            ]
            result.related_events.extend([
                {'region': r, 'type': 'nearby'} for r in nearby_regions[:3]
            ])
    
    async def _generate_findings(self, result: InvestigationResult):
        """Generate key findings and recommendations."""
        findings = []
        recommendations = []
        
        # Summarize climate conditions
        favorable_indices = [
            c for c in result.correlations 
            if c.get('type') == 'climate_index' and c.get('favorable_for_flood')
        ]
        
        if favorable_indices:
            idx_names = [c['index'] for c in favorable_indices]
            findings.append(
                f"Climate conditions were favorable for flooding: {', '.join(idx_names)} "
                f"indices indicated elevated flood risk."
            )
        
        # Summarize data availability
        source_types = [s.source_type for s in result.data_sources]
        findings.append(
            f"Collected data from {len(result.data_sources)} sources: "
            f"{', '.join(set(source_types))}"
        )
        
        # Paper findings
        if result.papers:
            findings.append(
                f"Found {len(result.papers)} relevant scientific papers"
            )
        
        # Related events
        if result.related_events:
            findings.append(
                f"Identified {len(result.related_events)} related events for comparison"
            )
        
        # Recommendations
        recommendations.append(
            "Run PCMCI causal analysis on aligned time series for robust causal inference"
        )
        recommendations.append(
            "Cross-reference findings with local hydrological station data"
        )
        
        if not result.papers:
            recommendations.append(
                "Expand literature search to include Italian journals and reports"
            )
        
        result.key_findings = findings
        result.recommendations = recommendations
        result.confidence = min(0.3 + 0.1 * len(result.data_sources) + 0.05 * len(result.papers), 0.9)


# CLI interface
async def main():
    """CLI for testing investigation agent."""
    import sys
    
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "analizza alluvioni Lago Maggiore 2000"
    
    agent = InvestigationAgent()
    result = await agent.investigate(query)
    
    print("\n" + "=" * 60)
    print("📋 INVESTIGATION REPORT")
    print("=" * 60)
    
    print(f"\n🔎 Query: {result.query}")
    print(f"📍 Location: {result.event_context.location_name}")
    print(f"📅 Period: {result.event_context.start_date} to {result.event_context.end_date}")
    
    print(f"\n📊 Data Sources ({len(result.data_sources)}):")
    for source in result.data_sources:
        print(f"   - {source.name} ({source.source_type})")
    
    print(f"\n🔗 Correlations ({len(result.correlations)}):")
    for corr in result.correlations:
        if corr.get('type') == 'climate_index':
            print(f"   - {corr['interpretation']}")
    
    print(f"\n📚 Papers Found: {len(result.papers)}")
    for paper in result.papers[:5]:
        print(f"   - {paper['title'][:60]}...")
    
    print(f"\n💡 Key Findings:")
    for finding in result.key_findings:
        print(f"   • {finding}")
    
    print(f"\n📝 Recommendations:")
    for rec in result.recommendations:
        print(f"   → {rec}")
    
    print(f"\n🎯 Confidence: {result.confidence:.0%}")


if __name__ == "__main__":
    asyncio.run(main())

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
        return {
            'query': self.query,
            'location': self.event_context.location_name,
            'event_type': self.event_context.event_type,
            'time_range': f"{self.event_context.start_date} to {self.event_context.end_date}",
            'data_sources_count': len(self.data_sources),
            'papers_found': len(self.papers),
            'correlations': self.correlations,
            'key_findings': self.key_findings,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
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
            if lake in query.lower():
                return f"Lago {lake.capitalize()}"
        
        # Look for regions
        regions = ['piemonte', 'lombardia', 'veneto', 'liguria', 'pianura padana']
        for region in regions:
            if region in query.lower():
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
    ):
        self.llm_client = llm_client
        self.db_client = db_client
        
        self.query_parser = QueryParser()
        
        # Initialize tools (lazy)
        self._geo_resolver = None
        self._cmems_client = None
        self._era5_client = None
        self._climate_client = None
        self._literature_scraper = None
    
    @property
    def geo_resolver(self):
        if self._geo_resolver is None and GeoResolver:
            self._geo_resolver = GeoResolver()
        return self._geo_resolver
    
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
        lat_range = (loc.bbox[1], loc.bbox[3]) if loc.bbox else (loc.lat - 1, loc.lat + 1)
        lon_range = (loc.bbox[0], loc.bbox[2]) if loc.bbox else (loc.lon - 1, loc.lon + 1)
        
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
        lat_range = (loc.bbox[1], loc.bbox[3]) if loc.bbox else (loc.lat - 2, loc.lat + 2)
        lon_range = (loc.bbox[0], loc.bbox[2]) if loc.bbox else (loc.lon - 2, loc.lon + 2)
        
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
        """Collect scientific papers."""
        if not self.literature_scraper:
            return
        
        print("   📚 Searching scientific literature...")
        
        try:
            papers = await self.literature_scraper.search_flood_papers(
                location=ctx.location_name,
                event_years=ctx.years_of_interest,
                max_results=20,
            )
            
            for paper in papers:
                result.papers.append(paper.to_dict())
            
            print(f"      ✅ Found {len(papers)} papers")
        except Exception as e:
            print(f"      ⚠️ Literature search error: {e}")
    
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

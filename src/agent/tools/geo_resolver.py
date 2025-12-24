"""
🌍 Geographic Entity Resolver
==============================

Convert natural language locations to coordinates and geographic metadata.

Uses:
- Nominatim (OpenStreetMap) - Free, no API key
- GeoNames fallback
- Local cache for common locations

Examples:
- "Lago Maggiore" → 45.95°N, 8.65°E, bbox, region info
- "Fram Strait" → 78.5°N, 0°E
- "alluvioni nord Italia 1994" → extracts "nord Italia" → bbox
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import aiohttp
import ssl
import certifi
import time

# Create SSL context with certifi certificates
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@dataclass
class GeoLocation:
    """Resolved geographic location."""
    name: str
    lat: float
    lon: float
    
    # Bounding box [south, north, west, east]
    bbox: Tuple[float, float, float, float] = None
    
    # Metadata
    country: str = ""
    region: str = ""
    place_type: str = ""  # lake, strait, river, city, etc.
    osm_id: Optional[int] = None
    
    # For water bodies
    area_km2: Optional[float] = None
    elevation_m: Optional[float] = None
    
    # Search context
    original_query: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def lat_range(self) -> Tuple[float, float]:
        """Get latitude range from bbox or point ± 0.5°."""
        if self.bbox:
            return (self.bbox[0], self.bbox[1])
        return (self.lat - 0.5, self.lat + 0.5)
    
    @property
    def lon_range(self) -> Tuple[float, float]:
        """Get longitude range from bbox or point ± 0.5°."""
        if self.bbox:
            return (self.bbox[2], self.bbox[3])
        return (self.lon - 0.5, self.lon + 0.5)


# Pre-defined locations for common oceanographic/hydrological features
KNOWN_LOCATIONS = {
    # Italian Lakes
    "lago maggiore": GeoLocation(
        name="Lago Maggiore",
        lat=45.95, lon=8.65,
        bbox=(45.72, 46.18, 8.38, 8.88),
        country="Italy/Switzerland",
        region="Lombardia/Piemonte/Ticino",
        place_type="lake",
        area_km2=212.5,
        elevation_m=193,
    ),
    "lago di como": GeoLocation(
        name="Lago di Como",
        lat=46.0, lon=9.27,
        bbox=(45.77, 46.17, 9.05, 9.45),
        country="Italy",
        region="Lombardia",
        place_type="lake",
        area_km2=146,
        elevation_m=198,
    ),
    "lago di garda": GeoLocation(
        name="Lago di Garda",
        lat=45.65, lon=10.65,
        bbox=(45.42, 45.88, 10.52, 10.95),
        country="Italy",
        region="Lombardia/Veneto/Trentino",
        place_type="lake",
        area_km2=370,
        elevation_m=65,
    ),
    
    # Arctic/Ocean straits
    "fram strait": GeoLocation(
        name="Fram Strait",
        lat=78.5, lon=0.0,
        bbox=(76.0, 81.0, -10.0, 10.0),
        country="International",
        region="Arctic Ocean",
        place_type="strait",
    ),
    "bering strait": GeoLocation(
        name="Bering Strait",
        lat=65.75, lon=-169.0,
        bbox=(65.0, 66.5, -171.0, -167.0),
        country="USA/Russia",
        region="Arctic Ocean/Pacific",
        place_type="strait",
    ),
    "davis strait": GeoLocation(
        name="Davis Strait",
        lat=66.0, lon=-57.0,
        bbox=(63.0, 70.0, -65.0, -50.0),
        country="Canada/Greenland",
        region="North Atlantic",
        place_type="strait",
    ),
    
    # Rivers
    "fiume ticino": GeoLocation(
        name="Fiume Ticino",
        lat=45.5, lon=8.85,
        bbox=(45.0, 46.2, 8.4, 9.2),
        country="Italy/Switzerland",
        region="Lombardia/Piemonte/Ticino",
        place_type="river",
    ),
    "fiume po": GeoLocation(
        name="Fiume Po",
        lat=45.0, lon=10.5,
        bbox=(44.7, 45.5, 7.0, 12.5),
        country="Italy",
        region="Pianura Padana",
        place_type="river",
    ),
    
    # Regions
    "nord italia": GeoLocation(
        name="Nord Italia",
        lat=45.5, lon=10.0,
        bbox=(44.5, 47.0, 6.5, 14.0),
        country="Italy",
        region="Northern Italy",
        place_type="region",
    ),
    "pianura padana": GeoLocation(
        name="Pianura Padana",
        lat=45.0, lon=10.5,
        bbox=(44.5, 46.0, 7.5, 12.5),
        country="Italy",
        region="Po Valley",
        place_type="region",
    ),
}


class GeoResolver:
    """
    Resolve geographic locations from natural language.
    
    Usage:
        resolver = GeoResolver()
        
        # Single location
        loc = await resolver.resolve("Lago Maggiore")
        print(f"{loc.name}: {loc.lat}, {loc.lon}")
        
        # From event description
        loc = await resolver.resolve_from_text(
            "Le alluvioni del Lago Maggiore nel 1994 hanno causato..."
        )
        
        # Expand area for investigation
        expanded = resolver.expand_bbox(loc, factor=2.0)
    """
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    USER_AGENT = "SurgeShazam/1.0 (research project)"
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "geo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0
        self._rate_limit_seconds = 1.0  # Nominatim requires 1 req/sec
    
    async def resolve(self, query: str) -> Optional[GeoLocation]:
        """
        Resolve a location query to coordinates.
        
        Checks:
        1. Known locations cache
        2. Local file cache
        3. Nominatim API
        """
        # Normalize query
        query_lower = query.lower().strip()
        
        # Check known locations
        if query_lower in KNOWN_LOCATIONS:
            loc = KNOWN_LOCATIONS[query_lower]
            loc.original_query = query
            return loc
        
        # Check partial matches in known locations
        for key, loc in KNOWN_LOCATIONS.items():
            if key in query_lower or query_lower in key:
                loc.original_query = query
                return loc
        
        # Check file cache
        cached = self._load_from_cache(query_lower)
        if cached:
            return cached
        
        # Query Nominatim
        result = await self._query_nominatim(query)
        if result:
            self._save_to_cache(query_lower, result)
            return result
        
        return None
    
    async def resolve_from_text(self, text: str) -> List[GeoLocation]:
        """
        Extract and resolve all geographic entities from text.
        
        Uses regex patterns and known location matching.
        """
        locations = []
        text_lower = text.lower()
        
        # Check all known locations
        for key, loc in KNOWN_LOCATIONS.items():
            if key in text_lower:
                loc_copy = GeoLocation(**asdict(loc))
                loc_copy.original_query = key
                locations.append(loc_copy)
        
        # Extract potential location patterns
        patterns = [
            r'\b[Ll]ago\s+(?:di\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # Lago di Como
            r'\b[Ff]iume\s+([A-Z][a-z]+)',  # Fiume Ticino
            r'\b([A-Z][a-z]+)\s+[Ss]trait',  # Fram Strait
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*(?:Italy|Italia|Switzerland)',  # City, Country
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                # Try to resolve
                loc = await self.resolve(match)
                if loc and loc not in locations:
                    locations.append(loc)
        
        return locations
    
    async def _query_nominatim(self, query: str) -> Optional[GeoLocation]:
        """Query Nominatim API with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_seconds:
            await asyncio.sleep(self._rate_limit_seconds - elapsed)
        
        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "extratags": 1,
        }
        
        headers = {"User-Agent": self.USER_AGENT}
        
        try:
            connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    self.NOMINATIM_URL,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    self._last_request_time = time.time()
                    
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data:
                        return None
                    
                    result = data[0]
                    
                    # Parse bounding box
                    bbox = None
                    if "boundingbox" in result:
                        bb = result["boundingbox"]
                        bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                    
                    # Extract address details
                    address = result.get("address", {})
                    
                    return GeoLocation(
                        name=result.get("display_name", query).split(",")[0],
                        lat=float(result["lat"]),
                        lon=float(result["lon"]),
                        bbox=bbox,
                        country=address.get("country", ""),
                        region=address.get("state", address.get("region", "")),
                        place_type=result.get("type", result.get("class", "")),
                        osm_id=int(result.get("osm_id", 0)) or None,
                        original_query=query,
                        confidence=float(result.get("importance", 0.5)),
                    )
                    
        except Exception as e:
            print(f"Nominatim query failed: {e}")
            return None
    
    def _cache_key(self, query: str) -> str:
        """Generate cache filename."""
        h = hashlib.md5(query.encode()).hexdigest()[:12]
        return f"geo_{h}.json"
    
    def _load_from_cache(self, query: str) -> Optional[GeoLocation]:
        """Load from file cache."""
        cache_file = self.cache_dir / self._cache_key(query)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return GeoLocation(**data)
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, query: str, location: GeoLocation):
        """Save to file cache."""
        cache_file = self.cache_dir / self._cache_key(query)
        try:
            with open(cache_file, "w") as f:
                json.dump(location.to_dict(), f, indent=2)
        except Exception:
            pass
    
    def expand_bbox(
        self,
        location: GeoLocation,
        factor: float = 2.0,
        min_size_deg: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """
        Expand bounding box for investigation.
        
        Args:
            location: Original location
            factor: Expansion factor (2.0 = double the area)
            min_size_deg: Minimum size in degrees
            
        Returns:
            (south, north, west, east) expanded bbox
        """
        lat_range = location.lat_range
        lon_range = location.lon_range
        
        # Current size
        lat_size = lat_range[1] - lat_range[0]
        lon_size = lon_range[1] - lon_range[0]
        
        # Ensure minimum size
        lat_size = max(lat_size, min_size_deg)
        lon_size = max(lon_size, min_size_deg)
        
        # Expand
        lat_expand = lat_size * (factor - 1) / 2
        lon_expand = lon_size * (factor - 1) / 2
        
        return (
            lat_range[0] - lat_expand,
            lat_range[1] + lat_expand,
            lon_range[0] - lon_expand,
            lon_range[1] + lon_expand,
        )
    
    def get_temporal_context(
        self,
        location: GeoLocation,
        event_date: datetime,
        days_before: int = 90,
        days_after: int = 30,
    ) -> Dict[str, Any]:
        """
        Get temporal context for investigation.
        
        Returns time windows for:
        - Precursor analysis (before event)
        - Event window
        - Consequence analysis (after event)
        """
        from datetime import timedelta
        
        return {
            "event_date": event_date.isoformat(),
            "precursor_window": {
                "start": (event_date - timedelta(days=days_before)).isoformat(),
                "end": event_date.isoformat(),
                "days": days_before,
            },
            "event_window": {
                "start": (event_date - timedelta(days=7)).isoformat(),
                "end": (event_date + timedelta(days=7)).isoformat(),
                "days": 14,
            },
            "consequence_window": {
                "start": event_date.isoformat(),
                "end": (event_date + timedelta(days=days_after)).isoformat(),
                "days": days_after,
            },
            "location": location.to_dict(),
        }


# Convenience functions
async def resolve_location(query: str) -> Optional[GeoLocation]:
    """Quick location resolution."""
    resolver = GeoResolver()
    return await resolver.resolve(query)


async def extract_locations(text: str) -> List[GeoLocation]:
    """Extract all locations from text."""
    resolver = GeoResolver()
    return await resolver.resolve_from_text(text)


# CLI test
if __name__ == "__main__":
    async def test():
        resolver = GeoResolver()
        
        # Test known locations
        print("=== Testing Known Locations ===")
        for query in ["Lago Maggiore", "Fram Strait", "fiume ticino"]:
            loc = await resolver.resolve(query)
            if loc:
                print(f"✅ {query}: {loc.lat:.2f}°N, {loc.lon:.2f}°E")
                print(f"   bbox: {loc.bbox}")
                print(f"   type: {loc.place_type}")
            else:
                print(f"❌ {query}: not found")
        
        # Test Nominatim
        print("\n=== Testing Nominatim ===")
        loc = await resolver.resolve("Locarno, Switzerland")
        if loc:
            print(f"✅ Locarno: {loc.lat:.2f}°N, {loc.lon:.2f}°E")
        
        # Test text extraction
        print("\n=== Testing Text Extraction ===")
        text = """
        Le alluvioni del Lago Maggiore nel settembre 1993 e ottobre 2000 
        hanno causato gravi danni nelle zone del Fiume Ticino e nella 
        Pianura Padana. L'evento del 1994 è stato particolarmente grave.
        """
        locs = await resolver.resolve_from_text(text)
        print(f"Found {len(locs)} locations:")
        for loc in locs:
            print(f"  - {loc.name}: {loc.lat:.2f}°N, {loc.lon:.2f}°E")
    
    asyncio.run(test())

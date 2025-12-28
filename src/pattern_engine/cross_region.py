"""
🌍 Cross-Region Pattern Matcher
===============================

Finds similar causal patterns across different geographic regions.
When a pattern is discovered in one region, searches for analogous
patterns elsewhere that could indicate similar risks.

Use case:
    "I found that NAO → heavy rain → flooding in Lake Maggiore.
     Where else might this pattern occur?"

Features:
- Pattern similarity scoring based on causal structure
- Geographic filtering by climate zone, latitude band, coastline type
- Historical event matching
- Transferability assessment

Usage:
    matcher = CrossRegionMatcher()
    
    # Find regions with similar patterns
    matches = await matcher.find_similar_regions(
        pattern=discovered_pattern,
        source_region="Lake Maggiore",
        search_radius_km=2000,
    )
    
    # Check if a pattern applies to a new region
    score = await matcher.assess_transferability(
        pattern=source_pattern,
        target_region="Lake Como",
    )
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import math


class ClimateZone(Enum):
    """Köppen climate classification zones."""
    TROPICAL = "tropical"          # Af, Am, Aw
    ARID = "arid"                  # BWh, BWk, BSh, BSk
    TEMPERATE = "temperate"        # Csa, Csb, Cwa, Cwb, Cfa, Cfb
    CONTINENTAL = "continental"    # Dsa, Dsb, Dwa, Dwb, Dfa, Dfb
    POLAR = "polar"                # ET, EF
    MEDITERRANEAN = "mediterranean"  # Csa, Csb
    OCEANIC = "oceanic"            # Cfb, Cfc
    SUBARCTIC = "subarctic"        # Dfc, Dwc, Dsc


class CoastlineType(Enum):
    """Type of coastline/water body."""
    OCEAN_EXPOSED = "ocean_exposed"
    OCEAN_SHELTERED = "ocean_sheltered"
    INLAND_SEA = "inland_sea"
    LARGE_LAKE = "large_lake"
    RIVER_DELTA = "river_delta"
    FJORD = "fjord"


class TerrainType(Enum):
    """Type of surrounding terrain."""
    COASTAL_PLAIN = "coastal_plain"
    MOUNTAIN_VALLEY = "mountain_valley"
    LOWLAND = "lowland"
    PLATEAU = "plateau"
    ISLAND = "island"


@dataclass
class Region:
    """A geographic region with metadata."""
    id: str
    name: str
    latitude: float
    longitude: float
    climate_zone: ClimateZone
    coastline_type: Optional[CoastlineType] = None
    terrain_type: Optional[TerrainType] = None
    catchment_area_km2: Optional[float] = None
    elevation_m: Optional[float] = None
    population: Optional[int] = None
    historical_events: List[str] = field(default_factory=list)
    known_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other: "Region") -> float:
        """Calculate distance in km to another region (Haversine)."""
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "climate_zone": self.climate_zone.value,
            "coastline_type": self.coastline_type.value if self.coastline_type else None,
            "terrain_type": self.terrain_type.value if self.terrain_type else None,
            "catchment_area_km2": self.catchment_area_km2,
            "elevation_m": self.elevation_m,
        }


@dataclass
class CausalPattern:
    """A causal pattern that can be matched across regions."""
    id: str
    name: str
    causes: List[Dict[str, Any]]  # [{var, lag, score}, ...]
    effect: str
    source_region: str
    climate_drivers: List[str] = field(default_factory=list)  # NAO, ENSO, etc.
    seasonality: Optional[str] = None  # "winter", "summer", "all"
    confidence: float = 0.5
    n_observations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_signature(self) -> Tuple:
        """Get pattern signature for comparison."""
        # Normalize cause names
        causes_sig = tuple(sorted([
            (c["var"].lower(), c.get("lag", 0))
            for c in self.causes
        ]))
        return (self.effect.lower(), causes_sig, tuple(sorted(self.climate_drivers)))
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "causes": self.causes,
            "effect": self.effect,
            "source_region": self.source_region,
            "climate_drivers": self.climate_drivers,
            "seasonality": self.seasonality,
            "confidence": self.confidence,
            "n_observations": self.n_observations,
        }


@dataclass
class RegionMatch:
    """A matched region with similarity score."""
    region: Region
    pattern: CausalPattern
    similarity_score: float
    geographic_score: float
    climatic_score: float
    structural_score: float
    transferability: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "region": self.region.to_dict(),
            "similarity_score": self.similarity_score,
            "components": {
                "geographic": self.geographic_score,
                "climatic": self.climatic_score,
                "structural": self.structural_score,
            },
            "transferability": self.transferability,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


# Known regions database (European flood-prone areas)
KNOWN_REGIONS = [
    Region(
        id="lago_maggiore",
        name="Lake Maggiore",
        latitude=45.95,
        longitude=8.65,
        climate_zone=ClimateZone.TEMPERATE,
        coastline_type=CoastlineType.LARGE_LAKE,
        terrain_type=TerrainType.MOUNTAIN_VALLEY,
        catchment_area_km2=6599,
        elevation_m=194,
        historical_events=["flood_2000_10", "flood_1993_09", "flood_2014_11"],
        known_patterns=["nao_precipitation_flood"],
    ),
    Region(
        id="lake_como",
        name="Lake Como",
        latitude=46.02,
        longitude=9.26,
        climate_zone=ClimateZone.TEMPERATE,
        coastline_type=CoastlineType.LARGE_LAKE,
        terrain_type=TerrainType.MOUNTAIN_VALLEY,
        catchment_area_km2=4572,
        elevation_m=198,
        historical_events=["flood_2002_11"],
    ),
    Region(
        id="danube_delta",
        name="Danube Delta",
        latitude=45.25,
        longitude=29.75,
        climate_zone=ClimateZone.CONTINENTAL,
        coastline_type=CoastlineType.RIVER_DELTA,
        terrain_type=TerrainType.LOWLAND,
        catchment_area_km2=817000,
        elevation_m=0,
        historical_events=["flood_2006_spring", "flood_2010_06"],
    ),
    Region(
        id="rhone_delta",
        name="Rhône Delta (Camargue)",
        latitude=43.50,
        longitude=4.60,
        climate_zone=ClimateZone.MEDITERRANEAN,
        coastline_type=CoastlineType.RIVER_DELTA,
        terrain_type=TerrainType.COASTAL_PLAIN,
        catchment_area_km2=98000,
        elevation_m=0,
    ),
    Region(
        id="elbe_hamburg",
        name="Elbe at Hamburg",
        latitude=53.55,
        longitude=9.99,
        climate_zone=ClimateZone.OCEANIC,
        coastline_type=CoastlineType.OCEAN_SHELTERED,
        terrain_type=TerrainType.LOWLAND,
        catchment_area_km2=148268,
        elevation_m=6,
        historical_events=["flood_2013_06", "storm_surge_1962"],
    ),
    Region(
        id="venice_lagoon",
        name="Venice Lagoon",
        latitude=45.44,
        longitude=12.32,
        climate_zone=ClimateZone.TEMPERATE,
        coastline_type=CoastlineType.INLAND_SEA,
        terrain_type=TerrainType.COASTAL_PLAIN,
        elevation_m=0,
        historical_events=["acqua_alta_2019_11", "acqua_alta_1966_11"],
        known_patterns=["adriatic_sla_flood"],
    ),
    Region(
        id="netherlands_coast",
        name="Netherlands Coast",
        latitude=52.10,
        longitude=4.30,
        climate_zone=ClimateZone.OCEANIC,
        coastline_type=CoastlineType.OCEAN_EXPOSED,
        terrain_type=TerrainType.COASTAL_PLAIN,
        elevation_m=-2,
        historical_events=["flood_1953_02"],
        known_patterns=["nao_storm_surge"],
    ),
    Region(
        id="loire_valley",
        name="Loire Valley",
        latitude=47.40,
        longitude=0.68,
        climate_zone=ClimateZone.OCEANIC,
        terrain_type=TerrainType.LOWLAND,
        catchment_area_km2=117000,
        elevation_m=50,
    ),
    Region(
        id="baltic_coast_germany",
        name="Baltic Coast Germany",
        latitude=54.18,
        longitude=12.08,
        climate_zone=ClimateZone.OCEANIC,
        coastline_type=CoastlineType.INLAND_SEA,
        terrain_type=TerrainType.COASTAL_PLAIN,
        elevation_m=5,
    ),
    Region(
        id="po_delta",
        name="Po Delta",
        latitude=44.95,
        longitude=12.25,
        climate_zone=ClimateZone.TEMPERATE,
        coastline_type=CoastlineType.RIVER_DELTA,
        terrain_type=TerrainType.COASTAL_PLAIN,
        catchment_area_km2=71000,
        elevation_m=0,
    ),
]


class CrossRegionMatcher:
    """
    Matches causal patterns across geographic regions.
    """
    
    def __init__(self, regions: List[Region] = None):
        """
        Args:
            regions: List of regions to search. Uses KNOWN_REGIONS if None.
        """
        self.regions = {r.id: r for r in (regions or KNOWN_REGIONS)}
        self._patterns: Dict[str, CausalPattern] = {}
    
    def add_region(self, region: Region) -> None:
        """Add a region to the database."""
        self.regions[region.id] = region
    
    def add_pattern(self, pattern: CausalPattern) -> None:
        """Add a discovered pattern."""
        self._patterns[pattern.id] = pattern
    
    def find_similar_regions(
        self,
        pattern: CausalPattern,
        source_region: str,
        max_distance_km: float = None,
        same_climate_zone: bool = False,
        min_similarity: float = 0.3,
    ) -> List[RegionMatch]:
        """
        Find regions where a pattern might also occur.
        
        Args:
            pattern: The causal pattern to match
            source_region: ID of the region where pattern was discovered
            max_distance_km: Maximum distance to search
            same_climate_zone: Require same climate zone
            min_similarity: Minimum similarity score
            
        Returns:
            List of RegionMatch sorted by similarity
        """
        if source_region not in self.regions:
            raise ValueError(f"Unknown source region: {source_region}")
        
        source = self.regions[source_region]
        matches = []
        
        for region_id, region in self.regions.items():
            if region_id == source_region:
                continue
            
            # Distance filter
            if max_distance_km:
                distance = source.distance_to(region)
                if distance > max_distance_km:
                    continue
            
            # Climate zone filter
            if same_climate_zone and region.climate_zone != source.climate_zone:
                continue
            
            # Calculate similarity
            match = self._calculate_match(pattern, source, region)
            
            if match.similarity_score >= min_similarity:
                matches.append(match)
        
        # Sort by similarity
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
    
    def _calculate_match(
        self,
        pattern: CausalPattern,
        source: Region,
        target: Region,
    ) -> RegionMatch:
        """Calculate similarity between source and target regions for a pattern."""
        
        # 1. Geographic similarity
        distance = source.distance_to(target)
        geo_score = max(0, 1 - distance / 5000)  # 5000km = 0 similarity
        
        # Latitude band bonus
        lat_diff = abs(source.latitude - target.latitude)
        if lat_diff < 5:
            geo_score += 0.1
        
        # 2. Climatic similarity
        climate_score = 0.0
        
        # Same climate zone is strong indicator
        if source.climate_zone == target.climate_zone:
            climate_score = 1.0
        else:
            # Partial scores for similar zones
            similar_zones = {
                ClimateZone.TEMPERATE: [ClimateZone.OCEANIC, ClimateZone.MEDITERRANEAN],
                ClimateZone.OCEANIC: [ClimateZone.TEMPERATE, ClimateZone.SUBARCTIC],
                ClimateZone.MEDITERRANEAN: [ClimateZone.TEMPERATE],
                ClimateZone.CONTINENTAL: [ClimateZone.SUBARCTIC],
            }
            if target.climate_zone in similar_zones.get(source.climate_zone, []):
                climate_score = 0.6
        
        # 3. Structural similarity (terrain, coastline)
        struct_score = 0.0
        
        if source.coastline_type == target.coastline_type:
            struct_score += 0.4
        elif source.coastline_type and target.coastline_type:
            # Similar coastline types
            coastal_similar = {
                CoastlineType.LARGE_LAKE: [CoastlineType.INLAND_SEA],
                CoastlineType.RIVER_DELTA: [CoastlineType.OCEAN_SHELTERED],
                CoastlineType.OCEAN_EXPOSED: [CoastlineType.OCEAN_SHELTERED],
            }
            if target.coastline_type in coastal_similar.get(source.coastline_type, []):
                struct_score += 0.2
        
        if source.terrain_type == target.terrain_type:
            struct_score += 0.4
        
        # Catchment area similarity (log scale)
        if source.catchment_area_km2 and target.catchment_area_km2:
            log_ratio = abs(
                math.log10(source.catchment_area_km2) - 
                math.log10(target.catchment_area_km2)
            )
            struct_score += max(0, 0.2 * (1 - log_ratio / 2))
        
        # Combined score
        similarity = (
            0.25 * geo_score +
            0.35 * climate_score +
            0.40 * struct_score
        )
        
        # Transferability assessment
        transferability = self._assess_transferability(
            pattern, source, target, similarity
        )
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        
        if distance > 2000:
            warnings.append(f"Large distance ({distance:.0f} km) - different regional forcings likely")
        
        if source.climate_zone != target.climate_zone:
            warnings.append(f"Different climate zones ({source.climate_zone.value} vs {target.climate_zone.value})")
        
        if "nao" in [d.lower() for d in pattern.climate_drivers]:
            if target.latitude < 35:
                warnings.append("NAO influence weaker at low latitudes")
            else:
                recommendations.append("Check NAO correlation at target location")
        
        if struct_score > 0.5:
            recommendations.append("Similar terrain suggests pattern may transfer well")
        
        return RegionMatch(
            region=target,
            pattern=pattern,
            similarity_score=float(similarity),
            geographic_score=float(geo_score),
            climatic_score=float(climate_score),
            structural_score=float(struct_score),
            transferability=float(transferability),
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def _assess_transferability(
        self,
        pattern: CausalPattern,
        source: Region,
        target: Region,
        similarity: float,
    ) -> float:
        """Assess how likely a pattern is to transfer to new region."""
        
        transfer = similarity
        
        # Climate driver availability
        for driver in pattern.climate_drivers:
            driver_lower = driver.lower()
            
            # NAO affects North Atlantic / Europe
            if "nao" in driver_lower:
                if target.latitude > 30 and -60 < target.longitude < 40:
                    transfer += 0.1
                else:
                    transfer -= 0.2
            
            # ENSO affects tropical Pacific, but teleconnections reach far
            if "enso" in driver_lower:
                if abs(target.latitude) < 30:
                    transfer += 0.1
        
        # Historical events at target increase confidence
        if target.historical_events:
            effect_type = pattern.effect.lower()
            if any(effect_type in e.lower() for e in target.historical_events):
                transfer += 0.15
        
        return float(np.clip(transfer, 0, 1))
    
    def suggest_monitoring_regions(
        self,
        pattern: CausalPattern,
        source_region: str,
        n_regions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Suggest regions that should be monitored for similar events.
        
        Returns ranked list with monitoring priorities.
        """
        matches = self.find_similar_regions(
            pattern=pattern,
            source_region=source_region,
            min_similarity=0.2,
        )[:n_regions * 2]  # Get extra to filter
        
        suggestions = []
        for match in matches[:n_regions]:
            # Priority based on transferability and whether region has alerts
            priority = "high" if match.transferability > 0.7 else "medium" if match.transferability > 0.4 else "low"
            
            suggestions.append({
                "region": match.region.name,
                "region_id": match.region.id,
                "latitude": match.region.latitude,
                "longitude": match.region.longitude,
                "priority": priority,
                "transferability": match.transferability,
                "similarity": match.similarity_score,
                "reasons": match.recommendations,
                "cautions": match.warnings,
            })
        
        return suggestions


# Convenience functions

async def find_analogous_regions(
    effect: str,
    causes: List[Dict],
    source_region: str,
    climate_drivers: List[str] = None,
) -> List[Dict]:
    """
    Find regions where a discovered pattern might also occur.
    
    Args:
        effect: The effect (e.g., "flood")
        causes: List of causes [{var, lag, score}, ...]
        source_region: Where pattern was discovered
        climate_drivers: Climate indices involved (NAO, ENSO, etc.)
        
    Returns:
        List of matching regions with similarity scores
    """
    pattern = CausalPattern(
        id="query_pattern",
        name=f"{effect} pattern",
        causes=causes,
        effect=effect,
        source_region=source_region,
        climate_drivers=climate_drivers or [],
    )
    
    matcher = CrossRegionMatcher()
    matches = matcher.find_similar_regions(
        pattern=pattern,
        source_region=source_region,
    )
    
    return [m.to_dict() for m in matches]


# CLI test
if __name__ == "__main__":
    print("=== Cross-Region Pattern Matcher Test ===\n")
    
    # Create a pattern discovered at Lake Maggiore
    pattern = CausalPattern(
        id="lago_maggiore_flood_pattern",
        name="NAO-driven Alpine Flooding",
        causes=[
            {"var": "NAO_index", "lag": 7, "score": 0.85},
            {"var": "SST_anomaly", "lag": 5, "score": 0.72},
            {"var": "wind_stress", "lag": 3, "score": 0.68},
            {"var": "precipitation", "lag": 1, "score": 0.95},
        ],
        effect="flood",
        source_region="lago_maggiore",
        climate_drivers=["NAO"],
        seasonality="autumn",
        confidence=0.85,
    )
    
    matcher = CrossRegionMatcher()
    
    print(f"Source: Lake Maggiore")
    print(f"Pattern: {pattern.name}")
    print(f"Causes: {[c['var'] for c in pattern.causes]}")
    print(f"\nSearching for similar regions...\n")
    
    matches = matcher.find_similar_regions(
        pattern=pattern,
        source_region="lago_maggiore",
        max_distance_km=2000,
    )
    
    for match in matches[:5]:
        print(f"📍 {match.region.name}")
        print(f"   Similarity: {match.similarity_score:.2f}")
        print(f"   - Geographic: {match.geographic_score:.2f}")
        print(f"   - Climatic: {match.climatic_score:.2f}")
        print(f"   - Structural: {match.structural_score:.2f}")
        print(f"   Transferability: {match.transferability:.2f}")
        if match.warnings:
            print(f"   ⚠️ {match.warnings[0]}")
        print()
    
    # Get monitoring suggestions
    print("=== Monitoring Suggestions ===\n")
    suggestions = matcher.suggest_monitoring_regions(pattern, "lago_maggiore", n_regions=3)
    
    for sug in suggestions:
        print(f"🔍 {sug['region']} ({sug['priority']} priority)")
        print(f"   Location: {sug['latitude']:.1f}°N, {sug['longitude']:.1f}°E")
        print(f"   Transfer score: {sug['transferability']:.2f}")

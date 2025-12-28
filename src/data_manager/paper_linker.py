"""
📚🛰️ Paper-Data Linker
======================

Automatically connects scientific papers to satellite datasets.
Uses NLP extraction of:
- Geographic regions mentioned
- Variables discussed (SST, SLA, currents, etc.)
- Time periods studied
- Dataset references

Usage:
    linker = PaperDataLinker(catalog)
    
    # Analyze a paper
    links = await linker.analyze_paper(paper)
    
    # Get relevant datasets for a paper
    datasets = await linker.get_datasets_for_paper(paper_id)
    
    # Get papers relevant to a dataset
    papers = await linker.get_papers_for_dataset(dataset_id)
"""

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum
import hashlib

from .catalog import CopernicusCatalog, DataProduct, DataCategory, get_catalog


class LinkType(Enum):
    """Type of link between paper and dataset."""
    USES_DATA = "uses_data"           # Paper explicitly uses this dataset
    MENTIONS_VARIABLE = "mentions_variable"  # Paper mentions variable in dataset
    SAME_REGION = "same_region"       # Paper studies same geographic region
    SAME_TIMEFRAME = "same_timeframe"  # Paper covers same time period
    CITES_DOI = "cites_doi"           # Paper cites dataset DOI
    KEYWORD_MATCH = "keyword_match"   # Keywords overlap
    METHODOLOGY = "methodology"       # Paper uses relevant methodology


@dataclass
class DatasetLink:
    """Link between a paper and a dataset."""
    paper_id: str
    dataset_id: str
    link_type: LinkType
    confidence: float  # 0-1
    evidence: List[str]  # Text snippets that support link
    variables_matched: List[str] = field(default_factory=list)
    regions_matched: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "dataset_id": self.dataset_id,
            "link_type": self.link_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "variables_matched": self.variables_matched,
            "regions_matched": self.regions_matched,
            "timestamp": self.timestamp,
        }


@dataclass 
class ExtractedEntities:
    """Entities extracted from paper text."""
    variables: List[str]
    regions: List[Tuple[str, Optional[Tuple[float, float, float, float]]]]  # name, (lat_min, lat_max, lon_min, lon_max)
    time_periods: List[Tuple[str, str]]  # (start, end) dates
    cited_datasets: List[str]  # DOIs or dataset names
    keywords: List[str]
    methodologies: List[str]


# Vocabulary mappings for NLP extraction
VARIABLE_PATTERNS = {
    # Sea level
    r'\b(sea\s*level\s*anomal[yies]|SLA)\b': 'sla',
    r'\b(absolute\s*dynamic\s*topography|ADT)\b': 'adt',
    r'\b(SSH|sea\s*surface\s*height)\b': 'sla',
    r'\b(geostrophic\s*(current|velocity|flow))\b': 'ugos',
    
    # Temperature
    r'\b(sea\s*surface\s*temperature|SST)\b': 'analysed_sst',
    r'\b(ocean\s*temperature)\b': 'thetao',
    r'\b(heat\s*content|OHC)\b': 'thetao',
    
    # Currents
    r'\b(ocean\s*current|surface\s*current)\b': 'uo',
    r'\b(velocity\s*field)\b': 'uo',
    r'\b(Ekman\s*transport)\b': 'uo',
    
    # Waves
    r'\b(significant\s*wave\s*height|Hs|SWH)\b': 'VHM0',
    r'\b(wave\s*period)\b': 'VTM10',
    r'\b(swell)\b': 'VHM0',
    
    # Wind
    r'\b(wind\s*speed|wind\s*stress)\b': 'wind_speed',
    r'\b(scatterometer)\b': 'wind_speed',
    
    # Salinity
    r'\b(salinity|SSS)\b': 'so',
    r'\b(freshwater\s*flux)\b': 'so',
    
    # Ice
    r'\b(sea\s*ice\s*(concentration|extent))\b': 'ice_conc',
    r'\b(ice\s*edge)\b': 'ice_edge',
}

REGION_PATTERNS = {
    # European seas
    r'\b(Mediterranean\s*Sea)\b': ('Mediterranean', (30, 46, -6, 36)),
    r'\b(Baltic\s*Sea)\b': ('Baltic Sea', (53, 66, 10, 30)),
    r'\b(North\s*Sea)\b': ('North Sea', (51, 62, -5, 10)),
    r'\b(Black\s*Sea)\b': ('Black Sea', (40, 47, 27, 42)),
    r'\b(Adriatic\s*Sea)\b': ('Adriatic Sea', (39, 46, 12, 20)),
    
    # Atlantic
    r'\b(North\s*Atlantic)\b': ('North Atlantic', (20, 65, -80, 0)),
    r'\b(Gulf\s*Stream)\b': ('Gulf Stream', (25, 45, -80, -40)),
    r'\b(Labrador\s*Sea)\b': ('Labrador Sea', (50, 65, -65, -45)),
    
    # Arctic
    r'\b(Arctic\s*Ocean)\b': ('Arctic Ocean', (65, 90, -180, 180)),
    r'\b(Barents\s*Sea)\b': ('Barents Sea', (70, 82, 15, 60)),
    r'\b(Greenland\s*Sea)\b': ('Greenland Sea', (65, 82, -20, 10)),
    r'\b(Fram\s*Strait)\b': ('Fram Strait', (76, 82, -15, 15)),
    
    # Pacific
    r'\b(Pacific\s*Ocean)\b': ('Pacific Ocean', (-60, 60, 100, -80)),
    r'\b(ENSO|El\s*Ni[ñn]o)\b': ('Tropical Pacific', (-5, 5, -170, -80)),
    
    # Other
    r'\b(Southern\s*Ocean)\b': ('Southern Ocean', (-80, -40, -180, 180)),
    r'\b(Indian\s*Ocean)\b': ('Indian Ocean', (-40, 30, 20, 120)),
    
    # Italian lakes/regions (for flood studies)
    r'\b(Lago\s*Maggiore|Lake\s*Maggiore)\b': ('Lake Maggiore', (45.7, 46.2, 8.4, 8.9)),
    r'\b(Po\s*Valley|Pianura\s*Padana)\b': ('Po Valley', (44.5, 46.0, 7.5, 12.5)),
    r'\b(Ligurian\s*Sea)\b': ('Ligurian Sea', (42, 45, 7, 11)),
}

DATASET_DOI_PATTERNS = [
    r'\b(10\.48670/moi-\d+)\b',  # Copernicus DOIs
    r'\b(doi:\s*10\.\d+/[^\s]+)\b',
    r'\b(CMEMS|Copernicus\s*Marine)\b',
    r'\b(AVISO|DUACS)\b',
    r'\b(OSTIA)\b',
    r'\b(GLORYS)\b',
]

METHODOLOGY_PATTERNS = [
    r'\b(PCMCI|causal\s*discovery)\b',
    r'\b(Granger\s*causality)\b',
    r'\b(correlation\s*analysis)\b',
    r'\b(EOF|empirical\s*orthogonal\s*function)\b',
    r'\b(wavelet\s*analysis)\b',
    r'\b(spectral\s*analysis)\b',
    r'\b(satellite\s*altimetry)\b',
    r'\b(remote\s*sensing)\b',
    r'\b(composite\s*analysis)\b',
]


class TextExtractor:
    """Extract entities from scientific text using regex patterns."""
    
    def extract_variables(self, text: str) -> List[str]:
        """Extract oceanographic variables mentioned."""
        variables = set()
        for pattern, var_name in VARIABLE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                variables.add(var_name)
        return list(variables)
    
    def extract_regions(self, text: str) -> List[Tuple[str, Optional[Tuple[float, float, float, float]]]]:
        """Extract geographic regions mentioned."""
        regions = []
        for pattern, (name, bbox) in REGION_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                regions.append((name, bbox))
        
        # Also try to extract coordinate mentions
        coord_pattern = r'(\d+\.?\d*)\s*°?\s*([NS])\s*[,\-–]\s*(\d+\.?\d*)\s*°?\s*([NS])'
        for match in re.finditer(coord_pattern, text):
            lat1 = float(match.group(1)) * (1 if match.group(2) == 'N' else -1)
            lat2 = float(match.group(3)) * (1 if match.group(4) == 'N' else -1)
            # This is partial - would need full implementation
        
        return regions
    
    def extract_time_periods(self, text: str) -> List[Tuple[str, str]]:
        """Extract time periods mentioned."""
        periods = []
        
        # Year ranges: "1993-2020", "from 1993 to 2020"
        year_range = r'(?:from\s+)?(\d{4})\s*[-–to]+\s*(\d{4})'
        for match in re.finditer(year_range, text):
            start_year = match.group(1)
            end_year = match.group(2)
            periods.append((f"{start_year}-01-01", f"{end_year}-12-31"))
        
        # Specific periods: "October 2000", "2000-10"
        month_year = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
        month_map = {
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'May': '05', 'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        for match in re.finditer(month_year, text, re.IGNORECASE):
            month = month_map.get(match.group(1).capitalize(), '01')
            year = match.group(2)
            periods.append((f"{year}-{month}-01", f"{year}-{month}-28"))
        
        return periods
    
    def extract_cited_datasets(self, text: str) -> List[str]:
        """Extract dataset references."""
        datasets = []
        for pattern in DATASET_DOI_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                datasets.append(match.group(1))
        return list(set(datasets))
    
    def extract_methodologies(self, text: str) -> List[str]:
        """Extract methodologies mentioned."""
        methods = []
        for pattern in METHODOLOGY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                methods.append(match.group(1).lower())
        return list(set(methods))
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from abstract/title."""
        # Basic keyword extraction - in production would use NLP
        keywords = set()
        
        # Scientific terms
        sci_terms = [
            'climate', 'variability', 'trend', 'anomaly', 'forcing',
            'circulation', 'transport', 'flux', 'heat', 'freshwater',
            'stratification', 'mixing', 'upwelling', 'downwelling',
            'eddy', 'mesoscale', 'submesoscale', 'fronts',
            'teleconnection', 'NAO', 'AMO', 'ENSO', 'PDO',
            'extreme', 'flood', 'storm', 'surge', 'drought',
        ]
        
        text_lower = text.lower()
        for term in sci_terms:
            if term.lower() in text_lower:
                keywords.add(term)
        
        return list(keywords)
    
    def extract_all(self, text: str) -> ExtractedEntities:
        """Extract all entities from text."""
        return ExtractedEntities(
            variables=self.extract_variables(text),
            regions=self.extract_regions(text),
            time_periods=self.extract_time_periods(text),
            cited_datasets=self.extract_cited_datasets(text),
            keywords=self.extract_keywords(text),
            methodologies=self.extract_methodologies(text),
        )


class PaperDataLinker:
    """
    Link scientific papers to relevant datasets.
    """
    
    def __init__(self, catalog: CopernicusCatalog = None):
        self.catalog = catalog or get_catalog()
        self.extractor = TextExtractor()
        self._links_cache: Dict[str, List[DatasetLink]] = {}
    
    def analyze_paper(self, paper: Dict) -> List[DatasetLink]:
        """
        Analyze a paper and find relevant datasets.
        
        Args:
            paper: Dict with keys: id, title, abstract, content (optional)
            
        Returns:
            List of DatasetLink objects
        """
        paper_id = paper.get('id', hashlib.md5(paper.get('title', '').encode()).hexdigest())
        
        # Combine text sources
        text_parts = [
            paper.get('title', ''),
            paper.get('abstract', ''),
            paper.get('content', ''),
        ]
        full_text = ' '.join(filter(None, text_parts))
        
        # Extract entities
        entities = self.extractor.extract_all(full_text)
        
        # Find matching datasets
        links = []
        
        # 1. DOI/Dataset citation matches (highest confidence)
        for dataset_ref in entities.cited_datasets:
            for product in self.catalog.list_products():
                if product.doi and dataset_ref in product.doi:
                    links.append(DatasetLink(
                        paper_id=paper_id,
                        dataset_id=product.product_id,
                        link_type=LinkType.CITES_DOI,
                        confidence=0.95,
                        evidence=[f"Paper cites DOI: {dataset_ref}"],
                    ))
        
        # 2. Variable matches
        for var in entities.variables:
            matching_products = self.catalog.search(variable=var)
            for product in matching_products:
                evidence = [f"Paper mentions variable '{var}' which is in dataset"]
                links.append(DatasetLink(
                    paper_id=paper_id,
                    dataset_id=product.product_id,
                    link_type=LinkType.MENTIONS_VARIABLE,
                    confidence=0.7,
                    evidence=evidence,
                    variables_matched=[var],
                ))
        
        # 3. Geographic region matches
        for region_name, bbox in entities.regions:
            if bbox:
                lat_range = (bbox[0], bbox[1])
                lon_range = (bbox[2], bbox[3])
                matching_products = self.catalog.search(
                    lat_range=lat_range,
                    lon_range=lon_range,
                )
                for product in matching_products:
                    links.append(DatasetLink(
                        paper_id=paper_id,
                        dataset_id=product.product_id,
                        link_type=LinkType.SAME_REGION,
                        confidence=0.6,
                        evidence=[f"Paper studies region: {region_name}"],
                        regions_matched=[region_name],
                    ))
        
        # 4. Time period matches
        for start_date, end_date in entities.time_periods:
            matching_products = self.catalog.search(
                time_range=(start_date, end_date),
            )
            for product in matching_products:
                links.append(DatasetLink(
                    paper_id=paper_id,
                    dataset_id=product.product_id,
                    link_type=LinkType.SAME_TIMEFRAME,
                    confidence=0.5,
                    evidence=[f"Paper covers time period: {start_date} to {end_date}"],
                ))
        
        # 5. Keyword matches
        for product in self.catalog.list_products():
            matching_keywords = set(entities.keywords) & set(k.lower() for k in product.keywords)
            if matching_keywords:
                links.append(DatasetLink(
                    paper_id=paper_id,
                    dataset_id=product.product_id,
                    link_type=LinkType.KEYWORD_MATCH,
                    confidence=0.4 + 0.1 * len(matching_keywords),
                    evidence=[f"Matching keywords: {', '.join(matching_keywords)}"],
                ))
        
        # Deduplicate and merge links
        merged = self._merge_links(links)
        
        # Cache results
        self._links_cache[paper_id] = merged
        
        return merged
    
    def _merge_links(self, links: List[DatasetLink]) -> List[DatasetLink]:
        """Merge multiple links to same dataset."""
        merged_map: Dict[Tuple[str, str], DatasetLink] = {}
        
        for link in links:
            key = (link.paper_id, link.dataset_id)
            if key not in merged_map:
                merged_map[key] = link
            else:
                existing = merged_map[key]
                # Combine evidence
                existing.evidence.extend(link.evidence)
                existing.evidence = list(set(existing.evidence))
                # Combine matched items
                existing.variables_matched = list(set(existing.variables_matched + link.variables_matched))
                existing.regions_matched = list(set(existing.regions_matched + link.regions_matched))
                # Boost confidence (but cap at 0.99)
                existing.confidence = min(0.99, existing.confidence + link.confidence * 0.2)
                # Use highest-priority link type
                type_priority = {
                    LinkType.CITES_DOI: 1,
                    LinkType.USES_DATA: 2,
                    LinkType.MENTIONS_VARIABLE: 3,
                    LinkType.SAME_REGION: 4,
                    LinkType.SAME_TIMEFRAME: 5,
                    LinkType.METHODOLOGY: 6,
                    LinkType.KEYWORD_MATCH: 7,
                }
                if type_priority[link.link_type] < type_priority[existing.link_type]:
                    existing.link_type = link.link_type
        
        # Sort by confidence
        return sorted(merged_map.values(), key=lambda x: x.confidence, reverse=True)
    
    def get_datasets_for_paper(self, paper_id: str) -> List[Dict]:
        """Get datasets relevant to a paper."""
        links = self._links_cache.get(paper_id, [])
        results = []
        for link in links:
            product = self.catalog.get_product(link.dataset_id)
            if product:
                results.append({
                    "dataset": product.to_summary(),
                    "link": link.to_dict(),
                })
        return results
    
    def get_papers_for_dataset(self, dataset_id: str, papers: List[Dict]) -> List[Dict]:
        """Find papers relevant to a dataset."""
        results = []
        for paper in papers:
            links = self.analyze_paper(paper)
            for link in links:
                if link.dataset_id == dataset_id and link.confidence > 0.5:
                    results.append({
                        "paper": paper,
                        "link": link.to_dict(),
                    })
        return sorted(results, key=lambda x: x['link']['confidence'], reverse=True)
    
    def get_download_suggestions(
        self,
        paper: Dict,
        variables: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Suggest datasets to download based on paper analysis.
        
        Returns download configs ready for CMEMSClient.
        """
        links = self.analyze_paper(paper)
        entities = self.extractor.extract_all(
            paper.get('title', '') + ' ' + paper.get('abstract', '')
        )
        
        suggestions = []
        seen_datasets = set()
        
        for link in links:
            if link.dataset_id in seen_datasets:
                continue
            if link.confidence < 0.5:
                continue
                
            seen_datasets.add(link.dataset_id)
            
            # Determine download parameters
            lat_range = None
            lon_range = None
            time_range = None
            
            # Use first region from paper
            if entities.regions:
                _, bbox = entities.regions[0]
                if bbox:
                    lat_range = (bbox[0], bbox[1])
                    lon_range = (bbox[2], bbox[3])
            
            # Use first time period from paper
            if entities.time_periods:
                time_range = entities.time_periods[0]
            
            # Get download config
            config = self.catalog.get_download_config(
                product_id=link.dataset_id,
                variables=variables or link.variables_matched or None,
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=time_range,
            )
            
            if config:
                suggestions.append({
                    "config": config,
                    "reason": link.evidence[0] if link.evidence else "Relevant to paper",
                    "confidence": link.confidence,
                })
        
        return suggestions


# Convenience functions
def link_paper_to_datasets(paper: Dict, catalog: CopernicusCatalog = None) -> List[Dict]:
    """Quick function to link a paper to datasets."""
    linker = PaperDataLinker(catalog)
    links = linker.analyze_paper(paper)
    return [link.to_dict() for link in links]


def suggest_datasets_for_paper(paper: Dict, catalog: CopernicusCatalog = None) -> List[Dict]:
    """Suggest datasets to download based on paper content."""
    linker = PaperDataLinker(catalog)
    return linker.get_download_suggestions(paper)


# CLI test
if __name__ == "__main__":
    # Test with sample paper
    test_paper = {
        "id": "test_001",
        "title": "Sea level anomaly and Mediterranean circulation during the October 2000 flood event",
        "abstract": """
        We analyze the relationship between sea level anomaly (SLA) and atmospheric forcing
        during the severe flood event affecting Lake Maggiore and the Po Valley in October 2000.
        Using satellite altimetry data from AVISO/DUACS and ERA5 reanalysis, we examine
        the role of Mediterranean Sea circulation in moisture transport. SST anomalies in the
        Ligurian Sea are correlated with the extreme precipitation using PCMCI causal discovery.
        The analysis covers the period 2000-2001 with focus on the NAO index teleconnection.
        """,
    }
    
    print("=== Paper-Dataset Linker Test ===\n")
    
    linker = PaperDataLinker()
    
    # Extract entities
    entities = linker.extractor.extract_all(test_paper['abstract'])
    print("Extracted Entities:")
    print(f"  Variables: {entities.variables}")
    print(f"  Regions: {[(r[0], r[1]) for r in entities.regions]}")
    print(f"  Time periods: {entities.time_periods}")
    print(f"  Cited datasets: {entities.cited_datasets}")
    print(f"  Methodologies: {entities.methodologies}")
    print(f"  Keywords: {entities.keywords}")
    
    # Find links
    print("\nDataset Links:")
    links = linker.analyze_paper(test_paper)
    for link in links[:10]:  # Top 10
        print(f"\n  {link.dataset_id}")
        print(f"    Type: {link.link_type.value}")
        print(f"    Confidence: {link.confidence:.2f}")
        print(f"    Evidence: {link.evidence[0] if link.evidence else 'N/A'}")
    
    # Get download suggestions
    print("\nDownload Suggestions:")
    suggestions = linker.get_download_suggestions(test_paper)
    for sug in suggestions[:5]:
        print(f"\n  {sug['config']['product_id']}")
        print(f"    Reason: {sug['reason']}")
        print(f"    Size: ~{sug['config'].get('estimated_size_mb', 'N/A')} MB")

"""
📚 Scientific Literature Scraper
================================

Scrape scientific papers from:
- arXiv (open access preprints)
- Semantic Scholar API
- CrossRef (DOI metadata)

Extract structured information for knowledge graph.
"""

import os
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import xml.etree.ElementTree as ET

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class Paper:
    """Scientific paper metadata."""
    title: str
    authors: List[str]
    abstract: str
    source: str  # arxiv, semantic_scholar, crossref
    
    # IDs
    arxiv_id: str = ""
    doi: str = ""
    semantic_scholar_id: str = ""
    
    # Dates
    published_date: str = ""
    updated_date: str = ""
    
    # Categories
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Links
    pdf_url: str = ""
    html_url: str = ""
    
    # Citations
    citation_count: int = 0
    references: List[str] = field(default_factory=list)
    
    # Full text (if extracted)
    full_text: str = ""
    
    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'source': self.source,
            'arxiv_id': self.arxiv_id,
            'doi': self.doi,
            'published_date': self.published_date,
            'categories': self.categories,
            'keywords': self.keywords,
            'pdf_url': self.pdf_url,
            'citation_count': self.citation_count,
        }


class ArxivScraper:
    """
    arXiv API client for scientific papers.
    
    Usage:
        scraper = ArxivScraper()
        
        # Search for flood papers
        papers = await scraper.search(
            query="flood prediction machine learning",
            max_results=20
        )
        
        # Search by category
        papers = await scraper.search(
            query="lake level",
            categories=["physics.ao-ph", "physics.geo-ph"]
        )
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # Relevant arXiv categories
    CATEGORIES = {
        "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.geo-ph": "Geophysics",
        "stat.ML": "Machine Learning",
        "cs.LG": "Machine Learning (CS)",
        "cs.AI": "Artificial Intelligence",
        "eess.SP": "Signal Processing",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.data-an": "Data Analysis",
    }
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent.parent / "data" / "cache" / "arxiv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = 3.0  # seconds between requests (arXiv requires this)
        self._last_request = 0
    
    async def search(
        self,
        query: str,
        categories: List[str] = None,
        max_results: int = 50,
        sort_by: str = "relevance",  # relevance, lastUpdatedDate, submittedDate
        start: int = 0,
    ) -> List[Paper]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query (supports arXiv search syntax)
            categories: Filter by arXiv categories
            max_results: Maximum papers to return
            sort_by: Sort order
            start: Pagination offset
            
        Returns:
            List of Paper objects
        """
        if not HAS_AIOHTTP:
            print("⚠️ aiohttp required: pip install aiohttp")
            return []
        
        # Build query
        search_query = self._build_query(query, categories)
        
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        
        # Rate limiting
        await self._rate_limit()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params, timeout=30) as response:
                    if response.status != 200:
                        print(f"❌ arXiv API error: {response.status}")
                        return []
                    
                    xml_text = await response.text()
                    return self._parse_response(xml_text)
                    
        except Exception as e:
            print(f"❌ arXiv search error: {e}")
            return []
    
    def _build_query(self, query: str, categories: List[str] = None) -> str:
        """Build arXiv query string."""
        # Main search in title and abstract
        parts = [f'all:{query}']
        
        # Add category filter
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            parts.append(f"({cat_query})")
        
        return " AND ".join(parts)
    
    def _parse_response(self, xml_text: str) -> List[Paper]:
        """Parse arXiv API XML response."""
        papers = []
        
        # Parse XML
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}
        
        root = ET.fromstring(xml_text)
        
        for entry in root.findall('atom:entry', ns):
            try:
                # Extract fields
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                
                abstract = entry.find('atom:summary', ns).text.strip()
                
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns).text
                    authors.append(name)
                
                # IDs
                entry_id = entry.find('atom:id', ns).text
                arxiv_id = entry_id.split('/abs/')[-1]
                
                # Dates
                published = entry.find('atom:published', ns).text[:10]
                updated = entry.find('atom:updated', ns).text[:10]
                
                # Categories
                categories = []
                for cat in entry.findall('atom:category', ns):
                    categories.append(cat.get('term'))
                
                # PDF link
                pdf_url = ""
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break
                
                # DOI if present
                doi = ""
                doi_elem = entry.find('arxiv:doi', ns)
                if doi_elem is not None:
                    doi = doi_elem.text
                
                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    source="arxiv",
                    arxiv_id=arxiv_id,
                    doi=doi,
                    published_date=published,
                    updated_date=updated,
                    categories=categories,
                    pdf_url=pdf_url,
                    html_url=f"https://arxiv.org/abs/{arxiv_id}",
                )
                papers.append(paper)
                
            except Exception as e:
                print(f"⚠️ Error parsing entry: {e}")
                continue
        
        return papers
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        import time
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    async def download_pdf(
        self,
        paper: Paper,
        output_dir: Path = None,
    ) -> Optional[Path]:
        """Download paper PDF."""
        if not paper.pdf_url:
            return None
        
        output_dir = output_dir or self.cache_dir / "pdfs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
        output_path = output_dir / filename
        
        if output_path.exists():
            return output_path
        
        await self._rate_limit()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(paper.pdf_url, timeout=60) as response:
                    if response.status == 200:
                        with open(output_path, 'wb') as f:
                            f.write(await response.read())
                        print(f"✅ Downloaded: {filename}")
                        return output_path
        except Exception as e:
            print(f"❌ PDF download error: {e}")
        
        return None


class SemanticScholarScraper:
    """
    Semantic Scholar API client.
    
    Better for citation analysis and finding related papers.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
    
    async def search(
        self,
        query: str,
        max_results: int = 50,
        year_range: Tuple[int, int] = None,
        fields_of_study: List[str] = None,
    ) -> List[Paper]:
        """
        Search Semantic Scholar.
        
        Fields of study: "Environmental Science", "Geology", "Computer Science", etc.
        """
        if not HAS_AIOHTTP:
            return []
        
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,authors,abstract,year,citationCount,externalIds,publicationTypes,fieldsOfStudy,openAccessPdf",
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                url = f"{self.BASE_URL}/paper/search"
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        print(f"❌ Semantic Scholar API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return self._parse_results(data.get('data', []))
                    
        except Exception as e:
            print(f"❌ Semantic Scholar error: {e}")
            return []
    
    def _parse_results(self, results: List[dict]) -> List[Paper]:
        """Parse API results."""
        papers = []
        
        for item in results:
            try:
                authors = [a.get('name', '') for a in item.get('authors', [])]
                
                external_ids = item.get('externalIds', {})
                
                pdf_url = ""
                if item.get('openAccessPdf'):
                    pdf_url = item['openAccessPdf'].get('url', '')
                
                paper = Paper(
                    title=item.get('title', ''),
                    authors=authors,
                    abstract=item.get('abstract', '') or '',
                    source="semantic_scholar",
                    arxiv_id=external_ids.get('ArXiv', ''),
                    doi=external_ids.get('DOI', ''),
                    semantic_scholar_id=item.get('paperId', ''),
                    published_date=str(item.get('year', '')),
                    citation_count=item.get('citationCount', 0),
                    pdf_url=pdf_url,
                    categories=item.get('fieldsOfStudy', []) or [],
                )
                papers.append(paper)
                
            except Exception as e:
                continue
        
        return papers
    
    async def get_citations(
        self,
        paper_id: str,
        max_results: int = 100,
    ) -> List[Paper]:
        """Get papers that cite this paper."""
        if not HAS_AIOHTTP:
            return []
        
        params = {
            "limit": min(max_results, 1000),
            "fields": "title,authors,year,citationCount",
        }
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                url = f"{self.BASE_URL}/paper/{paper_id}/citations"
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        citing_papers = [item['citingPaper'] for item in data.get('data', [])]
                        return self._parse_results(citing_papers)
        except Exception as e:
            print(f"❌ Citations error: {e}")
        
        return []


class LiteratureScraper:
    """
    Combined literature scraper for investigation.
    
    Usage:
        scraper = LiteratureScraper()
        
        # Search for papers on Lake Maggiore floods
        papers = await scraper.search(
            query="Lake Maggiore flood October 2000",
            sources=["arxiv", "semantic_scholar"]
        )
        
        # Get flood-related papers
        papers = await scraper.search_flood_papers(
            location="Lago Maggiore",
            event_years=[1993, 2000, 2014]
        )
    """
    
    def __init__(self):
        self.arxiv = ArxivScraper()
        self.semantic = SemanticScholarScraper()
    
    async def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 50,
    ) -> List[Paper]:
        """
        Search multiple sources.
        
        Args:
            query: Search query
            sources: ["arxiv", "semantic_scholar"]
            max_results: Max per source
        """
        sources = sources or ["arxiv", "semantic_scholar"]
        all_papers = []
        
        tasks = []
        
        if "arxiv" in sources:
            tasks.append(self.arxiv.search(query, max_results=max_results))
        
        if "semantic_scholar" in sources:
            tasks.append(self.semantic.search(query, max_results=max_results))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            elif isinstance(result, Exception):
                print(f"⚠️ Search error: {result}")
        
        # Deduplicate by DOI
        seen_dois = set()
        unique_papers = []
        for paper in all_papers:
            if paper.doi:
                if paper.doi not in seen_dois:
                    seen_dois.add(paper.doi)
                    unique_papers.append(paper)
            else:
                unique_papers.append(paper)
        
        return unique_papers
    
    async def search_flood_papers(
        self,
        location: str,
        event_years: List[int] = None,
        max_results: int = 50,
    ) -> List[Paper]:
        """Search for flood-related papers."""
        queries = [
            f"{location} flood",
            f"{location} extreme precipitation",
            f"{location} flood prediction",
            f"{location} hydrological modeling",
        ]
        
        if event_years:
            for year in event_years:
                queries.append(f"{location} flood {year}")
        
        all_papers = []
        
        for q in queries[:3]:  # Limit to first 3 queries
            papers = await self.search(q, max_results=max_results // 3)
            all_papers.extend(papers)
            await asyncio.sleep(1)  # Rate limiting
        
        return all_papers
    
    async def search_climate_papers(
        self,
        topic: str,
        max_results: int = 50,
    ) -> List[Paper]:
        """Search for climate/meteorology papers."""
        categories = ["physics.ao-ph", "physics.geo-ph"]
        
        return await self.arxiv.search(
            query=topic,
            categories=categories,
            max_results=max_results,
        )


# CLI test
if __name__ == "__main__":
    async def test():
        scraper = LiteratureScraper()
        
        print("=== Searching arXiv ===")
        arxiv_papers = await scraper.arxiv.search(
            query="flood prediction neural network",
            max_results=5,
        )
        for p in arxiv_papers:
            print(f"\n📄 {p.title}")
            print(f"   Authors: {', '.join(p.authors[:3])}")
            print(f"   arXiv: {p.arxiv_id}")
        
        print("\n=== Searching Semantic Scholar ===")
        semantic_papers = await scraper.semantic.search(
            query="Lake Maggiore flood",
            max_results=5,
        )
        for p in semantic_papers:
            print(f"\n📄 {p.title}")
            print(f"   Citations: {p.citation_count}")
        
        print("\n=== Combined Search ===")
        papers = await scraper.search_flood_papers(
            location="Italy Alps",
            event_years=[2000],
            max_results=10,
        )
        print(f"Found {len(papers)} papers")
    
    asyncio.run(test())

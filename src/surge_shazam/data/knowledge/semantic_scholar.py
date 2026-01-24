"""
📚 Semantic Scholar Client
==========================

Academic paper search and knowledge extraction.
Free API (rate limited) - no auth for basic usage.

For Early Warning:
- Find relevant flood/surge research
- Extract causal relationships from abstracts
- Build knowledge graph of precursors
- Track citation networks

API: https://api.semanticscholar.org/
Rate: 100 requests/5 min (unauthenticated)
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class Paper:
    """Academic paper metadata."""
    paper_id: str
    title: str
    abstract: str = ""
    year: int = 0
    authors: List[str] = field(default_factory=list)
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    url: str = ""
    
    # Extracted info
    fields_of_study: List[str] = field(default_factory=list)
    tldr: str = ""  # AI-generated summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract[:500] if self.abstract else "",
            "year": self.year,
            "authors": self.authors[:5],
            "venue": self.venue,
            "citations": self.citation_count,
            "doi": self.doi,
            "fields": self.fields_of_study,
        }
    
    @classmethod
    def from_api(cls, data: Dict) -> "Paper":
        """Create Paper from Semantic Scholar API response."""
        authors = [a.get("name", "") for a in data.get("authors", [])]
        
        return cls(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", "") or "",
            year=data.get("year", 0) or 0,
            authors=authors,
            venue=data.get("venue", "") or "",
            citation_count=data.get("citationCount", 0) or 0,
            doi=data.get("externalIds", {}).get("DOI", ""),
            url=data.get("url", ""),
            fields_of_study=[f.get("category", "") for f in data.get("s2FieldsOfStudy", [])],
            tldr=data.get("tldr", {}).get("text", "") if data.get("tldr") else "",
        )


@dataclass
class SearchResult:
    """Search results container."""
    query: str
    total: int
    papers: List[Paper]
    offset: int = 0


class SemanticScholarClient:
    """
    Semantic Scholar API Client.
    
    Usage:
        client = SemanticScholarClient()
        
        # Search for papers
        results = await client.search(
            query="storm surge prediction machine learning",
            year_range=(2015, 2024),
            limit=50,
        )
        
        # Get paper details
        paper = await client.get_paper("10.1038/s41586-021-03819-2")
        
        # Get citations
        citations = await client.get_citations(paper.paper_id)
        
        # Bulk search for flood-related papers
        flood_papers = await client.search_flood_papers()
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Default fields to request
    DEFAULT_FIELDS = "paperId,title,abstract,year,authors,venue,citationCount,externalIds,url,s2FieldsOfStudy,tldr"
    
    def __init__(
        self,
        api_key: str = None,
        cache_dir: Path = None,
    ):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent.parent / "data" / "cache" / "papers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self._last_request = 0
        self._min_interval = 0.5  # seconds between requests
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        import time
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request = time.time()
    
    async def search(
        self,
        query: str,
        year_range: tuple = None,
        fields_of_study: List[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> SearchResult:
        """
        Search for papers.
        
        Args:
            query: Search query
            year_range: (start_year, end_year) filter
            fields_of_study: Filter by field
            limit: Max results (up to 100)
            offset: Pagination offset
            
        Returns:
            SearchResult with papers
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp required")
            return SearchResult(query, 0, [])
        
        await self._rate_limit()
        
        params = {
            "query": query,
            "fields": self.DEFAULT_FIELDS,
            "limit": min(limit, 100),
            "offset": offset,
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self._get_headers(),
                    timeout=30,
                ) as resp:
                    if resp.status == 429:
                        logger.warning("Rate limited, waiting...")
                        await asyncio.sleep(60)
                        return await self.search(query, year_range, fields_of_study, limit, offset)
                    
                    if resp.status != 200:
                        logger.error(f"Search failed: {resp.status}")
                        return SearchResult(query, 0, [])
                    
                    data = await resp.json()
                    
                    papers = [Paper.from_api(p) for p in data.get("data", [])]
                    total = data.get("total", len(papers))
                    
                    logger.info(f"Found {total} papers for '{query}'")
                    
                    return SearchResult(
                        query=query,
                        total=total,
                        papers=papers,
                        offset=offset,
                    )
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResult(query, 0, [])
    
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Get paper details by ID.
        
        Args:
            paper_id: Semantic Scholar ID, DOI, or arXiv ID
            
        Returns:
            Paper or None
        """
        if not HAS_AIOHTTP:
            return None
        
        await self._rate_limit()
        
        # Handle DOI format
        if paper_id.startswith("10."):
            paper_id = f"DOI:{paper_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/paper/{paper_id}",
                    params={"fields": self.DEFAULT_FIELDS},
                    headers=self._get_headers(),
                    timeout=30,
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Get paper failed: {resp.status}")
                        return None
                    
                    data = await resp.json()
                    return Paper.from_api(data)
                    
        except Exception as e:
            logger.error(f"Get paper error: {e}")
            return None
    
    async def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> List[Paper]:
        """
        Get papers that cite this paper.
        
        Args:
            paper_id: Paper ID
            limit: Max results
            
        Returns:
            List of citing papers
        """
        if not HAS_AIOHTTP:
            return []
        
        await self._rate_limit()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/paper/{paper_id}/citations",
                    params={"fields": self.DEFAULT_FIELDS, "limit": limit},
                    headers=self._get_headers(),
                    timeout=30,
                ) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    return [Paper.from_api(c.get("citingPaper", {})) for c in data.get("data", [])]
                    
        except Exception as e:
            logger.error(f"Citations error: {e}")
            return []
    
    async def get_references(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> List[Paper]:
        """
        Get papers referenced by this paper.
        """
        if not HAS_AIOHTTP:
            return []
        
        await self._rate_limit()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/paper/{paper_id}/references",
                    params={"fields": self.DEFAULT_FIELDS, "limit": limit},
                    headers=self._get_headers(),
                    timeout=30,
                ) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    return [Paper.from_api(r.get("citedPaper", {})) for r in data.get("data", [])]
                    
        except Exception as e:
            logger.error(f"References error: {e}")
            return []
    
    # =========================================================================
    # Domain-Specific Searches
    # =========================================================================
    
    async def search_flood_papers(
        self,
        year_range: tuple = (2015, 2025),
        limit: int = 100,
    ) -> List[Paper]:
        """Search for flood-related papers."""
        queries = [
            "flood prediction machine learning",
            "storm surge forecasting",
            "flash flood early warning",
            "flood precursors satellite",
        ]
        
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            result = await self.search(query, year_range, limit=limit//len(queries))
            for paper in result.papers:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    all_papers.append(paper)
        
        return all_papers
    
    async def search_causal_discovery_papers(
        self,
        year_range: tuple = (2015, 2025),
        limit: int = 50,
    ) -> List[Paper]:
        """Search for causal discovery in climate papers."""
        queries = [
            "causal discovery climate time series",
            "PCMCI climate",
            "Granger causality meteorology",
            "teleconnections causal",
        ]
        
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            result = await self.search(query, year_range, limit=limit//len(queries))
            for paper in result.papers:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    all_papers.append(paper)
        
        return all_papers
    
    async def search_physics_ml_papers(
        self,
        year_range: tuple = (2018, 2025),
        limit: int = 50,
    ) -> List[Paper]:
        """Search for physics-informed ML papers."""
        queries = [
            "physics informed neural network",
            "physics constrained machine learning",
            "hybrid physics data-driven",
            "scientific machine learning PDE",
        ]
        
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            result = await self.search(query, year_range, limit=limit//len(queries))
            for paper in result.papers:
                if paper.paper_id not in seen_ids:
                    seen_ids.add(paper.paper_id)
                    all_papers.append(paper)
        
        return all_papers
    
    def papers_to_surrealdb_format(self, papers: List[Paper]) -> List[Dict]:
        """
        Convert papers to SurrealDB import format.
        
        Compatible with seed_knowledge_graph.py
        """
        return [
            {
                "paper_id": p.paper_id,
                "cite_key": f"s2_{p.paper_id[:10]}",
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "abstract": p.abstract,
                "venue": p.venue,
                "citation_count": p.citation_count,
                "doi": p.doi,
                "fields": p.fields_of_study,
                "tldr": p.tldr,
                "source": "semantic_scholar",
                "imported_at": datetime.now().isoformat(),
            }
            for p in papers
        ]


# Convenience
Client = SemanticScholarClient


async def search_papers(query: str, limit: int = 50) -> List[Paper]:
    """Quick paper search."""
    client = SemanticScholarClient()
    result = await client.search(query, limit=limit)
    return result.papers


# CLI
if __name__ == "__main__":
    async def test():
        print("=== Semantic Scholar Client Test ===\n")
        
        client = SemanticScholarClient()
        
        print("1. Searching flood prediction papers...")
        result = await client.search(
            "flood prediction machine learning",
            year_range=(2020, 2024),
            limit=10,
        )
        
        print(f"   Found {result.total} total, got {len(result.papers)}")
        
        for paper in result.papers[:5]:
            print(f"\n   📄 {paper.title[:60]}...")
            print(f"      Year: {paper.year}, Citations: {paper.citation_count}")
            print(f"      Authors: {', '.join(paper.authors[:3])}")
            if paper.tldr:
                print(f"      TLDR: {paper.tldr[:100]}...")
        
        if result.papers:
            print(f"\n2. Getting citations for first paper...")
            cites = await client.get_citations(result.papers[0].paper_id, limit=5)
            print(f"   Got {len(cites)} citing papers")
        
        print("\n3. Domain searches...")
        flood = await client.search_flood_papers(limit=20)
        print(f"   Flood papers: {len(flood)}")
        
        causal = await client.search_causal_discovery_papers(limit=20)
        print(f"   Causal discovery: {len(causal)}")
        
        pinn = await client.search_physics_ml_papers(limit=20)
        print(f"   Physics-ML: {len(pinn)}")
        
        print("\n✅ Test complete")
    
    asyncio.run(test())

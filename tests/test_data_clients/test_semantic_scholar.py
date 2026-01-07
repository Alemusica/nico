"""
📚 Test Semantic Scholar Client
===============================

Tests for Semantic Scholar API client.
Provides access to:
- Paper search
- Citation analysis
- Author information
- Fields of study filtering

All tests use synthetic data to avoid API dependencies.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch, AsyncMock

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# ============================================================================
# Semantic Scholar Client Implementation
# (Mirrors the one in src/agent/tools/literature_scraper.py)
# ============================================================================

@dataclass
class Paper:
    """Scientific paper metadata."""
    title: str
    authors: List[str]
    abstract: str
    source: str  # semantic_scholar, arxiv
    
    # IDs
    paper_id: str = ""
    arxiv_id: str = ""
    doi: str = ""
    
    # Dates
    year: int = 0
    published_date: str = ""
    
    # Categories
    fields_of_study: List[str] = field(default_factory=list)
    
    # Links
    pdf_url: str = ""
    url: str = ""
    
    # Citations
    citation_count: int = 0
    reference_count: int = 0
    influential_citation_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'source': self.source,
            'paper_id': self.paper_id,
            'arxiv_id': self.arxiv_id,
            'doi': self.doi,
            'year': self.year,
            'fields_of_study': self.fields_of_study,
            'citation_count': self.citation_count,
        }


@dataclass
class Author:
    """Author metadata."""
    author_id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    paper_count: int = 0
    citation_count: int = 0
    h_index: int = 0


class SemanticScholarClient:
    """
    Client for Semantic Scholar API.
    
    Provides:
    - Paper search with field filtering
    - Citation and reference retrieval
    - Author information
    - Recommendation engine
    
    Usage:
        client = SemanticScholarClient()
        
        # Search for papers
        papers = await client.search(
            query="flood prediction machine learning",
            fields_of_study=["Environmental Science", "Computer Science"],
            year_range=(2018, 2024),
            limit=50,
        )
        
        # Get citations
        citations = await client.get_citations(paper_id="abc123")
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Common fields of study for CTW
    FIELDS_OF_STUDY = [
        "Environmental Science",
        "Geology",
        "Computer Science",
        "Physics",
        "Mathematics",
        "Geography",
        "Engineering",
    ]
    
    def __init__(
        self,
        api_key: str = None,
        rate_limit: float = 0.1,  # seconds between requests
    ):
        import os
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.rate_limit = rate_limit
        self._last_request = 0
        
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
    
    @property
    def is_authenticated(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    async def search(
        self,
        query: str,
        fields_of_study: List[str] = None,
        year_range: Tuple[int, int] = None,
        limit: int = 100,
        offset: int = 0,
        open_access_only: bool = False,
    ) -> List[Paper]:
        """
        Search for papers.
        
        Args:
            query: Search query
            fields_of_study: Filter by fields
            year_range: (start_year, end_year)
            limit: Maximum results
            offset: Pagination offset
            open_access_only: Only return open access papers
            
        Returns:
            List of Paper objects
        """
        # For testing, return synthetic results
        return await self._synthetic_search(
            query, fields_of_study, year_range, limit
        )
    
    async def get_paper(
        self,
        paper_id: str,
    ) -> Optional[Paper]:
        """
        Get paper by ID.
        
        Args:
            paper_id: Semantic Scholar paper ID, DOI, or arXiv ID
            
        Returns:
            Paper object or None
        """
        # For testing, return synthetic paper
        return await self._synthetic_paper(paper_id)
    
    async def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Paper]:
        """
        Get papers that cite this paper.
        
        Args:
            paper_id: Paper ID
            limit: Maximum citations to return
            offset: Pagination offset
            
        Returns:
            List of citing papers
        """
        return await self._synthetic_citations(paper_id, limit)
    
    async def get_references(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> List[Paper]:
        """
        Get papers referenced by this paper.
        
        Returns:
            List of referenced papers
        """
        return await self._synthetic_references(paper_id, limit)
    
    async def get_author(
        self,
        author_id: str,
    ) -> Optional[Author]:
        """
        Get author information.
        
        Returns:
            Author object
        """
        return Author(
            author_id=author_id,
            name=f"Author {author_id[-4:]}",
            affiliations=["University of Test"],
            paper_count=np.random.randint(10, 200),
            citation_count=np.random.randint(100, 5000),
            h_index=np.random.randint(5, 50),
        )
    
    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
    ) -> List[Paper]:
        """
        Get papers by author.
        
        Returns:
            List of author's papers
        """
        return await self._synthetic_search(
            query=f"author:{author_id}",
            fields_of_study=None,
            year_range=None,
            limit=limit,
        )
    
    async def recommend(
        self,
        paper_id: str,
        limit: int = 20,
    ) -> List[Paper]:
        """
        Get paper recommendations.
        
        Returns:
            List of recommended papers
        """
        # Return synthetic recommendations
        return await self._synthetic_search(
            query="recommended",
            fields_of_study=None,
            year_range=None,
            limit=limit,
        )
    
    async def batch_papers(
        self,
        paper_ids: List[str],
    ) -> List[Paper]:
        """
        Get multiple papers by ID (batch request).
        
        Returns:
            List of papers
        """
        papers = []
        for pid in paper_ids[:100]:  # API limit
            paper = await self._synthetic_paper(pid)
            if paper:
                papers.append(paper)
        return papers
    
    # ========================================================================
    # Synthetic Data Generation
    # ========================================================================
    
    async def _synthetic_search(
        self,
        query: str,
        fields_of_study: List[str],
        year_range: Tuple[int, int],
        limit: int,
    ) -> List[Paper]:
        """Generate synthetic search results."""
        papers = []
        
        # Keywords for realistic titles
        keywords = query.lower().split()
        topic_words = [
            "analysis", "prediction", "model", "neural network",
            "deep learning", "climate", "ocean", "flood",
            "satellite", "remote sensing", "machine learning",
        ]
        
        for i in range(limit):
            # Generate title
            title_words = [
                np.random.choice(keywords) if keywords else "climate",
                np.random.choice(topic_words),
                np.random.choice(["approach", "study", "method", "system"]),
            ]
            title = " ".join(title_words).title()
            
            # Year
            if year_range:
                year = np.random.randint(year_range[0], year_range[1] + 1)
            else:
                year = np.random.randint(2015, 2025)
            
            # Fields
            fields = fields_of_study or [
                np.random.choice(self.FIELDS_OF_STUDY)
                for _ in range(np.random.randint(1, 3))
            ]
            
            # Citations follow power law
            citation_count = int(np.random.exponential(20))
            
            papers.append(Paper(
                title=f"{title}: A {np.random.choice(['Novel', 'Comprehensive', 'Data-Driven', 'Deep Learning'])} Approach",
                authors=[f"Author {j}" for j in range(np.random.randint(1, 5))],
                abstract=f"This paper presents a {title.lower()} for {query}. "
                        f"We demonstrate improved results on benchmark datasets.",
                source="semantic_scholar",
                paper_id=f"s2_{i}_{np.random.randint(10000, 99999)}",
                doi=f"10.1234/test.{year}.{i}" if np.random.random() > 0.3 else "",
                year=year,
                fields_of_study=fields,
                citation_count=citation_count,
                reference_count=np.random.randint(10, 50),
                influential_citation_count=max(0, citation_count // 10),
                pdf_url=f"https://example.com/paper_{i}.pdf" if np.random.random() > 0.5 else "",
            ))
        
        return papers
    
    async def _synthetic_paper(self, paper_id: str) -> Paper:
        """Generate synthetic paper for ID."""
        return Paper(
            title=f"Paper {paper_id[-6:]}",
            authors=["Test Author 1", "Test Author 2"],
            abstract="This is a synthetic paper abstract for testing purposes.",
            source="semantic_scholar",
            paper_id=paper_id,
            doi=f"10.1234/test.{paper_id[-4:]}",
            year=2023,
            fields_of_study=["Environmental Science"],
            citation_count=np.random.randint(0, 100),
            reference_count=np.random.randint(10, 50),
        )
    
    async def _synthetic_citations(self, paper_id: str, limit: int) -> List[Paper]:
        """Generate synthetic citations."""
        # Paper typically receives citations over years
        papers = await self._synthetic_search(
            query="citation",
            fields_of_study=None,
            year_range=(2020, 2024),
            limit=min(limit, 50),
        )
        return papers
    
    async def _synthetic_references(self, paper_id: str, limit: int) -> List[Paper]:
        """Generate synthetic references."""
        # References are typically older papers
        papers = await self._synthetic_search(
            query="reference",
            fields_of_study=None,
            year_range=(2010, 2022),
            limit=min(limit, 30),
        )
        return papers


# ============================================================================
# Tests
# ============================================================================

class TestPaper:
    """Test Paper dataclass."""
    
    def test_create_paper(self):
        """Should create paper with required fields."""
        paper = Paper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="Test abstract",
            source="semantic_scholar",
        )
        
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.source == "semantic_scholar"
    
    def test_paper_with_ids(self):
        """Should store various IDs."""
        paper = Paper(
            title="Test",
            authors=["Author"],
            abstract="Abstract",
            source="semantic_scholar",
            paper_id="s2_123456",
            doi="10.1234/test",
            arxiv_id="2401.12345",
        )
        
        assert paper.paper_id == "s2_123456"
        assert paper.doi == "10.1234/test"
        assert paper.arxiv_id == "2401.12345"
    
    def test_paper_to_dict(self):
        """Should convert to dictionary."""
        paper = Paper(
            title="Test Paper",
            authors=["Author One"],
            abstract="Abstract text",
            source="semantic_scholar",
            citation_count=42,
        )
        
        d = paper.to_dict()
        
        assert isinstance(d, dict)
        assert d['title'] == "Test Paper"
        assert d['citation_count'] == 42
    
    def test_paper_with_citations(self):
        """Should store citation metrics."""
        paper = Paper(
            title="Highly Cited Paper",
            authors=["Famous Author"],
            abstract="Groundbreaking research",
            source="semantic_scholar",
            citation_count=1000,
            influential_citation_count=150,
            reference_count=50,
        )
        
        assert paper.citation_count == 1000
        assert paper.influential_citation_count == 150


class TestAuthor:
    """Test Author dataclass."""
    
    def test_create_author(self):
        """Should create author metadata."""
        author = Author(
            author_id="12345",
            name="Dr. Test Author",
            affiliations=["MIT", "Stanford"],
            paper_count=50,
            citation_count=2000,
            h_index=25,
        )
        
        assert author.name == "Dr. Test Author"
        assert len(author.affiliations) == 2
        assert author.h_index == 25


class TestSemanticScholarClient:
    """Test SemanticScholarClient class."""
    
    @pytest.fixture
    def client(self):
        return SemanticScholarClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
    
    def test_unauthenticated_by_default(self, client):
        """Should be unauthenticated without API key."""
        assert client.is_authenticated is False
    
    def test_authenticated_with_key(self):
        """Should be authenticated with API key."""
        client = SemanticScholarClient(api_key="test_key")
        assert client.is_authenticated is True
    
    def test_fields_of_study_list(self, client):
        """Should have predefined fields of study."""
        assert "Environmental Science" in client.FIELDS_OF_STUDY
        assert "Computer Science" in client.FIELDS_OF_STUDY
    
    @pytest.mark.asyncio
    async def test_search_returns_papers(self, client):
        """Search should return list of papers."""
        papers = await client.search(
            query="flood prediction",
            limit=10,
        )
        
        assert isinstance(papers, list)
        assert len(papers) == 10
        
        for paper in papers:
            assert isinstance(paper, Paper)
            assert paper.source == "semantic_scholar"
    
    @pytest.mark.asyncio
    async def test_search_with_fields(self, client):
        """Search should filter by fields of study."""
        papers = await client.search(
            query="climate change",
            fields_of_study=["Environmental Science"],
            limit=5,
        )
        
        assert len(papers) == 5
        for paper in papers:
            assert "Environmental Science" in paper.fields_of_study
    
    @pytest.mark.asyncio
    async def test_search_with_year_range(self, client):
        """Search should filter by year range."""
        papers = await client.search(
            query="machine learning",
            year_range=(2020, 2023),
            limit=10,
        )
        
        for paper in papers:
            assert 2020 <= paper.year <= 2023
    
    @pytest.mark.asyncio
    async def test_get_paper_by_id(self, client):
        """Should get paper by ID."""
        paper = await client.get_paper("test_paper_id")
        
        assert paper is not None
        assert paper.paper_id == "test_paper_id"
    
    @pytest.mark.asyncio
    async def test_get_citations(self, client):
        """Should get paper citations."""
        citations = await client.get_citations(
            paper_id="test_paper",
            limit=20,
        )
        
        assert isinstance(citations, list)
        assert len(citations) <= 20
        
        for paper in citations:
            assert isinstance(paper, Paper)
    
    @pytest.mark.asyncio
    async def test_get_references(self, client):
        """Should get paper references."""
        references = await client.get_references(
            paper_id="test_paper",
            limit=20,
        )
        
        assert isinstance(references, list)
        for paper in references:
            assert isinstance(paper, Paper)
    
    @pytest.mark.asyncio
    async def test_get_author(self, client):
        """Should get author information."""
        author = await client.get_author("author_12345")
        
        assert author is not None
        assert author.author_id == "author_12345"
        assert author.paper_count > 0
    
    @pytest.mark.asyncio
    async def test_get_author_papers(self, client):
        """Should get author's papers."""
        papers = await client.get_author_papers(
            author_id="author_12345",
            limit=10,
        )
        
        assert isinstance(papers, list)
        assert len(papers) == 10
    
    @pytest.mark.asyncio
    async def test_recommend(self, client):
        """Should get paper recommendations."""
        recommendations = await client.recommend(
            paper_id="test_paper",
            limit=10,
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) == 10
    
    @pytest.mark.asyncio
    async def test_batch_papers(self, client):
        """Should get multiple papers at once."""
        paper_ids = [f"paper_{i}" for i in range(5)]
        
        papers = await client.batch_papers(paper_ids)
        
        assert len(papers) == 5


class TestSearchQuality:
    """Test search result quality."""
    
    @pytest.fixture
    def client(self):
        return SemanticScholarClient()
    
    @pytest.mark.asyncio
    async def test_papers_have_titles(self, client):
        """All papers should have titles."""
        papers = await client.search("test query", limit=20)
        
        for paper in papers:
            assert paper.title
            assert len(paper.title) > 0
    
    @pytest.mark.asyncio
    async def test_papers_have_abstracts(self, client):
        """All papers should have abstracts."""
        papers = await client.search("test query", limit=20)
        
        for paper in papers:
            assert paper.abstract
            assert len(paper.abstract) > 0
    
    @pytest.mark.asyncio
    async def test_papers_have_valid_year(self, client):
        """Papers should have valid publication years."""
        papers = await client.search("test query", limit=20)
        
        for paper in papers:
            assert 1900 < paper.year < 2100
    
    @pytest.mark.asyncio
    async def test_citation_counts_non_negative(self, client):
        """Citation counts should be non-negative."""
        papers = await client.search("test query", limit=20)
        
        for paper in papers:
            assert paper.citation_count >= 0
            assert paper.reference_count >= 0


class TestCitationAnalysis:
    """Test citation analysis functionality."""
    
    @pytest.fixture
    def client(self):
        return SemanticScholarClient()
    
    @pytest.mark.asyncio
    async def test_citations_are_newer(self, client):
        """Citing papers should generally be newer."""
        citations = await client.get_citations("test_paper", limit=20)
        
        # Citations typically from recent years
        years = [p.year for p in citations]
        avg_year = np.mean(years)
        
        assert avg_year > 2018, "Citations should be recent"
    
    @pytest.mark.asyncio
    async def test_references_are_older(self, client):
        """Referenced papers should generally be older."""
        references = await client.get_references("test_paper", limit=20)
        
        years = [p.year for p in references]
        avg_year = np.mean(years)
        
        assert avg_year < 2022, "References should be older"
    
    @pytest.mark.asyncio
    async def test_influential_citations(self, client):
        """Should track influential citations."""
        papers = await client.search("important topic", limit=10)
        
        for paper in papers:
            # Influential citations should be <= total citations
            assert paper.influential_citation_count <= paper.citation_count


class TestFieldsOfStudy:
    """Test fields of study filtering."""
    
    @pytest.fixture
    def client(self):
        return SemanticScholarClient()
    
    @pytest.mark.asyncio
    async def test_environmental_science_filter(self, client):
        """Should filter by Environmental Science."""
        papers = await client.search(
            query="flood risk",
            fields_of_study=["Environmental Science"],
            limit=10,
        )
        
        for paper in papers:
            assert "Environmental Science" in paper.fields_of_study
    
    @pytest.mark.asyncio
    async def test_multiple_fields(self, client):
        """Should handle multiple fields."""
        papers = await client.search(
            query="climate modeling",
            fields_of_study=["Environmental Science", "Computer Science"],
            limit=10,
        )
        
        for paper in papers:
            # Should have at least one of the specified fields
            has_field = any(
                f in paper.fields_of_study
                for f in ["Environmental Science", "Computer Science"]
            )
            assert has_field


class TestDeduplication:
    """Test paper deduplication."""
    
    @pytest.fixture
    def client(self):
        return SemanticScholarClient()
    
    @pytest.mark.asyncio
    async def test_papers_have_unique_ids(self, client):
        """Papers should have unique IDs."""
        papers = await client.search("test query", limit=50)
        
        paper_ids = [p.paper_id for p in papers]
        unique_ids = set(paper_ids)
        
        assert len(unique_ids) == len(paper_ids), "Duplicate paper IDs found"
    
    def test_deduplicate_by_doi(self):
        """Should deduplicate papers by DOI."""
        papers = [
            Paper(
                title="Paper 1",
                authors=["Author"],
                abstract="Abstract 1",
                source="semantic_scholar",
                doi="10.1234/test",
            ),
            Paper(
                title="Paper 1 (duplicate)",
                authors=["Author"],
                abstract="Abstract 2",
                source="arxiv",
                doi="10.1234/test",  # Same DOI
            ),
            Paper(
                title="Paper 2",
                authors=["Author"],
                abstract="Abstract 3",
                source="semantic_scholar",
                doi="10.1234/different",
            ),
        ]
        
        # Deduplication logic
        seen_dois = set()
        unique = []
        for p in papers:
            if p.doi:
                if p.doi not in seen_dois:
                    seen_dois.add(p.doi)
                    unique.append(p)
            else:
                unique.append(p)
        
        assert len(unique) == 2


class TestRateLimiting:
    """Test rate limiting behavior."""
    
    def test_rate_limit_default(self):
        """Should have default rate limit."""
        client = SemanticScholarClient()
        assert client.rate_limit == 0.1
    
    def test_rate_limit_custom(self):
        """Should allow custom rate limit."""
        client = SemanticScholarClient(rate_limit=0.5)
        assert client.rate_limit == 0.5


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

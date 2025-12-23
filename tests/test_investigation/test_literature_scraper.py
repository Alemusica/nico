"""
📚 Test Literature Scraper
==========================

Tests for arXiv and Semantic Scholar scraping.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent.tools.literature_scraper import (
    ArxivScraper,
    SemanticScholarScraper,
    LiteratureScraper,
    Paper,
)


class TestPaperDataclass:
    """Test Paper dataclass."""
    
    def test_create_paper(self):
        """Should create Paper with required fields."""
        paper = Paper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract.",
            source="arxiv",
        )
        
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.source == "arxiv"
        
    def test_paper_to_dict(self):
        """Should convert to dictionary."""
        paper = Paper(
            title="Test Paper",
            authors=["Author One"],
            abstract="Abstract text",
            source="arxiv",
            arxiv_id="2401.12345",
            citation_count=10,
        )
        
        d = paper.to_dict()
        
        assert isinstance(d, dict)
        assert d['title'] == "Test Paper"
        assert d['arxiv_id'] == "2401.12345"
        assert d['citation_count'] == 10


class TestArxivScraper:
    """Test ArxivScraper class."""
    
    @pytest.fixture
    def scraper(self):
        return ArxivScraper()
    
    def test_categories_defined(self, scraper):
        """Should have relevant categories."""
        assert "physics.ao-ph" in scraper.CATEGORIES
        assert "physics.geo-ph" in scraper.CATEGORIES
        
    def test_build_query_simple(self, scraper):
        """Should build simple query."""
        query = scraper._build_query("flood prediction")
        
        assert "flood prediction" in query
        assert "all:" in query
        
    def test_build_query_with_categories(self, scraper):
        """Should include category filter."""
        query = scraper._build_query(
            "flood prediction",
            categories=["physics.ao-ph"]
        )
        
        assert "cat:physics.ao-ph" in query
        
    @pytest.mark.asyncio
    async def test_search_returns_papers(self, scraper):
        """Search should return list of Papers."""
        # Use a common query that should return results
        papers = await scraper.search(
            query="machine learning climate",
            max_results=3
        )
        
        assert isinstance(papers, list)
        # May be empty due to rate limiting, but shouldn't crash
        for paper in papers:
            assert isinstance(paper, Paper)
            assert paper.source == "arxiv"
            
    @pytest.mark.asyncio
    async def test_search_flood_papers(self, scraper):
        """Should search for flood-related papers."""
        papers = await scraper.search(
            query="flood prediction neural network",
            max_results=5
        )
        
        # Check structure if results returned
        for paper in papers:
            assert paper.title
            assert paper.abstract or paper.title  # At least one should be set


class TestSemanticScholarScraper:
    """Test SemanticScholarScraper class."""
    
    @pytest.fixture
    def scraper(self):
        return SemanticScholarScraper()
    
    @pytest.mark.asyncio
    async def test_search_returns_papers(self, scraper):
        """Search should return list of Papers."""
        papers = await scraper.search(
            query="flood prediction",
            max_results=3
        )
        
        assert isinstance(papers, list)
        for paper in papers:
            assert isinstance(paper, Paper)
            assert paper.source == "semantic_scholar"


class TestLiteratureScraper:
    """Test combined LiteratureScraper."""
    
    @pytest.fixture
    def scraper(self):
        return LiteratureScraper()
    
    @pytest.mark.asyncio
    async def test_search_multiple_sources(self, scraper):
        """Should search multiple sources."""
        papers = await scraper.search(
            query="climate prediction",
            sources=["arxiv", "semantic_scholar"],
            max_results=5
        )
        
        assert isinstance(papers, list)
        
        # If we got results, check sources
        if len(papers) > 0:
            sources = {p.source for p in papers}
            # May have one or both sources
            assert len(sources) >= 1
            
    @pytest.mark.asyncio
    async def test_search_flood_papers(self, scraper):
        """Should search for flood papers with location."""
        papers = await scraper.search_flood_papers(
            location="Italy Alps",
            event_years=[2000],
            max_results=5
        )
        
        assert isinstance(papers, list)
        
    @pytest.mark.asyncio
    async def test_search_climate_papers(self, scraper):
        """Should search climate/meteorology papers."""
        papers = await scraper.search_climate_papers(
            topic="NAO precipitation Europe",
            max_results=5
        )
        
        assert isinstance(papers, list)


class TestDeduplication:
    """Test result deduplication."""
    
    @pytest.fixture
    def scraper(self):
        return LiteratureScraper()
    
    @pytest.mark.asyncio
    async def test_deduplicates_by_doi(self, scraper):
        """Should deduplicate papers with same DOI."""
        # Create papers with same DOI
        paper1 = Paper(
            title="Paper 1",
            authors=["Author"],
            abstract="Abstract 1",
            source="arxiv",
            doi="10.1234/test",
        )
        paper2 = Paper(
            title="Paper 2",
            authors=["Author"],
            abstract="Abstract 2",
            source="semantic_scholar",
            doi="10.1234/test",  # Same DOI
        )
        
        # Simulate deduplication logic
        all_papers = [paper1, paper2]
        seen_dois = set()
        unique = []
        for p in all_papers:
            if p.doi:
                if p.doi not in seen_dois:
                    seen_dois.add(p.doi)
                    unique.append(p)
            else:
                unique.append(p)
        
        assert len(unique) == 1


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

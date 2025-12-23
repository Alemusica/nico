"""
🕵️ Test Investigation Agent
============================

Integration tests for the full investigation pipeline.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent.investigation_agent import (
    InvestigationAgent,
    InvestigationResult,
    EventContext,
    QueryParser,
)


class TestQueryParser:
    """Test natural language query parsing."""
    
    @pytest.fixture
    def parser(self):
        return QueryParser()
    
    def test_parse_lago_maggiore_query(self, parser):
        """Should parse Lago Maggiore flood query."""
        ctx = parser.parse("analizza le alluvioni del Lago Maggiore nel 2000")
        
        assert ctx.location_name is not None
        assert "maggiore" in ctx.location_name.lower()
        assert ctx.event_type == "flood"
        assert 2000 in ctx.years_of_interest or "2000" in ctx.start_date
        
    def test_parse_known_event_lago_maggiore_2000(self, parser):
        """Should recognize known Lago Maggiore 2000 event."""
        ctx = parser.parse("lago maggiore 2000")
        
        # Should match known event
        assert "maggiore" in ctx.location_name.lower()
        assert ctx.start_date  # Should have specific dates for known event
        
    def test_parse_extract_flood_event_type(self, parser):
        """Should extract flood event type from Italian."""
        ctx = parser.parse("alluvione in Piemonte 1994")
        assert ctx.event_type == "flood"
        
    def test_parse_extract_drought_event_type(self, parser):
        """Should extract drought event type."""
        ctx = parser.parse("siccità in Lombardia 2022")
        assert ctx.event_type == "drought"
        
    def test_parse_extract_years(self, parser):
        """Should extract years from query."""
        years = parser._extract_years("eventi tra il 1990 e il 2000")
        
        assert isinstance(years, list)
        assert 1990 in years or 2000 in years
        
    def test_parse_extract_year_range(self, parser):
        """Should extract year range."""
        years = parser._extract_years("1993-2000")
        
        assert 1993 in years
        assert 2000 in years
        
    def test_parse_build_keywords(self, parser):
        """Should build search keywords."""
        keywords = parser._build_keywords(
            query="alluvione lago maggiore",
            event_type="flood",
            location="Lago Maggiore"
        )
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "Lago Maggiore" in keywords


class TestEventContext:
    """Test EventContext dataclass."""
    
    def test_create_event_context(self):
        """Should create EventContext."""
        ctx = EventContext(
            location_name="Lago Maggiore",
            event_type="flood",
            start_date="2000-10-01",
            end_date="2000-10-31",
        )
        
        assert ctx.location_name == "Lago Maggiore"
        assert ctx.event_type == "flood"
        
    def test_default_values(self):
        """Should have reasonable defaults."""
        ctx = EventContext(location_name="Test")
        
        assert ctx.event_type == "flood"  # Default
        assert ctx.temporal_window_days == 90
        assert ctx.spatial_buffer_deg == 1.0


class TestInvestigationResult:
    """Test InvestigationResult dataclass."""
    
    def test_create_result(self):
        """Should create InvestigationResult."""
        result = InvestigationResult(
            query="test query",
            event_context=EventContext(location_name="Test"),
        )
        
        assert result.query == "test query"
        assert isinstance(result.data_sources, list)
        assert isinstance(result.papers, list)
        
    def test_result_to_dict(self):
        """Should convert to dictionary."""
        result = InvestigationResult(
            query="test query",
            event_context=EventContext(
                location_name="Test Location",
                event_type="flood",
                start_date="2000-01-01",
                end_date="2000-12-31",
            ),
        )
        result.confidence = 0.75
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d['query'] == "test query"
        assert d['location'] == "Test Location"
        assert d['confidence'] == 0.75


class TestInvestigationAgent:
    """Test InvestigationAgent class."""
    
    @pytest.fixture
    def agent(self):
        return InvestigationAgent()
    
    def test_create_agent(self, agent):
        """Should create agent."""
        assert agent is not None
        assert agent.query_parser is not None
        
    def test_lazy_tool_initialization(self, agent):
        """Tools should be lazily initialized."""
        # Before access, should be None
        assert agent._geo_resolver is None
        
        # After access (if available), should be initialized
        resolver = agent.geo_resolver
        # May be None if import fails, but shouldn't crash
        
    @pytest.mark.asyncio
    async def test_investigate_returns_result(self, agent):
        """Investigation should return InvestigationResult."""
        result = await agent.investigate(
            "analizza le alluvioni del Lago Maggiore nel 2000",
            collect_satellite=False,  # Skip to speed up test
            collect_reanalysis=False,
            collect_papers=False,
            collect_news=False,
            run_correlation=False,
            expand_search=False,
        )
        
        assert isinstance(result, InvestigationResult)
        assert result.query is not None
        assert result.event_context is not None
        
    @pytest.mark.asyncio
    async def test_investigate_parses_location(self, agent):
        """Investigation should parse location from query."""
        result = await agent.investigate(
            "alluvioni Lago Maggiore 2000",
            collect_satellite=False,
            collect_reanalysis=False,
            collect_papers=False,
            run_correlation=False,
            expand_search=False,
        )
        
        assert "maggiore" in result.event_context.location_name.lower()


class TestFullPipeline:
    """Full pipeline integration tests."""
    
    @pytest.fixture
    def agent(self):
        return InvestigationAgent()
    
    @pytest.mark.asyncio
    async def test_full_investigation_lago_maggiore(self, agent):
        """Full investigation for Lago Maggiore (with fallbacks)."""
        result = await agent.investigate(
            "analizza le alluvioni del Lago Maggiore nell'ottobre 2000",
            collect_satellite=True,  # Will use synthetic fallback
            collect_reanalysis=True,  # Will use synthetic fallback
            collect_climate_indices=True,  # Will use synthetic fallback
            collect_papers=True,  # Will try real APIs
            run_correlation=True,
            expand_search=True,
        )
        
        assert isinstance(result, InvestigationResult)
        
        # Check we got some data
        print(f"\n📊 Data sources collected: {len(result.data_sources)}")
        print(f"📚 Papers found: {len(result.papers)}")
        print(f"🔗 Correlations: {len(result.correlations)}")
        print(f"💡 Key findings: {len(result.key_findings)}")
        
        # Should have some findings even with fallback data
        assert result.event_context.location_name is not None
        
    @pytest.mark.asyncio
    async def test_pipeline_generates_findings(self, agent):
        """Pipeline should generate key findings."""
        result = await agent.investigate(
            "Lago Maggiore flood October 2000",
            collect_satellite=True,
            collect_reanalysis=True,
            collect_climate_indices=True,
            collect_papers=False,  # Skip papers for speed
            run_correlation=True,
            expand_search=True,
        )
        
        # Should generate findings
        assert isinstance(result.key_findings, list)
        assert isinstance(result.recommendations, list)
        
        # Confidence should be calculated
        assert 0 <= result.confidence <= 1


class TestKnownEvents:
    """Test with known historical events."""
    
    @pytest.fixture
    def parser(self):
        return QueryParser()
    
    def test_lago_maggiore_2000_known(self, parser):
        """Lago Maggiore 2000 should be recognized."""
        ctx = parser.parse("lago maggiore 2000")
        
        # Should use known event data
        assert ctx.start_date  # Has specific date
        assert "2000-10" in ctx.start_date  # October 2000
        
    def test_lago_maggiore_1993_known(self, parser):
        """Lago Maggiore 1993 should be recognized."""
        ctx = parser.parse("lago maggiore 1993")
        
        assert ctx.start_date
        assert "1993" in ctx.start_date


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

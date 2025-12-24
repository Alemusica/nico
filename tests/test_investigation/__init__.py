"""
🧪 Test Suite for Investigation Agent
=====================================

Modular tests for:
1. Geo Resolver
2. CMEMS Client
3. ERA5 Client
4. Climate Indices
5. Literature Scraper
6. PDF Parser
7. Investigation Agent (full pipeline)

Run all tests:
    pytest tests/test_investigation/ -v

Run specific module:
    pytest tests/test_investigation/test_geo_resolver.py -v

Run with coverage:
    pytest tests/test_investigation/ --cov=src/agent --cov=src/surge_shazam/data
"""

import pytest
import asyncio
from pathlib import Path

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Test configuration
TEST_LOCATION = "Lago Maggiore"
TEST_LAT = 45.95
TEST_LON = 8.65
TEST_DATE_RANGE = ("2000-10-01", "2000-10-31")
TEST_EVENT_DATE = "2000-10-15"


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def lago_maggiore_bbox():
    """Bounding box for Lago Maggiore area."""
    return (8.0, 45.0, 9.5, 46.5)


@pytest.fixture
def test_query():
    """Sample investigation query."""
    return "analizza le alluvioni del Lago Maggiore nell'ottobre 2000"

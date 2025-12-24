"""
API Test Fixtures
=================
Shared fixtures for API endpoint testing.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent.parent.absolute()
tests_dir = str(project_root / "tests")
if tests_dir in sys.path:
    sys.path.remove(tests_dir)
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_data_service():
    """Mock DataService for testing."""
    service = Mock()
    service.get_dataset = Mock(return_value=None)
    service.get_metadata = Mock(return_value=None)
    service.get_sample_data = Mock(return_value=None)
    service.list_datasets = Mock(return_value=[])
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = AsyncMock()
    service.check_availability = AsyncMock(return_value=True)
    service.generate_hypotheses = AsyncMock(return_value=[])
    service.interpret_dataset = AsyncMock(return_value=Mock(
        columns=[],
        temporal_column=None,
        suggested_targets=[],
        domain=None,
        summary="Test summary"
    ))
    return service


@pytest.fixture
def mock_knowledge_service():
    """Mock knowledge service for testing."""
    service = AsyncMock()
    service.add_paper = AsyncMock(return_value={"id": "test-paper-id"})
    service.get_paper = AsyncMock(return_value=None)
    service.search_papers = AsyncMock(return_value=[])
    return service


@pytest.fixture
def test_client():
    """FastAPI test client."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def mock_investigation_agent():
    """Mock investigation agent."""
    agent = AsyncMock()
    agent.investigate = AsyncMock(return_value=Mock(
        query="test query",
        event_context=Mock(
            location_name="Test Location",
            event_type="flood",
            start_date="2000-01-01",
            end_date="2000-01-31"
        ),
        data_sources=[],
        papers=[],
        correlations=[],
        key_findings=["Test finding"],
        recommendations=["Test recommendation"],
        confidence=0.8,
        to_dict=lambda: {}
    ))
    return agent

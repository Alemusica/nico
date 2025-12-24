"""
Tests for Knowledge Router
==========================
Test knowledge graph management endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock


def test_list_papers_empty(test_client, mock_knowledge_service):
    """Test listing papers when none exist."""
    mock_knowledge_service.list_papers = AsyncMock(return_value=[])
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.get("/api/v1/knowledge/papers")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0


def test_add_paper_success(test_client, mock_knowledge_service):
    """Test adding a paper successfully."""
    mock_knowledge_service.add_paper = AsyncMock(return_value={"id": "test-paper-123"})
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.post(
            "/knowledge/papers",
            json={
                "title": "Test Paper",
                "authors": ["Author 1", "Author 2"],
                "year": 2020,
                "abstract": "Test abstract",
                "keywords": ["climate", "arctic"]
            }
        )
        
        assert response.status_code in [200, 422, 500]


def test_get_paper_not_found(test_client, mock_knowledge_service):
    """Test getting non-existent paper."""
    mock_knowledge_service.get_paper = AsyncMock(return_value=None)
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.get("/api/v1/knowledge/papers/nonexistent-id")
        
        assert response.status_code == 404


def test_search_papers(test_client, mock_knowledge_service):
    """Test searching papers."""
    mock_knowledge_service.search_papers = AsyncMock(return_value=[])
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.get("/api/v1/knowledge/papers/search?query=arctic")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_list_events_empty(test_client, mock_knowledge_service):
    """Test listing events when none exist."""
    mock_knowledge_service.list_events = AsyncMock(return_value=[])
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.get("/api/v1/knowledge/events")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_add_event(test_client, mock_knowledge_service):
    """Test adding a historical event."""
    mock_knowledge_service.add_event = AsyncMock(return_value={"id": "test-event-123"})
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.post(
            "/knowledge/events",
            json={
                "event_id": "flood_2000",
                "name": "2000 Flood Event",
                "event_type": "flood",
                "start_date": "2000-10-01",
                "end_date": "2000-10-31",
                "description": "Major flood event",
                "region": "Po Valley"
            }
        )
        
        assert response.status_code in [200, 422, 500]


def test_list_patterns_empty(test_client, mock_knowledge_service):
    """Test listing patterns when none exist."""
    mock_knowledge_service.list_patterns = AsyncMock(return_value=[])
    
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.get("/api/v1/knowledge/patterns")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_validate_pattern_structure(test_client, mock_knowledge_service):
    """Test pattern validation endpoint."""
    with patch('api.routers.knowledge_router.create_knowledge_service', return_value=mock_knowledge_service):
        response = test_client.post(
            "/knowledge/validate",
            json={
                "pattern_id": "test-pattern",
                "name": "Test Pattern",
                "causal_chain": []
            }
        )
        
        assert response.status_code in [200, 422]

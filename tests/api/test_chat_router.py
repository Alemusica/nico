"""
Tests for Chat Router
====================
Test chat and WebSocket endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock


def test_chat_endpoint_success(test_client, mock_llm_service):
    """Test chat endpoint with LLM available."""
    mock_llm_service.check_availability = AsyncMock(return_value=True)
    mock_llm_service.chat = AsyncMock(return_value="Test response")
    
    with patch('api.routers.chat_router.get_llm_service', return_value=mock_llm_service):
        response = test_client.post(
            "/chat",
            json={"message": "What is causal discovery?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data


def test_chat_endpoint_llm_unavailable(test_client, mock_llm_service):
    """Test chat endpoint when LLM is unavailable."""
    mock_llm_service.check_availability = AsyncMock(return_value=False)
    
    with patch('api.routers.chat_router.get_llm_service', return_value=mock_llm_service):
        response = test_client.post(
            "/chat",
            json={"message": "Test message"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "LLM not available" in data["response"] or "error" in data["response"].lower()


def test_chat_endpoint_with_context(test_client, mock_llm_service):
    """Test chat with context."""
    mock_llm_service.check_availability = AsyncMock(return_value=True)
    mock_llm_service.chat = AsyncMock(return_value="Contextual response")
    
    with patch('api.routers.chat_router.get_llm_service', return_value=mock_llm_service):
        response = test_client.post(
            "/chat",
            json={
                "message": "Explain this dataset",
                "context": {"dataset": "test.csv", "variables": ["temp", "precip"]}
            }
        )
        
        assert response.status_code == 200


def test_websocket_chat_connection():
    """Test WebSocket chat connection."""
    # WebSocket testing requires special handling
    # This is a placeholder for WebSocket tests
    pass

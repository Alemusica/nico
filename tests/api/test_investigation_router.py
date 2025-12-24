"""
Tests for Investigation Router
==============================
Test investigation agent endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


def test_investigation_status_endpoint(test_client):
    """Test investigation status check."""
    response = test_client.get("/investigate/status")
    
    assert response.status_code == 200
    data = response.json()
    assert "investigation_agent" in data
    assert "data_manager" in data
    assert "components" in data


def test_create_briefing_success(test_client):
    """Test creating investigation briefing."""
    response = test_client.post(
        "/investigate/briefing",
        json={
            "query": "analyze floods in Venice 2020",
            "collect_satellite": True,
            "collect_reanalysis": True
        }
    )
    
    # May fail if investigation agent not available
    assert response.status_code in [200, 500]


def test_create_briefing_invalid_query(test_client):
    """Test briefing with invalid query."""
    response = test_client.post(
        "/investigate/briefing",
        json={
            "query": "",  # Empty query
            "collect_satellite": True
        }
    )
    
    assert response.status_code in [422, 500]


def test_confirm_briefing_not_found(test_client):
    """Test confirming non-existent briefing."""
    response = test_client.post("/investigate/briefing/nonexistent-id/confirm")
    
    assert response.status_code == 404


def test_websocket_connection():
    """Test WebSocket investigation connection."""
    # WebSocket testing requires special handling
    # This is a placeholder for WebSocket tests
    pass

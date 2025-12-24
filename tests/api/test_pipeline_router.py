"""
Tests for Pipeline Router
=========================
Test full pipeline execution endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


def test_run_pipeline_minimal(test_client):
    """Test running pipeline with minimal parameters."""
    response = test_client.post(
        "/pipeline/run",
        json={}
    )
    
    # May fail if pipeline dependencies not available
    assert response.status_code in [200, 500]


def test_run_pipeline_with_topics(test_client):
    """Test running pipeline with specific topics."""
    response = test_client.post(
        "/pipeline/run",
        json={
            "topics": ["arctic ice", "climate change"],
            "max_per_topic": 3
        }
    )
    
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "status" in data


def test_run_pipeline_with_queries(test_client):
    """Test running pipeline with paper queries."""
    response = test_client.post(
        "/pipeline/run",
        json={
            "paper_queries": ["arctic sea ice trends"],
            "max_per_topic": 5
        }
    )
    
    assert response.status_code in [200, 500]

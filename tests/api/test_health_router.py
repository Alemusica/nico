"""
Tests for Health Router
=======================
Test health check endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(test_client):
    """Test root health check endpoint."""
    response = test_client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Causal Discovery API"
    assert "version" in data


def test_health_endpoint(test_client):
    """Test detailed health check endpoint."""
    response = test_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "components" in data
    assert "llm" in data["components"]
    assert "causal_discovery" in data["components"]
    assert "databases" in data["components"]
    assert data["robustness"] == "All components have fallbacks - system operational"


def test_health_components_structure(test_client):
    """Test health endpoint returns correct component structure."""
    response = test_client.get("/health")
    data = response.json()
    
    # Check LLM component
    llm = data["components"]["llm"]
    assert "status" in llm
    assert llm["status"] in ["available", "fallback (rules)"]
    
    # Check causal discovery component
    causal = data["components"]["causal_discovery"]
    assert "status" in causal
    assert "method" in causal
    assert causal["method"] in ["PCMCI", "cross-correlation"]
    
    # Check databases
    dbs = data["components"]["databases"]
    assert "neo4j" in dbs
    assert "surrealdb" in dbs

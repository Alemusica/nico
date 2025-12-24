"""
Tests for Analysis Router
=========================
Test root cause analysis and causal discovery endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import pandas as pd


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'temperature': [20, 21, 22, 23, 24, 25, 24, 23, 22, 21],
        'precipitation': [0, 5, 10, 15, 20, 25, 20, 15, 10, 5]
    })


def test_ishikawa_endpoint_no_dataset(test_client, mock_data_service):
    """Test Ishikawa diagram with non-existent dataset."""
    mock_data_service.get_dataset = Mock(return_value=None)
    
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service):
        response = test_client.post(
            "/analysis/ishikawa",
            json={"dataset_name": "nonexistent.csv", "target": "flood"}
        )
        
        assert response.status_code == 404


def test_ishikawa_endpoint_success(test_client, mock_data_service, sample_dataset):
    """Test Ishikawa diagram generation."""
    mock_data_service.get_dataset = Mock(return_value=sample_dataset)
    
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service):
        response = test_client.post(
            "/analysis/ishikawa",
            json={"dataset_name": "test.csv", "target": "flood"}
        )
        
        # Should return diagram structure
        assert response.status_code == 200
        data = response.json()
        assert "target" in data
        assert "categories" in data


def test_fmea_endpoint(test_client, mock_data_service, sample_dataset):
    """Test FMEA analysis endpoint."""
    mock_data_service.get_dataset = Mock(return_value=sample_dataset)
    
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service):
        response = test_client.post(
            "/analysis/fmea",
            json={"dataset_name": "test.csv"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "failure_modes" in data


def test_five_why_endpoint(test_client, mock_data_service):
    """Test 5-Why analysis endpoint."""
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service):
        response = test_client.post(
            "/analysis/5why",
            json={"dataset_name": "test.csv", "observation": "High flood risk"}
        )
        
        assert response.status_code in [200, 404]


def test_hypotheses_generation_no_dataset(test_client, mock_data_service):
    """Test hypothesis generation with missing dataset."""
    mock_data_service.get_dataset = Mock(return_value=None)
    
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service):
        response = test_client.post(
            "/analysis/hypotheses",
            params={"dataset_name": "nonexistent.csv"}
        )
        
        assert response.status_code == 404


def test_hypotheses_generation_llm_unavailable(test_client, mock_data_service, mock_llm_service, sample_dataset):
    """Test hypothesis generation when LLM is unavailable."""
    mock_data_service.get_dataset = Mock(return_value=sample_dataset)
    mock_llm_service.check_availability = Mock(return_value=False)
    
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service), \
         patch('api.routers.analysis_router.get_llm_service', return_value=mock_llm_service):
        
        response = test_client.post(
            "/analysis/hypotheses",
            params={"dataset_name": "test.csv"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["hypotheses"] == []
        assert "error" in data


def test_discover_endpoint_no_dataset(test_client, mock_data_service):
    """Test causal discovery with missing dataset."""
    mock_data_service.get_dataset = Mock(return_value=None)
    
    with patch('api.routers.analysis_router.get_data_service', return_value=mock_data_service):
        response = test_client.post(
            "/analysis/discover",
            json={"dataset_name": "nonexistent.csv"}
        )
        
        assert response.status_code == 404

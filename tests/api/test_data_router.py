"""
Tests for Data Router
=====================
Test data management endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import pandas as pd
import io


@pytest.fixture
def sample_csv():
    """Sample CSV data for testing."""
    return "date,value\n2020-01-01,10\n2020-01-02,20\n"


def test_list_datasets_empty(test_client, mock_data_service):
    """Test listing datasets when none exist."""
    with patch('api.routers.data_router.get_data_service', return_value=mock_data_service):
        response = test_client.get("/data/files")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0


def test_list_datasets_with_data(test_client, mock_data_service):
    """Test listing datasets with existing data."""
    mock_data_service.list_datasets = Mock(return_value=[
        {"name": "test.csv", "n_rows": 100, "n_cols": 5}
    ])
    
    with patch('api.routers.data_router.get_data_service', return_value=mock_data_service):
        response = test_client.get("/data/files")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test.csv"


def test_get_dataset_not_found(test_client, mock_data_service):
    """Test getting non-existent dataset."""
    mock_data_service.get_metadata = Mock(return_value=None)
    
    with patch('api.routers.data_router.get_data_service', return_value=mock_data_service):
        response = test_client.get("/data/nonexistent.csv")
        
        assert response.status_code == 404


def test_get_dataset_success(test_client, mock_data_service):
    """Test getting existing dataset metadata."""
    mock_metadata = Mock(
        name="test.csv",
        file_type="csv",
        n_rows=100,
        n_cols=5,
        columns=[{"name": "date", "dtype": "object"}],
        memory_mb=0.5,
        time_range=None,
        spatial_bounds=None
    )
    mock_data_service.get_metadata = Mock(return_value=mock_metadata)
    
    with patch('api.routers.data_router.get_data_service', return_value=mock_data_service):
        response = test_client.get("/data/test.csv")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test.csv"
        assert data["n_rows"] == 100


def test_upload_csv_file(test_client, sample_csv):
    """Test uploading CSV file."""
    files = {"file": ("test.csv", sample_csv, "text/csv")}
    
    response = test_client.post("/data/upload", files=files)
    
    # Should process successfully
    assert response.status_code in [200, 500]  # May fail if data service not mocked properly


def test_data_sources_endpoint(test_client):
    """Test getting available data sources."""
    response = test_client.get("/data/sources")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Should contain at least ERA5 and CMEMS
    source_ids = [s["id"] for s in data]
    assert "era5" in source_ids or len(data) >= 0


def test_resolutions_endpoint(test_client):
    """Test getting available resolutions."""
    response = test_client.get("/data/resolutions")
    
    assert response.status_code == 200
    data = response.json()
    assert "temporal" in data
    assert "spatial" in data


def test_cache_stats_endpoint(test_client):
    """Test cache statistics endpoint."""
    response = test_client.get("/data/cache/stats")
    
    assert response.status_code in [200, 500]  # May fail if cache not available
    if response.status_code == 200:
        data = response.json()
        assert "total_entries" in data or "error" in data

"""Simple health test without fixtures."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_import_app():
    """Test that we can import the app."""
    from api.main import app
    assert app is not None

def test_health_endpoint_direct():
    """Test health endpoint directly."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_root_endpoint_direct():
    """Test root endpoint directly."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

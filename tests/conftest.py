"""
Pytest Configuration
====================
Shared fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require real data files"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def slcci_data_dir(data_dir):
    """Return SLCCI data directory."""
    return data_dir / "slcci"

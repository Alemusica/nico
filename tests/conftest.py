"""
Pytest Configuration
====================
Shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path

# CRITICAL: Add project root to path BEFORE pytest collects tests
project_root = Path(__file__).parent.parent.absolute()

# Remove tests directory from sys.path if present (pytest adds it)
tests_dir = str(project_root / "tests")
if tests_dir in sys.path:
    sys.path.remove(tests_dir)

# Ensure project root is at position 0
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

# Also ensure we're in project root
os.chdir(project_root)


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

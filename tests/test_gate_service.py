"""
Tests for Gate Service
======================
Tests for src/services/gate_service.py
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


class TestGateService:
    """Tests for GateService."""
    
    @pytest.fixture
    def gate_service(self):
        """Create GateService instance."""
        from src.services.gate_service import GateService
        return GateService()
    
    def test_service_initialization(self, gate_service):
        """Test service initializes correctly."""
        assert gate_service is not None
        assert gate_service.catalog is not None
    
    def test_list_gates(self, gate_service):
        """Test listing all gates."""
        gates = gate_service.list_gates()
        
        assert isinstance(gates, list)
        # Should have gates defined in config
        assert len(gates) >= 0
    
    def test_list_regions(self, gate_service):
        """Test listing regions."""
        regions = gate_service.get_regions()
        
        assert isinstance(regions, list)
    
    def test_select_existing_gate(self, gate_service):
        """Test selecting a gate that exists."""
        gates = gate_service.list_gates()
        
        if gates:
            gate_id = gates[0].id
            gate = gate_service.select_gate(gate_id)
            
            assert gate is not None
            assert gate.id == gate_id
    
    def test_select_nonexistent_gate(self, gate_service):
        """Test selecting a gate that doesn't exist."""
        gate = gate_service.select_gate("nonexistent_gate_xyz")
        assert gate is None
    
    def test_gate_exists(self, gate_service):
        """Test gate_exists method."""
        gates = gate_service.list_gates()
        
        if gates:
            assert gate_service.gate_exists(gates[0].id) is True
        
        assert gate_service.gate_exists("nonexistent") is False
    
    def test_get_closest_passes(self, gate_service):
        """Test getting closest satellite passes."""
        gates = gate_service.list_gates()
        
        if gates:
            gate_id = gates[0].id
            passes = gate_service.get_closest_passes(gate_id, n=3)
            
            assert isinstance(passes, list)
            assert len(passes) <= 3
    
    def test_get_primary_pass(self, gate_service):
        """Test getting primary pass."""
        gates = gate_service.list_gates()
        
        if gates:
            gate_id = gates[0].id
            primary = gate_service.get_primary_pass(gate_id)
            
            # May be None if no passes defined
            if primary is not None:
                assert isinstance(primary, int)
    
    def test_clear_cache(self, gate_service):
        """Test cache clearing."""
        # Should not raise
        gate_service.clear_cache()
        gate_service.clear_cache("some_gate")


class TestGateServiceBBox:
    """Tests for GateService bounding box methods."""
    
    @pytest.fixture
    def gate_service(self):
        """Create GateService instance."""
        from src.services.gate_service import GateService
        return GateService()
    
    def test_get_bbox_returns_model(self, gate_service):
        """Test that get_bbox returns a BoundingBox."""
        gates = gate_service.list_gates()
        
        if gates:
            bbox = gate_service.get_bbox(gates[0].id, buffer_km=50)
            
            if bbox is not None:
                from src.core.models import BoundingBox
                assert isinstance(bbox, BoundingBox)
                assert bbox.lat_min <= bbox.lat_max
    
    def test_get_gate_info(self, gate_service):
        """Test getting complete gate info."""
        gates = gate_service.list_gates()
        
        if gates:
            info = gate_service.get_gate_info(gates[0].id)
            
            if info is not None:
                assert "gate" in info
                assert "closest_passes" in info


class TestGateCatalog:
    """Tests for GateCatalog."""
    
    def test_catalog_initialization(self):
        """Test catalog loads from config."""
        from src.gates.catalog import GateCatalog
        
        catalog = GateCatalog()
        assert catalog is not None
    
    def test_catalog_list_all(self):
        """Test listing all gates from catalog."""
        from src.gates.catalog import GateCatalog
        
        catalog = GateCatalog()
        gates = catalog.list_all()
        
        assert isinstance(gates, list)
    
    def test_catalog_by_region(self):
        """Test filtering by region."""
        from src.gates.catalog import GateCatalog
        
        catalog = GateCatalog()
        regions = catalog.get_regions()
        
        if regions:
            gates = catalog.by_region(regions[0])
            assert isinstance(gates, list)

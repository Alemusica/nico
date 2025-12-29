"""
Tests for Core Models
=====================
Tests for src/core/models.py
"""

import pytest
from datetime import datetime, timedelta


class TestBoundingBox:
    """Tests for BoundingBox model."""
    
    def test_valid_bbox(self):
        """Test valid bounding box creation."""
        from src.core.models import BoundingBox
        
        bbox = BoundingBox(
            lat_min=60.0,
            lat_max=85.0,
            lon_min=-20.0,
            lon_max=20.0
        )
        
        assert bbox.lat_min == 60.0
        assert bbox.lat_max == 85.0
        assert bbox.lon_min == -20.0
        assert bbox.lon_max == 20.0
    
    def test_invalid_lat_range(self):
        """Test that lat_min > lat_max raises error."""
        from src.core.models import BoundingBox
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            BoundingBox(
                lat_min=85.0,  # Greater than max
                lat_max=60.0,
                lon_min=-20.0,
                lon_max=20.0
            )
    
    def test_invalid_lon_range(self):
        """Test that lon_min > lon_max raises error (except for antimeridian)."""
        from src.core.models import BoundingBox
        from pydantic import ValidationError
        
        # This should raise because it's not spanning antimeridian
        with pytest.raises(ValidationError):
            BoundingBox(
                lat_min=60.0,
                lat_max=85.0,
                lon_min=20.0,  # Greater than max
                lon_max=-20.0
            )
    
    def test_lat_bounds(self):
        """Test latitude is within -90 to 90."""
        from src.core.models import BoundingBox
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            BoundingBox(
                lat_min=-100.0,  # Invalid
                lat_max=85.0,
                lon_min=-20.0,
                lon_max=20.0
            )
    
    def test_center_property(self):
        """Test center calculation."""
        from src.core.models import BoundingBox
        
        bbox = BoundingBox(
            lat_min=60.0,
            lat_max=80.0,
            lon_min=-20.0,
            lon_max=20.0
        )
        
        center = bbox.center
        assert center[0] == 70.0  # lat
        assert center[1] == 0.0   # lon


class TestTimeRange:
    """Tests for TimeRange model."""
    
    def test_valid_time_range(self):
        """Test valid time range creation."""
        from src.core.models import TimeRange
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        tr = TimeRange(start=start, end=end)
        
        assert tr.start == start
        assert tr.end == end
    
    def test_invalid_time_range(self):
        """Test that start > end raises error."""
        from src.core.models import TimeRange
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            TimeRange(
                start=datetime(2024, 12, 31),  # After end
                end=datetime(2024, 1, 1)
            )
    
    def test_days_property(self):
        """Test duration in days."""
        from src.core.models import TimeRange
        
        tr = TimeRange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 11)
        )
        
        assert tr.days == 10


class TestGateModel:
    """Tests for GateModel."""
    
    def test_gate_creation(self):
        """Test gate model creation."""
        from src.core.models import GateModel
        
        gate = GateModel(
            id="fram_strait",
            name="Fram Strait",
            file="fram_strait.shp",
            region="Atlantic Sector",
            lat_min=76.0,
            lat_max=82.0,
            lon_min=-20.0,
            lon_max=15.0
        )
        
        assert gate.id == "fram_strait"
        assert gate.name == "Fram Strait"
        assert gate.region == "Atlantic Sector"
    
    def test_gate_with_passes(self):
        """Test gate with closest passes."""
        from src.core.models import GateModel
        
        gate = GateModel(
            id="fram_strait",
            name="Fram Strait",
            file="fram_strait.shp",
            closest_passes=[481, 360, 239]
        )
        
        assert gate.closest_passes == [481, 360, 239]
    
    def test_gate_with_datasets(self):
        """Test gate with recommended datasets."""
        from src.core.models import GateModel
        
        gate = GateModel(
            id="fram_strait",
            name="Fram Strait",
            file="fram_strait.shp",
            datasets=["SLCCI", "ERA5", "CMEMS-SST"],
            default_buffer_km=75.0
        )
        
        assert gate.datasets == ["SLCCI", "ERA5", "CMEMS-SST"]
        assert gate.default_buffer_km == 75.0
    
    def test_gate_bbox_from_ranges(self):
        """Test bbox computed from latitude_range and longitude_range."""
        from src.core.models import GateModel
        
        gate = GateModel(
            id="fram_strait",
            name="Fram Strait",
            file="fram_strait.shp",
            latitude_range=[78.0, 80.0],
            longitude_range=[-20.0, 10.0]
        )
        
        bbox = gate.bbox
        assert bbox is not None
        assert bbox.lat_min == 78.0
        assert bbox.lat_max == 80.0
        assert bbox.lon_min == -20.0
        assert bbox.lon_max == 10.0
    
    def test_gate_no_bbox_without_ranges(self):
        """Test bbox is None when no ranges provided."""
        from src.core.models import GateModel
        
        gate = GateModel(
            id="fram_strait",
            name="Fram Strait",
            file="fram_strait.shp"
        )
        
        # Without lat/lon ranges, bbox should be None
        # (unless lat_min/max/lon_min/max are set)
        assert gate.bbox is None


class TestDataRequest:
    """Tests for DataRequest model."""
    
    def test_data_request(self):
        """Test data request creation."""
        from src.core.models import DataRequest, BoundingBox, TimeRange
        
        bbox = BoundingBox(
            lat_min=60.0, lat_max=85.0,
            lon_min=-20.0, lon_max=20.0
        )
        
        time_range = TimeRange(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31)
        )
        
        request = DataRequest(
            bbox=bbox,
            time_range=time_range,
            variables=["sla", "adt"],
            dataset_id="cmems_sla"
        )
        
        assert request.variables == ["sla", "adt"]
        assert request.dataset_id == "cmems_sla"


class TestResolution:
    """Tests for resolution enums."""
    
    def test_temporal_resolution(self):
        """Test temporal resolution enum."""
        from src.core.models import TemporalResolution
        
        assert TemporalResolution.DAILY.value == "daily"
        assert TemporalResolution.MONTHLY.value == "monthly"
    
    def test_spatial_resolution(self):
        """Test spatial resolution enum as float."""
        from src.core.models import SpatialResolution
        
        # Values are floats for direct use in binning
        assert SpatialResolution.MEDIUM.value == 0.25
        assert SpatialResolution.LOW.value == 0.5
        assert SpatialResolution.HIGH.value == 0.1
        assert SpatialResolution.COARSE.value == 1.0
        
        # Test from_degrees helper
        res = SpatialResolution.from_degrees(0.3)
        assert res == SpatialResolution.MEDIUM
        
        res = SpatialResolution.from_degrees(0.08)
        assert res == SpatialResolution.HIGH
    
    def test_spatial_resolution_list_all(self):
        """Test spatial resolution list_all method."""
        from src.core.models import SpatialResolution
        
        all_res = SpatialResolution.list_all()
        assert len(all_res) == 4
        assert all_res[0]["name"] == "HIGH"
        assert all_res[1]["degrees"] == 0.25

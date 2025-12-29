"""
Unit Tests for Core Models
==========================
Tests for src/core/models.py
"""

import pytest
from datetime import datetime

from src.core.models import (
    BoundingBox,
    TimeRange,
    GateModel,
    DataRequest,
    ResolutionConfig,
    TemporalResolution,
    SpatialResolution,
    DataSource,
    bbox_to_legacy_format,
    legacy_to_bbox,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""
    
    def test_valid_bbox(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        assert bbox.lat_min == 78.0
        assert bbox.lat_max == 80.0
        assert bbox.lon_min == -20.0
        assert bbox.lon_max == 10.0
    
    def test_lat_range_property(self):
        """Test lat_range property."""
        bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        assert bbox.lat_range == (78.0, 80.0)
    
    def test_lon_range_property(self):
        """Test lon_range property."""
        bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        assert bbox.lon_range == (-20.0, 10.0)
    
    def test_as_tuple(self):
        """Test as_tuple property."""
        bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        assert bbox.as_tuple == (78.0, 80.0, -20.0, 10.0)
    
    def test_as_list(self):
        """Test as_list property."""
        bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        assert bbox.as_list == [78.0, 80.0, -20.0, 10.0]
    
    def test_from_tuple(self):
        """Test creating from tuple."""
        bbox = BoundingBox.from_tuple((78.0, 80.0, -20.0, 10.0))
        assert bbox.lat_min == 78.0
        assert bbox.lon_max == 10.0
    
    def test_from_list(self):
        """Test creating from list."""
        bbox = BoundingBox.from_list([78.0, 80.0, -20.0, 10.0])
        assert bbox.lat_min == 78.0
        assert bbox.lon_max == 10.0
    
    def test_crosses_dateline(self):
        """Test dateline crossing detection."""
        # Does not cross dateline
        bbox1 = BoundingBox(lat_min=65.0, lat_max=66.0, lon_min=-170.0, lon_max=-168.0)
        assert not bbox1.crosses_dateline
        
        # Crosses dateline (lon_min > lon_max)
        bbox2 = BoundingBox(lat_min=65.0, lat_max=66.0, lon_min=170.0, lon_max=-170.0)
        assert bbox2.crosses_dateline
    
    def test_invalid_lat_range(self):
        """Test that lat_min > lat_max raises error."""
        with pytest.raises(ValueError, match="lat_min.*must be <= lat_max"):
            BoundingBox(lat_min=80.0, lat_max=78.0, lon_min=-20.0, lon_max=10.0)
    
    def test_invalid_lat_value(self):
        """Test that latitude out of range raises error."""
        with pytest.raises(ValueError):
            BoundingBox(lat_min=100.0, lat_max=110.0, lon_min=-20.0, lon_max=10.0)
    
    def test_invalid_lon_value(self):
        """Test that longitude out of range raises error."""
        with pytest.raises(ValueError):
            BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-200.0, lon_max=10.0)


class TestTimeRange:
    """Tests for TimeRange model."""
    
    def test_valid_time_range(self):
        """Test creating a valid time range."""
        tr = TimeRange(start="2024-01-01", end="2024-12-31")
        assert tr.start == "2024-01-01"
        assert tr.end == "2024-12-31"
    
    def test_start_date_property(self):
        """Test start_date property."""
        tr = TimeRange(start="2024-01-01", end="2024-12-31")
        assert tr.start_date == datetime(2024, 1, 1)
    
    def test_end_date_property(self):
        """Test end_date property."""
        tr = TimeRange(start="2024-01-01", end="2024-12-31")
        assert tr.end_date == datetime(2024, 12, 31)
    
    def test_days_property(self):
        """Test days property."""
        tr = TimeRange(start="2024-01-01", end="2024-01-31")
        assert tr.days == 30
    
    def test_invalid_date_format(self):
        """Test that invalid date format raises error."""
        with pytest.raises(ValueError, match="Invalid date format"):
            TimeRange(start="01-01-2024", end="2024-12-31")
    
    def test_invalid_range(self):
        """Test that start > end raises error."""
        with pytest.raises(ValueError, match="start.*must be <= end"):
            TimeRange(start="2024-12-31", end="2024-01-01")


class TestGateModel:
    """Tests for GateModel."""
    
    def test_valid_gate(self):
        """Test creating a valid gate."""
        gate = GateModel(
            id="fram_strait",
            name="🧊 Fram Strait",
            file="fram_strait_S3_pass_481.shp",
            description="Main Arctic-Atlantic exchange",
            region="Atlantic Sector",
            closest_passes=[481, 254, 127]
        )
        assert gate.id == "fram_strait"
        assert gate.name == "🧊 Fram Strait"
        assert gate.closest_passes == [481, 254, 127]
    
    def test_gate_without_passes(self):
        """Test gate without closest_passes."""
        gate = GateModel(
            id="test_gate",
            name="Test Gate",
            file="test.shp",
            description="Test",
            region="Atlantic Sector"
        )
        assert gate.closest_passes is None


class TestDataRequest:
    """Tests for DataRequest model."""
    
    def test_valid_request(self):
        """Test creating a valid data request."""
        request = DataRequest(
            bbox=BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0),
            time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
            variables=["sla", "adt"],
            gate_id="fram_strait",
            source=DataSource.CMEMS
        )
        assert request.gate_id == "fram_strait"
        assert request.source == DataSource.CMEMS
        assert len(request.variables) == 2
    
    def test_request_with_resolution(self):
        """Test request with custom resolution."""
        request = DataRequest(
            bbox=BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0),
            time_range=TimeRange(start="2024-01-01", end="2024-12-31"),
            resolution=ResolutionConfig(
                temporal=TemporalResolution.HOURLY,
                spatial=SpatialResolution.HIGH
            )
        )
        assert request.resolution.temporal == TemporalResolution.HOURLY
        assert request.resolution.spatial == SpatialResolution.HIGH


class TestLegacyCompatibility:
    """Tests for legacy compatibility functions."""
    
    def test_bbox_to_legacy_format(self):
        """Test converting bbox to legacy format."""
        bbox = BoundingBox(lat_min=78.0, lat_max=80.0, lon_min=-20.0, lon_max=10.0)
        legacy = bbox_to_legacy_format(bbox)
        assert legacy["lat_range"] == (78.0, 80.0)
        assert legacy["lon_range"] == (-20.0, 10.0)
    
    def test_legacy_to_bbox(self):
        """Test converting legacy format to bbox."""
        bbox = legacy_to_bbox(lat_range=(78.0, 80.0), lon_range=(-20.0, 10.0))
        assert bbox.lat_min == 78.0
        assert bbox.lat_max == 80.0
        assert bbox.lon_min == -20.0
        assert bbox.lon_max == 10.0


class TestEnums:
    """Tests for enum types."""
    
    def test_temporal_resolution_values(self):
        """Test temporal resolution enum values."""
        assert TemporalResolution.DAILY.value == "daily"
        assert TemporalResolution.HOURLY.value == "hourly"
    
    def test_spatial_resolution_values(self):
        """Test spatial resolution enum values."""
        assert SpatialResolution.HIGH.value == "0.1"
        assert SpatialResolution.MEDIUM.value == "0.25"
    
    def test_data_source_values(self):
        """Test data source enum values."""
        assert DataSource.CMEMS.value == "cmems"
        assert DataSource.ERA5.value == "era5"

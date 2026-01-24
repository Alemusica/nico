"""
Base Data Client Interface - The "Parking Spot" Contract.

All data clients MUST implement this interface to ensure:
- Consistent method signatures
- Predictable return types
- Unified error handling
- Standard caching behavior

Created: 2026-01-19
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import pandas as pd
import xarray as xr


class DataFormat(Enum):
    """Standard output formats for data clients."""
    XARRAY = "xarray"      # For gridded spatial-temporal data
    DATAFRAME = "dataframe"  # For time series / tabular data
    POINTS = "points"      # For point observations (converted to DataFrame)


class ClientStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Partial functionality
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Geographic bounding box (WGS84)."""
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    def __post_init__(self):
        if not (-180 <= self.lon_min <= 180 and -180 <= self.lon_max <= 180):
            raise ValueError(f"Longitude must be in [-180, 180]: {self.lon_min}, {self.lon_max}")
        if not (-90 <= self.lat_min <= 90 and -90 <= self.lat_max <= 90):
            raise ValueError(f"Latitude must be in [-90, 90]: {self.lat_min}, {self.lat_max}")

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (lon_min, lat_min, lon_max, lat_max)."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> "BoundingBox":
        """Create from tuple (lon_min, lat_min, lon_max, lat_max)."""
        return cls(lon_min=t[0], lat_min=t[1], lon_max=t[2], lat_max=t[3])


@dataclass
class TimeRange:
    """Time range for data requests."""
    start: datetime
    end: datetime

    def __post_init__(self):
        if isinstance(self.start, str):
            self.start = datetime.fromisoformat(self.start)
        if isinstance(self.end, str):
            self.end = datetime.fromisoformat(self.end)
        if self.start > self.end:
            raise ValueError(f"Start must be before end: {self.start} > {self.end}")

    @classmethod
    def from_strings(cls, start: str, end: str) -> "TimeRange":
        """Create from ISO format strings."""
        return cls(
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end)
        )


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: ClientStatus
    message: str
    latency_ms: Optional[float] = None
    last_check: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


@dataclass
class DataClientError(Exception):
    """
    Unified exception for all data client errors.

    Attributes:
        source_id: ID of the data source that failed
        operation: What operation was attempted
        original_error: The underlying exception
        fallback_available: Whether synthetic fallback data is available
        message: Human-readable error description
    """
    source_id: str
    operation: str
    original_error: Optional[Exception] = None
    fallback_available: bool = False
    message: str = ""

    def __str__(self):
        base = f"[{self.source_id}] {self.operation} failed"
        if self.message:
            base += f": {self.message}"
        if self.original_error:
            base += f" (caused by: {type(self.original_error).__name__}: {self.original_error})"
        if self.fallback_available:
            base += " [fallback available]"
        return base


class DataClient(ABC):
    """
    Abstract base class for all data clients.

    This is the "parking spot" contract - every data client ("car")
    must fit this interface to park in the garage.

    Subclasses must implement:
    - source_id: Unique identifier matching api_registry.py
    - output_format: Whether this client returns xarray or DataFrame
    - download(): Fetch data for given parameters
    - list_products(): List available products/variables
    - health_check(): Verify API availability

    Optional overrides:
    - generate_synthetic(): Create realistic fallback data
    - get_cache_key(): Custom cache key generation
    """

    # =========================================================================
    # REQUIRED PROPERTIES
    # =========================================================================

    @property
    @abstractmethod
    def source_id(self) -> str:
        """
        Unique identifier for this data source.

        Must match an entry in api_registry.py.
        Example: "CMEMS_SEALEVEL", "ERA5_SURFACE", "GPM_IMERG"
        """
        pass

    @property
    @abstractmethod
    def output_format(self) -> DataFormat:
        """
        The standard output format for this client.

        - XARRAY: For gridded data (CMEMS, ERA5, GPM, etc.)
        - DATAFRAME: For time series (Climate Indices, etc.)
        - POINTS: For point observations (Tide Gauges, ARGO, Aircraft)
                  Note: POINTS are converted to DataFrame internally
        """
        pass

    # =========================================================================
    # REQUIRED METHODS
    # =========================================================================

    @abstractmethod
    async def download(
        self,
        variables: list[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Download data for the specified parameters.

        This is the main entry point for data retrieval. Implementations
        should:
        1. Try to fetch real data from the API
        2. On failure, call generate_synthetic() if available
        3. Raise DataClientError if both fail

        Args:
            variables: List of variable names to download
            bbox: Geographic bounding box
            time_range: Start and end time
            **kwargs: Client-specific parameters (e.g., product type)

        Returns:
            xr.Dataset for gridded data (output_format=XARRAY)
            pd.DataFrame for time series (output_format=DATAFRAME/POINTS)

        Raises:
            DataClientError: If download fails and no fallback available
        """
        pass

    @abstractmethod
    def list_products(self) -> dict[str, str]:
        """
        List available products/variables from this data source.

        Returns:
            Dictionary mapping product/variable IDs to descriptions.
            Example: {"sla": "Sea Level Anomaly", "adt": "Absolute Dynamic Topography"}
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Check if the data source API is available.

        Should perform a minimal request to verify connectivity
        without downloading significant data.

        Returns:
            HealthCheckResult with status and diagnostics
        """
        pass

    # =========================================================================
    # OPTIONAL METHODS (with default implementations)
    # =========================================================================

    async def generate_synthetic(
        self,
        variables: list[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Generate realistic synthetic/fallback data.

        Called automatically when download() fails.
        Override this to provide domain-specific synthetic data.

        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.source_id} does not implement synthetic data generation"
        )

    def get_cache_key(
        self,
        variables: list[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> str:
        """
        Generate a cache key for the given request.

        Override for custom caching behavior.
        Default uses MD5 of parameters.
        """
        import hashlib
        import json

        params = {
            "source": self.source_id,
            "variables": sorted(variables),
            "bbox": bbox.to_tuple(),
            "time_start": time_range.start.isoformat(),
            "time_end": time_range.end.isoformat(),
            **kwargs
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def validate_variables(self, variables: list[str]) -> list[str]:
        """
        Validate that requested variables are available.

        Returns list of valid variables, logs warnings for invalid ones.
        """
        available = set(self.list_products().keys())
        valid = []
        for var in variables:
            if var in available:
                valid.append(var)
            else:
                import logging
                logging.warning(
                    f"[{self.source_id}] Variable '{var}' not available. "
                    f"Available: {sorted(available)}"
                )
        return valid

    def standardize_output(
        self,
        data: Any,
        variables: list[str],
        bbox: BoundingBox,
        time_range: TimeRange
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Ensure output conforms to standard format.

        Adds required attributes/metadata if missing.
        """
        if isinstance(data, xr.Dataset):
            # Ensure standard attributes
            data.attrs.setdefault("source", self.source_id)
            data.attrs.setdefault("created", datetime.now().isoformat())
            data.attrs.setdefault("bbox", str(bbox.to_tuple()))
            return data

        elif isinstance(data, pd.DataFrame):
            # Ensure DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                if "time" in data.columns:
                    data = data.set_index("time")
                elif "timestamp" in data.columns:
                    data = data.set_index("timestamp")
            return data

        else:
            raise TypeError(
                f"Unexpected data type from {self.source_id}: {type(data)}. "
                f"Expected xr.Dataset or pd.DataFrame."
            )


# =============================================================================
# CONVENIENCE MIXINS
# =============================================================================

class XArrayClientMixin:
    """Mixin for clients that output xarray.Dataset."""

    @property
    def output_format(self) -> DataFormat:
        return DataFormat.XARRAY

    def create_empty_dataset(
        self,
        variables: list[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        resolution: float = 0.25
    ) -> xr.Dataset:
        """Create an empty xarray Dataset with proper structure."""
        import numpy as np

        # Create coordinate arrays
        lons = np.arange(bbox.lon_min, bbox.lon_max + resolution, resolution)
        lats = np.arange(bbox.lat_min, bbox.lat_max + resolution, resolution)
        times = pd.date_range(time_range.start, time_range.end, freq="D")

        # Create empty data variables
        data_vars = {}
        for var in variables:
            data_vars[var] = (
                ["time", "latitude", "longitude"],
                np.full((len(times), len(lats), len(lons)), np.nan)
            )

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "source": getattr(self, "source_id", "unknown"),
                "synthetic": True,
            }
        )


class DataFrameClientMixin:
    """Mixin for clients that output pandas.DataFrame."""

    @property
    def output_format(self) -> DataFormat:
        return DataFormat.DATAFRAME

    def create_empty_dataframe(
        self,
        columns: list[str],
        time_range: TimeRange,
        freq: str = "D"
    ) -> pd.DataFrame:
        """Create an empty DataFrame with proper structure."""
        times = pd.date_range(time_range.start, time_range.end, freq=freq)
        return pd.DataFrame(index=times, columns=columns)


class PointDataClientMixin:
    """Mixin for clients that output point observations."""

    @property
    def output_format(self) -> DataFormat:
        return DataFormat.POINTS

    def points_to_dataframe(
        self,
        points: list[Any],
        time_field: str = "timestamp",
        value_fields: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Convert list of point dataclasses to DataFrame."""
        if not points:
            return pd.DataFrame()

        # Convert dataclasses to dicts
        from dataclasses import asdict, is_dataclass

        if is_dataclass(points[0]):
            records = [asdict(p) for p in points]
        elif isinstance(points[0], dict):
            records = points
        else:
            raise TypeError(f"Expected dataclass or dict, got {type(points[0])}")

        df = pd.DataFrame(records)

        # Set time index
        if time_field in df.columns:
            df[time_field] = pd.to_datetime(df[time_field])
            df = df.set_index(time_field)

        return df

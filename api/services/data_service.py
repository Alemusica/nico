"""
📊 Data Service
================
Dataset loading, preprocessing, and metadata extraction.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import io


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""
    name: str
    file_type: str
    n_rows: int
    n_cols: int
    columns: List[Dict[str, Any]]
    memory_mb: float
    time_range: Optional[Dict[str, str]] = None
    spatial_bounds: Optional[Dict[str, float]] = None


class DataService:
    """
    Service for loading and preprocessing various data formats.
    
    Supports:
    - CSV, Excel, Parquet
    - NetCDF (xarray)
    - JSON time series
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, DatasetMetadata] = {}
    
    def list_available_files(self) -> List[Dict[str, str]]:
        """List all data files in the data directory."""
        files = []
        if self.data_dir.exists():
            for ext in ['*.csv', '*.nc', '*.xlsx', '*.parquet', '*.json']:
                for f in self.data_dir.rglob(ext):
                    files.append({
                        "path": str(f.relative_to(self.data_dir)),
                        "name": f.name,
                        "type": f.suffix[1:],
                        "size_mb": f.stat().st_size / 1024 / 1024,
                    })
        return sorted(files, key=lambda x: x["path"])
    
    def _extract_column_info(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract detailed column information."""
        columns = []
        for col in df.columns:
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
            }
            
            # Add samples
            non_null = df[col].dropna()
            if len(non_null) > 0:
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    info["min"] = float(non_null.min())
                    info["max"] = float(non_null.max())
                    info["mean"] = float(non_null.mean())
                    info["samples"] = [float(x) for x in non_null.head(5).tolist()]
                else:
                    info["samples"] = [str(x) for x in non_null.head(5).tolist()]
            
            columns.append(info)
        
        return columns
    
    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the time column."""
        time_patterns = ['time', 'timestamp', 'date', 'datetime', 't', 'epoch']
        
        # Check column names
        for col in df.columns:
            if col.lower() in time_patterns:
                return col
        
        # Check datetime dtypes
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        # Try parsing columns as dates
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(10))
                    return col
                except:
                    pass
        
        return None
    
    def load_csv(
        self, 
        file_path: Union[str, Path, io.BytesIO],
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load CSV file."""
        if isinstance(file_path, (str, Path)):
            path = Path(file_path)
            if not path.is_absolute():
                path = self.data_dir / path
            df = pd.read_csv(path)
            name = name or path.stem
        else:
            df = pd.read_csv(file_path)
            name = name or "uploaded_csv"
        
        # Try to parse time column
        time_col = self._detect_time_column(df)
        if time_col:
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                pass
        
        self.datasets[name] = df
        self.metadata[name] = DatasetMetadata(
            name=name,
            file_type="csv",
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=self._extract_column_info(df),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            time_range=self._get_time_range(df, time_col),
        )
        
        return df
    
    def load_netcdf(
        self,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        flatten: bool = True,
    ) -> pd.DataFrame:
        """
        Load NetCDF file and convert to DataFrame.
        
        Args:
            file_path: Path to .nc file
            name: Dataset name
            flatten: If True, flatten spatial dimensions
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.data_dir / path
        
        ds = xr.open_dataset(path)
        name = name or path.stem
        
        # Convert to DataFrame
        if flatten:
            df = ds.to_dataframe().reset_index()
        else:
            # Stack spatial dimensions
            df = ds.to_dataframe()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
        
        # Extract metadata
        time_col = None
        for coord in ['time', 'Time', 'timestamp']:
            if coord in df.columns:
                time_col = coord
                break
        
        spatial_bounds = None
        for lat_name in ['lat', 'latitude', 'Latitude']:
            for lon_name in ['lon', 'longitude', 'Longitude']:
                if lat_name in df.columns and lon_name in df.columns:
                    spatial_bounds = {
                        "lat_min": float(df[lat_name].min()),
                        "lat_max": float(df[lat_name].max()),
                        "lon_min": float(df[lon_name].min()),
                        "lon_max": float(df[lon_name].max()),
                    }
                    break
        
        self.datasets[name] = df
        self.metadata[name] = DatasetMetadata(
            name=name,
            file_type="netcdf",
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=self._extract_column_info(df),
            memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
            time_range=self._get_time_range(df, time_col),
            spatial_bounds=spatial_bounds,
        )
        
        ds.close()
        return df
    
    def load_file(
        self,
        file_path: Union[str, Path],
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Auto-detect file type and load."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return self.load_csv(path, name)
        elif suffix == '.nc':
            return self.load_netcdf(path, name)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
            name = name or path.stem
            self.datasets[name] = df
            return df
        elif suffix == '.parquet':
            df = pd.read_parquet(path)
            name = name or path.stem
            self.datasets[name] = df
            return df
        elif suffix == '.json':
            df = pd.read_json(path)
            name = name or path.stem
            self.datasets[name] = df
            return df
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _get_time_range(
        self, 
        df: pd.DataFrame, 
        time_col: Optional[str]
    ) -> Optional[Dict[str, str]]:
        """Get time range if time column exists."""
        if time_col and time_col in df.columns:
            try:
                times = pd.to_datetime(df[time_col])
                return {
                    "start": str(times.min()),
                    "end": str(times.max()),
                    "column": time_col,
                }
            except:
                pass
        return None
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get loaded dataset by name."""
        return self.datasets.get(name)
    
    def get_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata."""
        return self.metadata.get(name)
    
    def get_sample_data(self, name: str, n_rows: int = 10) -> str:
        """Get sample data as formatted string."""
        df = self.datasets.get(name)
        if df is None:
            return ""
        return df.head(n_rows).to_string()
    
    def aggregate_by_time(
        self,
        name: str,
        time_col: str,
        freq: str = "D",  # Daily
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate dataset by time period.
        
        Args:
            name: Dataset name
            time_col: Time column
            freq: Frequency ('H', 'D', 'W', 'M')
            agg_func: Aggregation function ('mean', 'sum', 'max', 'min')
        """
        df = self.datasets[name].copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if agg_func == "mean":
            agg_df = df[numeric_cols].resample(freq).mean()
        elif agg_func == "sum":
            agg_df = df[numeric_cols].resample(freq).sum()
        elif agg_func == "max":
            agg_df = df[numeric_cols].resample(freq).max()
        elif agg_func == "min":
            agg_df = df[numeric_cols].resample(freq).min()
        else:
            agg_df = df[numeric_cols].resample(freq).mean()
        
        return agg_df.reset_index()


# Singleton
_data_service: Optional[DataService] = None

def get_data_service() -> DataService:
    """Get or create data service singleton."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service

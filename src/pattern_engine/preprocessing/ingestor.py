"""
Data ingestor for heterogeneous data sources.

Supports:
- CSV, JSON, Parquet, Excel files
- SQL databases
- REST APIs
- Real-time sensor streams
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from ..core.config import DataSourceConfig

logger = logging.getLogger(__name__)


@dataclass
class IngestedData:
    """Container for ingested data with metadata."""
    data: pd.DataFrame
    source_name: str
    source_type: str
    ingested_at: datetime
    row_count: int
    column_count: int
    schema: Dict[str, str]
    metadata: Dict[str, Any]
    
    @property
    def shape(self) -> tuple:
        return (self.row_count, self.column_count)


class DataIngestor:
    """
    Unified data ingestor for heterogeneous data sources.
    
    Example:
        ingestor = DataIngestor()
        
        # Ingest from CSV
        data = ingestor.ingest(DataSourceConfig(
            name="batch_data",
            source_type="csv",
            path_or_uri="data/batches.csv"
        ))
        
        # Ingest from multiple sources
        all_data = ingestor.ingest_multiple([config1, config2, config3])
    """
    
    # Supported file extensions and their handlers
    FILE_HANDLERS = {
        ".csv": "csv",
        ".tsv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".xlsx": "excel",
        ".xls": "excel",
        ".feather": "feather",
        ".pickle": "pickle",
        ".pkl": "pickle",
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cache: Dict[str, IngestedData] = {}
    
    def ingest(
        self,
        config: DataSourceConfig,
        use_cache: bool = True,
    ) -> IngestedData:
        """
        Ingest data from a single source.
        
        Args:
            config: Data source configuration
            use_cache: Whether to use cached data if available
            
        Returns:
            IngestedData object containing the DataFrame and metadata
        """
        cache_key = f"{config.name}:{config.path_or_uri}"
        
        if use_cache and cache_key in self._cache:
            if self.verbose:
                logger.info(f"Using cached data for {config.name}")
            return self._cache[cache_key]
        
        if self.verbose:
            logger.info(f"Ingesting data from {config.name} ({config.source_type})")
        
        # Dispatch to appropriate handler
        if config.source_type in ("csv", "tsv"):
            df = self._ingest_csv(config)
        elif config.source_type == "json":
            df = self._ingest_json(config)
        elif config.source_type == "jsonl":
            df = self._ingest_jsonl(config)
        elif config.source_type == "parquet":
            df = self._ingest_parquet(config)
        elif config.source_type == "excel":
            df = self._ingest_excel(config)
        elif config.source_type == "database":
            df = self._ingest_database(config)
        elif config.source_type == "api":
            df = self._ingest_api(config)
        elif config.source_type == "auto":
            df = self._ingest_auto(config)
        else:
            raise ValueError(f"Unsupported source type: {config.source_type}")
        
        # Infer schema
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Create result
        result = IngestedData(
            data=df,
            source_name=config.name,
            source_type=config.source_type,
            ingested_at=datetime.now(),
            row_count=len(df),
            column_count=len(df.columns),
            schema=schema,
            metadata=config.metadata,
        )
        
        if use_cache:
            self._cache[cache_key] = result
        
        if self.verbose:
            logger.info(f"Ingested {result.row_count} rows, {result.column_count} columns")
        
        return result
    
    def ingest_multiple(
        self,
        configs: List[DataSourceConfig],
        merge: bool = False,
        merge_on: Optional[Union[str, List[str]]] = None,
        merge_how: str = "outer",
    ) -> Union[List[IngestedData], IngestedData]:
        """
        Ingest data from multiple sources.
        
        Args:
            configs: List of data source configurations
            merge: Whether to merge all data into a single DataFrame
            merge_on: Column(s) to merge on
            merge_how: Merge type ('inner', 'outer', 'left', 'right')
            
        Returns:
            List of IngestedData objects, or single merged IngestedData if merge=True
        """
        results = [self.ingest(config) for config in configs]
        
        if not merge:
            return results
        
        # Merge all DataFrames
        merged_df = results[0].data
        for result in results[1:]:
            if merge_on:
                merged_df = pd.merge(merged_df, result.data, on=merge_on, how=merge_how)
            else:
                # Concatenate if no merge key
                merged_df = pd.concat([merged_df, result.data], ignore_index=True)
        
        return IngestedData(
            data=merged_df,
            source_name="merged",
            source_type="merged",
            ingested_at=datetime.now(),
            row_count=len(merged_df),
            column_count=len(merged_df.columns),
            schema={col: str(dtype) for col, dtype in merged_df.dtypes.items()},
            metadata={"sources": [r.source_name for r in results]},
        )
    
    def _ingest_csv(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest CSV file."""
        path = Path(config.path_or_uri)
        
        kwargs = {
            "sep": "," if config.source_type == "csv" else "\t",
            "parse_dates": [config.timestamp_column] if config.timestamp_column else False,
        }
        
        # Add schema if provided
        if config.schema:
            kwargs["dtype"] = config.schema
        
        return pd.read_csv(path, **kwargs)
    
    def _ingest_json(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest JSON file."""
        path = Path(config.path_or_uri)
        return pd.read_json(path)
    
    def _ingest_jsonl(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest JSON Lines file."""
        path = Path(config.path_or_uri)
        return pd.read_json(path, lines=True)
    
    def _ingest_parquet(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest Parquet file."""
        path = Path(config.path_or_uri)
        return pd.read_parquet(path)
    
    def _ingest_excel(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest Excel file."""
        path = Path(config.path_or_uri)
        sheet = config.metadata.get("sheet_name", 0)
        return pd.read_excel(path, sheet_name=sheet)
    
    def _ingest_database(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest from SQL database."""
        import sqlalchemy
        
        engine = sqlalchemy.create_engine(config.path_or_uri)
        query = config.metadata.get("query", f"SELECT * FROM {config.name}")
        return pd.read_sql(query, engine)
    
    def _ingest_api(self, config: DataSourceConfig) -> pd.DataFrame:
        """Ingest from REST API."""
        import requests
        
        headers = config.metadata.get("headers", {})
        params = config.metadata.get("params", {})
        
        response = requests.get(config.path_or_uri, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different response structures
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try common patterns
            for key in ["data", "results", "items", "records"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
        
        return pd.DataFrame()
    
    def _ingest_auto(self, config: DataSourceConfig) -> pd.DataFrame:
        """Auto-detect file type and ingest."""
        path = Path(config.path_or_uri)
        ext = path.suffix.lower()
        
        if ext in self.FILE_HANDLERS:
            handler_type = self.FILE_HANDLERS[ext]
            config.source_type = handler_type
            return self.ingest(config, use_cache=False).data
        
        raise ValueError(f"Cannot auto-detect type for: {path}")
    
    def stream(
        self,
        config: DataSourceConfig,
        chunk_size: int = 10000,
    ) -> Iterator[IngestedData]:
        """
        Stream large files in chunks.
        
        Useful for processing large datasets that don't fit in memory.
        """
        if config.source_type not in ("csv", "tsv", "jsonl"):
            raise ValueError(f"Streaming not supported for: {config.source_type}")
        
        path = Path(config.path_or_uri)
        
        if config.source_type in ("csv", "tsv"):
            sep = "," if config.source_type == "csv" else "\t"
            chunks = pd.read_csv(path, sep=sep, chunksize=chunk_size)
        else:  # jsonl
            chunks = pd.read_json(path, lines=True, chunksize=chunk_size)
        
        for i, chunk in enumerate(chunks):
            yield IngestedData(
                data=chunk,
                source_name=f"{config.name}_chunk_{i}",
                source_type=config.source_type,
                ingested_at=datetime.now(),
                row_count=len(chunk),
                column_count=len(chunk.columns),
                schema={col: str(dtype) for col, dtype in chunk.dtypes.items()},
                metadata={"chunk_index": i, **config.metadata},
            )
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

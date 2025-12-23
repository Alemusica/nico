"""
Schema inference and validation for heterogeneous data.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

import pandas as pd
import numpy as np


class DataType(Enum):
    """Standard data types for schema definition."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    data_type: DataType
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[Set[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    description: str = ""
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single value against this schema."""
        # Check null
        if pd.isna(value):
            if not self.nullable:
                return False, f"Column {self.name} does not allow null values"
            return True, None
        
        # Check type
        if self.data_type == DataType.INTEGER:
            if not isinstance(value, (int, np.integer)):
                return False, f"Expected integer for {self.name}, got {type(value)}"
        elif self.data_type == DataType.FLOAT:
            if not isinstance(value, (float, int, np.number)):
                return False, f"Expected numeric for {self.name}, got {type(value)}"
        elif self.data_type == DataType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string for {self.name}, got {type(value)}"
            if self.pattern:
                if not re.match(self.pattern, value):
                    return False, f"Value does not match pattern {self.pattern}"
        elif self.data_type == DataType.BOOLEAN:
            if not isinstance(value, (bool, np.bool_)):
                return False, f"Expected boolean for {self.name}, got {type(value)}"
        
        # Check range
        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} below minimum {self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} above maximum {self.max_value}"
        
        # Check allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Value {value} not in allowed values"
        
        return True, None


@dataclass
class DataSchema:
    """Complete schema for a dataset."""
    name: str
    columns: Dict[str, ColumnSchema] = field(default_factory=dict)
    primary_key: Optional[str] = None
    timestamp_column: Optional[str] = None
    target_column: Optional[str] = None
    description: str = ""
    
    def add_column(self, column: ColumnSchema) -> "DataSchema":
        """Add a column schema."""
        self.columns[column.name] = column
        return self
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a DataFrame against this schema."""
        errors = []
        
        # Check required columns
        missing_cols = set(self.columns.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Validate each column
        for col_name, col_schema in self.columns.items():
            if col_name not in df.columns:
                continue
            
            # Check uniqueness
            if col_schema.unique:
                if df[col_name].duplicated().any():
                    errors.append(f"Column {col_name} contains duplicate values")
            
            # Sample validation (avoid checking every row for large datasets)
            sample = df[col_name].dropna().sample(min(100, len(df)))
            for value in sample:
                is_valid, error = col_schema.validate(value)
                if not is_valid:
                    errors.append(error)
                    break  # One error per column is enough
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "columns": {
                name: {
                    "data_type": col.data_type.value,
                    "nullable": col.nullable,
                    "unique": col.unique,
                    "min_value": col.min_value,
                    "max_value": col.max_value,
                    "allowed_values": list(col.allowed_values) if col.allowed_values else None,
                    "pattern": col.pattern,
                    "description": col.description,
                }
                for name, col in self.columns.items()
            },
            "primary_key": self.primary_key,
            "timestamp_column": self.timestamp_column,
            "target_column": self.target_column,
            "description": self.description,
        }


class SchemaInferrer:
    """
    Infer schema from DataFrame.
    
    Example:
        inferrer = SchemaInferrer()
        schema = inferrer.infer(df)
        
        # Validate new data
        is_valid, errors = schema.validate(new_df)
    """
    
    # Common datetime patterns
    DATETIME_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    ]
    
    # Common patterns for specific column types
    COLUMN_PATTERNS = {
        "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        "phone": r"^\+?[\d\s\-\(\)]+$",
        "url": r"^https?://",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    }
    
    def __init__(
        self,
        sample_size: int = 1000,
        categorical_threshold: float = 0.05,
    ):
        """
        Args:
            sample_size: Number of rows to sample for inference
            categorical_threshold: If unique ratio < threshold, treat as categorical
        """
        self.sample_size = sample_size
        self.categorical_threshold = categorical_threshold
    
    def infer(
        self,
        df: pd.DataFrame,
        name: str = "inferred_schema",
    ) -> DataSchema:
        """Infer schema from DataFrame."""
        schema = DataSchema(name=name)
        
        for col in df.columns:
            col_schema = self._infer_column(df[col])
            schema.add_column(col_schema)
        
        # Try to identify special columns
        schema.primary_key = self._identify_primary_key(df)
        schema.timestamp_column = self._identify_timestamp(df)
        
        return schema
    
    def _infer_column(self, series: pd.Series) -> ColumnSchema:
        """Infer schema for a single column."""
        col_name = series.name
        nullable = series.isna().any()
        unique = series.nunique() == len(series)
        
        # Sample non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return ColumnSchema(
                name=col_name,
                data_type=DataType.UNKNOWN,
                nullable=True,
            )
        
        sample = non_null.sample(min(self.sample_size, len(non_null)))
        
        # Infer type
        data_type = self._infer_type(sample)
        
        # Get value range for numeric types
        min_value = None
        max_value = None
        if data_type in (DataType.INTEGER, DataType.FLOAT):
            min_value = float(non_null.min())
            max_value = float(non_null.max())
        
        # Check if categorical
        allowed_values = None
        unique_ratio = series.nunique() / len(series)
        if data_type == DataType.STRING and unique_ratio < self.categorical_threshold:
            data_type = DataType.CATEGORICAL
            allowed_values = set(series.unique())
        
        return ColumnSchema(
            name=col_name,
            data_type=data_type,
            nullable=nullable,
            unique=unique,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
        )
    
    def _infer_type(self, sample: pd.Series) -> DataType:
        """Infer data type from sample values."""
        dtype = sample.dtype
        
        # Check pandas dtype first
        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN
        elif pd.api.types.is_integer_dtype(dtype):
            return DataType.INTEGER
        elif pd.api.types.is_float_dtype(dtype):
            return DataType.FLOAT
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return DataType.DATETIME
        
        # For object dtype, sample values
        if dtype == "object":
            # Check if boolean strings
            bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f"}
            sample_lower = set(str(v).lower() for v in sample)
            if sample_lower.issubset(bool_values):
                return DataType.BOOLEAN
            
            # Check if datetime strings
            sample_str = sample.astype(str).iloc[0]
            for pattern in self.DATETIME_PATTERNS:
                if re.match(pattern, sample_str):
                    return DataType.DATETIME
            
            # Check if numeric strings
            try:
                pd.to_numeric(sample)
                return DataType.FLOAT
            except (ValueError, TypeError):
                pass
            
            return DataType.STRING
        
        return DataType.UNKNOWN
    
    def _identify_primary_key(self, df: pd.DataFrame) -> Optional[str]:
        """Try to identify a primary key column."""
        for col in df.columns:
            # Check common naming patterns
            if any(pk in col.lower() for pk in ["id", "key", "pk", "number"]):
                if df[col].nunique() == len(df):
                    return col
        return None
    
    def _identify_timestamp(self, df: pd.DataFrame) -> Optional[str]:
        """Try to identify timestamp column."""
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
            if any(ts in col.lower() for ts in ["date", "time", "timestamp", "created", "updated"]):
                return col
        return None

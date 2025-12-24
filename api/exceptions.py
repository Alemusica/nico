"""
🚨 Custom Exceptions
====================
Domain-specific exceptions for the Causal Discovery API.
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any


# ==========================
# BASE EXCEPTIONS
# ==========================

class CausalDiscoveryError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# ==========================
# DATA EXCEPTIONS
# ==========================

class DatasetNotFoundError(CausalDiscoveryError):
    """Raised when a requested dataset does not exist."""
    
    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"Dataset '{dataset_name}' not found",
            details={"dataset_name": dataset_name}
        )


class DatasetLoadError(CausalDiscoveryError):
    """Raised when dataset loading fails."""
    
    def __init__(self, dataset_name: str, reason: str):
        super().__init__(
            message=f"Failed to load dataset '{dataset_name}': {reason}",
            details={"dataset_name": dataset_name, "reason": reason}
        )


class InvalidDataFormatError(CausalDiscoveryError):
    """Raised when data format is invalid."""
    
    def __init__(self, expected: str, received: str):
        super().__init__(
            message=f"Invalid data format. Expected: {expected}, Received: {received}",
            details={"expected": expected, "received": received}
        )


class InsufficientDataError(CausalDiscoveryError):
    """Raised when dataset has insufficient data for analysis."""
    
    def __init__(self, required_rows: int, actual_rows: int):
        super().__init__(
            message=f"Insufficient data. Required: {required_rows} rows, Got: {actual_rows} rows",
            details={"required_rows": required_rows, "actual_rows": actual_rows}
        )


# ==========================
# LLM EXCEPTIONS
# ==========================

class LLMUnavailableError(CausalDiscoveryError):
    """Raised when LLM service is unavailable."""
    
    def __init__(self, service_url: str, reason: Optional[str] = None):
        message = f"LLM service unavailable at {service_url}"
        if reason:
            message += f": {reason}"
        super().__init__(
            message=message,
            details={"service_url": service_url, "reason": reason}
        )


class LLMTimeoutError(CausalDiscoveryError):
    """Raised when LLM request times out."""
    
    def __init__(self, timeout_seconds: int):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds}
        )


class LLMResponseError(CausalDiscoveryError):
    """Raised when LLM returns invalid or unexpected response."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"LLM response error: {reason}",
            details={"reason": reason}
        )


# ==========================
# KNOWLEDGE BASE EXCEPTIONS
# ==========================

class KnowledgeItemNotFoundError(CausalDiscoveryError):
    """Raised when knowledge base item is not found."""
    
    def __init__(self, item_type: str, item_id: str):
        super().__init__(
            message=f"{item_type} with ID '{item_id}' not found",
            details={"item_type": item_type, "item_id": item_id}
        )


class InvalidKnowledgeStructureError(CausalDiscoveryError):
    """Raised when knowledge item has invalid structure."""
    
    def __init__(self, item_type: str, reason: str):
        super().__init__(
            message=f"Invalid {item_type} structure: {reason}",
            details={"item_type": item_type, "reason": reason}
        )


class DatabaseConnectionError(CausalDiscoveryError):
    """Raised when database connection fails."""
    
    def __init__(self, database: str, reason: str):
        super().__init__(
            message=f"Failed to connect to {database}: {reason}",
            details={"database": database, "reason": reason}
        )


# ==========================
# INVESTIGATION EXCEPTIONS
# ==========================

class InvestigationFailedError(CausalDiscoveryError):
    """Raised when investigation process fails."""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Investigation failed for query '{query}': {reason}",
            details={"query": query, "reason": reason}
        )


class InvalidQueryError(CausalDiscoveryError):
    """Raised when investigation query is invalid."""
    
    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Invalid query '{query}': {reason}",
            details={"query": query, "reason": reason}
        )


class BriefingNotFoundError(CausalDiscoveryError):
    """Raised when investigation briefing is not found."""
    
    def __init__(self, briefing_id: str):
        super().__init__(
            message=f"Briefing with ID '{briefing_id}' not found",
            details={"briefing_id": briefing_id}
        )


# ==========================
# CAUSAL ANALYSIS EXCEPTIONS
# ==========================

class CausalAnalysisError(CausalDiscoveryError):
    """Raised when causal analysis fails."""
    
    def __init__(self, method: str, reason: str):
        super().__init__(
            message=f"Causal analysis ({method}) failed: {reason}",
            details={"method": method, "reason": reason}
        )


class InvalidVariableError(CausalDiscoveryError):
    """Raised when variable name is invalid."""
    
    def __init__(self, variable: str, available: list[str]):
        super().__init__(
            message=f"Invalid variable '{variable}'. Available: {', '.join(available)}",
            details={"variable": variable, "available_variables": available}
        )


class PCMCINotAvailableError(CausalDiscoveryError):
    """Raised when PCMCI/Tigramite is not available."""
    
    def __init__(self):
        super().__init__(
            message="PCMCI/Tigramite not available. Install with: pip install tigramite",
            details={"fallback": "correlation-based discovery"}
        )


# ==========================
# HTTP EXCEPTION MAPPERS
# ==========================

def map_to_http_exception(error: CausalDiscoveryError) -> HTTPException:
    """Map domain exception to HTTP exception."""
    
    # 404 Not Found
    if isinstance(error, (
        DatasetNotFoundError,
        KnowledgeItemNotFoundError,
        BriefingNotFoundError
    )):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": error.message, **error.details}
        )
    
    # 400 Bad Request
    if isinstance(error, (
        InvalidDataFormatError,
        InvalidQueryError,
        InvalidVariableError,
        InvalidKnowledgeStructureError
    )):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": error.message, **error.details}
        )
    
    # 422 Unprocessable Entity
    if isinstance(error, InsufficientDataError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": error.message, **error.details}
        )
    
    # 503 Service Unavailable
    if isinstance(error, (
        LLMUnavailableError,
        DatabaseConnectionError,
        PCMCINotAvailableError
    )):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": error.message, **error.details}
        )
    
    # 504 Gateway Timeout
    if isinstance(error, LLMTimeoutError):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": error.message, **error.details}
        )
    
    # 500 Internal Server Error (default)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"message": error.message, **error.details}
    )

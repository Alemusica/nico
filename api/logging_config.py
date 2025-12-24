"""
📝 Structured Logging Configuration
====================================
Structured logging with request ID tracking and JSON output.
"""

import structlog
import logging
import sys
from typing import Any, Dict
from pathlib import Path
from datetime import datetime
import uuid


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Path | None = None
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "text")
        log_file: Optional file path for logging
    """
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        stream=sys.stdout
    )
    
    # Processors for structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]
    
    # Add request ID processor
    processors.append(add_request_id)
    
    # Choose renderer based on format
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.root.addHandler(file_handler)


def add_request_id(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Add request ID to log events."""
    # Try to get from context vars, otherwise generate
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    event_dict["request_id"] = request_id
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name or __name__)


# Convenience loggers for different components
api_logger = get_logger("api")
data_logger = get_logger("data")
llm_logger = get_logger("llm")
knowledge_logger = get_logger("knowledge")
investigation_logger = get_logger("investigation")
causal_logger = get_logger("causal")

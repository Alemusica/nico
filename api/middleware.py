"""
🎯 Request ID Middleware
========================
Middleware for tracking requests with unique IDs.
"""

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import time


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        
        # Store in context vars for logging
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        # Add to request state
        request.state.request_id = request_id
        
        # Process request and measure time
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log request completion
            duration = time.time() - start_time
            logger = structlog.get_logger("api.request")
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger = structlog.get_logger("api.request")
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            raise
        finally:
            # Clear context vars
            structlog.contextvars.clear_contextvars()

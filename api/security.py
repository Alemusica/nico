"""
🔒 Security Middleware
======================
Security headers and validation middleware.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable
from api.config import get_settings
from api.logging_config import get_logger

logger = get_logger("api.security")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    
    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security (if HTTPS)
    - Content-Security-Policy
    - Referrer-Policy: strict-origin-when-cross-origin
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # HSTS (HTTP Strict Transport Security) - only if HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",  # Allow inline scripts for docs
            "style-src 'self' 'unsafe-inline'",   # Allow inline styles for docs
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate and sanitize request inputs.
    
    Checks:
    - Content-Length limits
    - Content-Type validation
    - Request size limits
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.settings = get_settings()
        self.max_request_size = 100 * 1024 * 1024  # 100 MB
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request before processing."""
        
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(
                "request_too_large",
                content_length=content_length,
                max_size=self.max_request_size,
                path=request.url.path
            )
            return Response(
                content='{"error": "Request payload too large"}',
                status_code=413,
                media_type="application/json"
            )
        
        # Validate Content-Type for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            # Skip validation for docs endpoints
            if "/docs" in request.url.path or "/redoc" in request.url.path:
                return await call_next(request)
            
            # Allow multipart/form-data, application/json, application/x-www-form-urlencoded
            allowed_types = [
                "application/json",
                "multipart/form-data",
                "application/x-www-form-urlencoded"
            ]
            
            if not any(ct in content_type for ct in allowed_types):
                logger.warning(
                    "invalid_content_type",
                    content_type=content_type,
                    path=request.url.path
                )
                return Response(
                    content='{"error": "Invalid Content-Type"}',
                    status_code=415,
                    media_type="application/json"
                )
        
        return await call_next(request)


def setup_security_middleware(app):
    """
    Setup security middleware for FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    settings = get_settings()
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("security_headers_middleware_added")
    
    # Add input validation middleware
    app.add_middleware(InputValidationMiddleware)
    logger.info("input_validation_middleware_added")
    
    logger.info("security_middleware_configured")

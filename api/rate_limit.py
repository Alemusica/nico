"""
🚦 Rate Limiting Middleware
============================
Request rate limiting using slowapi with configurable limits.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from api.config import get_settings
from api.logging_config import get_logger

logger = get_logger("api.rate_limit")


def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key from request.
    
    Uses X-Forwarded-For header if behind proxy, otherwise remote address.
    """
    settings = get_settings()
    
    # If behind proxy, use X-Forwarded-For
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP in chain
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = get_remote_address(request)
    
    logger.debug("rate_limit_key_extracted", client_ip=client_ip, forwarded=forwarded)
    return client_ip


def create_limiter():
    """Create and configure rate limiter."""
    settings = get_settings()
    
    if not settings.rate_limit_enabled:
        logger.info("rate_limiting_disabled", message="Rate limiting is disabled in settings")
        return None
    
    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=[
            f"{settings.rate_limit_per_minute}/minute",
        ],
        storage_uri="memory://",  # In-memory storage (use Redis for production)
        strategy="fixed-window",
    )
    
    logger.info(
        "rate_limiter_created",
        per_minute=settings.rate_limit_per_minute,
        storage="memory",
        strategy="fixed-window"
    )
    
    return limiter


def setup_rate_limiting(app):
    """
    Setup rate limiting for FastAPI application.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        limiter instance or None if disabled
    """
    settings = get_settings()
    
    if not settings.rate_limit_enabled:
        logger.info("rate_limiting_setup_skipped", enabled=False)
        return None
    
    limiter = create_limiter()
    
    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add middleware
    app.add_middleware(SlowAPIMiddleware)
    
    # Store limiter in app state
    app.state.limiter = limiter
    
    logger.info(
        "rate_limiting_configured",
        per_minute=settings.rate_limit_per_minute,
        enabled=True
    )
    
    return limiter


# Global limiter instance (initialized in main.py)
_limiter = None


def get_limiter() -> Limiter:
    """Get the global limiter instance."""
    return _limiter


def set_limiter(limiter: Limiter):
    """Set the global limiter instance."""
    global _limiter
    _limiter = limiter

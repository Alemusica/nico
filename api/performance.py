"""
⚡ Performance Optimization Utilities
=====================================
Caching, async optimization, and performance monitoring.
"""

from functools import wraps, lru_cache
from typing import Any, Callable, Optional
import time
import asyncio
from datetime import datetime, timedelta
from api.logging_config import get_logger
from api.config import get_settings

logger = get_logger("api.performance")


# =====================
# SIMPLE MEMORY CACHE
# =====================

class SimpleCache:
    """
    Simple in-memory cache with TTL.
    
    For production, use Redis or Memcached.
    """
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self.settings = get_settings()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        
        # Check TTL
        timestamp = self._timestamps.get(key)
        if timestamp:
            age_seconds = (datetime.now() - timestamp).total_seconds()
            ttl_seconds = self.settings.cache_ttl_days * 86400
            
            if age_seconds > ttl_seconds:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
                logger.debug("cache_expired", key=key, age_seconds=age_seconds)
                return None
        
        logger.debug("cache_hit", key=key)
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache with timestamp."""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
        logger.debug("cache_set", key=key)
    
    def clear(self):
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        logger.info("cache_cleared", entries_removed=count)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "oldest_entry": min(self._timestamps.values()) if self._timestamps else None,
            "newest_entry": max(self._timestamps.values()) if self._timestamps else None,
        }


# Global cache instance
_cache = SimpleCache()


def get_cache() -> SimpleCache:
    """Get global cache instance."""
    return _cache


# =====================
# CACHING DECORATOR
# =====================

def cached(ttl_seconds: Optional[int] = None, key_prefix: str = ""):
    """
    Cache function results in memory.
    
    Args:
        ttl_seconds: Time to live in seconds (default: from settings)
        key_prefix: Prefix for cache keys
    
    Example:
        @cached(ttl_seconds=300, key_prefix="datasets")
        async def get_datasets():
            return await expensive_operation()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                logger.debug("cache_hit", function=func.__name__, key=cache_key)
                return cached_value
            
            # Execute function
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Store in cache
            _cache.set(cache_key, result)
            
            logger.debug(
                "cache_miss_computed",
                function=func.__name__,
                duration_ms=duration * 1000,
                key=cache_key
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached_value = _cache.get(cache_key)
            if cached_value is not None:
                logger.debug("cache_hit", function=func.__name__, key=cache_key)
                return cached_value
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Store in cache
            _cache.set(cache_key, result)
            
            logger.debug(
                "cache_miss_computed",
                function=func.__name__,
                duration_ms=duration * 1000,
                key=cache_key
            )
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =====================
# PERFORMANCE MONITORING
# =====================

def timed(log_slow_threshold_ms: float = 1000.0):
    """
    Log execution time of functions.
    
    Args:
        log_slow_threshold_ms: Log as warning if slower than this (ms)
    
    Example:
        @timed(log_slow_threshold_ms=500)
        async def slow_operation():
            await asyncio.sleep(1)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            if duration_ms > log_slow_threshold_ms:
                logger.warning(
                    "slow_operation",
                    function=func.__name__,
                    duration_ms=duration_ms,
                    threshold_ms=log_slow_threshold_ms
                )
            else:
                logger.debug(
                    "operation_completed",
                    function=func.__name__,
                    duration_ms=duration_ms
                )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            if duration_ms > log_slow_threshold_ms:
                logger.warning(
                    "slow_operation",
                    function=func.__name__,
                    duration_ms=duration_ms,
                    threshold_ms=log_slow_threshold_ms
                )
            else:
                logger.debug(
                    "operation_completed",
                    function=func.__name__,
                    duration_ms=duration_ms
                )
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =====================
# ASYNC OPTIMIZATION
# =====================

async def gather_with_limit(tasks: list, max_concurrent: int = 10):
    """
    Run async tasks with concurrency limit.
    
    Prevents overwhelming system resources when running many tasks.
    
    Args:
        tasks: List of coroutines to execute
        max_concurrent: Maximum concurrent tasks
    
    Returns:
        List of results in same order as tasks
    
    Example:
        results = await gather_with_limit([
            fetch_data(1),
            fetch_data(2),
            fetch_data(3),
        ], max_concurrent=2)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[bounded_task(task) for task in tasks])


# =====================
# BATCH PROCESSING
# =====================

def batch_processor(batch_size: int = 100):
    """
    Process items in batches.
    
    Args:
        batch_size: Number of items per batch
    
    Example:
        @batch_processor(batch_size=50)
        async def process_items(items: list):
            # Process batch of items
            return [process(item) for item in items]
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(items: list, *args, **kwargs):
            results = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                logger.debug(
                    "processing_batch",
                    function=func.__name__,
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                    total=len(items)
                )
                
                batch_result = await func(batch, *args, **kwargs)
                results.extend(batch_result)
            
            return results
        
        @wraps(func)
        def sync_wrapper(items: list, *args, **kwargs):
            results = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                logger.debug(
                    "processing_batch",
                    function=func.__name__,
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                    total=len(items)
                )
                
                batch_result = func(batch, *args, **kwargs)
                results.extend(batch_result)
            
            return results
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

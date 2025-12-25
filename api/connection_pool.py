"""
🔌 Connection Pool Manager
===========================
Production-grade connection pooling with:
- Async connection management
- Health checks and auto-reconnection
- Circuit breaker pattern
- Retry logic with exponential backoff
- Connection lifecycle management
"""

import asyncio
import logging
from typing import Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failure threshold exceeded
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: int = 60  # Seconds to wait before half-open
    


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    failed_attempts: int = 0
    successful_requests: int = 0
    last_health_check: Optional[datetime] = None
    circuit_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    Prevents cascading failures by temporarily blocking requests to failing services.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker HALF_OPEN - testing service")
                    return True
            return False
        
        # HALF_OPEN state
        return True


class AsyncConnectionPool:
    """
    Async connection pool with health checks and retry logic.
    """
    
    def __init__(
        self,
        create_connection: Callable,
        max_connections: int = 10,
        min_connections: int = 2,
        health_check_interval: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.pool: list[Any] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.stats = ConnectionStats()
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize pool with minimum connections."""
        logger.info(f"Initializing connection pool (min={self.min_connections}, max={self.max_connections})")
        
        for _ in range(self.min_connections):
            try:
                conn = await self._create_connection_with_retry()
                if conn:
                    self.pool.append(conn)
                    await self.available.put(conn)
                    self.stats.total_connections += 1
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
        
        # Start health check background task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Connection pool initialized with {len(self.pool)} connections")
    
    async def _create_connection_with_retry(self) -> Optional[Any]:
        """Create connection with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                conn = await self.create_connection()
                self.circuit_breaker.record_success()
                return conn
            except Exception as e:
                self.circuit_breaker.record_failure()
                self.stats.failed_attempts += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} connection attempts failed")
                    return None
    
    async def acquire(self) -> Optional[Any]:
        """Acquire connection from pool."""
        if not self.circuit_breaker.can_attempt():
            logger.warning("Circuit breaker OPEN - rejecting connection request")
            return None
        
        async with self._lock:
            self.stats.active_connections += 1
            
            # Try to get available connection
            if not self.available.empty():
                return await self.available.get()
            
            # Create new connection if under max
            if len(self.pool) < self.max_connections:
                conn = await self._create_connection_with_retry()
                if conn:
                    self.pool.append(conn)
                    self.stats.total_connections += 1
                    return conn
        
        # Wait for available connection
        logger.debug("Waiting for available connection...")
        return await self.available.get()
    
    async def release(self, conn: Any):
        """Release connection back to pool."""
        async with self._lock:
            self.stats.active_connections -= 1
            self.stats.successful_requests += 1
        await self.available.put(conn)
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._health_check()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
    
    async def _health_check(self):
        """Perform health check on all connections."""
        logger.debug("Running connection pool health check...")
        self.stats.last_health_check = datetime.now()
        
        healthy_count = 0
        unhealthy = []
        
        for conn in self.pool:
            try:
                # Simple ping/health check (customize based on connection type)
                if hasattr(conn, 'ping'):
                    await conn.ping()
                healthy_count += 1
            except Exception as e:
                logger.warning(f"Unhealthy connection detected: {e}")
                unhealthy.append(conn)
        
        # Remove unhealthy connections
        for conn in unhealthy:
            try:
                await conn.close()
            except:
                pass
            self.pool.remove(conn)
            self.stats.total_connections -= 1
        
        # Replenish to minimum
        while len(self.pool) < self.min_connections:
            conn = await self._create_connection_with_retry()
            if conn:
                self.pool.append(conn)
                await self.available.put(conn)
                self.stats.total_connections += 1
        
        self.stats.circuit_state = self.circuit_breaker.state
        logger.info(f"Health check complete: {healthy_count}/{len(self.pool)} healthy, circuit: {self.circuit_breaker.state.value}")
    
    async def close(self):
        """Close all connections and cleanup."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        for conn in self.pool:
            try:
                if hasattr(conn, 'close'):
                    await conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self.pool.clear()
        logger.info("Connection pool closed")
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "total_connections": self.stats.total_connections,
            "active_connections": self.stats.active_connections,
            "available_connections": self.available.qsize(),
            "failed_attempts": self.stats.failed_attempts,
            "successful_requests": self.stats.successful_requests,
            "last_health_check": self.stats.last_health_check.isoformat() if self.stats.last_health_check else None,
            "circuit_state": self.stats.circuit_state.value,
            "max_connections": self.max_connections,
            "min_connections": self.min_connections
        }

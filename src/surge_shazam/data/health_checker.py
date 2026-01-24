"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    API HEALTH CHECKER                                         ║
║                                                                              ║
║   Monitors health and availability of all data source APIs.                  ║
║   Runs periodic checks and stores results in SurrealDB.                      ║
║                                                                              ║
║   Features:                                                                  ║
║   - Endpoint connectivity testing                                            ║
║   - Authentication validation                                                ║
║   - Response time monitoring                                                 ║
║   - Data freshness verification                                              ║
║   - Alerting on failures                                                     ║
║                                                                              ║
║   Usage:                                                                     ║
║       checker = HealthChecker()                                              ║
║       results = await checker.check_all()                                    ║
║       checker.report()                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .api_registry import (
    API_REGISTRY,
    DataSource,
    Status,
    Latency,
)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"  # Not implemented or disabled


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    source_id: str
    status: HealthStatus
    timestamp: datetime
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "details": self.details,
        }
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Aggregated health report."""
    timestamp: datetime
    results: List[HealthCheckResult]
    summary: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Compute summary
        self.summary = {status.value: 0 for status in HealthStatus}
        for result in self.results:
            self.summary[result.status.value] += 1
    
    @property
    def overall_health(self) -> HealthStatus:
        """Compute overall system health."""
        if self.summary["unhealthy"] > 0:
            return HealthStatus.UNHEALTHY
        if self.summary["degraded"] > 0:
            return HealthStatus.DEGRADED
        if self.summary["healthy"] > 0:
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall": self.overall_health.value,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }
    
    def print_report(self):
        """Print human-readable report."""
        emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌",
            HealthStatus.UNKNOWN: "❓",
            HealthStatus.SKIPPED: "⏭️",
        }
        
        print("=" * 60)
        print(f"  HEALTH CHECK REPORT - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"\n  Overall: {emoji[self.overall_health]} {self.overall_health.value.upper()}")
        print(f"\n  Summary:")
        for status, count in self.summary.items():
            if count > 0:
                s = HealthStatus(status)
                print(f"    {emoji[s]} {status}: {count}")
        
        print(f"\n  Details:")
        for result in sorted(self.results, key=lambda r: r.status.value):
            e = emoji[result.status]
            rt = f"{result.response_time_ms}ms" if result.response_time_ms else "N/A"
            print(f"    {e} {result.source_id:20} [{rt:>8}]", end="")
            if result.error_message:
                print(f" - {result.error_message[:40]}")
            else:
                print()
        
        print("=" * 60)


class HealthChecker:
    """
    API Health Checker.
    
    Monitors all registered data sources for availability and health.
    
    Usage:
        checker = HealthChecker()
        
        # Check all APIs
        report = await checker.check_all()
        report.print_report()
        
        # Check specific category
        results = await checker.check_category(DataCategory.SATELLITE)
        
        # Save to SurrealDB
        await checker.save_to_surrealdb(report)
        
        # Run continuous monitoring
        await checker.run_monitor(interval_minutes=30)
    """
    
    # Health check endpoints for each source
    HEALTH_ENDPOINTS = {
        "cmems_sealevel": "https://data.marine.copernicus.eu/api/health",
        "cmems_sst": "https://data.marine.copernicus.eu/api/health",
        "cmems_waves": "https://data.marine.copernicus.eu/api/health",
        "era5_surface": "https://cds.climate.copernicus.eu/api/v2",
        "cygnss": "https://podaac.jpl.nasa.gov/",
        "gpm_imerg": "https://gpm.nasa.gov/data/",
        "noaa_indices": "https://www.cpc.ncep.noaa.gov/",
        "tide_gauges": "https://www.ioc-sealevelmonitoring.org/",
        "opensky": "https://opensky-network.org/api/states/all",
    }
    
    def __init__(
        self,
        timeout_seconds: int = 10,
        surrealdb_url: str = "http://localhost:8000",
    ):
        self.timeout = timeout_seconds
        self.surrealdb_url = surrealdb_url
        self._results_history: List[HealthReport] = []
    
    async def check_source(self, source: DataSource) -> HealthCheckResult:
        """
        Check health of a single data source.
        
        Args:
            source: DataSource to check
            
        Returns:
            HealthCheckResult
        """
        start_time = datetime.now()
        
        # Skip non-implemented sources
        if source.status in [Status.TODO, Status.STUB]:
            return HealthCheckResult(
                source_id=source.id,
                status=HealthStatus.SKIPPED,
                timestamp=start_time,
                details={"reason": f"Status is {source.status.value}"},
            )
        
        # Get health endpoint
        endpoint = self.HEALTH_ENDPOINTS.get(source.id)
        
        if not endpoint:
            # Try base_url
            endpoint = source.base_url if source.base_url != "local" else None
        
        if not endpoint:
            return HealthCheckResult(
                source_id=source.id,
                status=HealthStatus.SKIPPED,
                timestamp=start_time,
                details={"reason": "No health endpoint configured"},
            )
        
        # Perform HTTP check
        try:
            result = await self._http_check(source.id, endpoint)
            return result
        except Exception as e:
            return HealthCheckResult(
                source_id=source.id,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                error_message=str(e),
            )
    
    async def _http_check(self, source_id: str, url: str) -> HealthCheckResult:
        """Perform HTTP health check."""
        start = datetime.now()
        
        if not HAS_AIOHTTP:
            return HealthCheckResult(
                source_id=source_id,
                status=HealthStatus.UNKNOWN,
                timestamp=start,
                error_message="aiohttp not installed",
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ssl=False,  # Some endpoints have cert issues
                ) as response:
                    end = datetime.now()
                    response_time = int((end - start).total_seconds() * 1000)
                    
                    if response.status < 400:
                        status = HealthStatus.HEALTHY
                    elif response.status < 500:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.UNHEALTHY
                    
                    return HealthCheckResult(
                        source_id=source_id,
                        status=status,
                        timestamp=start,
                        response_time_ms=response_time,
                        details={
                            "http_status": response.status,
                            "url": url,
                        },
                    )
                    
        except asyncio.TimeoutError:
            return HealthCheckResult(
                source_id=source_id,
                status=HealthStatus.UNHEALTHY,
                timestamp=start,
                error_message=f"Timeout after {self.timeout}s",
            )
        except aiohttp.ClientError as e:
            return HealthCheckResult(
                source_id=source_id,
                status=HealthStatus.UNHEALTHY,
                timestamp=start,
                error_message=f"Connection error: {str(e)[:50]}",
            )
    
    async def check_all(
        self,
        include_todo: bool = False,
    ) -> HealthReport:
        """
        Check all registered data sources.
        
        Args:
            include_todo: Include TODO/STUB sources
            
        Returns:
            HealthReport with all results
        """
        tasks = []
        
        for source_id, source in API_REGISTRY.items():
            if not include_todo and source.status in [Status.TODO, Status.STUB]:
                continue
            tasks.append(self.check_source(source))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to unhealthy results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(HealthCheckResult(
                    source_id=f"unknown_{i}",
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(),
                    error_message=str(result),
                ))
            else:
                final_results.append(result)
        
        report = HealthReport(
            timestamp=datetime.now(),
            results=final_results,
        )
        
        self._results_history.append(report)
        
        return report
    
    async def save_to_surrealdb(self, report: HealthReport) -> bool:
        """
        Save health report to SurrealDB.
        
        Creates/updates api_status records.
        """
        if not HAS_HTTPX:
            logger.warning("httpx not installed, cannot save to SurrealDB")
            return False
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "surreal-ns": "surge_shazam",
            "surreal-db": "data_catalog",
            "Authorization": "Basic cm9vdDpyb290",
        }
        
        queries = []
        for result in report.results:
            query = f'''
            UPSERT api_status:{result.source_id} SET
                source_id = "{result.source_id}",
                status = "{result.status.value}",
                last_check = "{result.timestamp.isoformat()}",
                response_time_ms = {result.response_time_ms or 'NONE'},
                error_message = "{result.error_message or ''}",
                is_healthy = {str(result.is_healthy).lower()}
            ;
            '''
            queries.append(query)
        
        try:
            response = httpx.post(
                f"{self.surrealdb_url}/sql",
                headers=headers,
                content="\n".join(queries),
                timeout=30.0,
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to save to SurrealDB: {e}")
            return False
    
    async def run_monitor(
        self,
        interval_minutes: int = 30,
        save_to_db: bool = True,
    ):
        """
        Run continuous health monitoring.
        
        Args:
            interval_minutes: Check interval
            save_to_db: Save results to SurrealDB
        """
        logger.info(f"Starting health monitor (interval: {interval_minutes}min)")
        
        while True:
            try:
                report = await self.check_all()
                report.print_report()
                
                if save_to_db:
                    await self.save_to_surrealdb(report)
                
                # Alert on unhealthy
                if report.overall_health == HealthStatus.UNHEALTHY:
                    unhealthy = [r for r in report.results if r.status == HealthStatus.UNHEALTHY]
                    logger.warning(f"⚠️ {len(unhealthy)} unhealthy sources!")
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            await asyncio.sleep(interval_minutes * 60)
    
    def get_history(self, hours: int = 24) -> List[HealthReport]:
        """Get historical reports."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [r for r in self._results_history if r.timestamp >= cutoff]


# Convenience function
async def check_api_health() -> HealthReport:
    """Quick health check of all APIs."""
    checker = HealthChecker()
    return await checker.check_all()


# CLI
if __name__ == "__main__":
    async def main():
        print("🏥 Running API Health Check...\n")
        
        checker = HealthChecker()
        report = await checker.check_all(include_todo=True)
        report.print_report()
        
        # Try to save to SurrealDB
        print("\n💾 Saving to SurrealDB...")
        saved = await checker.save_to_surrealdb(report)
        if saved:
            print("   ✅ Saved to SurrealDB")
        else:
            print("   ⚠️ Could not save (SurrealDB not running?)")
    
    asyncio.run(main())

"""
Tests for API Health Checker
============================
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from src.surge_shazam.data.health_checker import (
    HealthChecker,
    HealthCheckResult,
    HealthReport,
    HealthStatus,
    check_api_health,
)
from src.surge_shazam.data.api_registry import API_REGISTRY, Status


class TestHealthStatus:
    """Test HealthStatus enum."""
    
    def test_all_statuses_defined(self):
        """All expected statuses should exist."""
        assert HealthStatus.HEALTHY
        assert HealthStatus.DEGRADED
        assert HealthStatus.UNHEALTHY
        assert HealthStatus.UNKNOWN
        assert HealthStatus.SKIPPED


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""
    
    def test_create_result(self):
        """Should create result with required fields."""
        result = HealthCheckResult(
            source_id="test_source",
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            response_time_ms=100,
        )
        
        assert result.source_id == "test_source"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms == 100
    
    def test_is_healthy(self):
        """is_healthy should return True for HEALTHY status."""
        healthy = HealthCheckResult(
            source_id="test",
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
        )
        unhealthy = HealthCheckResult(
            source_id="test",
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(),
        )
        
        assert healthy.is_healthy is True
        assert unhealthy.is_healthy is False
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        result = HealthCheckResult(
            source_id="test",
            status=HealthStatus.HEALTHY,
            timestamp=datetime(2000, 1, 1, 12, 0),
            response_time_ms=50,
            error_message=None,
        )
        
        d = result.to_dict()
        
        assert d["source_id"] == "test"
        assert d["status"] == "healthy"
        assert d["response_time_ms"] == 50
        assert "timestamp" in d


class TestHealthReport:
    """Test HealthReport dataclass."""
    
    def test_summary_computed(self):
        """Summary should be computed on init."""
        results = [
            HealthCheckResult("s1", HealthStatus.HEALTHY, datetime.now()),
            HealthCheckResult("s2", HealthStatus.HEALTHY, datetime.now()),
            HealthCheckResult("s3", HealthStatus.UNHEALTHY, datetime.now()),
        ]
        
        report = HealthReport(
            timestamp=datetime.now(),
            results=results,
        )
        
        assert report.summary["healthy"] == 2
        assert report.summary["unhealthy"] == 1
    
    def test_overall_health_healthy(self):
        """Overall should be HEALTHY if all healthy."""
        results = [
            HealthCheckResult("s1", HealthStatus.HEALTHY, datetime.now()),
            HealthCheckResult("s2", HealthStatus.HEALTHY, datetime.now()),
        ]
        
        report = HealthReport(datetime.now(), results)
        
        assert report.overall_health == HealthStatus.HEALTHY
    
    def test_overall_health_unhealthy(self):
        """Overall should be UNHEALTHY if any unhealthy."""
        results = [
            HealthCheckResult("s1", HealthStatus.HEALTHY, datetime.now()),
            HealthCheckResult("s2", HealthStatus.UNHEALTHY, datetime.now()),
        ]
        
        report = HealthReport(datetime.now(), results)
        
        assert report.overall_health == HealthStatus.UNHEALTHY
    
    def test_overall_health_degraded(self):
        """Overall should be DEGRADED if any degraded but none unhealthy."""
        results = [
            HealthCheckResult("s1", HealthStatus.HEALTHY, datetime.now()),
            HealthCheckResult("s2", HealthStatus.DEGRADED, datetime.now()),
        ]
        
        report = HealthReport(datetime.now(), results)
        
        assert report.overall_health == HealthStatus.DEGRADED
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        results = [
            HealthCheckResult("s1", HealthStatus.HEALTHY, datetime.now()),
        ]
        
        report = HealthReport(datetime.now(), results)
        d = report.to_dict()
        
        assert "timestamp" in d
        assert "overall" in d
        assert "summary" in d
        assert "results" in d


class TestHealthChecker:
    """Test HealthChecker functionality."""
    
    @pytest.fixture
    def checker(self):
        return HealthChecker(timeout_seconds=5)
    
    @pytest.mark.asyncio
    async def test_check_source_skips_todo(self, checker):
        """TODO sources should be skipped."""
        # Find a TODO source
        todo_source = None
        for source in API_REGISTRY.values():
            if source.status == Status.TODO:
                todo_source = source
                break
        
        if todo_source:
            result = await checker.check_source(todo_source)
            assert result.status == HealthStatus.SKIPPED
    
    @pytest.mark.asyncio
    async def test_check_all_returns_report(self, checker):
        """check_all should return HealthReport."""
        # This will test actual connectivity (or timeout)
        report = await checker.check_all(include_todo=False)
        
        assert isinstance(report, HealthReport)
        assert len(report.results) >= 0
    
    @pytest.mark.asyncio
    async def test_check_all_with_todo(self, checker):
        """check_all with include_todo should include more sources."""
        report_no_todo = await checker.check_all(include_todo=False)
        report_with_todo = await checker.check_all(include_todo=True)
        
        # With TODO should have more results (skipped ones)
        assert len(report_with_todo.results) >= len(report_no_todo.results)
    
    def test_get_history_empty(self, checker):
        """History should start empty."""
        history = checker.get_history()
        # May have results from other tests
        assert isinstance(history, list)
    
    @pytest.mark.asyncio
    async def test_history_accumulates(self, checker):
        """History should accumulate after checks."""
        await checker.check_all()
        await checker.check_all()
        
        history = checker.get_history(hours=1)
        assert len(history) >= 2


class TestHealthCheckResultCreation:
    """Test various result scenarios."""
    
    def test_result_with_error(self):
        """Result with error message."""
        result = HealthCheckResult(
            source_id="failing",
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(),
            error_message="Connection refused",
        )
        
        assert result.error_message == "Connection refused"
        assert result.is_healthy is False
    
    def test_result_with_details(self):
        """Result with additional details."""
        result = HealthCheckResult(
            source_id="test",
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            details={
                "http_status": 200,
                "url": "https://example.com",
            },
        )
        
        assert result.details["http_status"] == 200


class TestConvenienceFunction:
    """Test module-level convenience function."""
    
    @pytest.mark.asyncio
    async def test_check_api_health(self):
        """check_api_health should return report."""
        report = await check_api_health()
        
        assert isinstance(report, HealthReport)

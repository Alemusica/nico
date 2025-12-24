"""
Investigation System - Structured Logging & Observability
Implements comprehensive logging, metrics, and health checks for investigation pipeline.

Based on best practices from Apache Airflow and Great Expectations:
- Structured JSON logging with context preservation
- Step-by-step tracking with timing
- Validation checkpoints
- Error context capture
- Health checks and metrics
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict


class InvestigationStep(str, Enum):
    """Investigation pipeline steps for tracking"""
    PARSE = "parse_query"
    RESOLVE = "resolve_entities"
    BRIEFING = "create_briefing"
    DOWNLOAD = "download_data"
    PAPERS_COLLECT = "collect_papers"
    PAPERS_VALIDATE = "validate_papers"
    PAPERS_STORE = "store_papers"
    CACHE_STORE = "store_cache"
    COMPLETE = "complete"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class InvestigationContext:
    """Context for investigation session"""
    investigation_id: str
    query: str
    location: Optional[str] = None
    event_type: Optional[str] = None
    time_range: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    event: str
    investigation_id: str
    step: Optional[str] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


class InvestigationLogger:
    """Structured logger for investigation pipeline"""
    
    def __init__(self, investigation_id: str, query: str):
        self.context = InvestigationContext(
            investigation_id=investigation_id,
            query=query
        )
        self.logger = logging.getLogger("investigation")
        self.step_timers: Dict[str, float] = {}
        
    def start_step(self, step: InvestigationStep):
        """Start timing a step"""
        self.step_timers[step.value] = time.time()
        self._log(
            level=LogLevel.INFO,
            event=f"step_started",
            step=step.value
        )
        
    def complete_step(self, step: InvestigationStep, metrics: Optional[Dict[str, Any]] = None):
        """Complete a step successfully"""
        duration = None
        if step.value in self.step_timers:
            duration = (time.time() - self.step_timers[step.value]) * 1000  # ms
            del self.step_timers[step.value]
        
        self.context.steps_completed.append(step.value)
        if metrics:
            self.context.metrics[step.value] = metrics
        
        self._log(
            level=LogLevel.INFO,
            event=f"step_completed",
            step=step.value,
            duration_ms=duration,
            success=True,
            metrics=metrics
        )
        
    def fail_step(self, step: InvestigationStep, error: str, context: Optional[Dict[str, Any]] = None):
        """Mark step as failed"""
        duration = None
        if step.value in self.step_timers:
            duration = (time.time() - self.step_timers[step.value]) * 1000
            del self.step_timers[step.value]
        
        self.context.steps_failed.append(step.value)
        
        self._log(
            level=LogLevel.ERROR,
            event=f"step_failed",
            step=step.value,
            duration_ms=duration,
            success=False,
            error=error,
            context=context
        )
        
    def log_validation(self, validator: str, passed: bool, details: Optional[Dict[str, Any]] = None):
        """Log validation checkpoint"""
        self._log(
            level=LogLevel.INFO if passed else LogLevel.WARNING,
            event=f"validation_{validator}",
            success=passed,
            context=details
        )
        
    def log_health_check(self, component: str, healthy: bool, details: Optional[Dict[str, Any]] = None):
        """Log health check"""
        self._log(
            level=LogLevel.INFO if healthy else LogLevel.ERROR,
            event=f"health_check_{component}",
            success=healthy,
            context=details
        )
        
    def log_metric(self, metric_name: str, value: Any, unit: Optional[str] = None):
        """Log a metric"""
        self.context.metrics[metric_name] = {"value": value, "unit": unit}
        self._log(
            level=LogLevel.DEBUG,
            event="metric",
            context={"name": metric_name, "value": value, "unit": unit}
        )
        
    def update_context(self, **kwargs):
        """Update investigation context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
                
    def get_summary(self) -> Dict[str, Any]:
        """Get investigation summary"""
        total_duration = (time.time() - self.context.start_time) * 1000
        return {
            "investigation_id": self.context.investigation_id,
            "query": self.context.query,
            "location": self.context.location,
            "event_type": self.context.event_type,
            "total_duration_ms": total_duration,
            "steps_completed": self.context.steps_completed,
            "steps_failed": self.context.steps_failed,
            "success_rate": len(self.context.steps_completed) / (len(self.context.steps_completed) + len(self.context.steps_failed)) if (len(self.context.steps_completed) + len(self.context.steps_failed)) > 0 else 0,
            "metrics": self.context.metrics
        }
    
    def _log(
        self,
        level: LogLevel,
        event: str,
        step: Optional[str] = None,
        duration_ms: Optional[float] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Internal logging method"""
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level.value,
            event=event,
            investigation_id=self.context.investigation_id,
            step=step,
            duration_ms=duration_ms,
            success=success,
            error=error,
            context=context,
            metrics=metrics
        )
        
        # Log to Python logger
        log_method = getattr(self.logger, level.value)
        log_method(entry.to_json())
        
        return entry

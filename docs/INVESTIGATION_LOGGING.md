# Investigation System - Logging & Validation Framework

## 🎯 Overview

Sistema di **observability** e **data quality** production-ready per la pipeline di investigation, basato su best practices da **Apache Airflow** e **Great Expectations**.

## ✨ Features

### 📊 Structured Logging (`investigation_logger.py`)

- **JSON logging** con timestamp, level, event, context
- **Step tracking** per ogni fase della pipeline
- **Performance metrics**: duration tracking automatico
- **Investigation context** persistente durante tutta l'esecuzione
- **Health checks** per componenti esterni (knowledge_base, cache)
- **Summary automatico** con success rate e metriche aggregate

#### Usage Example

```python
from src.core.investigation_logger import InvestigationLogger, InvestigationStep

# Initialize logger
logger = InvestigationLogger(investigation_id, query)

# Track a step
logger.start_step(InvestigationStep.PAPERS_COLLECT)
# ... do work ...
logger.complete_step(InvestigationStep.PAPERS_COLLECT, {"papers_found": 10})

# Or handle failures
try:
    # ... risky operation ...
except Exception as e:
    logger.fail_step(InvestigationStep.PAPERS_STORE, str(e), context={"details": "..."})

# Get summary
summary = logger.get_summary()
print(f"Duration: {summary['total_duration_ms']}ms")
print(f"Success rate: {summary['success_rate']*100}%")
```

### ✅ Data Validation (`investigation_validators.py`)

#### PaperValidator

Valida papers scientifici prima del salvataggio:

- **CRITICAL**: Required fields (title), DOI format valido
- **WARNING**: Title length ragionevole, abstract presente
- **INFO**: Authors presenti, anno pubblicazione valido

#### PaperSanitizer

Pulisce e normalizza i dati:

- Trim whitespace da stringhe
- Normalizza DOI (rimuove prefix http/https)
- Converte authors in lista
- Converte year in int
- Rimuove valori None

#### DuplicateDetector

Identifica duplicati basandosi su:

- DOI identico (case-insensitive)
- Title identico (case-insensitive, trimmed)

#### Batch Validation

```python
from src.core.investigation_validators import validate_papers_batch

papers = [{"title": "My Paper", "doi": "10.1234/abc"}, ...]
validated_papers, validation_results = validate_papers_batch(papers)

# Check results
for result in validation_results:
    if not result.passed and result.level == ValidationLevel.CRITICAL:
        print(f"REJECTED: {result.message}")
```

## 🔗 Integration with InvestigationAgent

Il sistema è integrato automaticamente nell'InvestigationAgent:

### 1. **Logger Initialization**

```python
# Auto-initialized in investigate_streaming()
investigation_id = str(uuid.uuid4())
self._logger = InvestigationLogger(investigation_id, query)
```

### 2. **Step Tracking**

Ogni step principale è tracciato:

- `PARSE`: Query parsing
- `RESOLVE`: Location resolution
- `PAPERS_COLLECT`: Paper search
- `PAPERS_VALIDATE`: Validation & sanitization
- `PAPERS_STORE`: Storage in SurrealDB
- `CACHE_STORE`: Data caching
- `COMPLETE`: Investigation finished

### 3. **Validation Pipeline**

Papers passano attraverso 3 fasi:

```
Search → Validate → Store
   ↓         ↓        ↓
 Papers → Valid Papers → SurrealDB
```

### 4. **Health Checks**

Dopo ogni operazione critica:

```python
logger.log_health_check("knowledge_base", True, {
    "papers_stored": count,
    "backend": "surrealdb"
})
```

## 📈 Monitoring & Metrics

### Log Output Example

```json
{
  "timestamp": "2025-12-24T12:00:00Z",
  "level": "info",
  "event": "step_completed",
  "investigation_id": "abc-123",
  "step": "papers_validate",
  "duration_ms": 150.5,
  "success": true,
  "metrics": {
    "original_count": 20,
    "validated_count": 18,
    "rejected_count": 2,
    "warnings": 5
  }
}
```

### Summary Report

```json
{
  "investigation_id": "abc-123",
  "query": "Lago Maggiore 2000 floods",
  "location": "Lago Maggiore",
  "event_type": "flood",
  "total_duration_ms": 5230.2,
  "steps_completed": ["parse", "papers_collect", "papers_validate", "papers_store"],
  "steps_failed": [],
  "success_rate": 1.0,
  "metrics": {
    "papers_found": 20,
    "papers_validated": 18,
    "papers_stored": 18
  }
}
```

## 🐛 Debugging

### Check Logs

```python
# In Python logger output
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run investigation and check console
```

### Analyze Validation Results

```python
for result in validation_results:
    if not result.passed:
        print(f"{result.level}: {result.validator} - {result.message}")
        if result.details:
            print(f"  Details: {result.details}")
```

### Health Check Status

```python
# Check if papers were successfully stored
logger.log_health_check("knowledge_base", healthy, details)
```

## 🔧 Configuration

### Logging Levels

- `DEBUG`: Detailed metrics and intermediate steps
- `INFO`: Normal operation progress
- `WARNING`: Issues that don't prevent completion
- `ERROR`: Failed operations
- `CRITICAL`: System-level failures

### Validation Levels

- `CRITICAL`: Must pass - paper rejected if fails
- `WARNING`: Should pass - logged but not rejected
- `INFO`: Nice to have - informational only

## 📊 Best Practices Applied

### From Apache Airflow

- ✅ Structured logging with consistent format
- ✅ Step-by-step progress tracking
- ✅ Performance metrics per operation
- ✅ Health checks for external dependencies
- ✅ Error context preservation

### From Great Expectations

- ✅ Validation as "expectations" (unit tests for data)
- ✅ Severity levels (critical/warning/info)
- ✅ Batch validation with detailed results
- ✅ Data sanitization before storage
- ✅ Duplicate detection

## 🚀 Future Enhancements

1. **Metrics Export**: Prometheus/Grafana integration
2. **Alert System**: Email/Slack notifications on failures
3. **Audit Trail**: Store all logs in database
4. **Dashboard**: Real-time investigation monitoring
5. **Replay**: Re-run failed investigations with saved context

## 📝 Notes

- Logger is **optional** - system works without it (graceful degradation)
- Validation can be **disabled** for testing (papers used as-is)
- All metrics are **JSON-serializable** for easy export
- Health checks are **non-blocking** - failures are logged but don't stop pipeline

---

**Status**: ✅ Production-ready | **Version**: 1.0.0 | **Last Updated**: 2025-12-24

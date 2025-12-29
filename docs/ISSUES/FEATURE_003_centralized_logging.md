# 🔧 Centralized Logging System

## Issue Type
Feature / Infrastructure

## Summary
Implemented a centralized logging system with best practices for debugging across the entire codebase.

## Problem
- No consistent logging across components
- Errors only visible via screenshots
- No log files for post-mortem analysis
- No structured logging for production

## Solution
Created `src/core/logging_config.py` with:

### Features
1. **Multiple Formatters**
   - `ColoredFormatter` - Colorized console for development
   - `JSONFormatter` - Structured JSON for production/log aggregation
   - `StreamlitFormatter` - Clean output for Streamlit apps

2. **File Handlers**
   - Rotating log files (10MB max, 5 backups)
   - Separate error log file
   - JSON format for easy parsing

3. **Decorators**
   - `@log_call()` - Log function calls with args/results
   - `@log_errors()` - Catch and log exceptions with context

4. **Context Manager**
   - `LogContext` - Log operations with timing

### Usage
```python
from src.core.logging_config import setup_logging, get_logger, LogContext

# Setup
setup_logging(level="DEBUG", env="development")
logger = get_logger(__name__)

# Basic logging
logger.info("Processing gate", extra={"gate_id": "fram_strait"})
logger.error("Failed to load", exc_info=True)

# With context
with LogContext(logger, "Loading data", gate_id="fram"):
    load_data()

# Decorator
@log_errors()
def risky_function():
    ...
```

### Environment Variables
- `NICO_ENV` - "development", "production", "streamlit"
- `NICO_LOG_LEVEL` - "DEBUG", "INFO", "WARNING", "ERROR"

### Log Locations
- `logs/nico.log` - All logs (JSON)
- `logs/nico_errors.log` - Errors only (JSON)

## Files Created
- `src/core/logging_config.py`

## Files Modified
- `app/main.py` - Integrated logging
- `.gitignore` - Added `*.log`

## Status
✅ Implemented

## Related
- Issue #12: Unified Architecture Refactoring

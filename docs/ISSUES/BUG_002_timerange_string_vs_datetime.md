# 🐛 TimeRange String vs DateTime Type Error

## Issue Type
Bug / Type Mismatch

## Summary
`TimeRange` model stores `start` and `end` as ISO strings internally (due to Pydantic serialization), but code attempted datetime arithmetic directly.

## Error Message
```
TypeError: unsupported operand type(s) for -: 'str' and 'str'

File "data_selector.py", line 499, in _render_confirmation
    time_days = (selection.time_range.end - selection.time_range.start).days
```

## Root Cause
The `TimeRange` Pydantic model serializes datetime objects to ISO strings. When accessing `.start` and `.end`, they may be strings instead of `datetime` objects depending on context.

## Solution
Added type checking before arithmetic:
```python
time_days = 0
if selection.time_range:
    try:
        start = selection.time_range.start
        end = selection.time_range.end
        # Handle both datetime and string
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace('Z', '+00:00'))
        time_days = (end - start).days
    except Exception:
        time_days = 30  # Default fallback
```

## Files Modified
- `app/components/data_selector.py`

## Prevention
- Use `@validator` in Pydantic to ensure consistent types
- Add `model_validator` to always return datetime
- Consider using `pendulum` library for robust datetime handling

## Status
✅ Fixed

## Related
- Issue #12: Unified Architecture Refactoring

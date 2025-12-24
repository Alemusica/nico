# API Versioning Strategy

## Overview

The Causal Discovery API uses URL-based versioning with the `/api/v{version}` prefix pattern.

## Current Version

**Version 1 (v1)** - Current stable API
- Base path: `/api/v1`
- Status: Active
- Endpoints: All current endpoints

## Versioning Rules

### 1. Breaking Changes
Breaking changes require a new major version:
- Removing endpoints
- Changing required fields
- Modifying response structure
- Changing authentication method

**Examples:**
```python
# Breaking: Renamed field
# v1
{"dataset_name": "fram_strait"}
# v2
{"name": "fram_strait"}  # ❌ Breaking change

# Non-breaking: Added optional field
# v1
{"dataset_name": "fram_strait"}
# v1
{"dataset_name": "fram_strait", "metadata": {...}}  # ✅ Non-breaking
```

### 2. Non-Breaking Changes
Non-breaking changes can be added to existing versions:
- Adding optional fields
- Adding new endpoints
- Adding optional parameters
- Deprecation warnings

### 3. Version Support Policy
- **Current version (v1)**: Full support, active development
- **Previous version (v0)**: Not applicable (v1 is first versioned release)
- **Future versions**: Maintain N-1 version (support previous major version)
- **Deprecation period**: 6 months minimum before removal

## Endpoint Structure

### Format
```
{protocol}://{host}/api/v{version}/{resource}/{action}
```

### Examples
```bash
# Health check
GET https://api.example.com/api/v1/health

# List datasets
GET https://api.example.com/api/v1/data/datasets

# Causal discovery
POST https://api.example.com/api/v1/analysis/discover

# Chat
POST https://api.example.com/api/v1/chat
```

## Version-Specific Routers

Each router is mounted with version prefix in `api/main.py`:

```python
API_V1_PREFIX = "/api/v1"

app.include_router(analysis_router, prefix=API_V1_PREFIX)
app.include_router(chat_router, prefix=API_V1_PREFIX)
app.include_router(data_router, prefix=API_V1_PREFIX)
app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(investigation_router, prefix=API_V1_PREFIX)
app.include_router(knowledge_router, prefix=API_V1_PREFIX)
app.include_router(pipeline_router, prefix=API_V1_PREFIX)
```

## Version Negotiation

### Default Version
Requests without version prefix are **not supported** (explicit versioning required).

```bash
# ❌ Invalid - no version
GET https://api.example.com/data/datasets

# ✅ Valid - explicit version
GET https://api.example.com/api/v1/data/datasets
```

### Version Header (Future)
In future releases, support for version negotiation via headers:

```bash
GET https://api.example.com/data/datasets
Header: API-Version: 1
```

## Migration Guide

### Client Migration (v1 → v2)
When v2 is released, clients should:

1. **Review changelog**: Check `/docs/CHANGELOG.md` for breaking changes
2. **Update base URL**: Change from `/api/v1` to `/api/v2`
3. **Test thoroughly**: Validate all endpoints with v2
4. **Gradual rollout**: Deploy to staging first
5. **Monitor deprecation warnings**: Watch for deprecation notices in v1

### Example Migration
```python
# Before (v1)
import requests

BASE_URL = "https://api.example.com/api/v1"

response = requests.get(f"{BASE_URL}/data/datasets")

# After (v2)
BASE_URL = "https://api.example.com/api/v2"

response = requests.get(f"{BASE_URL}/data/datasets")
# Check for schema changes in response
```

## Deprecation Process

### Step 1: Announce (v1.x)
Add deprecation warnings to affected endpoints:
```python
@router.get("/old-endpoint")
async def old_endpoint():
    """
    ⚠️ DEPRECATED: Use /new-endpoint instead.
    This endpoint will be removed in v2.
    Deprecation date: 2024-12-24
    Removal date: 2025-06-24
    """
    return {"warning": "This endpoint is deprecated"}
```

### Step 2: Document (CHANGELOG.md)
```markdown
## v1.5.0 (2024-12-24)

### Deprecated
- `/api/v1/old-endpoint` - Use `/api/v1/new-endpoint` instead
  - Removal scheduled: v2.0.0 (2025-06-24)
```

### Step 3: Remove (v2.0)
Endpoint removed from codebase, returns 410 Gone:
```python
@router.get("/old-endpoint")
async def old_endpoint_removed():
    raise HTTPException(
        status_code=410,
        detail="This endpoint was removed in v2.0. Use /new-endpoint."
    )
```

## Version-Specific Features

### v1 (Current)

**Features:**
- Data management (upload, list, cache)
- Causal discovery (PCMCI, correlation)
- Root cause analysis (Ishikawa, FMEA, 5-Why)
- Knowledge base (papers, events, patterns)
- Investigation agent (WebSocket)
- LLM chat interface
- Rate limiting
- Structured logging
- Request ID tracking

**Base Path:** `/api/v1`

**Status:** Active development

## Future Versions

### v2 (Planned)

**Possible Breaking Changes:**
- Authentication required (JWT tokens)
- Renamed fields for consistency
- Pagination for all list endpoints
- GraphQL endpoint (alternative to REST)

**Base Path:** `/api/v2`

**Status:** Not yet released

## Documentation

- **API Usage Guide**: `/docs/API_USAGE.md`
- **OpenAPI Schema**: `/api/v1/openapi.json`
- **Interactive Docs**: `/docs` (Swagger UI)
- **Alternative Docs**: `/redoc` (ReDoc)

## Version Discovery

Clients can discover supported API versions via health endpoint:

```bash
GET /api/v1/health
```

Response includes version info:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "api_version": "v1",
  "supported_versions": ["v1"],
  "latest_version": "v1"
}
```

## Best Practices

### For API Consumers

1. **Always use explicit versions** in production
2. **Pin to specific version** in client configuration
3. **Monitor deprecation warnings** in response headers
4. **Test against new versions** before migration
5. **Subscribe to changelogs** for breaking changes

### For API Developers

1. **Never break existing versions** without major version bump
2. **Document all changes** in CHANGELOG.md
3. **Provide migration guides** for breaking changes
4. **Maintain backwards compatibility** when possible
5. **Test version-specific behavior** thoroughly

## Questions & Support

- **Documentation**: `/docs/API_USAGE.md`
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@causaldiscovery.io

---

**Current Version**: v1  
**Last Updated**: 2024-12-24

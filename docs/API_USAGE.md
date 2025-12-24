# API Usage Guide

Comprehensive guide for using the Causal Discovery API.

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Core Workflows](#core-workflows)
- [Endpoints Reference](#endpoints-reference)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Quick Start

### Starting the Server

```bash
# Development mode
uvicorn api.main:app --reload --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Basic Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "llm": {
      "status": "available",
      "model": "llama3.1:8b-instruct-q8_0"
    },
    "causal_discovery": {
      "status": "available",
      "method": "PCMCI"
    },
    "databases": {
      "neo4j": "connected",
      "surrealdb": "available"
    }
  },
  "robustness": "All components have fallbacks - system operational"
}
```

## Authentication

Currently using development mode without authentication.

**Coming soon**: JWT-based authentication.

```bash
# Future: Include bearer token
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/data/datasets
```

## Core Workflows

### 1. Investigation Workflow

**Step 1: Check investigation status**
```bash
curl http://localhost:8000/investigation/status
```

**Step 2: Get current briefing**
```bash
curl http://localhost:8000/investigation/briefing
```

Response:
```json
{
  "region": "Fram Strait",
  "timeframe": "2020-01-01 to 2023-12-31",
  "question": "What caused the anomalous warm water inflow event in January 2022?",
  "datasets_available": ["CMEMS", "ERA5", "AVISO"],
  "variables_of_interest": ["sea_surface_temperature", "salinity", "sea_level_anomaly"]
}
```

**Step 3: Real-time investigation via WebSocket**
```python
import asyncio
import websockets
import json

async def investigate():
    uri = "ws://localhost:8000/investigation/ws"
    async with websockets.connect(uri) as websocket:
        # Send investigation request
        await websocket.send(json.dumps({
            "query": "Analyze temperature anomaly in Fram Strait January 2022",
            "sources": ["CMEMS", "ERA5"],
            "time_range": ["2021-12-01", "2022-02-28"]
        }))
        
        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            print(f"{data['stage']}: {data['message']}")
            
            if data.get("complete"):
                print("Investigation complete!")
                break

asyncio.run(investigate())
```

### 2. Data Management Workflow

**List available datasets**
```bash
curl http://localhost:8000/data/datasets
```

Response:
```json
{
  "datasets": [
    {
      "name": "CMEMS_FRAM_STRAIT_2020_2023",
      "source": "CMEMS",
      "time_range": ["2020-01-01", "2023-12-31"],
      "variables": ["temperature", "salinity", "velocity_u", "velocity_v"],
      "resolution": "daily",
      "size_mb": 450.5
    }
  ],
  "count": 1
}
```

**Upload custom dataset**
```bash
curl -X POST http://localhost:8000/data/upload \
  -F "file=@my_data.csv" \
  -F "dataset_name=custom_fram_strait" \
  -F "source=local"
```

**Get dataset statistics**
```bash
curl http://localhost:8000/data/datasets/CMEMS_FRAM_STRAIT_2020_2023/stats
```

### 3. Causal Analysis Workflow

**Step 1: Generate hypotheses**
```bash
curl -X POST http://localhost:8000/analysis/hypotheses \
  -H "Content-Type: application/json" \
  -d '{
    "observation": "Sharp temperature increase in Fram Strait January 2022",
    "context": ["North Atlantic Oscillation positive phase", "Reduced sea ice coverage"]
  }'
```

Response:
```json
{
  "hypotheses": [
    {
      "id": 1,
      "description": "NAO positive phase increased warm Atlantic water inflow",
      "likelihood": "high",
      "mechanisms": ["atmospheric pressure gradients", "wind-driven currents"],
      "testable_variables": ["NAO_index", "wind_stress", "ocean_velocity"]
    }
  ]
}
```

**Step 2: Discover causal links**
```bash
curl -X POST http://localhost:8000/analysis/discover \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "CMEMS_FRAM_STRAIT_2020_2023",
    "target": "temperature",
    "variables": ["NAO_index", "wind_stress", "salinity"],
    "lag_max": 30,
    "alpha": 0.05
  }'
```

Response:
```json
{
  "target": "temperature",
  "causal_links": [
    {
      "source": "wind_stress",
      "target": "temperature",
      "lag": 5,
      "strength": 0.72,
      "p_value": 0.001,
      "confidence": "high"
    }
  ],
  "method": "PCMCI",
  "computation_time_seconds": 12.5
}
```

**Step 3: Create Ishikawa diagram**
```bash
curl -X POST http://localhost:8000/analysis/ishikawa \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Warm water anomaly January 2022",
    "categories": ["atmospheric", "oceanic", "sea_ice"]
  }'
```

### 4. Knowledge Management Workflow

**Add scientific paper**
```bash
curl -X POST http://localhost:8000/knowledge/papers \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Atlantic Water Inflow variability in Fram Strait",
    "authors": ["Smith et al."],
    "year": 2023,
    "doi": "10.1234/oceanography.2023.001",
    "key_findings": [
      "NAO controls 60% of transport variability",
      "5-day lag between wind forcing and transport response"
    ],
    "relevant_variables": ["NAO_index", "ocean_transport", "wind_stress"]
  }'
```

**Search knowledge base**
```bash
curl http://localhost:8000/knowledge/papers/search?query=NAO+fram+strait&limit=5
```

**Link knowledge to causal pattern**
```bash
curl -X POST http://localhost:8000/knowledge/patterns \
  -H "Content-Type: application/json" \
  -d '{
    "pattern_type": "wind_driven_transport",
    "description": "NAO-driven Atlantic water transport",
    "typical_lag_days": 5,
    "strength_range": [0.6, 0.8],
    "evidence_papers": ["10.1234/oceanography.2023.001"]
  }'
```

### 5. Chat Interface Workflow

**Ask question with context**
```bash
curl -X POST http://localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why does NAO affect Fram Strait temperature?",
    "context": {
      "current_investigation": "fram_strait_2022_anomaly",
      "recent_findings": ["wind_stress -> temperature lag=5d"]
    },
    "include_sources": true
  }'
```

Response:
```json
{
  "answer": "The North Atlantic Oscillation (NAO) affects Fram Strait temperature through wind-driven ocean circulation. During positive NAO phases, enhanced westerly winds strengthen the northward transport of warm Atlantic water...",
  "sources": [
    "Smith et al. (2023): Atlantic Water Inflow variability",
    "Causal analysis: wind_stress -> temperature (lag=5d, p<0.001)"
  ],
  "confidence": 0.85
}
```

## Endpoints Reference

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Simple health check |
| `/health` | GET | Detailed component status |

### Investigation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/investigation/status` | GET | Current investigation state |
| `/investigation/briefing` | GET | Active investigation briefing |
| `/investigation/ws` | WebSocket | Real-time investigation updates |
| `/investigation/history` | GET | Past investigation results |

### Data Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/data/datasets` | GET | List all datasets |
| `/data/datasets/{name}` | GET | Get dataset details |
| `/data/datasets/{name}/stats` | GET | Dataset statistics |
| `/data/upload` | POST | Upload new dataset |
| `/data/sources` | GET | Available data sources |
| `/data/resolutions` | GET | Supported temporal/spatial resolutions |
| `/data/cache/stats` | GET | Cache usage statistics |
| `/data/cache/clear` | POST | Clear data cache |

### Causal Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analysis/discover` | POST | Run causal discovery (PCMCI) |
| `/analysis/hypotheses` | POST | Generate testable hypotheses |
| `/analysis/ishikawa` | POST | Create Ishikawa diagram |
| `/analysis/fmea` | POST | Failure Mode Effects Analysis |
| `/analysis/5why` | POST | 5-Why root cause analysis |
| `/analysis/interpret` | POST | Interpret causal graph |

### Knowledge Base

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/knowledge/papers` | GET/POST | List or add papers |
| `/knowledge/papers/{id}` | GET/PUT/DELETE | Manage specific paper |
| `/knowledge/papers/search` | GET | Search papers |
| `/knowledge/events` | GET/POST | Historical oceanographic events |
| `/knowledge/patterns` | GET/POST | Known causal patterns |
| `/knowledge/graph` | GET | Knowledge graph export |

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/ask` | POST | Ask question with context |
| `/chat/stream` | POST | Streaming chat response |

### Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pipeline/execute` | POST | Run full analysis pipeline |

## Error Handling

The API uses structured error responses with HTTP status codes:

### Error Response Format

```json
{
  "error_type": "DatasetNotFoundError",
  "message": "Dataset 'INVALID_NAME' not found",
  "details": {
    "dataset_name": "INVALID_NAME",
    "available_datasets": ["CMEMS_FRAM_STRAIT_2020_2023"]
  },
  "timestamp": "2024-12-24T16:08:00Z",
  "request_id": "a1b2c3d4"
}
```

### HTTP Status Codes

| Code | Error Type | Description |
|------|------------|-------------|
| 400 | `InvalidDataFormatError` | Malformed request data |
| 404 | `DatasetNotFoundError` | Resource not found |
| 422 | `InvalidVariableError` | Invalid input parameters |
| 500 | `CausalAnalysisError` | Server error during analysis |
| 503 | `LLMUnavailableError` | LLM service unavailable |
| 504 | `LLMTimeoutError` | Request timeout |

### Common Errors

**DatasetNotFoundError (404)**
```bash
curl http://localhost:8000/data/datasets/NONEXISTENT
# Response: 404 with available datasets
```

**InvalidDataFormatError (400)**
```bash
curl -X POST http://localhost:8000/analysis/discover -d '{}'
# Response: 400 with required fields
```

**LLMUnavailableError (503)**
```bash
curl -X POST http://localhost:8000/chat/ask -d '{"question": "test"}'
# Response: 503 with fallback info (system continues with rule-based fallback)
```

## Rate Limiting

**Coming soon**: Rate limiting with configurable thresholds.

Default limits (planned):
- 100 requests/minute per IP
- 1000 requests/hour per IP
- WebSocket: 10 concurrent connections per IP

Headers returned:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640000000
```

## Examples

### Complete Investigation Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Check system health
health = requests.get(f"{BASE_URL}/health").json()
print(f"System status: {health['status']}")

# 2. List available datasets
datasets = requests.get(f"{BASE_URL}/data/datasets").json()
print(f"Found {datasets['count']} datasets")

# 3. Generate hypotheses
hypotheses_req = {
    "observation": "Temperature spike in January 2022",
    "context": ["NAO positive phase", "Low sea ice"]
}
hypotheses = requests.post(
    f"{BASE_URL}/analysis/hypotheses",
    json=hypotheses_req
).json()
print(f"Generated {len(hypotheses['hypotheses'])} hypotheses")

# 4. Run causal discovery
discovery_req = {
    "dataset": datasets['datasets'][0]['name'],
    "target": "temperature",
    "variables": ["NAO_index", "wind_stress", "salinity", "sea_ice_concentration"],
    "lag_max": 30,
    "alpha": 0.05
}
causal_links = requests.post(
    f"{BASE_URL}/analysis/discover",
    json=discovery_req
).json()

print(f"\nFound {len(causal_links['causal_links'])} causal links:")
for link in causal_links['causal_links']:
    print(f"  {link['source']} -> {link['target']} (lag={link['lag']}d, p={link['p_value']:.4f})")

# 5. Save findings to knowledge base
for link in causal_links['causal_links']:
    pattern = {
        "pattern_type": f"{link['source']}_to_{link['target']}",
        "description": f"Causal link: {link['source']} influences {link['target']}",
        "typical_lag_days": link['lag'],
        "strength_range": [link['strength'] - 0.1, link['strength'] + 0.1],
        "discovery_date": "2024-12-24"
    }
    requests.post(f"{BASE_URL}/knowledge/patterns", json=pattern)

print("\n✅ Complete investigation workflow finished!")
```

### Batch Dataset Processing

```python
import os
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
DATA_DIR = Path("./my_datasets")

# Upload multiple CSV files
for csv_file in DATA_DIR.glob("*.csv"):
    with open(csv_file, 'rb') as f:
        files = {'file': f}
        data = {
            'dataset_name': csv_file.stem,
            'source': 'local'
        }
        response = requests.post(
            f"{BASE_URL}/data/upload",
            files=files,
            data=data
        )
        if response.status_code == 200:
            print(f"✅ Uploaded {csv_file.name}")
        else:
            print(f"❌ Failed to upload {csv_file.name}: {response.text}")

# List all uploaded datasets
datasets = requests.get(f"{BASE_URL}/data/datasets").json()
print(f"\nTotal datasets: {datasets['count']}")
for ds in datasets['datasets']:
    print(f"  - {ds['name']} ({ds['size_mb']:.1f} MB)")
```

### Knowledge Graph Export

```python
import requests
import json
import matplotlib.pyplot as plt
import networkx as nx

# Fetch knowledge graph
graph_data = requests.get("http://localhost:8000/knowledge/graph").json()

# Create NetworkX graph
G = nx.DiGraph()

# Add nodes
for node in graph_data['nodes']:
    G.add_node(node['id'], **node)

# Add edges
for edge in graph_data['edges']:
    G.add_edge(edge['source'], edge['target'], **edge)

# Visualize
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=2, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, arrows=True)
plt.title("Causal Knowledge Graph")
plt.savefig("knowledge_graph.png", dpi=300, bbox_inches='tight')
print("✅ Knowledge graph saved to knowledge_graph.png")
```

## Interactive API Documentation

Visit the auto-generated interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Support & Feedback

- Documentation: `/docs/`
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Next Steps

1. Set up authentication (JWT tokens)
2. Configure rate limiting
3. Set up monitoring and alerting
4. Review security best practices
5. Optimize for production workloads

---

**Version**: 1.0.0  
**Last Updated**: 2024-12-24

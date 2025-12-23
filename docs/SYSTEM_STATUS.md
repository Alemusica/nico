# 🔧 System Status & Dependencies

**Last Updated:** December 23, 2025

## 📦 Core Dependencies Status

| Package | Status | Purpose | Fallback |
|---------|--------|---------|----------|
| **tigramite** | ✅ Installed | PCMCI causal discovery | Correlation analysis |
| **neo4j** | ✅ Installed | Graph database client | In-memory NetworkX |
| **surrealdb** | ✅ Installed | Multi-model database | JSON file storage |
| **networkx** | ✅ Installed | Graph algorithms | Always available |
| **scipy** | ✅ Installed | Statistical tests | Always available |
| **numpy** | ✅ Installed | Numerical computing | Required |
| **xarray** | ✅ Installed | NetCDF handling | Required |

## 🗄️ Database Configuration

### Currently Active: **Hybrid Mode**

The system supports dual-database operation:

| Database | Port | Purpose | Status |
|----------|------|---------|--------|
| **Neo4j** | 7474 (HTTP), 7687 (Bolt) | Graph relationships, causal chains | Optional |
| **SurrealDB** | 8529 | Document storage, time-series | Optional |

### Running Without Docker

The system is designed to be **robust** and work without Docker:

```python
# Automatic fallback hierarchy:
1. Neo4j/SurrealDB (if Docker running)
2. SQLite local database
3. In-memory storage with JSON persistence
```

## 🔬 Causal Discovery Engine

### PCMCI (Tigramite) - Primary Method
- **Package:** tigramite 5.2.9.4
- **Tests:** ParCorr (linear), GPDC (non-linear)
- **Features:** Time-lagged cross-correlation with conditional independence

### Correlation Fallback - Secondary Method
When tigramite is unavailable:
- Scipy cross-correlation
- Lag detection via signal correlation
- Bootstrap significance testing

## 📊 Data Sources Integrated

| Source | Type | Coverage | Status |
|--------|------|----------|--------|
| SLCCI | Along-track altimetry | ≤66°N | ✅ Loader ready |
| CMEMS L3 | Along-track gridded | Global | ✅ Loader ready |
| CMEMS L4 | Daily gridded | 70-85°N Arctic | ✅ Loader ready |
| AVISO | Along-track | Global | ✅ Loader ready |
| TUM Geoid | Static grid | Global | ✅ Loader ready |

## 🚀 Starting the System

### Minimal (No Docker)
```bash
# Start API only (uses in-memory/SQLite)
cd /Users/alessioivoycazzaniga/nico
uvicorn api.main:app --port 8000

# Start frontend
cd frontend && npm run dev
```

### Full Stack (Docker)
```bash
docker-compose up -d
# Neo4j: http://localhost:7474
# API: http://localhost:8000
# Frontend: http://localhost:5173
```

## 🔌 API Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "llm_available": true/false,
  "databases": {
    "neo4j": "connected/fallback",
    "surrealdb": "connected/fallback"
  },
  "tigramite": "available/fallback"
}
```

## 📚 External Libraries Used

### From Grok/Research Recommendations:

1. **Tigramite** (jakobrunge/tigramite)
   - PCMCI algorithm for time-series causal discovery
   - Status: ✅ Installed and integrated
   
2. **NetworkX** 
   - Graph representation and algorithms
   - Status: ✅ Core dependency

3. **Neo4j Python Driver**
   - Graph database connectivity
   - Status: ✅ Installed, optional use

4. **SurrealDB Python SDK**
   - Multi-model database client
   - Status: ✅ Installed, optional use

### Not Yet Integrated:
- CausalNex (PyMC based)
- DoWhy (Microsoft causal inference)
- EconML (heterogeneous treatment effects)

## 🛡️ Robustness Features

1. **Graceful Degradation**
   - Missing tigramite → correlation fallback
   - Missing Neo4j → in-memory graphs
   - Missing SurrealDB → JSON file storage
   - Missing Ollama → rule-based explanations

2. **Health Monitoring**
   - `/health` endpoint checks all services
   - Automatic reconnection on failure
   - Clear status reporting

3. **Data Validation**
   - Physics validator checks correlation lags
   - Experience engine learns from validated patterns
   - Cross-correlation against known oceanography

## 📈 Performance Notes

- **Tigramite PCMCI:** ~2-5 seconds for 8 variables, 100 time points
- **Correlation fallback:** ~0.5 seconds same data
- **Neo4j queries:** <100ms for typical graph traversals
- **API response:** <200ms for most endpoints

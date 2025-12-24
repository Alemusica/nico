# 🚀 Quick Start: Fram Strait Experiment

Deploy the multi-layer causal discovery system and run your first experiment.

## Option 1: Local Development (Fastest)

### Prerequisites
```bash
# Python 3.11+
python --version

# Install dependencies
pip install -r requirements.txt

# Optional: Ollama for LLM features
brew install ollama && ollama pull llama2
```

### Run the Experiment
```bash
# Run Fram Strait multi-satellite tracking
python experiments/fram_strait_experiment.py
```

### Run the API (without databases)
```bash
# Start API with in-memory mode
uvicorn api.main:app --reload --port 8000

# Open API docs
open http://localhost:8000/docs
```

### Run Frontend
```bash
cd frontend
npm install
npm run dev

# Open frontend
open http://localhost:5173
```

---

## Option 2: Docker Deployment (Full Stack)

### Start Everything
```bash
# Build and run all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Access Points
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (neo4j/causalpass123)
- **SurrealDB**: http://localhost:8529

### Initialize Databases
```bash
# Run database initialization
docker-compose exec api python -c "
from api.services.knowledge_service import create_knowledge_service, KnowledgeBackend
import asyncio

async def init():
    # Neo4j
    neo4j = create_knowledge_service(KnowledgeBackend.NEO4J)
    await neo4j.connect()
    print('Neo4j connected and initialized')
    
    # SurrealDB
    surreal = create_knowledge_service(KnowledgeBackend.SURREALDB)
    await surreal.connect()
    print('SurrealDB connected and initialized')

asyncio.run(init())
"
```

---

## Option 3: Just Databases + Local Code

### Start only Neo4j and SurrealDB
```bash
docker-compose up -d neo4j surrealdb

# Wait for startup
sleep 10

# Run experiment locally
python experiments/fram_strait_experiment.py
```

---

## Running the Fram Strait Experiment

### What it does:
1. **Loads multi-satellite data** from Fram Strait (77-81°N, -10-15°E)
   - SLCCI Jason-1 (10-day repeat orbit)
   - CMEMS L3 (along-track) and L4 (gridded)
   
2. **Builds time series** for sea level anomaly at different temporal resolutions

3. **Calculates cross-correlations** with various lags to discover:
   - How do Jason-1 and CMEMS signals relate?
   - What lag exists between different products?
   
4. **Physics Engine validates** each correlation:
   - Is the lag physically plausible?
   - Does the sign match expected physics?
   - Is the causality direction correct?
   
5. **Experience Engine learns** patterns:
   - Builds confidence through repeated observations
   - Only promotes physics-valid patterns to "knowledge"

### Expected Output:
```
╔══════════════════════════════════════════════════╗
║      FRAM STRAIT EVOLUTION EXPERIMENT            ║
║  Multi-satellite tracking with Physics +         ║
║  Experience Engine validation                    ║
╚══════════════════════════════════════════════════╝

📍 Study Area: Fram Strait
   Lat: 77.0° - 81.0°
   Lon: -10.0° - 15.0°
   Controls ~50% of Arctic freshwater export

📡 STEP 1: Loading satellite data...
   SLCCI (Jason-1): 8 observations
   CMEMS (L3/L4): 12 observations
   Total: 20 observations

📊 STEP 2: Building time series...
   slcci_j1_sla: 8 points (2002-01 to 2002-03)
   cmems_l4_sla: 12 points (2020-01 to 2020-12)

🔗 STEP 3: Calculating cross-correlations...
   Calculated 156 lag-correlation pairs

⚡ STEP 4: Physics + Experience Engine validation...
   Physics validation:
     ✓ Valid: 89
     ✗ Invalid: 67
   
   Experience learning:
     📚 Learned patterns: 5

🏆 STEP 5: Best cross-satellite correlations:
   1. slcci_j1 ↔ cmems_l4
      Lag: 7 days, r=0.823
      Physics: 0.85, Experience: 0.72
```

---

## Adding Your Data

### Put data in the right places:
```
data/
├── slcci/           # SLCCI altimetry cycles
│   └── SLCCI_ALTDB_J1_Cycle*.nc
├── cmems/           # CMEMS products
│   ├── cmems_l3_*.nc
│   └── cmems_l4_*.nc
└── aviso/           # AVISO gridded
    └── aviso_*.nc
```

### Change study area (edit experiment):
```python
# In experiments/fram_strait_experiment.py

FRAM_STRAIT = {
    "name": "Your Study Area",
    "lat_min": 60.0,
    "lat_max": 70.0,
    "lon_min": -30.0,
    "lon_max": -10.0,
}
```

---

## API Endpoints for Results

### Store discovered patterns:
```bash
# Add a learned pattern to knowledge graph
curl -X POST http://localhost:8000/knowledge/patterns \
  -H "Content-Type: application/json" \
  -d '{
    "id": "fram_sla_lag_7d",
    "source_variable": "slcci_sla",
    "target_variable": "cmems_sla",
    "lag_days": 7,
    "strength": 0.823,
    "confidence": 0.85,
    "physics_valid": true,
    "discovery_method": "cross_correlation"
  }'
```

### Query causal chains:
```bash
# Find patterns affecting Fram Strait
curl "http://localhost:8000/knowledge/patterns/search?domain=fram_strait"
```

---

## Next Steps

1. **Add more data sources**: Sentinel-3, ICESat-2, Argo floats
2. **Extend study area**: Barents Sea, Denmark Strait
3. **Add climate indices**: NAO, AMO, AO correlations
4. **Build Agent layer**: Model physical processes as agents

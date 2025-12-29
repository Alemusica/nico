# 🤖 Copilot/AI Agent Instructions

## 🌿 BRANCH STRATEGY (READ FIRST!)

**Check which branch you're on:**
```bash
git branch --show-current
```

| Branch | Agent Doc | Focus |
|--------|-----------|-------|
| `master` | `docs/AGENT_FULLSTACK.md` | React + API + Knowledge Graph |
| `feature/gates-streamlit` | `docs/AGENT_GATES.md` | Streamlit + Gates + Dataset |

**📖 Read your agent doc BEFORE starting any work!**

See `docs/BRANCH_STRATEGY.md` for full details.

---

## 📋 TASK SYSTEM

**Before ANY task, read:**
1. Your branch-specific agent doc (see table above)
2. `docs/TASKS/CONTEXT.md` - Awareness del codice esistente
3. `docs/PROGRESS.md` - Stato attuale

## ⚠️ CRITICAL: Python Environment

**ALWAYS use the project virtual environment:**
```bash
source .venv/bin/activate
python -m <module>
```

**NEVER use:**
- `python3` (points to system Python)
- `pip3` (points to system pip)
- `pip install` without activating venv first

## 🚫 EXISTING CODE (Do NOT duplicate!)

- `src/data_manager/catalog.py` (736 lines) - CopernicusCatalog
- `src/surge_shazam/data/era5_client.py` - ERA5 client
- `src/surge_shazam/data/cmems_client.py` - CMEMS client
- `src/pattern_engine/causal/pcmci_engine.py` - PCMCI

## 🗄️ Database: SurrealDB

This project uses **SurrealDB** as the primary database, NOT Neo4j.

- **URL**: `ws://localhost:8001/rpc`
- **Namespace**: `causal`
- **Database**: `knowledge`
- **Auth**: Unauthenticated mode (no signin needed)

```python
from surrealdb import Surreal

db = Surreal("ws://localhost:8001/rpc")
db.use("causal", "knowledge")
# No signin needed - unauthenticated mode
```

## 🚀 Starting Services

Use the existing scripts:
```bash
./start.sh  # Starts everything
```

Or use VS Code tasks (Cmd+Shift+P → "Tasks: Run Task"):
- 🚀 Start All Services
- 🔧 Start API Only
- 🌐 Start Frontend Only

## 📦 Dependencies

All dependencies are already installed in `.venv`. 

**DO NOT reinstall packages.** If you get import errors:
1. Check you're using `.venv/bin/python`
2. Run `source .venv/bin/activate` first

## 🔌 Service Ports

| Service   | Port  | URL                          |
|-----------|-------|------------------------------|
| API       | 8000  | http://localhost:8000/docs   |
| Frontend  | 5173  | http://localhost:5173        |
| SurrealDB | 8001  | ws://localhost:8001/rpc      |
| Ollama    | 11434 | http://localhost:11434       |

## 📁 Project Structure

```
/api          - FastAPI backend
/frontend     - React + Vite frontend  
/src          - Core Python modules
/.venv        - Python virtual environment (USE THIS!)
/data         - Data files and cache
/scripts      - Utility scripts
```

## 🧪 Running Code

```bash
# Correct way
source .venv/bin/activate
python -c "from api.services.knowledge_service import *; print('OK')"

# Or use the venv python directly
.venv/bin/python -c "import api; print('OK')"
```

## 🐳 Docker Services

```bash
# Check running containers
docker ps

# Start SurrealDB if not running
docker start surrealdb
```

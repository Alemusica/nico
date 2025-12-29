# 🤖 Copilot/AI Agent Instructions

> ⚠️ **CRITICAL: READ THIS ENTIRE FILE FIRST** - Every agent, every session, every time.

---

## 🔄 STEP 0: MANDATORY GIT PULL (DO THIS FIRST!)

**ALWAYS execute these commands when starting ANY new session:**

```bash
cd /Users/nicolocaron/Documents/GitHub/nico
git fetch origin
git pull origin $(git branch --show-current)
```

**Why?** Multiple agents work on this repo. Without pulling first, you WILL have stale context and create conflicts.

---

## 📚 STEP 1: Required Reading Order

After pulling, read these docs IN THIS EXACT ORDER:

| # | File | What You Learn |
|---|------|----------------|
| 1 | `docs/PROGRESS.md` | Current state, what's done, what's broken |
| 2 | `docs/FEATURE_INVENTORY.md` | All features across all branches |
| 3 | `docs/CHAT_HISTORY.md` | Previous conversation context (if exists) |
| 4 | Your branch agent doc (see table below) | Branch-specific tasks |
| 5 | `docs/TASKS/CONTEXT.md` | Code awareness |

---

## 🌿 STEP 2: Branch Strategy

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

---

## 📊 Logging (IMPORTANT)

Use centralized logging from `src/core/logging_config.py`:

```python
from src.core.logging_config import get_logger, log_call, log_errors

logger = get_logger(__name__)

@log_call(logger)  # Auto-log function entry/exit
def my_function():
    logger.info("Processing", extra={"key": "value"})
    return result
```

**Log files:**
- `logs/nico.log` - All logs (JSON format)
- `logs/nico_errors.log` - Errors only

---

## 📝 When Creating New Features

1. **Check `docs/FEATURE_INVENTORY.md`** - Is it already implemented?
2. **Add to services layer first** - `src/services/`
3. **Use existing models** - `src/core/models.py`
4. **Document in FEATURE_INVENTORY** - Add your feature
5. **Update PROGRESS.md** - Track completion
6. **Create GitHub Issue** - For visibility

---

## ✅ Pre-Commit Checklist

Before EVERY commit:
- [ ] Run `git pull origin <branch>` first
- [ ] Update `docs/PROGRESS.md`
- [ ] Update `docs/FEATURE_INVENTORY.md` if new features
- [ ] Run tests: `.venv/bin/python -m pytest tests/ -v`
- [ ] Create GitHub Issue for bugs/features found

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                      │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Streamlit     │    │      React + Vite               │ │
│  │   (Port 8501)   │    │      (Port 5173)                │ │
│  └────────┬────────┘    └───────────────┬─────────────────┘ │
│           └──────────────┬──────────────┘                    │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   FastAPI Backend                      │  │
│  │                    (Port 8000)                         │  │
│  │  /api/v1/data, /api/v1/gates, /api/v1/knowledge       │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┼───────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVICES LAYER                          │
│  src/services/                                               │
│  ├── gate_service.py    - Gate operations                   │
│  ├── data_service.py    - Unified data loading              │
│  └── analysis_service.py - Analysis pipelines               │
└──────────────────────────┼───────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ SurrealDB    │  │ Intake       │  │ External APIs    │   │
│  │ (Port 8001)  │  │ catalog.yaml │  │ CMEMS, ERA5, etc │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

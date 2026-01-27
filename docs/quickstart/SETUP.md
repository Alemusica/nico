# 🔧 Project Setup for AI Agents & Developers

## 🐍 Python Environment

This project uses a **virtual environment** at `.venv/`. 

### DO THIS:
```bash
source .venv/bin/activate
python script.py
```

### DON'T DO THIS:
```bash
python3 script.py  # ❌ Uses system Python
pip3 install xyz   # ❌ Installs to system
```

## Why This Matters

All 100+ dependencies are already installed in `.venv`. Using system Python will:
1. Fail with "ModuleNotFoundError" 
2. Waste time reinstalling everything
3. Potentially break the system Python

## 🗄️ Database

**SurrealDB** (not Neo4j!) on `ws://localhost:8001`

Namespace: `causal`, Database: `knowledge`

## 🚀 Quick Start

```bash
./start.sh  # Starts API + Frontend + checks SurrealDB
```

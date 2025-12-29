# 🚀 Agent Full Stack - Istruzioni

> **Branch**: `master`  
> **Focus**: React Frontend, API FastAPI, Knowledge Graph, LLM Cockpit  
> **Last Updated**: 2025-12-29

---

## ⚠️ LEGGI PRIMA DI INIZIARE

Questo documento è per l'agent che lavora sul **progetto principale**.  
Se devi lavorare su **gates/dataset Streamlit**, vai su branch `feature/gates-streamlit` e leggi `docs/AGENT_GATES.md`.

---

## 🎯 Scope di questo Agent

### ✅ PUOI modificare:
```
frontend/           # React + Vite + Cosmograph
api/                # FastAPI backend
src/                # Core Python modules
scripts/            # Utility scripts
docs/               # Documentation (non AGENT_GATES.md)
```

### ❌ NON modificare:
```
gates/              # Shapefile gates (gestito da Agent Gates)
streamlit_app.py    # App Streamlit legacy (gestito da Agent Gates)
demo_dashboard.py   # Dashboard demo (gestito da Agent Gates)
```

---

## 📋 TODO Prioritizzati

### 🔴 ALTA PRIORITÀ
1. **Verificare React + Cosmograph** nel browser
   - URL: http://localhost:5173
   - Endpoint dati: http://localhost:8000/api/v1/knowledge/graph
   - Verificare che il grafo si renderizzi

2. **LLM Cockpit Commands** (da implementare)
   - "Expand geographically" → trova eventi in regioni adiacenti
   - "Find correlations" → LLM analizza nodi non collegati
   - "Show precursors" → mostra precursori climatici storici
   - File: `frontend/src/components/ChatPanel.tsx`

3. **Cosmograph con dati reali**
   - Connettere `/api/v1/knowledge/graph` a `CosmographView.tsx`
   - Verificare che nodes/links siano renderizzati

### 🟡 MEDIA PRIORITÀ
4. **Timeline slider** - animazione temporale nel grafo
5. **Drill-down grafo** - double-click espande nodi
6. **Test coverage** → 80%

### 🟢 BASSA PRIORITÀ
7. Export grafo (PNG, SVG)
8. Miglioramenti UX

---

## 🔧 Servizi da avviare

```bash
# Terminal 1: API FastAPI
cd /Users/alessioivoycazzaniga/nico
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2: React Frontend
cd /Users/alessioivoycazzaniga/nico/frontend
npm run dev  # oppure: npx vite --port 5173

# Terminal 3: SurrealDB (se non già running)
docker start surrealdb
```

### Verifica servizi:
```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/knowledge/stats
```

---

## 📂 Struttura Chiave

```
frontend/
├── src/
│   ├── components/
│   │   ├── CosmographView.tsx    # GPU-accelerated graph
│   │   ├── KnowledgeGraph3DView.tsx  # Three.js 3D graph
│   │   ├── ChatPanel.tsx         # LLM Cockpit
│   │   ├── CausalGraphView.tsx   # Causal relationships
│   │   └── PCMCIPanel.tsx        # PCMCI controls
│   ├── api.ts                    # API client (785 linee)
│   ├── config.ts                 # Endpoint configuration
│   └── store.ts                  # Zustand state
│
api/
├── main.py                       # FastAPI app
├── routers/
│   ├── knowledge_router.py       # /knowledge/* endpoints
│   ├── data_router.py            # /data/* endpoints
│   └── investigation_router.py   # WebSocket + briefing
└── services/
    ├── knowledge_service.py      # Abstract service
    └── surrealdb_knowledge.py    # SurrealDB implementation
```

---

## 🗄️ Database: SurrealDB

```python
# Connessione
from surrealdb import Surreal
db = Surreal("ws://localhost:8001/rpc")
db.use("causal", "knowledge")

# Dati attuali (seeded):
# - 6 events (Lago Maggiore 2000, Venice Acqua Alta, etc.)
# - 6 papers (flood research)
# - 5 patterns (NAO chain, SST teleconnection, etc.)
```

---

## 🔗 API Endpoints Principali

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/api/v1/knowledge/graph` | GET | Nodi e link per Cosmograph |
| `/api/v1/knowledge/stats` | GET | Statistiche knowledge base |
| `/api/v1/knowledge/events` | GET | Lista eventi |
| `/api/v1/knowledge/papers` | GET | Lista papers |
| `/api/v1/knowledge/patterns` | GET | Lista patterns causali |

---

## ⚠️ Known Issues

1. **Vite Proxy Timeout**: Le chiamate `/api/*` via porta 5173 vanno in timeout
   - Workaround: Il frontend usa fetch diretto a `localhost:8000`
   - CORS è configurato per `localhost:5173`

2. **HistoricalEvent mismatch**: La dataclass in `knowledge_service.py` ha campi diversi da quelli usati in `surrealdb_knowledge.py`
   - Da allineare in futuro

---

## 📚 Documentazione Correlata

- `docs/KNOWLEDGE_GRAPH_EXPLORER.md` - Roadmap Cosmograph
- `docs/ARCHITECTURE.md` - Architettura generale
- `docs/REFACTORING_ROADMAP.md` - Piano refactoring
- `docs/NEXT_STEPS_GRAPH_UX.md` - UX graph-centric

---

**Autore**: NICO Project  
**Branch**: master  
**Ultimo aggiornamento**: 2025-12-29

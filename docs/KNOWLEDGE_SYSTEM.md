# 🌊 Causal Discovery Dashboard

## Dual-Backend Knowledge Graph System
**Swiss Design • Blue-Emerald Oceanography Theme**

---

## 🗄️ Database Backends

### Neo4j
- **Vector Search**: HNSW indexes for paper embeddings (up to 4096 dimensions)
- **Graph Traversal**: Native Cypher queries for causal chains
- **Full-text Search**: Built-in search across papers and events
- **Connections**: Async driver with connection pooling

### SurrealDB  
- **Multi-Model**: Document + Graph + Vector in one database
- **LIVE SELECT**: Real-time subscriptions for pattern updates
- **Graph Edges**: Type-safe relationships with `->` operators
- **MTREE Vectors**: Fast similarity search with MTREE indexes

---

## 🎨 Frontend Design System

### Swiss Typography
- **Font**: Inter (display + body), JetBrains Mono (code)
- **Scale**: PHI-based (1.618 ratio) - 10px → 68px

### Color Palette
```
Blue (Ocean Deep)    #2563eb → Primary
Emerald (Ocean Life) #059669 → Secondary  
Cyan (Ocean Surface) #06b6d4 → Accent
Slate (Swiss Clean)  #0f172a → Dark
```

### PHI Spacing
```
phi-xs:  5px   (8 / 1.618)
phi-sm:  8px   (base)
phi-md:  13px  (8 × 1.618)
phi-lg:  21px  (13 × 1.618)
phi-xl:  34px  (21 × 1.618)
phi-2xl: 55px  (34 × 1.618)
phi-3xl: 89px  (55 × 1.618)
```

---

## 📁 Project Structure

```
api/
  services/
    knowledge_service.py    # Abstract base + dataclasses
    neo4j_knowledge.py      # Neo4j implementation
    surrealdb_knowledge.py  # SurrealDB implementation
    llm_service.py          # Ollama LLM integration
    causal_service.py       # PCMCI causal discovery
  main.py                   # FastAPI endpoints

frontend/
  src/
    components/
      Sidebar.tsx           # Navigation
      Header.tsx            # Backend toggle
      CausalGraphView.tsx   # D3.js force graph
      ChatPanel.tsx         # AI assistant
      DataPanel.tsx         # File browser
      KnowledgeSearch.tsx   # Papers/Events/Patterns
    api.ts                  # API client
    store.ts                # Zustand state
  tailwind.config.js        # Swiss design tokens
  
tests/
  test_backends.py          # Neo4j vs SurrealDB benchmark
```

---

## 🚀 Quick Start

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn api.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

### Databases

**Neo4j** (Docker):
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

**SurrealDB** (Docker):
```bash
docker run -d \
  --name surrealdb \
  -p 8000:8000 \
  surrealdb/surrealdb:latest \
  start --user root --pass root
```

---

## 📊 API Endpoints

### Knowledge Base
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/knowledge/papers` | POST | Add paper |
| `/knowledge/papers/search` | GET | Search papers |
| `/knowledge/events` | POST | Add event |
| `/knowledge/events/search` | GET | Search events |
| `/knowledge/patterns` | POST | Add pattern |
| `/knowledge/patterns/{id}/causal-chain` | GET | Get causal chain |
| `/knowledge/patterns/{id}/evidence` | GET | Get supporting evidence |
| `/knowledge/climate-indices` | GET | List indices |
| `/knowledge/stats` | GET | Get statistics |
| `/knowledge/compare` | GET | Compare backends |

### Discovery
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/discover` | POST | Run causal discovery |
| `/interpret` | POST | LLM interpretation |
| `/chat` | POST | Chat with AI |

---

## 🔬 Benchmark

Run the backend comparison:
```bash
python tests/test_backends.py
```

Output:
```
📊 BENCHMARK RESULTS
================================================================================
Operation            Backend      Count   Total (ms)   Avg (ms)   Ops/sec    Status
--------------------------------------------------------------------------------
write_papers         neo4j           50       245.32       4.91      203.8    ✅ OK
write_papers         surrealdb       50       312.45       6.25      160.0    ✅ OK

search_papers        neo4j           20        89.12       4.46      448.0    ✅ OK
search_papers        surrealdb       20       102.34       5.12      390.6    ✅ OK
```

---

## 🏗️ Schema

### Neo4j
```cypher
(:Paper {id, title, authors, abstract, doi, year, journal, keywords, embedding})
(:Event {id, name, description, event_type, start_date, end_date, location, severity})
(:Pattern {id, name, description, pattern_type, variables, lag_days, strength, confidence})
(:ClimateIndex {id, name, abbreviation, description, source_url, time_series})

(Paper)-[:VALIDATES {validation_type, confidence}]->(Pattern)
(Pattern)-[:OBSERVED_IN {correlation, lag_observed}]->(Event)
(Pattern)-[:CAUSES {mechanism, strength, lag_days}]->(Pattern)
(ClimateIndex)-[:CORRELATES_WITH {correlation, lag_months}]->(Pattern)
```

### SurrealDB
```sql
DEFINE TABLE paper SCHEMAFULL;
DEFINE TABLE event SCHEMAFULL;
DEFINE TABLE pattern SCHEMAFULL;
DEFINE TABLE climate_index SCHEMAFULL;

DEFINE TABLE validates TYPE RELATION FROM paper TO pattern;
DEFINE TABLE observed_in TYPE RELATION FROM pattern TO event;
DEFINE TABLE causes TYPE RELATION FROM pattern TO pattern;
DEFINE TABLE correlates_with TYPE RELATION FROM climate_index TO pattern;
```

---

## 📝 License

MIT

# 🤖 README per Agent AI

> **⚠️ LEGGI QUESTO DOCUMENTO PRIMA DI FARE QUALSIASI COSA**

---

## 🚦 Workflow Standard

```
1. Controlla il branch     →  git branch --show-current
2. Leggi il tuo agent doc  →  docs/AGENT_*.md
3. Verifica i servizi      →  docker ps && curl localhost:8000/health
4. Lavora sul tuo scope    →  NON toccare file di altri agent
5. Commit frequenti        →  git commit -m "feat/fix: descrizione"
```

---

## 🌿 Quale Branch? Quale Doc?

```bash
# Prima cosa: scopri dove sei
git branch --show-current
```

| Branch | Documento da leggere | Focus |
|--------|---------------------|-------|
| `master` | 📄 `docs/AGENT_FULLSTACK.md` | React + API + Knowledge Graph |
| `feature/gates-streamlit` | 📄 `docs/AGENT_GATES.md` | Streamlit + Gates + Dataset |

**📖 LEGGI IL TUO DOCUMENTO AGENT PRIMA DI INIZIARE!**

---

## ⚙️ Setup Ambiente

### Python Virtual Environment
```bash
# SEMPRE usare il venv del progetto
source .venv/bin/activate

# Verifica
which python  # Deve mostrare: .../nico/.venv/bin/python
```

**❌ MAI usare:**
- `python3` (punta al Python di sistema)
- `pip3` (punta al pip di sistema)
- `pip install` senza attivare venv prima

### Servizi Docker
```bash
# Verifica container attivi
docker ps

# Se SurrealDB non è attivo:
docker start surrealdb
```

---

## 🔌 Porte e Servizi

| Servizio | Porta | URL | Stato |
|----------|-------|-----|-------|
| API FastAPI | 8000 | http://localhost:8000/docs | `curl localhost:8000/health` |
| React Frontend | 5173 | http://localhost:5173 | `curl localhost:5173` |
| SurrealDB | 8001 | ws://localhost:8001/rpc | `docker ps \| grep surreal` |
| Streamlit | 8501/8502 | http://localhost:8501 | `streamlit run ...` |
| Ollama LLM | 11434 | http://localhost:11434 | `ollama list` |

---

## 🚀 Avviare i Servizi

### Tutto insieme
```bash
./start.sh
```

### Singolarmente
```bash
# API
cd /Users/alessioivoycazzaniga/nico
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# React Frontend
cd frontend && npm run dev

# Streamlit
streamlit run demo_dashboard.py --server.port 8502
```

---

## 📂 Struttura Progetto

```
nico/
├── api/                 # 🔴 FastAPI backend (Agent Full Stack)
├── frontend/            # 🔴 React + Vite (Agent Full Stack)
├── src/                 # 🔴 Core Python modules (Agent Full Stack)
├── gates/               # 🟢 Shapefile oceanografici (Agent Gates)
├── streamlit_app.py     # 🟢 App Streamlit (Agent Gates)
├── demo_dashboard.py    # 🟢 Dashboard demo (Agent Gates)
├── data/                # 🟡 Dataset (entrambi, con cautela)
├── notebooks/           # 🟡 Jupyter notebooks (entrambi)
├── docs/                # 📚 Documentazione
│   ├── AGENT_FULLSTACK.md
│   ├── AGENT_GATES.md
│   └── BRANCH_STRATEGY.md
└── .venv/               # Python virtual environment
```

**Legenda:**
- 🔴 Solo Agent Full Stack (master)
- 🟢 Solo Agent Gates (feature/gates-streamlit)
- 🟡 Condiviso (attenzione ai conflitti)

---

## 🗄️ Database: SurrealDB

**NON Neo4j!** Questo progetto usa SurrealDB.

```python
from surrealdb import Surreal

async def connect():
    db = Surreal("ws://localhost:8001/rpc")
    await db.connect()
    await db.use("causal", "knowledge")
    # NO signin - modalità non autenticata
    return db
```

---

## ✅ Checklist Prima di Lavorare

- [ ] Ho verificato il branch: `git branch --show-current`
- [ ] Ho letto il mio agent doc in `docs/`
- [ ] Ho attivato il venv: `source .venv/bin/activate`
- [ ] I servizi sono attivi: `docker ps`
- [ ] So quali file posso modificare (vedi scope nel mio doc)

---

## 🔄 Workflow Git

### Commit
```bash
# Commit frequenti e descrittivi
git add <files>
git commit -m "feat(scope): descrizione breve"

# Esempi:
# feat(react): add Cosmograph clustering
# fix(api): correct knowledge endpoint
# feat(streamlit): add gate selector
# docs: update agent instructions
```

### Sincronizzazione
```bash
# Prima di iniziare, aggiorna
git pull origin <tuo-branch>

# Dopo aver finito
git push origin <tuo-branch>
```

---

## ⚠️ Regole d'Oro

1. **NON installare pacchetti** - sono già in `.venv`
2. **NON modificare file fuori dal tuo scope** - vedi agent doc
3. **NON usare python3/pip3** - usa il venv
4. **COMMIT frequenti** - piccoli e atomici
5. **TESTA prima di committare** - verifica che funzioni

---

## 🆘 Troubleshooting

### Import Error
```bash
# Hai attivato il venv?
source .venv/bin/activate
which python  # Deve essere .venv/bin/python
```

### SurrealDB non risponde
```bash
docker start surrealdb
docker logs surrealdb
```

### API non risponde
```bash
# Verifica che non ci siano altri processi sulla porta
lsof -i :8000
# Se occupata, killa il processo
kill -9 <PID>
```

### Frontend non parte
```bash
cd frontend
npm install  # Solo se mancano node_modules
npm run dev
```

---

## 📚 Documentazione Completa

| Documento | Descrizione |
|-----------|-------------|
| `README.md` | Overview progetto (per umani) |
| `README_AGENT.md` | **Questo file** (per agent AI) |
| `docs/AGENT_FULLSTACK.md` | Istruzioni Agent Full Stack |
| `docs/AGENT_GATES.md` | Istruzioni Agent Gates |
| `docs/BRANCH_STRATEGY.md` | Strategia branch paralleli |
| `QUICKSTART.md` | Setup rapido |

---

**Buon lavoro! 🚀**

*Ultimo aggiornamento: 2025-12-29*

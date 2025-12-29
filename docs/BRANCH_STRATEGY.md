# 🌿 Branch Strategy - NICO Project

> **Ultimo aggiornamento**: 2025-12-29

---

## 📊 Overview

Questo progetto usa **due branch paralleli** per permettere a più agent/sviluppatori di lavorare simultaneamente senza conflitti.

```
                    ┌─────────────────────────────────────┐
                    │              master                  │
                    │   React + API + Knowledge Graph      │
                    │   (Agent Full Stack)                 │
                    └─────────────────────────────────────┘
                                      │
                                      │ branch
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      feature/gates-streamlit         │
                    │   Streamlit + Gates + Dataset        │
                    │   (Agent Gates)                      │
                    └─────────────────────────────────────┘
```

---

## 🔀 Branch: `master`

### Focus
- **React Frontend** (Cosmograph, 3D visualization)
- **FastAPI Backend** (Knowledge Graph, LLM)
- **Core Modules** (pattern engine, data manager)

### Agent Doc
📄 `docs/AGENT_FULLSTACK.md`

### Cartelle Gestite
```
frontend/           ✅
api/               ✅
src/               ✅
scripts/           ✅
```

### Cartelle OFF-LIMITS
```
gates/             ❌
streamlit_app.py   ❌
demo_dashboard.py  ❌
```

---

## 🔀 Branch: `feature/gates-streamlit`

### Focus
- **Streamlit App** (UI per analisi oceanografica)
- **Gates Analysis** (shapefile, flussi)
- **Dataset Selection** (catalogo multi-provider)

### Agent Doc
📄 `docs/AGENT_GATES.md`

### Cartelle Gestite
```
gates/             ✅
streamlit_app.py   ✅
demo_dashboard.py  ✅
notebooks/         ✅
data/              ✅ (con cautela)
```

### Cartelle OFF-LIMITS
```
frontend/          ❌
api/               ❌
src/               ❌
```

---

## 🔄 Workflow di Merge

### Quando fare merge?

1. **Gates → Master**: Quando la feature Streamlit è completa e testata
2. **Master → Gates**: Per prendere aggiornamenti critici (API changes)

### Come fare merge:

```bash
# Da master, per prendere gates
git checkout master
git merge feature/gates-streamlit

# Da gates, per aggiornare da master
git checkout feature/gates-streamlit
git merge master
```

### Risoluzione Conflitti

I conflitti dovrebbero essere **rari** se ogni agent rispetta il proprio scope.  
File potenzialmente in conflitto:
- `requirements.txt` - aggiungere dipendenze in sezioni separate
- `pyproject.toml` - idem
- `.github/copilot-instructions.md` - sezione specifica per branch

---

## 📋 Checklist Pre-Merge

### Agent Full Stack (master)
- [ ] `npm run build` in frontend/ passa
- [ ] `pytest tests/` passa
- [ ] API risponde su :8000
- [ ] Nessun file in gates/ modificato

### Agent Gates (feature/gates-streamlit)
- [ ] `streamlit run streamlit_app.py` funziona
- [ ] Shapefile validi in gates/
- [ ] Nessun file in frontend/ modificato
- [ ] Nessun file in api/ modificato

---

## 🏷️ Convenzioni Commit

### Master
```
feat(react): add Cosmograph clustering
fix(api): correct knowledge endpoint
docs(fullstack): update agent instructions
```

### Gates Branch
```
feat(streamlit): add gate selector
feat(gates): new Arctic gates
fix(data): correct catalog path
docs(gates): update analysis workflow
```

---

## 📊 Status Tracker

| Branch | Status | Agent | Focus |
|--------|--------|-------|-------|
| `master` | 🟢 Active | Full Stack | React+API |
| `feature/gates-streamlit` | 🟡 Ready | Gates | Streamlit |

---

## 📞 Comunicazione tra Agent

Gli agent **NON** comunicano direttamente, ma attraverso:

1. **Commit messages** - descrittivi e dettagliati
2. **Questo documento** - aggiornato quando serve
3. **Issue/PR** - per task cross-branch

---

**Autore**: NICO Project  
**Documento**: docs/BRANCH_STRATEGY.md

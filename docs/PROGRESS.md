# 📊 Surge Shazam - Progress Tracker

> Last Updated: 2024-12-28
> Agent: Use this file to track progress. Update after each task.

---

## 🧠 Pre-Task: Awareness Check

**PRIMA di ogni task, verifica:**
- [ ] Letto `docs/TASKS/CONTEXT.md`?
- [ ] Verificato codice esistente?
- [ ] Usando `.venv/bin/python`?

---

## 🎯 Phase 1: Catalog Foundation

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 1.1 catalog.yaml | ✅ DONE | 2024-12-28 | 2024-12-28 | 11 datasets, latency badges |
| 1.2 intake_bridge.py | ✅ DONE | 2024-12-28 | 2024-12-28 | Bridge + search + summary |
| 1.3 catalog API | ✅ DONE | 2024-12-28 | 2024-12-28 | 4 endpoints added |

## 🛰️ Phase 2: CYGNSS Client (PARALLELIZZABILE)

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 2.1 cygnss_client.py | ✅ DONE | 2024-12-28 | 2024-12-28 | HIGH priority, NASA near-RT |

## 🔗 Phase 3: Causal Graph (PARALLELIZZABILE)

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 3.1 causal_graph.py | ✅ DONE | 2024-12-28 | 2024-12-28 | SurrealDB + 4 known chains |

---

## 📝 Progress Log

### 2024-12-28 - Task 3.1 Causal Graph
**Status**: ✅ DONE
**What was done**:
- Created `src/data_manager/causal_graph.py`
- CausalEdge dataclass + CausalGraphDB async class
- 4 KNOWN_CAUSAL_CHAINS pre-seeded (NAO→precipitation, precipitation→runoff, wind→SST, SLCCI→SLA r=0.866)
- Methods: add_edge, get_precursors, get_effects, get_causal_chain

**Blockers**:
- Docker not running (runtime test skipped)

**Next**:
- Integration with pcmci_engine.py

---

### 2024-12-28 - Task 2.1 CYGNSS Client
**Status**: ✅ DONE
**What was done**:
- Created `src/surge_shazam/data/cygnss_client.py`
- CYGNSS L3 Global Daily V3.1 client using `earthaccess`
- Search granules, download to xarray.Dataset
- Latency: 2-24h (NASA PO.DAAC)

**Blockers**:
- None

**Next**:
- Task 3.1 causal_graph.py

---

### 2025-12-29 - Unified Architecture Planning
**Status**: ✅ DONE
**What was done**:
- Created `docs/ROADMAP_UNIFIED_ARCHITECTURE.md` - Full refactoring plan
- Created `docs/GATES_CATALOG.md` - Gates documentation
- Created `docs/MODELS.md` - Pydantic models reference
- Created `docs/ISSUES/ISSUE_001_unified_architecture.md` - GitHub issue
- Updated `docs/ARCHITECTURE.md` with v2.0 section

**Blockers**:
- None

**Next**:
- Phase 1: Create `src/core/models.py`
- Phase 2: Create `config/` directory with YAML files
- Phase 3: Create `src/gates/` module

---

### [DATE] - Task X.X
**Status**: ✅ / ❌ / 🔄
**What was done**:
- ...

**Blockers**:
- ...

**Next**:
- ...

---

## 🏗️ Architecture Refactoring (v2.0)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Documentation | ✅ DONE | Roadmap, Models, Gates docs |
| Phase 1: Core Models | ⬜ TODO | `src/core/models.py` |
| Phase 2: Config | ⬜ TODO | `config/` directory |
| Phase 3: Gates Module | ⬜ TODO | `src/gates/` |
| Phase 4: Services | ⬜ TODO | `src/services/` |
| Phase 5: API | ⬜ TODO | `gates_router.py` |
| Phase 6: Streamlit | ⬜ TODO | Refactor sidebar |
| Phase 7: React | ⬜ TODO | Gates component |
| Phase 8: Loaders | ⬜ TODO | Migrate from Legacy |
| Phase 9: Testing | ⬜ TODO | 80% coverage |
| Phase 10: Merge | ⬜ TODO | Final cleanup |

📄 See `docs/ROADMAP_UNIFIED_ARCHITECTURE.md` for details.

---

## ✅ Esistente (NON toccare)

| File | Linee | Cosa fa |
|------|-------|---------|
| `src/data_manager/catalog.py` | 736 | CopernicusCatalog (solo CMEMS) |
| `src/surge_shazam/data/era5_client.py` | ~200 | ERA5 download |
| `src/surge_shazam/data/cmems_client.py` | ~300 | CMEMS download |
| `src/surge_shazam/data/climate_indices.py` | ~150 | NOAA indices |

---

## ✅ Legend

- ⬜ TODO
- 🔄 IN PROGRESS  
- ✅ DONE
- ❌ BLOCKED

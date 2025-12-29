# 🏗️ ROADMAP: Unified Architecture Refactoring

> **Version**: 1.0  
> **Created**: 2025-12-29  
> **Branch**: Started on `feature/gates-streamlit`, will merge to `master`  
> **Status**: 📋 PLANNING

---

## 📊 Executive Summary

Questo documento descrive il piano di refactoring per unificare l'architettura del progetto NICO, consolidando il meglio di:
- **Legacy nico** (gate handling, data loaders)
- **feature/gates-streamlit** (Streamlit UI, gates integration)
- **master** (React frontend, FastAPI, Knowledge Graph)

### 🎯 Obiettivo
Creare un'architettura **modulare, scalabile e asset-centric** ispirata a Dagster/Kedro best practices.

---

## ✅ TODO LIST - Master Checklist

### Phase 0: Setup & Documentation
- [ ] Create this roadmap document
- [ ] Create `docs/MODELS.md` - Pydantic models reference
- [ ] Create `docs/GATES_CATALOG.md` - Gates documentation
- [ ] Update `docs/ARCHITECTURE.md` with new structure
- [ ] Create GitHub Issue for tracking

### Phase 1: Core Models (Foundation)
- [ ] Create `src/core/models.py` - Shared Pydantic models
  - [ ] `BoundingBox` model
  - [ ] `GateModel` model
  - [ ] `TimeRange` model
  - [ ] `DataRequest` model
- [ ] Add unit tests `tests/unit/test_core_models.py`

### Phase 2: Centralized Config
- [ ] Create `config/` directory
- [ ] Create `config/gates.yaml` - Gates catalog (from GATE_CATALOG)
- [ ] Create `config/datasets.yaml` - Dataset providers
- [ ] Create `config/regions.yaml` - Pre-defined regions
- [ ] Create `config/defaults.yaml` - Default parameters
- [ ] Add config loader in `src/core/config.py`

### Phase 3: Gates Module
- [ ] Create `src/gates/` directory
- [ ] Create `src/gates/__init__.py`
- [ ] Create `src/gates/catalog.py` - GateCatalog class
- [ ] Create `src/gates/loader.py` - Shapefile loading
- [ ] Create `src/gates/passes.py` - Pass filtering logic
- [ ] Create `src/gates/buffer.py` - Buffer calculations
- [ ] Add unit tests `tests/unit/test_gates/`
- [ ] Migrate `GATE_CLOSEST_PASSES` from Legacy

### Phase 4: Services Layer
- [ ] Create `src/services/` directory
- [ ] Create `src/services/__init__.py`
- [ ] Create `src/services/gate_service.py` - Gate business logic
- [ ] Create `src/services/data_service.py` - Data operations
- [ ] Create `src/services/analysis_service.py` - Analysis operations
- [ ] Add integration tests `tests/integration/test_services/`

### Phase 5: API Integration
- [ ] Create `api/routers/gates_router.py` - Gates REST endpoints
- [ ] Update `api/routers/data_router.py` to use services
- [ ] Register gates router in `api/main.py`
- [ ] Add API tests `tests/api/test_gates_router.py`

### Phase 6: Streamlit Integration
- [ ] Refactor `app/components/sidebar.py` to use `GateService`
- [ ] Enable gate selector (currently disabled)
- [ ] Test Streamlit app with new architecture
- [ ] Fix hardcoded paths

### Phase 7: React Integration (master branch)
- [ ] Add gates API client in `frontend/src/api.ts`
- [ ] Create `GateSelector` component
- [ ] Integrate with `DataExplorer.tsx`

### Phase 8: Data Loaders Enhancement
- [ ] Migrate `load_filtered_cycles` from Legacy to `src/data/loaders.py`
- [ ] Ensure consistency with existing loaders
- [ ] Add pass filtering support

### Phase 9: Testing & Documentation
- [ ] Achieve 80% test coverage on new modules
- [ ] Update all docstrings
- [ ] Create API documentation
- [ ] Update README.md

### Phase 10: Merge & Cleanup
- [ ] Merge `feature/gates-streamlit` to `master`
- [ ] Remove duplicated code
- [ ] Final audit
- [ ] Tag release v2.0

---

## 📁 New Directory Structure

```
nico/
├── config/                         # 🆕 CENTRALIZED CONFIG
│   ├── gates.yaml                  # Gates metadata
│   ├── datasets.yaml               # Dataset providers
│   ├── regions.yaml                # Pre-defined regions
│   └── defaults.yaml               # Default parameters
│
├── src/
│   ├── core/                       # 🔄 SHARED CORE
│   │   ├── models.py               # 🆕 Pydantic models
│   │   ├── coordinates.py          # ✅ Exists
│   │   ├── config.py               # 🔄 Enhanced
│   │   └── ...
│   │
│   ├── gates/                      # 🆕 GATES MODULE
│   │   ├── __init__.py
│   │   ├── catalog.py              # GateCatalog class
│   │   ├── loader.py               # Shapefile loading
│   │   ├── passes.py               # Pass filtering
│   │   └── buffer.py               # Buffer calculations
│   │
│   ├── services/                   # 🆕 DOMAIN SERVICES
│   │   ├── __init__.py
│   │   ├── gate_service.py         # Gate operations
│   │   ├── data_service.py         # Data operations
│   │   └── analysis_service.py     # Analysis operations
│   │
│   └── ...                         # Existing modules
│
├── api/routers/
│   ├── gates_router.py             # 🆕 Gates API
│   └── ...
│
└── tests/
    ├── unit/
    │   ├── test_core_models.py     # 🆕
    │   └── test_gates/             # 🆕
    └── integration/
        └── test_services/          # 🆕
```

---

## 🔄 Migration Strategy

### From Legacy nico
| Component | Source | Destination |
|-----------|--------|-------------|
| `GATE_CATALOG` | `sidebar.py` | `config/gates.yaml` |
| `GATE_CLOSEST_PASSES` | `config.py` | `config/gates.yaml` |
| `wrap_longitudes()` | `coordinates.py` | ✅ Already in `src/core/` |
| `load_filtered_cycles()` | `loaders.py` | `src/data/loaders.py` |

### From feature/gates-streamlit
| Component | Source | Destination |
|-----------|--------|-------------|
| `_load_gate_geometry()` | `sidebar.py` | `src/gates/loader.py` |
| `AppConfig` dataclass | `sidebar.py` | Keep + use `BoundingBox` |

### From master
| Component | Status |
|-----------|--------|
| `CopernicusCatalog` | ✅ Keep as-is |
| `DataManager` | ✅ Keep as-is |
| `catalog.yaml` | ✅ Link to `config/datasets.yaml` |
| React components | 🔄 Add gates support |

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   React     │  │  Streamlit  │  │     CLI/Notebooks   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    API GATEWAY (FastAPI)                     │
│  /gates    /data    /analysis    /knowledge    /pipeline    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    SERVICES LAYER                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐│
│  │ GateService  │ │ DataService  │ │   AnalysisService    ││
│  └──────────────┘ └──────────────┘ └──────────────────────┘│
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CORE LAYER (Shared)                       │
│  models.py │ coordinates.py │ config.py │ helpers.py        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                         │
│  GateCatalog │ DatasetCatalog │ Loaders │ Cache             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                            │
│  SurrealDB │ NetCDF Files │ Shapefiles │ External APIs      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Decisions

### 1. Config Format: YAML
- Human-readable
- Easy to version control
- Compatible with Kedro/Dagster patterns

### 2. Models: Pydantic v2
- Type safety
- Validation built-in
- FastAPI native support
- Serialization for both API and Streamlit

### 3. Service Layer Pattern
- Business logic separated from UI
- Same services used by API and Streamlit
- Easier testing

### 4. Gates as First-Class Citizens
- Dedicated module `src/gates/`
- REST API endpoints
- Consistent across branches

---

## 📅 Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0 | 1 day | None |
| Phase 1 | 1 day | None |
| Phase 2 | 1 day | Phase 1 |
| Phase 3 | 2 days | Phase 1, 2 |
| Phase 4 | 2 days | Phase 3 |
| Phase 5 | 1 day | Phase 4 |
| Phase 6 | 1 day | Phase 4 |
| Phase 7 | 2 days | Phase 5, master branch |
| Phase 8 | 1 day | Phase 4 |
| Phase 9 | 2 days | All phases |
| Phase 10 | 1 day | All phases |

**Total: ~15 days**

---

## 🚨 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing Streamlit app | HIGH | Feature flag for new code |
| Breaking API compatibility | HIGH | Version API (v1 → v2) |
| Merge conflicts | MEDIUM | Small, focused PRs |
| Missing Legacy functionality | MEDIUM | Audit before removing |

---

## 📝 Related Documents

- `docs/ARCHITECTURE.md` - Current architecture
- `docs/BRANCH_STRATEGY.md` - Branch management
- `docs/AGENT_FULLSTACK.md` - Master branch agent
- `docs/AGENT_GATES.md` - Gates branch agent
- `docs/TASKS/CONTEXT.md` - Task context

---

## 👥 Stakeholders

- **Agent Full Stack** - React/API development
- **Agent Gates** - Streamlit/Gates development
- **Human Developer** - Review and approval

---

*Last updated: 2025-12-29*

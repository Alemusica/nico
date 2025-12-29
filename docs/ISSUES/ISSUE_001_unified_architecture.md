# 🏗️ Unified Architecture Refactoring

## Summary
Refactor the NICO project to create a unified, modular, and scalable architecture that consolidates the best from Legacy nico, feature/gates-streamlit, and master branches.

## Motivation
- **Legacy nico** has excellent gate handling and data loaders
- **feature/gates-streamlit** has Streamlit UI integration
- **master** has React frontend, FastAPI, and Knowledge Graph
- Current code has duplication and inconsistent patterns
- Need merge-ready architecture for both branches

## Proposed Architecture
See `docs/ROADMAP_UNIFIED_ARCHITECTURE.md` for full details.

### Key Changes
1. **Centralized Config** (`config/`) - YAML files for gates, datasets, regions
2. **Core Models** (`src/core/models.py`) - Shared Pydantic models
3. **Gates Module** (`src/gates/`) - Dedicated gate handling
4. **Services Layer** (`src/services/`) - Business logic shared by API and Streamlit
5. **Gates API** (`api/routers/gates_router.py`) - REST endpoints for gates

## Tasks

### Phase 0: Documentation ✅
- [x] Create `docs/ROADMAP_UNIFIED_ARCHITECTURE.md`
- [x] Create `docs/GATES_CATALOG.md`
- [x] Create `docs/MODELS.md`
- [x] Create this issue

### Phase 1: Core Models
- [ ] Create `src/core/models.py`
  - [ ] `BoundingBox` model
  - [ ] `GateModel` model
  - [ ] `TimeRange` model
  - [ ] `DataRequest` model
- [ ] Add tests `tests/unit/test_core_models.py`

### Phase 2: Centralized Config
- [ ] Create `config/gates.yaml`
- [ ] Create `config/datasets.yaml`
- [ ] Create `config/regions.yaml`
- [ ] Create `config/defaults.yaml`
- [ ] Update `src/core/config.py` to load from YAML

### Phase 3: Gates Module
- [ ] Create `src/gates/__init__.py`
- [ ] Create `src/gates/catalog.py`
- [ ] Create `src/gates/loader.py`
- [ ] Create `src/gates/passes.py`
- [ ] Create `src/gates/buffer.py`
- [ ] Migrate `GATE_CLOSEST_PASSES` from Legacy
- [ ] Add tests `tests/unit/test_gates/`

### Phase 4: Services Layer
- [ ] Create `src/services/__init__.py`
- [ ] Create `src/services/gate_service.py`
- [ ] Create `src/services/data_service.py`
- [ ] Create `src/services/analysis_service.py`
- [ ] Add tests `tests/integration/test_services/`

### Phase 5: API Integration
- [ ] Create `api/routers/gates_router.py`
- [ ] Update `api/routers/data_router.py` to use services
- [ ] Register in `api/main.py`
- [ ] Add API tests

### Phase 6: Streamlit Integration
- [ ] Refactor `app/components/sidebar.py`
- [ ] Enable gate selector
- [ ] Test with new architecture

### Phase 7: React Integration (master branch)
- [ ] Add gates API client
- [ ] Create `GateSelector` component
- [ ] Integrate with `DataExplorer`

### Phase 8: Data Loaders
- [ ] Migrate `load_filtered_cycles` from Legacy
- [ ] Add pass filtering support
- [ ] Ensure consistency

### Phase 9: Testing & Docs
- [ ] 80% test coverage
- [ ] Update all docstrings
- [ ] Update README.md

### Phase 10: Merge
- [ ] Merge to master
- [ ] Cleanup
- [ ] Tag release v2.0

## Files to Create
```
config/
├── gates.yaml
├── datasets.yaml
├── regions.yaml
└── defaults.yaml

src/core/
└── models.py

src/gates/
├── __init__.py
├── catalog.py
├── loader.py
├── passes.py
└── buffer.py

src/services/
├── __init__.py
├── gate_service.py
├── data_service.py
└── analysis_service.py

api/routers/
└── gates_router.py
```

## Labels
- `enhancement`
- `architecture`
- `documentation`

## Milestone
v2.0 - Unified Architecture

## Related
- `docs/ROADMAP_UNIFIED_ARCHITECTURE.md`
- `docs/BRANCH_STRATEGY.md`
- `docs/ARCHITECTURE.md`

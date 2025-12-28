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
| 3.1 causal_graph.py | ⬜ TODO | - | - | SurrealDB storage |

---

## 📝 Progress Log

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

### [DATE] - Task X.X
**Status**: ✅ / ❌ / 🔄
**What was done**:
- ...

**Blockers**:
- ...

**Next**:
- ...

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

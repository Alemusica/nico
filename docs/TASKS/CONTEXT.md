# 🧠 Agent Context File

> **LEGGI QUESTO PRIMA DI OGNI TASK**
> Questo file dà awareness su cosa esiste già per evitare duplicazioni.
> 
> **Last Updated**: 2025-12-28 (Sprint completo)

## 📂 Codice Esistente (NON riscrivere!)

### Data Clients ✅
| File | Linee | Status | Copre |
|------|-------|--------|-------|
| `src/data_manager/catalog.py` | 736 | ✅ Working | **Solo CMEMS** (8 prodotti) |
| `src/data_manager/intake_bridge.py` | 180 | ✅ **NEW** | Bridge multi-provider |
| `src/data_manager/causal_graph.py` | 300 | ✅ **NEW** | SurrealDB causal storage |
| `src/surge_shazam/data/era5_client.py` | ~200 | ✅ Working | ERA5 |
| `src/surge_shazam/data/cmems_client.py` | ~300 | ✅ Working | CMEMS download |
| `src/surge_shazam/data/climate_indices.py` | ~150 | ✅ Working | NAO, ENSO, etc |
| `src/surge_shazam/data/cygnss_client.py` | 63 | ✅ **NEW** | NASA CYGNSS wind |

### Multi-Provider Catalog ✅ (Sprint Dec 2025)
| File | Cosa fa |
|------|---------|
| `catalog.yaml` | 11 datasets, latency badges 🟢🟡🔴⚫ |
| `intake_bridge.py` | `get_catalog()`, `search()`, `search_by_latency()` |
| `causal_graph.py` | `CausalGraphDB`, `CausalEdge`, 4 known chains |
| `cygnss_client.py` | NASA earthaccess, near real-time wind |

### Cosa FA `catalog.py` esistente:
- ✅ `list_products()` - lista prodotti CMEMS
- ✅ `search(variable, category, bbox, time)` - ricerca
- ✅ `check_availability()` - verifica copertura
- ✅ `get_download_config()` - config per download
- ✅ Caching JSON con TTL 24h

### Cosa ERA mancante (ORA IMPLEMENTATO ✅):
- ✅ **Latency metadata** → `catalog.yaml` con `latency_badge`
- ✅ **ERA5 nel catalog** → `catalog.yaml` entry
- ✅ **Climate Indices (NOAA)** → `catalog.yaml` entry
- ✅ **CYGNSS (NASA)** → `cygnss_client.py` + `catalog.yaml`
- ✅ **SLCCI (ESA CCI)** → `catalog.yaml` entry
- ✅ **Multi-provider unified** → `intake_bridge.py`

### Causal Discovery ✅
| File | Linee | Status |
|------|-------|--------|
| `src/pattern_engine/causal/pcmci_engine.py` | ~400 | Working |
| `src/pattern_engine/causal/discovery.py` | ~200 | Working |

### Fusion ✅
| File | Linee | Status |
|------|-------|--------|
| `src/data/satellite_fusion.py` | ~500 | Working |

## 🎯 STRATEGIA: Estendere, NON sostituire

```
catalog.yaml (Intake)          ← 🆕 Multi-provider + latency
       ↓
IntakeCatalogBridge            ← 🆕 Bridge unificato
       ↓
┌──────────────────────────────────────────┐
│ CopernicusCatalog (esistente)            │ ← CMEMS
│ ERA5Client (esistente)                   │ ← ERA5  
│ ClimateIndices (esistente)               │ ← NOAA
│ CYGNSSClient (🆕 da fare)                │ ← NASA
└──────────────────────────────────────────┘
```

## 🔗 Dipendenze tra Task

```
Task 1.1 (catalog.yaml) ← Aggiunge latency + multi-provider
    ↓
Task 1.2 (intake_bridge.py) ← Collega a client esistenti
    ↓
Task 1.3 (API endpoints)
    
Task 2.1 (CYGNSS) ← Indipendente, parallelizzabile
    
Task 3.1 (SurrealDB) ← Indipendente, parallelizzabile
```

## 🔀 Parallelizzazione (Chat Separate)

| Chat | Task | Blocca | Note |
|------|------|--------|------|
| **Chat A** | 1.1 → 1.2 → 1.3 | - | Sequential |
| **Chat B** | 2.1 CYGNSS | Niente | ✅ Parallelizza |
| **Chat C** | 3.1 SurrealDB | Niente | ✅ Parallelizza |

## ⚠️ Trappole da Evitare

1. **NON sovrascrivere catalog.py** - estendi con bridge
2. **NON reinstallare pacchetti** - tutto in `.venv`
3. **NON usare python3** - usa `source .venv/bin/activate`
4. **Elimina** `src/surge_shazam/data/catalog.py` (file vuoto duplicato)

## 📋 Checklist Pre-Task

Prima di iniziare qualsiasi task:
- [ ] Ho letto questo file?
- [ ] Ho verificato se esiste già codice simile?
- [ ] Sto usando `.venv/bin/python`?
- [ ] Il task è parallelizzabile o sequenziale?
